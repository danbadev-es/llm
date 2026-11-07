"""Checks for leon.sft, the fine-tuning loop of phases 5, 6 and 8. Run: pytest tests/test_sft.py"""

import argparse
import tempfile
from pathlib import Path

import numpy as np
import torch

import leon.sft as sft
from leon.chat import encode_chat_for_training
from leon.model import Leon, LeonConfig
from leon.run import RunFiles
from leon.tokenizer import BPETokenizer, count_chunks
from leon.training import save_checkpoint

TEXT = "hola qué tal, muy bien gracias. ¿cuánto es dos más dos? cuatro. adiós y hasta luego"


def make_data(folder: Path, tokenizer: BPETokenizer, n: int) -> None:
    """A tiny {train,val}.npz in the format of phases/5_sft/prepare.py."""
    rng = np.random.default_rng(0)
    words = TEXT.split()
    folder.mkdir(parents=True, exist_ok=True)
    for split, count in (("train", n), ("val", 3)):
        ids, learn, offsets = [], [], [0]
        for _ in range(count):
            question = " ".join(rng.choice(words, 4))
            reply = " ".join(rng.choice(words, 6))
            tokens, targets = encode_chat_for_training(
                tokenizer, [{"role": "user", "content": question}, {"role": "assistant", "content": reply}]
            )
            ids += tokens
            learn += targets
            offsets.append(len(ids))
        np.savez(
            folder / f"{split}.npz",
            ids=np.array(ids, dtype=np.uint16),
            learn=np.array(learn, dtype=bool),
            offsets=np.array(offsets, dtype=np.int64),
            sources=np.array(["test"] * count),
            tokenizer=np.array("v2"),
            exchanges=np.ones(count, dtype=np.int16),
        )


def args_for(base: Path, max_steps: int, resume: bool = False) -> argparse.Namespace:
    parser = sft.add_sft_args(argparse.ArgumentParser(), base=base, run="tiny")
    return parser.parse_args(
        ["--device", "cpu", "--no-compile", "--batch-size", "2", "--grad-accum", "2", "--epochs", "2"]
        + ["--eval-interval", "100", "--max-steps", str(max_steps), "--warmup-steps", "2"]
        + (["--resume"] if resume else [])
    )


def test_pause_and_resume_give_the_same_weights_as_an_uninterrupted_run(monkeypatch) -> None:
    tokenizer = BPETokenizer.train(count_chunks([TEXT] * 5), 300)
    monkeypatch.setattr(sft, "load_tokenizer", lambda *_: tokenizer)
    monkeypatch.setattr(sft, "keep_system_awake", lambda: None)
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        make_data(tmp / "data", tokenizer, n=40)
        config = LeonConfig(vocab_size=tokenizer.vocab_size, block_size=64, n_layer=1, n_head=2, n_kv_head=1, n_embd=16)
        torch.manual_seed(0)
        save_checkpoint(tmp / "base.pt", Leon(config), step=1, best_val=0.0, tokenizer="v2")

        sft.train_sft(args_for(tmp / "base.pt", 6), tmp / "data", tmp / "straight", ["hola"])
        straight = torch.load(tmp / "straight" / "tiny" / "last.pt", weights_only=True)

        # The same run, paused after step 3 (as `control.py pause` would) and resumed.
        calls = {"n": 0}

        def pause_once(self) -> bool:
            calls["n"] += 1
            return calls["n"] == 3

        monkeypatch.setattr(RunFiles, "pause_requested", pause_once)
        sft.train_sft(args_for(tmp / "base.pt", 6), tmp / "data", tmp / "paused", ["hola"])
        run = RunFiles(tmp / "paused" / "tiny")
        assert run.read_status()["state"] == "paused" and run.read_status()["step"] == 3
        monkeypatch.setattr(RunFiles, "pause_requested", lambda self: False)
        sft.train_sft(args_for(tmp / "base.pt", 6, resume=True), tmp / "data", tmp / "paused", ["hola"])
        resumed = torch.load(tmp / "paused" / "tiny" / "last.pt", weights_only=True)

        assert run.read_status()["state"] == "finished" and resumed["step"] == straight["step"] == 6
        assert resumed["stage"] == "chat"
        for name, weight in straight["model"].items():
            assert torch.equal(weight, resumed["model"][name]), name


def test_answer_keeps_a_tool_call_and_stops_at_its_end() -> None:
    """Probes show tool calls: generation stops after <|/tool_call|>, which stays in the text."""
    tokenizer = BPETokenizer.train(count_chunks([TEXT] * 5), 300)
    special = tokenizer.special_tokens
    script = [special["<|tool_call|>"], *tokenizer.encode("{}"), special["<|/tool_call|>"], *tokenizer.encode("no")]

    class Scripted(torch.nn.Module):
        def stream(self, ids, max_new, stop_ids=(), **_):
            for token in script:
                yield torch.tensor([[token]])
                if token in stop_ids:
                    return

    reply = sft.answer(Scripted(), tokenizer, "hola", "cpu", lambda: torch.autocast("cpu", enabled=False))
    assert reply == "<|tool_call|>{}<|/tool_call|>"
