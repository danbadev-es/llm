"""Phase 7: build preference pairs for DPO (uses the GPU: started with `control.py prepare`).

For each prompt of the phase 5 data written by people (oasst2, aya, dolly; one exchange):

    chosen    the human answer from the dataset
    rejected  answers Leon itself samples for the same prompt (the model to improve: phase 6's,
              or phase 5's if there is none), at temperature 0.8

So DPO learns to prefer the human answer over Leon's own typical mistakes: repeating itself,
drifting off topic, inventing. Several samples per prompt are generated in one batch (the same
prompt repeated: the batch costs about as much as one sample), and up to --keep are kept.

The run is resumable: pairs are appended to data/dpo/pairs.jsonl as they are made, and a restart
skips the prompts already done. It writes status.json like a training run, so control.py's
status/watch/pause work on it (run "prepare"). At the end, pairs.jsonl is split into
data/dpo/{train,val}.jsonl (5 % val, by prompt).
"""

import argparse
import hashlib
import json
import os
import random
import time
from pathlib import Path

import torch

from leon.chat import encode_chat
from leon.data import load_tokenizer
from leon.dpo import encode_answer
from leon.model import Leon, LeonConfig
from leon.paths import CHECKPOINT_DIR, DATA_DIR
from leon.run import RunFiles, keep_system_awake

DPO_DIR = DATA_DIR / "dpo"
AGENT = CHECKPOINT_DIR / "6_agent" / "leon_0.1_agent" / "best.pt"
CHAT = CHECKPOINT_DIR / "5_sft" / "leon_0.1_chat" / "best.pt"
HUMAN_SOURCES = ("oasst2", "aya", "dolly")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--policy", default=str(AGENT if AGENT.exists() else CHAT), help="the model that answers")
    parser.add_argument("--run", default="prepare", help="status files go to checkpoints/7_dpo/<run>/")
    parser.add_argument("--prompts", type=int, default=3000)
    parser.add_argument("--samples", type=int, default=4, help="answers sampled per prompt (one batch)")
    parser.add_argument("--keep", type=int, default=2, help="rejected answers kept per prompt")
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--resume", action="store_true", help="accepted for control.py; resuming is automatic")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def stable_fraction(text: str) -> float:
    return int(hashlib.sha256(text.encode("utf-8")).hexdigest()[:8], 16) / 0xFFFFFFFF


def load_prompts(tokenizer, count: int, block_size: int, seed: int) -> list[dict]:
    """Single-exchange human conversations whose prompt + answer fit in the context."""
    items = []
    with open(DATA_DIR / "sft" / "train.jsonl", encoding="utf-8") as f:
        for line in f:
            item = json.loads(line)
            messages = item["messages"]
            if item["source"] not in HUMAN_SOURCES or [m["role"] for m in messages if m["role"] != "system"] != [
                "user",
                "assistant",
            ]:
                continue
            answer = messages[-1]["content"]
            if not 40 <= len(answer) <= 1500:
                continue
            prompt = messages[:-1]
            if len(encode_answer(tokenizer, prompt, answer)[0]) > block_size // 2:  # room for long rejections
                continue
            items.append({"prompt": prompt, "chosen": answer, "source": item["source"]})
    random.Random(seed).shuffle(items)
    return items[:count]


def normalized(text: str) -> str:
    return " ".join(text.lower().split())


@torch.no_grad()
def sample(model: Leon, tokenizer, prompt: list[dict], args, use_bf16: bool) -> list[str]:
    """args.samples answers to one prompt, generated together (the prompt repeated in a batch)."""
    ids = encode_chat(tokenizer, prompt)
    idx = torch.tensor([ids] * args.samples, device=args.device)
    stop = set(tokenizer.special_tokens.values())
    rows: list[list[int]] = [[] for _ in range(args.samples)]
    finished = [False] * args.samples
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=use_bf16):
        for step in model.stream(idx, args.max_new_tokens, temperature=args.temperature, top_k=40, stop_ids=stop):
            for row, token in enumerate(step.view(-1).tolist()):
                if finished[row]:
                    continue
                if token in stop:
                    finished[row] = True
                else:
                    rows[row].append(token)
    return [tokenizer.decode(row).strip() for row in rows]


def split(pairs_path: Path) -> dict:
    counts = {"train": 0, "val": 0}
    with (
        open(pairs_path, encoding="utf-8") as pairs,
        open(DPO_DIR / "train.jsonl", "w", encoding="utf-8") as train,
        open(DPO_DIR / "val.jsonl", "w", encoding="utf-8") as val,
    ):
        for line in pairs:
            item = json.loads(line)
            name = "val" if stable_fraction(item["prompt"][-1]["content"]) < 0.05 else "train"
            (val if name == "val" else train).write(line)
            counts[name] += 1
    return counts


def main() -> None:
    args = parse_args()
    use_bf16 = args.device.startswith("cuda") and torch.cuda.is_bf16_supported()
    checkpoint = torch.load(args.policy, map_location="cpu", weights_only=True)
    model = Leon(LeonConfig(**checkpoint["config"])).to(args.device).eval()
    model.load_state_dict(checkpoint["model"])
    tokenizer = load_tokenizer(checkpoint.get("tokenizer", "v1"))
    prompts = load_prompts(tokenizer, args.prompts, model.config.block_size, args.seed)
    print(f"Policy: {args.policy} | {len(prompts):,} prompts x {args.samples} samples, keep {args.keep}", flush=True)

    DPO_DIR.mkdir(parents=True, exist_ok=True)
    pairs_path = DPO_DIR / "pairs.jsonl"
    done = set()
    if pairs_path.exists():
        done = {json.loads(line)["index"] for line in open(pairs_path, encoding="utf-8")}
        print(f"Resuming: {len(done):,} prompts already done", flush=True)

    run = RunFiles(CHECKPOINT_DIR / "7_dpo" / args.run)
    run.dir.mkdir(parents=True, exist_ok=True)
    run.pid.write_text(str(os.getpid()))
    run.clear_pause()
    keep_system_awake()
    rng = random.Random(args.seed)
    started, made, pairs_written = time.perf_counter(), 0, 0
    status = {"state": "running", "max_steps": len(prompts), "perplexity": False}

    def report(**fields) -> None:
        status.update(fields)
        run.write_status(**status, step=len(done))

    report()
    with open(pairs_path, "a", encoding="utf-8") as out:
        for index, item in enumerate(prompts):
            if index in done:
                continue
            answers = sample(model, tokenizer, item["prompt"], args, use_bf16)
            rejected = [a for a in answers if a and normalized(a) != normalized(item["chosen"])]
            for answer in rng.sample(rejected, min(args.keep, len(rejected))):
                record = {"index": index, **item, "rejected": answer}
                out.write(json.dumps(record, ensure_ascii=False) + "\n")
                pairs_written += 1
            if not rejected:  # still mark the prompt as done
                out.write(json.dumps({"index": index, "skipped": True}) + "\n")
            out.flush()
            done.add(index)
            made += 1
            if made % 10 == 0:
                seconds = (time.perf_counter() - started) / made
                report(
                    eta_hours=(len(prompts) - len(done)) * seconds / 3600,
                    metrics={"pairs": pairs_written},
                    probes=[[item["prompt"][-1]["content"][:120], (rejected or [""])[0][:160]]],
                )
            if run.pause_requested():
                run.clear_pause()
                report(state="paused")
                print(f"Paused after {len(done):,} prompts. Continue with: control.py prepare", flush=True)
                return

    # Drop the "skipped" markers and split by prompt.
    lines = [line for line in open(pairs_path, encoding="utf-8") if '"skipped": true' not in line]
    clean = DPO_DIR / "pairs.clean.jsonl"
    clean.write_text("".join(lines), encoding="utf-8")
    counts = split(clean)
    clean.unlink()
    report(state="finished", eta_hours=0, metrics={"train pairs": counts["train"], "val pairs": counts["val"]})
    print(f"Pairs: {counts['train']:,} train, {counts['val']:,} val in {DPO_DIR}", flush=True)


if __name__ == "__main__":
    main()
