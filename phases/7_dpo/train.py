"""Phase 7: DPO, learning from preferences (pairs made by phases/7_dpo/prepare.py).

Starts from the model that produced the rejected answers (phase 6's, else phase 5's) and trains
it with leon.dpo.dpo_loss against a frozen reference: the same starting weights, whose
log-probabilities are computed once at the start and kept in <run>/reference.pt.
Usually run through phases/7_dpo/control.py (start / pause / resume / status / watch).

The dashboard shows the DPO loss (it starts at log 2 = 0.693, when chosen and rejected are tied)
and two metrics on the validation pairs: reward accuracy (the share of pairs where the chosen
answer wins) and the mean margin between their rewards.
"""

import argparse
import json
import math
import os
import time
from dataclasses import asdict, replace

import numpy as np
import torch

from leon.data import load_tokenizer
from leon.dpo import batch, dpo_loss, encode_answer, sequence_logprobs
from leon.model import Leon, LeonConfig
from leon.paths import CHECKPOINT_DIR, DATA_DIR
from leon.run import RunFiles, keep_system_awake
from leon.sft import answer
from leon.training import load_checkpoint, lr_at, save_checkpoint

CHECKPOINT_ROOT = CHECKPOINT_DIR / "7_dpo"
DPO_DIR = DATA_DIR / "dpo"
AGENT = CHECKPOINT_DIR / "6_agent" / "leon_0.1_agent" / "best.pt"
CHAT = CHECKPOINT_DIR / "5_sft" / "leon_0.1_chat" / "best.pt"
PROBES = [
    "Hola, ¿quién eres?",
    "Escribe una frase sobre el mar.",
    "¿Qué consejos me das para dormir mejor?",
    "Explícame qué es la fotosíntesis.",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--base", default=str(AGENT if AGENT.exists() else CHAT), help="policy and reference")
    parser.add_argument("--run", default="leon_0.1_dpo")
    parser.add_argument("--beta", type=float, default=0.1, help="how far the policy may move from the reference")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--pairs-per-micro-batch", type=int, default=2, help="each pair is 2 sequences")
    parser.add_argument("--grad-accum", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-6, help="DPO moves fast: 100x below SFT")
    parser.add_argument("--warmup-steps", type=int, default=20)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--eval-interval", type=int, default=50)
    parser.add_argument("--max-steps", type=int, default=0, help="stop early (smoke tests); 0 = all epochs")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def load_pairs(path, tokenizer, block_size: int) -> list[tuple]:
    """(chosen, rejected) sequences with their answer masks; pairs that do not fit are skipped."""
    pairs = []
    for line in open(path, encoding="utf-8"):
        item = json.loads(line)
        chosen = encode_answer(tokenizer, item["prompt"], item["chosen"])
        rejected = encode_answer(tokenizer, item["prompt"], item["rejected"])
        if max(len(chosen[0]), len(rejected[0])) <= block_size + 1:
            pairs.append((chosen, rejected))
    return pairs


def logprobs_of(model, pairs, indices, pad_id, device, autocast) -> tuple[torch.Tensor, torch.Tensor]:
    """log π(chosen) and log π(rejected) for some pairs, in one forward pass (2 rows per pair)."""
    sequences = [pairs[i][0] for i in indices] + [pairs[i][1] for i in indices]
    x, y = batch(sequences, pad_id, device)
    with autocast():
        scores = sequence_logprobs(model, x, y)
    return scores[: len(indices)], scores[len(indices) :]


@torch.no_grad()
def reference_logprobs(model, pairs, size, pad_id, device, autocast) -> torch.Tensor:
    """(pairs, 2) log-probabilities under the frozen starting model."""
    model.eval()
    out = []
    for start in range(0, len(pairs), size):
        chosen, rejected = logprobs_of(
            model, pairs, range(start, min(start + size, len(pairs))), pad_id, device, autocast
        )
        out.append(torch.stack([chosen, rejected], dim=1).float().cpu())
    model.train()
    return torch.cat(out)


@torch.no_grad()
def evaluate(model, pairs, reference, size, pad_id, beta, device, autocast) -> tuple[float, dict]:
    model.eval()
    losses, metrics = [], []
    for start in range(0, len(pairs), size):
        indices = range(start, min(start + size, len(pairs)))
        chosen, rejected = logprobs_of(model, pairs, indices, pad_id, device, autocast)
        ref = reference[start : start + len(indices)].to(device)
        loss, m = dpo_loss(chosen.float(), rejected.float(), ref[:, 0], ref[:, 1], beta)
        losses.append(loss.item() * len(indices))
        metrics.append({k: v * len(indices) for k, v in m.items()})
    model.train()
    n = len(pairs)
    return sum(losses) / n, {k: sum(m[k] for m in metrics) / n for k in metrics[0]}


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    device = args.device
    use_bf16 = device.startswith("cuda") and torch.cuda.is_bf16_supported()

    def autocast():
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=use_bf16)

    run_dir = CHECKPOINT_ROOT / args.run
    base = torch.load(args.base, map_location="cpu", weights_only=True)
    model_tokenizer = base.get("tokenizer", "v1")
    tokenizer = load_tokenizer(model_tokenizer)
    pad_id = tokenizer.eot_id
    config = replace(LeonConfig(**base["config"]), dropout=0.0)  # no dropout: policy and reference must agree at step 0
    model = Leon(config).to(device)
    model.load_state_dict(base["model"])
    stage = base.get("stage", "chat")
    train_pairs = load_pairs(DPO_DIR / "train.jsonl", tokenizer, config.block_size)
    val_pairs = load_pairs(DPO_DIR / "val.jsonl", tokenizer, config.block_size)
    if not train_pairs or not val_pairs:
        raise SystemExit(f"Not enough pairs ({len(train_pairs)} train, {len(val_pairs)} val): run control.py prepare")
    size = args.pairs_per_micro_batch

    reference_path = run_dir / "reference.pt"
    if args.resume and reference_path.exists():
        reference = torch.load(reference_path, weights_only=True)
    else:
        print(f"Reference log-probabilities of {len(train_pairs) + len(val_pairs):,} pairs...", flush=True)
        reference = {
            "train": reference_logprobs(model, train_pairs, size, pad_id, device, autocast),
            "val": reference_logprobs(model, val_pairs, size, pad_id, device, autocast),
        }
        run_dir.mkdir(parents=True, exist_ok=True)
        torch.save(reference, reference_path)

    optimizer = model.configure_optimizer(args.lr, args.weight_decay, (0.9, 0.95), device)
    start_step, best_val = 0, float("inf")
    if args.resume:
        checkpoint = load_checkpoint(run_dir / "last.pt", model, optimizer, device)
        start_step, best_val = checkpoint["step"], checkpoint["best_val"]
        print(f"Resumed from step {start_step} (best val {best_val:.4f})", flush=True)
    else:
        print(f"Base model: {args.base} (step {base['step']}, stage {stage})", flush=True)

    order = np.concatenate(
        [np.random.default_rng(args.seed + e).permutation(len(train_pairs)) for e in range(args.epochs)]
    )
    pairs_per_step = size * args.grad_accum
    total_steps = len(order) // pairs_per_step
    if args.max_steps:
        total_steps = min(total_steps, args.max_steps)
    settings = {**vars(args), "model": asdict(config)}
    print(
        f"{len(train_pairs):,} train pairs, {len(val_pairs):,} val | {total_steps:,} steps of {pairs_per_step} pairs "
        f"| beta {args.beta} | lr {args.lr}\n",
        flush=True,
    )

    def schedule(step: int) -> float:
        return lr_at(step, args.lr, 0.0, args.warmup_steps, total_steps)

    def save(path, step, with_optimizer):
        optimizer_state = optimizer if with_optimizer else None
        save_checkpoint(path, model, step, best_val, optimizer_state, settings, stage=stage, tokenizer=model_tokenizer)

    run = RunFiles(run_dir)
    run.pid.write_text(str(os.getpid()))
    run.clear_pause()
    keep_system_awake()
    status = {"state": "running", "max_steps": total_steps, "perplexity": False}
    if args.resume:
        status.update(
            {k: v for k, v in run.read_status().items() if k in ("val_loss", "best_val", "metrics", "probes")}
        )

    def report(step: int, **fields) -> None:
        status.update(fields)
        run.write_status(**status, step=step)

    report(start_step)
    step, window_start, window_steps = start_step, time.perf_counter(), 0
    loss_sum, accuracy_sum = 0.0, 0.0
    try:
        for step in range(start_step, total_steps + 1):
            if (step % args.eval_interval == 0 or step == total_steps) and not (args.resume and step == start_step):
                val_loss, metrics = evaluate(
                    model, val_pairs, reference["val"], size, pad_id, args.beta, device, autocast
                )
                print(
                    f"step {step:5d}/{total_steps} | val dpo loss {val_loss:.4f} | reward accuracy "
                    f"{metrics['reward_accuracy']:.3f} | margin {metrics['margin']:.3f} | lr {schedule(step):.2e}",
                    flush=True,
                )
                probes = []
                for question in PROBES:
                    reply = answer(model, tokenizer, question, device, autocast)[:160]
                    probes.append([question, reply])
                    print(f"    {question} → {reply!r}", flush=True)
                shown = {"reward accuracy": metrics["reward_accuracy"], "margin": metrics["margin"]}
                report(step, val_loss=val_loss, best_val=min(best_val, val_loss), metrics=shown, probes=probes)
                window_start, window_steps = time.perf_counter(), 0
                if val_loss < best_val:
                    best_val = val_loss
                    save(run_dir / "best.pt", step, with_optimizer=False)
                save(run_dir / "last.pt", step, with_optimizer=True)
            if step == total_steps:
                break

            for group in optimizer.param_groups:
                group["lr"] = schedule(step)
            for micro in range(args.grad_accum):
                start = step * pairs_per_step + micro * size
                indices = order[start : start + size].tolist()
                chosen, rejected = logprobs_of(model, train_pairs, indices, pad_id, device, autocast)
                ref = reference["train"][indices].to(device)
                loss, m = dpo_loss(chosen.float(), rejected.float(), ref[:, 0], ref[:, 1], args.beta)
                (loss / args.grad_accum).backward()
                loss_sum += loss.item() / args.grad_accum
                accuracy_sum += m["reward_accuracy"] / args.grad_accum
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            window_steps += 1

            done = step + 1
            if done % 10 == 0:
                seconds = (time.perf_counter() - window_start) / window_steps
                report(done, train_loss=loss_sum / 10, eta_hours=(total_steps - done) * seconds / 3600)
                loss_sum, accuracy_sum = 0.0, 0.0
            if run.pause_requested():
                save(run_dir / "last.pt", done, with_optimizer=True)
                run.clear_pause()
                report(done, state="paused")
                print(f"\nPaused: saved last.pt at step {done}. Continue with: control.py resume", flush=True)
                return
    except KeyboardInterrupt:
        save(run_dir / "last.pt", step, with_optimizer=True)
        report(step, state="paused")
        return

    report(total_steps, state="finished", eta_hours=0)
    print(f"\nBest val DPO loss: {best_val:.4f} (log 2 = {math.log(2):.4f} at the start). Model: {run_dir / 'best.pt'}")


if __name__ == "__main__":
    main()
