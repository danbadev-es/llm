"""Supervised fine-tuning (SFT) on packed conversations, shared by phases 5, 6 and 8.

The model learns only the tokens marked in the data's `learn` mask (the assistant's answers and,
from phase 6, its tool calls), never the system, user or tool-result text. Each phase has a thin
train.py that adds its defaults (base model, data folder, probes) and calls train_sft():

    parser = argparse.ArgumentParser()
    add_sft_args(parser, base=..., run="leon_0.1_chat", upsample=["oasst2=3"])
    train_sft(parser.parse_args(), data_dir, checkpoint_root, probes)

The run writes the control files of leon.run (pid, status.json, pause flag), so the phase's
control.py can start, pause, resume and watch it in the background.
"""

import argparse
import math
import os
import time
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np
import torch

from leon.chat import encode_chat
from leon.data import ConversationFile, load_tokenizer
from leon.model import Leon, LeonConfig
from leon.run import RunFiles, keep_system_awake
from leon.training import load_checkpoint, lr_at, save_checkpoint

MICRO_BATCH_TOKENS = 4096  # what fits in 8 GB with a 16k vocabulary: 4 blocks of 1024, 2 of 2048
PROBE_TOKENS = 80


def add_sft_args(
    parser: argparse.ArgumentParser,
    base: Path | str,
    run: str,
    epochs: int = 3,
    upsample: list[str] | None = None,
    lr: float = 1e-4,
) -> argparse.ArgumentParser:
    """The command-line options of an SFT run, with the phase's own defaults."""
    parser.add_argument("--base", default=str(base), help="the checkpoint to fine-tune")
    parser.add_argument("--run", default=run, help="checkpoints go to <checkpoint root>/<run>/")
    parser.add_argument("--epochs", type=int, default=epochs)
    parser.add_argument("--upsample", nargs="*", default=upsample or [], help="repeat sources: name=times")
    parser.add_argument(
        "--multi-turn", type=int, default=3, help="repeat conversations of 2+ exchanges (rare: ~7%% of the data)"
    )
    parser.add_argument(
        "--batch-size", type=int, default=0, help=f"packed blocks per micro-batch; 0 = {MICRO_BATCH_TOKENS:,} tokens"
    )
    parser.add_argument("--grad-accum", type=int, default=8)
    parser.add_argument("--lr", type=float, default=lr, help="10x below pretraining: adapt, do not forget")
    parser.add_argument("--min-lr", type=float, default=lr / 10)
    parser.add_argument("--warmup-steps", type=int, default=50)
    parser.add_argument("--weight-decay", type=float, default=0.1)
    parser.add_argument("--dropout", type=float, default=0.1, help="regularization: the datasets are small")
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--eval-interval", type=int, default=100)
    parser.add_argument("--max-steps", type=int, default=0, help="stop early (smoke tests); 0 = all epochs")
    parser.add_argument("--resume", action="store_true", help="continue from <run>/last.pt")
    parser.add_argument("--no-compile", action="store_true")
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser


def micro_batch_plan(data: ConversationFile, args: argparse.Namespace, block_size: int) -> list[list[list[int]]]:
    """Every micro-batch of the whole run, decided up front from the seed.

    Each epoch shuffles the conversations and packs them into blocks. Sources named in --upsample
    are repeated, and multi-turn conversations --multi-turn times (the larger of the two, not
    their product, so a multi-turn oasst2 chat is not seen 9 times per epoch).
    Because the plan is a pure function of the seed, --resume only needs the step number to
    continue with exactly the same batches.
    """
    repeats = {name: int(times) for name, times in (item.split("=") for item in args.upsample)}
    times = np.array([repeats.get(str(s), 1) for s in data.sources])
    times = np.where(data.exchanges >= 2, np.maximum(times, args.multi_turn), times)
    pool = np.repeat(np.arange(len(data)), times)
    plan = []
    for epoch in range(args.epochs):
        order = np.random.default_rng(args.seed + epoch).permutation(pool)
        blocks = data.pack(block_size, order)
        plan += [blocks[i : i + args.batch_size] for i in range(0, len(blocks), args.batch_size)]
    return plan


@torch.no_grad()
def validation_loss(model: Leon, val: ConversationFile, block_size: int, pad_id: int, args, autocast) -> float:
    """Mean loss over every learned token of the validation conversations."""
    model.eval()
    blocks = val.pack(block_size, np.arange(len(val)))
    total, count = 0.0, 0
    for i in range(0, len(blocks), args.batch_size):
        x, y = val.batch(blocks[i : i + args.batch_size], block_size, pad_id, args.device)
        with autocast():
            _, loss = model(x, y)
        n = int((y != -100).sum())
        total, count = total + loss.item() * n, count + n
    model.train()
    return total / count


def probe_question(probe: str | list[dict]) -> str:
    """What the dashboard shows for a probe: the question, or the last user message."""
    if isinstance(probe, str):
        return probe
    return next(m["content"] for m in reversed(probe) if m["role"] == "user")


@torch.no_grad()
def answer(model: Leon, tokenizer, probe: str | list[dict], device: str, autocast) -> str:
    """The model's answer to a probe (a question, or whole messages such as a system prompt with tools).

    Generation stops at the end of the turn or at the end of a tool call, which is kept in the
    text, so the probes show when the model decides to call a tool.
    """
    messages = [{"role": "user", "content": probe}] if isinstance(probe, str) else probe
    ids = torch.tensor([encode_chat(tokenizer, messages)], device=device)
    special = tokenizer.special_tokens
    stop = set(special.values()) - {special["<|tool_call|>"]}
    model.eval()
    with autocast():
        tokens = [t.item() for t in model.stream(ids, PROBE_TOKENS, temperature=0.7, top_k=40, stop_ids=stop)]
    model.train()
    keep = {special["<|tool_call|>"], special["<|/tool_call|>"]}
    return tokenizer.decode([t for t in tokens if t not in stop or t in keep]).strip()


def train_sft(
    args: argparse.Namespace,
    data_dir: Path,
    checkpoint_root: Path,
    probes: list[str | list[dict]],
    stage: str = "chat",
) -> None:
    """Fine-tune args.base on data_dir/{train,val}.npz; checkpoints go to checkpoint_root/args.run/."""
    torch.manual_seed(args.seed)
    device = args.device
    use_bf16 = device.startswith("cuda") and torch.cuda.is_bf16_supported()

    def autocast():
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=use_bf16)

    if device.startswith("cuda"):
        torch.backends.cuda.matmul.allow_tf32 = True

    run_dir = checkpoint_root / args.run
    train, val = ConversationFile(data_dir / "train.npz"), ConversationFile(data_dir / "val.npz")

    source = run_dir / "last.pt" if args.resume else args.base
    base = torch.load(source, map_location="cpu", weights_only=True)
    # The model and the data must speak the same tokenizer: the same text has different ids in
    # different versions, and a mismatch would train on noise without any error.
    model_tokenizer = base.get("tokenizer", "v1")
    if model_tokenizer != train.tokenizer:
        raise SystemExit(
            f"Tokenizer mismatch: {source} uses {model_tokenizer} but {data_dir} was encoded with {train.tokenizer}. "
            "Use a base model of the same version, or rerun the phase's prepare.py."
        )
    tokenizer = load_tokenizer(model_tokenizer)
    pad_id = tokenizer.eot_id  # padding is never a target, so any id works
    config = replace(LeonConfig(**base["config"]), dropout=args.dropout)
    model = Leon(config).to(device)
    optimizer = model.configure_optimizer(args.lr, args.weight_decay, (0.9, 0.95), device)
    start_step, best_val = 0, float("inf")
    if args.resume:
        checkpoint = load_checkpoint(run_dir / "last.pt", model, optimizer, device)
        start_step, best_val = checkpoint["step"], checkpoint["best_val"]
        print(f"Resumed from step {start_step} (best val {best_val:.4f})", flush=True)
    else:
        model.load_state_dict(base["model"])
        print(f"Base model: {args.base} (step {base['step']}, stage {base.get('stage', 'base')})", flush=True)

    block_size = config.block_size
    args.batch_size = args.batch_size or max(1, MICRO_BATCH_TOKENS // block_size)
    plan = micro_batch_plan(train, args, block_size)
    total_steps = len(plan) // args.grad_accum
    if args.max_steps:
        total_steps = min(total_steps, args.max_steps)
    settings = {**vars(args), "model": asdict(config)}
    train_model = model if args.no_compile else torch.compile(model)

    def schedule(step: int) -> float:
        return lr_at(step, args.lr, args.min_lr, args.warmup_steps, total_steps)

    def save(path: Path, step: int, with_optimizer: bool) -> None:
        save_checkpoint(
            path,
            model,
            step,
            best_val,
            optimizer if with_optimizer else None,
            settings,
            stage=stage,
            tokenizer=model_tokenizer,
        )

    print(
        f"Params {model.num_params() / 1e6:.1f}M | {len(train):,} train conversations | upsample {args.upsample} | "
        f"{len(plan):,} micro-batches = {total_steps:,} steps of {args.batch_size * args.grad_accum} blocks "
        f"x {block_size + 1} tokens | {args.epochs} epochs\n",
        flush=True,
    )

    # Control files for control.py: pid, pause request and status.json (progress, losses, ETA).
    run = RunFiles(run_dir)
    run.dir.mkdir(parents=True, exist_ok=True)
    run.pid.write_text(str(os.getpid()))
    run.clear_pause()
    keep_system_awake()
    tokens_per_step = args.batch_size * args.grad_accum * (block_size + 1)  # packed blocks, padding included
    status = {"state": "running", "max_steps": total_steps}
    if args.resume:  # keep the latest evaluation on screen until the next one
        status.update({k: v for k, v in run.read_status().items() if k in ("val_loss", "probes", "best_val")})

    def report(step: int, **fields) -> None:
        status.update(fields)
        run.write_status(**status, step=step, tokens_seen=step * tokens_per_step)

    report(start_step)
    step = start_step
    started = time.perf_counter()
    window_start, window_steps = time.perf_counter(), 0  # throughput between evaluations
    loss_sum = torch.zeros((), device=device)
    try:
        for step in range(start_step, total_steps + 1):
            already_evaluated = args.resume and step == start_step
            if (step % args.eval_interval == 0 or step == total_steps) and not already_evaluated:
                val_loss = validation_loss(model, val, block_size, pad_id, args, autocast)
                elapsed = time.perf_counter() - started
                print(
                    f"step {step:5d}/{total_steps} | val loss (answers) {val_loss:.4f} "
                    f"| ppl {math.exp(val_loss):.1f} | lr {schedule(step):.2e} | {elapsed / 60:.1f} min",
                    flush=True,
                )
                answers = []
                for probe in probes:
                    question = probe_question(probe)
                    reply = answer(model, tokenizer, probe, device, autocast)[:160]
                    answers.append([question, reply])
                    print(f"    {question} → {reply!r}", flush=True)
                report(step, val_loss=val_loss, best_val=min(best_val, val_loss), probes=answers)
                window_start, window_steps = time.perf_counter(), 0
                if val_loss < best_val:
                    best_val = val_loss
                    save(run_dir / "best.pt", step, with_optimizer=False)
                save(run_dir / "last.pt", step, with_optimizer=True)
            if step == total_steps:
                break

            for group in optimizer.param_groups:
                group["lr"] = schedule(step)
            for blocks in plan[step * args.grad_accum : (step + 1) * args.grad_accum]:
                x, y = train.batch(blocks, block_size, pad_id, device)
                with autocast():
                    _, loss = train_model(x, y)
                (loss / args.grad_accum).backward()
                loss_sum += loss.detach()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            window_steps += 1

            done = step + 1
            if done % 10 == 0:  # .item() waits for the GPU: only every few steps
                seconds_per_step = (time.perf_counter() - window_start) / window_steps
                report(
                    done,
                    train_loss=loss_sum.item() / (10 * args.grad_accum),
                    tok_per_s=round(tokens_per_step / seconds_per_step),
                    eta_hours=(total_steps - done) * seconds_per_step / 3600,
                )
                loss_sum.zero_()
            if run.pause_requested():
                save(run_dir / "last.pt", done, with_optimizer=True)
                run.clear_pause()
                report(done, state="paused")
                print(f"\nPaused: saved last.pt at step {done}. Continue with: control.py resume", flush=True)
                return
    except KeyboardInterrupt:
        save(run_dir / "last.pt", step, with_optimizer=True)
        report(step, state="paused")
        print(f"\nInterrupted: saved last.pt at step {step}. Continue with --resume.", flush=True)
        return

    report(total_steps, state="finished", eta_hours=0)
    print(f"\nBest val loss (answers): {best_val:.4f}. Chat model: {run_dir / 'best.pt'}", flush=True)
