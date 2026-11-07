"""Phase 5: supervised fine-tuning (SFT) of the pretrained Leon on Spanish conversations.

Starts from a base checkpoint (phase 4) and trains it on the chat template, learning only the
assistant's answers. The result is saved with stage="chat", which the chat app recognizes.
Usually run through phases/5_sft/control.py (start / pause / resume / status / watch). Run by
hand, Ctrl+C also saves last.pt; continue with --resume.
"""

import argparse
import math
import os
import time
from dataclasses import asdict, replace

import numpy as np
import torch

from leon.chat import encode_chat
from leon.data import ConversationFile, load_tokenizer
from leon.model import Leon, LeonConfig
from leon.paths import CHECKPOINT_DIR, DATA_DIR
from leon.run import RunFiles, keep_system_awake
from leon.training import load_checkpoint, lr_at, save_checkpoint

CHECKPOINT_ROOT = CHECKPOINT_DIR / "5_sft"
# The base model: the 2048-token context extension if it has been trained, else the 1024 one.
EXTENDED_BASE = CHECKPOINT_DIR / "4_pretrain" / "leon_0.1_ctx2048" / "best.pt"
BASE = EXTENDED_BASE if EXTENDED_BASE.exists() else CHECKPOINT_DIR / "4_pretrain" / "leon_0.1" / "best.pt"
MICRO_BATCH_TOKENS = 4096  # what fits in 8 GB with a 16k vocabulary: 4 blocks of 1024, 2 of 2048
SFT_DIR = DATA_DIR / "sft"
# Fixed questions answered at every evaluation, to watch the chat format being learned.
PROBES = [
    "Hola, ¿quién eres?",
    "¿Cuál es la capital de Francia?",
    "Escribe una frase sobre el mar.",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--base", default=str(BASE))
    parser.add_argument("--run", default="leon_0.1_chat", help="checkpoints go to checkpoints/5_sft/<run>/")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--upsample", nargs="*", default=["oasst2=3", "aya=2"], help="repeat human-written sources")
    parser.add_argument(
        "--multi-turn", type=int, default=3, help="repeat conversations of 2+ exchanges (rare: ~7%% of the data)"
    )
    parser.add_argument(
        "--batch-size", type=int, default=0, help=f"packed blocks per micro-batch; 0 = {MICRO_BATCH_TOKENS:,} tokens"
    )
    parser.add_argument("--grad-accum", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-4, help="10x below pretraining: adapt, do not forget")
    parser.add_argument("--min-lr", type=float, default=1e-5)
    parser.add_argument("--warmup-steps", type=int, default=50)
    parser.add_argument("--weight-decay", type=float, default=0.1)
    parser.add_argument("--dropout", type=float, default=0.1, help="regularization: the dataset is small")
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--eval-interval", type=int, default=100)
    parser.add_argument("--max-steps", type=int, default=0, help="stop early (smoke tests); 0 = all epochs")
    parser.add_argument("--resume", action="store_true", help="continue from <run>/last.pt")
    parser.add_argument("--no-compile", action="store_true")
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def micro_batch_plan(data: ConversationFile, args: argparse.Namespace, block_size: int) -> list[list[list[int]]]:
    """Every micro-batch of the whole run, decided up front from the seed.

    Each epoch shuffles the conversations and packs them into blocks. Human-written sources are
    repeated per --upsample and multi-turn conversations --multi-turn times (the larger of the
    two, not their product, so a multi-turn oasst2 chat is not seen 9 times per epoch).
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
    """Mean loss over every answer token of the validation conversations."""
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


@torch.no_grad()
def answer(model: Leon, tokenizer, question: str, device: str, autocast) -> str:
    ids = torch.tensor([encode_chat(tokenizer, [{"role": "user", "content": question}])], device=device)
    stop = set(tokenizer.special_tokens.values())
    model.eval()
    with autocast():
        tokens = [t.item() for t in model.stream(ids, 80, temperature=0.7, top_k=40, stop_ids=stop)]
    model.train()
    return tokenizer.decode([t for t in tokens if t not in stop]).strip()


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    device = args.device
    use_bf16 = device.startswith("cuda") and torch.cuda.is_bf16_supported()

    def autocast():
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=use_bf16)

    if device.startswith("cuda"):
        torch.backends.cuda.matmul.allow_tf32 = True

    run_dir = CHECKPOINT_ROOT / args.run
    train, val = ConversationFile(SFT_DIR / "train.npz"), ConversationFile(SFT_DIR / "val.npz")

    source = run_dir / "last.pt" if args.resume else args.base
    base = torch.load(source, map_location="cpu", weights_only=True)
    # The model and the data must speak the same tokenizer: the same text has different ids in
    # different versions, and a mismatch would train on noise without any error.
    model_tokenizer = base.get("tokenizer", "v1")
    if model_tokenizer != train.tokenizer:
        raise SystemExit(
            f"Tokenizer mismatch: {source} uses {model_tokenizer} but data/sft was encoded with {train.tokenizer}. "
            "Use a base model of the same version, or rerun phases/5_sft/prepare.py."
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
                probes = []
                for question in PROBES:
                    reply = answer(model, tokenizer, question, device, autocast)[:160]
                    probes.append([question, reply])
                    print(f"    {question} → {reply!r}", flush=True)
                report(step, val_loss=val_loss, best_val=min(best_val, val_loss), probes=probes)
                window_start, window_steps = time.perf_counter(), 0
                if val_loss < best_val:
                    best_val = val_loss
                    save_checkpoint(
                        run_dir / "best.pt",
                        model,
                        step,
                        best_val,
                        settings=settings,
                        stage="chat",
                        tokenizer=model_tokenizer,
                    )
                save_checkpoint(
                    run_dir / "last.pt",
                    model,
                    step,
                    best_val,
                    optimizer,
                    settings,
                    stage="chat",
                    tokenizer=model_tokenizer,
                )
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
                save_checkpoint(
                    run_dir / "last.pt",
                    model,
                    done,
                    best_val,
                    optimizer,
                    settings,
                    stage="chat",
                    tokenizer=model_tokenizer,
                )
                run.clear_pause()
                report(done, state="paused")
                print(f"\nPaused: saved last.pt at step {done}. Continue with: control.py resume", flush=True)
                return
    except KeyboardInterrupt:
        save_checkpoint(
            run_dir / "last.pt", model, step, best_val, optimizer, settings, stage="chat", tokenizer=model_tokenizer
        )
        report(step, state="paused")
        print(f"\nInterrupted: saved last.pt at step {step}. Continue with --resume.", flush=True)
        return

    report(total_steps, state="finished", eta_hours=0)
    print(f"\nBest val loss (answers): {best_val:.4f}. Chat model: {run_dir / 'best.pt'}", flush=True)


if __name__ == "__main__":
    main()
