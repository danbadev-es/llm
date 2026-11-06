"""Phase 4: pretrain Leon 0.1 (56M, context 1024, ~2B tokens of Spanish Wikipedia).

Usually driven by control.py (start / pause / resume / status), but it can also be run
directly. It can be stopped at any moment without losing work:
  - `control.py pause` (or creating the PAUSE file) or Ctrl+C: finishes the current step,
    saves last.pt and exits;
  - `--resume` continues from last.pt with the same batches an uninterrupted run would see.
"""

import argparse
import os
import time
from dataclasses import asdict

import torch

from leon.data import DATASETS, load_splits, load_tokenizer
from leon.model import Leon, LeonConfig
from leon.paths import CHECKPOINT_DIR
from leon.run import RunFiles, keep_system_awake
from leon.training import bits_per_byte, estimate_loss, load_checkpoint, lr_at, save_checkpoint

CHECKPOINT_ROOT = CHECKPOINT_DIR / "4_pretrain"
FIXED_PROMPT = "En un lugar de la Mancha"
STATUS_EVERY = 10  # steps between status.json updates (~30 s at full size)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run", default="leon_0.1", help="checkpoints go to checkpoints/4_pretrain/<run>/")
    parser.add_argument("--dataset", choices=list(DATASETS), default="wiki")
    # Model: Leon 0.1, fixed in phase 3 from measured throughput
    parser.add_argument("--n-layer", type=int, default=10)
    parser.add_argument("--n-head", type=int, default=10)
    parser.add_argument("--n-kv-head", type=int, default=5)
    parser.add_argument("--n-embd", type=int, default=640)
    parser.add_argument("--block-size", type=int, default=1024)
    parser.add_argument("--dropout", type=float, default=0.0)
    # Optimization: 4 x 1024 = 4,096 tokens per micro-batch, x32 = 131,072 tokens per step
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--grad-accum", type=int, default=32)
    parser.add_argument("--max-steps", type=int, default=15_259, help="15,259 x 131,072 tokens = 2.0B tokens")
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--min-lr", type=float, default=1e-4)
    parser.add_argument("--warmup-steps", type=int, default=500)
    parser.add_argument("--weight-decay", type=float, default=0.1)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    # Evaluation, checkpoints, runtime
    parser.add_argument("--eval-interval", type=int, default=250, help="~13 min at full size")
    parser.add_argument("--eval-iters", type=int, default=40)
    parser.add_argument("--resume", action="store_true", help="continue from <run>/last.pt")
    parser.add_argument("--overwrite", action="store_true", help="start from scratch even if <run>/last.pt exists")
    parser.add_argument("--no-compile", action="store_true", help="skip torch.compile (slower, starts faster)")
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args(argv)


# Everything that defines the run. On --resume these come from the checkpoint, so a resumed run
# follows the same schedule and data no matter what the command line says.
RUN_SETTINGS = (
    "dataset", "batch_size", "grad_accum", "max_steps", "lr", "min_lr", "warmup_steps",
    "weight_decay", "grad_clip", "eval_interval", "eval_iters", "seed",
)  # fmt: skip


def main() -> None:
    args = parse_args()
    run = RunFiles(CHECKPOINT_ROOT / args.run)
    if not args.resume and run.last.exists() and not args.overwrite:
        raise SystemExit(f"{run.last} already exists: use --resume to continue it, or --overwrite to start over.")
    if args.resume:
        if not run.last.exists():
            raise SystemExit(f"Nothing to resume: {run.last} does not exist.")
        saved = torch.load(run.last, map_location="cpu", weights_only=True, mmap=True)["settings"]
        for key in RUN_SETTINGS:
            setattr(args, key, saved[key])
    run.dir.mkdir(parents=True, exist_ok=True)
    run.pid.write_text(str(os.getpid()))
    run.clear_pause()  # a stale request from an earlier run must not stop this one
    keep_system_awake()

    torch.manual_seed(args.seed)
    device = args.device
    use_bf16 = device.startswith("cuda") and torch.cuda.is_bf16_supported()

    def autocast():
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=use_bf16)

    if device.startswith("cuda"):
        torch.backends.cuda.matmul.allow_tf32 = True

    tokenizer = load_tokenizer()
    splits = load_splits(args.dataset)
    eval_splits = dict(splits)
    if args.dataset != "quijote":
        eval_splits["quijote"] = load_splits("quijote")["val"]  # out of domain, comparable across phases

    config = LeonConfig(
        vocab_size=tokenizer.vocab_size,
        block_size=args.block_size,
        n_layer=args.n_layer,
        n_head=args.n_head,
        n_kv_head=args.n_kv_head,
        n_embd=args.n_embd,
        dropout=args.dropout,
    )
    if args.resume:  # the architecture always comes from the checkpoint
        config = LeonConfig(**torch.load(run.last, map_location="cpu", weights_only=True, mmap=True)["config"])
    model = Leon(config).to(device)
    optimizer = model.configure_optimizer(args.lr, args.weight_decay, (0.9, 0.95), device)

    start_step, best_val, last_val = 0, float("inf"), None
    if args.resume:
        checkpoint = load_checkpoint(run.last, model, optimizer, device)  # also restores the RNG
        start_step, best_val = checkpoint["step"], checkpoint["best_val"]
        print(f"Resumed from step {start_step} (best val {best_val:.4f})", flush=True)

    train_model = model if args.no_compile else torch.compile(model)
    tokens_per_step = args.batch_size * args.grad_accum * config.block_size
    settings = {**vars(args), "model": asdict(config)}

    def schedule(step: int) -> float:
        return lr_at(step, args.lr, args.min_lr, args.warmup_steps, args.max_steps)

    print(
        f"Params: {model.num_params() / 1e6:.2f}M ({model.num_params(non_embedding=True) / 1e6:.2f}M non-embedding) "
        f"| dataset: {args.dataset} ({len(splits['train']) / 1e6:,.0f}M train tokens) | bf16: {use_bf16} "
        f"| compile: {not args.no_compile}",
        flush=True,
    )
    print(
        f"tokens/step: {tokens_per_step:,} | steps {start_step:,} -> {args.max_steps:,} "
        f"| total: {tokens_per_step * args.max_steps / 1e9:.2f}B tokens\n",
        flush=True,
    )

    session_start = time.perf_counter()
    train_time, steps_timed, tok_per_s = 0.0, 0, 0.0
    loss_sum = torch.zeros((), device=device)  # summed on the GPU, read only for status
    sample = ""

    def report(state: str, step: int) -> None:
        remaining = (args.max_steps - step) * tokens_per_step
        run.write_status(
            state=state,
            run=args.run,
            step=step,
            max_steps=args.max_steps,
            progress=round(step / args.max_steps, 4),
            tokens_seen=step * tokens_per_step,
            train_loss=round(loss_sum.item() / max(1, steps_timed * args.grad_accum), 4) if steps_timed else None,
            val_loss=last_val,
            val_bits_per_byte=round(bits_per_byte(last_val, splits["val"].bytes_per_token), 4) if last_val else None,
            best_val=best_val if best_val < float("inf") else None,
            tok_per_s=round(tok_per_s),
            eta_hours=round(remaining / tok_per_s / 3600, 2) if tok_per_s else None,
            session_hours=round((time.perf_counter() - session_start) / 3600, 2),
            sample=sample,
            pid=os.getpid(),
        )

    def save_and_exit(step: int, reason: str) -> None:
        save_checkpoint(run.last, model, step, best_val, optimizer, settings)
        run.clear_pause()
        report("paused", step)
        print(f"\n{reason}: saved {run.last} at step {step}. Continue with: control.py resume", flush=True)

    step = start_step
    try:
        for step in range(start_step, args.max_steps + 1):
            already_evaluated = args.resume and step == start_step  # evaluated (and saved) before pausing
            if (step % args.eval_interval == 0 or step == args.max_steps) and not already_evaluated:
                losses = estimate_loss(
                    model, eval_splits, args.batch_size, config.block_size, args.eval_iters, device, autocast
                )
                last_val = losses["val"]
                line = (
                    f"step {step:6d} | train {losses['train']:.4f} | val {losses['val']:.4f} "
                    f"| val bits/byte {bits_per_byte(losses['val'], splits['val'].bytes_per_token):.3f}"
                )
                if "quijote" in losses:
                    quijote_bpb = bits_per_byte(losses["quijote"], eval_splits["quijote"].bytes_per_token)
                    line += f" | quijote bits/byte {quijote_bpb:.3f}"
                line += f" | lr {schedule(step):.2e} | {tok_per_s:,.0f} tok/s"
                print(line, flush=True)
                if losses["val"] < best_val:
                    best_val = losses["val"]
                    save_checkpoint(run.best, model, step, best_val, settings=settings)
                save_checkpoint(run.last, model, step, best_val, optimizer, settings)
                prompt = torch.tensor([tokenizer.encode(FIXED_PROMPT)], device=device)
                with autocast():
                    out = model.generate(prompt, 60, temperature=0.8, top_k=40)
                sample = tokenizer.decode(out[0].tolist())
                print(f"  sample: {sample!r}", flush=True)
            if step == args.max_steps:
                break
            if run.pause_requested():
                save_and_exit(step, "Paused")
                return

            if device.startswith("cuda"):
                torch.cuda.synchronize()
            step_start = time.perf_counter()
            for group in optimizer.param_groups:
                group["lr"] = schedule(step)
            for _ in range(args.grad_accum):
                x, y = splits["train"].get_batch(args.batch_size, config.block_size, device)
                with autocast():
                    _, loss = train_model(x, y)
                (loss / args.grad_accum).backward()
                loss_sum += loss.detach()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            if device.startswith("cuda"):
                torch.cuda.synchronize()
            train_time += time.perf_counter() - step_start
            steps_timed += 1

            if (step + 1) % STATUS_EVERY == 0:
                tok_per_s = steps_timed * tokens_per_step / train_time
                report("running", step + 1)
                loss_sum.zero_()
                train_time, steps_timed = 0.0, 0
    except KeyboardInterrupt:  # Ctrl+C in a console: same as a pause
        save_and_exit(step, "Interrupted")
        return
    finally:
        run.pid.unlink(missing_ok=True)

    report("finished", args.max_steps)
    print(
        f"\nFinished. Best val loss: {best_val:.4f} ({bits_per_byte(best_val, splits['val'].bytes_per_token):.3f} "
        f"bits/byte). Best model: {run.best}",
        flush=True,
    )
    if device.startswith("cuda"):
        print(f"Peak VRAM allocated: {torch.cuda.max_memory_allocated() / 2**30:.2f} GiB", flush=True)


if __name__ == "__main__":
    main()
