"""Phase 3: train Leon (RoPE, RMSNorm, SwiGLU, GQA) on BPE tokens read with np.memmap."""

import argparse
import math
import time
from dataclasses import asdict

import torch

from leon.data import DATASETS, TokenFile, load_splits, load_tokenizer
from leon.model import Leon, LeonConfig
from leon.paths import CHECKPOINT_DIR
from leon.training import bits_per_byte, load_checkpoint, lr_at, save_checkpoint

CHECKPOINT_ROOT = CHECKPOINT_DIR / "3_modern"
FIXED_PROMPT = "En un lugar de la Mancha"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", default="wiki_32m", help="checkpoints go to checkpoints/3_modern/<run>/")
    parser.add_argument("--dataset", choices=list(DATASETS), default="wiki")
    # Model
    parser.add_argument("--n-layer", type=int, default=8)
    parser.add_argument("--n-head", type=int, default=8)
    parser.add_argument("--n-kv-head", type=int, default=4)
    parser.add_argument("--n-embd", type=int, default=512)
    parser.add_argument("--block-size", type=int, default=512)
    parser.add_argument("--dropout", type=float, default=0.0)
    # Optimization
    parser.add_argument(
        "--batch-size", type=int, default=8, help="sequences per micro-batch (4096 tokens fit well in 8 GB)"
    )
    parser.add_argument("--grad-accum", type=int, default=16, help="micro-batches per optimizer step")
    parser.add_argument("--max-steps", type=int, default=2000)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--min-lr", type=float, default=1e-4)
    parser.add_argument("--warmup-steps", type=int, default=200)
    parser.add_argument("--weight-decay", type=float, default=0.1)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    # Evaluation, checkpoints, runtime
    parser.add_argument("--eval-interval", type=int, default=100)
    parser.add_argument("--eval-iters", type=int, default=40)
    parser.add_argument("--resume", action="store_true", help="continue from <run>/last.pt")
    parser.add_argument("--compile", action="store_true", help="torch.compile (on Windows needs triton-windows)")
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


@torch.no_grad()
def estimate_loss(
    model: Leon, splits: dict[str, TokenFile], block_size: int, args: argparse.Namespace, autocast
) -> dict[str, float]:
    model.eval()
    result = {}
    for name, split in splits.items():
        losses = torch.zeros(args.eval_iters)
        for k in range(args.eval_iters):
            x, y = split.get_batch(args.batch_size, block_size, args.device)
            with autocast():
                _, loss = model(x, y)
            losses[k] = loss.item()
        result[name] = losses.mean().item()
    model.train()
    return result


def main() -> None:
    args = parse_args()
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
        # Out-of-domain check. Phases 0-1 report bits per character: multiply bits/byte by the
        # Quijote's 1.028 bytes/char to compare.
        eval_splits["quijote"] = load_splits("quijote")["val"]

    run_dir = CHECKPOINT_ROOT / args.run
    config = LeonConfig(
        vocab_size=tokenizer.vocab_size,
        block_size=args.block_size,
        n_layer=args.n_layer,
        n_head=args.n_head,
        n_kv_head=args.n_kv_head,
        n_embd=args.n_embd,
        dropout=args.dropout,
    )
    if args.resume:
        # The architecture comes from the checkpoint, never from the command line: resuming a
        # ctx-1024 run without --block-size 1024 must not silently continue at 512.
        saved = LeonConfig(
            **torch.load(run_dir / "last.pt", map_location="cpu", weights_only=True, mmap=True)["config"]
        )
        if saved != config:
            print(f"Note: using the model config stored in the checkpoint, not the command line: {saved}")
        config = saved
    model = Leon(config).to(device)
    optimizer = model.configure_optimizer(args.lr, args.weight_decay, (0.9, 0.95), device)

    start_step, best_val = 0, float("inf")
    if args.resume:
        checkpoint = load_checkpoint(run_dir / "last.pt", model, optimizer, device)  # also restores the RNG
        start_step, best_val = checkpoint["step"], checkpoint["best_val"]
        print(f"Resumed from step {start_step} (best val {best_val:.4f})")

    train_model = torch.compile(model) if args.compile else model

    tokens_per_step = args.batch_size * args.grad_accum * config.block_size

    def schedule(step: int) -> float:
        return lr_at(step, args.lr, args.min_lr, args.warmup_steps, args.max_steps)

    settings = {**vars(args), "model": asdict(config)}
    print(
        f"Params: {model.num_params() / 1e6:.2f}M ({model.num_params(non_embedding=True) / 1e6:.2f}M non-embedding) "
        f"| dataset: {args.dataset} ({len(splits['train']) / 1e6:.1f}M train tokens) | bf16: {use_bf16}"
    )
    print(
        f"tokens/step: {tokens_per_step:,} | total: {tokens_per_step * args.max_steps / 1e6:.0f}M tokens "
        f"| uniform baseline ln(V) = {math.log(config.vocab_size):.4f}\n"
    )

    train_time, steps_timed = 0.0, 0
    for step in range(start_step, args.max_steps + 1):
        # On resume, the start step was already evaluated and saved: evaluating it again would
        # consume random numbers and change every batch that follows.
        already_evaluated = args.resume and step == start_step
        if (step % args.eval_interval == 0 or step == args.max_steps) and not already_evaluated:
            # The uncompiled model: same weights, and no recompilation for eval mode.
            losses = estimate_loss(model, eval_splits, config.block_size, args, autocast)
            throughput = steps_timed * tokens_per_step / train_time if train_time else 0.0
            line = (
                f"step {step:5d} | train {losses['train']:.4f} | val {losses['val']:.4f} "
                f"| val bits/byte {bits_per_byte(losses['val'], splits['val'].bytes_per_token):.3f}"
            )
            if "quijote" in losses:
                quijote_bpb = bits_per_byte(losses["quijote"], eval_splits["quijote"].bytes_per_token)
                line += f" | quijote bits/byte {quijote_bpb:.3f}"
            line += (
                f" | lr {schedule(step):.2e} | {throughput:,.0f} tok/s "
                f"| {throughput * model.flops_per_token() / 1e12:.1f} TFLOP/s"
            )
            print(line, flush=True)
            if losses["val"] < best_val:
                best_val = losses["val"]
                save_checkpoint(run_dir / "best.pt", model, step, best_val, settings=settings)  # weights only
            save_checkpoint(run_dir / "last.pt", model, step, best_val, optimizer, settings)  # resumable
            train_time, steps_timed = 0.0, 0
        if step == args.max_steps:
            break

        if device.startswith("cuda"):
            torch.cuda.synchronize()
        step_start = time.perf_counter()  # throughput counts training only, not evaluation
        for group in optimizer.param_groups:
            group["lr"] = schedule(step)
        for _ in range(args.grad_accum):
            x, y = splits["train"].get_batch(args.batch_size, config.block_size, device)
            with autocast():
                _, loss = train_model(x, y)
            (loss / args.grad_accum).backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        if device.startswith("cuda"):
            torch.cuda.synchronize()
        train_time += time.perf_counter() - step_start
        steps_timed += 1

    print(f"\nBest val loss: {best_val:.4f} ({bits_per_byte(best_val, splits['val'].bytes_per_token):.3f} bits/byte)")
    if device.startswith("cuda"):
        print(f"Peak VRAM allocated: {torch.cuda.max_memory_allocated() / 2**30:.2f} GiB")
    best = torch.load(run_dir / "best.pt", map_location=device, weights_only=True)
    model.load_state_dict(best["model"])
    model.eval()
    prompt = torch.tensor([tokenizer.encode(FIXED_PROMPT)], dtype=torch.long, device=device)
    with autocast():
        sample = model.generate(prompt, max_new_tokens=200, temperature=0.8, top_k=40)
    print(f"\n--- Sample (best checkpoint, temperature=0.8, top_k=40) ---\n{tokenizer.decode(sample[0].tolist())}")


if __name__ == "__main__":
    main()
