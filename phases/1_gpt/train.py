"""Phase 1: train a minimal character-level GPT on Don Quijote."""

import argparse
import math
import os
import time
from dataclasses import asdict
from pathlib import Path

import torch
from model import GPT, GPTConfig

from data import CharTokenizer, get_batch, load_corpus, train_val_split

CHECKPOINT_DIR = Path(__file__).resolve().parents[2] / "checkpoints" / "1_gpt"
FIXED_PROMPT = "En un lugar de la Mancha"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    # Model
    parser.add_argument("--n-layer", type=int, default=6)
    parser.add_argument("--n-head", type=int, default=6)
    parser.add_argument("--n-embd", type=int, default=384)
    parser.add_argument("--block-size", type=int, default=256)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--sdpa", action="store_true", help="use the fused attention kernel")
    # Optimization
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--grad-accum", type=int, default=1, help="micro-batches per optimizer step")
    parser.add_argument("--max-steps", type=int, default=5000)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--min-lr", type=float, default=1e-4)
    parser.add_argument("--warmup-steps", type=int, default=200)
    parser.add_argument("--weight-decay", type=float, default=0.1)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    # Evaluation, checkpoints, runtime
    parser.add_argument("--eval-interval", type=int, default=250)
    parser.add_argument("--eval-iters", type=int, default=50)
    parser.add_argument("--resume", action="store_true", help="continue from checkpoints/1_gpt/last.pt")
    parser.add_argument("--compile", action="store_true", help="torch.compile (on Windows needs triton-windows)")
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def lr_at(step: int, args: argparse.Namespace) -> float:
    """Linear warmup followed by cosine decay down to min_lr."""
    if step < args.warmup_steps:
        return args.lr * (step + 1) / args.warmup_steps
    progress = min(1.0, (step - args.warmup_steps) / max(1, args.max_steps - args.warmup_steps))
    return args.min_lr + 0.5 * (1.0 + math.cos(math.pi * progress)) * (args.lr - args.min_lr)


@torch.no_grad()
def estimate_loss(model: GPT, splits: dict[str, torch.Tensor], args: argparse.Namespace, autocast) -> dict[str, float]:
    model.eval()
    result = {}
    for name, data in splits.items():
        losses = torch.zeros(args.eval_iters)
        for k in range(args.eval_iters):
            x, y = get_batch(data, args.batch_size, args.block_size)
            with autocast():
                _, loss = model(x, y)
            losses[k] = loss.item()
        result[name] = losses.mean().item()
    model.train()
    return result


def save_checkpoint(path: Path, model: GPT, optimizer, step: int, best_val: float, tokenizer: CharTokenizer) -> None:
    """Save atomically (temporary file, then rename) together with the random number generators,
    so that --resume continues with the same batches an uninterrupted run would have drawn."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "config": asdict(model.config),
            "step": step,
            "best_val": best_val,
            "chars": tokenizer.chars,
            "rng_cpu": torch.get_rng_state(),
            "rng_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
        },
        tmp,
    )
    os.replace(tmp, path)


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    device = args.device

    use_bf16 = device.startswith("cuda") and torch.cuda.is_bf16_supported()

    def autocast():
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=use_bf16)

    if device.startswith("cuda"):
        torch.backends.cuda.matmul.allow_tf32 = True

    text = load_corpus()
    tokenizer = CharTokenizer.from_text(text)
    data = torch.tensor(tokenizer.encode(text), dtype=torch.long, device=device)
    splits = train_val_split(data)

    config = GPTConfig(
        vocab_size=tokenizer.vocab_size,
        block_size=args.block_size,
        n_layer=args.n_layer,
        n_head=args.n_head,
        n_embd=args.n_embd,
        dropout=args.dropout,
        use_sdpa=args.sdpa,
    )
    checkpoint = None
    if args.resume:
        checkpoint = torch.load(CHECKPOINT_DIR / "last.pt", map_location=device, weights_only=True)
        config = GPTConfig(**checkpoint["config"])  # the architecture comes from the checkpoint, not the CLI
        args.block_size = config.block_size  # batches must match the checkpoint's context length
    model = GPT(config).to(device)
    optimizer = model.configure_optimizer(args.lr, args.weight_decay, (0.9, 0.95), device)

    start_step, best_val = 0, float("inf")
    if checkpoint is not None:
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        start_step, best_val = checkpoint["step"], checkpoint["best_val"]
        if "rng_cpu" in checkpoint:  # absent in checkpoints saved before this was added
            torch.set_rng_state(checkpoint["rng_cpu"].cpu())
        if checkpoint.get("rng_cuda") and torch.cuda.is_available():
            torch.cuda.set_rng_state_all([state.cpu() for state in checkpoint["rng_cuda"]])
        print(f"Resumed from step {start_step} (best val {best_val:.4f})")

    train_model = torch.compile(model) if args.compile else model

    tokens_per_step = args.batch_size * args.grad_accum * config.block_size
    print(
        f"Params: {model.num_params() / 1e6:.2f}M | vocab: {tokenizer.vocab_size} | device: {device} "
        f"| bf16: {use_bf16} | tokens/step: {tokens_per_step:,}"
    )
    print(f"Uniform baseline loss ln(V) = {math.log(tokenizer.vocab_size):.4f}\n")

    step_start, last_eval_step = time.perf_counter(), start_step
    for step in range(start_step, args.max_steps + 1):
        # On resume, the start step was already evaluated and saved: evaluating it again would
        # consume random numbers and change every batch that follows.
        already_evaluated = args.resume and step == start_step
        if (step % args.eval_interval == 0 or step == args.max_steps) and not already_evaluated:
            losses = estimate_loss(train_model, splits, args, autocast)
            elapsed = time.perf_counter() - step_start
            steps_done = step - last_eval_step
            throughput = steps_done * tokens_per_step / elapsed if steps_done else 0.0
            print(
                f"step {step:5d} | train {losses['train']:.4f} | val {losses['val']:.4f} "
                f"| val bits/char {losses['val'] / math.log(2):.3f} | lr {lr_at(step, args):.2e} "
                f"| {throughput:,.0f} tok/s"
            )
            if losses["val"] < best_val:
                best_val = losses["val"]
                save_checkpoint(CHECKPOINT_DIR / "best.pt", model, optimizer, step, best_val, tokenizer)
            save_checkpoint(CHECKPOINT_DIR / "last.pt", model, optimizer, step, best_val, tokenizer)
            step_start, last_eval_step = time.perf_counter(), step
        if step == args.max_steps:
            break

        for group in optimizer.param_groups:
            group["lr"] = lr_at(step, args)

        for _ in range(args.grad_accum):
            x, y = get_batch(splits["train"], args.batch_size, args.block_size)
            with autocast():
                _, loss = train_model(x, y)
            (loss / args.grad_accum).backward()  # gradients accumulate across micro-batches

        torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)

    print(f"\nBest val loss: {best_val:.4f} ({best_val / math.log(2):.3f} bits/char)")
    if device.startswith("cuda"):
        print(f"Peak VRAM allocated: {torch.cuda.max_memory_allocated() / 2**30:.2f} GiB")
    best = torch.load(CHECKPOINT_DIR / "best.pt", map_location=device, weights_only=True)
    model.load_state_dict(best["model"])
    model.eval()
    prompt = torch.tensor([tokenizer.encode(FIXED_PROMPT)], dtype=torch.long, device=device)
    with autocast():
        sample = model.generate(prompt, max_new_tokens=500, temperature=0.8, top_k=40)
    print(f"\n--- Sample (best checkpoint, temperature=0.8, top_k=40) ---\n{tokenizer.decode(sample[0].tolist())}")


if __name__ == "__main__":
    main()
