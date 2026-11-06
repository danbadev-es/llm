"""Measure real training throughput of candidate Leon 0.1 sizes on this GPU.

Each candidate runs a few full training steps (forward, backward, AdamW) on random tokens
with bf16 autocast, exactly like train.py, and reports tokens/s, effective TFLOP/s, peak
VRAM and the projected time to train on a given number of tokens. Data loading is left out
on purpose: it measures the model, not the disk.
"""

import argparse
import time

import torch

from leon.model import Leon, LeonConfig

# name: (n_layer, n_embd, n_head, n_kv_head)
CANDIDATES = {
    "20M d384 L8": (8, 384, 6, 2),
    "32M d512 L8": (8, 512, 8, 4),
    "44M d512 L12": (12, 512, 8, 4),
    "56M d640 L10": (10, 640, 10, 5),
    "63M d768 L8": (8, 768, 12, 4),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--block-sizes", type=int, nargs="+", default=[512, 1024])
    parser.add_argument("--tokens-per-micro-batch", type=int, default=16384)
    parser.add_argument("--warmup-steps", type=int, default=5)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--train-tokens", type=float, default=2e9, help="for the projected training time")
    parser.add_argument("--only", nargs="*", help="subset of candidate names")
    parser.add_argument("--compile", action="store_true", help="torch.compile (on Windows needs triton-windows)")
    return parser.parse_args()


def benchmark(config: LeonConfig, batch_size: int, args: argparse.Namespace) -> dict:
    torch.manual_seed(0)
    model = Leon(config).cuda()
    optimizer = model.configure_optimizer(1e-3, 0.1, (0.9, 0.95), "cuda")
    train_model = torch.compile(model) if args.compile else model
    x = torch.randint(config.vocab_size, (batch_size, config.block_size), device="cuda")
    y = torch.randint(config.vocab_size, (batch_size, config.block_size), device="cuda")

    def step() -> None:
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            _, loss = train_model(x, y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)

    for _ in range(args.warmup_steps):  # first steps include allocation, kernel selection and compilation
        step()
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    start = time.perf_counter()
    for _ in range(args.steps):
        step()
    torch.cuda.synchronize()  # GPU work is asynchronous: wait before stopping the clock
    elapsed = time.perf_counter() - start

    tok_per_s = args.steps * batch_size * config.block_size / elapsed
    return {
        "params": model.num_params(),
        "tok_per_s": tok_per_s,
        "tflops": tok_per_s * model.flops_per_token() / 1e12,
        "vram_gib": torch.cuda.max_memory_allocated() / 2**30,
    }  # the model and optimizer are freed when this function returns


def main() -> None:
    args = parse_args()
    torch.backends.cuda.matmul.allow_tf32 = True
    print(f"GPU: {torch.cuda.get_device_name(0)} | {args.tokens_per_micro_batch:,} tokens per micro-batch\n")
    print(
        f"{'candidate':14s} {'params':>7s} {'ctx':>5s} {'batch':>5s} {'tok/s':>8s} {'TFLOP/s':>7s} "
        f"{'VRAM':>8s} {f'h@{args.train_tokens / 1e9:g}B':>7s} {'h@20N':>6s}"
    )
    for name, (n_layer, n_embd, n_head, n_kv_head) in CANDIDATES.items():
        if args.only and name not in args.only:
            continue
        for block_size in args.block_sizes:
            config = LeonConfig(
                block_size=block_size, n_layer=n_layer, n_embd=n_embd, n_head=n_head, n_kv_head=n_kv_head
            )
            batch_size = max(1, args.tokens_per_micro_batch // block_size)
            try:
                r = benchmark(config, batch_size, args)
            except torch.OutOfMemoryError:
                print(f"{name:14s} {'':>7s} {block_size:5d} {batch_size:5d}  out of memory")
                continue
            finally:
                torch.cuda.empty_cache()  # start every candidate with a clean allocator
            hours = args.train_tokens / r["tok_per_s"] / 3600
            chinchilla_hours = 20 * r["params"] / r["tok_per_s"] / 3600  # ~20 tokens per parameter
            print(
                f"{name:14s} {r['params'] / 1e6:6.1f}M {block_size:5d} {batch_size:5d} {r['tok_per_s']:8,.0f} "
                f"{r['tflops']:7.1f} {r['vram_gib']:6.2f}GiB {hours:7.1f} {chinchilla_hours:6.1f}",
                flush=True,
            )


if __name__ == "__main__":
    main()
