"""Generate text from a phase 3 checkpoint, with or without the KV cache."""

import argparse
import time
from pathlib import Path

import torch

from leon.data import load_tokenizer
from leon.model import Leon, LeonConfig
from leon.paths import CHECKPOINT_DIR

CHECKPOINT_ROOT = CHECKPOINT_DIR / "3_modern"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", default="wiki_32m")
    parser.add_argument("--checkpoint", type=Path, default=None, help="default: <run>/best.pt")
    parser.add_argument("--prompt", default="En un lugar de la Mancha")
    parser.add_argument("--max-new-tokens", type=int, default=200)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=40)
    parser.add_argument("--no-cache", action="store_true", help="recompute the whole context at every step")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.seed is not None:
        torch.manual_seed(args.seed)
    path = args.checkpoint or CHECKPOINT_ROOT / args.run / "best.pt"
    checkpoint = torch.load(path, map_location=args.device, weights_only=True)
    model = Leon(LeonConfig(**checkpoint["config"])).to(args.device).eval()
    model.load_state_dict(checkpoint["model"])
    tokenizer = load_tokenizer(checkpoint.get("tokenizer", "v1"))  # the version the model was trained with

    prompt = torch.tensor([tokenizer.encode(args.prompt)], dtype=torch.long, device=args.device)
    use_bf16 = args.device.startswith("cuda") and torch.cuda.is_bf16_supported()
    start = time.perf_counter()
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=use_bf16):
        out = model.generate(
            prompt, args.max_new_tokens, temperature=args.temperature, top_k=args.top_k, use_cache=not args.no_cache
        )
    if args.device.startswith("cuda"):
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    new_tokens = out.size(1) - prompt.size(1)
    print(tokenizer.decode(out[0].tolist()))
    print(f"\n[{new_tokens} tokens in {elapsed:.2f}s = {new_tokens / elapsed:.0f} tok/s, cache: {not args.no_cache}]")


if __name__ == "__main__":
    main()
