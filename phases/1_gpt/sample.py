"""Generate text from a trained phase 1 checkpoint."""

import argparse
from pathlib import Path

import torch
from model import GPT, GPTConfig

from data import CharTokenizer

DEFAULT_CHECKPOINT = Path(__file__).resolve().parents[2] / "checkpoints" / "1_gpt" / "best.pt"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prompt", default="En un lugar de la Mancha")
    parser.add_argument("--max-new-tokens", type=int, default=500)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=40)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    if args.seed is not None:
        torch.manual_seed(args.seed)

    checkpoint = torch.load(args.checkpoint, map_location=args.device, weights_only=True)
    tokenizer = CharTokenizer(checkpoint["chars"])
    model = GPT(GPTConfig(**checkpoint["config"])).to(args.device)
    model.load_state_dict(checkpoint["model"])
    model.eval()

    unknown = set(args.prompt) - set(tokenizer.chars)
    if unknown:
        raise SystemExit(f"Prompt contains characters outside the vocabulary: {sorted(unknown)}")

    prompt = torch.tensor([tokenizer.encode(args.prompt)], dtype=torch.long, device=args.device)
    output = model.generate(prompt, args.max_new_tokens, args.temperature, args.top_k or None)
    print(tokenizer.decode(output[0].tolist()))


if __name__ == "__main__":
    main()
