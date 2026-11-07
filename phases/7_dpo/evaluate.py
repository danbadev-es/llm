"""Phase 7: compare answers before and after DPO on held-out prompts (data/dpo/val.jsonl).

For each model, one sampled answer per prompt (same seed and settings for both), and:

    repite       share of answers stuck in a loop: a 4-word sequence that appears 3+ times
    termina      share that end on their own (<|end|>) instead of running out of tokens
    palabras     mean length in words

How often the trained model prefers the human answer (reward accuracy) is in its training
dashboard and log, measured on the same val pairs.

    python phases/7_dpo/evaluate.py                         # before: the DPO base, after: DPO's best
    python phases/7_dpo/evaluate.py --before X.pt --after Y.pt
"""

import argparse
import json
from collections import Counter
from pathlib import Path

import torch

from leon.chat import encode_chat
from leon.data import load_tokenizer
from leon.model import Leon, LeonConfig
from leon.paths import CHECKPOINT_DIR, DATA_DIR

AFTER = CHECKPOINT_DIR / "7_dpo" / "leon_0.1_dpo" / "best.pt"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--before", type=Path, default=None, help="default: the base the DPO run started from")
    parser.add_argument("--after", type=Path, default=AFTER)
    parser.add_argument("--prompts", type=int, default=100)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def loops(text: str, n: int = 4, times: int = 3) -> bool:
    words = text.lower().split()
    grams = Counter(tuple(words[i : i + n]) for i in range(len(words) - n + 1))
    return any(count >= times for count in grams.values())


@torch.no_grad()
def answers(path: Path, prompts: list[list[dict]], args) -> list[tuple[str, bool]]:
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    model = Leon(LeonConfig(**checkpoint["config"])).to(args.device).eval()
    model.load_state_dict(checkpoint["model"])
    tokenizer = load_tokenizer(checkpoint.get("tokenizer", "v1"))
    stop = set(tokenizer.special_tokens.values())
    use_bf16 = args.device.startswith("cuda") and torch.cuda.is_bf16_supported()
    torch.manual_seed(args.seed)
    out = []
    for prompt in prompts:
        idx = torch.tensor([encode_chat(tokenizer, prompt)], device=args.device)
        tokens, ended = [], False
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=use_bf16):
            for token in model.stream(idx, args.max_new_tokens, temperature=args.temperature, top_k=40, stop_ids=stop):
                if token.item() in stop:
                    ended = True
                    break
                tokens.append(token.item())
        out.append((tokenizer.decode(tokens).strip(), ended))
    return out


def main() -> None:
    args = parse_args()
    if args.before is None:
        settings = torch.load(args.after, map_location="cpu", weights_only=True).get("settings", {})
        args.before = Path(settings["base"])
    items = [json.loads(line) for line in open(DATA_DIR / "dpo" / "val.jsonl", encoding="utf-8")]
    seen, prompts = set(), []
    for item in items:  # one per prompt (prepare.py keeps up to 2 pairs per prompt)
        key = item["prompt"][-1]["content"]
        if key not in seen:
            seen.add(key)
            prompts.append(item["prompt"])
    prompts = prompts[: args.prompts]

    print(f"{len(prompts)} prompts, temperature {args.temperature}, up to {args.max_new_tokens} tokens\n")
    print("| Modelo | Repite | Termina | Palabras |")
    print("|---|---|---|---|")
    examples = {}
    for name, path in (("antes", args.before), ("después", args.after)):
        results = answers(path, prompts, args)
        n = len(results)
        repeat = sum(loops(text) for text, _ in results) / n
        ended = sum(end for _, end in results) / n
        words = sum(len(text.split()) for text, _ in results) / n
        model = f"{path.parent.parent.name}/{path.parent.name}"
        print(f"| {name} (`{model}`) | {100 * repeat:.0f} % | {100 * ended:.0f} % | {words:.0f} |")
        examples[name] = [text for text, _ in results[:5]]
    print("\nPrimeras respuestas:")
    for i, prompt in enumerate(prompts[:5]):
        print(f"\n» {prompt[-1]['content'][:150]}")
        for name in examples:
            print(f"  {name}: {examples[name][i][:200]!r}")


if __name__ == "__main__":
    main()
