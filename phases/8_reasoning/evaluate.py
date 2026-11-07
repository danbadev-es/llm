"""Phase 8: accuracy by level on held-out problems (data/reasoning/eval.jsonl), greedy decoding.

    python phases/8_reasoning/evaluate.py      # every phase 8 model that exists: direct, steps, grpo
    python phases/8_reasoning/evaluate.py --checkpoints a.pt b.pt

The comparison that matters: the same problems answered directly, with steps, and with steps
after GRPO. Levels: 1 sums, 2 two digits × one, 3 two × two, 4 mixed with parentheses, 5 word problems.
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path

import torch

from leon.chat import encode_chat
from leon.data import load_tokenizer
from leon.grpo import extract_answer
from leon.model import Leon, LeonConfig
from leon.paths import CHECKPOINT_DIR, DATA_DIR

ROOT = CHECKPOINT_DIR / "8_reasoning"
DEFAULTS = [ROOT / run / "best.pt" for run in ("leon_0.1_direct", "leon_0.1_steps", "leon_0.1_grpo")]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoints", type=Path, nargs="+", default=[p for p in DEFAULTS if p.exists()])
    parser.add_argument("--per-level", type=int, default=100)
    parser.add_argument("--max-new-tokens", type=int, default=200)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


@torch.no_grad()
def solve(path: Path, problems: list[dict], args) -> tuple[list[bool], str]:
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    model = Leon(LeonConfig(**checkpoint["config"])).to(args.device).eval()
    model.load_state_dict(checkpoint["model"])
    tokenizer = load_tokenizer(checkpoint.get("tokenizer", "v1"))
    stop = set(tokenizer.special_tokens.values())
    use_bf16 = args.device.startswith("cuda") and torch.cuda.is_bf16_supported()
    right, example = [], ""
    for item in problems:
        idx = torch.tensor(
            [encode_chat(tokenizer, [{"role": "user", "content": item["question"]}])], device=args.device
        )
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=use_bf16):
            tokens = [t.item() for t in model.stream(idx, args.max_new_tokens, temperature=0.0, stop_ids=stop)]
        text = tokenizer.decode([t for t in tokens if t not in stop])
        right.append(extract_answer(text) == item["answer"])
        if not example and item["level"] == 3:
            example = f"{item['question']}\n{text}"
    return right, example


def main() -> None:
    args = parse_args()
    if not args.checkpoints:
        raise SystemExit("No phase 8 model yet: train one with phases/8_reasoning/control.py start")
    by_level = defaultdict(list)
    for line in open(DATA_DIR / "reasoning" / "eval.jsonl", encoding="utf-8"):
        item = json.loads(line)
        if len(by_level[item["level"]]) < args.per_level:
            by_level[item["level"]].append(item)
    problems = [item for level in sorted(by_level) for item in by_level[level]]
    levels = sorted(by_level)

    print("| Modelo | " + " | ".join(f"Nivel {level}" for level in levels) + " | Total |")
    print("|---|" + "---|" * (len(levels) + 1))
    examples = {}
    for path in args.checkpoints:
        right, examples[path.parent.name] = solve(path, problems, args)
        cells = []
        for level in levels:
            hits = [ok for ok, item in zip(right, problems) if item["level"] == level]
            cells.append(f"{100 * sum(hits) / len(hits):.0f} %")
        print(f"| `{path.parent.name}` | " + " | ".join(cells) + f" | {100 * sum(right) / len(right):.0f} % |")
    for name, example in examples.items():
        print(f"\n{name}:\n{example}")


if __name__ == "__main__":
    main()
