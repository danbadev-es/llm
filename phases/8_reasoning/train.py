"""Phase 8: teach Leon to solve arithmetic step by step (supervised fine-tuning, leon.sft).

Two runs on the same problems, to measure what writing the steps is worth:

    leon_0.1_steps    answers with the solution line by line, then "Respuesta: N"   (--data steps)
    leon_0.1_direct   answers only "Respuesta: N"                                 (--data direct)

The data follows the run's name unless --data says otherwise. Starts from the newest
conversational model (phase 7, else 6, else 5). Usually run through phases/8_reasoning/control.py:

    control.py start                          the steps run
    control.py start --run leon_0.1_direct    the control run
"""

import argparse

import torch

from leon.paths import CHECKPOINT_DIR, DATA_DIR
from leon.sft import add_sft_args, train_sft

CHECKPOINT_ROOT = CHECKPOINT_DIR / "8_reasoning"
REASONING_DIR = DATA_DIR / "reasoning"
CANDIDATES = [
    CHECKPOINT_DIR / "7_dpo" / "leon_0.1_dpo" / "best.pt",
    CHECKPOINT_DIR / "6_agent" / "leon_0.1_agent" / "best.pt",
    CHECKPOINT_DIR / "5_sft" / "leon_0.1_chat" / "best.pt",
]
BASE = next((path for path in CANDIDATES if path.exists()), CANDIDATES[-1])
PROBES = [
    "¿Cuánto es 47 × 36?",
    "¿Cuánto es (25 + 38) × 4?",
    "Tengo 7 cajas con 24 lápices cada una y regalo 15 lápices. ¿Cuántos lápices me quedan?",
    "Hola, ¿quién eres?",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_sft_args(parser, base=BASE, run="leon_0.1_steps", epochs=2)
    parser.add_argument("--data", choices=["steps", "direct"], default=None, help="default: from the run's name")
    args = parser.parse_args()
    args.data = args.data or ("direct" if "direct" in args.run else "steps")
    return args


def main() -> None:
    args = parse_args()
    stage = torch.load(args.base, map_location="cpu", weights_only=True, mmap=True).get("stage", "chat")
    print(f"Data: {REASONING_DIR / args.data}", flush=True)
    train_sft(args, REASONING_DIR / args.data, CHECKPOINT_ROOT, PROBES, stage=stage)


if __name__ == "__main__":
    main()
