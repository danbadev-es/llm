"""Phase 5: supervised fine-tuning (SFT) of the pretrained Leon on Spanish conversations.

Starts from a base checkpoint (phase 4) and trains it on the chat template, learning only the
assistant's answers. The result is saved with stage="chat", which the chat app recognizes.
Usually run through phases/5_sft/control.py (start / pause / resume / status / watch). Run by
hand, Ctrl+C also saves last.pt; continue with --resume.

The training loop itself lives in leon/sft.py: phases 6 and 8 fine-tune the same way.
"""

import argparse

from leon.paths import CHECKPOINT_DIR, DATA_DIR
from leon.sft import add_sft_args, train_sft

CHECKPOINT_ROOT = CHECKPOINT_DIR / "5_sft"
# The base model: the 2048-token context extension if it has been trained, else the 1024 one.
EXTENDED_BASE = CHECKPOINT_DIR / "4_pretrain" / "leon_0.1_ctx2048" / "best.pt"
BASE = EXTENDED_BASE if EXTENDED_BASE.exists() else CHECKPOINT_DIR / "4_pretrain" / "leon_0.1" / "best.pt"
SFT_DIR = DATA_DIR / "sft"
# Fixed questions answered at every evaluation, to watch the chat format being learned.
PROBES = [
    "Hola, ¿quién eres?",
    "¿Cuál es la capital de Francia?",
    "Escribe una frase sobre el mar.",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_sft_args(parser, base=BASE, run="leon_0.1_chat", upsample=["oasst2=3", "aya=2"])
    return parser.parse_args()


def main() -> None:
    train_sft(parse_args(), SFT_DIR, CHECKPOINT_ROOT, PROBES)


if __name__ == "__main__":
    main()
