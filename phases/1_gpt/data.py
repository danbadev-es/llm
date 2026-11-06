"""Character-level data utilities.

Uses the same corpus and the same 90/10 split as phase 0, so validation
losses (and bits per character) are directly comparable across both phases.
"""

from pathlib import Path

import torch

DATA_PATH = Path(__file__).resolve().parents[2] / "data" / "quijote" / "input.txt"
TRAIN_FRACTION = 0.9


class CharTokenizer:
    """Maps each distinct character in the corpus to an integer id."""

    def __init__(self, chars: list[str]) -> None:
        self.chars = chars
        self.char_to_id = {char: i for i, char in enumerate(chars)}

    @classmethod
    def from_text(cls, text: str) -> "CharTokenizer":
        return cls(sorted(set(text)))

    @property
    def vocab_size(self) -> int:
        return len(self.chars)

    def encode(self, text: str) -> list[int]:
        return [self.char_to_id[char] for char in text]

    def decode(self, ids: list[int]) -> str:
        return "".join(self.chars[i] for i in ids)


def load_corpus() -> str:
    if not DATA_PATH.exists():
        raise SystemExit(f"Missing {DATA_PATH}. Run: python phases/0_bigram/prepare.py")
    return DATA_PATH.read_text(encoding="utf-8")


def train_val_split(data: torch.Tensor) -> dict[str, torch.Tensor]:
    split_at = int(TRAIN_FRACTION * len(data))
    return {"train": data[:split_at], "val": data[split_at:]}


def get_batch(data: torch.Tensor, batch_size: int, block_size: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Sample random windows; targets are the inputs shifted one position to the right.

    The corpus is small (a few MB), so it lives entirely on the training device
    and batches are sliced there without host-to-device copies.
    """
    offsets = torch.randint(len(data) - block_size - 1, (batch_size,), device=data.device)
    window = torch.arange(block_size + 1, device=data.device)
    chunks = data[offsets[:, None] + window]  # (batch, block_size + 1)
    return chunks[:, :-1], chunks[:, 1:]
