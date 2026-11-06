"""Batches from the token files written by phases/2_tokenizer/encode.py.

The .bin files are opened with np.memmap: nothing is read until a window is sliced, so a
corpus of billions of tokens costs no RAM. Each batch copies only batch_size windows.
"""

import json
from pathlib import Path

import numpy as np
import torch

from leon.paths import DATA_DIR, TOKENIZER_PATH
from leon.tokenizer import BPETokenizer

DATASETS = {"wiki": DATA_DIR / "wiki_es", "quijote": DATA_DIR / "quijote"}


def load_tokenizer(path: Path = TOKENIZER_PATH) -> BPETokenizer:
    if not path.exists():
        raise SystemExit(f"Missing {path}. Run: python phases/2_tokenizer/train_tokenizer.py")
    return BPETokenizer.load(path)


class TokenFile:
    """One encoded split (e.g. data/wiki_es/val.bin) plus its byte count, for bits per byte."""

    def __init__(self, path: Path) -> None:
        if not path.exists():
            raise SystemExit(f"Missing {path}. Run: python phases/2_tokenizer/encode.py")
        self.tokens = np.memmap(path, dtype=np.uint16, mode="r")
        meta = json.loads(path.with_suffix(".json").read_text(encoding="utf-8"))
        self.bytes_per_token = meta["bytes"] / meta["tokens"]

    def __len__(self) -> int:
        return len(self.tokens)

    def get_batch(self, batch_size: int, block_size: int, device: str) -> tuple[torch.Tensor, torch.Tensor]:
        """Random windows; targets are the inputs shifted one token to the right.

        A window takes block_size + 1 tokens, so valid starts are 0 .. len - block_size - 1
        (torch.randint's upper bound is exclusive).
        """
        offsets = torch.randint(len(self.tokens) - block_size, (batch_size,)).tolist()
        chunks = np.stack([self.tokens[i : i + block_size + 1] for i in offsets]).astype(np.int64)
        chunks = torch.from_numpy(chunks)
        x, y = chunks[:, :-1], chunks[:, 1:]
        if device.startswith("cuda"):
            # Pinned (page-locked) memory allows an asynchronous copy that overlaps with compute.
            return x.pin_memory().to(device, non_blocking=True), y.pin_memory().to(device, non_blocking=True)
        return x.to(device), y.to(device)


def load_splits(dataset: str) -> dict[str, TokenFile]:
    directory = DATASETS[dataset]
    return {"train": TokenFile(directory / "train.bin"), "val": TokenFile(directory / "val.bin")}
