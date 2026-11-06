"""Training batches.

TokenFile          pretraining: the .bin token files of phases/2_tokenizer/encode.py, read with
                   np.memmap (nothing is loaded until a window is sliced: billions of tokens, no RAM)
ConversationFile   fine-tuning: tokenized conversations with a mask of the tokens to learn,
                   packed into blocks (phases/5_sft/prepare.py)
"""

import json
from pathlib import Path

import numpy as np
import torch

from leon.paths import DATA_DIR, DEFAULT_TOKENIZER, TOKENIZERS
from leon.tokenizer import BPETokenizer

DATASETS = {"wiki": DATA_DIR / "wiki_es", "quijote": DATA_DIR / "quijote"}


def load_tokenizer(version: str | Path = DEFAULT_TOKENIZER) -> BPETokenizer:
    """A tokenizer by version ("v1", "v2": see leon.paths.TOKENIZERS) or by file path."""
    path = TOKENIZERS[version] if isinstance(version, str) else version
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


IGNORE = -100  # target value that the loss skips (see Leon.forward's ignore_index)


class ConversationFile:
    """Tokenized conversations for supervised fine-tuning (written by phases/5_sft/prepare.py).

    ids holds every conversation back to back, learn marks the tokens the model must produce
    (the assistant's answers), and offsets[i]:offsets[i + 1] is conversation i.

    Training packs whole conversations into blocks of block_size + 1 tokens (never cutting one in
    half): fewer padding tokens than one conversation per row, and every answer keeps its context.
    """

    def __init__(self, path: Path) -> None:
        if not path.exists():
            raise SystemExit(f"Missing {path}. Run: python phases/5_sft/prepare.py")
        data = np.load(path)
        self.ids, self.learn, self.offsets = data["ids"], data["learn"], data["offsets"]
        self.sources = data["sources"] if "sources" in data else np.array(["unknown"] * len(self))
        self.tokenizer = str(data["tokenizer"]) if "tokenizer" in data else "v1"  # version of the ids

    def __len__(self) -> int:
        return len(self.offsets) - 1

    def pack(self, block_size: int, order: np.ndarray) -> list[list[int]]:
        """Greedily fill blocks with the conversations in `order` (a permutation of indices)."""
        blocks, current, used = [], [], 0
        for i in order:
            length = int(self.offsets[i + 1] - self.offsets[i])
            if length > block_size + 1:
                continue  # prepare.py already drops these; skip defensively
            if used + length > block_size + 1:
                blocks.append(current)
                current, used = [], 0
            current.append(int(i))
            used += length
        if current:
            blocks.append(current)
        return blocks

    def batch(self, blocks: list[list[int]], block_size: int, pad_id: int, device: str) -> tuple:
        """Inputs and targets for some packed blocks. Targets are the next token where it must
        be learned, and IGNORE elsewhere (user turns, system prompt, padding)."""
        x = np.full((len(blocks), block_size + 1), pad_id, dtype=np.int64)
        learn = np.zeros((len(blocks), block_size + 1), dtype=bool)
        for row, conversations in enumerate(blocks):
            position = 0
            for i in conversations:
                start, end = self.offsets[i], self.offsets[i + 1]
                x[row, position : position + end - start] = self.ids[start:end]
                learn[row, position : position + end - start] = self.learn[start:end]
                position += end - start
        inputs = torch.from_numpy(x[:, :-1])
        targets = torch.from_numpy(np.where(learn[:, 1:], x[:, 1:], IGNORE))
        return inputs.to(device), targets.to(device)
