"""Correctness checks for leon.data. Run: python tests/test_data.py (or pytest)"""

import json
import tempfile
from pathlib import Path

import numpy as np
import torch

from leon.data import IGNORE, ConversationFile, TokenFile


def make_token_file(directory: Path, n_tokens: int) -> Path:
    """A .bin holding 0, 1, 2, ... so every window's content reveals where it came from."""
    path = directory / "toy.bin"
    np.arange(n_tokens, dtype=np.uint16).tofile(path)
    path.with_suffix(".json").write_text(json.dumps({"tokens": n_tokens, "bytes": 3 * n_tokens}))
    return path


def test_targets_are_inputs_shifted_by_one() -> None:
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:  # Windows keeps memmaps open
        data = TokenFile(make_token_file(Path(tmp), 100))
        x, y = data.get_batch(batch_size=16, block_size=8, device="cpu")
        assert x.shape == y.shape == (16, 8) and x.dtype == torch.int64
        assert torch.equal(y, x + 1)
        assert data.bytes_per_token == 3.0


def test_every_token_can_be_a_target() -> None:
    """Regression test: an off-by-one once made the last token of a file unreachable."""
    torch.manual_seed(0)
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:  # Windows keeps memmaps open
        data = TokenFile(make_token_file(Path(tmp), 20))
        seen = set()
        for _ in range(300):
            _, y = data.get_batch(batch_size=4, block_size=8, device="cpu")
            seen.update(y.reshape(-1).tolist())
        assert seen == set(range(1, 20)), f"missing targets: {set(range(1, 20)) - seen}"


def test_conversations_are_packed_whole_and_only_answers_are_targets() -> None:
    # Three conversations of 4, 3 and 5 tokens; in each, the last two tokens are "answers".
    lengths = [4, 3, 5]
    ids = np.arange(1, sum(lengths) + 1, dtype=np.uint16)
    learn = np.concatenate([[False] * (n - 2) + [True, True] for n in lengths])
    offsets = np.concatenate([[0], np.cumsum(lengths)])
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
        path = Path(tmp) / "conversations.npz"
        np.savez(path, ids=ids, learn=learn, offsets=offsets)
        data = ConversationFile(path)
        blocks = data.pack(block_size=7, order=np.arange(3))  # 8 tokens per block
        assert blocks == [[0, 1], [2]]  # 4 + 3 fit together; 5 more would split conversation 2
        x, y = data.batch(blocks, block_size=7, pad_id=0, device="cpu")
        assert x.shape == y.shape == (2, 7)
        assert x[0].tolist() == [1, 2, 3, 4, 5, 6, 7]
        # The target at position t is token t+1 if that token is learned: 3 and 4 close conversation 0,
        # 6 and 7 close conversation 1. Position 6 would predict the padding: never learned.
        assert y[0].tolist() == [IGNORE, 3, 4, IGNORE, 6, 7, IGNORE]
        assert x[1].tolist() == [8, 9, 10, 11, 12, 0, 0]
        assert y[1].tolist() == [IGNORE, IGNORE, 11, 12, IGNORE, IGNORE, IGNORE]


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"ok  {name}")
