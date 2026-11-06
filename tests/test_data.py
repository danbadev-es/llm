"""Correctness checks for leon.data. Run: python tests/test_data.py (or pytest)"""

import json
import tempfile
from pathlib import Path

import numpy as np
import torch

from leon.data import TokenFile


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


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"ok  {name}")
