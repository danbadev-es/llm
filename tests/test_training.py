"""Correctness checks for leon.training. Run: python tests/test_training.py (or pytest)"""

import math
import tempfile
from pathlib import Path

import torch

from leon.model import Leon, LeonConfig
from leon.training import bits_per_byte, load_checkpoint, lr_at, save_checkpoint

CONFIG = LeonConfig(vocab_size=64, block_size=16, n_layer=1, n_head=2, n_kv_head=1, n_embd=16)


def test_lr_schedule() -> None:
    kwargs = dict(lr=1e-3, min_lr=1e-4, warmup_steps=10, max_steps=110)
    assert math.isclose(lr_at(0, **kwargs), 1e-4)  # warmup starts at lr / warmup_steps
    assert math.isclose(lr_at(9, **kwargs), 1e-3)  # warmup ends at lr
    assert math.isclose(lr_at(60, **kwargs), 5.5e-4)  # halfway through the cosine: the midpoint
    assert math.isclose(lr_at(110, **kwargs), 1e-4)  # ends at min_lr
    assert math.isclose(lr_at(500, **kwargs), 1e-4)  # and stays there


def test_bits_per_byte() -> None:
    assert math.isclose(bits_per_byte(math.log(2), 1.0), 1.0)  # ln 2 nats = 1 bit per 1-byte token
    assert math.isclose(bits_per_byte(4 * math.log(2), 4.0), 1.0)  # 4 bits spread over 4 bytes


def test_checkpoint_roundtrip_restores_weights_optimizer_and_rng() -> None:
    torch.manual_seed(0)
    model = Leon(CONFIG)
    optimizer = model.configure_optimizer(1e-3, 0.1, (0.9, 0.95), "cpu")
    model(torch.randint(64, (2, 16)), torch.randint(64, (2, 16)))[1].backward()
    optimizer.step()  # creates optimizer state worth saving

    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "last.pt"
        save_checkpoint(path, model, step=7, best_val=1.5, optimizer=optimizer, settings={"lr": 1e-3})
        assert not path.with_suffix(".tmp").exists(), "the temporary file must be renamed away"
        expected_batch = torch.randint(1000, (5,))  # what the RNG draws right after saving

        torch.manual_seed(123)  # scramble everything
        restored = Leon(CONFIG)
        restored_opt = restored.configure_optimizer(1e-3, 0.1, (0.9, 0.95), "cpu")
        checkpoint = load_checkpoint(path, restored, restored_opt, "cpu")

        assert torch.equal(torch.randint(1000, (5,)), expected_batch), "RNG must continue where it was saved"
        assert checkpoint["step"] == 7 and checkpoint["best_val"] == 1.5 and checkpoint["settings"]["lr"] == 1e-3
        for a, b in zip(model.state_dict().values(), restored.state_dict().values()):
            assert torch.equal(a, b)
        assert restored_opt.state_dict()["state"].keys() == optimizer.state_dict()["state"].keys()

        save_checkpoint(Path(tmp) / "best.pt", model, step=7, best_val=1.5)  # no optimizer
        assert "optimizer" not in torch.load(Path(tmp) / "best.pt", weights_only=True)


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"ok  {name}")
