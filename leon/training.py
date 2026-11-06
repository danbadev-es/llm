"""Training utilities shared by every training script from phase 3 on.

lr_at              warmup + cosine learning-rate schedule
estimate_loss      average loss over random batches of each split
bits_per_byte      loss per token -> bits per byte of text (comparable across tokenizers)
save_checkpoint    atomic save of model, optimizer, RNG state and run settings
load_checkpoint    the inverse, restoring the RNG so a resumed run continues the same data stream
"""

import math
import os
from collections.abc import Callable
from contextlib import AbstractContextManager
from dataclasses import asdict
from pathlib import Path

import torch

from leon.data import TokenFile
from leon.model import Leon


def lr_at(step: int, lr: float, min_lr: float, warmup_steps: int, max_steps: int) -> float:
    """Linear warmup to lr, then cosine decay down to min_lr at max_steps."""
    if step < warmup_steps:
        return lr * (step + 1) / warmup_steps
    progress = min(1.0, (step - warmup_steps) / max(1, max_steps - warmup_steps))
    return min_lr + 0.5 * (1.0 + math.cos(math.pi * progress)) * (lr - min_lr)


@torch.no_grad()
def estimate_loss(
    model: Leon,
    splits: dict[str, TokenFile],
    batch_size: int,
    block_size: int,
    eval_iters: int,
    device: str,
    autocast: Callable[[], AbstractContextManager],
) -> dict[str, float]:
    """Mean loss over eval_iters random batches of every split (dropout off during the estimate)."""
    model.eval()
    result = {}
    for name, split in splits.items():
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            x, y = split.get_batch(batch_size, block_size, device)
            with autocast():
                _, loss = model(x, y)
            losses[k] = loss.item()
        result[name] = losses.mean().item()
    model.train()
    return result


def bits_per_byte(loss: float, bytes_per_token: float) -> float:
    """Average loss in nats per token -> bits per byte of the original text."""
    return loss / math.log(2) / bytes_per_token


def save_checkpoint(
    path: Path,
    model: Leon,
    step: int,
    best_val: float,
    optimizer: torch.optim.Optimizer | None = None,
    settings: dict | None = None,
    stage: str = "base",
) -> None:
    """Write a checkpoint atomically: to a temporary file first, then rename over the old one.

    A crash or power cut in the middle of a write leaves the previous checkpoint intact,
    which matters when the file is the only resume point of a 15-hour run.
    The optimizer (2x the model size) is only needed to resume; best.pt can skip it.
    stage is "base" (pretrained) or "chat" (fine-tuned on conversations): the chat API reads it.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint = {
        "model": model.state_dict(),
        "config": asdict(model.config),
        "step": step,
        "best_val": best_val,
        "settings": settings or {},
        "stage": stage,
        "rng_cpu": torch.get_rng_state(),
        "rng_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
    }
    if optimizer is not None:
        checkpoint["optimizer"] = optimizer.state_dict()
    tmp = path.with_suffix(".tmp")
    torch.save(checkpoint, tmp)
    os.replace(tmp, path)  # atomic on the same filesystem


def load_checkpoint(path: Path, model: Leon, optimizer: torch.optim.Optimizer | None, device: str) -> dict:
    """Load weights (and optimizer) in place and restore the random number generators.

    Restoring the RNG makes the resumed run draw the batches the uninterrupted run would
    have drawn, instead of replaying the ones from step 0.
    """
    checkpoint = torch.load(path, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint["model"])
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint["optimizer"])
    if "rng_cpu" in checkpoint:  # absent in checkpoints written before this was added
        torch.set_rng_state(checkpoint["rng_cpu"].cpu())
    if checkpoint.get("rng_cuda") and torch.cuda.is_available():
        torch.cuda.set_rng_state_all([state.cpu() for state in checkpoint["rng_cuda"]])
    return checkpoint
