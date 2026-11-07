"""Reinforcement learning with a verifiable reward (phase 8): GRPO, Group Relative Policy
Optimization (Shao et al., 2024, DeepSeekMath; the algorithm behind DeepSeek-R1).

For each problem the model writes G answers (a "group"). Each gets a reward that a program can
check: 1 if the final "Respuesta: N" is right. Its *advantage* is how much better it did than its
own group:

    A = (reward − mean of the group) / std of the group

so no value model is needed: the group is the baseline. Then every token of an answer is pushed
up in probability if A > 0 and down if A < 0, while a KL penalty keeps the model close to a frozen
reference (the model before RL), as in DPO:

    loss = mean over tokens of  −A · ratio  +  β · KL(π ‖ π_ref)

with ratio = π(token) / π_old(token). This implementation makes one update per batch of samples,
so π_old = π and ratio = 1 (its gradient is A · ∇log π): GRPO's clipping never activates, and the
loss is REINFORCE with the group as baseline. The KL uses the k3 estimator,
exp(r) − r − 1 with r = log π_ref − log π, which is always ≥ 0.

Groups where every answer got the same reward (all right or all wrong) have A = 0: they teach
nothing, which is why the problems must be neither too easy nor too hard for the model.
"""

import re

import torch
import torch.nn.functional as F

from leon.dpo import IGNORE

ANSWER = re.compile(r"Respuesta:\s*(-?\d+)")


def extract_answer(text: str) -> int | None:
    """The number after the last "Respuesta:", or None."""
    found = ANSWER.findall(text)
    return int(found[-1]) if found else None


def reward(text: str, target: int, format_bonus: float = 0.1) -> float:
    """1 for the right answer; a small bonus for at least writing "Respuesta: N"; else 0."""
    answer = extract_answer(text)
    if answer is None:
        return 0.0
    return 1.0 if answer == target else format_bonus


def group_advantages(rewards: torch.Tensor, eps: float = 1e-4) -> torch.Tensor:
    """(problems, G) rewards → advantages normalized within each group."""
    mean = rewards.mean(dim=1, keepdim=True)
    std = rewards.std(dim=1, keepdim=True, unbiased=False)
    return (rewards - mean) / (std + eps)


def token_logprobs(model, x: torch.Tensor, y: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """log π of each target token (0 where IGNORE) and the mask of scored tokens."""
    logits, _ = model(x)
    logprobs = F.log_softmax(logits.float(), dim=-1)
    mask = (y != IGNORE).float()
    picked = logprobs.gather(-1, y.clamp(min=0).unsqueeze(-1)).squeeze(-1)
    return picked * mask, mask


def grpo_loss(
    policy: torch.Tensor, reference: torch.Tensor, mask: torch.Tensor, advantages: torch.Tensor, beta: float
) -> tuple[torch.Tensor, dict]:
    """GRPO's loss for sampled answers: per-token log-probabilities (B, T) under the policy (with
    gradient) and the reference, the answers' mask and their advantages (B,)."""
    ratio = torch.exp(policy - policy.detach())  # = 1, but carries the gradient of log π
    log_ratio = (reference - policy) * mask
    kl = torch.exp(log_ratio) - log_ratio - 1
    per_token = -advantages.unsqueeze(1) * ratio + beta * kl
    lengths = mask.sum(dim=1).clamp(min=1)
    loss = ((per_token * mask).sum(dim=1) / lengths).mean()
    return loss, {"kl": ((kl * mask).sum(dim=1) / lengths).mean().item()}
