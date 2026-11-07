"""Preference learning (phase 7): DPO, Direct Preference Optimization (Rafailov et al., 2023).

Given a prompt, a *chosen* answer and a *rejected* one, DPO raises the probability of the chosen
answer relative to the rejected one, measured against a frozen copy of the starting model (the
*reference*), so the model does not drift far from what it already knew:

    loss = -log σ( β · [(log π(chosen) − log π_ref(chosen)) − (log π(rejected) − log π_ref(rejected))] )

where log π(answer) is the sum of the log-probabilities of the answer's tokens given the prompt.
β·(log π − log π_ref) is the answer's implicit "reward": the loss falls as the chosen answer's
reward grows above the rejected one's. No reward model and no sampling during training (unlike
RLHF with PPO): just two forward passes per pair.

The same helpers (sequences with a learn mask, summed log-probabilities) serve GRPO in phase 8.
"""

import numpy as np
import torch
import torch.nn.functional as F

from leon.chat import encode_chat_for_training

IGNORE = -100


def encode_answer(tokenizer, prompt: list[dict], answer: str) -> tuple[list[int], list[bool]]:
    """The conversation prompt + answer, and which tokens are the answer's (the ones scored)."""
    return encode_chat_for_training(tokenizer, [*prompt, {"role": "assistant", "content": answer}])


def batch(sequences: list[tuple[list[int], list[bool]]], pad_id: int, device: str) -> tuple[torch.Tensor, torch.Tensor]:
    """Inputs and targets for some sequences, padded to the longest: targets are the next token
    where it is learned (scored) and IGNORE elsewhere (prompt, padding)."""
    length = max(len(ids) for ids, _ in sequences)
    x = np.full((len(sequences), length), pad_id, dtype=np.int64)
    learn = np.zeros((len(sequences), length), dtype=bool)
    for row, (ids, mask) in enumerate(sequences):
        x[row, : len(ids)] = ids
        learn[row, : len(ids)] = mask
    inputs = torch.from_numpy(x[:, :-1])
    targets = torch.from_numpy(np.where(learn[:, 1:], x[:, 1:], IGNORE))
    return inputs.to(device), targets.to(device)


def sequence_logprobs(model, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """log π(answer | prompt) per row: the summed log-probabilities of the scored tokens."""
    logits, _ = model(x)
    logprobs = F.log_softmax(logits.float(), dim=-1)
    scored = y != IGNORE
    token_logprobs = logprobs.gather(-1, y.clamp(min=0).unsqueeze(-1)).squeeze(-1)
    return (token_logprobs * scored).sum(dim=-1)


def dpo_loss(
    policy_chosen: torch.Tensor,
    policy_rejected: torch.Tensor,
    reference_chosen: torch.Tensor,
    reference_rejected: torch.Tensor,
    beta: float,
) -> tuple[torch.Tensor, dict]:
    """The DPO loss over a batch of pairs, and its diagnostics (as Python floats):

    reward_accuracy  share of pairs where the chosen answer's implicit reward is the higher one
    margin           mean reward(chosen) − reward(rejected)
    chosen, rejected mean implicit rewards (both may fall: DPO only needs the gap to grow)
    """
    chosen = beta * (policy_chosen - reference_chosen)
    rejected = beta * (policy_rejected - reference_rejected)
    loss = -F.logsigmoid(chosen - rejected).mean()
    metrics = {
        "reward_accuracy": (chosen > rejected).float().mean().item(),
        "margin": (chosen - rejected).mean().item(),
        "chosen": chosen.mean().item(),
        "rejected": rejected.mean().item(),
    }
    return loss, metrics
