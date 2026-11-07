"""Checks for leon.dpo (phase 7). Run: pytest tests/test_rl.py"""

import math

import torch
import torch.nn.functional as F

from leon.dpo import IGNORE, batch, dpo_loss, encode_answer, sequence_logprobs
from leon.model import Leon, LeonConfig
from leon.tokenizer import BPETokenizer, count_chunks

TOKENIZER = BPETokenizer.train(count_chunks(["hola qué tal, muy bien gracias"] * 5), 300)


def test_dpo_loss_on_hand_computed_values() -> None:
    zero = torch.zeros(2)
    loss, metrics = dpo_loss(zero, zero, zero, zero, beta=0.1)
    assert math.isclose(loss.item(), math.log(2), rel_tol=1e-6)  # a tie: σ(0) = 1/2
    assert metrics["reward_accuracy"] == 0.0 and metrics["margin"] == 0.0
    # The chosen answer gained 10 nats over the reference, the rejected lost 10: margin β·20 = 2.
    loss, metrics = dpo_loss(torch.tensor([10.0]), torch.tensor([-10.0]), torch.zeros(1), torch.zeros(1), beta=0.1)
    assert math.isclose(loss.item(), -F.logsigmoid(torch.tensor(2.0)).item(), rel_tol=1e-6)
    assert metrics == {"reward_accuracy": 1.0, "margin": 2.0, "chosen": 1.0, "rejected": -1.0}


def test_sequence_logprobs_sum_only_the_answer_tokens() -> None:
    torch.manual_seed(0)
    config = LeonConfig(vocab_size=TOKENIZER.vocab_size, block_size=64, n_layer=1, n_head=2, n_kv_head=1, n_embd=16)
    model = Leon(config).eval()
    prompt = [{"role": "user", "content": "hola"}]
    short, long = encode_answer(TOKENIZER, prompt, "bien"), encode_answer(TOKENIZER, prompt, "muy bien gracias")
    x, y = batch([short, long], TOKENIZER.eot_id, "cpu")
    assert (y[0] != IGNORE).sum() == sum(short[1]) - short[1][0]  # every answer token after the first position
    scores = sequence_logprobs(model, x, y)
    # By hand for the first row: log-softmax of the logits at each scored position.
    logits, _ = model(x[:1])
    logprobs = F.log_softmax(logits[0].float(), dim=-1)
    expected = sum(logprobs[t, y[0, t]].item() for t in range(y.size(1)) if y[0, t] != IGNORE)
    assert math.isclose(scores[0].item(), expected, rel_tol=1e-5)
    assert scores[1] < scores[0]  # more tokens, more negative log-probability (random model)


def test_grpo_reward_and_group_advantages() -> None:
    from leon.grpo import extract_answer, group_advantages, reward

    assert extract_answer("47 × 36 = …\nRespuesta: 1692") == 1692
    assert extract_answer("Respuesta: 3\nRespuesta: -5") == -5  # the last one counts
    assert extract_answer("no lo sé") is None
    assert reward("Respuesta: 1692", 1692) == 1.0
    assert reward("Respuesta: 1600", 1692) == 0.1  # right format, wrong number
    assert reward("mil seiscientos", 1692) == 0.0
    advantages = group_advantages(torch.tensor([[1.0, 0.0, 0.0, 0.0], [1.0, 1.0, 1.0, 1.0]]))
    assert advantages[0, 0] > 0 > advantages[0, 1] and math.isclose(advantages[0].sum().item(), 0, abs_tol=1e-5)
    assert torch.all(advantages[1] == 0)  # a group that all got right teaches nothing


def test_grpo_loss_pushes_good_answers_up() -> None:
    from leon.grpo import grpo_loss

    policy = torch.zeros(2, 3, requires_grad=True)
    mask = torch.tensor([[1.0, 1.0, 0.0], [1.0, 1.0, 1.0]])
    loss, metrics = grpo_loss(policy, policy.detach(), mask, torch.tensor([1.0, -1.0]), beta=0.04)
    loss.backward()
    assert metrics["kl"] == 0.0  # policy = reference
    assert torch.all(policy.grad[0, :2] < 0) and torch.all(policy.grad[1] > 0)  # descent raises row 0, lowers row 1
    assert policy.grad[0, 2] == 0  # masked token
