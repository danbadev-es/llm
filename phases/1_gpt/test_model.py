"""Correctness checks for the phase 1 model. Run: python phases/1_gpt/test_model.py"""

import torch
from model import GPT, GPTConfig

CONFIG = GPTConfig(vocab_size=50, block_size=32, n_layer=2, n_head=4, n_embd=64, dropout=0.0)


def test_causality() -> None:
    """Changing future tokens must not change predictions for earlier positions."""
    torch.manual_seed(0)
    model = GPT(CONFIG).eval()
    idx = torch.randint(CONFIG.vocab_size, (1, 16))
    altered = idx.clone()
    altered[0, 10:] = torch.randint(CONFIG.vocab_size, (6,))
    with torch.no_grad():
        logits_a, _ = model(idx)
        logits_b, _ = model(altered)
    assert torch.allclose(logits_a[:, :10], logits_b[:, :10], atol=1e-5), "future tokens leak into the past"
    assert not torch.allclose(logits_a[:, 10:], logits_b[:, 10:]), "later positions should change"


def test_manual_attention_matches_sdpa() -> None:
    """The explicit attention implementation must match PyTorch's fused kernel."""
    torch.manual_seed(0)
    manual = GPT(CONFIG).eval()
    fused = GPT(GPTConfig(**{**CONFIG.__dict__, "use_sdpa": True})).eval()
    fused.load_state_dict(manual.state_dict())
    idx = torch.randint(CONFIG.vocab_size, (2, 32))
    with torch.no_grad():
        assert torch.allclose(manual(idx)[0], fused(idx)[0], atol=1e-5), "manual attention differs from SDPA"


def test_initial_loss_near_uniform() -> None:
    """With small random init, the initial loss should be close to ln(vocab_size)."""
    torch.manual_seed(0)
    model = GPT(CONFIG).eval()
    idx = torch.randint(CONFIG.vocab_size, (8, 32))
    with torch.no_grad():
        _, loss = model(idx, torch.randint(CONFIG.vocab_size, (8, 32)))
    expected = torch.log(torch.tensor(float(CONFIG.vocab_size)))
    assert abs(loss.item() - expected.item()) < 0.1, f"initial loss {loss.item():.3f} far from {expected:.3f}"


if __name__ == "__main__":
    for test in (test_causality, test_manual_attention_matches_sdpa, test_initial_loss_near_uniform):
        test()
        print(f"ok  {test.__name__}")
