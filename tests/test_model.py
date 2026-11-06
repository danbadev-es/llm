"""Correctness checks for leon.model. Run: python tests/test_model.py (or pytest)"""

import math

import torch
import torch.nn.functional as F

from leon.model import KVCache, Leon, LeonConfig, RMSNorm, _nucleus, apply_rope, precompute_rope

CONFIG = LeonConfig(vocab_size=64, block_size=32, n_layer=2, n_head=4, n_kv_head=2, n_embd=32)


def test_rmsnorm_unit_rms() -> None:
    x = torch.randn(4, 10, 32) * 7 + 3
    y = RMSNorm(32, 1e-5)(x)
    assert torch.allclose(y.pow(2).mean(dim=-1).sqrt(), torch.ones(4, 10), atol=1e-4)


def test_rope_preserves_norm_and_is_relative() -> None:
    """A rotation keeps vector lengths, and q_m · k_n depends only on m - n."""
    torch.manual_seed(0)
    head_dim = 16
    cos, sin = precompute_rope(head_dim, 64, 10000.0)
    q, k = torch.randn(1, 1, 1, head_dim), torch.randn(1, 1, 1, head_dim)

    def rotated(x, pos):
        return apply_rope(x, cos[pos : pos + 1], sin[pos : pos + 1])

    assert torch.allclose(rotated(q, 9).norm(), q.norm(), atol=1e-5)
    dot_a = (rotated(q, 5) * rotated(k, 2)).sum()  # distance 3
    dot_b = (rotated(q, 40) * rotated(k, 37)).sum()  # distance 3, other absolute positions
    dot_c = (rotated(q, 40) * rotated(k, 30)).sum()  # distance 10
    assert torch.allclose(dot_a, dot_b, atol=1e-4), "RoPE scores must depend on relative position only"
    assert not torch.allclose(dot_a, dot_c, atol=1e-2)


def test_causality() -> None:
    torch.manual_seed(0)
    model = Leon(CONFIG).eval()
    idx = torch.randint(CONFIG.vocab_size, (1, 16))
    altered = idx.clone()
    altered[0, 10:] = torch.randint(CONFIG.vocab_size, (6,))
    with torch.no_grad():
        a, _ = model(idx)
        b, _ = model(altered)
    assert torch.allclose(a[:, :10], b[:, :10], atol=1e-5), "future tokens leak into the past"
    assert not torch.allclose(a[:, 10:], b[:, 10:])


def test_gqa_attention_matches_manual() -> None:
    """The SDPA + repeat_interleave path must equal an explicit softmax(QK^T/sqrt(d))V per head."""
    torch.manual_seed(0)
    model = Leon(CONFIG).eval()
    attn = model.blocks[0].attn
    T, d = 12, CONFIG.head_dim
    x = torch.randn(1, T, CONFIG.n_embd)
    cos, sin = model.rope_cos[:T], model.rope_sin[:T]
    with torch.no_grad():
        expected_heads = []
        q = apply_rope(attn.q_proj(x).view(1, T, CONFIG.n_head, d).transpose(1, 2), cos, sin)
        k = apply_rope(attn.k_proj(x).view(1, T, CONFIG.n_kv_head, d).transpose(1, 2), cos, sin)
        v = attn.v_proj(x).view(1, T, CONFIG.n_kv_head, d).transpose(1, 2)
        mask = torch.tril(torch.ones(T, T, dtype=torch.bool))
        for h in range(CONFIG.n_head):
            kv = h // (CONFIG.n_head // CONFIG.n_kv_head)  # query head h reads key/value head kv
            scores = (q[0, h] @ k[0, kv].T / math.sqrt(d)).masked_fill(~mask, float("-inf"))
            expected_heads.append(F.softmax(scores, dim=-1) @ v[0, kv])
        expected = attn.out_proj(torch.cat(expected_heads, dim=-1)[None])
        actual = attn(x, cos, sin, cache=None)
    assert torch.allclose(actual, expected, atol=1e-5)


def test_kv_cache_matches_full_forward() -> None:
    """Prefill + chunked + token-by-token decoding with the cache must give the same logits."""
    torch.manual_seed(0)
    model = Leon(CONFIG).eval()
    idx = torch.randint(CONFIG.vocab_size, (2, 20))
    with torch.no_grad():
        full, _ = model(idx)
        cache = KVCache(CONFIG, batch_size=2)
        pieces = [model(idx[:, :8], cache=cache)[0], model(idx[:, 8:13], cache=cache)[0]]  # prefill, then a chunk
        pieces += [model(idx[:, t : t + 1], cache=cache)[0] for t in range(13, 20)]  # one token at a time
    assert torch.allclose(torch.cat(pieces, dim=1), full, atol=1e-5)


def test_generate_same_with_and_without_cache() -> None:
    torch.manual_seed(0)
    model = Leon(CONFIG).eval()
    prompt = torch.randint(CONFIG.vocab_size, (1, 5))
    with_cache = model.generate(prompt, 20, temperature=0, use_cache=True)
    without = model.generate(prompt, 20, temperature=0, use_cache=False)
    assert torch.equal(with_cache, without)


def test_generation_fills_the_context_exactly() -> None:
    """With the cache, block_size - prompt + 1 tokens fit (the last one is never fed back),
    and they must match the uncached path right up to the boundary."""
    torch.manual_seed(0)
    model = Leon(CONFIG).eval()
    prompt = torch.randint(CONFIG.vocab_size, (1, 20))
    with_cache = model.generate(prompt, 100, temperature=0, use_cache=True)
    assert with_cache.size(1) == CONFIG.block_size + 1
    without = model.generate(prompt, CONFIG.block_size + 1 - 20, temperature=0, use_cache=False)
    assert torch.equal(with_cache, without)

    too_long = torch.randint(CONFIG.vocab_size, (1, CONFIG.block_size + 10))  # cropped, not an error
    assert model.generate(too_long, 3, temperature=0).size(1) == too_long.size(1) + 1


def test_stream_stops_at_stop_ids() -> None:
    torch.manual_seed(0)
    model = Leon(CONFIG).eval()
    prompt = torch.randint(CONFIG.vocab_size, (1, 5))
    first = next(model.stream(prompt, 10, temperature=0)).item()
    tokens = list(model.stream(prompt, 10, temperature=0, stop_ids={first}))
    assert len(tokens) == 1 and tokens[0].item() == first


def test_top_p_keeps_the_smallest_likely_set() -> None:
    logits = torch.tensor([[0.5, 0.3, 0.15, 0.05]]).log()
    kept = _nucleus(logits, 0.75).isfinite().tolist()
    assert kept == [[True, True, False, False]]  # 0.5 < 0.75 <= 0.5 + 0.3
    assert _nucleus(logits, 1e-6).isfinite().sum() == 1  # the most likely token always stays


def test_stream_with_tiny_top_p_is_greedy() -> None:
    torch.manual_seed(0)
    model = Leon(CONFIG).eval()
    prompt = torch.randint(CONFIG.vocab_size, (1, 5))
    greedy = model.generate(prompt, 10, temperature=0)
    assert torch.equal(model.generate(prompt, 10, temperature=1.0, top_p=1e-6), greedy)


def test_initial_loss_near_uniform() -> None:
    torch.manual_seed(0)
    model = Leon(CONFIG).eval()
    idx = torch.randint(CONFIG.vocab_size, (8, 32))
    # Targets independent of the inputs. (With idx as its own target the loss starts below ln V:
    # with weight tying, the normalized embedding of a token has a large dot product with itself.)
    targets = torch.randint(CONFIG.vocab_size, (8, 32))
    with torch.no_grad():
        _, loss = model(idx, targets)
    assert abs(loss.item() - math.log(CONFIG.vocab_size)) < 0.3, loss.item()


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"ok  {name}")
