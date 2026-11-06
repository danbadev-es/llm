"""Leon: a modern decoder-only transformer (the architecture of Leon 0.1).

Changes with respect to phases/1_gpt/model.py, one idea each:
    RMSNorm   instead of LayerNorm      cheaper: no mean subtraction, no bias
    RoPE      instead of learned positions   positions enter as a rotation of q and k
    SwiGLU    instead of GELU MLP       gated feed-forward, better quality at equal cost
    GQA       instead of full multi-head    several query heads share one key/value head
    KV cache  during generation          past keys and values are computed only once
    No biases in linear layers          they add parameters and almost nothing else
"""

import math
from collections.abc import Collection, Iterator
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class LeonConfig:
    vocab_size: int = 16384
    block_size: int = 512  # maximum context length, in tokens
    n_layer: int = 8
    n_head: int = 8  # query heads
    n_kv_head: int = 4  # key/value heads; n_head // n_kv_head query heads share each one
    n_embd: int = 512
    ffn_hidden: int = 0  # SwiGLU hidden size; 0 = about 8/3 * n_embd (same cost as a 4x GELU MLP)
    rope_theta: float = 10000.0
    norm_eps: float = 1e-5
    dropout: float = 0.0

    def __post_init__(self) -> None:
        if self.n_embd % self.n_head != 0:
            raise ValueError("n_embd must be divisible by n_head")
        if self.n_head % self.n_kv_head != 0:
            raise ValueError("n_head must be divisible by n_kv_head")
        if self.ffn_hidden == 0:
            self.ffn_hidden = 64 * math.ceil(8 * self.n_embd / 3 / 64)  # rounded up for GPU efficiency

    @property
    def head_dim(self) -> int:
        return self.n_embd // self.n_head


class RMSNorm(nn.Module):
    """Rescale each vector to unit root-mean-square, then apply a learned per-channel gain.

    LayerNorm also subtracts the mean and adds a bias; in practice that is not needed.
    """

    def __init__(self, dim: int, eps: float) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_float = x.float()  # the mean of squares is computed in fp32 even under bf16 autocast
        normed = x_float * torch.rsqrt(x_float.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return (normed * self.weight).type_as(x)


def precompute_rope(head_dim: int, max_len: int, theta: float) -> tuple[torch.Tensor, torch.Tensor]:
    """Rotation angles for every (position, pair of channels): angle = position * theta^(-2i/head_dim).

    Early channel pairs rotate fast (they encode short distances), late pairs rotate slowly
    (long distances), like the hands of a clock.
    """
    inv_freq = 1.0 / theta ** (torch.arange(0, head_dim, 2).float() / head_dim)  # (head_dim/2,)
    angles = torch.outer(torch.arange(max_len).float(), inv_freq)  # (max_len, head_dim/2)
    return angles.cos(), angles.sin()


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """Rotate each pair of channels (x[2i], x[2i+1]) by its position's angle.

    x: (batch, heads, time, head_dim); cos, sin: (time, head_dim/2).
    Because rotations compose, q_m · k_n only depends on the distance m - n: the model sees
    relative positions without any position embedding.
    """
    x1, x2 = x.float()[..., 0::2], x.float()[..., 1::2]
    rotated = torch.stack((x1 * cos - x2 * sin, x1 * sin + x2 * cos), dim=-1)
    return rotated.flatten(-2).type_as(x)


class KVCache:
    """Keys and values of already processed positions, one slot per layer.

    Without a cache, generating token t recomputes keys and values for all t previous tokens.
    With it, each step only computes the newest token's k and v and reuses the rest.
    It stores n_kv_head heads, not n_head: this is where GQA saves memory.
    """

    def __init__(self, config: LeonConfig, batch_size: int) -> None:
        self.shape = (config.n_layer, batch_size, config.n_kv_head, config.block_size, config.head_dim)
        self.k: torch.Tensor | None = None
        self.v: torch.Tensor | None = None
        self.pos = 0  # number of positions already stored

    def update(self, layer: int, k: torch.Tensor, v: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if self.k is None:  # allocated lazily, so it takes the dtype used under autocast
            self.k = torch.zeros(self.shape, dtype=k.dtype, device=k.device)
            self.v = torch.zeros(self.shape, dtype=v.dtype, device=v.device)
        end = self.pos + k.size(2)
        self.k[layer, :, :, self.pos : end] = k
        self.v[layer, :, :, self.pos : end] = v
        return self.k[layer, :, :, :end], self.v[layer, :, :, :end]


class Attention(nn.Module):
    """Causal self-attention with RoPE and grouped-query attention (GQA)."""

    def __init__(self, config: LeonConfig, layer: int) -> None:
        super().__init__()
        self.layer = layer
        self.n_head, self.n_kv_head, self.head_dim = config.n_head, config.n_kv_head, config.head_dim
        self.dropout = config.dropout
        self.q_proj = nn.Linear(config.n_embd, config.n_head * config.head_dim, bias=False)
        self.k_proj = nn.Linear(config.n_embd, config.n_kv_head * config.head_dim, bias=False)
        self.v_proj = nn.Linear(config.n_embd, config.n_kv_head * config.head_dim, bias=False)
        self.out_proj = nn.Linear(config.n_head * config.head_dim, config.n_embd, bias=False)
        self.resid_dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, cache: KVCache | None) -> torch.Tensor:
        batch, time, _ = x.shape
        q = self.q_proj(x).view(batch, time, self.n_head, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(batch, time, self.n_kv_head, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(batch, time, self.n_kv_head, self.head_dim).transpose(1, 2)
        q, k = apply_rope(q, cos, sin), apply_rope(k, cos, sin)  # v carries content, not position

        start = cache.pos if cache is not None else 0
        if cache is not None:
            k, v = cache.update(self.layer, k, v)  # now covers positions 0 .. start + time - 1

        # GQA: each key/value head serves a group of query heads (query head h reads kv head h // group).
        # SDPA's enable_gqa=True avoids this copy, but measured on this setup (torch 2.11, RTX 5060)
        # it is ~20% slower and uses more memory: it falls off the fused flash-attention kernel.
        group = self.n_head // self.n_kv_head
        k = k.repeat_interleave(group, dim=1)
        v = v.repeat_interleave(group, dim=1)

        if start == 0:
            mask, is_causal = None, True
        elif time == 1:
            mask, is_causal = None, False  # one new token sees every cached position: no mask needed
        else:
            # Query i sits at absolute position start + i and may see keys 0 .. start + i.
            mask = torch.ones(time, start + time, dtype=torch.bool, device=x.device).tril(diagonal=start)
            is_causal = False
        y = F.scaled_dot_product_attention(
            q, k, v, attn_mask=mask, is_causal=is_causal, dropout_p=self.dropout if self.training else 0.0
        )
        y = y.transpose(1, 2).contiguous().view(batch, time, self.n_head * self.head_dim)
        return self.resid_dropout(self.out_proj(y))


class SwiGLU(nn.Module):
    """Gated feed-forward: down(silu(gate(x)) * up(x)).

    The gate decides, per hidden unit and per token, how much of the "up" signal passes.
    Three matrices instead of two, so the hidden size shrinks to ~8/3·n_embd to keep the cost.
    """

    def __init__(self, config: LeonConfig) -> None:
        super().__init__()
        self.gate_proj = nn.Linear(config.n_embd, config.ffn_hidden, bias=False)
        self.up_proj = nn.Linear(config.n_embd, config.ffn_hidden, bias=False)
        self.down_proj = nn.Linear(config.ffn_hidden, config.n_embd, bias=False)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x)))


class Block(nn.Module):
    def __init__(self, config: LeonConfig, layer: int) -> None:
        super().__init__()
        self.attn_norm = RMSNorm(config.n_embd, config.norm_eps)
        self.attn = Attention(config, layer)
        self.mlp_norm = RMSNorm(config.n_embd, config.norm_eps)
        self.mlp = SwiGLU(config)

    def forward(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, cache: KVCache | None) -> torch.Tensor:
        x = x + self.attn(self.attn_norm(x), cos, sin, cache)
        x = x + self.mlp(self.mlp_norm(x))
        return x


class Leon(nn.Module):
    def __init__(self, config: LeonConfig) -> None:
        super().__init__()
        self.config = config
        self.token_embedding = nn.Embedding(config.vocab_size, config.n_embd)
        self.dropout = nn.Dropout(config.dropout)
        self.blocks = nn.ModuleList(Block(config, i) for i in range(config.n_layer))
        self.final_norm = RMSNorm(config.n_embd, config.norm_eps)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.lm_head.weight = self.token_embedding.weight  # weight tying, as in phase 1

        cos, sin = precompute_rope(config.head_dim, config.block_size, config.rope_theta)
        self.register_buffer("rope_cos", cos, persistent=False)
        self.register_buffer("rope_sin", sin, persistent=False)

        self.apply(self._init_weights)
        for name, param in self.named_parameters():
            if name.endswith(("out_proj.weight", "down_proj.weight")):
                nn.init.normal_(param, mean=0.0, std=0.02 / math.sqrt(2 * config.n_layer))

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def num_params(self, non_embedding: bool = False) -> int:
        n = sum(p.numel() for p in self.parameters())
        return n - self.token_embedding.weight.numel() if non_embedding else n

    def flops_per_token(self) -> int:
        """Training FLOPs per token: 6 per parameter (forward + backward) plus attention scores.

        The embedding lookup costs nothing, but the tied lm_head matmul does, so the full
        parameter count is used. The attention term is 12 · n_layer · block_size · n_embd.
        """
        c = self.config
        return 6 * self.num_params() + 12 * c.n_layer * c.block_size * c.n_embd

    def forward(
        self, idx: torch.Tensor, targets: torch.Tensor | None = None, cache: KVCache | None = None
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        _, time = idx.shape
        start = cache.pos if cache is not None else 0
        if start + time > self.config.block_size:
            raise ValueError(f"Sequence length {start + time} exceeds block_size {self.config.block_size}")

        cos, sin = self.rope_cos[start : start + time], self.rope_sin[start : start + time]
        x = self.dropout(self.token_embedding(idx))  # no position embedding: RoPE handles order
        for block in self.blocks:
            x = block(x, cos, sin, cache)
        if cache is not None:
            cache.pos += time
        logits = self.lm_head(self.final_norm(x))

        if targets is None:
            return logits, None
        # ignore_index lets phase 5 skip the loss on user turns (loss masking).
        loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)).float(), targets.reshape(-1), ignore_index=-100)
        return logits, loss

    def configure_optimizer(
        self, lr: float, weight_decay: float, betas: tuple[float, float], device: str
    ) -> torch.optim.AdamW:
        """Weight decay only on matrices; norm gains are not decayed."""
        decay = [p for p in self.parameters() if p.requires_grad and p.dim() >= 2]
        no_decay = [p for p in self.parameters() if p.requires_grad and p.dim() < 2]
        groups = [{"params": decay, "weight_decay": weight_decay}, {"params": no_decay, "weight_decay": 0.0}]
        return torch.optim.AdamW(groups, lr=lr, betas=betas, fused=device.startswith("cuda"))

    @torch.no_grad()
    def stream(
        self,
        idx: torch.Tensor,
        max_new_tokens: int,
        temperature: float = 1.0,
        top_k: int | None = None,
        use_cache: bool = True,
        stop_ids: Collection[int] = (),
    ) -> Iterator[torch.Tensor]:
        """Autoregressive sampling, yielding each new token (batch, 1) as soon as it exists.

        With the cache, the prompt is processed once ("prefill") and every later step feeds
        only the newest token. The cache holds at most block_size positions: the prompt is
        cropped to fit and generation stops when the cache is full (the last sampled token
        is never fed back, so block_size - prompt + 1 tokens fit). Without the cache the
        context window slides instead. Generation also stops after any token in stop_ids.
        """
        block_size = self.config.block_size
        cache = KVCache(self.config, idx.size(0)) if use_cache else None
        next_input = idx[:, -block_size:]
        if use_cache:
            max_new_tokens = min(max_new_tokens, block_size - next_input.size(1) + 1)
        for _ in range(max_new_tokens):
            if cache is None:
                next_input = idx[:, -block_size:]
            logits, _ = self(next_input, cache=cache)
            logits = logits[:, -1, :].float()
            if temperature == 0:
                next_id = logits.argmax(dim=-1, keepdim=True)
            else:
                logits = logits / temperature
                if top_k is not None:
                    kth_value = torch.topk(logits, min(top_k, logits.size(-1))).values[:, [-1]]
                    logits = logits.masked_fill(logits < kth_value, float("-inf"))
                next_id = torch.multinomial(F.softmax(logits, dim=-1), num_samples=1)
            idx = torch.cat([idx, next_id], dim=1)
            next_input = next_id
            yield next_id
            if stop_ids and all(token in stop_ids for token in next_id.view(-1).tolist()):
                break

    @torch.no_grad()
    def generate(self, idx: torch.Tensor, max_new_tokens: int, **kwargs) -> torch.Tensor:
        """Like stream(), but returns the prompt followed by all generated tokens at once."""
        new = list(self.stream(idx, max_new_tokens, **kwargs))
        return torch.cat([idx, *new], dim=1)
