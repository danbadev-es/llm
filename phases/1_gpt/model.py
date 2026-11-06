"""Minimal GPT (decoder-only transformer), written for readability over speed."""

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class GPTConfig:
    vocab_size: int
    block_size: int = 256  # maximum context length, in tokens
    n_layer: int = 6
    n_head: int = 6
    n_embd: int = 384
    dropout: float = 0.2
    use_sdpa: bool = False  # True: fused PyTorch attention kernel; False: explicit implementation


class CausalSelfAttention(nn.Module):
    """Multi-head self-attention where each position can only attend to itself and earlier positions."""

    def __init__(self, config: GPTConfig) -> None:
        super().__init__()
        if config.n_embd % config.n_head != 0:
            raise ValueError("n_embd must be divisible by n_head")
        self.n_head = config.n_head
        self.head_dim = config.n_embd // config.n_head
        self.dropout = config.dropout
        self.use_sdpa = config.use_sdpa

        # Query, key and value projections for all heads, computed in a single matmul.
        self.qkv_proj = nn.Linear(config.n_embd, 3 * config.n_embd)
        self.out_proj = nn.Linear(config.n_embd, config.n_embd)
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)

        # Lower-triangular mask: position t may attend to positions 0..t only.
        mask = torch.tril(torch.ones(config.block_size, config.block_size, dtype=torch.bool))
        self.register_buffer("causal_mask", mask.view(1, 1, config.block_size, config.block_size), persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, time, channels = x.shape
        q, k, v = self.qkv_proj(x).split(channels, dim=2)

        # (batch, time, channels) -> (batch, n_head, time, head_dim): each head attends independently.
        q = q.view(batch, time, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(batch, time, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(batch, time, self.n_head, self.head_dim).transpose(1, 2)

        if self.use_sdpa:
            y = F.scaled_dot_product_attention(
                q, k, v, dropout_p=self.dropout if self.training else 0.0, is_causal=True
            )
        else:
            # Similarity of every query with every key, scaled to keep softmax inputs well-conditioned.
            scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)  # (batch, n_head, time, time)
            scores = scores.masked_fill(~self.causal_mask[:, :, :time, :time], float("-inf"))
            weights = F.softmax(scores, dim=-1)
            weights = self.attn_dropout(weights)
            y = weights @ v  # weighted sum of values: (batch, n_head, time, head_dim)

        # Concatenate heads back into a single channel dimension.
        y = y.transpose(1, 2).contiguous().view(batch, time, channels)
        return self.resid_dropout(self.out_proj(y))


class MLP(nn.Module):
    """Position-wise feed-forward network: expands to 4x width, applies a non-linearity, projects back."""

    def __init__(self, config: GPTConfig) -> None:
        super().__init__()
        self.up_proj = nn.Linear(config.n_embd, 4 * config.n_embd)
        self.down_proj = nn.Linear(4 * config.n_embd, config.n_embd)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.down_proj(F.gelu(self.up_proj(x))))


class Block(nn.Module):
    """Transformer block with pre-normalization and residual connections."""

    def __init__(self, config: GPTConfig) -> None:
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp = MLP(config)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln_1(x))  # tokens exchange information
        x = x + self.mlp(self.ln_2(x))  # each token processes what it gathered
        return x


class GPT(nn.Module):
    def __init__(self, config: GPTConfig) -> None:
        super().__init__()
        self.config = config
        self.token_embedding = nn.Embedding(config.vocab_size, config.n_embd)
        self.position_embedding = nn.Embedding(config.block_size, config.n_embd)
        self.dropout = nn.Dropout(config.dropout)
        self.blocks = nn.ModuleList(Block(config) for _ in range(config.n_layer))
        self.ln_f = nn.LayerNorm(config.n_embd)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        # Weight tying: the same matrix maps ids -> vectors and vectors -> logits.
        self.lm_head.weight = self.token_embedding.weight

        self.apply(self._init_weights)
        # Scale residual output projections so the residual stream variance does not grow with depth.
        for name, param in self.named_parameters():
            if name.endswith(("out_proj.weight", "down_proj.weight")):
                nn.init.normal_(param, mean=0.0, std=0.02 / math.sqrt(2 * config.n_layer))

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def num_params(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def forward(
        self, idx: torch.Tensor, targets: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        _, time = idx.shape
        if time > self.config.block_size:
            raise ValueError(f"Sequence length {time} exceeds block_size {self.config.block_size}")

        positions = torch.arange(time, device=idx.device)
        x = self.dropout(self.token_embedding(idx) + self.position_embedding(positions))
        for block in self.blocks:
            x = block(x)
        logits = self.lm_head(self.ln_f(x))  # (batch, time, vocab)

        if targets is None:
            return logits, None
        loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)).float(), targets.reshape(-1))
        return logits, loss

    def configure_optimizer(
        self, lr: float, weight_decay: float, betas: tuple[float, float], device: str
    ) -> torch.optim.AdamW:
        """Apply weight decay only to matrices; biases and norm parameters are not decayed."""
        decay = [p for p in self.parameters() if p.requires_grad and p.dim() >= 2]
        no_decay = [p for p in self.parameters() if p.requires_grad and p.dim() < 2]
        groups = [
            {"params": decay, "weight_decay": weight_decay},
            {"params": no_decay, "weight_decay": 0.0},
        ]
        return torch.optim.AdamW(groups, lr=lr, betas=betas, fused=device.startswith("cuda"))

    @torch.no_grad()
    def generate(
        self, idx: torch.Tensor, max_new_tokens: int, temperature: float = 1.0, top_k: int | None = None
    ) -> torch.Tensor:
        for _ in range(max_new_tokens):
            context = idx[:, -self.config.block_size :]  # the model cannot see beyond block_size
            logits, _ = self(context)
            logits = logits[:, -1, :]
            if temperature == 0:
                next_id = logits.argmax(dim=-1, keepdim=True)
            else:
                logits = logits / temperature
                if top_k is not None:
                    kth_value = torch.topk(logits, min(top_k, logits.size(-1))).values[:, [-1]]
                    logits = logits.masked_fill(logits < kth_value, float("-inf"))
                next_id = torch.multinomial(F.softmax(logits, dim=-1), num_samples=1)
            idx = torch.cat([idx, next_id], dim=1)
        return idx
