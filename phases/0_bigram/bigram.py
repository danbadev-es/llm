"""Phase 0: character-level bigram language model.

The model predicts the next character using only the current one. It is
deliberately the simplest possible language model, so that the full training
pipeline (tokenization, batching, loss, optimization, sampling) is visible
without any architectural complexity.
"""

import argparse
import math
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

DATA_PATH = Path(__file__).resolve().parents[2] / "data" / "quijote" / "input.txt"
FIXED_PROMPT = "En un lugar de la Mancha"


class CharTokenizer:
    """Maps each distinct character in the corpus to an integer id."""

    def __init__(self, text: str) -> None:
        self.chars = sorted(set(text))
        self.char_to_id = {char: i for i, char in enumerate(self.chars)}

    @property
    def vocab_size(self) -> int:
        return len(self.chars)

    def encode(self, text: str) -> list[int]:
        return [self.char_to_id[char] for char in text]

    def decode(self, ids: list[int]) -> str:
        return "".join(self.chars[i] for i in ids)


class BigramLanguageModel(nn.Module):
    """A (vocab_size x vocab_size) table: row i holds the logits of the next char after char i."""

    def __init__(self, vocab_size: int) -> None:
        super().__init__()
        self.next_token_logits = nn.Embedding(vocab_size, vocab_size)

    def forward(
        self, idx: torch.Tensor, targets: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        logits = self.next_token_logits(idx)  # (batch, time, vocab)
        if targets is None:
            return logits, None
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss

    @torch.no_grad()
    def generate(self, idx: torch.Tensor, max_new_tokens: int, temperature: float = 1.0) -> torch.Tensor:
        for _ in range(max_new_tokens):
            logits, _ = self(idx[:, -1:])  # a bigram only needs the last token
            logits = logits[:, -1, :]
            if temperature == 0:
                next_id = logits.argmax(dim=-1, keepdim=True)
            else:
                probs = F.softmax(logits / temperature, dim=-1)
                next_id = torch.multinomial(probs, num_samples=1)
            idx = torch.cat([idx, next_id], dim=1)
        return idx


def get_batch(data: torch.Tensor, batch_size: int, block_size: int, device: str) -> tuple[torch.Tensor, torch.Tensor]:
    """Sample random windows; targets are the inputs shifted one position to the right."""
    offsets = torch.randint(len(data) - block_size - 1, (batch_size,))
    x = torch.stack([data[i : i + block_size] for i in offsets])
    y = torch.stack([data[i + 1 : i + block_size + 1] for i in offsets])
    return x.to(device), y.to(device)


@torch.no_grad()
def estimate_loss(
    model: nn.Module, splits: dict[str, torch.Tensor], args: argparse.Namespace, device: str
) -> dict[str, float]:
    """Average the loss over several batches to get a low-noise estimate."""
    model.eval()
    result = {}
    for name, data in splits.items():
        losses = torch.zeros(args.eval_iters)
        for k in range(args.eval_iters):
            x, y = get_batch(data, args.batch_size, args.block_size, device)
            _, loss = model(x, y)
            losses[k] = loss.item()
        result[name] = losses.mean().item()
    model.train()
    return result


def counting_bigram_val_loss(train: torch.Tensor, val: torch.Tensor, vocab_size: int) -> float:
    """Closed-form reference: estimate P(next | current) by counting (add-one smoothing).

    Gradient descent on the bigram table should converge close to this value,
    because both minimize the same objective. Comparing them validates the training loop.
    """
    pair_ids = train[:-1] * vocab_size + train[1:]
    counts = torch.bincount(pair_ids, minlength=vocab_size * vocab_size).view(vocab_size, vocab_size)
    probs = (counts + 1).float()
    probs /= probs.sum(dim=1, keepdim=True)
    return -probs[val[:-1], val[1:]].log().mean().item()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--steps", type=int, default=3000)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--block-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-2)
    parser.add_argument("--eval-interval", type=int, default=300)
    parser.add_argument("--eval-iters", type=int, default=100)
    parser.add_argument("--sample-tokens", type=int, default=400)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)

    if not DATA_PATH.exists():
        raise SystemExit(f"Missing {DATA_PATH}. Run: python phases/0_bigram/prepare.py")
    text = DATA_PATH.read_text(encoding="utf-8")

    tokenizer = CharTokenizer(text)
    data = torch.tensor(tokenizer.encode(text), dtype=torch.long)
    split_at = int(0.9 * len(data))
    splits = {"train": data[:split_at], "val": data[split_at:]}

    print(f"Corpus: {len(text):,} chars | vocab: {tokenizer.vocab_size} | device: {args.device}")
    print(f"Uniform baseline loss  ln(V) = {math.log(tokenizer.vocab_size):.4f}")
    reference = counting_bigram_val_loss(splits["train"], splits["val"], tokenizer.vocab_size)
    print(f"Counting bigram val loss     = {reference:.4f}  (target for gradient descent)\n")

    model = BigramLanguageModel(tokenizer.vocab_size).to(args.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    start = time.perf_counter()
    for step in range(args.steps + 1):
        if step % args.eval_interval == 0:
            losses = estimate_loss(model, splits, args, args.device)
            bits_per_char = losses["val"] / math.log(2)
            print(
                f"step {step:5d} | train {losses['train']:.4f} | val {losses['val']:.4f} "
                f"| val bits/char {bits_per_char:.3f}"
            )
        if step == args.steps:
            break
        x, y = get_batch(splits["train"], args.batch_size, args.block_size, args.device)
        _, loss = model(x, y)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

    elapsed = time.perf_counter() - start
    tokens_per_sec = args.steps * args.batch_size * args.block_size / elapsed
    print(f"\nTraining time: {elapsed:.1f}s | throughput: {tokens_per_sec:,.0f} tokens/s")

    prompt = torch.tensor([tokenizer.encode(FIXED_PROMPT)], dtype=torch.long, device=args.device)
    sample = model.generate(prompt, args.sample_tokens, args.temperature)[0].tolist()
    print(f"\n--- Sample (temperature={args.temperature}) ---\n{tokenizer.decode(sample)}")


if __name__ == "__main__":
    main()
