"""Inspect a trained tokenizer: how it splits text and how much it compresses."""

import argparse
import json
import time
from pathlib import Path

from leon.paths import DATA_DIR, TOKENIZER_PATH
from leon.tokenizer import BPETokenizer

WIKI_VAL = DATA_DIR / "wiki_es" / "val.jsonl"
QUIJOTE_PATH = DATA_DIR / "quijote" / "input.txt"
FIXED_PROMPT = "En un lugar de la Mancha"
EXAMPLES = [
    FIXED_PROMPT,
    "La inteligencia artificial generativa transformó la programación en 2023.",
    "¿Cuánto es 1234 * 5678? Llamo a la calculadora.",
    "desoxirribonucleico, otorrinolaringólogo, electroencefalografista",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tokenizer", type=Path, default=TOKENIZER_PATH)
    parser.add_argument(
        "--wiki-mchars", type=float, default=10, help="millions of characters of Wikipedia val to measure"
    )
    return parser.parse_args()


def show_split(tokenizer: BPETokenizer, text: str) -> str:
    """Render each token between brackets, e.g. [En][ un][ lugar].

    A token holding only part of a multi-byte character is shown as raw bytes, e.g. [\\xc2].
    """
    pieces = (tokenizer.token_bytes(i).decode("utf-8", errors="backslashreplace") for i in tokenizer.encode(text))
    return "".join(f"[{piece}]" for piece in pieces)


def compression_stats(tokenizer: BPETokenizer, name: str, text: str) -> None:
    start = time.perf_counter()
    ids = tokenizer.encode(text)
    elapsed = time.perf_counter() - start
    n_bytes, n_chars, n_words = len(text.encode("utf-8")), len(text), len(text.split())
    print(
        f"{name:14s} {n_chars / 1e6:6.2f}M chars | {n_bytes / len(ids):.2f} bytes/token | "
        f"{n_chars / len(ids):.2f} chars/token | {len(ids) / n_words:.2f} tokens/word | "
        f"{n_bytes / n_chars:.3f} bytes/char | encode {n_bytes / elapsed / 1e6:.2f} MB/s (1 process)"
    )


def read_wiki_val(max_chars: int) -> str:
    docs, total = [], 0
    with open(WIKI_VAL, encoding="utf-8") as f:
        for line in f:
            docs.append(json.loads(line)["text"])
            total += len(docs[-1])
            if total >= max_chars:
                break
    return "\n\n".join(docs)


def main() -> None:
    args = parse_args()
    tokenizer = BPETokenizer.load(args.tokenizer)
    print(
        f"Tokenizer: {args.tokenizer.name} | vocab {tokenizer.vocab_size:,} "
        f"({len(tokenizer.merges):,} merges + 256 bytes + {len(tokenizer.special_tokens)} special)\n"
    )

    print("How text is split:")
    for text in EXAMPLES:
        print(f"  {len(tokenizer.encode(text)):3d} tokens  {show_split(tokenizer, text)}")

    print("\nFirst 20 merges:", [tokenizer.decode([256 + i]) for i in range(20)])
    print(
        "Last 20 merges: ",
        [tokenizer.decode([256 + i]) for i in range(len(tokenizer.merges) - 20, len(tokenizer.merges))],
    )
    longest = sorted(range(256, 256 + len(tokenizer.merges)), key=lambda i: len(tokenizer.vocab[i]), reverse=True)[:10]
    print("Longest tokens: ", [tokenizer.decode([i]) for i in longest])

    print("\nCompression (char-level tokenizer = 1 char/token):")
    quijote = QUIJOTE_PATH.read_text(encoding="utf-8")
    compression_stats(tokenizer, "Quijote val", quijote[int(0.9 * len(quijote)) :])
    if WIKI_VAL.exists():
        compression_stats(tokenizer, "Wikipedia val", read_wiki_val(int(args.wiki_mchars * 1e6)))


if __name__ == "__main__":
    main()
