"""Phase 2: train the Leon BPE tokenizer on a sample of Spanish Wikipedia."""

import argparse
import json
import time
from collections.abc import Iterator
from pathlib import Path

from leon.paths import DATA_DIR, TOKENIZER_PATH
from leon.tokenizer import BPETokenizer, count_chunks

WIKI_TRAIN = DATA_DIR / "wiki_es" / "train.jsonl"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vocab-size", type=int, default=16384, help="must stay below 65536 to fit in uint16")
    parser.add_argument(
        "--train-mchars", type=float, default=100, help="millions of characters of Wikipedia to learn from"
    )
    parser.add_argument("--output", type=Path, default=TOKENIZER_PATH)
    return parser.parse_args()


def read_documents(path: Path, max_chars: int) -> Iterator[str]:
    """Stream articles from the start of the JSONL file until max_chars have been read."""
    if not path.exists():
        raise SystemExit(f"Missing {path}. Run: python phases/2_tokenizer/prepare.py")
    total = 0
    with open(path, encoding="utf-8") as f:
        for line in f:
            text = json.loads(line)["text"]
            yield text
            total += len(text)
            if total >= max_chars:
                return


def main() -> None:
    args = parse_args()
    if args.vocab_size > 65536:
        raise SystemExit("vocab_size must be <= 65536: token ids are stored as uint16")

    start = time.perf_counter()
    chunk_counts = count_chunks(read_documents(WIKI_TRAIN, int(args.train_mchars * 1e6)))
    print(
        f"Pre-tokenized in {time.perf_counter() - start:.1f}s: {sum(chunk_counts.values()):,} chunks, "
        f"{len(chunk_counts):,} unique. Most common: {[c for c, _ in chunk_counts.most_common(8)]}\n"
    )

    start = time.perf_counter()
    tokenizer = BPETokenizer.train(chunk_counts, args.vocab_size, verbose=True)
    print(f"\nTrained {len(tokenizer.merges):,} merges in {time.perf_counter() - start:.1f}s")

    tokenizer.save(args.output)
    print(f"Saved {args.output} (vocab {tokenizer.vocab_size:,}) and {args.output.with_suffix('.vocab.txt').name}")


if __name__ == "__main__":
    main()
