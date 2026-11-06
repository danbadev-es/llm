"""Encode the corpora into flat uint16 token files, ready for np.memmap.

Outputs (next to each corpus):
    data/wiki_es/{train,val}.bin     every article followed by <|endoftext|>
    data/quijote/{train,val}.bin     same 90/10 character split as phases 0 and 1
and a .json with the token and byte counts of each file. Those counts are what turn
a loss per token into bits per byte:  bits/byte = loss_nats * tokens / (bytes * ln 2).

Encoding is pure Python, so documents are spread over a pool of worker processes.
Input is streamed and output is appended in blocks: RAM use does not grow with corpus size.
"""

import argparse
import json
import multiprocessing as mp
import time
from collections.abc import Iterator
from pathlib import Path

import numpy as np

from leon.paths import DATA_DIR, ROOT, TOKENIZER_PATH
from leon.tokenizer import BPETokenizer

WIKI_DIR = DATA_DIR / "wiki_es"
QUIJOTE_PATH = DATA_DIR / "quijote" / "input.txt"
TRAIN_FRACTION = 0.9  # same as phases/1_gpt/data.py

_tokenizer: BPETokenizer | None = None  # one copy per worker process


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--tokenizer", type=Path, default=TOKENIZER_PATH)
    parser.add_argument("--corpora", nargs="+", default=["quijote", "wiki_val", "wiki_train"])
    parser.add_argument("--workers", type=int, default=max(1, mp.cpu_count() - 2))
    return parser.parse_args()


def init_worker(tokenizer_path: Path) -> None:
    global _tokenizer
    _tokenizer = BPETokenizer.load(tokenizer_path)


def encode_document(text: str) -> tuple[np.ndarray, int]:
    ids = _tokenizer.encode_ordinary(text)
    ids.append(_tokenizer.eot_id)
    return np.array(ids, dtype=np.uint16), len(text.encode("utf-8", errors="replace"))


def iter_jsonl(path: Path) -> Iterator[str]:
    with open(path, encoding="utf-8") as f:
        for line in f:
            yield json.loads(line)["text"]


def write_tokens(out_path: Path, documents: Iterator[str], pool, tokenizer_path: Path) -> None:
    n_tokens = n_bytes = n_docs = 0
    start = time.perf_counter()
    with open(out_path, "wb") as f:
        for ids, size in pool.imap(encode_document, documents, chunksize=16):
            ids.tofile(f)
            n_tokens += len(ids)
            n_bytes += size
            n_docs += 1
            if n_docs % 5000 == 0:
                rate = n_bytes / (time.perf_counter() - start) / 1e6
                print(f"  {out_path.name}: {n_docs:,} docs, {n_tokens:,} tokens, {rate:.1f} MB/s", flush=True)
    elapsed = time.perf_counter() - start
    meta = {
        "tokens": n_tokens,
        "bytes": n_bytes,
        "documents": n_docs,
        "bytes_per_token": n_bytes / n_tokens,
        "tokenizer": str(tokenizer_path.relative_to(ROOT))
        if tokenizer_path.is_relative_to(ROOT)
        else str(tokenizer_path),
    }
    out_path.with_suffix(".json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(
        f"{out_path.relative_to(ROOT)}: {n_tokens:,} tokens from {n_bytes / 1e6:,.1f} MB "
        f"({meta['bytes_per_token']:.2f} bytes/token) in {elapsed:.1f}s ({n_bytes / elapsed / 1e6:.1f} MB/s)"
    )


def main() -> None:
    args = parse_args()
    if not args.tokenizer.exists():
        raise SystemExit(f"Missing {args.tokenizer}. Run: python phases/2_tokenizer/train_tokenizer.py")

    with mp.Pool(args.workers, initializer=init_worker, initargs=(args.tokenizer,)) as pool:
        for corpus in args.corpora:
            if corpus == "quijote":
                text = QUIJOTE_PATH.read_text(encoding="utf-8")
                split_at = int(TRAIN_FRACTION * len(text))
                # One "document" per split. The trailing <|endoftext|> is harmless and keeps one code path.
                write_tokens(QUIJOTE_PATH.with_name("train.bin"), iter([text[:split_at]]), pool, args.tokenizer)
                write_tokens(QUIJOTE_PATH.with_name("val.bin"), iter([text[split_at:]]), pool, args.tokenizer)
            elif corpus in ("wiki_train", "wiki_val"):
                split = corpus.removeprefix("wiki_")
                write_tokens(WIKI_DIR / f"{split}.bin", iter_jsonl(WIKI_DIR / f"{split}.jsonl"), pool, args.tokenizer)
            else:
                raise SystemExit(f"Unknown corpus: {corpus}")


if __name__ == "__main__":
    main()
