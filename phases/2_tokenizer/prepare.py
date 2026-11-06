"""Download Spanish Wikipedia (parquet shards) and convert it to JSONL train/val files.

Source: the `wikimedia/wikipedia` dump 20231101.es, 13 parquet shards (~3 GB in total).
Files are fetched over plain HTTP and read with pyarrow, one batch of rows at a time,
so memory stays flat no matter how many shards are processed.

Output: data/wiki_es/{train,val}.jsonl, one {"text": ...} per article. Every 10th
article goes to val: a deterministic 90/10 split by document.
"""

import argparse
import json
import sys
import unicodedata
import urllib.request
from pathlib import Path

import pyarrow.parquet as pq

from leon.paths import DATA_DIR

WIKI_DIR = DATA_DIR / "wiki_es"
SHARD_URL = "https://huggingface.co/datasets/wikimedia/wikipedia/resolve/main/20231101.es/train-{:05d}-of-00013.parquet"
NUM_SHARDS = 13
MIN_CHARS = 200  # stubs and redirects carry almost no language signal


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--shards", type=int, nargs="+", default=[0], help=f"shard indices, 0..{NUM_SHARDS - 1}")
    return parser.parse_args()


def download(url: str, path: Path) -> None:
    if path.exists():
        print(f"Already downloaded: {path.name}")
        return
    tmp = path.with_suffix(".part")
    with urllib.request.urlopen(url, timeout=60) as response, open(tmp, "wb") as f:
        total = int(response.headers.get("Content-Length", 0))
        done = 0
        while block := response.read(1 << 20):
            f.write(block)
            done += len(block)
            print(f"\r{path.name}: {done / 2**20:,.0f} / {total / 2**20:,.0f} MiB", end="", file=sys.stderr)
    print(file=sys.stderr)
    tmp.rename(path)  # only complete files get the final name


def clean(title: str, text: str) -> str:
    # NFC: "á" as one code point, never "a" + combining accent. Otherwise the same word
    # would have two byte sequences and the tokenizer would waste merges on both.
    return unicodedata.normalize("NFC", f"{title}\n\n{text.strip()}")


def main() -> None:
    args = parse_args()
    raw_dir = WIKI_DIR / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)

    paths = []
    for shard in args.shards:
        path = raw_dir / f"train-{shard:05d}-of-{NUM_SHARDS:05d}.parquet"
        download(SHARD_URL.format(shard), path)
        paths.append(path)

    counts = {"train": [0, 0], "val": [0, 0]}  # split -> [articles, characters]
    doc_index = 0
    with (
        open(WIKI_DIR / "train.jsonl", "w", encoding="utf-8") as train_f,
        open(WIKI_DIR / "val.jsonl", "w", encoding="utf-8") as val_f,
    ):
        for path in paths:
            for batch in pq.ParquetFile(path).iter_batches(batch_size=1024, columns=["title", "text"]):
                for title, text in zip(batch.column("title").to_pylist(), batch.column("text").to_pylist()):
                    if len(text) < MIN_CHARS:
                        continue
                    split = "val" if doc_index % 10 == 9 else "train"
                    doc = clean(title, text)
                    (val_f if split == "val" else train_f).write(json.dumps({"text": doc}, ensure_ascii=False) + "\n")
                    counts[split][0] += 1
                    counts[split][1] += len(doc)
                    doc_index += 1
            print(f"Processed {path.name}")

    for split, (docs, chars) in counts.items():
        print(f"{split:5s}: {docs:,} articles, {chars:,} characters")


if __name__ == "__main__":
    main()
