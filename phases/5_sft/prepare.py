"""Phase 5: build the Spanish conversation dataset for supervised fine-tuning (SFT).

Sources (all downloadable without an account; Spanish rows only):
    oasst2   OpenAssistant/oasst2               human multi-turn conversations       Apache-2.0
    aya      CohereLabs/aya_dataset             human instruction / answer pairs     Apache-2.0
    dolly    databricks-dolly-15k (curated es)  translated instruction / answer      CC BY-SA 3.0
    alpaca   bertin-project/alpaca-spanish      translated instruction / answer      CC BY 4.0 (*)
    wildchat allenai/WildChat-1M                real multi-turn chats with ChatGPT   ODC-BY (*)
(*) Generated with OpenAI models: fine for study, not for a commercial product.

Conversations longer than the context are cut to their first turns that fit (whole exchanges),
so long multi-turn chats still teach how to follow a conversation.

Output in data/sft/:
    {train,val}.jsonl   {"source", "messages": [{"role", "content"}, ...]} per conversation
    {train,val}.npz     ids (uint16), learn (bool: is this token a training target?), offsets, sources,
                        and the tokenizer version that produced the ids
    stats.json          conversations and tokens per source, and what was filtered out
"""

import argparse
import hashlib
import json
import re
import sys
import urllib.request
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pyarrow.compute as pc
import pyarrow.parquet as pq

from leon.chat import DEFAULT_SYSTEM, encode_chat_for_training
from leon.data import load_tokenizer
from leon.paths import DATA_DIR, DEFAULT_TOKENIZER

SFT_DIR = DATA_DIR / "sft"
PARQUET_API = "https://huggingface.co/api/datasets/{dataset}/parquet/{config}/{split}"
SOURCES = {
    "oasst2": [("OpenAssistant/oasst2", "default", "train"), ("OpenAssistant/oasst2", "default", "validation")],
    "aya": [("CohereLabs/aya_dataset", "default", "train")],
    "dolly": [("argilla/databricks-dolly-15k-curated-multilingual", "default", "es")],
    "alpaca": [("bertin-project/alpaca-spanish", "default", "train")],
    "wildchat": [("allenai/WildChat-1M", "default", "train")],
}
SYSTEM_PROMPT = DEFAULT_SYSTEM
# The assistant in oasst2 introduces itself by its original name; Leon should not.
IDENTITY = re.compile(r"\b(Open[ -]?Assistant|LAION)\b", re.IGNORECASE)
# WildChat answers that would teach Leon to be someone else, or to answer with code: skipped.
FOREIGN_ANSWER = re.compile(
    r"openai|chatgpt|gpt-[34]|modelo de lenguaje|soy una (ia|inteligencia artificial)|como (una )?ia\b|```",
    re.IGNORECASE,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--sources", nargs="+", default=list(SOURCES), choices=list(SOURCES))
    parser.add_argument("--max-tokens", type=int, default=2048, help="the context: longer chats keep their first turns")
    parser.add_argument("--system-fraction", type=float, default=0.3, help="share given a generic system prompt")
    parser.add_argument("--max-paths-per-tree", type=int, default=3, help="oasst2: conversations taken per tree")
    return parser.parse_args()


def stable_fraction(text: str) -> float:
    """A deterministic pseudo-random number in [0, 1) derived from a text."""
    return int(hashlib.sha256(text.encode("utf-8")).hexdigest()[:8], 16) / 2**32


def download(source: str) -> list[Path]:
    """Fetch the parquet files of a source once (Hugging Face's parquet export)."""
    paths = []
    for dataset, config, split in SOURCES[source]:
        with urllib.request.urlopen(PARQUET_API.format(dataset=dataset, config=config, split=split), timeout=60) as r:
            urls = json.load(r)
        for i, url in enumerate(urls):
            path = SFT_DIR / "raw" / source / f"{split}-{i}.parquet"
            if not path.exists():
                path.parent.mkdir(parents=True, exist_ok=True)
                print(f"  downloading {dataset} {split} [{i}]", flush=True)
                tmp = path.with_suffix(".part")
                urllib.request.urlretrieve(url, tmp)
                tmp.rename(path)
            paths.append(path)
    return paths


def read_rows(paths: list[Path], columns: list[str]) -> list[dict]:
    return [row for path in paths for row in pq.read_table(path, columns=columns).to_pylist()]


def oasst2_conversations(paths: list[Path], max_paths: int) -> list[list[dict]]:
    """Turn oasst2's message trees into linear conversations.

    Every user message may have several ranked assistant replies: the best-ranked one is kept.
    Different user follow-ups branch the tree; up to max_paths branches are taken per tree.
    """
    columns = ["message_id", "parent_id", "text", "role", "lang", "rank", "deleted", "review_result", "synthetic"]
    rows = [
        row
        for row in read_rows(paths, columns)
        if row["lang"] == "es" and not row["deleted"] and row["review_result"] and not row["synthetic"]
    ]
    children = defaultdict(list)
    for row in rows:
        children[row["parent_id"]].append(row)
    roots = [row for row in children[None] if row["role"] == "prompter"]

    def best_reply(prompt: dict) -> dict | None:
        replies = [c for c in children[prompt["message_id"]] if c["role"] == "assistant"]
        return min(replies, key=lambda r: (r["rank"] is None, r["rank"] or 0), default=None)

    conversations = []
    for root in roots:
        found = []
        stack = [(root, [])]
        while stack and len(found) < max_paths:
            prompt, history = stack.pop()
            answer = best_reply(prompt)
            if answer is None:
                continue
            path = history + [
                {"role": "user", "content": prompt["text"]},
                {"role": "assistant", "content": answer["text"]},
            ]
            follow_ups = [c for c in children[answer["message_id"]] if c["role"] == "prompter"]
            if not follow_ups:
                found.append(path)
            stack.extend((follow_up, path) for follow_up in follow_ups)
        conversations += found
    return conversations


def pair(user: str, answer: str) -> list[dict]:
    return [{"role": "user", "content": user}, {"role": "assistant", "content": answer}]


def load_source(source: str, paths: list[Path], args: argparse.Namespace) -> list[list[dict]]:
    if source == "oasst2":
        return oasst2_conversations(paths, args.max_paths_per_tree)
    if source == "aya":
        rows = read_rows(paths, ["inputs", "targets", "language"])
        return [pair(r["inputs"], r["targets"]) for r in rows if r["language"] == "Spanish"]
    if source == "dolly":
        rows = read_rows(paths, ["instruction", "context", "response"])
        return [
            pair(
                f"{r['context'].strip()}\n\n{r['instruction']}" if (r["context"] or "").strip() else r["instruction"],
                r["response"],
            )
            for r in rows
        ]
    if source == "wildchat":
        conversations = []
        for path in paths:  # one file at a time: the dataset is 3.4 GB
            table = pq.read_table(path, columns=["conversation", "language", "toxic"])
            # Filter in Arrow first: only the ~2% Spanish rows become Python objects (RAM stays low).
            spanish = pc.and_(pc.equal(table["language"], "Spanish"), pc.invert(table["toxic"]))
            for row in table.filter(spanish).to_pylist():
                turns = [{"role": m["role"], "content": m["content"]} for m in row["conversation"]]
                if any(m["role"] == "assistant" and FOREIGN_ANSWER.search(m["content"] or "") for m in turns):
                    continue
                conversations.append(turns)
        return conversations
    rows = read_rows(paths, ["instruction", "input", "output"])  # alpaca
    return [
        pair(f"{r['instruction']}\n\n{r['input']}" if (r["input"] or "").strip() else r["instruction"], r["output"])
        for r in rows
    ]


def clean(conversation: list[dict]) -> list[dict] | None:
    """Trim whitespace, rename the assistant's identity, and reject conversations with empty turns."""
    messages = []
    for message in conversation:
        content = (message["content"] or "").strip()
        if not content:
            return None
        if message["role"] == "assistant":
            content = IDENTITY.sub("Leon", content)
        messages.append({"role": message["role"], "content": content})
    return messages


def fit(tokenizer, messages: list[dict], max_tokens: int) -> tuple[list[dict], list[int], list[bool]] | None:
    """The conversation, or its longest beginning made of whole exchanges, that fits in max_tokens."""
    for end in range(len(messages), 0, -1):
        if messages[end - 1]["role"] != "assistant":
            continue  # a training conversation always ends with an answer
        tokens, targets = encode_chat_for_training(tokenizer, messages[:end])
        if len(tokens) <= max_tokens:
            return messages[:end], tokens, targets
    return None


def main() -> None:
    args = parse_args()
    tokenizer = load_tokenizer()
    stats = {"tokenizer": DEFAULT_TOKENIZER, "sources": {}, "dropped": Counter()}
    splits = {"train": [], "val": []}
    seen = set()

    for source in args.sources:
        print(f"{source}:", flush=True)
        conversations = load_source(source, download(source), args)
        kept = Counter()
        for conversation in conversations:
            messages = clean(conversation)
            if messages is None:
                stats["dropped"]["empty turn"] += 1
                continue
            roles = [m["role"] for m in messages]
            if roles[0] != "user" or roles[-1] != "assistant" or any(a == b for a, b in zip(roles, roles[1:])):
                stats["dropped"]["turns out of order"] += 1
                continue
            # Duplicates: the whole conversation. Branches of one oasst2 tree share their beginning
            # but continue differently, so they are different conversations.
            whole = " ".join(" ⟂ ".join(m["content"] for m in messages).lower().split())
            if whole in seen:
                stats["dropped"]["duplicate"] += 1
                continue
            seen.add(whole)
            # Split: the first exchange, so every branch of a tree lands in the same split (a shared
            # beginning in train and val would let validation measure memorized text).
            key = " ".join((messages[0]["content"] + " ⟂ " + messages[1]["content"][:300]).lower().split())
            if stable_fraction("system:" + key) < args.system_fraction:
                messages = [{"role": "system", "content": SYSTEM_PROMPT}, *messages]
            split = "val" if stable_fraction("split:" + key) < 0.05 else "train"
            splits[split].append({"source": source, "messages": messages})
            kept[split] += 1
        stats["sources"][source] = {"read": len(conversations), **kept}
        print(
            f"  {len(conversations):,} conversations read, {kept['train']:,} train / {kept['val']:,} val kept",
            flush=True,
        )

    for split, items in splits.items():
        ids, learn, offsets, sources, lengths, exchanges = [], [], [0], [], Counter(), Counter()
        turns = []  # exchanges (user question + answer) of each conversation
        with open(SFT_DIR / f"{split}.jsonl", "w", encoding="utf-8") as jsonl:
            for item in items:
                fitted = fit(tokenizer, item["messages"], args.max_tokens)
                if fitted is None:
                    stats["dropped"][f"first exchange longer than {args.max_tokens} tokens"] += 1
                    continue
                messages, tokens, targets = fitted
                if len(messages) < len(item["messages"]):
                    stats["dropped"]["later turns cut to fit the context"] += 1
                item = {**item, "messages": messages}
                turns.append(sum(m["role"] == "user" for m in messages))
                exchanges[min(turns[-1], 5)] += 1
                jsonl.write(json.dumps(item, ensure_ascii=False) + "\n")
                ids += tokens
                learn += targets
                offsets.append(len(ids))
                sources.append(item["source"])
                lengths[item["source"]] += len(tokens)
        np.savez(
            SFT_DIR / f"{split}.npz",
            ids=np.array(ids, dtype=np.uint16),
            learn=np.array(learn, dtype=bool),
            offsets=np.array(offsets, dtype=np.int64),
            sources=np.array(sources),
            exchanges=np.array(turns, dtype=np.int16),
            tokenizer=np.array(DEFAULT_TOKENIZER),
        )
        stats[split] = {
            "conversations": len(offsets) - 1,
            "tokens": len(ids),
            "learned_tokens": int(np.sum(learn)),
            "tokens_by_source": dict(lengths),
            "conversations_by_exchanges": {f"{k}+" if k == 5 else str(k): exchanges[k] for k in sorted(exchanges)},
        }
        print(f"{split}: {len(offsets) - 1:,} conversations, {len(ids):,} tokens ({int(np.sum(learn)):,} learned)")
        print(f"  by number of exchanges: {stats[split]['conversations_by_exchanges']}")

    stats["dropped"] = dict(stats["dropped"])
    (SFT_DIR / "stats.json").write_text(json.dumps(stats, indent=2, ensure_ascii=False), encoding="utf-8")
    print("dropped:", stats["dropped"])


if __name__ == "__main__":
    sys.exit(main())
