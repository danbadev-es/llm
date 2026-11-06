"""Byte-level BPE tokenizer, implemented from scratch.

Vocabulary layout (ids):
    0..255                      the 256 raw bytes (any text can always be encoded)
    256..256+n_merges-1         one new token per learned merge, in training order
    after that                  special tokens (chat template and tool calling, used from phase 5)

Text is first split into chunks by a regex (pre-tokenization) so that merges never
cross word boundaries: " Mancha" can become one token, but "a," can never be.
The `regex` module is used instead of `re` because it understands Unicode classes
such as \\p{L} (any letter) and \\p{N} (any number), which the pre-tokenizer needs.
"""

import heapq
import json
import time
from collections import Counter, defaultdict
from collections.abc import Iterable
from pathlib import Path

import regex as re  # drop-in replacement for re, with \p{...} classes

# The pre-tokenizer of Llama 3, exactly as llama.cpp implements it (its "llama-bpe" type),
# so a GGUF export tokenizes identically in LM Studio and Ollama. A chunk is:
#   - an English contraction ('s, 't, 're, 've, 'm, 'll, 'd)
#   - letters, optionally preceded by ONE other character: " Mancha", "¿Cuánto", "(hola"
#   - 1 to 3 digits (numbers are split in groups: 1234 -> 123 | 4)
#   - an optional space and punctuation, with any newlines that follow it
#   - runs of whitespace
# Every character falls into some class, so the split is lossless: "".join(chunks) == text.
PRETOKENIZE_PATTERN = (
    r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}"
    r"| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+"
)
# The v1 tokenizer's own pattern (it is stored in leon_bpe_v1.json too): an optional space plus
# letters, 1-3 digits or punctuation, or whitespace. ("_" counts as \w, hence the explicit "|_".)
PRETOKENIZE_PATTERN_V1 = r" ?[^\W\d_]+| ?\d{1,3}| ?(?:[^\w\s]|_)+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+"

# Reserved now so the vocabulary never changes between pretraining (phase 4),
# chat SFT (phase 5) and tool-calling SFT (phase 6).
SPECIAL_TOKENS = [
    "<|endoftext|>",
    "<|system|>",
    "<|user|>",
    "<|assistant|>",
    "<|end|>",
    "<|tool_call|>",
    "<|/tool_call|>",
    "<|tool_result|>",
    "<|/tool_result|>",
] + [f"<|reserved_{i}|>" for i in range(7)]


def merge_pair(ids: list[int], pair: tuple[int, int], new_id: int) -> list[int]:
    """Replace every non-overlapping occurrence of `pair` in `ids` with `new_id`."""
    out, i, n = [], 0, len(ids)
    while i < n:
        if i < n - 1 and ids[i] == pair[0] and ids[i + 1] == pair[1]:
            out.append(new_id)
            i += 2
        else:
            out.append(ids[i])
            i += 1
    return out


def count_chunks(texts: Iterable[str], pattern: str = PRETOKENIZE_PATTERN) -> Counter[str]:
    """Frequency of every pre-tokenized chunk. BPE trains on this table, not on the raw text."""
    regex = re.compile(pattern)
    counts: Counter[str] = Counter()
    for text in texts:
        counts.update(regex.findall(text))
    return counts


class BPETokenizer:
    def __init__(
        self,
        merges: list[tuple[int, int]],
        special_tokens: list[str] = SPECIAL_TOKENS,
        pattern: str = PRETOKENIZE_PATTERN,
    ) -> None:
        self.merges = merges
        self.pattern = pattern
        self.regex = re.compile(pattern)
        # Rank of a merge = its new id; lower ids were learned earlier and are applied first.
        self.ranks = {pair: 256 + i for i, pair in enumerate(merges)}

        self.vocab = [bytes([b]) for b in range(256)]
        for a, b in merges:
            self.vocab.append(self.vocab[a] + self.vocab[b])

        first_special = 256 + len(merges)
        self.special_tokens = {token: first_special + i for i, token in enumerate(special_tokens)}
        self.special_regex = re.compile("(" + "|".join(re.escape(t) for t in special_tokens) + ")")
        self.id_to_special = {i: t for t, i in self.special_tokens.items()}
        self.cache: dict[str, list[int]] = {}

    @property
    def vocab_size(self) -> int:
        return len(self.vocab) + len(self.special_tokens)

    @property
    def eot_id(self) -> int:
        return self.special_tokens["<|endoftext|>"]

    # ------------------------------------------------------------------ training

    @classmethod
    def train(
        cls,
        chunk_counts: Counter[str],
        vocab_size: int,
        special_tokens: list[str] = SPECIAL_TOKENS,
        verbose: bool = False,
    ) -> "BPETokenizer":
        """Learn merges greedily: repeatedly fuse the most frequent adjacent pair.

        Recounting every pair after each merge would cost O(corpus) per merge. Instead we
        keep `pair_counts` up to date incrementally and only revisit the chunks that
        contain the merged pair (`where`). A max-heap with lazy deletion finds the most
        frequent pair: stale entries are discarded when popped.
        """
        n_merges = vocab_size - 256 - len(special_tokens)
        if n_merges < 0:
            raise ValueError(f"vocab_size must be at least {256 + len(special_tokens)}")

        words = [list(chunk.encode("utf-8", errors="replace")) for chunk in chunk_counts]
        freqs = list(chunk_counts.values())

        pair_counts: dict[tuple[int, int], int] = defaultdict(int)
        where: dict[tuple[int, int], set[int]] = defaultdict(set)  # pair -> indices of words containing it
        for i, word in enumerate(words):
            for pair in zip(word, word[1:]):
                pair_counts[pair] += freqs[i]
                where[pair].add(i)

        # Ties are broken by the smallest pair, so training is deterministic.
        heap = [(-count, pair) for pair, count in pair_counts.items()]
        heapq.heapify(heap)

        merges: list[tuple[int, int]] = []
        vocab = [bytes([b]) for b in range(256)]
        start = time.perf_counter()
        while len(merges) < n_merges and heap:
            neg_count, pair = heapq.heappop(heap)
            if pair_counts.get(pair, 0) != -neg_count:
                continue  # stale heap entry
            new_id = 256 + len(merges)
            merges.append(pair)
            vocab.append(vocab[pair[0]] + vocab[pair[1]])

            changed = set()
            for i in where.pop(pair):
                word = words[i]
                merged = merge_pair(word, pair, new_id)
                if len(merged) == len(word):
                    continue  # `where` may keep indices of words that no longer contain the pair
                for p in zip(word, word[1:]):
                    pair_counts[p] -= freqs[i]
                    changed.add(p)
                for p in zip(merged, merged[1:]):
                    pair_counts[p] += freqs[i]
                    where[p].add(i)
                    changed.add(p)
                words[i] = merged

            for p in changed:
                if pair_counts[p] > 0:
                    heapq.heappush(heap, (-pair_counts[p], p))
                else:
                    del pair_counts[p]

            if verbose and (len(merges) % 1000 == 0 or len(merges) == n_merges):
                elapsed = time.perf_counter() - start
                print(
                    f"merge {len(merges):6d}/{n_merges} | count {-neg_count:>9,} | {vocab[-1]!r:<24} | {elapsed:6.1f}s"
                )
        return cls(merges, special_tokens)

    # ------------------------------------------------------------------ encoding

    def _encode_chunk(self, chunk: str) -> list[int]:
        """Apply the learned merges to one chunk, earliest-learned first (as in training)."""
        cached = self.cache.get(chunk)
        if cached is not None:
            return cached
        # errors="replace": a lone surrogate (invalid Unicode, e.g. from a broken JSON escape)
        # becomes "?" instead of crashing. Valid text is always encoded losslessly.
        ids = list(chunk.encode("utf-8", errors="replace"))
        while len(ids) >= 2:
            pair = min(zip(ids, ids[1:]), key=lambda p: self.ranks.get(p, float("inf")))
            if pair not in self.ranks:
                break
            ids = merge_pair(ids, pair, self.ranks[pair])
        if len(self.cache) < 1_000_000:  # bounded memory; frequent chunks get cached first
            self.cache[chunk] = ids
        return ids

    def encode_ordinary(self, text: str) -> list[int]:
        """Encode text, treating anything that looks like a special token as plain text."""
        ids = []
        for chunk in self.regex.findall(text):
            ids.extend(self._encode_chunk(chunk))
        return ids

    def encode(self, text: str, allow_special: bool = False) -> list[int]:
        """Encode text. With allow_special, strings like "<|user|>" map to their special id.

        Off by default: text scraped from the web must never be able to inject control tokens.
        """
        if not allow_special:
            return self.encode_ordinary(text)
        ids = []
        for part in self.special_regex.split(text):
            if part in self.special_tokens:
                ids.append(self.special_tokens[part])
            elif part:
                ids.extend(self.encode_ordinary(part))
        return ids

    def token_bytes(self, token_id: int) -> bytes:
        if token_id in self.id_to_special:
            return self.id_to_special[token_id].encode("utf-8")
        return self.vocab[token_id]

    def decode(self, ids: list[int]) -> str:
        # A token may hold half of a multi-byte character, so join bytes before decoding.
        return b"".join(self.token_bytes(i) for i in ids).decode("utf-8", errors="replace")

    # ------------------------------------------------------------------ persistence

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "version": 1,
            "pattern": self.pattern,
            "merges": self.merges,
            "special_tokens": list(self.special_tokens),
        }
        path.write_text(json.dumps(data), encoding="utf-8")
        # Human-readable dump, to browse what the tokenizer learned.
        lines = [
            f"{i}\t{self.token_bytes(i).decode('utf-8', errors='backslashreplace')!r}" for i in range(self.vocab_size)
        ]
        path.with_suffix(".vocab.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> "BPETokenizer":
        data = json.loads(path.read_text(encoding="utf-8"))
        return cls([tuple(m) for m in data["merges"]], data["special_tokens"], data["pattern"])
