"""Correctness checks for leon.tokenizer. Run: python tests/test_tokenizer.py (or pytest)"""

import random
import re
import tempfile
from collections import Counter
from pathlib import Path

from leon.tokenizer import PRETOKENIZE_PATTERN, SPECIAL_TOKENS, BPETokenizer, count_chunks, merge_pair

SAMPLE = (
    "En un lugar de la Mancha, de cuyo nombre no quiero acordarme, no ha mucho tiempo "
    "que vivía un hidalgo de los de lanza en astillero, adarga antigua, rocín flaco y galgo corredor.\n"
    "¿Qué día es hoy? ¡El 23 de abril de 1616! Año, niño, pingüino, señor_ñandú 3,14159.\r\n"
    "Emoji 🐉 y acentos combinados: é à.   Fin.\n\n"
)


def naive_train(text: str, n_merges: int) -> tuple[list[tuple[int, int]], dict[str, list[int]]]:
    """Reference trainer: recount every pair from scratch before each merge (slow but obvious).

    Returns the merges and the final segmentation of every chunk.
    """
    frequency = Counter(re.findall(PRETOKENIZE_PATTERN, text))
    words = {chunk: list(chunk.encode("utf-8")) for chunk in frequency}
    merges = []
    for i in range(n_merges):
        counts = Counter()
        for chunk, word in words.items():
            for pair in zip(word, word[1:]):
                counts[pair] += frequency[chunk]
        if not counts:
            break
        best_count = max(counts.values())
        pair = min(p for p, c in counts.items() if c == best_count)  # same tie-break as BPETokenizer
        merges.append(pair)
        words = {chunk: merge_pair(word, pair, 256 + i) for chunk, word in words.items()}
    return merges, words


def test_pretokenizer_is_lossless() -> None:
    """Every character must land in some chunk, otherwise encoding would silently drop text."""
    rng = random.Random(0)
    alphabet = "aáñÑ1²_ \t\n\r.,¿?¡!🐉́-'\"«»€"
    for _ in range(200):
        text = "".join(rng.choice(alphabet) for _ in range(rng.randint(0, 40)))
        assert "".join(re.findall(PRETOKENIZE_PATTERN, text)) == text, repr(text)


def test_pretokenizer_is_lossless_on_all_of_unicode() -> None:
    """Random code points from the whole Unicode range (surrogates excluded: not valid text)."""
    rng = random.Random(1)
    for _ in range(300):
        chars = []
        while len(chars) < 30:
            code = rng.randrange(0x110000)
            if not 0xD800 <= code <= 0xDFFF:
                chars.append(chr(code))
        text = "".join(chars)
        assert "".join(re.findall(PRETOKENIZE_PATTERN, text)) == text, repr(text)


def test_incremental_training_matches_naive() -> None:
    text = SAMPLE * 3
    n_merges = 60
    tokenizer = BPETokenizer.train(count_chunks([text]), 256 + n_merges + len(SPECIAL_TOKENS))
    merges, _ = naive_train(text, n_merges)
    assert tokenizer.merges == merges


def test_encode_reproduces_training_segmentation() -> None:
    """Applying merges by rank at encode time must give the segmentation training ended with."""
    text = SAMPLE * 3
    n_merges = 80
    tokenizer = BPETokenizer.train(count_chunks([text]), 256 + n_merges + len(SPECIAL_TOKENS))
    _, words = naive_train(text, n_merges)
    for chunk, ids in words.items():
        assert tokenizer.encode(chunk) == ids, repr(chunk)


def test_lone_surrogate_does_not_crash() -> None:
    tokenizer = BPETokenizer.train(count_chunks([SAMPLE]), 400)
    assert tokenizer.decode(tokenizer.encode("hola \ud800 adiós")) == "hola ? adiós"


def test_roundtrip() -> None:
    tokenizer = BPETokenizer.train(count_chunks([SAMPLE]), 400)
    ids = tokenizer.encode(SAMPLE)
    assert tokenizer.decode(ids) == SAMPLE
    assert len(ids) < len(SAMPLE.encode("utf-8")), "merges should compress the text"
    assert all(0 <= i < tokenizer.vocab_size for i in ids)
    unseen = "Texto nunca visto: ZXQ ж 漢字 ⚔️"
    assert tokenizer.decode(tokenizer.encode(unseen)) == unseen, "byte fallback must cover any text"


def test_special_tokens() -> None:
    tokenizer = BPETokenizer.train(count_chunks([SAMPLE]), 400)
    text = "<|user|>Hola<|end|>"
    ids = tokenizer.encode(text, allow_special=True)
    assert ids[0] == tokenizer.special_tokens["<|user|>"]
    assert ids[-1] == tokenizer.special_tokens["<|end|>"]
    assert tokenizer.decode(ids) == text
    plain = tokenizer.encode(text)  # default: special strings are just text
    assert not set(plain) & set(tokenizer.special_tokens.values())
    assert tokenizer.decode(plain) == text


def test_save_load() -> None:
    tokenizer = BPETokenizer.train(count_chunks([SAMPLE]), 400)
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "tok.json"
        tokenizer.save(path)
        loaded = BPETokenizer.load(path)
    assert loaded.merges == tokenizer.merges
    assert loaded.special_tokens == tokenizer.special_tokens
    assert loaded.encode(SAMPLE) == tokenizer.encode(SAMPLE)


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"ok  {name}")
