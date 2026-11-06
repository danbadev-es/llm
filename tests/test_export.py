"""Checks for leon.export without llama.cpp: export a tiny model and read the GGUF back.
The end-to-end check against llama.cpp itself is `python -m leon.export ... --verify <llama.cpp dir>`.
Run: pytest tests/test_export.py"""

import tempfile
from pathlib import Path

import numpy as np
import pytest
import torch

gguf = pytest.importorskip("gguf")  # optional extra: pip install -e ".[export]"

from leon import export  # noqa: E402
from leon.model import Leon, LeonConfig  # noqa: E402
from leon.tokenizer import BPETokenizer, count_chunks  # noqa: E402
from leon.training import save_checkpoint  # noqa: E402


def field(reader, key: str):
    """The Python value of a GGUF metadata field."""
    value = reader.fields[key]
    if value.types[0] == gguf.GGUFValueType.ARRAY:
        return [value.contents(i) for i in range(len(value.data))]
    return value.contents()


def test_export_writes_a_llama_gguf(monkeypatch) -> None:
    tokenizer = BPETokenizer.train(count_chunks(["hola qué tal, muy bien gracias. ¿Y tú? 1234"] * 5), 300)
    monkeypatch.setattr(export, "load_tokenizer", lambda *_: tokenizer)
    config = LeonConfig(vocab_size=tokenizer.vocab_size, block_size=32, n_layer=2, n_head=4, n_kv_head=2, n_embd=32)
    torch.manual_seed(0)
    model = Leon(config)
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
        checkpoint = Path(tmp) / "best.pt"
        save_checkpoint(checkpoint, model, step=1, best_val=0.0, stage="chat")
        output = Path(tmp) / "tiny.gguf"
        export.export(checkpoint, output, dtype="f32")
        reader = gguf.GGUFReader(output)

        assert field(reader, "general.architecture") == "llama"
        assert field(reader, "tokenizer.ggml.pre") == "llama-bpe"
        assert field(reader, "llama.attention.head_count_kv") == 2
        assert field(reader, "llama.context_length") == 32
        assert field(reader, "tokenizer.ggml.add_bos_token") is False
        assert field(reader, "tokenizer.ggml.eos_token_id") == tokenizer.special_tokens["<|end|>"]
        assert field(reader, "tokenizer.chat_template") == export.CHAT_TEMPLATE
        tokens = field(reader, "tokenizer.ggml.tokens")
        assert len(tokens) == tokenizer.vocab_size and tokens[ord("a")] == "a" and tokens[ord(" ")] == "Ġ"
        assert len(field(reader, "tokenizer.ggml.merges")) == len(tokenizer.merges)

        tensors = {t.name: t for t in reader.tensors}
        assert "output.weight" not in tensors  # tied: llama.cpp reuses token_embd
        assert len(tensors) == 2 + 9 * config.n_layer
        for leon_name, gguf_name in export.tensor_names(config.n_layer).items():
            original = model.state_dict()[leon_name].numpy()
            stored = np.asarray(tensors[gguf_name].data).reshape(original.shape)
            assert np.array_equal(stored, original), gguf_name  # f32: bit-exact, no permutation


def test_byte_map_is_reversible() -> None:
    mapping = export.bytes_to_unicode()
    assert len(set(mapping.values())) == 256  # every byte gets its own character
    assert mapping[ord("A")] == "A" and mapping[ord("\n")] == "Ċ"
