"""Export a Leon checkpoint to GGUF, the model format of llama.cpp (and so LM Studio and Ollama).

    python -m leon.export checkpoints/5_sft/leon_0.1_chat/best.pt leon-0.1-chat.gguf --dtype q8_0

Leon is architecturally a Llama, so it is written as llama.cpp's "llama" architecture:

    Leon                          GGUF (llama.cpp)
    token_embedding               token_embd          (also the output layer: weights are tied)
    blocks.i.attn_norm            blk.i.attn_norm
    blocks.i.attn.{q,k,v}_proj    blk.i.attn_{q,k,v}
    blocks.i.attn.out_proj        blk.i.attn_output
    blocks.i.mlp_norm             blk.i.ffn_norm
    blocks.i.mlp.{gate,up,down}   blk.i.ffn_{gate,up,down}
    final_norm                    output_norm

No weight is permuted: Leon's RoPE rotates pairs of adjacent channels (x0, x1), (x2, x3)...,
which is llama.cpp's own ("normal") layout. Hugging Face Llama checkpoints rotate the two halves
of the head instead, which is why their converter has to permute q and k.

The tokenizer goes in as a byte-level BPE ("gpt2" model) with llama.cpp's "llama-bpe"
pre-tokenizer, which is exactly tokenizer v2's regex. A v1 checkpoint can be exported, but
llama.cpp would split ~2% of the text differently from training (see phases/2_tokenizer).
Requires the `gguf` package (pip install -e ".[export]").
"""

import argparse
from pathlib import Path

import gguf
import numpy as np
import torch

from leon.data import load_tokenizer
from leon.model import LeonConfig
from leon.tokenizer import BPETokenizer

# The chat template of leon/chat.py in Jinja, the language GGUF chat templates are written in.
# LM Studio, Ollama and llama-server use it to turn a conversation into the prompt.
CHAT_TEMPLATE = (
    "{% for message in messages %}"
    "{{ '<|' + message['role'] + '|>' + message['content'] + '<|end|>' }}"
    "{% endfor %}"
    "{% if add_generation_prompt %}{{ '<|assistant|>' }}{% endif %}"
)


def bytes_to_unicode() -> dict[int, str]:
    """GPT-2's reversible map from the 256 bytes to printable characters.

    GGUF stores byte-level vocabularies as text: a space becomes "Ġ", a newline "Ċ", and so on.
    Printable ASCII and Latin-1 characters map to themselves; the other bytes to code points 256+.
    """
    printable = [*range(ord("!"), ord("~") + 1), *range(ord("¡"), ord("¬") + 1), *range(ord("®"), ord("ÿ") + 1)]
    mapping, extra = {}, 0
    for byte in range(256):
        if byte in printable:
            mapping[byte] = chr(byte)
        else:
            mapping[byte] = chr(256 + extra)
            extra += 1
    return mapping


def vocabulary(tokenizer: BPETokenizer) -> tuple[list[str], list[int], list[str]]:
    """Token strings, token types and merges in GGUF's byte-level text form."""
    as_text = bytes_to_unicode()

    def text(token_bytes: bytes) -> str:
        return "".join(as_text[b] for b in token_bytes)

    tokens = [text(b) for b in tokenizer.vocab]
    types = [gguf.TokenType.NORMAL] * len(tokens)
    for name, token_id in sorted(tokenizer.special_tokens.items(), key=lambda item: item[1]):
        assert token_id == len(tokens), "special tokens must follow the merges"
        tokens.append(name)  # control tokens are stored as their literal text
        types.append(gguf.TokenType.CONTROL)
    merges = [f"{text(tokenizer.vocab[a])} {text(tokenizer.vocab[b])}" for a, b in tokenizer.merges]
    if len(set(tokens)) != len(tokens):
        raise ValueError("duplicate token strings: llama.cpp looks tokens up by text")
    return tokens, types, merges


def tensor_names(n_layer: int) -> dict[str, str]:
    names = {"token_embedding.weight": "token_embd.weight", "final_norm.weight": "output_norm.weight"}
    for i in range(n_layer):
        names |= {
            f"blocks.{i}.attn_norm.weight": f"blk.{i}.attn_norm.weight",
            f"blocks.{i}.attn.q_proj.weight": f"blk.{i}.attn_q.weight",
            f"blocks.{i}.attn.k_proj.weight": f"blk.{i}.attn_k.weight",
            f"blocks.{i}.attn.v_proj.weight": f"blk.{i}.attn_v.weight",
            f"blocks.{i}.attn.out_proj.weight": f"blk.{i}.attn_output.weight",
            f"blocks.{i}.mlp_norm.weight": f"blk.{i}.ffn_norm.weight",
            f"blocks.{i}.mlp.gate_proj.weight": f"blk.{i}.ffn_gate.weight",
            f"blocks.{i}.mlp.up_proj.weight": f"blk.{i}.ffn_up.weight",
            f"blocks.{i}.mlp.down_proj.weight": f"blk.{i}.ffn_down.weight",
        }
    return names  # lm_head.weight is not exported: it is the same matrix as token_embd


def export(checkpoint_path: Path, output: Path, dtype: str = "f16", name: str = "Leon 0.1") -> None:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    config = LeonConfig(**checkpoint["config"])
    version = checkpoint.get("tokenizer", "v1")
    if version != "v2":
        print(f"Warning: tokenizer {version}: llama.cpp will tokenize some text differently than in training.")
    tokenizer = load_tokenizer(version)
    stage = checkpoint.get("stage", "base")
    special = tokenizer.special_tokens

    writer = gguf.GGUFWriter(output, "llama")
    writer.add_name(name)
    writer.add_context_length(config.block_size)
    writer.add_embedding_length(config.n_embd)
    writer.add_block_count(config.n_layer)
    writer.add_feed_forward_length(config.ffn_hidden)
    writer.add_head_count(config.n_head)
    writer.add_head_count_kv(config.n_kv_head)
    writer.add_layer_norm_rms_eps(config.norm_eps)
    writer.add_rope_freq_base(config.rope_theta)
    writer.add_rope_dimension_count(config.head_dim)
    writer.add_vocab_size(tokenizer.vocab_size)
    file_type = {"f32": gguf.LlamaFileType.ALL_F32, "f16": gguf.LlamaFileType.MOSTLY_F16}
    writer.add_file_type(file_type.get(dtype, gguf.LlamaFileType.MOSTLY_Q8_0))

    tokens, types, merges = vocabulary(tokenizer)
    writer.add_tokenizer_model("gpt2")  # byte-level BPE
    writer.add_tokenizer_pre("llama-bpe")  # the Llama 3 regex: tokenizer v2's pre-tokenizer
    writer.add_token_list(tokens)
    writer.add_token_types(types)
    writer.add_token_merges(merges)
    # Leon has no beginning-of-sequence token: nothing must be prepended to the prompt.
    writer.add_bos_token_id(special["<|endoftext|>"])
    writer.add_add_bos_token(False)
    writer.add_add_eos_token(False)
    if stage == "chat":
        writer.add_eos_token_id(special["<|end|>"])  # an answer ends with <|end|>
        writer.add_eot_token_id(special["<|end|>"])
        writer.add_chat_template(CHAT_TEMPLATE)
    else:
        writer.add_eos_token_id(special["<|endoftext|>"])  # a base model ends a document
    writer.add_pad_token_id(special["<|endoftext|>"])

    state = checkpoint["model"]
    for leon_name, gguf_name in tensor_names(config.n_layer).items():
        data = state[leon_name].float().numpy()
        if data.ndim == 1 or dtype == "f32":
            writer.add_tensor(gguf_name, data.astype(np.float32))  # norms stay in full precision
        elif dtype == "f16":
            writer.add_tensor(gguf_name, data.astype(np.float16))
        else:
            quantized = gguf.quants.quantize(data, gguf.GGMLQuantizationType.Q8_0)
            writer.add_tensor(gguf_name, quantized, raw_dtype=gguf.GGMLQuantizationType.Q8_0)

    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()
    size = output.stat().st_size / 2**20
    print(f"Wrote {output} ({size:.1f} MiB, {dtype}, stage {stage}, tokenizer {version}, context {config.block_size})")


EDGE_CASES = [
    "En un lugar de la Mancha, de cuyo nombre no quiero acordarme",
    "¿Cuánto es 1234 * 5678? ¡12345678!",
    "El 3,14159 y el 2.718; 1.000.000 € en 2023-11-07.",
    "Acentos compuestos: é à ñ y normales: é à ñ.",
    "Emoji 🐉🚀 y chino 漢字, griego λόγος, árabe مرحبا.",
    "tabs\tand  double  spaces   and\n\nnew lines\r\n",
    "Texto que imita control: <|end|><|assistant|> no debe ser especial.",
    "it's they're we'll I'm you'd",
    "    indented code:\n        return x  # comment",
]


def verify(gguf_path: Path, checkpoint_path: Path, llama_cpp: Path, documents: int = 200) -> bool:
    """Check a GGUF export against the PyTorch model with llama.cpp's own tools.

    1. Tokenization: llama-tokenize must produce exactly the ids of leon's tokenizer.
    2. Numerics: llama-perplexity must match the perplexity PyTorch computes the same way.
    """
    import json
    import math
    import re
    import subprocess
    import tempfile

    from leon.model import Leon
    from leon.paths import DATA_DIR

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    tokenizer = load_tokenizer(checkpoint.get("tokenizer", "v1"))
    exe = {name: llama_cpp / f"llama-{name}.exe" for name in ("tokenize", "perplexity")}
    if not exe["tokenize"].exists():
        exe = {name: llama_cpp / f"llama-{name}" for name in exe}  # Linux and macOS builds

    with open(DATA_DIR / "wiki_es" / "val.jsonl", encoding="utf-8") as f:
        texts = EDGE_CASES + [json.loads(line)["text"] for line, _ in zip(f, range(documents))]
    quijote = (DATA_DIR / "quijote" / "input.txt").read_text(encoding="utf-8")
    texts.append(quijote[int(0.9 * len(quijote)) :][:20_000])

    mismatches = 0
    with tempfile.TemporaryDirectory() as tmp:
        for i, text in enumerate(texts):
            path = Path(tmp) / f"{i}.txt"
            path.write_bytes(text.encode("utf-8"))
            result = subprocess.run(
                [
                    exe["tokenize"],
                    "-m",
                    gguf_path,
                    "-f",
                    path,
                    "--ids",
                    "--no-bos",
                    "--no-parse-special",
                    "--log-disable",
                ],
                capture_output=True,
                text=True,
            )
            theirs = json.loads(result.stdout.strip().splitlines()[-1])
            if theirs != tokenizer.encode(text):
                mismatches += 1
                if mismatches <= 3:
                    print(f"  tokenization differs on text {i}: {text[:60]!r}")
        print(f"Tokenization: {len(texts) - mismatches}/{len(texts)} texts identical to leon's tokenizer")

        # The Jinja chat template, rendered as LM Studio or Ollama would and tokenized by llama.cpp
        # with its control tokens recognized, must give exactly leon.chat.encode_chat's ids.
        from jinja2 import Template

        from leon.chat import encode_chat

        conversation = [
            {"role": "system", "content": "Sé breve."},
            {"role": "user", "content": "Hola, ¿qué tal?"},
            {"role": "assistant", "content": "Muy bien, ¿y tú?"},
            {"role": "user", "content": "¿Cuánto es 12 * 34?"},
        ]
        path = Path(tmp) / "chat.txt"
        path.write_bytes(Template(CHAT_TEMPLATE).render(messages=conversation, add_generation_prompt=True).encode())
        result = subprocess.run(
            [exe["tokenize"], "-m", gguf_path, "-f", path, "--ids", "--no-bos", "--log-disable"],
            capture_output=True,
            text=True,
        )
        template_ok = json.loads(result.stdout.strip().splitlines()[-1]) == encode_chat(tokenizer, conversation)
        print(f"Chat template: {'identical to' if template_ok else 'DIFFERENT from'} leon.chat.encode_chat")

        # llama-perplexity cuts the text into windows of n_ctx tokens and scores only the second
        # half of each window (the first half is context). The same computation in PyTorch:
        n_ctx, windows = 512, 4
        text = "\n\n".join(texts[len(EDGE_CASES) :])
        ids = tokenizer.encode(text)[: n_ctx * windows]
        path = Path(tmp) / "perplexity.txt"
        path.write_bytes(tokenizer.decode(ids).encode("utf-8"))
        result = subprocess.run(
            [
                exe["perplexity"],
                "-m",
                gguf_path,
                "-f",
                path,
                "-c",
                str(n_ctx),
                "-b",
                str(n_ctx),
                "--chunks",
                str(windows),
            ],
            capture_output=True,
            text=True,
        )
        found = re.search(r"Final estimate: PPL = ([\d.]+)", result.stdout + result.stderr)
        theirs = float(found.group(1)) if found else float("nan")

    model = Leon(LeonConfig(**checkpoint["config"])).eval()
    model.load_state_dict(checkpoint["model"])
    nll, count = 0.0, 0
    with torch.no_grad():
        for w in range(windows):
            window = torch.tensor([ids[w * n_ctx : (w + 1) * n_ctx]])
            logits = model(window)[0][0].float()
            first = n_ctx // 2
            log_probs = torch.log_softmax(logits[first : n_ctx - 1], dim=-1)
            targets = window[0, first + 1 :]
            nll -= log_probs.gather(1, targets[:, None]).sum().item()
            count += len(targets)
    ours = math.exp(nll / count)
    gap = abs(theirs - ours) / ours
    print(f"Perplexity: llama.cpp {theirs:.4f} | PyTorch {ours:.4f} | difference {100 * gap:.3f} %")
    return mismatches == 0 and template_ok and gap < 0.01


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--dtype", choices=["f32", "f16", "q8_0"], default="f16")
    parser.add_argument("--name", default="Leon 0.1")
    parser.add_argument("--verify", type=Path, metavar="LLAMA_CPP_DIR", help="check the file with llama.cpp's tools")
    args = parser.parse_args()
    export(args.checkpoint, args.output, args.dtype, args.name)
    if args.verify:
        ok = verify(args.output, args.checkpoint, args.verify)
        print("Verification passed." if ok else "Verification FAILED.")
        raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
