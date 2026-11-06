"""Project locations. Data and checkpoints live next to the package and are not versioned."""

from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
CHECKPOINT_DIR = ROOT / "checkpoints"
TOKENIZER_DIR = DATA_DIR / "tokenizer"

# Tokenizer versions. A checkpoint records the one it was trained with ("tokenizer" key);
# checkpoints written before that key existed used v1.
#   v1  phase 2's first tokenizer (own pre-tokenizer regex); the phase 3 models use it
#   v2  same data and vocabulary size, llama.cpp's Llama 3 pre-tokenizer regex: exportable to
#       GGUF (LM Studio, Ollama) with identical tokenization. Leon 0.1 uses it from phase 4 on.
TOKENIZERS = {"v1": TOKENIZER_DIR / "leon_bpe_v1.json", "v2": TOKENIZER_DIR / "leon_bpe_v2.json"}
DEFAULT_TOKENIZER = "v2"
LEGACY_TOKENIZER = "v1"
TOKENIZER_PATH = TOKENIZERS[DEFAULT_TOKENIZER]
