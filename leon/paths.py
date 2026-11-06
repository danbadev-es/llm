"""Project locations. Data and checkpoints live next to the package and are not versioned."""

from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
CHECKPOINT_DIR = ROOT / "checkpoints"
TOKENIZER_PATH = DATA_DIR / "tokenizer" / "leon_bpe.json"
