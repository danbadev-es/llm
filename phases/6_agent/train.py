"""Phase 6: teach the chat model to call tools (supervised fine-tuning on phases/6_agent/prepare.py's data).

Starts from the phase 5 chat model and fine-tunes it on dialogs where the answer calls a tool
(calculadora, fecha_hora, buscar_documentos) and continues after its result, plus phase 5
conversations that need no tool. Only the calls and the answers are learned, never the results.
Usually run through phases/6_agent/control.py (start / pause / resume / status / watch).

The training loop is leon/sft.py, the same as phase 5.
"""

import argparse

from leon.chat import DEFAULT_SYSTEM
from leon.paths import CHECKPOINT_DIR, DATA_DIR
from leon.sft import add_sft_args, train_sft
from leon.tools import system_prompt

CHECKPOINT_ROOT = CHECKPOINT_DIR / "6_agent"
BASE = CHECKPOINT_DIR / "5_sft" / "leon_0.1_chat" / "best.pt"
AGENT_DIR = DATA_DIR / "agent"
SYSTEM = system_prompt()
# One question per tool, one that needs none: the dashboard shows whether the model calls the
# right tool (the call is kept in the answer's text) or answers directly.
PROBES = [
    [{"role": "system", "content": SYSTEM}, {"role": "user", "content": "¿Cuánto es 128 por 37?"}],
    [{"role": "system", "content": SYSTEM}, {"role": "user", "content": "¿Qué día es hoy?"}],
    [{"role": "system", "content": SYSTEM}, {"role": "user", "content": "¿Qué dicen mis documentos sobre Anoeta?"}],
    [{"role": "system", "content": SYSTEM}, {"role": "user", "content": "Hola, ¿quién eres?"}],
    [{"role": "system", "content": DEFAULT_SYSTEM}, {"role": "user", "content": "¿Cuál es la capital de Francia?"}],
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    # Two epochs: the tool dialogs are short and repetitive, more would overfit their templates.
    add_sft_args(parser, base=BASE, run="leon_0.1_agent", epochs=2, upsample=["varias=2", "no_encontrado=2"])
    return parser.parse_args()


def main() -> None:
    train_sft(parse_args(), AGENT_DIR, CHECKPOINT_ROOT, PROBES, stage="agent")


if __name__ == "__main__":
    main()
