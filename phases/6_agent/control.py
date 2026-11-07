"""Start, pause, resume and watch the Leon 0.1 tool-calling run (phase 6).

    python phases/6_agent/control.py start      start training in the background
    python phases/6_agent/control.py pause      finish the current step, save and stop
    python phases/6_agent/control.py resume     continue where it paused
    python phases/6_agent/control.py status     the dashboard once: progress, loss, ETA, latest answers
    python phases/6_agent/control.py watch      live dashboard that refreshes itself

The trainer is an independent process with no window; `start` and `resume` also open a `watch`
window (skip it with --no-watch), and closing it does not stop the training.
Arguments after `--` are passed to train.py.
"""

from pathlib import Path

from leon.paths import CHECKPOINT_DIR, DATA_DIR
from leon.run import Control


def check() -> None:
    if not (DATA_DIR / "agent" / "train.npz").exists():
        raise SystemExit("No tool-calling data yet. Run: python phases/6_agent/prepare.py")
    if not (CHECKPOINT_DIR / "5_sft" / "leon_0.1_chat" / "best.pt").exists():
        raise SystemExit("No phase 5 chat model yet: phase 6 starts from checkpoints/5_sft/leon_0.1_chat/best.pt")


CONTROL = Control(
    title="Leon 0.1 · tools (phase 6)",
    runs_dir=CHECKPOINT_DIR / "6_agent",
    script=Path(__file__).resolve().parent / "train.py",
    control=Path(__file__),
    default_run="leon_0.1_agent",
    check=check,
)

if __name__ == "__main__":
    CONTROL.main(__doc__)
