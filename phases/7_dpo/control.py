"""Make preference pairs and train Leon 0.1 with DPO (phase 7), in the background.

    python phases/7_dpo/control.py prepare    sample Leon's answers to build the pairs (GPU, ~30 min)
    python phases/7_dpo/control.py start      train with DPO (GPU)
    python phases/7_dpo/control.py pause      save and stop (training, or prepare with --run prepare)
    python phases/7_dpo/control.py resume     continue training where it paused
    python phases/7_dpo/control.py status     the dashboard once (--run prepare: the pairs being made)
    python phases/7_dpo/control.py watch      live dashboard that refreshes itself

Both steps use the GPU, so start them only when it is free. `prepare` can be paused and run
again: it skips the prompts already done. Arguments after `--` go to the script.
"""

from pathlib import Path

from leon.paths import CHECKPOINT_DIR, DATA_DIR
from leon.run import Control, RunFiles, launch, open_watch_window

PHASE = Path(__file__).resolve().parent
AGENT = CHECKPOINT_DIR / "6_agent" / "leon_0.1_agent" / "best.pt"
CHAT = CHECKPOINT_DIR / "5_sft" / "leon_0.1_chat" / "best.pt"


def check() -> None:
    if not (DATA_DIR / "dpo" / "train.jsonl").exists():
        raise SystemExit("No preference pairs yet. Make them with: python phases/7_dpo/control.py prepare")


def prepare(args) -> None:
    if not AGENT.exists() and not CHAT.exists():
        raise SystemExit("No chat model yet: phase 7 needs the phase 5 (or 6) model to sample answers.")
    run = RunFiles(CONTROL.runs_dir / "prepare")
    if pid := run.running_pid():
        raise SystemExit(f"Already running (pid {pid}). See: {CONTROL.command} status --run prepare")
    launch(run, PHASE / "prepare.py", ["--run", "prepare", *args.train_args], CONTROL.control)
    if not args.no_watch:
        open_watch_window(CONTROL.control, "prepare")


CONTROL = Control(
    title="Leon 0.1 · DPO (phase 7)",
    runs_dir=CHECKPOINT_DIR / "7_dpo",
    script=PHASE / "train.py",
    control=Path(__file__),
    default_run="leon_0.1_dpo",
    check=check,
    extra=[("prepare", prepare, "sample Leon's answers to build the preference pairs (GPU)")],
)

if __name__ == "__main__":
    CONTROL.main(__doc__)
