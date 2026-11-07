"""Train Leon 0.1 to reason step by step (phase 8), in the background.

    python phases/8_reasoning/control.py start                          SFT with steps (leon_0.1_steps)
    python phases/8_reasoning/control.py start --run leon_0.1_direct    SFT without steps (the control)
    python phases/8_reasoning/control.py grpo                           GRPO from leon_0.1_steps (continues if paused)
    python phases/8_reasoning/control.py pause [--run …]                save and stop
    python phases/8_reasoning/control.py resume [--run …]               continue an SFT run
    python phases/8_reasoning/control.py status [--run …]               the dashboard once
    python phases/8_reasoning/control.py watch [--run …]                live dashboard

All of them use the GPU. Data first (CPU): python phases/8_reasoning/prepare.py
Arguments after `--` go to the script.
"""

from pathlib import Path

from leon.paths import CHECKPOINT_DIR, DATA_DIR
from leon.run import Control, RunFiles, launch, open_watch_window

PHASE = Path(__file__).resolve().parent
GRPO_RUN = "leon_0.1_grpo"


def check() -> None:
    if not (DATA_DIR / "reasoning" / "steps" / "train.npz").exists():
        raise SystemExit("No reasoning data yet. Run: python phases/8_reasoning/prepare.py")


def grpo(args) -> None:
    if not (CONTROL.runs_dir / "leon_0.1_steps" / "best.pt").exists():
        raise SystemExit("GRPO starts from the step-by-step model: first run `control.py start`.")
    run = RunFiles(CONTROL.runs_dir / GRPO_RUN)
    if pid := run.running_pid():
        raise SystemExit(f"Already running (pid {pid}). See: {CONTROL.command} status --run {GRPO_RUN}")
    resume = ["--resume"] if run.last.exists() and not args.fresh else []
    launch(run, PHASE / "grpo.py", ["--run", GRPO_RUN, *resume, *args.train_args], CONTROL.control)
    if not args.no_watch:
        open_watch_window(CONTROL.control, GRPO_RUN)


CONTROL = Control(
    title="Leon 0.1 · reasoning (phase 8)",
    runs_dir=CHECKPOINT_DIR / "8_reasoning",
    script=PHASE / "train.py",
    control=Path(__file__),
    default_run="leon_0.1_steps",
    check=check,
    extra=[("grpo", grpo, "reinforcement learning (GRPO) from the step-by-step model; continues if paused")],
)

if __name__ == "__main__":
    CONTROL.main(__doc__)
