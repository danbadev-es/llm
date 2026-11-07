"""Start, pause, resume and watch the Leon 0.1 SFT run (phase 5).

    python phases/5_sft/control.py start      start SFT in the background
    python phases/5_sft/control.py pause      finish the current step, save and stop
    python phases/5_sft/control.py resume     continue where it paused
    python phases/5_sft/control.py status     the dashboard once: progress, loss, ETA, latest answers
    python phases/5_sft/control.py watch      live dashboard that refreshes itself

Like phase 4, the trainer is an independent process with no window: it keeps going if this
terminal, VS Code or the chat session is closed. `start` and `resume` also open a `watch`
window (skip it with --no-watch); closing that window does not stop the training.
Arguments after `--` are passed to train.py.

A run started before train.py wrote status.json is still shown: the dashboard then reads
the progress from the evaluations in train.log.
"""

import re
from datetime import datetime
from pathlib import Path

from leon.paths import CHECKPOINT_DIR, DATA_DIR
from leon.run import Control, RunFiles

EVAL_LINE = re.compile(r"^step\s+(\d+)/(\d+) \| val loss \(answers\) ([\d.]+) .*\| ([\d.]+) min$")
PROBE_LINE = re.compile(r"^    (.+?) → (['\"])(.*)\2$")


def status_from_log(run: RunFiles) -> dict:
    """Progress read from train.log's evaluations, for runs that do not write status.json.

    Evaluations happen every 100 steps, so the step, the loss and the ETA lag behind by up to
    one evaluation interval.
    """
    if not run.log.exists():
        return {}
    lines = run.log.read_text(encoding="utf-8", errors="replace").splitlines()
    evals = [(i, m) for i, line in enumerate(lines) if (m := EVAL_LINE.match(line))]
    if not evals:
        return {}
    index, match = evals[-1]
    step, max_steps, val_loss, minutes = int(match[1]), int(match[2]), float(match[3]), float(match[4])
    probes = []
    for line in lines[index + 1 : index + 1 + 3]:
        if probe := PROBE_LINE.match(line):
            probes.append([probe[1], probe[3].encode("latin-1", "backslashreplace").decode("unicode_escape")])
    finished = any(line.startswith("Best val loss (answers)") for line in lines[index:])
    status = {
        "state": "finished" if finished else "running",
        "step": step,
        "max_steps": max_steps,
        "val_loss": val_loss,
        "best_val": min(float(m[3]) for _, m in evals),
        "probes": probes,
        "updated_at": datetime.fromtimestamp(run.log.stat().st_mtime).strftime("%Y-%m-%d %H:%M:%S"),
        "from_log": True,
    }
    if step and not finished:
        status["eta_hours"] = (max_steps - step) * minutes / step / 60
    return status


def check_data() -> None:
    if not (DATA_DIR / "sft" / "train.npz").exists():
        raise SystemExit("No SFT data yet. Run: python phases/5_sft/prepare.py")


CONTROL = Control(
    title="Leon 0.1 · SFT (phase 5)",
    runs_dir=CHECKPOINT_DIR / "5_sft",
    script=Path(__file__).resolve().parent / "train.py",
    control=Path(__file__),
    default_run="leon_0.1_chat",
    check=check_data,
    status_reader=status_from_log,
)

if __name__ == "__main__":
    CONTROL.main(__doc__)
