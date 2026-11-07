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

import argparse
import re
import sys
from datetime import datetime
from pathlib import Path

from leon.paths import CHECKPOINT_DIR, DATA_DIR
from leon.run import RunFiles, dashboard, launch, open_watch_window, wait_for_pause, watch

PHASE_DIR = Path(__file__).resolve().parent
TRAIN_SCRIPT = PHASE_DIR / "train.py"
CONTROL = Path(__file__).resolve()
CONTROL_CMD = "python phases/5_sft/control.py"
TITLE = "Leon 0.1 · SFT (phase 5)"
RUNS_DIR = CHECKPOINT_DIR / "5_sft"
DEFAULT_RUN = "leon_0.1_chat"
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


def read_status(run: RunFiles) -> dict:
    return run.read_status() or status_from_log(run)


def check_data() -> None:
    if not (DATA_DIR / "sft" / "train.npz").exists():
        raise SystemExit("No SFT data yet. Run: python phases/5_sft/prepare.py")


def cmd_start(args: argparse.Namespace) -> None:
    run = RunFiles(RUNS_DIR / args.run)
    if pid := run.running_pid():
        raise SystemExit(f"Already running (pid {pid}). See: control.py status")
    if run.last.exists() and not args.fresh:
        raise SystemExit(
            f"{run.last} already exists. Continue it with `control.py resume`, "
            "or start over (deleting its progress) with `control.py start --fresh`."
        )
    check_data()
    run.status.unlink(missing_ok=True)
    launch(run, TRAIN_SCRIPT, ["--run", args.run, *args.train_args], CONTROL)
    if not args.no_watch:
        open_watch_window(CONTROL, args.run)


def cmd_resume(args: argparse.Namespace) -> None:
    run = RunFiles(RUNS_DIR / args.run)
    if pid := run.running_pid():
        raise SystemExit(f"Already running (pid {pid}). See: control.py status")
    if not run.last.exists():
        raise SystemExit(f"Nothing to resume: {run.last} does not exist. Use `control.py start`.")
    launch(run, TRAIN_SCRIPT, ["--run", args.run, "--resume", *args.train_args], CONTROL)
    if not args.no_watch:
        open_watch_window(CONTROL, args.run)


def cmd_pause(args: argparse.Namespace) -> None:
    run = RunFiles(RUNS_DIR / args.run)
    if not run.read_status() and run.running_pid():
        raise SystemExit(
            "This run was started before train.py could pause cleanly. It saves last.pt at every "
            "evaluation (every 100 steps): stopping it loses at most the steps since then."
        )
    wait_for_pause(run, args.timeout)


def cmd_status(args: argparse.Namespace) -> None:
    run = RunFiles(RUNS_DIR / args.run)
    print(dashboard(run, TITLE, CONTROL_CMD, read_status(run)))


def cmd_watch(args: argparse.Namespace) -> None:
    run = RunFiles(RUNS_DIR / args.run)
    watch(run, TITLE, CONTROL_CMD, lambda: read_status(run), args.interval)


def main() -> None:
    argv = sys.argv[1:]
    train_args = argv[argv.index("--") + 1 :] if "--" in argv else []
    argv = argv[: argv.index("--")] if "--" in argv else argv

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    for name, fn, help_text in (
        ("start", cmd_start, "start SFT in the background"),
        ("pause", cmd_pause, "save and stop cleanly"),
        ("resume", cmd_resume, "continue a paused run"),
        ("status", cmd_status, "show progress"),
        ("watch", cmd_watch, "live dashboard (closing it does not stop training)"),
    ):
        command = sub.add_parser(name, help=help_text)
        command.set_defaults(fn=fn)
        command.add_argument("--run", default=DEFAULT_RUN)
    sub.choices["start"].add_argument("--fresh", action="store_true", help="discard an existing run's progress")
    for name in ("start", "resume"):
        sub.choices[name].add_argument("--no-watch", action="store_true", help="do not open the dashboard window")
    sub.choices["watch"].add_argument("--interval", type=float, default=5, help="seconds between refreshes")
    sub.choices["pause"].add_argument("--timeout", type=int, default=300)
    args = parser.parse_args(argv)
    args.train_args = train_args
    args.fn(args)


if __name__ == "__main__":
    main()
