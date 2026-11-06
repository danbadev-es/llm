"""Start, pause, resume and watch the Leon 0.1 pretraining run.

    python phases/4_pretrain/control.py prepare    download all of Wikipedia and tokenize it (once)
    python phases/4_pretrain/control.py start      start training in the background
    python phases/4_pretrain/control.py pause      finish the current step, save and stop
    python phases/4_pretrain/control.py resume     continue exactly where it paused
    python phases/4_pretrain/control.py status     progress, losses, ETA, GPU temperature
    python phases/4_pretrain/control.py watch      live dashboard that refreshes itself
    python phases/4_pretrain/control.py extend     after `start` finishes: extend the context to 2048

pause / resume / status / watch act on the latest stage that has started (leon_0.1, then
leon_0.1_ctx2048 once `extend` has run); `--run` picks another one.

The trainer runs as an independent process with no window: it keeps going if this terminal,
VS Code or the chat session is closed. `start` and `resume` also open a `watch` window
(skip it with --no-watch); closing that window does not stop the training.
Arguments after `--` are passed to train.py (e.g. a tiny smoke test:
`control.py start --run smoke -- --dataset quijote --device cpu --n-layer 1 ...`).
"""

import argparse
import os
import shutil
import subprocess
import sys
import textwrap
import time
from datetime import datetime, timedelta
from pathlib import Path

from leon.paths import CHECKPOINT_DIR, DATA_DIR, ROOT
from leon.run import RunFiles

PHASE_DIR = Path(__file__).resolve().parent
TRAIN_SCRIPT = PHASE_DIR / "train.py"
RUNS_DIR = CHECKPOINT_DIR / "4_pretrain"
WIKI_SHARDS = range(13)
FULL_WIKI_TOKENS = 1_000_000_000  # below this, data/wiki_es holds only part of Wikipedia
BASE_RUN = "leon_0.1"
EXTEND_RUN = "leon_0.1_ctx2048"
# Context extension 1024 -> 2048: 1,150 steps x 131,072 tokens = 151M tokens (~1.5 h of GPU).
# The base model's weights, a 4x larger RoPE base so that 2048 positions rotate like ~1024 did
# before, half of every batch from inside articles longer than 2048 tokens (137k of them), and a
# short schedule that starts below the base run's peak learning rate so it refines, not relearns.
EXTEND_ARGS = [
    "--init-from", str(RUNS_DIR / BASE_RUN / "best.pt"),
    "--block-size", "2048", "--rope-theta", "50000", "--long-fraction", "0.5",
    "--batch-size", "2", "--grad-accum", "32", "--max-steps", "1150",
    "--lr", "3e-4", "--min-lr", "3e-5", "--warmup-steps", "100", "--eval-interval", "100",
]  # fmt: skip


def latest_run() -> str:
    """The run that pause / resume / status / watch act on by default: the latest started stage."""
    return EXTEND_RUN if (RUNS_DIR / EXTEND_RUN / "status.json").exists() else BASE_RUN


def launch(run: RunFiles, train_args: list[str]) -> None:
    """Start train.py detached from this terminal, appending its output to train.log."""
    run.dir.mkdir(parents=True, exist_ok=True)
    env = {**os.environ, "PYTHONUNBUFFERED": "1", "PYTHONIOENCODING": "utf-8"}
    flags = 0
    if sys.platform == "win32":
        # A hidden console of its own (no window that could be closed by mistake) and its own
        # process group (Ctrl+C here does not reach it): it survives closing this terminal.
        # DETACHED_PROCESS is not enough: the venv's python.exe is a launcher, and its child,
        # the real interpreter, would open a new, empty console window.
        flags = subprocess.CREATE_NO_WINDOW | subprocess.CREATE_NEW_PROCESS_GROUP
    with open(run.log, "a", encoding="utf-8") as log:
        log.write(f"\n===== {time.strftime('%Y-%m-%d %H:%M:%S')} {' '.join(train_args)}\n")
        log.flush()
        process = subprocess.Popen(
            [sys.executable, str(TRAIN_SCRIPT), *train_args],
            stdout=log,
            stderr=subprocess.STDOUT,
            stdin=subprocess.DEVNULL,
            cwd=ROOT,
            env=env,
            creationflags=flags,
            start_new_session=sys.platform != "win32",
        )
    run.pid.write_text(str(process.pid))
    time.sleep(5)  # catch immediate failures (missing data, bad arguments)
    if process.poll() is not None:
        print(f"train.py exited right away (code {process.returncode}). Last lines of {run.log}:\n")
        print_log_tail(run, 15)
        raise SystemExit(1)
    print(f"Training running in the background (pid {process.pid}).")
    print(f"  log:    {run.log}")
    print("  status: python phases/4_pretrain/control.py status")


def open_watch_window(run_name: str) -> None:
    """Open the live dashboard in a terminal window of its own (Windows)."""
    if sys.platform != "win32":
        print(f"  watch:  python phases/4_pretrain/control.py watch --run {run_name}")
        return
    subprocess.Popen(
        [sys.executable, str(Path(__file__).resolve()), "watch", "--run", run_name],
        cwd=ROOT,
        env={**os.environ, "PYTHONIOENCODING": "utf-8"},
        creationflags=subprocess.CREATE_NEW_CONSOLE,
    )
    print("  watch:  opened a dashboard window (closing it does not stop the training)")


def print_log_tail(run: RunFiles, lines: int) -> None:
    if run.log.exists():
        tail = run.log.read_text(encoding="utf-8", errors="replace").splitlines()[-lines:]
        print("\n".join(f"  {line}" for line in tail))


def gpu_summary() -> str:
    if not shutil.which("nvidia-smi"):
        return "n/a"
    query = "temperature.gpu,clocks.sm,utilization.gpu,memory.used,power.draw"
    result = subprocess.run(
        ["nvidia-smi", f"--query-gpu={query}", "--format=csv,noheader,nounits"], capture_output=True, text=True
    )
    if result.returncode != 0:
        return "n/a"
    temp, clock, util, memory, power = [value.strip() for value in result.stdout.splitlines()[0].split(",")]
    return f"{temp} °C · {clock} MHz · {util} % · {memory} MiB · {power} W"


def check_data() -> None:
    train_bin = DATA_DIR / "wiki_es" / "train.bin"
    if not train_bin.exists():
        raise SystemExit("No tokenized Wikipedia yet. Run: python phases/4_pretrain/control.py prepare")
    tokens = train_bin.stat().st_size // 2
    if tokens < FULL_WIKI_TOKENS:
        print(
            f"Warning: data/wiki_es has only {tokens / 1e6:,.0f}M tokens (part of Wikipedia). "
            "For Leon 0.1 run `control.py prepare` first; training on this would repeat it many times."
        )


def cmd_prepare(_: argparse.Namespace) -> None:
    """Download the 13 Wikipedia shards and tokenize them (CPU and network, not GPU)."""
    tokenizer_dir = ROOT / "phases" / "2_tokenizer"
    shards = [str(i) for i in WIKI_SHARDS]
    steps = [
        [sys.executable, str(tokenizer_dir / "prepare.py"), "--shards", *shards],
        [sys.executable, str(tokenizer_dir / "encode.py"), "--corpora", "wiki_val", "wiki_train"],
    ]
    for command in steps:
        print(f"\n$ {' '.join(command)}", flush=True)
        subprocess.run(command, check=True, cwd=ROOT, env={**os.environ, "PYTHONIOENCODING": "utf-8"})


def cmd_start(args: argparse.Namespace) -> None:
    run = RunFiles(RUNS_DIR / args.run)
    if pid := run.running_pid():
        raise SystemExit(f"Already running (pid {pid}). See: control.py status")
    if run.last.exists() and not args.fresh:
        raise SystemExit(
            f"{run.last} already exists. Continue it with `control.py resume`, "
            "or start over (deleting its progress) with `control.py start --fresh`."
        )
    if "--dataset" not in args.train_args:
        check_data()
    launch(run, ["--run", args.run, *(["--overwrite"] if args.fresh else []), *args.train_args])
    if not args.no_watch:
        open_watch_window(args.run)


def cmd_resume(args: argparse.Namespace) -> None:
    run = RunFiles(RUNS_DIR / args.run)
    if pid := run.running_pid():
        raise SystemExit(f"Already running (pid {pid}). See: control.py status")
    if not run.last.exists():
        raise SystemExit(f"Nothing to resume: {run.last} does not exist. Use `control.py start`.")
    launch(run, ["--run", args.run, "--resume", *args.train_args])
    if not args.no_watch:
        open_watch_window(args.run)


def cmd_extend(args: argparse.Namespace) -> None:
    """Start the context extension from the finished base run."""
    base = RunFiles(RUNS_DIR / BASE_RUN)
    if base.running_pid() or base.read_status().get("state") != "finished":
        raise SystemExit(f"The base run '{BASE_RUN}' has not finished yet: the extension starts from its best.pt.")
    run = RunFiles(RUNS_DIR / args.run)
    if pid := run.running_pid():
        raise SystemExit(f"Already running (pid {pid}). See: control.py status")
    if run.last.exists() and not args.fresh:
        raise SystemExit(f"{run.last} already exists. Continue it with `control.py resume`.")
    launch(run, ["--run", args.run, *(["--overwrite"] if args.fresh else []), *EXTEND_ARGS, *args.train_args])
    if not args.no_watch:
        open_watch_window(args.run)


def cmd_pause(args: argparse.Namespace) -> None:
    run = RunFiles(RUNS_DIR / args.run)
    pid = run.running_pid()
    if pid is None:
        raise SystemExit("Not running: nothing to pause.")
    run.request_pause()
    print(f"Pause requested (pid {pid}): the current step finishes, last.pt is saved, then it exits.")
    if args.no_wait:
        return
    deadline = time.time() + args.timeout
    while run.running_pid() and time.time() < deadline:  # a step takes seconds; an eval up to ~1 min
        time.sleep(2)
    if run.running_pid():
        print(f"Still running after {args.timeout} s (probably evaluating). It will stop on its own.")
    else:
        status = run.read_status()
        print(f"Paused at step {status.get('step', '?'):,} of {status.get('max_steps', '?'):,}. Continue with: resume")


def cmd_status(args: argparse.Namespace) -> None:
    run = RunFiles(RUNS_DIR / args.run)
    status = run.read_status()
    pid = run.running_pid()
    if not status and pid is None:
        print(f"No run '{args.run}' yet. Start it with: control.py start")
        return
    state = "running" if pid else status.get("state", "stopped")
    if not pid and state == "running":
        state = "stopped unexpectedly (see the log)"
    print(f"Run '{args.run}': {state}" + (f" (pid {pid})" if pid else ""))
    if status:
        step, max_steps = status["step"], status["max_steps"]
        print(f"  step       {step:,} / {max_steps:,}  ({100 * step / max_steps:.1f} %)")
        tokens = status["tokens_seen"]
        print(f"  tokens     {tokens / 1e9:.3f}B" if tokens >= 1e8 else f"  tokens     {tokens / 1e6:.1f}M")
        if status.get("train_loss") is not None:
            print(f"  train loss {status['train_loss']:.4f} (last 10 steps)")
        if status.get("val_loss") is not None:
            print(f"  val loss   {status['val_loss']:.4f} = {status['val_bits_per_byte']:.3f} bits/byte")
        if status.get("best_val") is not None:
            print(f"  best val   {status['best_val']:.4f}")
        if status.get("val_loss_second_half") is not None:
            print(
                f"  long text  loss {status['val_loss_first_half']:.4f} on the first half of the context, "
                f"{status['val_loss_second_half']:.4f} on the second"
            )
        if status.get("tok_per_s"):
            print(f"  speed      {status['tok_per_s']:,} tok/s")
        if status.get("eta_hours") is not None and state == "running":
            print(f"  remaining  {status['eta_hours']:.1f} h of GPU")
        print(f"  updated    {status['updated_at']}")
        if status.get("sample"):
            print(f"  sample     {status['sample']!r}")
    if pid:
        print(f"  GPU        {gpu_summary()}")
    print("  log tail:")
    print_log_tail(run, args.lines)


def finish_time(hours: float) -> str:
    """'today 23:40', 'tomorrow 12:15' or '08/11 03:00': when the remaining GPU hours end."""
    end = datetime.now() + timedelta(hours=hours)
    days = (end.date() - datetime.now().date()).days
    day = {0: "today", 1: "tomorrow"}.get(days, end.strftime("%d/%m"))
    return f"{day} {end:%H:%M}"


def dashboard(run: RunFiles, run_name: str) -> str:
    """The text of the live dashboard."""
    status, pid = run.read_status(), run.running_pid()
    width = max(60, min(shutil.get_terminal_size().columns, 110))
    rule = "─" * width
    state = status.get("state")
    if pid:
        label = f"● training (pid {pid})"
    elif state == "paused":
        label = "⏸ paused · continue with: control.py resume"
    elif state == "finished":
        label = "✓ finished"
    elif status:
        label = "✕ stopped unexpectedly (see the log)"
    else:
        label = "not started · start it with: control.py start"
    lines = [f" Leon 0.1 · pretraining (phase 4)   run: {run_name}", rule, f" State       {label}"]
    if status:
        step, max_steps = status["step"], status["max_steps"]
        done = int(30 * step / max_steps)
        percent = 100 * step / max_steps
        lines.append(f" Progress    [{'█' * done}{'░' * (30 - done)}] {percent:5.1f} %   step {step:,} / {max_steps:,}")
        tokens = status["tokens_seen"]
        total = max_steps * tokens / max(step, 1)
        lines.append(f" Tokens      {tokens / 1e9:.3f}B of {total / 1e9:.2f}B")
        losses = []
        if status.get("train_loss") is not None:
            losses.append(f"train {status['train_loss']:.3f}")
        if status.get("val_loss") is not None:
            losses.append(f"val {status['val_loss']:.3f} ({status['val_bits_per_byte']:.3f} bits/byte)")
        if status.get("best_val") is not None:
            losses.append(f"best {status['best_val']:.3f}")
        if losses:
            lines.append(f" Loss        {' · '.join(losses)}")
        if status.get("val_loss_second_half") is not None:
            lines.append(
                f" Long text   first half {status['val_loss_first_half']:.3f} · "
                f"second half {status['val_loss_second_half']:.3f} (lower = uses the longer context)"
            )
        if status.get("tok_per_s"):
            lines.append(f" Speed       {status['tok_per_s']:,} tok/s")
        if pid and status.get("eta_hours") is not None:
            eta = status["eta_hours"]
            lines.append(f" Remaining   {eta:.1f} h  → finishes {finish_time(eta)} if not paused")
        if pid:
            lines.append(f" GPU         {gpu_summary()}")
        lines.append(f" Updated     {status['updated_at']}")
        if status.get("sample"):
            lines += ["", " Latest sample (renewed at every evaluation):"]
            lines += textwrap.wrap(
                status["sample"].replace("\n", " ⏎ "),
                width - 4,
                initial_indent="   ",
                subsequent_indent="   ",
                max_lines=4,
                placeholder=" …",
            )
    if run.log.exists():
        log = [line for line in run.log.read_text(encoding="utf-8", errors="replace").splitlines() if line.strip()]
        lines += ["", " Log:"] + [f"   {line[: width - 4]}" for line in log[-5:]]
    lines += [
        rule,
        " Closing this window does NOT stop the training.",
        " Pause: python phases/4_pretrain/control.py pause",
    ]
    return "\n".join(lines)


def cmd_watch(args: argparse.Namespace) -> None:
    """Redraw the dashboard every few seconds until the run stops (Ctrl+C quits the dashboard only)."""
    run = RunFiles(RUNS_DIR / args.run)
    if sys.platform == "win32":
        import ctypes

        ctypes.windll.kernel32.SetConsoleTitleW(f"Leon 0.1 · pretraining ({args.run})")
        os.system("")  # enables ANSI escape codes in the classic Windows console
    try:
        while True:
            print("\x1b[2J\x1b[H" + dashboard(run, args.run), flush=True)  # clear the screen, cursor home
            if run.running_pid() is None and run.read_status().get("state") in ("paused", "finished"):
                break
            time.sleep(args.interval)
        input("\n Training is not running. Press Enter to close this window.")
    except (KeyboardInterrupt, EOFError):
        pass


def main() -> None:
    argv = sys.argv[1:]
    train_args = argv[argv.index("--") + 1 :] if "--" in argv else []
    argv = argv[: argv.index("--")] if "--" in argv else argv

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    for name, fn, help_text in (
        ("prepare", cmd_prepare, "download and tokenize all of Wikipedia (once)"),
        ("start", cmd_start, "start training in the background"),
        ("pause", cmd_pause, "save and stop cleanly"),
        ("resume", cmd_resume, "continue a paused run"),
        ("status", cmd_status, "show progress"),
        ("watch", cmd_watch, "live dashboard (closing it does not stop training)"),
        ("extend", cmd_extend, "extend the context to 2048 once the base run has finished"),
    ):
        command = sub.add_parser(name, help=help_text)
        command.set_defaults(fn=fn)
        default = {"start": BASE_RUN, "prepare": BASE_RUN, "extend": EXTEND_RUN}.get(name)
        command.add_argument("--run", default=default, help="default: " + (default or "the latest started stage"))
    for name in ("start", "extend"):
        sub.choices[name].add_argument("--fresh", action="store_true", help="discard an existing run's progress")
    for name in ("start", "resume", "extend"):
        sub.choices[name].add_argument("--no-watch", action="store_true", help="do not open the dashboard window")
    sub.choices["watch"].add_argument("--interval", type=float, default=5, help="seconds between refreshes")
    sub.choices["pause"].add_argument("--no-wait", action="store_true", help="return without waiting for the save")
    sub.choices["pause"].add_argument("--timeout", type=int, default=180)
    sub.choices["status"].add_argument("--lines", type=int, default=5, help="log lines to show")
    args = parser.parse_args(argv)
    args.run = args.run or latest_run()
    args.train_args = train_args
    args.fn(args)


if __name__ == "__main__":
    main()
