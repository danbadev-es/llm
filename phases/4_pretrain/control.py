"""Start, pause, resume and watch the Leon 0.1 pretraining run.

    python phases/4_pretrain/control.py prepare    download all of Wikipedia and tokenize it (once)
    python phases/4_pretrain/control.py start      start training in the background
    python phases/4_pretrain/control.py pause      finish the current step, save and stop
    python phases/4_pretrain/control.py resume     continue exactly where it paused
    python phases/4_pretrain/control.py status     progress, losses, ETA, GPU temperature

The trainer runs as an independent process: it keeps going if this terminal, VS Code or the
chat session is closed. Arguments after `--` are passed to train.py (e.g. a tiny smoke test:
`control.py start --run smoke -- --dataset quijote --device cpu --n-layer 1 ...`).
"""

import argparse
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

from leon.paths import CHECKPOINT_DIR, DATA_DIR, ROOT
from leon.run import RunFiles

PHASE_DIR = Path(__file__).resolve().parent
TRAIN_SCRIPT = PHASE_DIR / "train.py"
RUNS_DIR = CHECKPOINT_DIR / "4_pretrain"
WIKI_SHARDS = range(13)
FULL_WIKI_TOKENS = 1_000_000_000  # below this, data/wiki_es holds only part of Wikipedia


def launch(run: RunFiles, train_args: list[str]) -> None:
    """Start train.py detached from this terminal, appending its output to train.log."""
    run.dir.mkdir(parents=True, exist_ok=True)
    env = {**os.environ, "PYTHONUNBUFFERED": "1", "PYTHONIOENCODING": "utf-8"}
    flags = 0
    if sys.platform == "win32":  # own process group, no console: survives closing this terminal
        flags = subprocess.DETACHED_PROCESS | subprocess.CREATE_NEW_PROCESS_GROUP
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


def cmd_resume(args: argparse.Namespace) -> None:
    run = RunFiles(RUNS_DIR / args.run)
    if pid := run.running_pid():
        raise SystemExit(f"Already running (pid {pid}). See: control.py status")
    if not run.last.exists():
        raise SystemExit(f"Nothing to resume: {run.last} does not exist. Use `control.py start`.")
    launch(run, ["--run", args.run, "--resume", *args.train_args])


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
    ):
        command = sub.add_parser(name, help=help_text)
        command.set_defaults(fn=fn)
        command.add_argument("--run", default="leon_0.1")
    sub.choices["start"].add_argument("--fresh", action="store_true", help="discard an existing run's progress")
    sub.choices["pause"].add_argument("--no-wait", action="store_true", help="return without waiting for the save")
    sub.choices["pause"].add_argument("--timeout", type=int, default=180)
    sub.choices["status"].add_argument("--lines", type=int, default=5, help="log lines to show")
    args = parser.parse_args(argv)
    args.train_args = train_args
    args.fn(args)


if __name__ == "__main__":
    main()
