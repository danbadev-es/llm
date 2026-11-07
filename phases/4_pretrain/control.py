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
import subprocess
import sys
from pathlib import Path

from leon.paths import CHECKPOINT_DIR, DATA_DIR, ROOT
from leon.run import (
    RunFiles,
    gpu_summary,
    launch,
    open_watch_window,
    print_log_tail,
    wait_for_pause,
    watch,
)

PHASE_DIR = Path(__file__).resolve().parent
TRAIN_SCRIPT = PHASE_DIR / "train.py"
CONTROL = Path(__file__).resolve()
CONTROL_CMD = "python phases/4_pretrain/control.py"
TITLE = "Leon 0.1 · pretraining (phase 4)"
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
    launch(run, TRAIN_SCRIPT, ["--run", args.run, *(["--overwrite"] if args.fresh else []), *args.train_args], CONTROL)
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
    launch(
        run,
        TRAIN_SCRIPT,
        ["--run", args.run, *(["--overwrite"] if args.fresh else []), *EXTEND_ARGS, *args.train_args],
        CONTROL,
    )
    if not args.no_watch:
        open_watch_window(CONTROL, args.run)


def cmd_pause(args: argparse.Namespace) -> None:
    run = RunFiles(RUNS_DIR / args.run)
    if args.no_wait:
        if (pid := run.running_pid()) is None:
            raise SystemExit("Not running: nothing to pause.")
        run.request_pause()
        print(f"Pause requested (pid {pid}): the current step finishes, last.pt is saved, then it exits.")
        return
    wait_for_pause(run, args.timeout)


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


def cmd_watch(args: argparse.Namespace) -> None:
    run = RunFiles(RUNS_DIR / args.run)
    watch(run, TITLE, CONTROL_CMD, run.read_status, args.interval)


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
