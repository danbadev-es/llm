"""Control files of a long training run, so it can be started, paused, resumed and watched
from another process (each phase's control.py) while it runs in the background.
`Control` is the command line those control.py files share.

All of them live in the run's checkpoint folder:

    train.pid      process id of the running trainer
    PAUSE          created by `control.py pause`; the trainer saves last.pt and exits cleanly
    status.json    progress, rewritten by the trainer every few steps
    train.log      everything the trainer prints
"""

import argparse
import json
import math
import os
import shutil
import subprocess
import sys
import textwrap
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path


class RunFiles:
    def __init__(self, run_dir: Path) -> None:
        self.dir = run_dir
        self.pid = run_dir / "train.pid"
        self.pause_flag = run_dir / "PAUSE"
        self.status = run_dir / "status.json"
        self.log = run_dir / "train.log"
        self.last = run_dir / "last.pt"
        self.best = run_dir / "best.pt"

    # --- pause request (written by the controller, read by the trainer) ---

    def request_pause(self) -> None:
        self.dir.mkdir(parents=True, exist_ok=True)
        self.pause_flag.touch()

    def pause_requested(self) -> bool:
        return self.pause_flag.exists()

    def clear_pause(self) -> None:
        self.pause_flag.unlink(missing_ok=True)

    # --- progress (written by the trainer, read by the controller) ---

    def write_status(self, **fields) -> None:
        """Replace status.json atomically, so a reader never sees a half-written file."""
        self.dir.mkdir(parents=True, exist_ok=True)
        fields["updated_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
        tmp = self.status.with_suffix(".tmp")
        tmp.write_text(json.dumps(fields, indent=2), encoding="utf-8")
        # On Windows the rename fails while a reader (the dashboard, `status`) has the file open.
        # Status is informative: retry briefly, then skip this update rather than stop the
        # training (the next update, a few seconds later, replaces it).
        for _ in range(20):
            try:
                os.replace(tmp, self.status)
                return
            except PermissionError:
                time.sleep(0.05)

    def read_status(self) -> dict:
        for _ in range(20):
            try:
                return json.loads(self.status.read_text(encoding="utf-8")) if self.status.exists() else {}
            except (PermissionError, json.JSONDecodeError):  # caught in the middle of a write
                time.sleep(0.05)
        return {}

    # --- process ---

    def running_pid(self) -> int | None:
        """The trainer's pid if it is alive, else None (a stale pid file is ignored)."""
        if not self.pid.exists():
            return None
        pid = int(self.pid.read_text().strip() or 0)
        return pid if pid and is_process_alive(pid) else None


def is_process_alive(pid: int) -> bool:
    if sys.platform == "win32":
        # os.kill(pid, 0) would *terminate* the process on Windows: ask the OS instead.
        import ctypes

        process_query_limited_information, still_active = 0x1000, 259
        kernel32 = ctypes.windll.kernel32
        handle = kernel32.OpenProcess(process_query_limited_information, False, pid)
        if not handle:
            return False
        exit_code = ctypes.c_ulong()
        kernel32.GetExitCodeProcess(handle, ctypes.byref(exit_code))
        kernel32.CloseHandle(handle)
        return exit_code.value == still_active
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    return True


def launch(run: RunFiles, script: Path, script_args: list[str], control: Path) -> None:
    """Start a trainer detached from this terminal, appending its output to the run's train.log."""
    from leon.paths import ROOT

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
        log.write(f"\n===== {time.strftime('%Y-%m-%d %H:%M:%S')} {' '.join(script_args)}\n")
        log.flush()
        process = subprocess.Popen(
            [sys.executable, str(script), *script_args],
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
        print(f"{script.name} exited right away (code {process.returncode}). Last lines of {run.log}:\n")
        print_log_tail(run, 15)
        raise SystemExit(1)
    print(f"Training running in the background (pid {process.pid}).")
    print(f"  log:    {run.log}")
    print(f"  status: python {control.relative_to(ROOT).as_posix()} status")


def open_watch_window(control: Path, run_name: str) -> None:
    """Open the control script's live dashboard in a terminal window of its own (Windows)."""
    from leon.paths import ROOT

    if sys.platform != "win32":
        print(f"  watch:  python {control.relative_to(ROOT).as_posix()} watch --run {run_name}")
        return
    subprocess.Popen(
        [sys.executable, str(control), "watch", "--run", run_name],
        cwd=ROOT,
        env={**os.environ, "PYTHONIOENCODING": "utf-8"},
        creationflags=subprocess.CREATE_NEW_CONSOLE,
    )
    print("  watch:  opened a dashboard window (closing it does not stop the training)")


def print_log_tail(run: RunFiles, lines: int) -> None:
    if run.log.exists():
        tail = run.log.read_text(encoding="utf-8", errors="replace").splitlines()[-lines:]
        print("\n".join(f"  {line}" for line in tail))


def wait_for_pause(run: RunFiles, timeout: int) -> None:
    """Request a pause and wait until the trainer has saved last.pt and exited."""
    pid = run.running_pid()
    if pid is None:
        raise SystemExit("Not running: nothing to pause.")
    run.request_pause()
    print(f"Pause requested (pid {pid}): the current step finishes, last.pt is saved, then it exits.")
    deadline = time.time() + timeout
    while run.running_pid() and time.time() < deadline:  # a step takes seconds; an eval up to ~1 min
        time.sleep(2)
    if run.running_pid():
        print(f"Still running after {timeout} s (probably evaluating). It will stop on its own.")
    else:
        status = run.read_status()
        print(f"Paused at step {status.get('step', '?'):,} of {status.get('max_steps', '?'):,}. Continue with: resume")


def gpu_summary() -> str:
    """Temperature, clock, utilization, memory and power of the GPU, from nvidia-smi."""
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


def finish_time(hours: float) -> str:
    """'today 23:40', 'tomorrow 12:15' or '08/11 03:00': when the remaining GPU hours end."""
    end = datetime.now() + timedelta(hours=hours)
    days = (end.date() - datetime.now().date()).days
    day = {0: "today", 1: "tomorrow"}.get(days, end.strftime("%d/%m"))
    return f"{day} {end:%H:%M}"


def progress_bar(fraction: float, width: int = 30) -> str:
    done = int(width * min(max(fraction, 0.0), 1.0))
    return f"[{'█' * done}{'░' * (width - done)}]"


def _wrap(text: str, width: int, indent: str, max_lines: int) -> list[str]:
    return textwrap.wrap(
        text.replace("\n", " ⏎ "),
        width - len(indent),
        initial_indent=indent,
        subsequent_indent=indent,
        max_lines=max_lines,
        placeholder=" …",
    )


def dashboard(run: RunFiles, title: str, control: str, status: dict) -> str:
    """The live dashboard of any phase: it shows whichever fields the trainer's status.json has.

    Common fields: state, step, max_steps, tokens_seen, train_loss, val_loss, best_val, tok_per_s,
    eta_hours, updated_at. Optional: val_bits_per_byte, val_loss_first_half/second_half (context
    extension), sample (a text continuation), probes ([question, answer] pairs), metrics ({name:
    value}, e.g. DPO's reward accuracy) and perplexity (False: the loss is not a language-model one).
    """
    pid = run.running_pid()
    width = max(60, min(shutil.get_terminal_size().columns, 110))
    rule = "─" * width
    state = status.get("state")
    if pid:
        label = f"● training (pid {pid})"
    elif state == "paused":
        label = f"⏸ paused · continue with: {control} resume"
    elif state == "finished":
        label = "✓ finished"
    elif status:
        label = "✕ stopped unexpectedly (see the log)"
    else:
        label = f"not started · start it with: {control} start"
    lines = [f" {title}   run: {run.dir.name}", rule, f" State       {label}"]
    if status:
        step, max_steps = status["step"], status["max_steps"]
        lag = "  (at the latest evaluation)" if status.get("from_log") else ""
        lines.append(
            f" Progress    {progress_bar(step / max_steps)} {100 * step / max_steps:5.1f} %   "
            f"step {step:,} / {max_steps:,}{lag}"
        )
        if tokens := status.get("tokens_seen"):
            total = max_steps * tokens / max(step, 1)
            unit, scale = ("B", 1e9) if total >= 1e9 else ("M", 1e6)
            digits = 2 if unit == "B" else 0
            lines.append(f" Tokens      {tokens / scale:,.{digits}f}{unit} of {total / scale:,.{digits}f}{unit}")
        losses = []
        if status.get("train_loss") is not None:
            losses.append(f"train {status['train_loss']:.3f}")
        if status.get("val_loss") is not None:
            if status.get("val_bits_per_byte") is not None:
                detail = f" ({status['val_bits_per_byte']:.3f} bits/byte)"
            elif status.get("perplexity", True):  # a language-model loss; DPO's is not one
                detail = f" (ppl {math.exp(min(status['val_loss'], 50)):.1f})"
            else:
                detail = ""
            losses.append(f"val {status['val_loss']:.3f}{detail}")
        if status.get("best_val") is not None:
            losses.append(f"best {status['best_val']:.3f}")
        if losses:
            lines.append(f" Loss        {' · '.join(losses)}")
        if metrics := status.get("metrics"):  # other numbers a phase tracks: {"reward accuracy": 0.71}
            shown = [
                f"{name} {value:.3f}" if isinstance(value, float) else f"{name} {value}"
                for name, value in metrics.items()
            ]
            lines.append(f" Metrics     {' · '.join(shown)}")
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
            lines += _wrap(status["sample"], width, "   ", 4)
        if status.get("probes"):
            lines += ["", " Latest answers (renewed at every evaluation):"]
            for question, reply in status["probes"]:
                lines.append(f"   » {question}")
                lines += _wrap(reply, width, "     ", 2)
    if run.log.exists():
        # Only the unindented lines (evaluations, start, pause, end): the indented ones are the
        # samples, probe answers and long-text losses, already shown above.
        text = run.log.read_text(encoding="utf-8", errors="replace").splitlines()
        log = [line for line in text if line.strip() and not line[0].isspace()]
        lines += ["", " Log:"] + [f"   {line[: width - 4]}" for line in log[-5:]]
    lines += [rule, " Closing this window does NOT stop the training.", f" Pause: {control} pause"]
    return "\n".join(lines)


def watch(run: RunFiles, title: str, control: str, read_status, interval: float) -> None:
    """Redraw the dashboard every few seconds until the run stops (Ctrl+C quits the dashboard only)."""
    if sys.platform == "win32":
        import ctypes

        ctypes.windll.kernel32.SetConsoleTitleW(f"{title} ({run.dir.name})")
        os.system("")  # enables ANSI escape codes in the classic Windows console
    try:
        while True:
            status = read_status()
            print("\x1b[2J\x1b[H" + dashboard(run, title, control, status), flush=True)  # clear, cursor home
            if run.running_pid() is None and status.get("state") in ("paused", "finished"):
                break
            time.sleep(interval)
        input("\n Training is not running. Press Enter to close this window.")
    except (KeyboardInterrupt, EOFError):
        pass


def keep_system_awake() -> None:
    """Ask Windows not to sleep while this process lives (released automatically on exit).

    It does not override closing the laptop lid if the power settings say "sleep" for that.
    """
    if sys.platform == "win32":
        import ctypes

        es_continuous, es_system_required = 0x80000000, 0x00000001
        ctypes.windll.kernel32.SetThreadExecutionState(es_continuous | es_system_required)


@dataclass
class Control:
    """The command line of a phase's control.py: start, pause, resume, status and watch.

    A phase only describes its run (title, trainer script, where checkpoints go) and, optionally,
    a data check before start, a fallback status reader and extra commands (such as `prepare`):

        Control("Leon 0.1 · SFT (phase 5)", RUNS_DIR, TRAIN_SCRIPT, Path(__file__), "leon_0.1_chat").main(__doc__)

    Arguments after `--` are passed to the trainer.
    """

    title: str
    runs_dir: Path
    script: Path
    control: Path
    default_run: str
    check: Callable[[], None] | None = None
    status_reader: Callable[[RunFiles], dict] | None = None
    extra: list[tuple[str, Callable[[argparse.Namespace], None], str]] = field(default_factory=list)

    @property
    def command(self) -> str:
        from leon.paths import ROOT

        return f"python {self.control.resolve().relative_to(ROOT).as_posix()}"

    def run_files(self, args: argparse.Namespace) -> RunFiles:
        return RunFiles(self.runs_dir / args.run)

    def read_status(self, run: RunFiles) -> dict:
        return run.read_status() or (self.status_reader(run) if self.status_reader else {})

    def start(self, args: argparse.Namespace) -> None:
        run = self.run_files(args)
        if pid := run.running_pid():
            raise SystemExit(f"Already running (pid {pid}). See: {self.command} status")
        if run.last.exists() and not args.fresh:
            raise SystemExit(
                f"{run.last} already exists. Continue it with `{self.command} resume`, "
                f"or start over (deleting its progress) with `{self.command} start --fresh`."
            )
        if self.check:
            self.check()
        run.status.unlink(missing_ok=True)
        launch(run, self.script, ["--run", args.run, *args.train_args], self.control)
        if not args.no_watch:
            open_watch_window(self.control, args.run)

    def resume(self, args: argparse.Namespace) -> None:
        run = self.run_files(args)
        if pid := run.running_pid():
            raise SystemExit(f"Already running (pid {pid}). See: {self.command} status")
        if not run.last.exists():
            raise SystemExit(f"Nothing to resume: {run.last} does not exist. Use `{self.command} start`.")
        launch(run, self.script, ["--run", args.run, "--resume", *args.train_args], self.control)
        if not args.no_watch:
            open_watch_window(self.control, args.run)

    def pause(self, args: argparse.Namespace) -> None:
        run = self.run_files(args)
        if not run.read_status() and run.running_pid():
            raise SystemExit(
                "This run was started before its trainer could pause cleanly. It saves last.pt at every "
                "evaluation: stopping it loses at most the steps since then."
            )
        wait_for_pause(run, args.timeout)

    def status(self, args: argparse.Namespace) -> None:
        run = self.run_files(args)
        print(dashboard(run, self.title, self.command, self.read_status(run)))

    def watch(self, args: argparse.Namespace) -> None:
        run = self.run_files(args)
        watch(run, self.title, self.command, lambda: self.read_status(run), args.interval)

    def main(self, description: str, argv: list[str] | None = None) -> None:
        argv = sys.argv[1:] if argv is None else argv
        train_args = argv[argv.index("--") + 1 :] if "--" in argv else []
        argv = argv[: argv.index("--")] if "--" in argv else argv

        parser = argparse.ArgumentParser(description=description, formatter_class=argparse.RawDescriptionHelpFormatter)
        sub = parser.add_subparsers(dest="command", required=True)
        commands = [
            ("start", self.start, "start training in the background"),
            ("pause", self.pause, "save and stop cleanly"),
            ("resume", self.resume, "continue a paused run"),
            ("status", self.status, "show the dashboard once"),
            ("watch", self.watch, "live dashboard (closing it does not stop training)"),
            *self.extra,
        ]
        for name, fn, help_text in commands:
            command = sub.add_parser(name, help=help_text)
            command.set_defaults(fn=fn)
            command.add_argument("--run", default=self.default_run)
            command.add_argument("--fresh", action="store_true", help="discard an existing run's progress")
            command.add_argument("--no-watch", action="store_true", help="do not open the dashboard window")
            command.add_argument("--interval", type=float, default=5, help="watch: seconds between refreshes")
            command.add_argument("--timeout", type=int, default=300, help="pause: seconds to wait for the save")
        args = parser.parse_args(argv)
        args.train_args = train_args
        args.fn(args)
