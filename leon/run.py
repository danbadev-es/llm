"""Control files of a long training run, so it can be started, paused, resumed and watched
from another process (phases/4_pretrain/control.py) while it runs in the background.

All of them live in the run's checkpoint folder:

    train.pid      process id of the running trainer
    PAUSE          created by `control.py pause`; the trainer saves last.pt and exits cleanly
    status.json    progress, rewritten by the trainer every few steps
    train.log      everything the trainer prints
"""

import json
import os
import sys
import time
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
        os.replace(tmp, self.status)

    def read_status(self) -> dict:
        return json.loads(self.status.read_text(encoding="utf-8")) if self.status.exists() else {}

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


def keep_system_awake() -> None:
    """Ask Windows not to sleep while this process lives (released automatically on exit).

    It does not override closing the laptop lid if the power settings say "sleep" for that.
    """
    if sys.platform == "win32":
        import ctypes

        es_continuous, es_system_required = 0x80000000, 0x00000001
        ctypes.windll.kernel32.SetThreadExecutionState(es_continuous | es_system_required)
