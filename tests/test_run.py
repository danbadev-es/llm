"""Checks for leon.run, the control files of background training. Run: pytest tests/test_run.py"""

import os
import subprocess
import sys
import tempfile
from pathlib import Path

from leon.run import RunFiles, is_process_alive


def test_pause_flag_roundtrip() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        run = RunFiles(Path(tmp) / "run")
        assert not run.pause_requested()
        run.request_pause()  # also creates the run folder
        assert run.pause_requested()
        run.clear_pause()
        run.clear_pause()  # clearing twice is harmless
        assert not run.pause_requested()


def test_status_is_written_atomically_and_read_back() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        run = RunFiles(Path(tmp) / "run")
        assert run.read_status() == {}
        run.write_status(state="running", step=10, max_steps=100)
        status = run.read_status()
        assert status["state"] == "running" and status["step"] == 10 and "updated_at" in status
        assert not run.status.with_suffix(".tmp").exists()


def test_process_liveness() -> None:
    assert is_process_alive(os.getpid())
    finished = subprocess.run([sys.executable, "-c", "pass"])
    assert finished.returncode == 0
    child = subprocess.Popen([sys.executable, "-c", "pass"])
    child.wait()
    assert not is_process_alive(child.pid)


def test_running_pid_ignores_stale_pid_files() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        run = RunFiles(Path(tmp) / "run")
        assert run.running_pid() is None
        run.dir.mkdir()
        run.pid.write_text(str(os.getpid()))
        assert run.running_pid() == os.getpid()
        child = subprocess.Popen([sys.executable, "-c", "pass"])
        child.wait()
        run.pid.write_text(str(child.pid))  # the trainer died without cleaning up
        assert run.running_pid() is None


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"ok  {name}")
