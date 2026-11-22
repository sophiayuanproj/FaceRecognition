"""Shared advisory lock for all Nicla/FNIRSI hardware sessions."""

import fcntl
import json
import os
from pathlib import Path
import socket
import time
from typing import Optional


class HardwareLock:
    def __init__(self, lock_path: Path, run_id: str, timeout_s: float = 0.0):
        self.lock_path = Path(lock_path)
        self.run_id = run_id
        self.timeout_s = timeout_s
        self.handle = None

    def __enter__(self):
        self.lock_path.parent.mkdir(parents=True, exist_ok=True)
        self.handle = self.lock_path.open("a+", encoding="utf-8")
        deadline = time.monotonic() + self.timeout_s
        while True:
            try:
                fcntl.flock(self.handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except BlockingIOError:
                if time.monotonic() >= deadline:
                    self.handle.close()
                    self.handle = None
                    raise TimeoutError("hardware.lock is held by another session")
                time.sleep(min(0.1, max(0.0, deadline - time.monotonic())))
        self.write_metadata({
            "state": "acquired",
            "run_id": self.run_id,
            "pid": os.getpid(),
            "host": socket.gethostname(),
            "acquired_monotonic_ns": time.monotonic_ns(),
            "acquired_utc_ns": time.time_ns(),
        })
        return self

    def write_metadata(self, data):
        self.handle.seek(0)
        self.handle.truncate()
        self.handle.write(json.dumps(data, sort_keys=True) + "\n")
        self.handle.flush()
        os.fsync(self.handle.fileno())

    def __exit__(self, exc_type, exc, tb):
        if self.handle is not None:
            self.write_metadata({
                "state": "released",
                "run_id": self.run_id,
                "pid": os.getpid(),
                "host": socket.gethostname(),
                "released_monotonic_ns": time.monotonic_ns(),
                "released_utc_ns": time.time_ns(),
                "exit_error": None if exc is None else "%s: %s" % (exc_type.__name__, exc),
            })
            fcntl.flock(self.handle.fileno(), fcntl.LOCK_UN)
            self.handle.close()
            self.handle = None
        return False


def default_lock_path(project_root: Path) -> Path:
    return Path(project_root) / "work" / "research-plan-2026-10-22" / "hardware.lock"
