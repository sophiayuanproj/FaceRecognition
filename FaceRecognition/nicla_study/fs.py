"""Crash-safe local state and cross-process leases for macOS and Windows."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import tempfile
import time
from typing import Any


def sha256_file(path: Path | str) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_json_bytes(value: Any) -> bytes:
    return (json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False, separators=(",", ":")) + "\n").encode("utf-8")


def atomic_write_bytes(path: Path | str, payload: bytes, *, failpoint: str | None = None) -> None:
    """Replace one file atomically; a failed write leaves the previous file intact."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_name = tempfile.mkstemp(prefix="." + target.name + ".", suffix=".partial", dir=str(target.parent))
    temp = Path(temp_name)
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        if failpoint == "before_replace":
            raise OSError("injected interruption before atomic replace")
        os.replace(temp, target)
        try:
            dir_fd = os.open(str(target.parent), os.O_RDONLY)
            try:
                os.fsync(dir_fd)
            finally:
                os.close(dir_fd)
        except OSError:
            # Directory fsync is not available on every filesystem.
            pass
    except BaseException:
        try:
            temp.unlink()
        except FileNotFoundError:
            pass
        raise


def atomic_write_json(path: Path | str, value: Any, *, failpoint: str | None = None) -> None:
    atomic_write_bytes(path, (json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n").encode("utf-8"),
                       failpoint=failpoint)


class FileLock:
    """Bounded cross-process exclusive lock. Uses flock on POSIX and msvcrt on Windows."""
    def __init__(self, path: Path | str, timeout_s: float = 0.0, poll_s: float = 0.025):
        self.path = Path(path)
        self.timeout_s = float(timeout_s)
        self.poll_s = float(poll_s)
        if self.timeout_s < 0 or self.poll_s <= 0:
            raise ValueError("lock timeout must be nonnegative and polling interval positive")
        self._stream = None
        self.backend = None

    def acquire(self) -> "FileLock":
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._stream = self.path.open("a+b")
        if os.name == "nt":
            import msvcrt
            self.backend = "msvcrt.locking"
            self._stream.seek(0, os.SEEK_END)
            if self._stream.tell() == 0:
                self._stream.write(b"0")
                self._stream.flush()
            deadline = time.monotonic() + self.timeout_s
            while True:
                try:
                    self._stream.seek(0)
                    msvcrt.locking(self._stream.fileno(), msvcrt.LK_NBLCK, 1)
                    return self
                except OSError:
                    if time.monotonic() >= deadline:
                        self._stream.close()
                        self._stream = None
                        raise TimeoutError("lock acquisition timed out: " + str(self.path))
                    time.sleep(self.poll_s)
        else:
            import fcntl
            self.backend = "fcntl.flock"
            deadline = time.monotonic() + self.timeout_s
            while True:
                try:
                    fcntl.flock(self._stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                    return self
                except BlockingIOError:
                    if time.monotonic() >= deadline:
                        self._stream.close()
                        self._stream = None
                        raise TimeoutError("lock acquisition timed out: " + str(self.path))
                    time.sleep(self.poll_s)

    def release(self) -> None:
        if self._stream is None:
            return
        if os.name == "nt":
            import msvcrt
            self._stream.seek(0)
            msvcrt.locking(self._stream.fileno(), msvcrt.LK_UNLCK, 1)
        else:
            import fcntl
            fcntl.flock(self._stream.fileno(), fcntl.LOCK_UN)
        self._stream.close()
        self._stream = None

    def __enter__(self) -> "FileLock":
        return self.acquire()

    def __exit__(self, exc_type, exc, tb) -> None:
        self.release()


class ResourceLease(FileLock):
    """Exclusive lease for heavy_training, nicla, meter, or another named resource."""
    def __init__(self, study_root: Path | str, resource: str, timeout_s: float = 0.0):
        if not resource or any(ch not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_." for ch in resource):
            raise ValueError("resource names must be simple portable identifiers")
        super().__init__(Path(study_root) / "runs" / "leases" / (resource + ".lock"), timeout_s)
        self.resource = resource
