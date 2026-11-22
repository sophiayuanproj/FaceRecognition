"""Native Windows cross-process FNIRSI lock using a named mutex.

A process-local non-reentrant guard supplements the kernel mutex because Windows
mutexes are recursive for their owning thread. The mutex remains the
cross-process exclusion mechanism; metadata is diagnostic only.
"""
import ctypes
from ctypes import wintypes
import hashlib
import json
import math
import os
from pathlib import Path
import socket
import threading
import time

WAIT_OBJECT_0 = 0x00000000
WAIT_ABANDONED = 0x00000080
WAIT_TIMEOUT = 0x00000102
INFINITE = 0xFFFFFFFF
_LOCAL_LOCKS = {}
_LOCAL_LOCKS_GUARD = threading.Lock()


def _local_lock_for(token):
    with _LOCAL_LOCKS_GUARD:
        return _LOCAL_LOCKS.setdefault(token, threading.Lock())


class HardwareLock:
    def __init__(self, lock_path, run_id, timeout_s=0.0):
        self.lock_path = Path(lock_path)
        self.run_id = str(run_id)
        self.timeout_s = float(timeout_s)
        if self.timeout_s < 0:
            raise ValueError("timeout_s must be nonnegative")
        self._handle = None
        self._acquired = False
        self._local_handle = None
        self._local_acquired = False
        self._abandoned_recovered = False

    def _write_metadata(self, state, **extra):
        record = {
            "lock_backend": "Windows named mutex plus process-local non-reentrant guard",
            "mutex_scope": "Local",
            "state": state,
            "run_id": self.run_id,
            "pid": os.getpid(),
            "host": socket.gethostname(),
            "lock_path": str(self.lock_path.resolve()),
            "host_monotonic_ns": time.monotonic_ns(),
            "host_utc_ns": time.time_ns(),
            "abandoned_mutex_recovered": self._abandoned_recovered,
        }
        record.update(extra)
        self.lock_path.parent.mkdir(parents=True, exist_ok=True)
        temp_path = self.lock_path.with_name(self.lock_path.name + ".tmp-" + str(os.getpid()))
        with temp_path.open("w", encoding="utf-8", newline="\n") as stream:
            json.dump(record, stream, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temp_path, self.lock_path)

    def __enter__(self):
        if os.name != "nt":
            raise OSError("hardware_lock_windows requires native Windows")
        canonical = str(self.lock_path.resolve()).casefold().encode("utf-8")
        token = hashlib.sha256(canonical).hexdigest()[:32]
        deadline = time.monotonic() + self.timeout_s
        self._local_handle = _local_lock_for(token)
        if self.timeout_s == 0:
            local_ok = self._local_handle.acquire(blocking=False)
        else:
            local_ok = self._local_handle.acquire(timeout=self.timeout_s)
        if not local_ok:
            raise TimeoutError("hardware.lock is already held in this Python process")
        self._local_acquired = True
        try:
            kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
            kernel32.CreateMutexW.argtypes = (wintypes.LPVOID, wintypes.BOOL, wintypes.LPCWSTR)
            kernel32.CreateMutexW.restype = wintypes.HANDLE
            kernel32.WaitForSingleObject.argtypes = (wintypes.HANDLE, wintypes.DWORD)
            kernel32.WaitForSingleObject.restype = wintypes.DWORD
            kernel32.ReleaseMutex.argtypes = (wintypes.HANDLE,)
            kernel32.ReleaseMutex.restype = wintypes.BOOL
            kernel32.CloseHandle.argtypes = (wintypes.HANDLE,)
            kernel32.CloseHandle.restype = wintypes.BOOL
            self._kernel32 = kernel32
            self._mutex_name = "Local\\FNIRSIHardwareLock_" + token
            handle = kernel32.CreateMutexW(None, False, self._mutex_name)
            if not handle:
                raise ctypes.WinError(ctypes.get_last_error())
            self._handle = handle
            remaining_s = max(0.0, deadline - time.monotonic())
            timeout_ms = INFINITE if remaining_s >= (INFINITE / 1000.0) else int(math.ceil(remaining_s * 1000.0))
            result = kernel32.WaitForSingleObject(handle, timeout_ms)
            if result == WAIT_TIMEOUT:
                kernel32.CloseHandle(handle)
                self._handle = None
                raise TimeoutError("hardware.lock is held by another Windows process")
            if result not in (WAIT_OBJECT_0, WAIT_ABANDONED):
                error = ctypes.get_last_error()
                kernel32.CloseHandle(handle)
                self._handle = None
                raise ctypes.WinError(error)
            self._acquired = True
            self._abandoned_recovered = result == WAIT_ABANDONED
            self._write_metadata("acquired",
                                 acquired_monotonic_ns=time.monotonic_ns(),
                                 acquired_utc_ns=time.time_ns(),
                                 mutex_name=self._mutex_name)
            return self
        except BaseException:
            self._release()
            raise

    def _release(self):
        try:
            if self._handle is not None:
                try:
                    if self._acquired:
                        self._write_metadata("released",
                                             released_monotonic_ns=time.monotonic_ns(),
                                             released_utc_ns=time.time_ns())
                finally:
                    if self._acquired:
                        self._kernel32.ReleaseMutex(self._handle)
                    self._kernel32.CloseHandle(self._handle)
                    self._handle = None
                    self._acquired = False
        finally:
            if self._local_acquired:
                self._local_handle.release()
                self._local_acquired = False

    def __exit__(self, exc_type, exc, tb):
        self._release()
        return False


def default_lock_path(project_root):
    return Path(project_root) / "logs" / "hardware.lock"



