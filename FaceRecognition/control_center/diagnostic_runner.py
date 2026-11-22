"""Lifecycle for the opt-in, local TF2.10 host reconstruction worker."""

from __future__ import annotations

import base64
import json
import os
import select
import subprocess
import threading
import time
import uuid
from pathlib import Path
from typing import Any


class HostWorkerError(RuntimeError):
    pass


class HostDiagnosticWorker:
    def __init__(self, workspace: Path):
        self.workspace = Path(workspace).resolve()
        self.face_root = self.workspace / "FaceRecognition"
        self.python = self.workspace / "work" / "model-training" / ".venv" / "bin" / "python"
        self.lock = threading.RLock()
        self.process: subprocess.Popen[str] | None = None
        self.status = "off"
        self.error: str | None = None
        self.ready: dict[str, Any] | None = None

    def _read_record(self, timeout_s: float) -> dict[str, Any]:
        process = self.process
        if process is None or process.stdout is None:
            raise HostWorkerError("worker_not_started")
        ready, _, _ = select.select([process.stdout], [], [], timeout_s)
        if not ready:
            raise HostWorkerError("worker_response_timeout")
        line = process.stdout.readline()
        if not line:
            code = process.poll()
            raise HostWorkerError(f"worker_exited:{code}")
        value = json.loads(line)
        if not isinstance(value, dict):
            raise HostWorkerError("worker_protocol_invalid")
        return value

    def start(self) -> dict[str, Any]:
        with self.lock:
            if self.process is not None and self.process.poll() is None and self.status == "ready":
                return self.status_record()
            self.stop()
            if not self.python.is_file():
                self.status, self.error = "error", "tensorflow_environment_missing"
                raise HostWorkerError(self.error)
            env = os.environ.copy()
            env["PYTHONPATH"] = str(self.face_root) + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
            env["PYTHONDONTWRITEBYTECODE"] = "1"
            env["TF_CPP_MIN_LOG_LEVEL"] = "3"
            env["TF_NUM_INTRAOP_THREADS"] = "2"
            env["TF_NUM_INTEROP_THREADS"] = "1"
            env["OMP_NUM_THREADS"] = "2"
            self.status, self.error, self.ready = "starting", None, None
            try:
                self.process = subprocess.Popen(
                    [str(self.python), "-m", "control_center.host_diagnostic_worker"],
                    cwd=str(self.workspace),
                    stdin=subprocess.PIPE,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.DEVNULL,
                    text=True,
                    encoding="utf-8",
                    bufsize=1,
                    env=env,
                )
                record = self._read_record(45.0)
                if record.get("type") != "ready":
                    raise HostWorkerError(record.get("error", "worker_model_validation_failed"))
                self.ready = record
                self.status = "ready"
                return self.status_record()
            except Exception as error:
                self.error = str(error)[:180]
                self.status = "error"
                self._terminate()
                raise HostWorkerError(self.error) from error

    def infer(self, jpeg: bytes) -> dict[str, Any]:
        if not self.lock.acquire(blocking=False):
            raise HostWorkerError("worker_busy")
        try:
            if self.status != "ready" or self.process is None or self.process.poll() is not None:
                raise HostWorkerError("worker_not_ready")
            if self.process.stdin is None:
                raise HostWorkerError("worker_stdin_closed")
            request_id = uuid.uuid4().hex
            message = {
                "command": "analyze",
                "request_id": request_id,
                "jpeg_base64": base64.b64encode(jpeg).decode("ascii"),
            }
            try:
                self.process.stdin.write(json.dumps(message, separators=(",", ":")) + "\n")
                self.process.stdin.flush()
                record = self._read_record(12.0)
            except Exception as error:
                self.error = str(error)[:180]
                self.status = "error"
                self._terminate()
                raise HostWorkerError(self.error) from error
            if record.get("type") != "result" or record.get("request_id") != request_id:
                self.error = "worker_response_binding_mismatch"
                self.status = "error"
                self._terminate()
                raise HostWorkerError(self.error)
            if not record.get("ok"):
                raise HostWorkerError(record.get("error", "host_inference_failed"))
            result = record.get("result")
            if not isinstance(result, dict):
                raise HostWorkerError("worker_result_invalid")
            return result
        finally:
            self.lock.release()

    def status_record(self) -> dict[str, Any]:
        if not self.lock.acquire(blocking=False):
            return {
                "status": "busy",
                "error": None,
                "ready": self.ready,
                "runtime": "TensorFlow Lite from workspace TF2.10 environment",
            }
        try:
            if self.status == "ready" and (self.process is None or self.process.poll() is not None):
                code = None if self.process is None else self.process.poll()
                self.status = "error"
                self.error = f"worker_exited:{code}"
            return {
                "status": self.status,
                "error": self.error,
                "ready": self.ready,
                "runtime": "TensorFlow Lite from workspace TF2.10 environment",
            }
        finally:
            self.lock.release()

    def _terminate(self) -> None:
        process, self.process = self.process, None
        if process is None:
            return
        try:
            if process.poll() is None:
                process.terminate()
                process.wait(timeout=2.0)
        except Exception:
            try:
                process.kill()
                process.wait(timeout=1.0)
            except Exception:
                pass
        for stream in (process.stdin, process.stdout):
            try:
                if stream:
                    stream.close()
            except Exception:
                pass

    def stop(self) -> None:
        with self.lock:
            process = self.process
            if process is not None and process.poll() is None and process.stdin is not None:
                try:
                    process.stdin.write('{"command":"stop"}\n')
                    process.stdin.flush()
                    # The child exits after acknowledging, but the dashboard
                    # does not need to wait on a display or camera operation.
                    select.select([process.stdout], [], [], 1.0) if process.stdout else None
                except Exception:
                    pass
            self._terminate()
            self.status, self.error, self.ready = "off", None, None


