"""Local-only Nicla control center with an explicit-connect device broker.

The NiclaBroker is constructed disconnected. Only its explicit v2 Connect
route may acquire serial; legacy preview and replay routes remain separate.
"""

from __future__ import annotations

import argparse
import errno
import hashlib
import importlib.util
import json
import math
import os
import re
import signal
import socket
import sys
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
import webbrowser
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

from control_center.diagnostic_runner import HostDiagnosticWorker, HostWorkerError
from control_center.host_rgb_reference_runner import (
    HostRgbReferenceWorker,
    HostRgbReferenceWorkerError,
)
from control_center import __version__
from nicla_study.instruments.windows_fnirsi.dashboard_meter import PowerMeter


PROJECT_ROOT = Path(__file__).resolve().parents[2]
RESEARCH_ROOT = PROJECT_ROOT / "work" / "project-research-v5"
RUN_RESULT = RESEARCH_ROOT / "runs" / "camera_identity_survey_v2" / "VERIFIED_SET_RESULT.json"
PRIVATE_FRAMES = RESEARCH_ROOT / "private" / "camera_identity_survey_v2" / "scene_frames_cursor_clear"
WEB_ROOT = Path(__file__).resolve().parent / "web"
CASE_ORDER = ("K01", "K02", "K03", "K04", "K05", "OTHER", "BLANK")
CASE_IMAGES = {case: PRIVATE_FRAMES / (case + "_preview.jpg") for case in CASE_ORDER}
MAX_BODY_BYTES = 4096
MAX_HISTORY = 50
LOCAL_HTTP = urllib.request.build_opener(urllib.request.ProxyHandler({}))


def _json_response(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")


def _input_dtype_label(value: Any) -> str:
    """Normalize the board's MicroPython typecode to a readable tensor dtype."""
    raw = str(value).strip()
    # OpenMV's `array` typecode uses `b` for signed int8; telemetry transports
    # that character as its ASCII integer (98) for JSON safety.
    if raw in {"b", "98", "int8", "i1", "<i1"}:
        return "int8"
    if raw in {"B", "66", "uint8", "u1", "|u1"}:
        return "uint8"
    if raw in {"h", "104", "int16", "i2", "<i2"}:
        return "int16"
    if raw in {"H", "72", "uint16", "u2", "<u2"}:
        return "uint16"
    if raw in {"i", "105", "int32", "i4", "<i4"}:
        return "int32"
    if raw in {"f", "102", "float32", "f4", "<f4"}:
        return "float32"
    return raw or "unknown"


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("expected a JSON object")
    return value


def _loopback_preview_url(value: str) -> str:
    parsed = urllib.parse.urlsplit(value)
    if parsed.scheme != "http" or parsed.hostname not in {"127.0.0.1", "localhost", "::1"}:
        raise ValueError("live preview must be an HTTP loopback URL")
    if parsed.port != 8766:
        raise ValueError("read-only preview adapter accepts only the existing port 8766")
    if parsed.path not in {"", "/"} or parsed.query or parsed.fragment or parsed.username or parsed.password:
        raise ValueError("preview URL must be the local service root")
    return "http://127.0.0.1:8766"


def _preview_age(age_seconds: Any) -> tuple[float | None, bool]:
    """Return a known age in ms and freshness; absent/bad values are stale."""
    if type(age_seconds) not in (int, float):
        return None, False
    try:
        age_seconds = float(age_seconds)
    except (OverflowError, ValueError):
        return None, False
    if not math.isfinite(age_seconds) or age_seconds < 0:
        return None, False
    try:
        age_ms = round(age_seconds * 1000, 1)
    except (OverflowError, ValueError):
        return None, False
    return age_ms, age_seconds <= 2.0


class ControlCenter:
    def __init__(self, preview_url: str = "http://127.0.0.1:8766"):
        self.preview_url = _loopback_preview_url(preview_url)
        self.lock = threading.RLock()
        self.capture_lock = threading.Lock()
        self.started = time.monotonic()
        self.session_id = "cc-" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        self.clients: dict[str, dict[str, Any]] = {}
        self.live_error: str | None = None
        self.live_status: dict[str, Any] | None = None
        self.live_snapshot: bytes | None = None
        self.live_snapshot_received: float | None = None
        self.host_diagnostic_worker: HostDiagnosticWorker | None = None
        self.host_rgb_reference_worker: HostRgbReferenceWorker | None = None
        self.device_reference_worker: HostRgbReferenceWorker | None = None
        self.device_reference_lock = threading.Lock()
        self.stop_requested = False
        self.history: list[dict[str, Any]] = []
        self.power_meter = PowerMeter(PROJECT_ROOT)
        self.device_broker: Any | None = None
        self.device_broker_import_error: str | None = None
        try:
            from host.nicla_broker import NiclaBroker
            self.device_broker = NiclaBroker()
        except Exception as error:  # Keep replay/host diagnostics usable without the optional serial dependency.
            self.device_broker_import_error = f"{type(error).__name__}: {error}"
        self.replay_issue: str | None = None
        try:
            self.result = _read_json(RUN_RESULT)
            if self.result.get("schema") != "nicla-live-camera-identity-verified-set-v2":
                raise ValueError("the configured replay report schema changed; select a reviewed run")
            rows = self.result.get("per_condition")
            if not isinstance(rows, list):
                raise ValueError("replay result has no per_condition rows")
            self.rows = {str(row.get("case")): row for row in rows if isinstance(row, dict)}
            if set(CASE_ORDER) - set(self.rows):
                raise ValueError("replay result is missing one or more reviewed scene cases")
            missing = [str(path) for path in CASE_IMAGES.values() if not path.is_file()]
            if missing:
                raise FileNotFoundError("replay scene frames missing: " + ", ".join(missing))
        except (OSError, ValueError, json.JSONDecodeError) as error:
            # Live device operation must not depend on an optional private replay set.
            self.result = {}
            self.rows = {}
            self.replay_issue = f"{type(error).__name__}: {error}"

    def client(self, client_id: str | None) -> dict[str, Any]:
        if not isinstance(client_id, str) or not re.fullmatch(r"[A-Za-z0-9_-]{1,64}", client_id):
            client_id = "anonymous"
        with self.lock:
            return self.clients.setdefault(client_id, {
                "mode": "replay", "case": "K01", "live_attached": False,
                "observation_id": 0, "host_diagnostic_enabled": False,
                "host_rgb_reference_enabled": False,
                "live_snapshot": None, "live_snapshot_received": None,
            })

    @property
    def live_attached(self) -> bool:
        with self.lock:
            return any(bool(client.get("live_attached")) for client in self.clients.values())

    def _preview_json(self, endpoint: str, timeout: float = 1.0) -> dict[str, Any]:
        request = urllib.request.Request(self.preview_url + endpoint, headers={"Cache-Control": "no-cache"})
        with LOCAL_HTTP.open(request, timeout=timeout) as response:
            raw = response.read(65536)
        value = json.loads(raw.decode("utf-8"))
        if not isinstance(value, dict):
            raise ValueError("preview endpoint returned a non-object")
        return value

    def refresh_live_status(self) -> dict[str, Any]:
        if not self.live_attached:
            return {"available": False, "connected": False, "reason": "not_attached"}
        try:
            status = self._preview_json("/status")
            if status.get("error"):
                raise RuntimeError(str(status["error"]))
            if type(status.get("frames")) is not int or status["frames"] < 0:
                raise ValueError("preview status does not contain a valid frame count")
            with self.lock:
                self.live_status = status
                self.live_error = None
            return {"available": True, "connected": True, **status}
        except (OSError, ValueError, RuntimeError, urllib.error.URLError) as error:
            with self.lock:
                self.live_error = f"{type(error).__name__}: {error}"
                self.live_status = None
            return {"available": False, "connected": False, "reason": self.live_error}

    def connect_live(self, client_id: str | None = None) -> tuple[bool, dict[str, Any]]:
        # This only attaches to an existing loopback preview service. It never
        # claims a serial port and never starts/restarts live_camera.py.
        client = self.client(client_id)
        client["live_attached"] = True
        client["mode"] = "live"
        status = self.refresh_live_status()
        if not status.get("connected"):
            self.disconnect_live(client_id)
            return False, {"error": "Existing live preview is unavailable; no serial connection was opened.", "status": status}
        return True, {"status": status, "note": "Read-only HTTP attachment; decisions are not streamed by the preview service."}

    def disconnect_live(self, client_id: str | None = None) -> None:
        client = self.client(client_id)
        with self.lock:
            client["live_attached"] = False
            client["mode"] = "replay"
            client["host_diagnostic_enabled"] = False
            client["host_rgb_reference_enabled"] = False
            client["live_snapshot"] = None
            client["live_snapshot_received"] = None
            if not any(bool(value.get("live_attached")) for value in self.clients.values()):
                self.live_status = None
                self.live_snapshot = None
                self.live_snapshot_received = None
                self.live_error = None
            stop_worker = not any(bool(value.get("host_diagnostic_enabled")) for value in self.clients.values())
            stop_rgb_worker = not any(bool(value.get("host_rgb_reference_enabled")) for value in self.clients.values())
        if stop_worker and self.host_diagnostic_worker is not None:
            self.host_diagnostic_worker.stop()
        if stop_rgb_worker and self.host_rgb_reference_worker is not None:
            self.host_rgb_reference_worker.stop()

    def capture_live_snapshot(self, client_id: str | None = None) -> bytes:
        if not self.capture_lock.acquire(blocking=False):
            raise RuntimeError("preview_capture_busy")
        try:
            return self._capture_live_snapshot_locked(client_id)
        finally:
            self.capture_lock.release()

    def _capture_live_snapshot_locked(self, client_id: str | None = None) -> bytes:
        client = self.client(client_id)
        with self.lock:
            if not client["live_attached"]:
                raise RuntimeError("attach to the existing preview first")
        # Double-check service health around the JPEG read. Its API does not
        # expose a board frame id, so this is labeled as a host observation.
        status = self.refresh_live_status()
        _, age_fresh = _preview_age(status.get("age_s"))
        if not status.get("connected") or not age_fresh:
            raise RuntimeError("preview frame age is missing, invalid, or stale; snapshot suppressed")
        request = urllib.request.Request(self.preview_url + "/snapshot.jpg", headers={"Cache-Control": "no-cache"})
        with LOCAL_HTTP.open(request, timeout=2.0) as response:
            jpeg = response.read(1_000_001)
        if not 100 < len(jpeg) <= 1_000_000 or not jpeg.startswith(b"\xff\xd8") or not jpeg.endswith(b"\xff\xd9"):
            raise ValueError("existing preview returned an invalid or oversized JPEG")
        status_after = self.refresh_live_status()
        _, age_fresh_after = _preview_age(status_after.get("age_s"))
        if not status_after.get("connected") or not age_fresh_after:
            raise RuntimeError("preview became stale while reading the snapshot; image suppressed")
        with self.lock:
            client["observation_id"] += 1
            client["live_snapshot"] = jpeg
            client["live_snapshot_received"] = time.monotonic()
            self.live_snapshot = jpeg
            self.live_snapshot_received = client["live_snapshot_received"]
        return jpeg

    def enable_host_diagnostic(self, client_id: str | None = None) -> dict[str, Any]:
        client = self.client(client_id)
        if not client["live_attached"]:
            raise RuntimeError("attach to the existing preview before enabling host reconstruction")
        client["host_diagnostic_enabled"] = True
        if self.host_diagnostic_worker is None:
            self.host_diagnostic_worker = HostDiagnosticWorker(PROJECT_ROOT)
        try:
            worker = self.host_diagnostic_worker.start()
        except HostWorkerError as error:
            return {"enabled": True, "status": "error", "error": str(error)}
        return {"enabled": True, **worker}

    def host_diagnostic_status(self, client_id: str | None = None) -> dict[str, Any]:
        client = self.client(client_id)
        if not client["host_diagnostic_enabled"]:
            return {"enabled": False, "status": "off", "error": None, "ready": None}
        if not client["live_attached"]:
            return {"enabled": False, "status": "off", "error": "preview_detached", "ready": None}
        if self.host_diagnostic_worker is None:
            return {"enabled": True, "status": "starting", "error": None, "ready": None}
        return {"enabled": True, **self.host_diagnostic_worker.status_record()}

    def disable_host_diagnostic(self, client_id: str | None = None) -> dict[str, Any]:
        client = self.client(client_id)
        with self.lock:
            client["host_diagnostic_enabled"] = False
            stop_worker = not any(bool(value.get("host_diagnostic_enabled")) for value in self.clients.values())
        if stop_worker and self.host_diagnostic_worker is not None:
            self.host_diagnostic_worker.stop()
        return {"enabled": False, "status": "off", "error": None}

    def enable_host_rgb_reference(self, client_id: str | None = None) -> dict[str, Any]:
        client = self.client(client_id)
        if not client["live_attached"]:
            raise RuntimeError("attach to the existing preview before enabling HOST RGB REFERENCE")
        client["host_rgb_reference_enabled"] = True
        if self.host_rgb_reference_worker is None:
            self.host_rgb_reference_worker = HostRgbReferenceWorker(PROJECT_ROOT)
        try:
            worker = self.host_rgb_reference_worker.start()
        except HostRgbReferenceWorkerError as error:
            return {"enabled": True, "status": "error", "error": str(error)}
        return {"enabled": True, **worker}

    def host_rgb_reference_status(self, client_id: str | None = None) -> dict[str, Any]:
        client = self.client(client_id)
        if not client["host_rgb_reference_enabled"]:
            return {"enabled": False, "status": "off", "error": None, "ready": None}
        if not client["live_attached"]:
            return {"enabled": False, "status": "off", "error": "preview_detached", "ready": None}
        if self.host_rgb_reference_worker is None:
            return {"enabled": True, "status": "starting", "error": None, "ready": None}
        return {"enabled": True, **self.host_rgb_reference_worker.status_record()}

    def disable_host_rgb_reference(self, client_id: str | None = None) -> dict[str, Any]:
        client = self.client(client_id)
        with self.lock:
            client["host_rgb_reference_enabled"] = False
            stop_worker = not any(bool(value.get("host_rgb_reference_enabled")) for value in self.clients.values())
        if stop_worker and self.host_rgb_reference_worker is not None:
            self.host_rgb_reference_worker.stop()
        return {"enabled": False, "status": "off", "error": None}

    def run_host_rgb_reference(self, client_id: str | None, expected_observation: int) -> dict[str, Any]:
        client = self.client(client_id)
        if not client["live_attached"]:
            raise RuntimeError("preview_detached")
        if not client["host_rgb_reference_enabled"]:
            raise RuntimeError("host_rgb_reference_not_opted_in")
        if self.host_rgb_reference_worker is None or self.host_rgb_reference_worker.status != "ready":
            raise RuntimeError("host_rgb_reference_worker_not_ready")
        if isinstance(expected_observation, bool) or not isinstance(expected_observation, int):
            raise RuntimeError("invalid_host_observation_index")

        status_before = self.refresh_live_status()
        preview_age_before, preview_fresh_before = _preview_age(status_before.get("age_s"))
        with self.lock:
            source_observation = int(client["observation_id"])
            jpeg = client.get("live_snapshot")
            snapshot_received = client.get("live_snapshot_received")
        if expected_observation != source_observation:
            raise RuntimeError("host_observation_superseded")
        if not isinstance(jpeg, bytes) or snapshot_received is None:
            raise RuntimeError("host_snapshot_unavailable")
        snapshot_age_before = max(0, round((time.monotonic() - snapshot_received) * 1000))
        if not status_before.get("connected") or not preview_fresh_before or snapshot_age_before > 2000:
            raise RuntimeError("preview_frame_stale_before_host_rgb_reference")

        result = self.host_rgb_reference_worker.infer(jpeg)
        status_after = self.refresh_live_status()
        preview_age_after, preview_fresh_after = _preview_age(status_after.get("age_s"))
        snapshot_age_after = max(0, round((time.monotonic() - snapshot_received) * 1000))
        with self.lock:
            current_observation = int(client["observation_id"])
            reference_still_enabled = bool(client["host_rgb_reference_enabled"])
            preview_still_attached = bool(client["live_attached"])
        if not reference_still_enabled or not preview_still_attached:
            raise RuntimeError("host_rgb_reference_disabled_during_inference")
        if current_observation != source_observation:
            raise RuntimeError("host_observation_superseded_during_rgb_reference")
        if not status_after.get("connected") or not preview_fresh_after or snapshot_age_after > 2000:
            raise RuntimeError("preview_frame_stale_after_host_rgb_reference")
        expected_jpeg_hash = hashlib.sha256(jpeg).hexdigest()
        if result.get("source", {}).get("jpeg_sha256") != expected_jpeg_hash:
            raise RuntimeError("host_rgb_reference_jpeg_binding_mismatch")
        result["host_observation_index"] = source_observation
        result["source"]["host_observation_index"] = source_observation
        result["source"]["snapshot_age_ms_at_start"] = snapshot_age_before
        result["source"]["snapshot_age_ms_at_finish"] = snapshot_age_after
        result["source"]["preview_service_age_ms_at_start"] = preview_age_before
        result["source"]["preview_service_age_ms_at_finish"] = preview_age_after
        result["source"]["binding"] = "host observation index + exact cached RGB JPEG SHA-256; no device frame ID"
        result["qualification"] = "HOST REFERENCE ONLY; NOT ON NICLA; recognition not qualified"
        result["recognition"]["qualification"] = result["qualification"]
        return result

    def run_host_diagnostic(self, client_id: str | None, expected_observation: int) -> dict[str, Any]:
        client = self.client(client_id)
        if not client["live_attached"]:
            raise RuntimeError("preview_detached")
        if not client["host_diagnostic_enabled"]:
            raise RuntimeError("host_diagnostic_not_opted_in")
        if self.host_diagnostic_worker is None or self.host_diagnostic_worker.status != "ready":
            raise RuntimeError("host_diagnostic_worker_not_ready")
        if isinstance(expected_observation, bool) or not isinstance(expected_observation, int):
            raise RuntimeError("invalid_host_observation_index")

        status_before = self.refresh_live_status()
        preview_age_before, preview_fresh_before = _preview_age(status_before.get("age_s"))
        with self.lock:
            source_observation = int(client["observation_id"])
            jpeg = client.get("live_snapshot")
            snapshot_received = client.get("live_snapshot_received")
        if expected_observation != source_observation:
            raise RuntimeError("host_observation_superseded")
        if not isinstance(jpeg, bytes) or snapshot_received is None:
            raise RuntimeError("host_snapshot_unavailable")
        snapshot_age_before = max(0, round((time.monotonic() - snapshot_received) * 1000))
        if not status_before.get("connected") or not preview_fresh_before or snapshot_age_before > 2000:
            raise RuntimeError("preview_frame_stale_before_host_inference")

        result = self.host_diagnostic_worker.infer(jpeg)
        status_after = self.refresh_live_status()
        preview_age_after, preview_fresh_after = _preview_age(status_after.get("age_s"))
        snapshot_age_after = max(0, round((time.monotonic() - snapshot_received) * 1000))
        with self.lock:
            current_observation = int(client["observation_id"])
            diagnostic_still_enabled = bool(client["host_diagnostic_enabled"])
            preview_still_attached = bool(client["live_attached"])
        if not diagnostic_still_enabled or not preview_still_attached:
            raise RuntimeError("host_diagnostic_disabled_during_inference")
        if current_observation != source_observation:
            raise RuntimeError("host_observation_superseded_during_inference")
        if not status_after.get("connected") or not preview_fresh_after or snapshot_age_after > 2000:
            raise RuntimeError("preview_frame_stale_after_host_inference")
        expected_jpeg_hash = hashlib.sha256(jpeg).hexdigest()
        if result.get("source", {}).get("jpeg_sha256") != expected_jpeg_hash:
            raise RuntimeError("host_diagnostic_jpeg_binding_mismatch")
        result["host_observation_index"] = source_observation
        result["source"]["host_observation_index"] = source_observation
        result["source"]["snapshot_age_ms_at_start"] = snapshot_age_before
        result["source"]["snapshot_age_ms_at_finish"] = snapshot_age_after
        result["source"]["preview_service_age_ms_at_start"] = preview_age_before
        result["source"]["preview_service_age_ms_at_finish"] = preview_age_after
        result["source"]["binding"] = "host observation index + exact cached JPEG SHA-256; no device frame ID"
        result["qualification"] = "EXPERIMENTAL / UNQUALIFIED; no identity is emitted"
        result["identity"]["qualification"] = result["qualification"]
        return result

    def run_device_host_reference(self, boot_id: str, frame_id: int) -> dict[str, Any]:
        """Analyze a broker-verified device JPEG without issuing a device command."""
        if not isinstance(boot_id, str) or not re.fullmatch(r"[A-Za-z0-9._:-]{1,128}", boot_id):
            raise ValueError("invalid boot_id")
        if type(frame_id) is not int or frame_id < 0:
            raise ValueError("invalid frame_id")
        if not self.device_reference_lock.acquire(blocking=False):
            raise RuntimeError("Mac reference is busy")
        try:
            jpeg, meta = self.device_broker.get_frame(boot_id, frame_id)
            digest = hashlib.sha256(jpeg).hexdigest()
            if (meta.get("boot_id") != boot_id or meta.get("frame_id") != frame_id
                    or meta.get("sha256") != digest):
                raise RuntimeError("device frame binding mismatch")
            if self.device_reference_worker is None:
                self.device_reference_worker = HostRgbReferenceWorker(PROJECT_ROOT)
            self.device_reference_worker.start()
            result = self.device_reference_worker.infer(jpeg)
            if result.get("source", {}).get("jpeg_sha256") != digest:
                raise RuntimeError("Mac result frame hash mismatch")
            result["source"].update(device_boot_id=boot_id, device_frame_id=frame_id)
            result["evidence_scope"] = "MAC REFERENCE ON A VERIFIED NICLA FRAME"
            return result
        finally:
            self.device_reference_lock.release()

    def close(self) -> None:
        try:
            self.power_meter.close()
        except Exception as error:
            print(f"Power meter close failed: {type(error).__name__}: {error}", flush=True)
        if self.host_diagnostic_worker is not None:
            self.host_diagnostic_worker.stop()
        if self.host_rgb_reference_worker is not None:
            self.host_rgb_reference_worker.stop()
        with self.device_reference_lock:
            if self.device_reference_worker is not None:
                self.device_reference_worker.stop()
        if self.device_broker is not None:
            try:
                self.device_broker.close()
            except Exception as error:
                print(f"Device broker close failed: {type(error).__name__}: {error}", flush=True)

    def health(self) -> dict[str, Any]:
        broker_status: dict[str, Any] | None = None
        if self.device_broker is not None:
            try:
                broker_status = self.device_broker.status()
            except Exception:
                broker_status = None
        serial_open = bool(broker_status and broker_status.get("serial_open") is True)
        device_connected = bool(broker_status and broker_status.get("connected") is True)
        return {
            "ok": True,
            "service": "nicla-control-center",
            "version": __version__,
            "bind": "127.0.0.1",
            "uptime_s": round(time.monotonic() - self.started, 2),
            "mode": "legacy_preview" if self.live_attached else "replay",
            "live_attached": self.live_attached,
            "serial_owner": "NiclaBroker in this process; serial handle open" if serial_open else "NiclaBroker in this process; no serial handle open until Connect by a leased controller",
            "serial_opened_by_service": serial_open,
            "power": None,
            "replay_available": self.replay_issue is None,
            "replay_error": self.replay_issue,
            "device_broker_available": self.device_broker is not None,
            "device_connected": device_connected,
            "device_connection_status": broker_status,
            "device_broker_error": self.device_broker_import_error,
            "stop_requested": self.stop_requested,
        }

    def replay_case(self, case: str | None = None, client_id: str | None = None) -> dict[str, Any]:
        if self.replay_issue is not None:
            raise RuntimeError("reviewed replay unavailable: " + self.replay_issue)
        client = self.client(client_id)
        case = case or client["case"]
        if case not in self.rows:
            raise KeyError(case)
        row = self.rows[case]
        index = CASE_ORDER.index(case) + 1
        client["case"] = case
        client["mode"] = "replay"
        return {
            "schema_version": "control-center-event-v1",
            "session_id": "camera_identity_survey_v2",
            "boot_id": "probe_boot_id_not_retained",
            "frame_id": index,
            "acquired_monotonic_ms": None,
            "clock_domain": "recorded_case_order; source acquisition tick not retained",
            "host_received_unix_ms": int(time.time() * 1000),
            "model_sha256": self.result.get("candidate", {}).get("encoder_sha256"),
            "head_sha256": self.result.get("candidate", {}).get("head_sha256"),
            "config_sha256": None,
            "contract_sha256": None,
            "connection": {
                "state": "disconnected",
                "device": "Nicla Vision (recorded result)",
                "firmware": "not_bound_in_this_report",
                "execution_location": "recorded_replay",
            },
            "frame": {
                "frame_id": index,
                "width": 240,
                "height": 240,
                "orientation": "recorded camera preview; orientation metadata not retained",
                "source": "recording",
                "age_ms": None,
                "stale": False,
                "frame_ref": f"/api/v1/replay/frame?case={case}",
            },
            "detection": {
                "frame_id": index,
                "coordinate_space": "sensor_px",
                "status": "face" if row.get("fomo_accepted_clusters", 0) == 1 else ("no_face" if row.get("fomo_accepted_clusters", 0) == 0 else "multiple"),
                "centers_xy": [],
                "regions_xywh": [],
                "quality": {"accepted_clusters": row.get("fomo_accepted_clusters"), "raw_clusters": row.get("fomo_raw_clusters"), "score_values_not_retained": True},
            },
            "model_input": {
                "frame_id": None,
                "source": "none",
                "provenance": "Exact board RGB565 frame, crop, resized tensor and scores were not retained for this historic probe.",
                "width": None,
                "height": None,
                "channels": None,
                "dtype": None,
                "scale": None,
                "zero_point": None,
                "sha256": None,
                "warnings": ["The adjacent saved preview is a scene reference, not the exact board input frame."],
            },
            "decision": {
                "frame_id": index,
                "status": row.get("decision", "ERROR"),
                "scores": {},
                "score_semantics": "none",
                "threshold": None,
                "margin": None,
                "temporal_state": "single volatile probe; no temporal confirmation",
                "reason": f"Recorded diagnostic for {case}: predicted={row.get('predicted_class')!r}; exact score/input rows were not retained.",
                "experimental": True,
            },
            "telemetry": {
                "preview_fps": None,
                "inference_hz": None,
                "timings_ms": {"probe_total_including_reset_and_warmup": row.get("elapsed_ms")},
                "heap_bytes": None,
                "arena_bytes": None,
                "framebuffer_bytes": None,
                "imu": None,
                "sensor_sleep": None,
                "model_slot": "one-root diagnostic head; not deployable",
                "dropped_frames": 0,
                "errors": [],
                "ack": None,
            },
            "policy": {
                "selected": "historical_volatile_identity_probe",
                "requested": {},
                "acknowledged": {},
                "trace": ["Replay only; no board command was sent by the control center."],
            },
            "case": case,
            "role": row.get("role"),
            "predicted_class": row.get("predicted_class"),
            "elapsed_ms": row.get("elapsed_ms"),
            "expected_identity_for_development_case": case if case.startswith("K0") else None,
            "result": row,
            "limitations": self.result.get("limitations", []),
        }

    def live_state(self, client_id: str | None = None) -> dict[str, Any]:
        client = self.client(client_id)
        status = self.refresh_live_status() if client["live_attached"] else {"available": False, "connected": False, "reason": "not_attached"}
        with self.lock:
            now_mono_ms = int(time.monotonic() * 1000)
            frame_id = int(client["observation_id"])
            snapshot_received = client.get("live_snapshot_received")
            snapshot_age = None if snapshot_received is None else max(0, round((time.monotonic() - snapshot_received) * 1000))
            last_status = self.live_status or {}
            attached = bool(client["live_attached"])
            selected_case = client["case"]
            error = self.live_error
        state = status.get("connected") is True
        age_ms, age_fresh = _preview_age(last_status.get("age_s"))
        age_known_and_fresh = state and age_fresh
        return {
            "schema_version": "control-center-event-v1",
            "session_id": self.session_id,
            "client_id": client_id or "anonymous",
            "boot_id": "device_boot_id_unavailable_from_preview_service",
            "frame_id": frame_id,
            "acquired_monotonic_ms": now_mono_ms,
            "clock_domain": "host_monotonic_when_preview_snapshot_observed; device tick unavailable",
            "host_received_unix_ms": int(time.time() * 1000),
            "model_sha256": None,
            "head_sha256": None,
            "config_sha256": None,
            "contract_sha256": None,
            "connection": {
                "state": "connected" if state else ("connecting" if attached else "disconnected"),
                "device": "Nicla Vision through existing read-only preview service" if state else None,
                "firmware": None,
                "execution_location": "on_nicla" if state else "host_reference",
            },
            "frame": {
                "frame_id": frame_id,
                "width": 240 if frame_id else None,
                "height": 240 if frame_id else None,
                "orientation": "unknown",
                "source": "nicla" if state else "none",
                "age_ms": age_ms if state else None,
                "stale": not age_known_and_fresh,
                "frame_ref": "/api/v1/live/snapshot.jpg" if frame_id and age_known_and_fresh else None,
            },
            "detection": {
                "frame_id": None,
                "coordinate_space": "none",
                "status": "not_run",
                "centers_xy": [],
                "regions_xywh": [],
                "quality": {},
            },
            "model_input": {
                "frame_id": None,
                "source": "none",
                "provenance": "Legacy v1 preview endpoint does not emit the board tensor or frame-linked decision stream; the v2 NiclaBroker API is separate.",
                "width": None,
                "height": None,
                "channels": None,
                "dtype": None,
                "scale": None,
                "zero_point": None,
                "sha256": None,
                "warnings": ["Preview frame observations use a host receive sequence; there is no shared firmware frame_id/boot_id."],
            },
            "decision": {
                "frame_id": None,
                "status": "not_run",
                "scores": {},
                "score_semantics": "none",
                "threshold": None,
                "margin": None,
                "temporal_state": "no live inference broker",
                "reason": "Live camera preview only. No decision is attached to this frame.",
                "experimental": True,
            },
            "telemetry": {
                "preview_fps": last_status.get("fps") if state else None,
                "inference_hz": None,
                "timings_ms": {},
                "heap_bytes": None,
                "arena_bytes": None,
                "framebuffer_bytes": None,
                "imu": None,
                "sensor_sleep": None,
                "model_slot": None,
                "dropped_frames": 0,
                "errors": [error] if error else [],
                "ack": None,
            },
            "policy": {
                "selected": "preview_only",
                "requested": {},
                "acknowledged": {},
                "trace": ["Read-only HTTP attachment to existing serial owner; control commands are unavailable."],
            },
            "preview_frames_received_by_service": last_status.get("frames"),
            "preview_snapshot_age_ms": snapshot_age,
            "live_status": status,
            "selected_case": selected_case,
        }


class ControlCenterHTTPServer(ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = True

    def __init__(self, server_address, RequestHandlerClass, bind_and_activate=True):
        self._shutdown_lock = threading.Lock()
        self._shutdown_requested = False
        super().__init__(server_address, RequestHandlerClass, bind_and_activate)


def request_server_shutdown(app: ControlCenter, server: ControlCenterHTTPServer, source: str) -> None:
    """Stop serve_forever from a helper thread exactly once."""
    with server._shutdown_lock:
        if server._shutdown_requested:
            return
        server._shutdown_requested = True
        app.stop_requested = True
    print(f"Stopping control center ({source})…", flush=True)
    threading.Thread(target=server.shutdown, name="control-center-shutdown", daemon=True).start()


def make_handler(app: ControlCenter, server_ref: dict[str, ControlCenterHTTPServer]):
    class Handler(BaseHTTPRequestHandler):
        server_version = "NiclaControlCenter/0.1"
        sys_version = ""

        def log_message(self, fmt, *args):
            rendered = fmt % args
            if '"/api/v1/health ' in rendered or rendered.startswith('"GET /api/v2/power/'):
                return
            print("[http] " + rendered, flush=True)

        def send_json(self, status: int, value: Any):
            raw = _json_response(value)
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(raw)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header("X-Frame-Options", "SAMEORIGIN")
            self.send_header("Cross-Origin-Resource-Policy", "same-origin")
            self.end_headers()
            self.wfile.write(raw)

        def send_bytes(self, status: int, data: bytes, content_type: str, cache: str = "no-store", extra_headers: dict[str, str] | None = None):
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(data)))
            self.send_header("Cache-Control", cache)
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header("X-Frame-Options", "SAMEORIGIN")
            self.send_header("Cross-Origin-Resource-Policy", "same-origin")
            for name, value in (extra_headers or {}).items():
                if re.fullmatch(r"[A-Za-z0-9-]{1,64}", name) and "\r" not in value and "\n" not in value:
                    self.send_header(name, value)
            self.end_headers()
            self.wfile.write(data)

        def require_device_broker(self):
            if app.device_broker is None:
                return None
            return app.device_broker

        def viewer_id(self) -> str:
            value = self.headers.get("X-Viewer-ID") or self.headers.get("X-Nicla-Viewer-ID") or self.client_id()
            return value[:128]

        def read_body(self) -> dict[str, Any]:
            try:
                length = int(self.headers.get("Content-Length", "0"))
            except ValueError as error:
                raise ValueError("invalid Content-Length") from error
            if length < 0 or length > MAX_BODY_BYTES:
                raise ValueError("request body exceeds the 4096-byte limit")
            raw = self.rfile.read(length)
            value = json.loads(raw.decode("utf-8")) if raw else {}
            if not isinstance(value, dict):
                raise ValueError("request body must be a JSON object")
            return value

        def client_id(self) -> str:
            return self.headers.get("X-Control-Center-Client", "anonymous")

        def local_power_request(self) -> bool:
            port = self.server.server_port
            host = self.headers.get("Host", "").lower()
            loopback = ("127.0.0.1", "localhost", "[::1]")
            allowed = {f"{address}:{port}" for address in loopback}
            if port == 80:
                allowed.update(loopback)
            origin = self.headers.get("Origin")
            return (host in allowed and origin in (None, f"http://{host}")
                    and self.headers.get("Sec-Fetch-Site") in (None, "none", "same-origin"))

        def power_post(self, path: str, body: dict[str, Any]):
            meter = app.power_meter
            if app.stop_requested:
                return self.send_json(503, {"error": "Control center is shutting down."})
            try:
                if path == "/api/v2/power/connect":
                    result = meter.connect(body.get("interface_id"))
                elif path == "/api/v2/power/start-recording":
                    result = meter.start_recording(body.get("topology_confirmed", False))
                else:
                    action = {
                        "/api/v2/power/scan": meter.scan,
                        "/api/v2/power/disconnect": meter.disconnect,
                        "/api/v2/power/stop-recording": meter.stop_recording,
                    }.get(path)
                    if action is None:
                        return self.send_json(404, {"error": "unknown power meter route"})
                    result = action()
            except ValueError as error:
                return self.send_json(400, {"error": str(error)})
            except RuntimeError as error:
                return self.send_json(409, {"error": str(error)})
            return self.send_json(200, result)

        def do_GET(self):
            parsed = urllib.parse.urlsplit(self.path)
            path = parsed.path
            try:
                if path == "/":
                    return self.send_bytes(200, (WEB_ROOT / "index.html").read_bytes(), "text/html; charset=utf-8")
                if path == "/app.css":
                    return self.send_bytes(200, (WEB_ROOT / "app.css").read_bytes(), "text/css; charset=utf-8")
                if path == "/app.js":
                    return self.send_bytes(200, (WEB_ROOT / "app.js").read_bytes(), "text/javascript; charset=utf-8")
                if path == "/device_ui.js":
                    return self.send_bytes(200, (WEB_ROOT / "device_ui.js").read_bytes(), "text/javascript; charset=utf-8")
                if path == "/power_ui.js":
                    return self.send_bytes(200, (WEB_ROOT / "power_ui.js").read_bytes(), "text/javascript; charset=utf-8")
                if path in ("/api/v2/power/status", "/api/v2/power/samples"):
                    if not self.local_power_request():
                        return self.send_json(403, {"error": "Power meter requests must use this loopback origin."})
                    if path.endswith("/status"):
                        return self.send_json(200, app.power_meter.status())
                    try:
                        query = urllib.parse.parse_qs(parsed.query, max_num_fields=2)
                        after = int(query.get("after", ["0"])[0])
                        limit = int(query.get("limit", ["60"])[0])
                        result = app.power_meter.samples(after, limit)
                    except ValueError as error:
                        return self.send_json(400, {"error": str(error)})
                    return self.send_json(200, result)
                if path == "/api/v1/health":
                    return self.send_json(200, app.health())
                if path == "/api/v2/device/status":
                    broker = self.require_device_broker()
                    if broker is None:
                        return self.send_json(503, {"error": "device broker unavailable", "detail": app.device_broker_import_error})
                    return self.send_json(200, broker.status(viewer_id=self.viewer_id()))
                if path == "/api/v2/device/battery":
                    broker = self.require_device_broker()
                    if broker is None:
                        return self.send_json(503, {"error": "device broker unavailable"})
                    return self.send_json(200, broker.battery_status(viewer_id=self.viewer_id()))
                if path in ("/api/v2/device/sightings", "/api/v2/device/sightings/frame"):
                    broker = self.require_device_broker()
                    if broker is None:
                        return self.send_json(503, {"error": "device broker unavailable"})
                    if path.endswith("/sightings"):
                        return self.send_json(200, broker.sightings_status())
                    index = urllib.parse.parse_qs(parsed.query).get("id", [""])[0]
                    if not re.fullmatch(r"[1-9][0-9]{0,17}", index):
                        return self.send_json(400, {"error": "Invalid sighting ID."})
                    try:
                        jpeg = broker.get_sighting_jpeg(int(index))
                    except KeyError:
                        return self.send_json(404, {"error": "This sighting photo is no longer retained."})
                    return self.send_bytes(200, jpeg, "image/jpeg")
                if path == "/api/v2/device/capabilities":
                    broker = self.require_device_broker()
                    if broker is None:
                        return self.send_json(503, {"error": "device broker unavailable", "detail": app.device_broker_import_error})
                    return self.send_json(200, broker.capabilities())
                if path == "/api/v2/device/events":
                    broker = self.require_device_broker()
                    if broker is None:
                        return self.send_json(503, {"error": "device broker unavailable", "detail": app.device_broker_import_error})
                    query = urllib.parse.parse_qs(parsed.query, strict_parsing=True)
                    after = int(query.get("after", ["0"])[0])
                    limit = int(query.get("limit", ["100"])[0])
                    if after < 0 or not 1 <= limit <= 100:
                        return self.send_json(400, {"error": "after must be non-negative and limit must be 1..100"})
                    return self.send_json(200, broker.events(after_seq=after, limit=limit))
                if path in ("/api/v2/device/video", "/api/v2/device/video/file",
                            "/api/v2/device/video/manifest", "/api/v2/device/video/frame"):
                    broker = self.require_device_broker()
                    if broker is None:
                        return self.send_json(503, {"error": "device broker unavailable"})
                    if path.endswith("/video"):
                        return self.send_json(200, broker.video_status())
                    query = urllib.parse.parse_qs(parsed.query)
                    clip_id = query.get("id", [""])[0]
                    if path.endswith("/frame"):
                        index = query.get("number", [""])[0]
                        if not re.fullmatch(r"[1-9][0-9]{0,5}", index):
                            return self.send_json(400, {"error": "Invalid video frame number."})
                        filename, content_type = f"frames/frame-{int(index):06d}.jpg", "image/jpeg"
                    elif path.endswith("/manifest"):
                        filename, content_type = "manifest.json", "application/json; charset=utf-8"
                    else:
                        filename, content_type = "clip.avi", "video/x-msvideo"
                    try:
                        data = (broker.get_video_file(clip_id) if filename == "clip.avi" else
                                broker.get_video_file(clip_id, filename))
                    except KeyError:
                        return self.send_json(404, {"error": "Video is unavailable."})
                    headers = {"Content-Disposition": f'attachment; filename="nicla-{clip_id}.avi"'} if filename == "clip.avi" else None
                    return self.send_bytes(200, data, content_type, extra_headers=headers)
                if path == "/api/v2/device/frame":
                    broker = self.require_device_broker()
                    if broker is None:
                        return self.send_json(503, {"error": "device broker unavailable", "detail": app.device_broker_import_error})
                    query = urllib.parse.parse_qs(parsed.query, strict_parsing=True)
                    boot_id = query.get("boot_id", [""])[0]
                    frame_id = int(query.get("frame_id", [""])[0])
                    if not re.fullmatch(r"[A-Za-z0-9._:-]{1,128}", boot_id) or frame_id < 0:
                        return self.send_json(400, {"error": "invalid boot_id or frame_id"})
                    jpeg, metadata = broker.get_frame(boot_id, frame_id)
                    if not isinstance(jpeg, bytes) or not jpeg.startswith(b"\xff\xd8") or not jpeg.endswith(b"\xff\xd9") or len(jpeg) > 1_000_000:
                        return self.send_json(502, {"error": "broker frame is not a valid JPEG under 1 MB"})
                    headers = {
                        "X-Device-Boot-ID": str(metadata.get("boot_id", boot_id)),
                        "X-Device-Frame-ID": str(metadata.get("frame_id", frame_id)),
                        "X-Frame-SHA256": str(metadata.get("sha256", "")),
                    }
                    return self.send_bytes(200, jpeg, "image/jpeg", extra_headers=headers)
                if path == "/api/v2/device/input":
                    broker = self.require_device_broker()
                    if broker is None:
                        return self.send_json(503, {"error": "device broker unavailable", "detail": app.device_broker_import_error})
                    query = urllib.parse.parse_qs(parsed.query, strict_parsing=True)
                    boot_id = query.get("boot_id", [""])[0]
                    frame_id = int(query.get("frame_id", [""])[0])
                    kind = query.get("kind", [""])[0]
                    if not re.fullmatch(r"[A-Za-z0-9._:-]{1,128}", boot_id) or frame_id < 0 or kind not in {"detector", "identity"}:
                        return self.send_json(400, {"error": "invalid boot_id, frame_id or kind"})
                    raw, metadata = broker.get_input(boot_id, frame_id, kind)
                    if not isinstance(raw, bytes) or len(raw) > 262_144:
                        return self.send_json(502, {"error": "broker input exceeds the 256 KiB diagnostic limit"})
                    shape = metadata.get("shape", [])
                    headers = {
                        "X-Device-Boot-ID": str(metadata.get("boot_id", boot_id)),
                        "X-Device-Frame-ID": str(metadata.get("frame_id", frame_id)),
                        "X-Input-Kind": kind,
                        "X-Input-SHA256": str(metadata.get("sha256", "")),
                        "X-Input-Shape": json.dumps(shape, separators=(",", ":")),
                        "X-Input-DType": _input_dtype_label(metadata.get("dtype", "int8")),
                    }
                    if metadata.get("scale") is not None: headers["X-Input-Scale"] = str(metadata["scale"])
                    if metadata.get("zero_point") is not None: headers["X-Input-Zero-Point"] = str(metadata["zero_point"])
                    return self.send_bytes(200, raw, "application/octet-stream", extra_headers=headers)
                if path == "/api/v1/state":
                    client = app.client(self.client_id())
                    if client["mode"] == "replay" and app.replay_issue is not None:
                        return self.send_json(503, {"error": "reviewed replay unavailable", "detail": app.replay_issue, "replay_available": False})
                    return self.send_json(200, app.live_state(self.client_id()) if client["mode"] == "live" else app.replay_case(client_id=self.client_id()))
                if path == "/api/v1/live/diagnostic/status":
                    return self.send_json(200, app.host_diagnostic_status(self.client_id()))
                if path == "/api/v1/live/rgb-reference/status":
                    return self.send_json(200, app.host_rgb_reference_status(self.client_id()))
                if path == "/api/v1/replay/cases":
                    if app.replay_issue is not None:
                        return self.send_json(503, {"error": "reviewed replay unavailable", "detail": app.replay_issue, "replay_available": False})
                    return self.send_json(200, {"mode": "RECORDED REPLAY", "cases": [
                        {"case": case, "role": app.rows[case].get("role"), "decision": app.rows[case].get("decision"),
                         "predicted_class": app.rows[case].get("predicted_class"), "elapsed_ms": app.rows[case].get("elapsed_ms")}
                        for case in CASE_ORDER]})
                if path == "/api/v1/replay/frame":
                    if app.replay_issue is not None:
                        return self.send_json(503, {"error": "reviewed replay unavailable", "detail": app.replay_issue, "replay_available": False})
                    query = urllib.parse.parse_qs(parsed.query, strict_parsing=True)
                    case = query.get("case", [""])[0]
                    if case not in CASE_IMAGES:
                        return self.send_json(404, {"error": "unknown replay case"})
                    return self.send_bytes(200, CASE_IMAGES[case].read_bytes(), "image/jpeg")
                if path == "/api/v1/live/snapshot.jpg":
                    if app.client(self.client_id())["mode"] != "live" or not app.client(self.client_id())["live_attached"]:
                        return self.send_json(409, {"error": "live preview is not attached"})
                    jpeg = app.capture_live_snapshot(self.client_id())
                    observation = app.client(self.client_id())["observation_id"]
                    self.send_response(200)
                    self.send_header("Content-Type", "image/jpeg")
                    self.send_header("Content-Length", str(len(jpeg)))
                    self.send_header("Cache-Control", "no-store")
                    self.send_header("X-Preview-Observation", str(observation))
                    self.end_headers()
                    self.wfile.write(jpeg)
                    return
                return self.send_json(404, {"error": "not found"})
            except KeyError:
                return self.send_json(404, {"error": "unknown case"})
            except (OSError, ValueError, RuntimeError, urllib.error.URLError, json.JSONDecodeError) as error:
                return self.send_json(502, {"error": f"{type(error).__name__}: {error}"})

        def do_POST(self):
            parsed = urllib.parse.urlsplit(self.path)
            try:
                if parsed.path.startswith("/api/v2/power/"):
                    if (not self.local_power_request()
                            or not re.fullmatch(r"[A-Za-z0-9_-]{1,64}", self.headers.get("X-Control-Center-Client", ""))):
                        return self.send_json(403, {"error": "Power meter actions require a local dashboard client."})
                    if self.headers.get_content_type() != "application/json":
                        return self.send_json(415, {"error": "Power meter actions require a JSON body."})
                    return self.power_post(parsed.path, self.read_body())
                body = self.read_body()
                if parsed.path.startswith("/api/v2/device/"):
                    broker = self.require_device_broker()
                    if broker is None:
                        return self.send_json(503, {"error": "device broker unavailable", "detail": app.device_broker_import_error})
                    if parsed.path in ("/api/v2/device/battery", "/api/v2/device/battery/sightings/delete"):
                        viewer = self.viewer_id()
                        if body.get("viewer_id") != viewer:
                            return self.send_json(403, {"error": "viewer_id must match X-Viewer-ID"})
                        try:
                            if parsed.path.endswith("/delete"):
                                deleted = broker.delete_battery_sighting(body.get("id"), body.get("controller_id"), viewer)
                                if not deleted:
                                    return self.send_json(404, {"error": "This local battery photo is no longer retained."})
                                return self.send_json(200, {"deleted": True, "id": body["id"]})
                            result = broker.battery_action(body.get("action"), body.get("controller_id"), viewer)
                        except PermissionError as error:
                            return self.send_json(403, {"error": str(error)})
                        except ValueError as error:
                            return self.send_json(400, {"error": str(error)})
                        except RuntimeError as error:
                            return self.send_json(409, {"error": str(error)})
                        return self.send_json(202, result)
                    if parsed.path == "/api/v2/device/sightings":
                        viewer = self.viewer_id()
                        if body.get("viewer_id") != viewer:
                            return self.send_json(403, {"error": "viewer_id must match X-Viewer-ID"})
                        try:
                            result = broker.set_sightings_enabled(body.get("enabled"), body.get("controller_id"), viewer)
                        except PermissionError as error:
                            return self.send_json(403, {"error": str(error)})
                        except ValueError as error:
                            return self.send_json(400, {"error": str(error)})
                        return self.send_json(200, result)
                    if parsed.path == "/api/v2/device/sightings/assign":
                        viewer = self.viewer_id()
                        if body.get("viewer_id") != viewer:
                            return self.send_json(403, {"error": "viewer_id must match X-Viewer-ID"})
                        if "person_id" not in body:
                            return self.send_json(400, {"error": "Choose a person or Other."})
                        try:
                            result = broker.assign_sighting(body.get("sighting_id"), body["person_id"],
                                                           body.get("controller_id"), viewer)
                        except PermissionError as error:
                            return self.send_json(403, {"error": str(error)})
                        except ValueError as error:
                            return self.send_json(400, {"error": str(error)})
                        except KeyError:
                            return self.send_json(404, {"error": "This sighting photo is no longer retained."})
                        return self.send_json(200, result)
                    if parsed.path in ("/api/v2/device/video/start", "/api/v2/device/video/stop"):
                        viewer = self.viewer_id()
                        if body.get("viewer_id") != viewer:
                            return self.send_json(403, {"error": "viewer_id must match X-Viewer-ID"})
                        try:
                            if parsed.path.endswith("/start"):
                                result = broker.start_video(body.get("controller_id"), viewer,
                                                            body.get("mode", "capture_only"))
                            else:
                                result = broker.stop_video(body.get("controller_id"), viewer)
                        except PermissionError as error:
                            return self.send_json(403, {"error": str(error)})
                        except RuntimeError as error:
                            return self.send_json(409, {"error": str(error)})
                        return self.send_json(200, result)
                    if parsed.path == "/api/v2/device/host-reference":
                        try:
                            result = app.run_device_host_reference(body.get("boot_id"), body.get("frame_id"))
                        except KeyError:
                            return self.send_json(404, {"error": "The selected frame is no longer retained. Capture a new frame."})
                        return self.send_json(200, result)
                    if parsed.path in ("/api/v2/device/connect", "/api/v2/device/disconnect"):
                        action = parsed.path.rsplit("/", 1)[-1]
                        schema = f"nicla-{action}-v2"
                        if body.get("schema", schema) != schema:
                            return self.send_json(400, {"error": f"schema must be {schema}"})
                        viewer = str(body.get("viewer_id", ""))
                        if viewer != self.viewer_id() or not re.fullmatch(r"[A-Za-z0-9_-]{1,128}", viewer):
                            return self.send_json(403, {"error": "viewer_id must match X-Viewer-ID"})
                        controller = str(body.get("controller_id", ""))
                        if not re.fullmatch(r"[A-Za-z0-9._:-]{1,128}", controller):
                            return self.send_json(403, {"status": "read_only", "error": "Claim camera control to change the USB connection."})
                        try:
                            lease = broker.heartbeat_controller(controller, viewer)
                        except (RuntimeError, PermissionError) as error:
                            return self.send_json(409, {"status": "read_only", "error": str(error)})
                        if lease.get("status") != "renewed":
                            return self.send_json(409, {"status": "read_only", "reason": lease.get("reason"),
                                "error": "Camera control expired or belongs to another tab."})
                        if action == "connect":
                            return self.send_json(200, broker.connect())
                        result = broker.disconnect()
                        return self.send_json(200, result if isinstance(result, dict) else {"ok": True, "message": "Broker disconnected; device processing was not stopped."})
                    if parsed.path == "/api/v2/device/controller/claim":
                        viewer = str(body.get("viewer_id", ""))
                        if not re.fullmatch(r"[A-Za-z0-9_-]{1,128}", viewer) or viewer != self.viewer_id():
                            return self.send_json(403, {"error": "viewer_id must match this tab's X-Viewer-ID header"})
                        try:
                            return self.send_json(200, broker.claim_controller(viewer))
                        except (RuntimeError, PermissionError) as error:
                            return self.send_json(409, {"status": "owned", "error": str(error)})
                    if parsed.path in ("/api/v2/device/controller/heartbeat", "/api/v2/device/controller/release"):
                        viewer = str(body.get("viewer_id", "")); controller = str(body.get("controller_id", ""))
                        if not re.fullmatch(r"[A-Za-z0-9_-]{1,128}", viewer) or viewer != self.viewer_id() or not re.fullmatch(r"[A-Za-z0-9._:-]{1,128}", controller):
                            return self.send_json(400, {"error": "invalid viewer_id/controller_id"})
                        try:
                            result = (broker.heartbeat_controller(controller, viewer)
                                      if parsed.path.endswith("/heartbeat") else broker.release_controller(controller, viewer))
                        except (RuntimeError, PermissionError) as error:
                            return self.send_json(409, {"status": "lease_lost", "error": str(error)})
                        if result.get("status") == "rejected":
                            return self.send_json(409, {"status": "lease_lost", "reason": result.get("reason"),
                                "error": "Camera control expired or belongs to another tab."})
                        return self.send_json(200, result)
                    if parsed.path == "/api/v2/device/commands":
                        viewer = str(body.get("viewer_id", self.viewer_id()))
                        if body.get("schema") != "nicla-command-v2":
                            return self.send_json(400, {"error": "schema must be nicla-command-v2"})
                        if viewer != self.viewer_id():
                            return self.send_json(403, {"error": "body viewer_id must match X-Viewer-ID"})
                        if not re.fullmatch(r"[A-Za-z0-9._:-]{1,128}", str(body.get("command_id", ""))) or not re.fullmatch(r"[a-z][a-z0-9_]{1,40}", str(body.get("cmd", ""))):
                            return self.send_json(400, {"error": "command_id or cmd is malformed"})
                        result = broker.command(body, viewer_id=viewer)
                        status_code = 202 if result.get("status") == "accepted" else 409 if result.get("status") in {"busy", "rejected"} else 200
                        return self.send_json(status_code, result)
                    return self.send_json(404, {"error": "unknown device broker route"})
                if parsed.path == "/api/v1/live/connect":
                    ok, result = app.connect_live(self.client_id())
                    return self.send_json(200 if ok else 503, result)
                if parsed.path == "/api/v1/live/disconnect":
                    app.disconnect_live(self.client_id())
                    return self.send_json(200, {"ok": True, "message": "Dashboard detached; Nicla and preview owner were not changed."})
                if parsed.path == "/api/v1/live/diagnostic/enable":
                    return self.send_json(200, app.enable_host_diagnostic(self.client_id()))
                if parsed.path == "/api/v1/live/diagnostic/disable":
                    return self.send_json(200, app.disable_host_diagnostic(self.client_id()))
                if parsed.path == "/api/v1/live/rgb-reference/enable":
                    return self.send_json(200, app.enable_host_rgb_reference(self.client_id()))
                if parsed.path == "/api/v1/live/rgb-reference/disable":
                    return self.send_json(200, app.disable_host_rgb_reference(self.client_id()))
                if parsed.path == "/api/v1/live/rgb-reference/run":
                    expected = body.get("host_observation_index")
                    result = app.run_host_rgb_reference(self.client_id(), expected)
                    return self.send_json(200, result)
                if parsed.path == "/api/v1/live/diagnostic/run":
                    expected = body.get("host_observation_index")
                    result = app.run_host_diagnostic(self.client_id(), expected)
                    return self.send_json(200, result)
                if parsed.path == "/api/v1/live/capture":
                    if app.client(self.client_id())["mode"] != "live":
                        return self.send_json(409, {"error": "select live preview mode first"})
                    jpeg = app.capture_live_snapshot(self.client_id())
                    observation = app.client(self.client_id())["observation_id"]
                    self.send_response(200)
                    self.send_header("Content-Type", "image/jpeg")
                    self.send_header("Content-Length", str(len(jpeg)))
                    self.send_header("Cache-Control", "no-store")
                    self.send_header("X-Preview-Observation", str(observation))
                    self.end_headers()
                    self.wfile.write(jpeg)
                    return
                if parsed.path == "/api/v1/replay/select":
                    case = str(body.get("case", ""))
                    if case not in app.rows:
                        return self.send_json(400, {"error": "case must be one of the reviewed replay cases"})
                    app.disconnect_live(self.client_id())
                    return self.send_json(200, app.replay_case(case, self.client_id()))
                if parsed.path == "/api/v1/mode":
                    mode = body.get("mode")
                    if mode not in {"live", "replay"}:
                        return self.send_json(400, {"error": "mode must be live or replay"})
                    client = app.client(self.client_id())
                    if mode == "live" and not client["live_attached"]:
                        return self.send_json(409, {"error": "attach to the existing preview before selecting live mode"})
                    if mode == "replay":
                        app.disconnect_live(self.client_id())
                    client["mode"] = mode
                    return self.send_json(200, app.live_state(self.client_id()) if mode == "live" else app.replay_case(client_id=self.client_id()))
                if parsed.path == "/api/v1/shutdown":
                    self.send_json(200, {"ok": True, "message": "Control-center HTTP service is shutting down. Preview owner is unchanged."})
                    server = server_ref.get("server")
                    if server:
                        request_server_shutdown(app, server, "dashboard request")
                    return
                return self.send_json(404, {"error": "not found"})
            except (ValueError, json.JSONDecodeError) as error:
                return self.send_json(400, {"error": f"{type(error).__name__}: {error}"})
            except (OSError, RuntimeError, urllib.error.URLError, HostWorkerError, HostRgbReferenceWorkerError) as error:
                status = 409 if str(error).startswith((
                    "attach to", "preview_detached", "preview_capture_busy", "host_diagnostic_not_opted_in",
                    "host_diagnostic_disabled", "worker_busy",
                    "host_rgb_reference_not_opted_in", "host_rgb_reference_disabled",
                    "host_observation_", "host_snapshot_", "preview_frame_stale_",
                    "invalid_host_observation_index",
                )) else 503 if "worker" in str(error) or "tensorflow_environment" in str(error) or "opencv_reference_environment" in str(error) else 502
                return self.send_json(status, {"error": f"{type(error).__name__}: {error}"})

    return Handler


def doctor(host: str, port: int, preview_url: str) -> int:
    errors: list[str] = []
    replay_available = RUN_RESULT.is_file() and all(path.is_file() for path in CASE_IMAGES.values())
    try:
        _loopback_preview_url(preview_url)
    except ValueError as error:
        errors.append(str(error))
    for name in ("index.html", "app.js", "device_ui.js", "power_ui.js", "app.css"):
        path = WEB_ROOT / name
        if not path.is_file():
            errors.append(f"missing required local resource: {path}")
    if host not in {"127.0.0.1", "localhost", "::1"}:
        errors.append("server bind must stay on loopback")
    if not 0 <= int(port) <= 65535:
        errors.append("port must be in [0, 65535]")
    if port:
        try:
            with socket.socket(socket.AF_INET6 if host == "::1" else socket.AF_INET, socket.SOCK_STREAM) as sock:
                sock.bind((host, int(port)))
        except OSError as error:
            errors.append(f"port {port} cannot be bound: {error}")
    print("Nicla Control Center doctor")
    print(f"  version: {__version__}")
    print("  device broker: NiclaBroker; serial opens only after explicit Connect by the controller lease owner")
    print(f"  pyserial: {'available' if importlib.util.find_spec('serial') else 'missing (live Connect unavailable)'}")
    print("  live install: python3 -m pip install -r FaceRecognition/requirements-host.txt")
    print(f"  replay data: {'available' if replay_available else 'unavailable; replay controls will be disabled, live mode can still start'}")
    print(f"  bind: {host}:{port}")
    if port == 0:
        print("  port discovery: OS selects an available port when launched; open the exact URL printed at startup")
    print(f"  existing preview URL: {preview_url}")
    print("  legacy preview adapter: optional HTTP attach to 127.0.0.1:8766; separate from device control")
    print("  broker port discovery: no hardware connection is attempted by doctor")
    if errors:
        for error in errors:
            print("  ERROR: " + error)
        return 1
    print("  result: ready")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1", help="loopback bind address only")
    parser.add_argument("--port", type=int, default=8780, help="HTTP port; 0 chooses an available port")
    parser.add_argument("--preview-url", default="http://127.0.0.1:8766", help="legacy read-only preview service; separate from NiclaBroker")
    parser.add_argument("--doctor", action="store_true", help="validate local files and bind settings, then exit")
    parser.add_argument("--open-browser", action="store_true", help="open the dashboard after start; avoid during optical capture")
    args = parser.parse_args(argv)
    if args.doctor:
        return doctor(args.host, args.port, args.preview_url)
    if args.host not in {"127.0.0.1", "localhost", "::1"}:
        parser.error("this service binds to loopback only")
    if not 0 <= args.port <= 65535:
        parser.error("port must be in [0, 65535]")
    app = ControlCenter(args.preview_url)
    server_ref: dict[str, ControlCenterHTTPServer] = {}
    try:
        server = ControlCenterHTTPServer((args.host, args.port), make_handler(app, server_ref))
    except OSError as error:
        app.close()
        if error.errno == errno.EADDRINUSE:
            print(f"Could not bind to port {args.port}: it is already in use. Retry with --port 0, then open the exact URL printed by that run.", file=sys.stderr)
        else:
            print(f"Could not start the loopback server on {args.host}:{args.port}: {error}", file=sys.stderr)
        return 2
    server_ref["server"] = server
    bound_host, port = server.server_address[:2]
    url_host = f"[{bound_host}]" if ":" in str(bound_host) else str(bound_host)
    url = f"http://{url_host}:{port}/"

    previous_signal_handlers = {}

    def handle_signal(signum, _frame):
        try:
            name = signal.Signals(signum).name
        except ValueError:
            name = str(signum)
        request_server_shutdown(app, server, name)

    for signum in (signal.SIGINT, signal.SIGTERM):
        previous_signal_handlers[signum] = signal.signal(signum, handle_signal)

    print(f"NICLA_CONTROL_CENTER {url}", flush=True)
    print("Open the exact URL above. If the default port is occupied, retry with --port 0 and use its new URL.", flush=True)
    print("Mode: RECORDED REPLAY when private data is available; otherwise live operator controls remain available.", flush=True)
    print("NiclaBroker stays disconnected. Serial opens only after explicit Connect by the active controller lease owner.", flush=True)
    print("Legacy preview adapter uses 127.0.0.1:8766 and does not expose device decisions or control commands.", flush=True)
    if args.open_browser:
        try:
            if not webbrowser.open(url):
                print("No browser accepted the open request; use the URL printed above.", flush=True)
        except Exception as error:
            print(f"Could not open a browser ({error}); use the URL printed above.", flush=True)
    try:
        server.serve_forever(poll_interval=0.2)
    except KeyboardInterrupt:
        request_server_shutdown(app, server, "keyboard interrupt")
    finally:
        try:
            app.close()
        finally:
            try:
                server.server_close()
            finally:
                for signum, previous_handler in previous_signal_handlers.items():
                    signal.signal(signum, previous_handler)
                print("Control center stopped. NiclaBroker disconnected; device processing was not stopped by broker close.", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
