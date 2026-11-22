"""Optional, passive Windows HID meter service for the local dashboard."""

from collections import deque
from concurrent.futures import Future, TimeoutError as FutureTimeout
from contextlib import ExitStack
from datetime import datetime, timezone
import hashlib
import importlib
import json
import os
from pathlib import Path
from queue import Empty, Queue
import sys
import threading
import time
from uuid import uuid4

from . import receipt_clock
from .upstream.power import fnirsi_protocol
from .upstream.power import run_io
from .upstream.scripts import hardware_lock_windows
from .upstream.scripts.device_selector_windows import describe


GAP_NS = 200_000_000
CLOCK_JUMP_NS = 100_000_000
STALE_NS = 1_000_000_000
PREVIEW_CAPACITY = 600
MAX_PAGE = 120
SCAN_WAIT_SECONDS = 1.0
ACTION_WAIT_SECONDS = 3.0
BOUNDARY = "Inline USB supply feeding the Nicla"
TOPOLOGY_ID = "Nicla-FNB-C2-inline-v1"


def _load_hid():
    return importlib.import_module("hid")


def _interfaces(hid):
    interfaces = {}
    for device in hid.enumerate(fnirsi_protocol.VID, fnirsi_protocol.PID):
        if not device.get("path"):
            continue
        descriptor = describe(device)
        identity = json.dumps(
            [descriptor, device.get("serial_number") or ""], sort_keys=True
        ).encode()
        descriptor["id"] = hashlib.sha256(identity).hexdigest()
        interfaces[descriptor["id"]] = (descriptor, device)
    return interfaces


class PowerMeter:
    """A single worker holds the HID handle and Windows mutex for each connection."""

    def __init__(
        self, project_root, recording_root=None, *, platform_name=None, hid_loader=None,
        lock_factory=None,
    ):
        self.project_root = Path(project_root).resolve()
        self.recording_root = Path(
            recording_root if recording_root is not None else self.project_root / "raw" / "dashboard-power"
        ).resolve()
        self.lock_path = hardware_lock_windows.default_lock_path(self.project_root)
        self._load_hid = hid_loader if hid_loader is not None else _load_hid
        self._lock_factory = lock_factory if lock_factory is not None else hardware_lock_windows.HardwareLock
        self._windows = (platform_name if platform_name is not None else os.name) == "nt"
        self._guard = threading.Lock()
        self._worker = None
        self._stop = threading.Event()
        self._commands = Queue()
        self._closed = False
        self._scan_in_progress = False
        self._state = "unknown" if self._windows else "unsupported"
        self._message = "Scan for the Windows HID meter." if self._windows else "Native meter capture is available on Windows."
        self._interfaces = []
        self._session_id = None
        self._opened_ns = None
        self._topology_confirmed = False
        self._cursor = 0
        self._points = deque(maxlen=PREVIEW_CAPACITY)
        self._recent_receipts = deque(maxlen=PREVIEW_CAPACITY)
        self._report_times = deque(maxlen=120)
        self._last_sample_ns = None
        self._recording = None
        self._record_started_ns = None
        self._stats = self._new_stats()

    @staticmethod
    def _new_stats():
        return dict(gap_count=0, invalid_count=0, crc_fail_count=0, timeout_count=0,
                    duplicate_count=0, clock_jump_count=0)

    def status(self):
        now = time.monotonic_ns()
        with self._guard:
            state, message = self._state, self._message
            age_ns = None if self._last_sample_ns is None else max(0, now - self._last_sample_ns)
            if state in ("connecting", "receiving") and self._opened_ns is not None:
                if now - (self._last_sample_ns or self._opened_ns) > STALE_NS:
                    state = "stale"
                    message = "No recent valid HID report. Check the meter and its PC/HID cable."
            recent = [stamp for stamp in self._report_times if stamp >= now - 2_000_000_000]
            rate = None
            if len(recent) > 1 and recent[-1] > recent[0]:
                rate = (len(recent) - 1) * 1_000_000_000 / (recent[-1] - recent[0])
            recording = None if self._recording is None else dict(self._recording)
            if recording and recording["state"] == "running":
                recording["duration_s"] = max(0, now - self._record_started_ns) / 1_000_000_000
            return {
                "state": state,
                "message": message,
                "connected": self._opened_ns is not None,
                "scan_in_progress": self._scan_in_progress,
                "interfaces": [dict(item) for item in self._interfaces],
                "session_id": self._session_id,
                "topology_confirmed": self._topology_confirmed,
                "boundary": BOUNDARY,
                "latest": dict(self._points[-1]) if self._points else None,
                "sample_age_ms": None if age_ns is None else age_ns / 1_000_000,
                "report_rate_hz": rate,
                "report_gap_limit_ms": GAP_NS // 1_000_000,
                "recording": recording,
                **self._stats,
            }

    def samples(self, after=0, limit=60):
        if type(after) is not int or after < 0:
            raise ValueError("after must be a nonnegative integer")
        if type(limit) is not int or not 1 <= limit <= MAX_PAGE:
            raise ValueError(f"limit must be between 1 and {MAX_PAGE}")
        with self._guard:
            reset = after > self._cursor or (
                bool(self._points) and 0 < after < self._points[0]["cursor"] - 1
            )
            if after == 0 or reset:
                points = list(self._points)[-limit:]
            else:
                points = [point for point in self._points if point["cursor"] > after][:limit]
            return {
                "samples": [dict(point) for point in points],
                "next_cursor": points[-1]["cursor"] if points else self._cursor,
                "session_id": self._session_id,
                "reset": reset,
            }

    def scan(self):
        if not self._windows:
            return self.status()
        done = threading.Event()
        with self._guard:
            if self._closed or self._worker is not None:
                raise RuntimeError("Disconnect the meter before scanning.")
            if not self._scan_in_progress:
                self._scan_in_progress = True
                self._message = "Scanning the Windows HID inventory."
                threading.Thread(target=self._scan, args=(done,), name="power-meter-scan", daemon=True).start()
            else:
                done.set()
        done.wait(SCAN_WAIT_SECONDS)
        return self.status()

    def _scan(self, done):
        interfaces = []
        try:
            interfaces = [row for row, _ in _interfaces(self._load_hid()).values()]
            state, message = self._inventory_status(interfaces)
        except ImportError:
            state, message = "dependency_missing", "Install the optional Windows hidapi dependency to use the meter."
        except Exception as error:
            state, message = "error", f"Meter enumeration failed: {error}"
        finally:
            with self._guard:
                self._interfaces = interfaces
                self._state, self._message = state, message
                self._scan_in_progress = False
            done.set()

    @staticmethod
    def _inventory_status(interfaces):
        if not interfaces:
            return "absent", "No matching FNIRSI HID interfaces found."
        if len(interfaces) > 1:
            return "ambiguous", "Choose the interface verified for your FNIRSI meter."
        return "available", "One matching HID interface found. Connect to begin receiving reports."

    def connect(self, interface_id):
        if not isinstance(interface_id, str) or not interface_id:
            raise ValueError("Select a meter interface from the current scan.")
        with self._guard:
            if not self._windows or self._closed:
                raise RuntimeError("The Windows meter service is unavailable.")
            if self._worker is not None:
                raise RuntimeError("Disconnect the existing meter session first.")
            if self._scan_in_progress:
                raise RuntimeError("Wait for the meter scan to complete.")
            if interface_id not in {row["id"] for row in self._interfaces}:
                raise ValueError("Select a meter interface from the current scan.")
            self._session_id = "power-" + uuid4().hex
            self._topology_confirmed = False
            self._points.clear()
            self._recent_receipts.clear()
            self._report_times.clear()
            self._stats = self._new_stats()
            self._last_sample_ns = None
            self._stop.clear()
            self._state, self._message = "connecting", "Acquiring the selected HID interface."
            self._worker = threading.Thread(
                target=self._capture, args=(interface_id, self._session_id),
                name="power-meter-owner", daemon=True,
            )
            self._worker.start()
        return self.status()

    def disconnect(self):
        with self._guard:
            worker = self._worker
            if worker is not None:
                self._state, self._message = "disconnecting", "Waiting for the HID handle and lock to close."
                self._stop.set()
        if worker is not None:
            worker.join(ACTION_WAIT_SECONDS)
            if worker.is_alive():
                raise RuntimeError("The meter worker is still releasing the HID handle and lock.")
        return self.status()

    def close(self):
        with self._guard:
            self._closed = True
        self.disconnect()

    def start_recording(self, topology_confirmed=False):
        if type(topology_confirmed) is not bool:
            raise ValueError("topology_confirmed must be a boolean")
        return self._record_action("start", topology_confirmed)

    def stop_recording(self):
        return self._record_action("stop", False)

    def _record_action(self, action, topology_confirmed):
        future = Future()
        with self._guard:
            if self._opened_ns is None or self._stop.is_set():
                raise RuntimeError("Connect the meter before changing the recording.")
            self._commands.put((action, topology_confirmed, future))
        try:
            future.result(timeout=ACTION_WAIT_SECONDS)
        except FutureTimeout as error:
            future.cancel()
            raise RuntimeError("The meter worker has not completed the recording request.") from error
        return self.status()

    def _capture(self, interface_id, session_id):
        state, message = self._inventory_status(self._interfaces)
        try:
            with ExitStack() as owned:
                try:
                    owned.enter_context(self._lock_factory(self.lock_path, session_id, 0.0))
                except TimeoutError:
                    state, message = "busy", "The project hardware lock is held. Stop the CLI capture before connecting."
                    return
                if self._stop.is_set():
                    return
                hid = self._load_hid()
                current = _interfaces(hid)
                with self._guard:
                    self._interfaces = [row for row, _ in current.values()]
                if self._stop.is_set():
                    return
                if interface_id not in current:
                    state, _ = self._inventory_status(self._interfaces)
                    message = "The selected HID descriptor changed. Scan and choose the current interface."
                    return
                descriptor, selected = current[interface_id]
                device = hid.device()
                owned.callback(device.close)
                device.open_path(selected["path"])
                init_written = self._write_command(device, 0x81)
                if self._stop.wait(0.05):
                    return
                with self._guard:
                    self._opened_ns = time.monotonic_ns()
                self._read_loop(device, descriptor, selected.get("serial_number"), init_written)
        except ImportError:
            state, message = "dependency_missing", "The optional Windows hidapi dependency is unavailable."
        except Exception as error:
            state, message = "error", f"Meter connection ended: {error}. Close the vendor app and scan before reconnecting."
        finally:
            with self._guard:
                self._opened_ns = None
                self._session_id = None
                self._topology_confirmed = False
                self._points.clear()
                self._report_times.clear()
                self._last_sample_ns = None
                self._state, self._message = state, message
                while not self._commands.empty():
                    _, _, future = self._commands.get_nowait()
                    if future.set_running_or_notify_cancel():
                        future.set_exception(RuntimeError("The meter disconnected before processing the request."))
                self._worker = None

    @staticmethod
    def _write_command(device, code):
        command = fnirsi_protocol.make_command(code)
        written = device.write([0, *command])
        if written not in (64, 65):
            raise OSError(f"HID write for 0x{code:02x} returned {written}; expected 64 or 65")
        return written

    def _read_loop(self, device, descriptor, serial_number, init_written):
        writer = None
        prior_receipt = None
        prior_payload = None
        break_pending = True
        arrival_index = 0
        start_ns = time.monotonic_ns()
        try:
            while not self._stop.is_set():
                writer = self._process_record_actions(writer, descriptor, serial_number, init_written)
                self._write_command(device, 0x82)
                payload = bytes(device.read(fnirsi_protocol.REPORT_LEN, timeout_ms=250))
                mono_ns, utc_ns = time.monotonic_ns(), receipt_clock.host_utc_ns()
                utc = receipt_clock.iso_utc_ns(utc_ns)
                with self._guard:
                    # Preview receipts roll in memory. During Record, the same
                    # receipts are persisted before any display filtering.
                    self._recent_receipts.append((arrival_index, mono_ns, utc, payload))
                if not payload:
                    if writer is not None:
                        writer.record_timeout(writer.report_count + writer.invalid_count, mono_ns, utc)
                    with self._guard:
                        self._stats["timeout_count"] += 1
                    break_pending = True
                    continue
                gap = clock_jump = False
                if prior_receipt is not None:
                    elapsed_ns = mono_ns - prior_receipt[0]
                    gap = elapsed_ns > GAP_NS
                    clock_jump = abs(utc_ns - prior_receipt[1] - elapsed_ns) > CLOCK_JUMP_NS
                if writer is not None:
                    record_index = writer.report_count + writer.invalid_count
                    packet = writer.record_report(payload, record_index, mono_ns, utc, gap)
                    slots, crc_failed = packet["samples"], packet["crc_status"] == "fail"
                    if clock_jump:
                        writer.event("clock_jump", {"report_arrival_index": record_index})
                else:
                    try:
                        slots, crc_failed = fnirsi_protocol.decode_report(payload), False
                    except fnirsi_protocol.PacketDecodeError as error:
                        slots, crc_failed = [], str(error) == "crc_mismatch"
                with self._guard:
                    self._report_times.append(mono_ns)
                    self._stats["gap_count"] += int(gap)
                    self._stats["clock_jump_count"] += int(clock_jump)
                    self._stats["duplicate_count"] += int(payload == prior_payload)
                    if slots:
                        self._append_point(slots, mono_ns, utc, start_ns, break_pending or gap or clock_jump)
                    else:
                        self._stats["invalid_count"] += 1
                        self._stats["crc_fail_count"] += int(crc_failed)
                break_pending = not slots
                prior_receipt = (mono_ns, utc_ns)
                prior_payload = payload
                arrival_index += 1
        except Exception as error:
            if writer is not None:
                self._finish_recording(writer, "failed", str(error))
            raise
        else:
            if writer is not None:
                self._finish_recording(writer, "interrupted", "Meter disconnected while recording.")

    def _append_point(self, slots, mono_ns, utc, start_ns, gap_before):
        summary = fnirsi_protocol.summarize_report(slots)
        self._cursor += 1
        self._points.append({
            "cursor": self._cursor,
            "elapsed_s": (mono_ns - start_ns) / 1_000_000_000,
            "host_receive_utc": utc,
            "voltage_v": summary["voltage_mean_v"],
            "current_ma": 1000 * summary["current_mean_a"],
            "power_w": summary["power_mean_w"],
            "gap_before": gap_before,
        })
        self._last_sample_ns = mono_ns
        if not self._stop.is_set():
            self._state, self._message = "receiving", "Receiving FNIRSI HID reports."

    def _process_record_actions(self, writer, descriptor, serial_number, init_written):
        # Process at most one command between physical HID reads.
        try:
            action, topology_confirmed, future = self._commands.get_nowait()
        except Empty:
            return writer
        if not future.set_running_or_notify_cancel():
            return writer
        try:
            if action == "start":
                if writer is not None:
                    raise RuntimeError("A meter recording is already running.")
                writer = self._start_recording(topology_confirmed, descriptor, serial_number, init_written)
            else:
                if writer is None:
                    raise RuntimeError("No meter recording is running.")
                active_writer, writer = writer, None
                if active_writer.report_count:
                    self._finish_recording(active_writer, "complete")
                else:
                    self._finish_recording(active_writer, "failed", "No valid HID reports received while recording.")
        except Exception as error:
            future.set_exception(error if isinstance(error, RuntimeError) else RuntimeError(str(error)))
        else:
            future.set_result(None)
        return writer

    def _start_recording(self, topology_confirmed, descriptor, serial_number, init_written):
        run_id = "dashboard-power-" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S") + "-" + uuid4().hex[:12]
        output_dir = self.recording_root / run_id
        artifact_paths = {
            "dashboard_meter_sha256": Path(__file__),
            "windows_lock_sha256": Path(hardware_lock_windows.__file__),
            "decoder_sha256": Path(fnirsi_protocol.__file__),
            "run_io_sha256": Path(run_io.__file__),
            "receipt_clock_sha256": Path(receipt_clock.__file__),
        }
        metadata = {
            "run_id": run_id,
            "experiment_id": "windows-fnirsi-dashboard",
            "scenario_id": "dashboard-power-preview",
            "policy_id": "diagnostic-only",
            "capture_kind": "dashboard_power",
            "meter_vid_pid": f"{fnirsi_protocol.VID:04x}:{fnirsi_protocol.PID:04x}",
            "meter_model_reported": descriptor["product"],
            "meter_firmware": "unverified; see HID release_number",
            "meter_hid_descriptor": {**descriptor, "serial_number": serial_number},
            "fresh_interface_selection": True,
            "topology": BOUNDARY if topology_confirmed else "unconfirmed",
            "topology_id": TOPOLOGY_ID if topology_confirmed else None,
            "topology_confirmation": "operator_confirmation" if topology_confirmed else "unconfirmed",
            "source_input_paths_sha256": {
                key: hashlib.sha256(path.read_bytes()).hexdigest()
                for key, path in artifact_paths.items()
            },
            "host": {"platform": sys.platform, "python": sys.version, "pid": os.getpid()},
            "read_size_bytes": fnirsi_protocol.REPORT_LEN,
            "hid_write_report_id_prefix": 0,
            "hid_read_convention": "raw HIDAPI payload without report-ID transformation",
            "acquisition_time_source": "host monotonic and UTC at HID read completion",
            "host_utc_clock": receipt_clock.UTC_SOURCE,
            "measurement_claim": "dashboard diagnostic capture; no qualified energy calculation",
            "hid_session_id": self._session_id,
        }
        writer = run_io.RunWriter(output_dir, metadata)
        try:
            writer.event("meter_already_open", {"selected_interface": descriptor,
                                                "protocol_commands": ["0x81 init", "0x82 read"],
                                                "init_hidapi_write_return": init_written})
        except Exception as error:
            writer.close("failed", str(error))
            raise
        with self._guard:
            self._record_started_ns = time.monotonic_ns()
            self._recording = {
                "run_id": run_id, "state": "running", "duration_s": 0.0,
                "output_dir": str(output_dir), "error": None,
            }
            self._topology_confirmed = topology_confirmed
        return writer

    def _finish_recording(self, writer, state, error=None):
        try:
            writer.close(state, error)
        except Exception as close_error:
            state, error = "failed", str(close_error)
            raise
        finally:
            with self._guard:
                self._recording.update(
                    state=state, error=error,
                    duration_s=(time.monotonic_ns() - self._record_started_ns) / 1_000_000_000,
                )
