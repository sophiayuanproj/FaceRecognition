#!/usr/bin/env python3
"""Sole-owner, bounded host broker for the Nicla control/event stream.

The broker holds the USB CDC handle, parses bounded JSONL and NVPK1 packets,
and exposes immutable observation/command events to the local control center.
It never stops the device loop when a browser disconnects.
"""

from __future__ import annotations

from collections import OrderedDict, deque
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
from queue import Empty, Full, Queue
import re
from sqlite3 import Error as SQLiteError
import threading
import time
import secrets

from .nicla_battery import NiclaBatteryService
from .nicla_clip import write_clip
from .nicla_sightings import NiclaSightingsStore

_NICLA_VID = 0x2341
_NICLA_PID = 0x045F
_NICLA_PRODUCT = "nicla vision"
_COMMAND_ID = re.compile(r"^[A-Za-z0-9_-]{1,64}$")
_READ_ONLY = {"discover", "identity", "config", "ping", "status"}
_SUPPORTED_COMMANDS = ("discover", "config", "status", "one_shot", "start", "stop",
                       "set_policy", "set_model", "sleep", "wake", "preload_model", "trial")
_TERMINAL_ACKS = {"completed", "error", "unsupported", "rejected", "timeout"}
_VIDEO_DURATION_MS = 12000
_VIDEO_FRAMES = 50
_LIVE_VIDEO_DURATION_MS = 30000
_LIVE_VIDEO_FRAMES = 180
_VIDEO_BYTES = 16 * 1024 * 1024
_VIDEO_STAGES = {"capture_only": "camera_capture_only", "inference": "full_pipeline"}
_VIDEO_ACTIVE = {"starting", "recording"}
_VIDEO_ID = re.compile(r"^\d{8}T\d{6}Z-[0-9a-f]{8}$")
_SIGHTING_CLEAR_NS = 2_000_000_000
_OPENMV_DTYPE_CODES = {98: "int8", 66: "uint8", 104: "int16", 72: "uint16", 102: "float32"}
_OPENMV_DTYPE_CHARS = {"b": "int8", "B": "uint8", "h": "int16", "H": "uint16", "f": "float32"}


def _canonical_packet_dtype(metadata):
    """Normalize OpenMV's byte-valued dtype code for the API/UI contract."""
    result = dict(metadata)
    value = result.get("dtype")
    code = value if type(value) is int else None
    if code is None and isinstance(value, str) and value.isdecimal():
        try:
            code = int(value)
        except ValueError:
            code = None
    normalized = _OPENMV_DTYPE_CODES.get(code)
    if normalized is None and isinstance(value, bytes):
        try:
            value = value.decode("ascii", "strict")
        except UnicodeDecodeError:
            value = None
    if normalized is None and isinstance(value, str):
        normalized = _OPENMV_DTYPE_CHARS.get(value)
    if normalized is not None:
        result["dtype_wire_value"] = result.get("dtype")
        result["dtype"] = normalized
    return result


class NiclaBrokerError(RuntimeError):
    pass


class NiclaPortNotFound(NiclaBrokerError):
    pass


class AmbiguousNiclaPort(NiclaBrokerError):
    def __init__(self, candidates):
        self.candidates = candidates
        super().__init__("More than one Nicla Vision serial port is attached; select one by port or serial number")


class NiclaPortBusy(NiclaBrokerError):
    pass


class NiclaStreamParser:
    """Parse JSONL records plus bounded `NVPK1 + u16 header length + JSON + bytes`."""

    MAGIC = b"NVPK1"
    HEADER_PREFIX_BYTES = 7
    HEADER_MAX_BYTES = 1024
    PACKET_MAX_BYTES = 65536
    JSON_LINE_MAX_BYTES = 8192
    BUFFER_MAX_BYTES = 2 * (PACKET_MAX_BYTES + HEADER_MAX_BYTES + HEADER_PREFIX_BYTES)

    def __init__(self):
        self.buffer = bytearray()
        self.packet_started_at = None
        self.errors = 0

    @staticmethod
    def _reject_constant(value):
        raise ValueError("non_finite_json_constant:" + value)

    def feed(self, data: bytes, now: float | None = None):
        now = time.monotonic() if now is None else float(now)
        self.buffer.extend(data)
        if len(self.buffer) > self.BUFFER_MAX_BYTES:
            self.errors += 1
            marker = self.buffer.rfind(self.MAGIC)
            newline = self.buffer.rfind(b"\n")
            start = max(marker, newline + 1)
            if start <= 0 or start >= len(self.buffer):
                self.buffer.clear()
            else:
                del self.buffer[:start]
            self.packet_started_at = None
        out = []
        while self.buffer:
            if self.buffer.startswith(self.MAGIC):
                if len(self.buffer) < self.HEADER_PREFIX_BYTES:
                    self.packet_started_at = self.packet_started_at or now
                    break
                header_bytes = self.buffer[5] | (self.buffer[6] << 8)
                if header_bytes <= 0 or header_bytes > self.HEADER_MAX_BYTES:
                    self.errors += 1
                    prefix_hex = bytes(self.buffer[:48]).hex()
                    del self.buffer[0]
                    self.packet_started_at = None
                    out.append({"type": "parse_error", "reason_code": "packet_header_length_invalid"})
                    out[-1]["prefix_hex"] = prefix_hex
                    continue
                prefix_bytes = self.HEADER_PREFIX_BYTES + header_bytes
                if len(self.buffer) < prefix_bytes:
                    self.packet_started_at = self.packet_started_at or now
                    break
                try:
                    header = json.loads(bytes(self.buffer[7:prefix_bytes]).decode("utf-8"),
                                        parse_constant=self._reject_constant)
                except (ValueError, UnicodeDecodeError):
                    self.errors += 1
                    del self.buffer[0]
                    self.packet_started_at = None
                    out.append({"type": "parse_error", "reason_code": "packet_header_invalid"})
                    continue
                if not isinstance(header, dict) or header.get("schema_id") != "nicla-packet-v1":
                    self.errors += 1
                    del self.buffer[0]
                    self.packet_started_at = None
                    out.append({"type": "parse_error", "reason_code": "packet_schema_invalid"})
                    continue
                byte_count = header.get("byte_count")
                checksum = header.get("sha256")
                if type(byte_count) is not int or not 0 < byte_count <= self.PACKET_MAX_BYTES:
                    self.errors += 1
                    del self.buffer[0]
                    self.packet_started_at = None
                    out.append({"type": "parse_error", "reason_code": "packet_size_invalid"})
                    continue
                if not isinstance(checksum, str) or not re.fullmatch(r"[0-9a-f]{64}", checksum):
                    self.errors += 1
                    del self.buffer[0]
                    self.packet_started_at = None
                    out.append({"type": "parse_error", "reason_code": "packet_hash_invalid"})
                    continue
                total = prefix_bytes + byte_count
                if len(self.buffer) < total:
                    self.packet_started_at = self.packet_started_at or now
                    break
                payload = bytes(self.buffer[prefix_bytes:total])
                del self.buffer[:total]
                self.packet_started_at = None
                actual = hashlib.sha256(payload).hexdigest()
                if actual != checksum:
                    self.errors += 1
                    out.append({"type": "packet_error", "reason_code": "packet_sha256_mismatch",
                                "header": header, "actual_sha256": actual})
                else:
                    out.append({"type": "packet", "header": header, "data": payload})
                continue

            newline = self.buffer.find(b"\n")
            marker = self.buffer.find(self.MAGIC)
            first_nonspace = next((value for value in self.buffer
                                   if value not in (9, 10, 13, 32)), None)
            line_like = first_nonspace == ord("{")
            # `NVPK1` may legitimately occur inside the JSON telemetry value
            # `payload_transport`; a JSON line beginning with `{` owns every
            # byte through its newline. Search for packet magic only when the
            # stream is not currently a JSON record.
            if newline >= 0 and (line_like or marker < 0 or newline < marker):
                raw = bytes(self.buffer[:newline + 1])
                del self.buffer[:newline + 1]
                self.packet_started_at = None
                if not raw.strip():
                    continue
                try:
                    value = json.loads(raw.decode("utf-8"), parse_constant=self._reject_constant)
                except (ValueError, UnicodeDecodeError):
                    self.errors += 1
                    out.append({"type": "parse_error", "reason_code": "json_line_invalid",
                                "raw_sha256": hashlib.sha256(raw).hexdigest(),
                                "byte_count": len(raw), "raw_prefix_hex": raw[:48].hex()})
                    continue
                if not isinstance(value, dict):
                    self.errors += 1
                    out.append({"type": "parse_error", "reason_code": "json_record_not_object"})
                    continue
                out.append({"type": "record", "record": value})
                continue
            if marker >= 0 and not line_like:
                # A partial line/noise before a frame marker is not allowed to
                # grow without bound. Preserve a complete JSON line above;
                # otherwise resynchronize at the packet's fixed magic bytes.
                if marker:
                    self.errors += 1
                    del self.buffer[:marker]
                    continue
            if len(self.buffer) > self.JSON_LINE_MAX_BYTES:
                self.errors += 1
                newline = self.buffer.find(b"\n")
                if newline >= 0:
                    del self.buffer[:newline + 1]
                else:
                    del self.buffer[:-len(self.MAGIC) + 1]
                out.append({"type": "parse_error", "reason_code": "json_line_oversize"})
                continue
            break
        return out

    def expire(self, now: float | None = None, timeout_s: float = 8.0):
        now = time.monotonic() if now is None else float(now)
        if self.packet_started_at is None or now - self.packet_started_at < timeout_s:
            return None
        byte_count = len(self.buffer)
        self.buffer.clear()
        self.packet_started_at = None
        self.errors += 1
        return {"type": "parse_error", "reason_code": "packet_timeout_or_truncated",
                "buffered_bytes": byte_count}


class NiclaBroker:
    """One active Nicla USB owner with bounded history and explicit viewer control."""

    def __init__(self, port=None, serial_number=None, *, event_capacity=512,
                 frame_capacity=32, controller_lease_s=15.0, serial_module=None,
                 lock_path=None, hello_timeout_s=6.0, monotonic=time.monotonic, monotonic_ns=time.monotonic_ns,
                 utc_now=None):
        self.requested_port = str(port) if port else None
        self.requested_serial_number = str(serial_number) if serial_number else None
        self.event_capacity = max(16, min(int(event_capacity), 4096))
        self.frame_capacity = max(2, min(int(frame_capacity), 128))
        self.controller_lease_s = max(5.0, min(float(controller_lease_s), 60.0))
        self.hello_timeout_s = max(0.05, min(float(hello_timeout_s), 30.0))
        self.serial_module = serial_module
        self.monotonic = monotonic
        self.monotonic_ns = monotonic_ns
        self.utc_now = utc_now or (lambda: datetime.now(timezone.utc).isoformat(timespec="milliseconds"))
        workspace = Path(__file__).resolve().parents[2]
        self.lock_path = Path(lock_path) if lock_path else workspace / "work/project-research-v5/private/nicla_device_serial.lock"
        self._lock = threading.RLock()
        self._write_lock = threading.Lock()
        self._reader = None
        self._stop = threading.Event()
        self._serial = None
        self._lock_file = None
        self._connected = False
        self._device_ready = False
        self._connecting = False
        self._connection_error = None
        self._connection_warning = None
        self._descriptor = None
        self._serial_bytes_read = 0
        self._serial_bytes_written = 0
        self._last_receive_ns = None
        self._last_receive_utc = None
        self._event_seq = 0
        self._events = deque(maxlen=self.event_capacity)
        self._frames = OrderedDict()
        self._latest_boot_id = None
        self._latest_observation = None
        self._latest_frame_id = None
        self._latest_frame_received_ns = None
        self._device = {}
        self._applied = {"running": None, "sleeping": None, "policy_id": None,
                         "model_override": None, "measurement_stage_id": None,
                         "model_slot": None, "trial_active": False,
                         "last_command_id": None}
        self._capabilities = {"protocol": None, "commands": list(_SUPPORTED_COMMANDS),
                              "models": [], "policies": [], "measurement_stage_ids": [],
                              "max_trial_frames": None, "max_trial_duration_ms": None,
                              "camera_sleep_supported": None}
        self._controller_id = None
        self._controller_viewer_id = None
        self._controller_expiry = 0.0
        self._pending = {}
        self._command_cache = OrderedDict()
        self._parser = NiclaStreamParser()
        self.video_root = self.lock_path.parent / "nicla-video-captures"
        self._last_video = self._last_saved_video()
        self._video = self._last_video
        self._sightings = NiclaSightingsStore(self.lock_path.parent / "nicla-sightings.sqlite3")
        self._battery = NiclaBatteryService(self, self._sightings)
        self._sightings_enabled = False
        self._sightings_writer = None
        self._sightings_queue = Queue(maxsize=8)
        self._sightings_shutdown = threading.Event()
        self._sightings_error = None
        self._reset_sighting()

    def _last_saved_video(self):
        if not self.video_root.is_dir():
            return None
        for directory in sorted(self.video_root.iterdir(), reverse=True):
            if not _VIDEO_ID.fullmatch(directory.name) or not (directory / "clip.avi").is_file():
                continue
            try:
                receipt = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
                count = receipt["frame_count"]
                if receipt["clip_id"] != directory.name or type(count) is not int or count < 1:
                    continue
                return {"state": "ready", "id": directory.name, "frames": count,
                        "mode": receipt.get("recording_mode", "capture_only"), "reason": receipt.get("reason"),
                        "matched_observations": receipt.get("matched_observations", 0),
                        "missed_frames": receipt.get("missed_frames", 0)}
            except (OSError, ValueError, KeyError, TypeError):
                continue
        return None

    @staticmethod
    def _descriptor_row(descriptor):
        return {"device": str(getattr(descriptor, "device", "")),
                "vid": getattr(descriptor, "vid", None),
                "pid": getattr(descriptor, "pid", None),
                "serial_number": getattr(descriptor, "serial_number", None),
                "product": getattr(descriptor, "product", None),
                "manufacturer": getattr(descriptor, "manufacturer", None),
                "hwid": str(getattr(descriptor, "hwid", ""))[:160]}

    def _list_ports(self):
        if self.serial_module is None:
            try:
                import serial as serial_module
                self.serial_module = serial_module
            except ImportError as error:
                raise NiclaBrokerError("pyserial is required; install FaceRecognition/requirements-host.txt") from error
        try:
            from serial.tools import list_ports
        except ImportError as error:
            raise NiclaBrokerError("pyserial list_ports is unavailable") from error
        return list(list_ports.comports())

    def discover(self):
        candidates = []
        all_ports = self._list_ports()
        for descriptor in all_ports:
            row = self._descriptor_row(descriptor)
            product = str(row.get("product") or "").casefold()
            # Windows usbser exposes the USB serial and hardware ID but can omit iProduct.
            windows_usbser = os.name == "nt" and not product and bool(row["serial_number"])
            if (row["vid"], row["pid"]) == (_NICLA_VID, _NICLA_PID) and (
                _NICLA_PRODUCT in product or windows_usbser
            ):
                candidates.append((descriptor, row))
        selected = None
        if self.requested_port:
            matching = [(d, r) for d, r in candidates if r["device"].casefold() == self.requested_port.casefold()]
            if len(matching) == 1:
                selected = matching[0]
            elif not matching:
                raise NiclaPortNotFound("Selected port is not a fresh Arduino Nicla Vision descriptor: " + self.requested_port)
        elif self.requested_serial_number:
            matching = [(d, r) for d, r in candidates if r["serial_number"] == self.requested_serial_number]
            if len(matching) == 1:
                selected = matching[0]
            elif not matching:
                raise NiclaPortNotFound("No Nicla Vision with serial number " + self.requested_serial_number)
        if selected is None:
            if len(candidates) == 1:
                selected = candidates[0]
            elif not candidates:
                raise NiclaPortNotFound("No Arduino Nicla Vision (VID 2341 / PID 045F) is attached")
            else:
                raise AmbiguousNiclaPort([row for _descriptor, row in candidates])
        return selected[1]

    def _acquire_port_lock(self, serial_number):
        self.lock_path.parent.mkdir(parents=True, exist_ok=True)
        handle = self.lock_path.open("a+", encoding="utf-8")
        try:
            handle.seek(0)
            try:
                try:
                    import fcntl
                except ImportError:
                    import msvcrt
                    msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
                else:
                    fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            except OSError as error:
                raise NiclaPortBusy("Another Nicla serial broker owns this port") from error
            handle.seek(0)
            handle.truncate()
            handle.write(json.dumps({"pid": os.getpid(), "serial_number": serial_number,
                                     "acquired_utc": self.utc_now()}) + "\n")
            handle.flush()
            self._lock_file = handle
        except Exception:
            handle.close()
            raise

    def _release_port_lock(self):
        handle = self._lock_file
        self._lock_file = None
        if handle is None:
            return
        try:
            try:
                import fcntl
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
            except ImportError:
                pass
        finally:
            handle.close()

    def connect(self):
        with self._lock:
            if self._connected:
                return self.status()
            if self._connecting:
                return {"status": "connecting"}
            self._connecting = True
            self._connection_error = None
            self._connection_warning = None
            self._device_ready = False
            # Never treat a prior boot/frame as a hello for this USB session.
            self._latest_boot_id = None
            self._latest_observation = None
            self._latest_frame_id = None
            self._latest_frame_received_ns = None
            self._device = {}
            self._applied.update({"running": None, "sleeping": None,
                                  "policy_id": None, "model_override": None,
                                  "measurement_stage_id": None, "model_slot": None,
                                  "trial_active": False})
        try:
            descriptor = self.discover()
            self._acquire_port_lock(descriptor.get("serial_number"))
            serial = self.serial_module.Serial(port=None, baudrate=115200, timeout=0.05,
                                               write_timeout=1, dsrdtr=False, rtscts=False)
            try:
                # USB_VCP.isconnected() on the Nicla runtime follows the host
                # CDC DTR state. Keep RTS low and assert DTR after open, matching
                # the existing board_lab transport; leaving DTR low makes the
                # board silently discard telemetry and USB control commands.
                serial.dtr = False
                serial.rts = False
                serial.port = descriptor["device"]
                serial.open()
                serial.dtr = True
            except Exception:
                serial.close()
                raise
            with self._lock:
                self._serial = serial
                self._descriptor = descriptor
                self._connected = True
                self._connecting = False
                self._connection_error = None
                self._connection_warning = None
                self._stop.clear()
                self._parser = NiclaStreamParser()
                self._reader = threading.Thread(target=self._read_loop,
                    name="nicla-usb-owner", daemon=True)
                self._reader.start()
            self._publish({"type": "connection", "connected": False,
                           "serial_open": True, "device_ready": False,
                           "device": descriptor, "protocol": "nicla-face-control-2",
                           "dtr_asserted": True, "rts_asserted": False})
            # DTR may trigger a board soft reset. Wait for an actual boot/frame
            # identity before probing control, so commands cannot be lost in
            # startup and a port-open alone is never mistaken for a connection.
            hello_deadline = self.monotonic() + self.hello_timeout_s
            while self.monotonic() < hello_deadline:
                with self._lock:
                    hello_seen = bool(self._latest_boot_id or self._latest_observation)
                    still_connected = self._connected
                if hello_seen or not still_connected:
                    break
                time.sleep(0.05)
            with self._lock:
                hello_seen = bool(self._latest_boot_id or self._latest_observation)
            if hello_seen:
                for name in ("discover", "config", "status"):
                    self.command({"cmd": name, "command_id":
                        "connect_%s_%s" % (name, str(int(self.monotonic() * 1000)))})
            else:
                warning = "nicla_hello_or_frame_timeout_after_usb_dtr_assert"
                self._connection_warning = warning
                self._publish({"type": "connection_warning", "reason_code":
                    warning})
                # A failed handshake must release CDC and the cross-process
                # lock. Otherwise the UI sees DISCONNECTED while this broker
                # silently owns the port, preventing retry or recovery.
                self.disconnect()
                with self._lock:
                    self._connection_warning = warning
            return self.status()
        except Exception as error:
            with self._lock:
                self._connected = False
                self._device_ready = False
                self._connecting = False
                self._connection_error = str(error)[:240]
            self._release_port_lock()
            raise

    def disconnect(self):
        """Close only the host USB handle; do not send a device stop command."""
        with self._lock:
            serial = self._serial
            self._serial = None
            self._connected = False
            self._device_ready = False
            self._connection_error = None
            self._connection_warning = None
            self._stop.set()
            reader = self._reader
            self._reader = None
            self._finish_video("disconnected")
            self._fail_pending("device_disconnected")
            self._reset_sighting()
        if serial is not None:
            try:
                serial.close()
            except Exception:
                pass
        if reader is not None and reader is not threading.current_thread():
            reader.join(timeout=1.0)
        self._release_port_lock()
        self._publish({"type": "connection", "connected": False,
                       "reason": "host_disconnect_does_not_stop_device_loop"})
        return self.status()

    def close(self):
        self.disconnect()
        self._battery.close()
        with self._lock:
            writer = self._video.get("writer") if self._video else None
        if writer:
            writer.join(timeout=5.0)
        self._sightings_shutdown.set()
        if self._sightings_writer:
            self._sightings_writer.join(timeout=5.0)

    def _read_loop(self):
        serial = None
        while not self._stop.is_set():
            with self._lock:
                serial = self._serial
                connected = self._connected
            if not serial or not connected:
                break
            try:
                waiting = int(getattr(serial, "in_waiting", 0))
                data = serial.read(min(max(waiting, 1), 65536))
            except Exception as error:
                with self._lock:
                    self._connection_error = str(error)[:240]
                    self._connected = False
                    self._device_ready = False
                self._publish({"type": "connection_error", "error": self._connection_error})
                break
            receive_ns = self.monotonic_ns()
            receive_utc = self.utc_now()
            if data:
                self._serial_bytes_read += len(data)
                self._last_receive_ns = receive_ns
                self._last_receive_utc = receive_utc
                for item in self._parser.feed(bytes(data), self.monotonic()):
                    self._handle_stream_item(item, receive_ns, receive_utc)
            expired = self._parser.expire(self.monotonic())
            if expired:
                self._publish({"type": "protocol_error", **expired})
            self._expire_commands()
        with self._lock:
            if serial is not None and self._serial is serial:
                self._connected = False
                self._device_ready = False
                self._serial = None
                self._fail_pending("device_disconnected")
                reader_failed = not self._stop.is_set()
            else:
                reader_failed = False
        if reader_failed:
            self._finish_video("disconnected")
            with self._lock:
                self._reset_sighting()
            try:
                serial.close()
            except Exception:
                pass
            self._release_port_lock()

    def _publish(self, event):
        row = dict(event)
        with self._lock:
            self._event_seq += 1
            row.setdefault("host_received_utc", self.utc_now())
            row.setdefault("host_received_monotonic_ns", self.monotonic_ns())
            row["event_seq"] = self._event_seq
            self._events.append(row)
        return row

    def _ensure_frame_group(self, boot_id, frame_id):
        key = (str(boot_id), int(frame_id))
        group = self._frames.setdefault(key, {"boot_id": key[0], "frame_id": key[1], "payloads": {}})
        self._frames.move_to_end(key)
        while len(self._frames) > self.frame_capacity:
            self._frames.popitem(last=False)
        return key, group

    def _reset_sighting(self, boot_id=None):
        self._sighting_boot = boot_id
        self._sighting_last_frame = -1
        self._sighting_saved = None
        self._sighting_known = False
        self._sighting_absent_since = None
        self._sighting_pending = None
        self._sighting_previous_hit = None

    def sightings_status(self):
        with self._lock:
            enabled, error = self._sightings_enabled, self._sightings_error
        try:
            result = self._sightings.summary()
        except (OSError, SQLiteError, ValueError) as cause:
            raise NiclaBrokerError("Could not read the local sightings database.") from cause
        items = result["items"]
        for row in items:
            row["utc"] = row.get("host_received_utc")
            row["image_url"] = f"/api/v2/device/sightings/frame?id={row['id']}"
        return {**result, "enabled": enabled, "error": error,
                "total_encounters": result["total_count"], "retained_bytes": result["retained_jpeg_bytes"],
                "max_entries": self._sightings.max_items, "max_jpeg_bytes": self._sightings.max_jpeg_bytes}

    def set_sightings_enabled(self, enabled, controller_id, viewer_id):
        if type(enabled) is not bool:
            raise ValueError("Choose whether to save sightings.")
        with self._lock:
            if not self._controller_valid(controller_id, viewer_id):
                raise PermissionError("Claim camera control to change sighting capture.")
            if enabled != self._sightings_enabled:
                self._reset_sighting()
                self._sightings_enabled = enabled
                self._sightings_error = None
        return self.sightings_status()

    def get_sighting_jpeg(self, sighting_id):
        try:
            jpeg, _ = self._sightings.get_jpeg(sighting_id)
        except (OSError, SQLiteError, ValueError) as cause:
            raise NiclaBrokerError("Could not read the saved sighting photo.") from cause
        return jpeg

    def assign_sighting(self, sighting_id, person_id, controller_id, viewer_id):
        with self._lock:
            if not self._controller_valid(controller_id, viewer_id):
                raise PermissionError("Claim camera control to file a saved sighting.")
        try:
            return self._sightings.assign_person(sighting_id, person_id)
        except (OSError, SQLiteError) as cause:
            raise NiclaBrokerError("Could not file the saved sighting.") from cause

    def battery_status(self, viewer_id=None):
        return self._battery.status(viewer_id)

    def battery_action(self, action, controller_id, viewer_id):
        return self._battery.action(action, controller_id, viewer_id)

    def delete_battery_sighting(self, sighting_id, controller_id, viewer_id):
        if type(sighting_id) is not int or not 1 <= sighting_id < 1 << 63:
            raise ValueError("Invalid battery photo ID.")
        with self._lock:
            if not self._controller_valid(controller_id, viewer_id):
                raise PermissionError("Claim camera control to delete a local battery photo.")
        try:
            return self._sightings.remove_battery_photo(sighting_id)
        except (OSError, SQLiteError, ValueError) as cause:
            raise NiclaBrokerError("Could not delete the local battery photo.") from cause

    def _sighting_observation(self, record, boot_id, receive_ns, receive_utc):
        with self._lock:
            applied = record.get("applied_state")
            if (not self._sightings_enabled or record.get("source") != "on_nicla" or not isinstance(boot_id, str)
                    or not isinstance(applied, dict) or applied.get("measurement_stage_id") != "full_pipeline"):
                return
            frame_id = record.get("frame_id")
            if type(frame_id) is not int or frame_id < 0:
                return
            if boot_id != self._sighting_boot:
                self._reset_sighting(boot_id)
            if frame_id <= self._sighting_last_frame:
                return
            self._sighting_last_frame = frame_id
            detection = record.get("detection")
            if not isinstance(detection, dict):
                return
            status = detection.get("status")
            if status == "no_face":
                self._sighting_previous_hit = None
                if self._sighting_absent_since is None:
                    self._sighting_absent_since = receive_ns
                elif receive_ns - self._sighting_absent_since >= _SIGHTING_CLEAR_NS:
                    self._sighting_saved = self._sighting_pending = None
                    self._sighting_known = False
                return
            if status not in ("face", "multiple"):
                return
            self._sighting_absent_since = None
            decision = record.get("decision")
            if not isinstance(decision, dict):
                decision = {}
            known = (status == "face" and decision.get("status") == "KNOWN_IDENTITY"
                     and decision.get("known_acceptance_enabled") is True
                     and decision.get("candidate") in ("K01", "K02", "K03", "K04", "K05"))
            if self._sighting_saved is not None and (self._sighting_known or not known):
                return
            frame = record.get("frame")
            if not isinstance(frame, dict):
                return
            digest = frame.get("sha256")
            if (frame.get("boot_id") != boot_id or frame.get("frame_id") != frame_id
                    or not isinstance(digest, str) or not re.fullmatch(r"[0-9a-f]{64}", digest)):
                return
            previous_hit = self._sighting_previous_hit
            self._sighting_previous_hit = (frame_id, receive_ns)
            if (self._sighting_saved is None and not known and
                    (previous_hit is None or previous_hit[0] + 1 != frame_id or
                     receive_ns - previous_hit[1] > _SIGHTING_CLEAR_NS)):
                return
            regions = detection.get("regions_xywh")
            face_count = max(2 if status == "multiple" else 1, min(8, len(regions)) if isinstance(regions, list) else 0)
            metadata = {"sha256": digest, "host_received_utc": receive_utc, "face_count": face_count,
                        "policy_id": applied.get("policy_id"), "firmware_sha256": record.get("firmware_sha256"),
                        "config_sha256": record.get("config_sha256"),
                        "decision": {"status": decision.get("status"), "known_acceptance_enabled": known,
                                     "candidate": decision.get("candidate") if known else None}}
            key = (boot_id, frame_id)
            self._sighting_pending = (key, metadata, known)
            payload = self._frames.get(key, {}).get("payloads", {}).get("frame_jpeg")
            if payload:
                self._sighting_packet(key, payload["metadata"], payload["data"])

    def _sighting_packet(self, key, header, data):
        with self._lock:
            pending = self._sighting_pending
            if not self._sightings_enabled or not pending or pending[0] != key:
                return
            _, metadata, known = pending
            if header.get("sha256") != metadata["sha256"]:
                return
            boot_id, frame_id = key
            first_frame = self._sighting_saved
            try:
                self._sightings_queue.put_nowait({"boot_id": boot_id, "frame_id": frame_id, "jpeg": data,
                    "metadata": metadata, "encounter_frame_id": first_frame})
            except Full:
                self._sightings_error = "The sightings writer was busy. The next visible frame can retry."
                return
            self._sighting_saved = frame_id if first_frame is None else first_frame
            self._sighting_known = known
            self._sighting_pending = None
            if self._sightings_writer is None:
                self._sightings_writer = threading.Thread(target=self._save_sightings, name="nicla-sightings", daemon=True)
                self._sightings_writer.start()

    def _save_sightings(self):
        while not self._sightings_shutdown.is_set() or not self._sightings_queue.empty():
            try:
                item = self._sightings_queue.get(timeout=.2)
            except Empty:
                continue
            try:
                self._sightings.add(**item)
            except (OSError, RuntimeError, SQLiteError, ValueError) as error:
                with self._lock:
                    self._sightings_error = f"Could not save sighting ({type(error).__name__})."
                    if self._sighting_boot == item["boot_id"] and self._sighting_saved == item["frame_id"]:
                        self._sighting_saved = None
                        self._sighting_known = False
            else:
                with self._lock:
                    self._sightings_error = None
            finally:
                self._sightings_queue.task_done()

    def video_status(self):
        with self._lock:
            video = self._video
            result = {"state": "idle", "frames": 0, "max_duration_ms": _VIDEO_DURATION_MS,
                      "live_max_duration_ms": _LIVE_VIDEO_DURATION_MS}
            if video:
                result.update({key: video[key] for key in
                               ("state", "id", "frames", "error", "reason", "missed_frames",
                                "mode", "owns_trial", "matched_observations", "draining")
                               if key in video})
            if self._last_video:
                saved = self._last_video
                clip_id = saved["id"]
                result["last_clip"] = {key: saved[key] for key in
                                       ("id", "frames", "mode", "reason", "missed_frames", "matched_observations")
                                       if key in saved}
                result["last_clip"].update(download_url="/api/v2/device/video/file?id=" + clip_id,
                                           manifest_url="/api/v2/device/video/manifest?id=" + clip_id)
                if result["state"] == "ready":
                    result.update(download_url=result["last_clip"]["download_url"],
                                  manifest_url=result["last_clip"]["manifest_url"])
            return result

    def start_video(self, controller_id, viewer_id, mode="capture_only"):
        with self._lock:
            if not self._controller_valid(controller_id, viewer_id):
                raise PermissionError("Claim camera control to record video.")
            if not self._device_ready or not self._latest_boot_id:
                raise NiclaBrokerError("Connect the Nicla before recording video.")
            running = self._applied.get("running")
            if running not in (True, False) or self._applied.get("trial_active") or any(
                    row["command"] not in _READ_ONLY for row in self._pending.values()):
                raise NiclaBrokerError("Wait for the current device action to finish before recording video.")
            if self._video and self._video["state"] in _VIDEO_ACTIVE | {"saving"}:
                raise NiclaBrokerError("A video capture is already in progress.")
            if not isinstance(mode, str) or mode not in _VIDEO_STAGES:
                raise NiclaBrokerError("Choose inference or camera-only video.")
            stage = _VIDEO_STAGES[mode]
            if stage not in self._capabilities["measurement_stage_ids"]:
                raise NiclaBrokerError("This firmware does not support the selected video mode.")
            if running and self._applied.get("measurement_stage_id") != stage:
                raise NiclaBrokerError("Stop processing before recording a different video mode.")
            clip_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ-") + secrets.token_hex(4)
            frame_limit = (_LIVE_VIDEO_FRAMES if running else
                min(_VIDEO_FRAMES, self._capabilities["max_trial_frames"] or _VIDEO_FRAMES))
            duration = (_LIVE_VIDEO_DURATION_MS if running else
                min(_VIDEO_DURATION_MS, self._capabilities["max_trial_duration_ms"] or _VIDEO_DURATION_MS))
            video = {"id": clip_id, "state": "recording" if running else "starting",
                     "boot_id": self._latest_boot_id, "mode": mode, "stage": stage,
                     "owns_trial": not running, "deadline": self.monotonic() + duration / 1000 + (0 if running else 8),
                     "frames": 0, "frame_limit": frame_limit, "bytes": 0, "items": {}, "starts": {},
                     "observations": {}, "draining": False, "matched_observations": 0}
            self._video = video
            policy = self._applied["policy_id"]
            if running:
                return self.video_status()
            model_slot = "large" if mode == "inference" and policy in {"H", "G", "L"} else "small"
        command = {"schema": "nicla-command-v2", "cmd": "trial", "command_id": clip_id,
                   "trial_id": clip_id, "controller_id": controller_id,
                   "frames": frame_limit, "duration_ms": duration,
                   "expected_boot_id": video["boot_id"], "model_slot": model_slot,
                   "measurement_stage_id": stage}
        if policy:
            command["policy_id"] = policy
        result = self.command(command, viewer_id=viewer_id)
        if result.get("status") not in ("accepted", "completed"):
            self._finish_video("command_rejected", result.get("message"))
        return self.video_status()

    def stop_video(self, controller_id, viewer_id):
        with self._lock:
            if not self._controller_valid(controller_id, viewer_id):
                raise PermissionError("Claim camera control to stop recording.")
            video = self._video
            if not video or video["state"] != "recording":
                return self.video_status()
            if not video["owns_trial"]:
                self._drain_live_video(video, "stopped")
                return self.video_status()
            boot_id = video["boot_id"]
        result = self.command({"schema": "nicla-command-v2", "cmd": "stop",
                               "command_id": "video-stop-" + secrets.token_hex(8),
                               "controller_id": controller_id, "expected_boot_id": boot_id,
                               "max_frame_lag": 5}, viewer_id=viewer_id)
        if result.get("status") not in ("accepted", "completed"):
            raise NiclaBrokerError(result.get("message", "Could not stop capture."))
        return self.video_status()

    def _video_record(self, record, boot_id):
        event = record.get("event")
        if event not in ("boot", "control_input_status", "control_ack", "capture_start",
                         "trial_finished", "observation_complete"):
            return
        with self._lock:
            video = self._video
            if not video or video["state"] not in _VIDEO_ACTIVE:
                return
            if boot_id != video["boot_id"]:
                if event in ("boot", "control_input_status") and boot_id:
                    self._finish_video("device_restarted")
                return
            if event == "control_ack" and record.get("command_id") == video["id"]:
                status = record.get("control_status")
                if status == "completed" and record.get("trial_id") == video["id"]:
                    video["state"] = "recording"
                elif status in _TERMINAL_ACKS:
                    self._finish_video("command_rejected", record.get("reason_code"))
            elif event == "trial_finished" and record.get("trial_id") == video["id"]:
                self._finish_video(record.get("reason_code", "trial_finished"))
            elif (event == "control_ack" and record.get("command") == "stop"
                  and record.get("control_status") == "completed" and not video["owns_trial"]):
                self._drain_live_video(video, "stopped")
            elif (event == "capture_start" and video["state"] == "recording"
                  and not video["draining"]):
                frame_id = record.get("frame_index")
                if type(frame_id) is int and len(video["starts"]) < video["frame_limit"]:
                    video["starts"][frame_id] = (record.get("ticks_ms"), record.get("ticks_modulus_ms"))
            elif event == "observation_complete" and video["mode"] == "inference":
                self._video_observation(video, record)
                self._maybe_finish_live_video(video)

    def _video_observation(self, video, record):
        frame_id = record.get("frame_id")
        frame = record.get("frame") or {}
        if (type(frame_id) is not int or not isinstance(frame, dict) or frame_id not in video["starts"]
                or frame.get("frame_id") != frame_id
                or frame.get("boot_id") != video["boot_id"] or not frame.get("sha256")
                or (record.get("applied_state") or {}).get("measurement_stage_id") != video["stage"]):
            return
        identity, decision = record.get("identity") or {}, record.get("decision") or {}
        video["observations"][frame_id] = {"frame_sha256": frame["sha256"],
            "detection": record.get("detection"), "crop": record.get("crop"),
            "identity": {k: identity.get(k) for k in ("status", "model_id", "model_warm")},
            "decision": {k: decision.get(k) for k in ("status", "reason", "known_acceptance_enabled")},
            "timings_ms": record.get("timings_ms"), "config_sha256": record.get("config_sha256"),
            "firmware_sha256": record.get("firmware_sha256"),
            "host_received_monotonic_ns": record.get("host_received_monotonic_ns"),
            "host_received_utc": record.get("host_received_utc"),
            "policy_id": (record.get("applied_state") or {}).get("policy_id")}
        self._match_video_observation(video, frame_id)

    @staticmethod
    def _match_video_observation(video, frame_id):
        frame = video["items"].get(frame_id)
        observation = video["observations"].get(frame_id)
        if frame and observation and observation["frame_sha256"] == frame["sha256"]:
            if "observation" not in frame:
                video["matched_observations"] += 1
            frame["observation"] = observation

    def _drain_live_video(self, video, reason):
        if not video["draining"]:
            video.update(draining=True, end_reason=reason, deadline=self.monotonic() + 1.5)
        self._maybe_finish_live_video(video)

    def _maybe_finish_live_video(self, video):
        if video["draining"] and video["starts"].keys() <= video["items"].keys() and (
                video["mode"] == "capture_only" or video["matched_observations"] == video["frames"]):
            self._finish_video(video["end_reason"])

    def _video_frame(self, metadata, data, receive_ns, receive_utc):
        with self._lock:
            video = self._video
            if not video or video["state"] != "recording" or metadata["boot_id"] != video["boot_id"]:
                return
            frame_id = metadata["frame_id"]
            if frame_id not in video["starts"] or frame_id in video["items"]:
                return
            if video["bytes"] + len(data) > _VIDEO_BYTES:
                self._finish_video("size_limit")
                return
            ticks, modulus = video["starts"].get(frame_id, (None, None))
            video["items"][frame_id] = {
                "data": data, "boot_id": metadata["boot_id"], "frame_id": frame_id,
                "width": metadata.get("width"), "height": metadata.get("height"),
                "sha256": metadata.get("sha256"), "capture_ticks_ms": ticks,
                "ticks_modulus_ms": modulus, "host_received_monotonic_ns": receive_ns,
                "host_received_utc": receive_utc,
            }
            video["frames"] = len(video["items"])
            video["bytes"] += len(data)
            self._match_video_observation(video, frame_id)
            if not video["owns_trial"] and video["frames"] >= video["frame_limit"]:
                self._drain_live_video(video, "frame_limit")
            else:
                self._maybe_finish_live_video(video)

    def _finish_video(self, reason, error=None):
        with self._lock:
            video = self._video
            if not video or video["state"] not in _VIDEO_ACTIVE:
                return
            video["reason"] = reason
            items = video.pop("items")
            starts = video.pop("starts")
            video.pop("observations")
            frame_ids = sorted(items)
            gaps = frame_ids[-1] - frame_ids[0] + 1 - len(frame_ids) if frame_ids else 0
            video["missed_frames"] = max(gaps, len(starts.keys() - items.keys()))
            if not items:
                video.update(state="error", error=error or "The Nicla sent no video frames.")
                return
            video["state"] = "saving"
            frames = [items[frame_id] for frame_id in frame_ids]
            writer = threading.Thread(target=self._save_video, args=(video, frames), daemon=True)
            video["writer"] = writer
            writer.start()

    def _save_video(self, video, frames):
        try:
            write_clip(self.video_root, video["id"], frames, reason=video["reason"], mode=video["mode"],
                       missed_frames=video["missed_frames"])
        except Exception as error:
            with self._lock:
                video.update(state="error", error=f"Could not save video: {error}")
        else:
            with self._lock:
                video["state"] = "ready"
                self._last_video = video

    def get_video_file(self, clip_id, asset="clip.avi"):
        if not _VIDEO_ID.fullmatch(clip_id):
            raise KeyError("unknown video")
        if asset not in ("clip.avi", "manifest.json") and not re.fullmatch(r"frames/frame-\d{6}\.jpg", asset):
            raise KeyError("unknown video asset")
        try:
            return (self.video_root / clip_id / asset).read_bytes()
        except FileNotFoundError as error:
            raise KeyError("unknown video") from error

    def _handle_stream_item(self, item, receive_ns, receive_utc):
        item_type = item.get("type")
        if item_type == "record":
            record = item["record"]
            record = dict(record)
            record["host_received_utc"] = receive_utc
            record["host_received_monotonic_ns"] = receive_ns
            boot_id = record.get("boot_id", record.get("run_id"))
            if self._battery.consume_ack(record, boot_id):
                return
            self._video_record(record, boot_id)
            if record.get("event") == "boot":
                with self._lock:
                    prior = self._latest_boot_id
                    if prior is not None and boot_id != prior:
                        self._fail_pending("device_restarted")
                    self._latest_boot_id = boot_id
                    self._device_ready = bool(self._connected and boot_id)
                    if self._device_ready:
                        self._connection_error = None
                        self._connection_warning = None
                    self._latest_observation = None
                    self._latest_frame_id = None
                    self._latest_frame_received_ns = None
                    self._applied.update({"running": None, "sleeping": None, "policy_id": None,
                                          "model_override": None, "measurement_stage_id": None,
                                          "model_slot": None, "trial_active": False})
                    self._device.update({"boot_id": boot_id, "reset_cause": record.get("reset_cause"),
                        "firmware_sha256": record.get("firmware_sha256"),
                        "config_sha256": record.get("model_config_sha256"),
                        "firmware_release": record.get("firmware_release"),
                        "protocol": record.get("protocol")})
                self._publish({"type": "device_boot", "record": record,
                               "previous_boot_id": prior})
                return
            event_name = record.get("event")
            if event_name == "control_input_status":
                # A stopped device may not emit a boot record or camera
                # observation after USB reconnect. The periodic control
                # status is still a fresh, versioned hello from this serial
                # session; accept it only when its session and image hashes
                # are complete and syntactically valid.
                firmware_hash = record.get("firmware_sha256")
                config_hash = record.get("model_config_sha256")
                schema_hash = record.get("schema_sha256")
                valid_hello = (
                    record.get("schema_id") == "nicla-face-telemetry-1"
                    and isinstance(boot_id, str)
                    and re.fullmatch(r"[A-Za-z0-9._:-]{1,128}", boot_id)
                    and all(isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value)
                            for value in (firmware_hash, config_hash, schema_hash))
                )
                if valid_hello:
                    with self._lock:
                        prior = self._latest_boot_id
                        if prior != boot_id:
                            if prior is not None:
                                self._fail_pending("device_restarted")
                            self._latest_observation = None
                            self._latest_frame_id = None
                            self._latest_frame_received_ns = None
                            self._applied.update({"running": None, "sleeping": None,
                                                  "policy_id": None, "model_override": None,
                                                  "measurement_stage_id": None, "model_slot": None,
                                                  "trial_active": False})
                        self._latest_boot_id = boot_id
                        self._device_ready = bool(self._connected)
                        if self._device_ready:
                            self._connection_error = None
                            self._connection_warning = None
                        self._device.update({"boot_id": boot_id,
                            "firmware_sha256": firmware_hash,
                            "config_sha256": config_hash})
                    self._publish({"type": "device_hello" if prior != boot_id else "telemetry",
                                   "event": event_name, "boot_id": boot_id,
                                   "record": record, "previous_boot_id": prior})
                    return
            if event_name == "observation_complete":
                record["type"] = "observation_complete"
                with self._lock:
                    self._device_ready = bool(self._connected and boot_id)
                    if self._device_ready:
                        self._connection_error = None
                        self._connection_warning = None
                    if boot_id != self._latest_boot_id:
                        if self._latest_boot_id is not None:
                            self._fail_pending("device_restarted")
                        self._latest_boot_id = boot_id
                        self._latest_frame_id = None
                    self._latest_observation = record
                    self._latest_frame_id = record.get("frame_id")
                    self._applied.update(record.get("applied_state") or {})
                self._sighting_observation(record, boot_id, receive_ns, receive_utc)
                self._publish(record)
                return
            if event_name == "control_ack":
                record["type"] = "command_ack"
                status = str(record.get("control_status", "error"))
                command_id = record.get("command_id")
                command_name = record.get("command")
                if command_name == "discover" and record.get("protocol"):
                    with self._lock:
                        self._capabilities["protocol"] = record.get("protocol")
                        self._device.update({"protocol": record.get("protocol"),
                            "device": record.get("device", self._device.get("device"))})
                if command_name == "config" and status in _TERMINAL_ACKS:
                    models = record.get("models") or {}
                    with self._lock:
                        if isinstance(models, dict):
                            self._capabilities["models"] = [
                                {"id": slot, "value": slot,
                                 "label": ("Small embedding · experimental" if slot == "small" else
                                           "Larger classifier · experimental" if slot == "large" else str(slot)),
                                 **(dict(details) if isinstance(details, dict) else {})}
                                for slot, details in models.items()]
                        policies = record.get("policy_ids")
                        if isinstance(policies, list):
                            self._capabilities["policies"] = [str(value) for value in policies
                                if isinstance(value, str) and value]
                        stages = record.get("measurement_stage_ids")
                        if isinstance(stages, list):
                            self._capabilities["measurement_stage_ids"] = [str(value) for value in stages
                                if isinstance(value, str) and value]
                        for key in ("max_trial_frames", "max_trial_duration_ms"):
                            value = record.get(key)
                            if type(value) is int and value > 0:
                                self._capabilities[key] = value
                        reported_commands = record.get("supported_commands")
                        if isinstance(reported_commands, list):
                            if status == "completed":
                                self._battery.config_ack(reported_commands, boot_id)
                            self._capabilities["commands"] = [value for value in reported_commands
                                if isinstance(value, str) and value in _SUPPORTED_COMMANDS]
                        if "camera_sleep_supported" in record:
                            self._capabilities["camera_sleep_supported"] = bool(record["camera_sleep_supported"])
                if status in _TERMINAL_ACKS and command_id:
                    with self._lock:
                        self._pending.pop(command_id, None)
                        self._command_cache[command_id] = {"status": status, "event": record}
                        self._command_cache.move_to_end(command_id)
                        while len(self._command_cache) > 128:
                            self._command_cache.popitem(last=False)
                        if record.get("running") is not None:
                            self._applied["running"] = bool(record["running"])
                        if record.get("sleeping") is not None:
                            self._applied["sleeping"] = bool(record["sleeping"])
                        if record.get("policy_id"):
                            self._applied["policy_id"] = record["policy_id"]
                        if record.get("measurement_stage_id"):
                            self._applied["measurement_stage_id"] = record["measurement_stage_id"]
                        if record.get("model_slot"):
                            self._applied["model_slot"] = record["model_slot"]
                        if record.get("trial_active") is not None:
                            self._applied["trial_active"] = bool(record["trial_active"])
                        if "model_override" in record:
                            self._applied["model_override"] = record["model_override"]
                        self._applied["last_command_id"] = command_id
                        if command_name == "stop" and status == "completed":
                            self._reset_sighting(boot_id)
                            self._battery.capture_stopped(boot_id)
                self._publish(record)
                return
            if event_name == "config":
                self._capabilities.update({
                    "protocol": record.get("protocol"),
                    "policies": record.get("policy_ids", []),
                    "measurement_stage_ids": record.get("measurement_stage_ids", []),
                    "max_trial_frames": record.get("max_trial_frames"),
                    "max_trial_duration_ms": record.get("max_trial_duration_ms"),
                    "models": list((record.get("models") or {}).values()),
                    "commands": [value for value in record.get("supported_commands", list(_SUPPORTED_COMMANDS))
                                 if isinstance(value, str) and value in _SUPPORTED_COMMANDS],
                    "camera_sleep_supported": record.get("camera_sleep_supported")})
            if event_name == "policy_changed":
                self._applied["policy_id"] = record.get("policy_id")
            elif event_name == "camera_sleep":
                self._applied["sleeping"] = True
            elif event_name == "camera_wake":
                self._applied["sleeping"] = False
            elif event_name == "trial_finished":
                with self._lock:
                    self._applied["running"] = False
                    self._applied["trial_active"] = False
                    self._reset_sighting(boot_id)
            self._publish({"type": "telemetry", "record": record,
                           "event": event_name, "boot_id": boot_id,
                           "frame_id": record.get("frame_id", record.get("frame_index"))})
            return
        if item_type == "packet":
            header = item["header"]
            boot_id, frame_id = header.get("boot_id"), header.get("frame_id")
            if not isinstance(boot_id, str) or type(frame_id) is not int:
                self._publish({"type": "packet_rejected", "reason_code": "packet_missing_frame_identity",
                               "header": header})
                return
            key, group = self._ensure_frame_group(boot_id, frame_id)
            kind = header.get("kind")
            if kind not in ("frame_jpeg", "detector_input", "identity_input"):
                self._publish({"type": "packet_rejected", "reason_code": "unsupported_packet_kind",
                               "header": header})
                return
            data = item["data"]
            if kind == "frame_jpeg" and not data.startswith(b"\xff\xd8"):
                self._publish({"type": "packet_rejected", "reason_code": "jpeg_magic_invalid",
                               "header": header})
                return
            metadata = _canonical_packet_dtype(header) if kind in ("detector_input", "identity_input") else dict(header)
            group["payloads"][kind] = {"metadata": metadata, "data": data}
            group["last_receive_monotonic_ns"] = receive_ns
            if kind == "frame_jpeg":
                self._latest_frame_received_ns = receive_ns
                self._video_frame(metadata, data, receive_ns, receive_utc)
                self._sighting_packet(key, metadata, data)
            self._publish({"type": "frame_payload", "boot_id": boot_id,
                "frame_id": frame_id, "kind": kind,
                "byte_count": len(data), "sha256": header.get("sha256"),
                "metadata": metadata})
            return
        self._publish({"type": "protocol_error", **item})

    def _fail_pending(self, reason):
        """Finish commands from the ended USB session while holding the broker lock."""
        for command_id, pending in self._pending.items():
            ack = self._publish({"type": "command_ack", "command_id": command_id,
                                 "command": pending["command"], "control_status": "error",
                                 "reason_code": reason, "terminal": True})
            self._command_cache[command_id] = {"status": "error", "event": ack}
        self._pending.clear()
        while len(self._command_cache) > 128:
            self._command_cache.popitem(last=False)

    def _expire_commands(self):
        now = self.monotonic()
        with self._lock:
            video = self._video
            if video and video["state"] in _VIDEO_ACTIVE and now >= video["deadline"]:
                if video["owns_trial"]:
                    self._finish_video("timeout")
                elif video["draining"]:
                    self._finish_video(video["end_reason"])
                else:
                    self._drain_live_video(video, "duration_limit")
        expired = []
        with self._lock:
            for command_id, pending in list(self._pending.items()):
                if now >= pending["deadline"]:
                    expired.append((command_id, pending))
                    self._pending.pop(command_id, None)
                    self._command_cache[command_id] = {"status": "timeout", "event": None}
                    self._command_cache.move_to_end(command_id)
                    while len(self._command_cache) > 128:
                        self._command_cache.popitem(last=False)
        for command_id, pending in expired:
            self._publish({"type": "command_ack", "command_id": command_id,
                "command": pending["command"], "control_status": "timeout",
                "reason_code": "device_ack_deadline_expired",
                "terminal": True})

    def _write(self, command):
        payload = (json.dumps(command, separators=(",", ":"), allow_nan=False) + "\n").encode("utf-8")
        if len(payload) > 384:
            raise ValueError("command_oversize")
        with self._write_lock:
            serial = self._serial
            if not self._connected or serial is None:
                raise NiclaBrokerError("device_not_connected")
            if not self._device_ready:
                raise NiclaBrokerError("device_not_ready_waiting_for_boot_or_frame")
            written = serial.write(payload)
            serial.flush()
        if int(written) != len(payload):
            raise IOError("partial command write")
        self._serial_bytes_written += len(payload)
        return len(payload)

    def claim_controller(self, viewer_id):
        viewer_id = str(viewer_id or "").strip()
        if not viewer_id or len(viewer_id) > 96:
            return {"status": "rejected", "reason": "invalid_viewer_id"}
        now = self.monotonic()
        with self._lock:
            if self._controller_id and now < self._controller_expiry:
                if self._controller_viewer_id == viewer_id:
                    self._controller_expiry = now + self.controller_lease_s
                    return self._controller_row(viewer_id)
                return {"status": "owned", "owner_viewer_id": self._controller_viewer_id,
                        "lease_remaining_ms": int((self._controller_expiry - now) * 1000)}
            self._controller_id = secrets.token_urlsafe(24)
            self._controller_viewer_id = viewer_id
            self._controller_expiry = now + self.controller_lease_s
            row = self._controller_row(viewer_id)
        self._publish({"type": "controller_claimed", "viewer_id": viewer_id})
        return row

    def _controller_row(self, viewer_id=None):
        now = self.monotonic()
        remaining = max(0, int((self._controller_expiry - now) * 1000))
        owned = viewer_id == self._controller_viewer_id and now < self._controller_expiry
        return {"status": "claimed", "controller_id": self._controller_id if owned else None,
                "viewer_id": self._controller_viewer_id,
                "owned_by_viewer": owned,
                "lease_remaining_ms": remaining,
                "lease_expires_host_monotonic_ns": self.monotonic_ns() + remaining * 1_000_000}

    def heartbeat_controller(self, controller_id, viewer_id):
        with self._lock:
            if (controller_id != self._controller_id or viewer_id != self._controller_viewer_id or
                    self.monotonic() >= self._controller_expiry):
                return {"status": "rejected", "reason": "controller_lease_expired_or_mismatched"}
            self._controller_expiry = self.monotonic() + self.controller_lease_s
            row = self._controller_row(viewer_id)
            row["status"] = "renewed"
            return row

    def release_controller(self, controller_id, viewer_id):
        with self._lock:
            if not self._controller_valid(controller_id, viewer_id):
                return {"status": "rejected", "reason": "controller_lease_mismatch"}
            self._controller_id = None
            self._controller_viewer_id = None
            self._controller_expiry = 0.0
        self._publish({"type": "controller_released", "viewer_id": viewer_id})
        return {"status": "released"}

    def _controller_valid(self, controller_id, viewer_id):
        return (bool(controller_id) and controller_id == self._controller_id and
                viewer_id == self._controller_viewer_id and self.monotonic() < self._controller_expiry)

    def command(self, command, viewer_id=None):
        if not isinstance(command, dict):
            return {"status": "rejected", "message": "command_must_be_object"}
        cmd = command.get("cmd")
        command_id = command.get("command_id")
        if cmd not in _SUPPORTED_COMMANDS:
            return {"command_id": command_id, "status": "rejected", "message": "unsupported_command"}
        if not isinstance(command_id, str) or not _COMMAND_ID.fullmatch(command_id):
            return {"command_id": command_id, "status": "rejected", "message": "invalid_command_id"}
        if cmd == "trial":
            with self._lock:
                advertised_commands = self._capabilities.get("commands") or []
            if "trial" not in advertised_commands:
                return {"command_id": command_id, "status": "rejected", "message": "unsupported_command"}
            frames = command.get("frames")
            duration_ms = command.get("duration_ms")
            if type(frames) is not int or frames < 1:
                return {"command_id": command_id, "status": "rejected", "message": "trial_frames_out_of_range"}
            if type(duration_ms) is not int or duration_ms < 1:
                return {"command_id": command_id, "status": "rejected", "message": "trial_duration_out_of_range"}
            with self._lock:
                max_frames = self._capabilities.get("max_trial_frames") or 50
                max_duration_ms = self._capabilities.get("max_trial_duration_ms") or 30000
                stage_ids = list(self._capabilities.get("measurement_stage_ids") or [])
                policies = list(self._capabilities.get("policies") or [])
            if frames > max_frames:
                return {"command_id": command_id, "status": "rejected", "message": "trial_frames_out_of_range"}
            if duration_ms > max_duration_ms:
                return {"command_id": command_id, "status": "rejected", "message": "trial_duration_out_of_range"}
            trial_id = command.get("trial_id")
            if not isinstance(trial_id, str) or not re.fullmatch(r"[A-Za-z0-9_-]{1,48}", trial_id):
                return {"command_id": command_id, "status": "rejected", "message": "invalid_trial_id"}
            stage_id = command.get("measurement_stage_id")
            if stage_id is not None and (not isinstance(stage_id, str) or not stage_ids or stage_id not in stage_ids):
                return {"command_id": command_id, "status": "rejected", "message": "unsupported_measurement_stage"}
            model_slot = command.get("model_slot")
            if model_slot is not None and model_slot not in ("small", "large"):
                return {"command_id": command_id, "status": "rejected", "message": "unsupported_model_slot"}
            policy_id = command.get("policy_id")
            if policy_id is not None and (not isinstance(policy_id, str) or (policies and policy_id not in policies)):
                return {"command_id": command_id, "status": "rejected", "message": "unsupported_policy"}
        now = self.monotonic()
        with self._lock:
            prior = self._command_cache.get(command_id)
            if prior is not None:
                return {"command_id": command_id, "status": prior["status"],
                        "message": "duplicate_command_id_not_resent"}
            if command_id in self._pending:
                return {"command_id": command_id, "status": "accepted",
                        "message": "command_already_pending"}
            if len(self._pending) >= 32:
                return {"command_id": command_id, "status": "busy",
                        "message": "device_ack_queue_full"}
            if cmd not in _READ_ONLY and not self._controller_valid(command.get("controller_id"), viewer_id):
                return {"command_id": command_id, "status": "rejected",
                        "message": "controller_authority_required"}
            if (self._video and self._video["state"] in _VIDEO_ACTIVE
                    and cmd not in _READ_ONLY | {"stop"} and command_id != self._video["id"]):
                return {"command_id": command_id, "status": "busy",
                        "message": "video_capture_in_progress"}
            if not self._connected:
                return {"command_id": command_id, "status": "rejected", "message": "device_not_connected"}
            if not self._device_ready:
                return {"command_id": command_id, "status": "rejected",
                        "message": "device_not_ready_waiting_for_boot_or_frame"}
            wire = {key: value for key, value in command.items()
                    if key not in ("controller_id", "viewer_id", "timeout_ms")}
            observation = self._latest_observation or {}
            if cmd not in _READ_ONLY and cmd != "one_shot":
                boot_id = observation.get("boot_id", self._latest_boot_id)
                frame_id = observation.get("frame_id", self._latest_frame_id)
                if boot_id:
                    wire.setdefault("expected_boot_id", boot_id)
                # Stop addresses the processing loop for this boot. The
                # camera can advance several frames while USB drains.
                if cmd != "stop" and type(frame_id) is int:
                    wire.setdefault("expected_frame_id", frame_id)
                    wire.setdefault("max_frame_lag", 2)
        timeout_ms = command.get("timeout_ms", 15000 if cmd == "one_shot" else 8000)
        try:
            timeout_s = max(1.0, min(float(timeout_ms) / 1000.0, 30.0))
        except (TypeError, ValueError):
            timeout_s = 15.0
        with self._lock:
            # Register before writing so a fast device ACK cannot race ahead
            # of the pending-command record.
            self._pending[command_id] = {"command": cmd,
                "deadline": now + timeout_s, "viewer_id": viewer_id}
        try:
            self._write(wire)
        except Exception as error:
            message = str(error)[:160]
            with self._lock:
                self._pending.pop(command_id, None)
            self._publish({"type": "command_ack", "command_id": command_id,
                "command": cmd, "control_status": "error", "reason_code": "serial_write_failed",
                "message": message})
            return {"command_id": command_id, "status": "rejected", "message": message}
        self._publish({"type": "command_ack", "command_id": command_id,
            "command": cmd, "control_status": "accepted", "terminal": False,
            "message": "serial_write_complete_waiting_for_device_ack"})
        return {"command_id": command_id, "status": "accepted", "message": "device_ack_pending"}

    def status(self, viewer_id=None):
        now_ns = self.monotonic_ns()
        with self._lock:
            connected = self._device_ready
            serial_open = self._connected
            observation = dict(self._latest_observation) if self._latest_observation else None
            boot_id = self._latest_boot_id
            frame_id = self._latest_frame_id
            frame_received_ns = self._latest_frame_received_ns
            last_receive_ns = self._last_receive_ns
            descriptor = dict(self._descriptor) if self._descriptor else None
            error = self._connection_error
            warning = self._connection_warning
            applied = dict(self._applied)
            authority = self._controller_row(viewer_id) if self._controller_id else {
                "owner_viewer_id": None, "owned_by_viewer": False, "lease_remaining_ms": 0,
                "lease_expires_host_monotonic_ns": None}
            if self._controller_id and self.monotonic() >= self._controller_expiry:
                authority["expired"] = True
            device = dict(self._device)
            caps = dict(self._capabilities)
            event_seq = self._event_seq
        latest_frame = None
        if observation is not None:
            obs_boot = observation.get("boot_id", boot_id)
            obs_frame = observation.get("frame_id")
            age_source_ns = frame_received_ns if frame_id == obs_frame and frame_received_ns else observation.get("host_received_monotonic_ns")
            age_ms = max(0, int((now_ns - age_source_ns) / 1_000_000)) if isinstance(age_source_ns, int) else None
            frame_ready = False
            if obs_boot is not None and type(obs_frame) is int:
                group = self._frames.get((str(obs_boot), int(obs_frame)))
                frame_ready = bool(group and "frame_jpeg" in group["payloads"])
            latest_frame = {"boot_id": obs_boot, "frame_id": obs_frame,
                "width": (observation.get("camera") or {}).get("frame_shape", [None, None])[0]
                    if isinstance(observation.get("camera"), dict) else None,
                "height": (observation.get("camera") or {}).get("frame_shape", [None, None])[1]
                    if isinstance(observation.get("camera"), dict) else None,
                "orientation": (observation.get("camera") or {}).get("camera_orientation"),
                "age_ms": age_ms, "stale": age_ms is None or age_ms > 2000,
                "encoding": ((observation.get("frame_encoding") or {}).get("encoding")),
                "frame_ref": (f"/api/v2/device/frame?boot_id={obs_boot}&frame_id={obs_frame}"
                              if frame_ready else None),
                "payload_ready": frame_ready}
        return {"schema_id": "nicla-device-status-v2", "mode": "ON NICLA" if connected else "DISCONNECTED",
            "connected": connected, "serial_open": serial_open,
            "device_ready": connected, "connecting": self._connecting,
            "connection_error": error, "connection_warning": warning,
            "device": device or descriptor,
            "serial_descriptor": descriptor, "boot_id": boot_id,
            "applied_state": {**applied,
                "processing": applied.get("running"),
                "camera_sleeping": applied.get("sleeping"),
                "policy": applied.get("policy_id"),
                # A cleared override means the policy selects the route. Do
                # not keep showing a stale model from the prior observation.
                "model": applied.get("model_override") or "policy"},
            "running": applied.get("running"),
            "sleeping": applied.get("sleeping"),
            "policy": applied.get("policy_id"),
            "last_observation": observation,
            "latest_frame": latest_frame, "controller_authority": authority,
            "capabilities": caps, "host_last_receive_age_ms":
                max(0, int((now_ns - last_receive_ns) / 1_000_000)) if last_receive_ns else None,
            "event_seq": event_seq, "serial_bytes_read": self._serial_bytes_read,
            "serial_bytes_written": self._serial_bytes_written,
            "parser_error_count": self._parser.errors}

    def capabilities(self):
        with self._lock:
            result = dict(self._capabilities)
        commands = result.get("commands")
        if not isinstance(commands, list):
            commands = list(_SUPPORTED_COMMANDS)
        else:
            commands = [value for value in commands if isinstance(value, str) and value in _SUPPORTED_COMMANDS]
        result["commands"] = commands
        result["supported_commands"] = list(commands)
        if not result.get("models"):
            result["models"] = [
                {"id": "small", "value": "small", "model_id": "d-a010-int8-seed-20261110",
                 "label": "Small embedding · experimental"},
                {"id": "large", "value": "large", "model_id": "e-capacity-a025-int8-seed-20261101",
                 "label": "Larger classifier · experimental"},
            ]
        result["policies"] = list(result.get("policies") or [])
        result["camera_sleep_supported"] = True
        result["schema_id"] = "nicla-device-capabilities-v2"
        result["identity_qualification"] = "experimental_unqualified"
        return result

    def events(self, after_seq=0, limit=100):
        try:
            after_seq = max(0, int(after_seq))
            limit = max(1, min(int(limit), 256))
        except (TypeError, ValueError):
            after_seq, limit = 0, 100
        with self._lock:
            rows = list(self._events)
        oldest = rows[0]["event_seq"] if rows else self._event_seq + 1
        pending = [row for row in rows if row["event_seq"] > after_seq]
        skipped_count = max(0, len(pending) - limit)
        selected = pending[-limit:] if skipped_count else pending
        next_seq = selected[-1]["event_seq"] if selected else after_seq
        return {"events": selected, "next_seq": next_seq, "oldest_seq": oldest,
                "skipped_count": skipped_count,
                "truncated": bool(skipped_count or (rows and after_seq < oldest - 1))}

    def get_frame(self, boot_id, frame_id):
        key = (str(boot_id), int(frame_id))
        with self._lock:
            group = self._frames.get(key)
            if not group or "frame_jpeg" not in group["payloads"]:
                raise KeyError("matched_frame_not_retained")
            item = group["payloads"]["frame_jpeg"]
            return item["data"], dict(item["metadata"])

    def get_input(self, boot_id, frame_id, kind="identity"):
        if kind not in ("detector", "identity"):
            raise ValueError("input_kind_must_be_detector_or_identity")
        key = (str(boot_id), int(frame_id))
        payload_kind = kind + "_input"
        with self._lock:
            group = self._frames.get(key)
            if not group or payload_kind not in group["payloads"]:
                raise KeyError("exact_input_not_retained")
            item = group["payloads"][payload_kind]
            return item["data"], dict(item["metadata"])


__all__ = ["NiclaBroker", "NiclaBrokerError", "NiclaPortNotFound",
           "AmbiguousNiclaPort", "NiclaPortBusy", "NiclaStreamParser"]
