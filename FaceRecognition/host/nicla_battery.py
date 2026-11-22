"""Private, bounded command lane for importing Nicla battery sightings."""

from __future__ import annotations

import hashlib
import json
import re
import secrets
from sqlite3 import Error as SQLiteError
import threading
import time


BATTERY_COMMANDS = frozenset(("battery_status", "battery_arm", "battery_disarm", "battery_list", "battery_read"))
_ACTION_COMMANDS = {"arm": "battery_arm", "disarm": "battery_disarm", "import": "battery_list"}
_ACK_STATUSES = {"completed", "error", "unsupported", "rejected", "timeout"}
_MAX_RECORDS = 24
_MAX_BYTES = 96 * 1024
_READ_SIZE = 512
_ACK_TIMEOUT = 3.0
_IMPORT_TIMEOUT = 90.0
_BOOT_ID = re.compile(r"[A-Za-z0-9._:-]{1,128}")
_SHA256 = re.compile(r"[0-9a-f]{64}")
_HEX = re.compile(r"[0-9a-fA-F]+")
_ROSTER = frozenset(("K01", "K02", "K03", "K04", "K05"))


def _integer(value, low, high):
    return type(value) is int and low <= value <= high


class NiclaBatteryService:
    """Keeps large private ACKs out of the public telemetry and frame caches."""

    def __init__(self, broker, sightings):
        self.broker = broker
        self.sightings = sightings
        self.lock = broker._lock
        self._config_session = None
        self._config_supported = False
        self._state_session = None
        self._state = None
        self._waiting = None
        self._worker = None
        self._job = None
        self._reason = None
        self._error = None
        self._next_probe_at = 0.0
        self._last_import_count = 0
        self._refresh_generation = 0

    def _session(self):
        broker = self.broker
        boot = broker._latest_boot_id
        if broker._connected and broker._device_ready and broker._serial is not None and isinstance(boot, str):
            return broker._serial, boot
        return None

    def _idle(self):
        broker = self.broker
        video = broker._video or {}
        return (broker._applied.get("running") is False and not broker._applied.get("trial_active")
                and video.get("state") not in ("starting", "recording", "saving"))

    def _refresh_session(self):
        session = self._session()
        if session != self._state_session:
            self._state_session = session
            self._state = self._reason = self._error = None
            self._last_import_count = 0
            self._next_probe_at = 0.0
            self._refresh_generation += 1
        return session

    def capture_stopped(self, boot_id):
        with self.lock:
            session = self._session()
            if session is not None and session[1] == boot_id:
                self._state = None
                self._next_probe_at = 0.0
                self._refresh_generation += 1

    def config_ack(self, commands, boot_id):
        if not isinstance(commands, list):
            return
        with self.lock:
            session = self._session()
            if session is not None and session[1] == boot_id:
                self._config_session = session
                self._config_supported = BATTERY_COMMANDS.issubset(value for value in commands if isinstance(value, str))

    def consume_ack(self, record, boot_id):
        command = record.get("command")
        if record.get("event") != "control_ack" or not isinstance(command, str) or command not in BATTERY_COMMANDS:
            return False
        with self.lock:
            pending = self._waiting
            if (pending is not None and pending["ack"] is None and record.get("command_id") == pending["id"] and command == pending["command"]
                    and boot_id == pending["session"][1] and self._session() == pending["session"]
                    and record.get("source") == "usb" and str(record.get("control_status")) in _ACK_STATUSES):
                if record.get("control_status") == "completed" and type(record.get("running")) is bool:
                    self.broker._applied["running"] = record["running"]
                pending["ack"] = record
                pending["event"].set()
        return True

    def status(self, viewer_id=None):
        with self.lock:
            session = self._refresh_session()
            advertised = session is not None and self._config_session == session and self._config_supported
            idle = self._idle()
            if advertised and idle and self._state is None and self._job is None and time.monotonic() >= self._next_probe_at:
                self._start("status", session)
            state = self._state or {}
            available = advertised and state.get("available") is True
            controller = self.broker._controller_valid(self.broker._controller_id, viewer_id)
            allowed = available and idle and controller and self._job is None
            armed = state.get("armed")
            pending = state.get("record_count")
            reason = self._reason or state.get("reason")
            if session is None:
                reason = "Connect the Nicla over USB to check battery capture and photos."
            elif self._config_session != session:
                reason = "Waiting for the Nicla to report its capabilities."
            elif not advertised:
                reason = "This installed Nicla release does not support battery capture."
            elif self._job is not None:
                reason = {"status": "Checking battery capture…", "arm": "Arming battery capture…",
                          "disarm": "Stopping battery capture…", "import": "Importing board photos…"}[self._job]
            elif state and not available:
                reason = state.get("reason") or "Battery commands require a safe native USB connection."
            elif not idle:
                reason = "Stop Nicla processing or recording before changing battery capture or importing photos."
            elif state and not controller and reason is None:
                reason = "Claim camera control to change battery capture or import photos."
            response = {"supported": advertised, "available": available, "armed": armed,
                        "checking": session is not None and self._config_session != session,
                        "pending_photos": pending, "reason": reason, "error": self._error,
                        "unimported_count": state.get("unimported_count"),
                        "busy": self._job is not None, "active_action": self._job,
                        "actions": {"arm": bool(allowed and armed is False), "disarm": bool(allowed and armed is True),
                                    "import": bool(allowed and pending)},
                        "max_records": state.get("max_records", _MAX_RECORDS),
                        "max_jpeg_bytes": state.get("max_jpeg_bytes", _MAX_BYTES),
                        "last_import_count": self._last_import_count}
        try:
            response["imported_count"] = self.sightings.summary(limit=0)["battery_import_count"]
        except (OSError, SQLiteError, ValueError):
            response["imported_count"] = None
            response["reason"] = response["error"] = "Could not read the local battery photo history."
        return response

    def action(self, action, controller_id, viewer_id):
        if not isinstance(action, str) or action not in _ACTION_COMMANDS:
            raise ValueError("Choose arm, disarm, or import.")
        with self.lock:
            if not self.broker._controller_valid(controller_id, viewer_id):
                raise PermissionError("Claim camera control to change battery capture or import photos.")
            session = self._refresh_session()
            state = self._state or {}
            if session is None or self._config_session != session or not self._config_supported or not state.get("available"):
                raise RuntimeError("Wait for supported battery capture on a native USB connection.")
            if self._job is not None or not self._idle():
                raise RuntimeError("Wait for the active operation and stop processing before using battery controls.")
            if action == "import" and state["record_count"] == 0:
                raise RuntimeError("No battery photos are available on the Nicla.")
            self._start(action, session, (controller_id, viewer_id))
        return self.status(viewer_id)

    def _start(self, action, session, authority=None):
        self._job = action
        self._reason = self._error = None
        if action == "import":
            self._last_import_count = 0
        self._worker = threading.Thread(target=self._run, args=(action, session, authority, self._refresh_generation),
                                        name="nicla-battery", daemon=True)
        self._worker.start()

    def _guard(self, session, authority):
        if self._session() != session:
            raise RuntimeError("The Nicla disconnected or restarted. Reconnect and retry.")
        if not self._idle():
            raise RuntimeError("Nicla processing resumed. Stop processing and retry.")
        if authority is not None and not self.broker._controller_valid(*authority):
            raise RuntimeError("Camera control changed. Claim it again and retry.")

    def _request(self, command, session, authority, deadline, **fields):
        pending = {"id": "battery-" + secrets.token_hex(10), "command": command,
                   "session": session, "event": threading.Event(), "ack": None}
        with self.lock:
            self._guard(session, authority)
            if time.monotonic() >= deadline:
                raise RuntimeError("Battery transfer exceeded its deadline. Reconnect and retry the import.")
            self._waiting = pending
        try:
            self.broker._write({"cmd": command, "command_id": pending["id"], "expected_boot_id": session[1], **fields})
            ack_deadline = min(time.monotonic() + _ACK_TIMEOUT, deadline)
            while not pending["event"].wait(max(0, min(.10, ack_deadline - time.monotonic()))):
                with self.lock:
                    self._guard(session, authority)
                if time.monotonic() >= ack_deadline:
                    raise RuntimeError(f"No reply from the Nicla to {command}.")
            ack = pending["ack"]
            if ack.get("control_status") != "completed":
                raise RuntimeError(f"Nicla rejected {command}: {str(ack.get('reason_code') or ack.get('control_status'))[:100]}.")
            return ack
        finally:
            with self.lock:
                if self._waiting is pending:
                    self._waiting = None

    @staticmethod
    def _board_state(ack):
        reason = ack.get("reason")
        reason = reason[:160] if isinstance(reason, str) else None
        if ack.get("available") is False and type(ack.get("armed")) is bool:
            return {"available": False, "armed": ack["armed"], "reason": reason}
        maximum, budget = ack.get("max_records"), ack.get("max_jpeg_bytes")
        if (type(ack.get("armed")) is not bool or type(ack.get("available")) is not bool
                or not _integer(maximum, 1, _MAX_RECORDS) or not _integer(budget, 4, _MAX_BYTES)
                or not _integer(ack.get("record_count"), 0, maximum) or not _integer(ack.get("jpeg_bytes"), 0, budget)):
            raise ValueError("Invalid battery status from the Nicla.")
        return {**{key: ack[key] for key in ("armed", "available", "max_records", "max_jpeg_bytes", "record_count", "jpeg_bytes")},
                "reason": reason}

    @staticmethod
    def _record(row, limit):
        if (not isinstance(row, dict) or not _integer(row.get("record_id"), 1, 0x0fffffff)
                or not isinstance(row.get("boot_id"), str) or not _BOOT_ID.fullmatch(row["boot_id"])
                or not _integer(row.get("frame_id"), 0, (1 << 63) - 1)
                or not _integer(row.get("ticks_ms"), 0, (1 << 63) - 1)
                or not _integer(row.get("byte_count"), 4, limit) or not isinstance(row.get("sha256"), str)
                or not _SHA256.fullmatch(row["sha256"]) or row.get("classification") not in ("known", "unknown")):
            raise ValueError("Invalid battery photo metadata from the Nicla.")
        identity = row.get("identity_id")
        if ((row["classification"] == "known" and (not isinstance(identity, str) or identity not in _ROSTER))
                or (row["classification"] == "unknown" and identity is not None)):
            raise ValueError("Battery photo identity metadata is inconsistent.")
        key = hashlib.sha256(json.dumps([row[k] for k in ("record_id", "boot_id", "frame_id", "sha256")],
                                        separators=(",", ":")).encode()).hexdigest()
        return key, row

    def _import(self, session, authority, deadline):
        state = self._board_state(self._request("battery_status", session, authority, deadline))
        if not state["available"]:
            raise RuntimeError(state["reason"] or "Battery photos are unavailable on this connection.")
        count, budget = state["record_count"], state["max_jpeg_bytes"]
        rows = []
        for index in range(count):
            ack = self._request("battery_list", session, authority, deadline, index=index)
            if ack.get("index") != index or ack.get("record_count") != count:
                raise ValueError("Battery photo list changed during import. Retry from the beginning.")
            rows.append(self._record(ack.get("record"), budget))
        if len({row["record_id"] for _, row in rows}) != count or sum(row["byte_count"] for _, row in rows) != state["jpeg_bytes"]:
            raise ValueError("Battery photo list failed its count or byte budget check.")
        for key, row in sorted(rows, key=lambda pair: pair[1]["record_id"]):
            if self.sightings.has_battery_import(key):
                continue
            record_id, length = row["record_id"], row["byte_count"]
            photo = bytearray()
            while len(photo) < length:
                offset = len(photo)
                size = min(_READ_SIZE, length - offset)
                ack = self._request("battery_read", session, authority, deadline,
                                    record_id=record_id, offset=offset, size=size)
                chunk = ack.get("chunk_hex")
                if (ack.get("record_id") != record_id or ack.get("offset") != offset or not isinstance(chunk, str)
                        or len(chunk) != size * 2 or not _HEX.fullmatch(chunk)):
                    raise ValueError("Battery photo chunk did not match the requested record and offset.")
                photo.extend(bytes.fromhex(chunk))
            if hashlib.sha256(photo).hexdigest() != row["sha256"]:
                raise ValueError("Battery photo checksum mismatch. No damaged photo was saved.")
            with self.lock:
                self._guard(session, authority)
            added = self.sightings.add_battery(record_key=key, record=row, jpeg=bytes(photo), imported_utc=self.broker.utc_now())
            if added:
                with self.lock:
                    self._last_import_count += 1
        return dict(state, unimported_count=0)

    def _run(self, action, session, authority, generation):
        deadline = time.monotonic() + (_IMPORT_TIMEOUT if action == "import" else _ACK_TIMEOUT)
        try:
            if action == "import":
                state = self._import(session, authority, deadline)
                with self.lock:
                    count = self._last_import_count
                reason = f"Imported {count} new {'photo' if count == 1 else 'photos'} to this computer. Board copies remain."
            else:
                state = self._board_state(self._request(f"battery_{action}", session, authority, deadline))
                reason = None
            with self.lock:
                if self._session() == session and generation == self._refresh_generation:
                    self._state, self._reason = state, reason
        except (OSError, RuntimeError, ValueError, SQLiteError) as error:
            with self.lock:
                if self._session() == session and generation == self._refresh_generation:
                    self._reason = self._error = str(error)[:200]
                    self._next_probe_at = time.monotonic() + 5.0
        finally:
            with self.lock:
                self._job = None

    def close(self):
        with self.lock:
            worker = self._worker
        if worker is not None:
            worker.join(timeout=_ACK_TIMEOUT + 1)
