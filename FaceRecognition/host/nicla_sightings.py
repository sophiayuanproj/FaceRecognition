"""Bounded local photo history and persistent Nicla encounter counts."""

from __future__ import annotations

import hashlib
import json
import os
import re
import sqlite3
from contextlib import closing
from pathlib import Path


_MAX_ITEMS = 100
_MAX_JPEG_BYTES = 8 * 1024 * 1024
_MAX_PHOTO_BYTES = 1_000_000
_MAX_BOOT_WATERMARKS = 32
_SQLITE_INT_MAX = (1 << 63) - 1
_ROSTER = ("K01", "K02", "K03", "K04", "K05")
_BOOT_ID = re.compile(r"[A-Za-z0-9._:-]{1,128}")
_STATUS = re.compile(r"[A-Za-z0-9_]{1,64}")
_RECORD_KEY = re.compile(r"[0-9a-f]{64}")
_TEXT_FIELDS = ("host_received_utc", "policy_id", "model_id", "firmware_sha256", "config_sha256")
_INTEGER_FIELDS = ("width", "height", "capture_ticks_ms", "host_received_monotonic_ns")
_SCHEMA = (
    "CREATE TABLE IF NOT EXISTS counters (name TEXT PRIMARY KEY, count INTEGER NOT NULL)",
    "CREATE TABLE IF NOT EXISTS recent_boots (boot_id TEXT PRIMARY KEY, "
    "frame_id INTEGER NOT NULL, last_encounter_id INTEGER NOT NULL)",
    "CREATE TABLE IF NOT EXISTS sightings (id INTEGER PRIMARY KEY, boot_id TEXT NOT NULL, "
    "encounter_frame_id INTEGER NOT NULL, metadata TEXT NOT NULL, jpeg BLOB NOT NULL, "
    "UNIQUE (boot_id, encounter_frame_id))",
    "CREATE TABLE IF NOT EXISTS battery_imports (record_key TEXT PRIMARY KEY, sighting_id INTEGER NOT NULL UNIQUE)",
)


def _valid_frame_id(value: object) -> bool:
    return type(value) is int and 0 <= value <= _SQLITE_INT_MAX


def _photo_metadata(boot_id: str, frame_id: int, encounter_frame_id: int, jpeg: bytes,
                    metadata: dict) -> dict:
    if not isinstance(metadata, dict):
        raise ValueError("sighting metadata must be a dictionary")
    sha256 = hashlib.sha256(jpeg).hexdigest()
    if metadata.get("sha256", sha256) != sha256:
        raise ValueError("sighting JPEG SHA-256 mismatch")
    decision = metadata.get("decision")
    if not isinstance(decision, dict):
        decision = {}
    status = decision.get("status")
    if not isinstance(status, str) or not _STATUS.fullmatch(status):
        status = "UNAVAILABLE"
    gate_enabled = decision.get("known_acceptance_enabled") is True
    candidate = decision.get("candidate")
    identity = candidate if status == "KNOWN_IDENTITY" and gate_enabled and candidate in _ROSTER else None
    result = {"boot_id": boot_id, "frame_id": frame_id, "encounter_frame_id": encounter_frame_id,
              "sha256": sha256, "byte_count": len(jpeg), "decision_status": status,
              "classification": "known" if identity is not None else "unknown",
              "identity_id": identity, "known_acceptance_enabled": gate_enabled}
    for key in _TEXT_FIELDS:
        value = metadata.get(key)
        if isinstance(value, str) and 0 < len(value) <= 128:
            result[key] = value
    for key in _INTEGER_FIELDS:
        value = metadata.get(key)
        if _valid_frame_id(value):
            result[key] = value
    face_count = metadata.get("face_count")
    if type(face_count) is int and 1 <= face_count <= 8:
        result["face_count"] = face_count
    return result


def _gallery_item(item_id: int, metadata: str) -> dict:
    photo = json.loads(metadata)
    manual = "assigned_person_id" in photo
    device_accepted = (photo.get("classification") == "known"
                       and photo.get("decision_status") == "KNOWN_IDENTITY"
                       and photo.get("known_acceptance_enabled") is True)
    device_person = photo.get("identity_id") if device_accepted else None
    person_id = photo.get("assigned_person_id") if manual else device_person
    assignment = "manual" if manual else ("device" if person_id in _ROSTER else "unassigned")
    return dict(photo, id=item_id, assigned_person_id=photo.get("assigned_person_id"),
                person_id=person_id if person_id in _ROSTER else None, person_assignment=assignment)


class NiclaSightingsStore:
    """Each call uses its own SQLite connection, with atomic serialized writes.

    Only a write creates storage. Encounter counts survive photo eviction.
    Frame IDs must arrive in order within a boot. The latest 32 boot watermarks
    suppress duplicate or stale additions even after their photos are evicted.
    """

    def __init__(self, path: Path, *, max_items: int = _MAX_ITEMS,
                 max_jpeg_bytes: int = _MAX_JPEG_BYTES):
        if type(max_items) is not int or not 1 <= max_items <= _MAX_ITEMS:
            raise ValueError("sighting item limit must be 1–100")
        if type(max_jpeg_bytes) is not int or not 1 <= max_jpeg_bytes <= _MAX_JPEG_BYTES:
            raise ValueError("sighting JPEG budget must be 1 byte–8 MiB")
        self.path = Path(path)
        self.max_items = max_items
        self.max_jpeg_bytes = max_jpeg_bytes

    def _writer(self) -> sqlite3.Connection:
        self.path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        try:
            descriptor = os.open(self.path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
        except FileExistsError:
            pass
        else:
            os.close(descriptor)
        connection = sqlite3.connect(self.path, timeout=8)
        try:
            connection.execute("PRAGMA auto_vacuum = FULL")
            connection.execute("PRAGMA secure_delete = ON")
            connection.execute("BEGIN IMMEDIATE")
            for statement in _SCHEMA:
                connection.execute(statement)
        except BaseException:
            connection.close()
            raise
        return connection

    def _reader(self) -> sqlite3.Connection:
        return sqlite3.connect(self.path.resolve().as_uri() + "?mode=ro", uri=True, timeout=8)

    @staticmethod
    def _schema_ready(connection: sqlite3.Connection, table: str = "sightings") -> bool:
        return connection.execute("SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = ?", (table,)).fetchone() is not None

    @staticmethod
    def _increment(connection: sqlite3.Connection, name: str) -> int:
        connection.execute("INSERT INTO counters VALUES (?, 1) "
                           "ON CONFLICT(name) DO UPDATE SET count = count + 1", (name,))
        return connection.execute("SELECT count FROM counters WHERE name = ?", (name,)).fetchone()[0]

    def _trim(self, connection: sqlite3.Connection) -> None:
        rows = connection.execute("SELECT id, length(jpeg) FROM sightings ORDER BY id").fetchall()
        count, byte_count = len(rows), sum(size for _, size in rows)
        for item_id, size in rows:
            if count <= self.max_items and byte_count <= self.max_jpeg_bytes:
                break
            connection.execute("DELETE FROM sightings WHERE id = ?", (item_id,))
            count -= 1
            byte_count -= size
        connection.execute("DELETE FROM recent_boots WHERE boot_id IN (SELECT boot_id FROM recent_boots "
                           "ORDER BY last_encounter_id DESC LIMIT -1 OFFSET ?)", (_MAX_BOOT_WATERMARKS,))

    def add(self, *, boot_id: str, frame_id: int, jpeg: bytes, metadata: dict,
            encounter_frame_id: int | None = None) -> dict:
        """Add one encounter or upgrade a retained unknown using a later known frame."""
        if not isinstance(boot_id, str) or not _BOOT_ID.fullmatch(boot_id) or not _valid_frame_id(frame_id):
            raise ValueError("invalid sighting boot or frame ID")
        first_frame = frame_id if encounter_frame_id is None else encounter_frame_id
        if not _valid_frame_id(first_frame) or first_frame > frame_id:
            raise ValueError("invalid encounter frame ID")
        if (not isinstance(jpeg, bytes) or not 4 <= len(jpeg) <= min(_MAX_PHOTO_BYTES, self.max_jpeg_bytes)
                or not jpeg.startswith(b"\xff\xd8") or not jpeg.endswith(b"\xff\xd9")):
            raise ValueError("sighting photo must be a bounded JPEG")
        photo = _photo_metadata(boot_id, frame_id, first_frame, jpeg, metadata)
        encoded = json.dumps(photo, separators=(",", ":"))
        identity = photo["identity_id"]
        with closing(self._writer()) as connection, connection:
            existing = connection.execute("SELECT id, metadata FROM sightings WHERE boot_id = ? "
                                          "AND encounter_frame_id = ?", (boot_id, first_frame)).fetchone()
            previous = connection.execute("SELECT frame_id FROM recent_boots WHERE boot_id = ?",
                                          (boot_id,)).fetchone()
            result = {"id": existing[0] if existing else None, "added": False,
                      "updated": False, "retained": existing is not None}
            if previous is not None and frame_id <= previous[0]:
                return result
            if encounter_frame_id is None:
                if existing is not None:
                    return result
                item_id = self._increment(connection, "total")
                connection.execute("INSERT INTO sightings VALUES (?, ?, ?, ?, ?)",
                                   (item_id, boot_id, first_frame, encoded, jpeg))
                result.update(id=item_id, added=True, retained=True)
            else:
                if existing is None or identity is None:
                    return result
                previous_photo = json.loads(existing[1])
                if previous_photo["identity_id"] is not None:
                    return result
                if "assigned_person_id" in previous_photo:
                    photo["assigned_person_id"] = previous_photo["assigned_person_id"]
                    encoded = json.dumps(photo, separators=(",", ":"))
                item_id = existing[0]
                connection.execute("UPDATE sightings SET metadata = ?, jpeg = ? WHERE id = ?",
                                   (encoded, jpeg, item_id))
                result["updated"] = True
            if identity is not None:
                self._increment(connection, identity)
            connection.execute("INSERT INTO recent_boots VALUES (?, ?, ?) ON CONFLICT(boot_id) "
                               "DO UPDATE SET frame_id = excluded.frame_id, last_encounter_id = "
                               "max(recent_boots.last_encounter_id, excluded.last_encounter_id)",
                               (boot_id, frame_id, item_id))
            self._trim(connection)
            result["retained"] = connection.execute("SELECT 1 FROM sightings WHERE id = ?",
                                                    (item_id,)).fetchone() is not None
            return result

    def has_battery_import(self, record_key: str) -> bool:
        if not isinstance(record_key, str) or not _RECORD_KEY.fullmatch(record_key):
            raise ValueError("invalid battery record key")
        if not self.path.exists():
            return False
        with closing(self._reader()) as connection:
            return self._schema_ready(connection, "battery_imports") and connection.execute(
                "SELECT 1 FROM battery_imports WHERE record_key = ?", (record_key,)).fetchone() is not None

    def add_battery(self, *, record_key: str, record: dict, jpeg: bytes, imported_utc: str) -> bool:
        """Commit a verified JPEG and its idempotence receipt in one transaction."""
        if not isinstance(record_key, str) or not _RECORD_KEY.fullmatch(record_key):
            raise ValueError("invalid battery record key")
        boot_id, frame_id = record.get("boot_id"), record.get("frame_id")
        if not isinstance(boot_id, str) or not _BOOT_ID.fullmatch(boot_id) or not _valid_frame_id(frame_id):
            raise ValueError("invalid battery boot or frame ID")
        if (not isinstance(jpeg, bytes) or not 4 <= len(jpeg) <= min(_MAX_PHOTO_BYTES, self.max_jpeg_bytes)
                or not jpeg.startswith(b"\xff\xd8") or not jpeg.endswith(b"\xff\xd9")):
            raise ValueError("battery sighting must be a bounded JPEG")
        if not isinstance(imported_utc, str) or not 0 < len(imported_utc) <= 128:
            raise ValueError("invalid battery import timestamp")
        known = record.get("classification") == "known" and record.get("identity_id") in _ROSTER
        metadata = {"sha256": record.get("sha256"), "capture_ticks_ms": record.get("ticks_ms"),
                    "face_count": record.get("face_count"), "decision": {
                        "status": "KNOWN_IDENTITY" if known else "UNKNOWN",
                        "known_acceptance_enabled": known,
                        "candidate": record.get("identity_id") if known else None}}
        photo = _photo_metadata(boot_id, frame_id, frame_id, jpeg, metadata)
        photo.update(source="on_nicla_battery", imported_utc=imported_utc,
                     capture_time_available=False, board_record_id=record.get("record_id"))
        if _valid_frame_id(record.get("ticks_modulus_ms")):
            photo["ticks_modulus_ms"] = record["ticks_modulus_ms"]
        with closing(self._writer()) as connection, connection:
            if connection.execute("SELECT 1 FROM battery_imports WHERE record_key = ?", (record_key,)).fetchone():
                return False
            item_id = self._increment(connection, "total")
            self._increment(connection, "battery_total")
            if photo["identity_id"] is not None:
                self._increment(connection, photo["identity_id"])
            connection.execute("INSERT INTO sightings VALUES (?, ?, ?, ?, ?)",
                               (item_id, "battery:" + record_key, frame_id, json.dumps(photo, separators=(",", ":")), jpeg))
            connection.execute("INSERT INTO battery_imports VALUES (?, ?)", (record_key, item_id))
            self._trim(connection)
        return True

    def remove_battery_photo(self, sighting_id: int) -> bool:
        """Delete a retained host copy while keeping its import receipt and counters."""
        if not _valid_frame_id(sighting_id) or sighting_id == 0 or not self.path.exists():
            return False
        with closing(self._writer()) as connection, connection:
            return connection.execute("DELETE FROM sightings WHERE id = ? AND id IN "
                                      "(SELECT sighting_id FROM battery_imports WHERE sighting_id = ?)",
                                      (sighting_id, sighting_id)).rowcount == 1

    def assign_person(self, sighting_id: int, person_id: str | None) -> dict:
        """Save a manual group without changing the device decision or its counters."""
        if not _valid_frame_id(sighting_id) or sighting_id == 0:
            raise ValueError("Invalid sighting ID.")
        if person_id is not None and person_id not in _ROSTER:
            raise ValueError("Choose Person 1–5 or Other.")
        if not self.path.exists():
            raise KeyError(sighting_id)
        with closing(self._writer()) as connection, connection:
            row = connection.execute("SELECT metadata FROM sightings WHERE id = ?", (sighting_id,)).fetchone()
            if row is None:
                raise KeyError(sighting_id)
            photo = json.loads(row[0])
            photo["assigned_person_id"] = person_id
            connection.execute("UPDATE sightings SET metadata = ? WHERE id = ?",
                               (json.dumps(photo, separators=(",", ":")), sighting_id))
        return {"sighting_id": sighting_id, "person_id": person_id, "person_assignment": "manual"}

    def summary(self, limit: int = _MAX_ITEMS) -> dict:
        """Return a consistent snapshot of counts and newest retained entries."""
        if type(limit) is not int or not 0 <= limit <= _MAX_ITEMS:
            raise ValueError("sighting summary limit must be 0–100")
        result = {"total_count": 0, "battery_import_count": 0, "identity_counts": dict.fromkeys(_ROSTER, 0),
                  "retained_count": 0, "retained_jpeg_bytes": 0, "items": []}
        if not self.path.exists():
            return result
        with closing(self._reader()) as connection, connection:
            connection.execute("BEGIN")
            if not self._schema_ready(connection):
                return result
            counts = dict(connection.execute("SELECT name, count FROM counters"))
            result["total_count"] = counts.get("total", 0)
            result["battery_import_count"] = counts.get("battery_total", 0)
            result["identity_counts"] = {identity: counts.get(identity, 0) for identity in _ROSTER}
            result["retained_count"], result["retained_jpeg_bytes"] = connection.execute(
                "SELECT count(*), coalesce(sum(length(jpeg)), 0) FROM sightings").fetchone()
            result["items"] = [_gallery_item(item_id, metadata) for item_id, metadata in
                               connection.execute("SELECT id, metadata FROM sightings ORDER BY id DESC LIMIT ?",
                                                  (limit,))]
        return result

    def get_jpeg(self, sighting_id: int) -> tuple[bytes, dict]:
        """Return a retained frame and its metadata for a positive SQLite ID."""
        if not _valid_frame_id(sighting_id) or sighting_id == 0 or not self.path.exists():
            raise KeyError("unknown sighting")
        with closing(self._reader()) as connection, connection:
            connection.execute("BEGIN")
            row = connection.execute("SELECT jpeg, metadata FROM sightings WHERE id = ?",
                                     (sighting_id,)).fetchone() if self._schema_ready(connection) else None
            if row is None:
                raise KeyError("unknown sighting")
            jpeg, metadata = row
            return jpeg, dict(json.loads(metadata), id=sighting_id)
