"""Bounded local encounter JPEGs for the Nicla's unattended operation.

Each committed .rec is NSP1, a little-endian uint16 JSON length, a 32-byte
JSON SHA-256 digest, the UTF-8 JSON, then the original JPEG. The JSON
includes the JPEG byte count and SHA-256.
List/read work without modifying the filesystem. A serial host can list the
records and fetch verified 2 KiB chunks, or split files from a read-only FAT
mount using the same format. Board ticks identify order within a boot.
"""

import binascii
import hashlib
import json
import os
import struct


MAX_RECORDS = 24
MAX_JPEG_BYTES = 96 * 1024
MAX_READ_BYTES = 2048
_MAX_HEADER_BYTES = 768
_MAX_RECORD_ID = 0x0FFFFFFF
_MAGIC = b"NSP1"
_STAGED = "write.tmp"
_ROSTER = ("K01", "K02", "K03", "K04", "K05")


def _name(record_id):
    return "s%07x.rec" % record_id


def _record_id(name):
    if len(name) != 12 or name[0] != "s" or name[8:] != ".rec":
        return None
    digits = name[1:8]
    if any(digit not in "0123456789abcdef" for digit in digits):
        return None
    value = int(digits, 16)
    return value if value else None


def _sha256_hex(digest):
    return binascii.hexlify(digest.digest()).decode("ascii")


def _missing(error):
    return bool(error.args and error.args[0] == 2)


def _sync():
    sync = getattr(os, "sync", None)
    if sync is not None:
        sync()


def _write(stream, data):
    if stream.write(data) != len(data):
        raise OSError("incomplete offline sighting write")


class OfflineSightings:
    """One caller owns this spool while the board is writing its flash."""

    def __init__(self, root):
        self.root = str(root).rstrip("/")
        if not self.root:
            raise ValueError("offline sightings requires a storage directory")

    def _path(self, name):
        return self.root + "/" + name

    def _names(self):
        try:
            ilistdir = getattr(os, "ilistdir", None)
            if ilistdir is not None:
                for entry in ilistdir(self.root):
                    yield entry[0]
            else:
                for name in os.listdir(self.root):
                    yield name
        except OSError as error:
            if not _missing(error):
                raise

    def _remove(self, name):
        try:
            os.remove(self._path(name))
        except OSError as error:
            if not _missing(error):
                raise

    def _header(self, stream, file_size, record_id):
        prefix = stream.read(6)
        if len(prefix) != 6:
            raise ValueError("truncated sighting prefix")
        magic, header_size = struct.unpack("<4sH", prefix)
        if magic != _MAGIC or not 1 <= header_size <= _MAX_HEADER_BYTES:
            raise ValueError("invalid sighting prefix")
        checksum = stream.read(32)
        data = stream.read(header_size)
        if len(data) != header_size:
            raise ValueError("truncated sighting header")
        if hashlib.sha256(data).digest() != checksum:
            raise ValueError("sighting header SHA-256 mismatch")
        header = json.loads(data.decode("utf-8"))
        if not isinstance(header, dict) or header.get("record_id") != record_id:
            raise ValueError("invalid sighting record ID")
        size = header.get("byte_count")
        sha256 = header.get("sha256")
        if (type(size) is not int or not 4 <= size <= MAX_JPEG_BYTES or
                file_size != 38 + header_size + size or not isinstance(sha256, str) or
                len(sha256) != 64 or any(ch not in "0123456789abcdef" for ch in sha256)):
            raise ValueError("invalid sighting photo length or digest")
        return header, 38 + header_size

    def _verify(self, name, record_id):
        path = self._path(name)
        file_size = os.stat(path)[6]
        with open(path, "rb") as stream:
            header, _offset = self._header(stream, file_size, record_id)
            digest = hashlib.sha256()
            remaining = header["byte_count"]
            while remaining:
                block = stream.read(min(MAX_READ_BYTES, remaining))
                if not block:
                    raise ValueError("truncated sighting JPEG")
                digest.update(block)
                remaining -= len(block)
            if _sha256_hex(digest) != header["sha256"]:
                raise ValueError("sighting JPEG SHA-256 mismatch")
        return header

    def _scan(self):
        committed = []
        invalid = []
        staged = False
        for name in self._names():
            if name == _STAGED:
                staged = True
                continue
            record_id = _record_id(name)
            if record_id is None:
                continue
            if len(committed) + len(invalid) >= MAX_RECORDS:
                raise OSError("too many offline sighting files")
            try:
                header = self._verify(name, record_id)
            except ValueError:
                invalid.append(name)
            else:
                committed.append(header)
        committed.sort(key=lambda row: row["record_id"])
        return committed, invalid, staged

    def list_records(self):
        records, _invalid, _staged = self._scan()
        records.reverse()
        return records

    def status(self):
        records, invalid, staged = self._scan()
        return {"record_count": len(records),
                "jpeg_bytes": sum(row["byte_count"] for row in records),
                "max_records": MAX_RECORDS, "max_jpeg_bytes": MAX_JPEG_BYTES,
                "invalid_count": len(invalid) + int(staged)}

    def save(self, jpeg, *, boot_id, frame_id, ticks_ms, ticks_modulus_ms=None,
             face_count=None, decision=None):
        if not isinstance(jpeg, (bytes, bytearray, memoryview)) or not 4 <= len(jpeg) <= MAX_JPEG_BYTES:
            raise ValueError("offline sighting needs a bounded JPEG")
        if jpeg[0] != 255 or jpeg[1] != 216 or jpeg[-2] != 255 or jpeg[-1] != 217:
            raise ValueError("offline sighting JPEG markers are invalid")
        if (not isinstance(boot_id, str) or not 1 <= len(boot_id) <= 96 or
                any(ch not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789._:-" for ch in boot_id) or
                type(frame_id) is not int or frame_id < 0 or type(ticks_ms) is not int or ticks_ms < 0):
            raise ValueError("invalid offline sighting clock or frame ID")
        if (ticks_modulus_ms is not None and
                (type(ticks_modulus_ms) is not int or ticks_ms >= ticks_modulus_ms)):
            raise ValueError("invalid offline sighting tick modulus")
        records, invalid, staged = self._scan()
        for row in records:
            if row["boot_id"] == boot_id and row["frame_id"] == frame_id:
                return row
        record_id = 1 + max([0] + [row["record_id"] for row in records] +
                            [_record_id(name) for name in invalid])
        if record_id > _MAX_RECORD_ID:
            raise OSError("offline sighting record IDs exhausted")
        decision = decision if isinstance(decision, dict) else {}
        candidate = decision.get("candidate")
        known = (decision.get("status") == "KNOWN_IDENTITY" and
                 decision.get("known_acceptance_enabled") is True and candidate in _ROSTER)
        digest = hashlib.sha256()
        digest.update(jpeg)
        header = {"record_id": record_id, "boot_id": boot_id, "frame_id": frame_id,
                  "ticks_ms": ticks_ms, "ticks_modulus_ms": ticks_modulus_ms,
                  "byte_count": len(jpeg), "sha256": _sha256_hex(digest),
                  "classification": "known" if known else "unknown", "identity_id": candidate if known else None}
        if type(face_count) is int and 1 <= face_count <= 8:
            header["face_count"] = face_count
        encoded = json.dumps(header, separators=(",", ":")).encode("utf-8")
        if len(encoded) > _MAX_HEADER_BYTES:
            raise ValueError("offline sighting header too large")
        try:
            os.stat(self.root)
        except OSError as error:
            if not _missing(error):
                raise
            os.mkdir(self.root)
        if staged:
            self._remove(_STAGED)
        for name in invalid:
            self._remove(name)
        retained_bytes = sum(row["byte_count"] for row in records)
        while len(records) >= MAX_RECORDS or retained_bytes + len(jpeg) > MAX_JPEG_BYTES:
            old = records.pop(0)
            self._remove(_name(old["record_id"]))
            retained_bytes -= old["byte_count"]
        try:
            with open(self._path(_STAGED), "wb") as stream:
                _write(stream, struct.pack("<4sH", _MAGIC, len(encoded)))
                _write(stream, hashlib.sha256(encoded).digest())
                _write(stream, encoded)
                _write(stream, jpeg)
                stream.flush()
            _sync()
            self._verify(_STAGED, record_id)
            os.rename(self._path(_STAGED), self._path(_name(record_id)))
            _sync()
        except Exception:
            self._remove(_STAGED)
            raise
        return header

    def read(self, record_id, offset=0, size=1024):
        if (type(record_id) is not int or not 1 <= record_id <= _MAX_RECORD_ID or
                type(offset) is not int or offset < 0 or
                type(size) is not int or not 1 <= size <= MAX_READ_BYTES):
            raise ValueError("invalid offline sighting read range")
        path = self._path(_name(record_id))
        file_size = os.stat(path)[6]
        with open(path, "rb") as stream:
            header, jpeg_offset = self._header(stream, file_size, record_id)
            remaining = header["byte_count"] - offset
            if remaining < 0:
                raise ValueError("offline sighting offset exceeds JPEG length")
            if not remaining:
                return b""
            stream.seek(jpeg_offset + offset)
            return stream.read(min(remaining, size))

    def clear(self):
        removed = False
        for name in self._names():
            if name == _STAGED or _record_id(name) is not None:
                self._remove(name)
                removed = True
        if removed:
            _sync()
        return self.status()
