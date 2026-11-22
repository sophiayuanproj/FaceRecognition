"""Save original Nicla JPEGs and a constant-rate Motion JPEG playback copy."""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import shutil
import statistics
import struct
import tempfile
from pathlib import Path


_JPEG_SOF = frozenset((0xC0, 0xC1, 0xC2, 0xC3, 0xC5, 0xC6, 0xC7,
                       0xC9, 0xCA, 0xCB, 0xCD, 0xCE, 0xCF))


def _jpeg_dimensions(data: bytes) -> tuple[int, int]:
    """Read dimensions from the JPEG frame header without decoding pixels."""
    if not isinstance(data, bytes) or not data.startswith(b"\xff\xd8") or not data.endswith(b"\xff\xd9"):
        raise ValueError("invalid JPEG start or end marker")
    position = 2
    dimensions = None
    while position < len(data) - 2:
        if data[position] != 0xFF:
            break
        while position < len(data) and data[position] == 0xFF:
            position += 1
        if position >= len(data):
            break
        marker = data[position]
        position += 1
        if marker == 0x01:  # JPEG TEM is a standalone marker.
            continue
        if position + 2 > len(data) - 2:
            break
        length = int.from_bytes(data[position:position + 2], "big")
        if length < 2 or position + length > len(data) - 2:
            break
        if marker in _JPEG_SOF:
            if length < 11 or length != 8 + 3 * data[position + 7]:
                break
            height, width = struct.unpack_from(">HH", data, position + 3)
            dimensions = (width, height)
        if marker == 0xDA and dimensions is not None and all(dimensions):
            return dimensions
        position += length
    raise ValueError("JPEG has no valid frame and scan headers")


def _validated_frames(frames: list[dict]) -> tuple[list[dict], int, int]:
    if not frames:
        raise ValueError("clip needs at least one frame")
    rows = []
    seen = set()
    dimensions = None
    for number, frame in enumerate(frames, 1):
        boot_id, frame_id = frame["boot_id"], frame["frame_id"]
        if not isinstance(boot_id, str) or not boot_id or type(frame_id) is not int or frame_id < 0:
            raise ValueError(f"invalid frame identity at clip frame {number}")
        key = (boot_id, frame_id)
        if key in seen:
            raise ValueError(f"duplicate frame identity at clip frame {number}")
        seen.add(key)
        data = frame["data"]
        width, height = _jpeg_dimensions(data)
        if (type(frame["width"]) is not int or type(frame["height"]) is not int or
                (width, height) != (frame["width"], frame["height"])):
            raise ValueError(f"JPEG dimension mismatch at clip frame {number}")
        if dimensions is None:
            dimensions = (width, height)
        if dimensions != (width, height) or max(dimensions) > 32767:
            raise ValueError("AVI requires fixed dimensions within 32767 pixels")
        digest = hashlib.sha256(data).hexdigest()
        if digest != frame["sha256"]:
            raise ValueError(f"JPEG SHA-256 mismatch at clip frame {number}")
        ticks, modulus = frame.get("capture_ticks_ms"), frame.get("ticks_modulus_ms")
        if ((ticks is not None and (type(ticks) is not int or ticks < 0)) or
                (modulus is not None and (type(modulus) is not int or modulus < 2)) or
                (ticks is not None and modulus is not None and ticks >= modulus)):
            raise ValueError(f"invalid capture clock at clip frame {number}")
        received_ns, received_utc = frame["host_received_monotonic_ns"], frame["host_received_utc"]
        if (type(received_ns) is not int or received_ns < 0 or
                not isinstance(received_utc, str) or not received_utc):
            raise ValueError(f"invalid host receipt clock at clip frame {number}")
        rows.append({"path": f"frames/frame-{number:06d}.jpg", "boot_id": boot_id,
                     "frame_id": frame_id, "width": width, "height": height, "sha256": digest,
                     "byte_count": len(data), "capture_ticks_ms": ticks, "ticks_modulus_ms": modulus,
                     "host_received_monotonic_ns": received_ns, "host_received_utc": received_utc})
        observation = frame.get("observation")
        if observation is not None:
            if observation.get("frame_sha256") != digest:
                raise ValueError(f"observation SHA-256 mismatch at clip frame {number}")
            rows[-1]["observation"] = observation
    return rows, *dimensions


def _playback_timing(rows: list[dict]) -> dict:
    intervals_ns = []
    board_intervals = host_intervals = 0
    for previous, current in zip(rows, rows[1:]):
        modulus = current["ticks_modulus_ms"]
        old_ticks, new_ticks = previous["capture_ticks_ms"], current["capture_ticks_ms"]
        if (current["boot_id"] == previous["boot_id"] and modulus is not None and
                modulus == previous["ticks_modulus_ms"] and old_ticks is not None and new_ticks is not None):
            delta = (new_ticks - old_ticks) % modulus
            if 0 < 2 * delta < modulus:
                intervals_ns.append(delta * 1_000_000)
                board_intervals += 1
                continue
        delta = current["host_received_monotonic_ns"] - previous["host_received_monotonic_ns"]
        if delta > 0:
            intervals_ns.append(delta)
            host_intervals += 1
    period_us = max(1, round(statistics.median(intervals_ns) / 1000)) if intervals_ns else 1_000_000
    if period_us > 0xFFFFFFFF:
        raise ValueError("capture interval exceeds AVI timing limits")
    return {"mode": "constant_frame_rate", "frame_interval_us": period_us,
            "fps": 1_000_000 / period_us, "board_capture_intervals": board_intervals,
            "host_receipt_intervals": host_intervals,
            "unusable_intervals": len(rows) - 1 - len(intervals_ns),
            "default_used": not intervals_ns,
            "basis": "Median adjacent positive interval, using wrap-aware board capture ticks "
                     "within a boot and host monotonic packet receipt otherwise. Defaults to 1 fps "
                     "without a usable interval. Original per-frame times remain in the frame list."}


def _chunk(fourcc: bytes, data: bytes) -> bytes:
    return struct.pack("<4sI", fourcc, len(data)) + data + b"\0" * (len(data) % 2)


def _write_avi(path: Path, frames: list[dict], width: int, height: int, period_us: int) -> None:
    frame_count = len(frames)
    max_frame = max(len(frame["data"]) for frame in frames)
    max_rate = math.ceil((max_frame + 9) * 1_000_000 / period_us)
    avih = struct.pack("<14I", period_us, min(max_rate, 0xFFFFFFFF), 0, 0x10,
                       frame_count, 0, 1, max_frame, width, height, 0, 0, 0, 0)
    strh = struct.pack("<4s4sIHHIIIIIIIIhhhh", b"vids", b"MJPG", 0, 0, 0,
                       0, period_us, 1_000_000, 0, frame_count, max_frame,
                       0xFFFFFFFF, 0, 0, 0, width, height)
    strf = struct.pack("<IiiHH4sIiiII", 40, width, height, 1, 24, b"MJPG",
                       max_frame, 0, 0, 0, 0)
    strl = _chunk(b"LIST", b"strl" + _chunk(b"strh", strh) + _chunk(b"strf", strf))
    hdrl = _chunk(b"LIST", b"hdrl" + _chunk(b"avih", avih) + strl)
    movi_size = 4 + sum(8 + len(frame["data"]) + len(frame["data"]) % 2 for frame in frames)
    index_size = 16 * frame_count
    riff_size = 4 + len(hdrl) + 8 + movi_size + 8 + index_size
    if riff_size > 0x7FFFFFFF:
        raise ValueError("clip exceeds the classic AVI size limit")

    with path.open("xb") as stream:
        stream.write(struct.pack("<4sI4s", b"RIFF", riff_size, b"AVI "))
        stream.write(hdrl)
        stream.write(struct.pack("<4sI4s", b"LIST", movi_size, b"movi"))
        for frame in frames:
            jpeg = frame["data"]
            stream.write(struct.pack("<4sI", b"00dc", len(jpeg)))
            stream.write(jpeg)
            if len(jpeg) % 2:
                stream.write(b"\0")
        stream.write(struct.pack("<4sI", b"idx1", index_size))
        offset = 4  # AVI index offsets start at the 'movi' list type.
        for frame in frames:
            size = len(frame["data"])
            stream.write(struct.pack("<4sIII", b"00dc", 0x10, offset, size))
            offset += 8 + size + size % 2


def write_clip(root: Path, clip_id: str, frames: list[dict], *, reason: str,
               mode: str = "capture_only", missed_frames: int = 0) -> dict:
    """Atomically publish a local clip directory and return its artifact paths."""
    if not isinstance(clip_id, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,95}", clip_id):
        raise ValueError("invalid clip ID")
    if not isinstance(reason, str) or not reason.strip():
        raise ValueError("clip needs a reason")
    rows, width, height = _validated_frames(frames)
    timing = _playback_timing(rows)
    root = Path(root)
    root.mkdir(mode=0o700, parents=True, exist_ok=True)
    directory = root / clip_id
    if directory.exists() or directory.is_symlink():
        raise FileExistsError(directory)
    staged = Path(tempfile.mkdtemp(prefix=f".{clip_id}-", dir=root))
    try:
        (staged / "frames").mkdir(mode=0o700)
        for frame, row in zip(frames, rows):
            (staged / row["path"]).write_bytes(frame["data"])
        _write_avi(staged / "clip.avi", frames, width, height, timing["frame_interval_us"])
        manifest = {"schema": "nicla-clip-v1", "clip_id": clip_id, "reason": reason,
                    "recording_mode": mode, "missed_frames": missed_frames,
                    "matched_observations": sum("observation" in row for row in rows),
                    "frame_count": len(rows), "width": width, "height": height,
                    "video": {"path": "clip.avi", "codec": "MJPG", "jpeg_bytes_preserved": True,
                              "playback_fps": timing["fps"]},
                    "timing": timing, "frames": rows}
        (staged / "manifest.json").write_text(json.dumps(manifest, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        os.rename(staged, directory)
    finally:
        if staged.exists():
            shutil.rmtree(staged)
    return {"clip_id": clip_id, "frame_count": len(rows), "playback_fps": timing["fps"],
            "directory": str(directory), "video_path": str(directory / "clip.avi"),
            "manifest_path": str(directory / "manifest.json")}
