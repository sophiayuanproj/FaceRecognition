"""Maintained portable surface over the supplied Windows HID/serial sources."""
from __future__ import annotations
from datetime import datetime, timezone
import hashlib
import importlib.util
from pathlib import Path
import sys
import time
from typing import Any, Callable, Iterable

ADAPTER_DIR = Path(__file__).resolve().parent
UPSTREAM_SCRIPTS = ADAPTER_DIR / "upstream" / "scripts"
UPSTREAM_POWER = ADAPTER_DIR / "upstream" / "power"
for _path in (UPSTREAM_POWER, UPSTREAM_SCRIPTS):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from fnirsi_protocol import PacketDecodeError, decode_report as _decode_report  # noqa: E402
from hardware_lock_windows import HardwareLock, default_lock_path  # noqa: E402
from nano_serial_capture_windows import NanoSerialCapture, describe_port  # noqa: E402


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def decode_raw_first(payload: bytes) -> dict:
    """Decode only after caller has persisted the raw payload."""
    raw = bytes(payload)
    samples = _decode_report(raw)
    return {"schema_version": "fnirsi-decoded-report-1",
            "raw_report_hex": raw.hex(), "raw_report_sha256": sha256_bytes(raw),
            "crc_status": "pass", "sample_count": len(samples), "samples": samples,
            "sample_time_policy": "unknown; report values are co-received and are not assigned sub-report times"}


def reconnect_candidates(previous: dict, candidates: Iterable[dict]) -> list[dict]:
    """Match fresh serial descriptors by hashed serial or full non-private descriptor tuple."""
    rows = list(candidates)
    identity = previous.get("serial_identity_sha256")
    if identity:
        return [row for row in rows if row.get("serial_identity_sha256") == identity]
    fields = ("vid", "pid", "product", "manufacturer")
    if any(previous.get(field) is None for field in fields):
        return []
    return [row for row in rows if all(row.get(field) == previous.get(field) for field in fields)]


def bounded_wait(predicate: Callable[[], Any], timeout_s: float, interval_s: float = 0.025,
                 monotonic: Callable[[], float] = time.monotonic) -> Any:
    """Poll with a monotonic deadline; portable to Windows without SIGALRM."""
    if timeout_s < 0 or interval_s <= 0:
        raise ValueError("timeout must be nonnegative and interval positive")
    deadline = monotonic() + timeout_s
    while True:
        value = predicate()
        if value:
            return value
        if monotonic() >= deadline:
            raise TimeoutError("bounded device operation timed out")
        time.sleep(min(interval_s, max(0.0, deadline - monotonic())))


def validate_capture_metadata(metadata: dict) -> list[str]:
    errors = []
    board = metadata.get("selected_board")
    if not metadata.get("fresh_interface_selection"):
        errors.append("meter interface was not selected from fresh enumeration")
    if not isinstance(board, dict):
        if metadata.get("board_provenance_status") != "not_applicable_passive_load":
            errors.append("active board provenance is absent")
    else:
        if not board.get("board_type") or not board.get("board_label"):
            errors.append("selected board identity is incomplete")
        if not board.get("reset_id"):
            errors.append("reset/session id is required")
        for field in ("firmware_source_sha256", "configuration_sha256"):
            value = board.get(field)
            if not isinstance(value, str) or len(value) != 64:
                errors.append("selected board is missing " + field)
        payload = board.get("release_payload_sha256") or board.get("firmware_binary_sha256")
        if not isinstance(payload, str) or len(payload) != 64:
            errors.append("selected board is missing release_payload_sha256")
    return errors


def steady_average_module():
    path = ADAPTER_DIR / "steady_average.py"
    spec = importlib.util.spec_from_file_location("nicla_steady_average_adapter", path)
    if spec is None or spec.loader is None:
        raise ImportError("cannot load maintained steady-window estimator")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def fresh_serial_descriptor(port_name: str, list_ports_module: Any = None) -> dict:
    if list_ports_module is None:
        import serial.tools.list_ports as list_ports_module
    matches = [port for port in list_ports_module.comports()
               if str(getattr(port, "device", "")).casefold() == port_name.casefold()]
    if len(matches) != 1:
        raise ValueError("selected serial port must match exactly one freshly enumerated descriptor")
    return describe_port(matches[0])


def verify_active_firmware_binding(board_type: str, source: Path | str | None,
                                   binary: Path | str, configuration: Path | str,
                                   expected_source_sha256: str | None = None,
                                   expected_release_payload_sha256: str | None = None,
                                   expected_configuration_sha256: str | None = None) -> dict:
    """Bind source, versioned release payload and config; `binary` is legacy API naming."""
    files = {"release_payload": Path(binary), "configuration": Path(configuration)}
    source_path = Path(source) if source is not None else None
    if source_path is not None:
        files["source"] = source_path
    missing = [name for name, path in files.items() if not path.is_file()]
    if missing:
        raise FileNotFoundError("missing selected-board files: " + ", ".join(missing))
    release_hash = _sha256_file(files["release_payload"])
    source_file_hash = _sha256_file(source_path) if source_path is not None else None
    source_hash = source_file_hash or expected_source_sha256
    if source_hash is None:
        raise ValueError("firmware source file or source SHA-256 from the deployment manifest is required")
    configuration_hash = _sha256_file(files["configuration"])
    for label, expected, actual in (
        ("firmware source", expected_source_sha256, source_hash),
        ("release payload", expected_release_payload_sha256, release_hash),
        ("board configuration", expected_configuration_sha256, configuration_hash),
    ):
        if expected is not None:
            if len(expected) != 64 or any(ch not in "0123456789abcdefABCDEF" for ch in expected):
                raise ValueError("expected " + label + " SHA-256 must contain 64 hexadecimal characters")
            if actual != expected.lower():
                raise ValueError("selected " + label + " SHA-256 does not match the declared deployment manifest")
    legacy_semantics = (
        "legacy compatibility alias of release_payload_sha256; not a compiled MCU firmware binary"
        if board_type == "nicla_vision" else
        "legacy compatibility alias of release_payload_sha256"
    )
    return {"board_type": board_type, "firmware_source_sha256": source_hash,
            "firmware_source_file_sha256": source_file_hash,
            "firmware_source_file_available": source_file_hash is not None,
            "firmware_source_hash_origin": "local_source_file" if source_file_hash else "deployment_manifest",
            "release_payload_sha256": release_hash,
            "firmware_binary_sha256": release_hash,
            "firmware_binary_sha256_semantics": legacy_semantics,
            "configuration_sha256": configuration_hash,
            "verified_utc": datetime.now(timezone.utc).isoformat(timespec="seconds")}


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


__all__ = ["HardwareLock", "NanoSerialCapture", "PacketDecodeError", "bounded_wait",
           "decode_raw_first", "default_lock_path", "describe_port", "fresh_serial_descriptor",
           "reconnect_candidates", "steady_average_module", "validate_capture_metadata",
           "verify_active_firmware_binding"]
