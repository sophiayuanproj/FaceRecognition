"""Host/environment and source-artifact inventory; does not train or open face data."""
from __future__ import annotations

import csv
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import re
import shutil
import subprocess
import sys


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _system_memory_bytes() -> int | None:
    if sys.platform == "darwin":
        try:
            result = subprocess.run(["sysctl", "-n", "hw.memsize"], capture_output=True,
                                    text=True, timeout=5, check=False)
            if result.returncode == 0:
                return int(result.stdout.strip())
        except Exception:
            pass
    if os.name == "nt":
        try:
            import ctypes
            class MemoryStatus(ctypes.Structure):
                _fields_ = [("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong),
                            ("ullTotalPhys", ctypes.c_ulonglong), ("ullAvailPhys", ctypes.c_ulonglong),
                            ("ullTotalPageFile", ctypes.c_ulonglong), ("ullAvailPageFile", ctypes.c_ulonglong),
                            ("ullTotalVirtual", ctypes.c_ulonglong), ("ullAvailVirtual", ctypes.c_ulonglong),
                            ("ullAvailExtendedVirtual", ctypes.c_ulonglong)]
            status = MemoryStatus()
            status.dwLength = ctypes.sizeof(status)
            if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
                return int(status.ullTotalPhys)
        except Exception:
            return None
    try:
        import psutil
        return int(psutil.virtual_memory().total)
    except Exception:
        if sys.platform == "darwin":
            try:
                result = subprocess.run(["system_profiler", "SPHardwareDataType", "-detailLevel", "mini"],
                                        capture_output=True, text=True, timeout=20, check=False)
                match = re.search(r"^\s*Memory:\s*([0-9.]+)\s*(GB|MB)\s*$", result.stdout, re.MULTILINE | re.IGNORECASE)
                if result.returncode == 0 and match:
                    scale = 1024 ** (3 if match.group(2).upper() == "GB" else 2)
                    return int(float(match.group(1)) * scale)
            except Exception:
                pass
        return None


def system_snapshot(workspace: Path) -> dict:
    stat = shutil.disk_usage(workspace)
    return {
        "captured_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "platform": platform.platform(), "os_name": os.name,
        "machine": platform.machine(), "python": sys.version,
        "logical_cpu_count": os.cpu_count(), "system_memory_bytes": _system_memory_bytes(),
        "workspace_disk": {"total_bytes": stat.total, "used_bytes": stat.used,
                           "free_bytes": stat.free, "free_gib": round(stat.free / (1024 ** 3), 2)},
    }


def source_manifest(paths: list[Path], workspace: Path) -> list[dict]:
    result = []
    for path in paths:
        if path.is_file():
            try:
                rel = path.resolve().relative_to(workspace.resolve()).as_posix()
            except ValueError:
                rel = None
            result.append({"path": rel, "absolute_local_path": str(path.resolve()),
                           "bytes": path.stat().st_size, "sha256": sha256_file(path)})
    return result


def model_inventory_summary(csv_path: Path) -> dict:
    if not csv_path.is_file():
        return {"path": str(csv_path), "status": "missing"}
    rows = list(csv.DictReader(csv_path.open("r", newline="", encoding="utf-8-sig")))
    hash_keys = [key for key in (rows[0].keys() if rows else []) if "sha256" in key.lower() or "hash" in key.lower()]
    hashes = set()
    for row in rows:
        for key in hash_keys:
            value = (row.get(key) or "").strip().lower()
            if len(value) == 64 and all(c in "0123456789abcdef" for c in value):
                hashes.add(value)
    return {"path": str(csv_path), "rows": len(rows), "hash_columns": hash_keys,
            "distinct_sha256_values": len(hashes), "status": "read"}


def make_inventory(workspace: Path, study_root: Path) -> dict:
    key_paths = [
        workspace / "FaceRecognition/README.md",
        workspace / "work/research-complete-v3/STUDY_PROTOCOL.md",
        workspace / "work/research-complete-v3/START_HERE.md",
        workspace / "work/research-complete-v3/reports/metric-repair-verification.json",
        workspace / "work/research-complete-v3/reports/host-qualification-summary-v3.json",
        workspace / "work/research-complete-v3/MODEL_INVENTORY.csv",
        workspace / "work/windows-fnirsi-results-import-2026-10-28/INDEPENDENT_INTEGRITY_CHECK.json",
        workspace / "work/windows-fnirsi-results-import-2026-10-28/CURRENT_WINDOWS_PACKAGE_SHA256.json",
        workspace / "work/windows-fnirsi-results-import-2026-10-28/WINDOWS_READINESS.json",
    ]
    app_model = workspace / "FaceRecognition/models/identity_embedding_int8.tflite"
    if app_model.exists():
        key_paths.append(app_model)
    return {
        "schema": "nicla-host-inventory-1", "system": system_snapshot(workspace),
        "source_artifacts": source_manifest(key_paths, workspace),
        "existing_v3_models": model_inventory_summary(workspace / "work/research-complete-v3/MODEL_INVENTORY.csv"),
        "disk_usage": {"study_root_bytes": sum(p.stat().st_size for p in study_root.rglob("*") if p.is_file())},
        "note": "Hash inventory covers declared source, receipt, inventory, and application model inputs; it does not hash every raw dataset image.",
    }


def fresh_device_discovery() -> dict:
    """Enumerate current USB/serial devices. Historical receipts are never used as discovery."""
    found = {"schema": "fresh-device-discovery-1",
             "enumerated_utc": datetime.now(timezone.utc).isoformat(timespec="microseconds"),
             "platform": platform.platform(), "hid_devices": [], "serial_ports": [],
             "usb_text": None, "errors": []}
    if os.name == "nt":
        try:
            import hid
            for row in hid.enumerate():
                safe = {key: row.get(key) for key in ("vendor_id", "product_id", "manufacturer_string",
                                                        "product_string", "interface_number", "usage_page", "usage")}
                path = row.get("path")
                if path is not None:
                    raw = path if isinstance(path, bytes) else str(path).encode("utf-8", "replace")
                    safe["enumerated_path_sha256"] = hashlib.sha256(raw).hexdigest()
                found["hid_devices"].append(safe)
        except Exception as exc:
            found["errors"].append("hid discovery: %s: %s" % (type(exc).__name__, exc))
        try:
            from serial.tools import list_ports
            for port in list_ports.comports():
                found["serial_ports"].append({"device": port.device, "description": port.description,
                                               "manufacturer": port.manufacturer,
                                               "vid": port.vid, "pid": port.pid})
        except Exception as exc:
            found["errors"].append("serial discovery: %s: %s" % (type(exc).__name__, exc))
    elif sys.platform == "darwin":
        found["serial_ports"] = [{"device": str(path)} for pattern in ("/dev/cu.usb*", "/dev/tty.usb*")
                                 for path in Path("/dev").glob(Path(pattern).name)]
        try:
            completed = subprocess.run(["system_profiler", "SPUSBDataType", "-detailLevel", "mini"],
                                       capture_output=True, text=True, timeout=20, check=False)
            found["usb_text"] = completed.stdout[-12000:] if completed.returncode == 0 else None
            if completed.returncode:
                found["errors"].append("system_profiler exit %d" % completed.returncode)
        except Exception as exc:
            found["errors"].append("USB discovery: %s: %s" % (type(exc).__name__, exc))
    else:
        found["errors"].append("fresh USB discovery adapter is not implemented for this platform")
    found["attached_device_count"] = len(found["hid_devices"]) + len(found["serial_ports"])
    return found


def probe_interpreter(executable: Path) -> dict:
    if not executable.is_file():
        return {"path": str(executable), "status": "missing"}
    script = """import json,sys,platform
d={'python':sys.version,'platform':platform.platform()}
try:
 import tensorflow as tf
 d['tensorflow']=tf.__version__
except Exception as e:
 d['tensorflow_error']=type(e).__name__+': '+str(e)
print(json.dumps(d))
"""
    try:
        result = subprocess.run([str(executable), "-c", script], capture_output=True, text=True,
                                timeout=60, check=False)
        return {"path": str(executable), "exit_code": result.returncode,
                "stdout": result.stdout.strip(), "stderr_tail": result.stderr[-2000:],
                "status": "pass" if result.returncode == 0 else "failed"}
    except Exception as exc:
        return {"path": str(executable), "status": "failed", "error": str(exc)}
