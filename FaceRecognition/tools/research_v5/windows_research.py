#!/usr/bin/env python3
"""Fail-closed Windows entry point for the v5 instrument-only handoff."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import struct
import subprocess
import sys
from datetime import datetime, timezone


HERE = Path(__file__).resolve().parent
APP_ROOT = HERE.parents[1]
if str(APP_ROOT) not in sys.path:
    sys.path.insert(0, str(APP_ROOT))

from nicla_study.v5_evidence import derive_metrics_from_predictions, strict_load, verify_bindings


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def workspace_arg(value: str | None) -> Path:
    candidate = Path(value or Path.cwd()).resolve()
    for root in (candidate, *candidate.parents):
        if (root / "work/project-research-v5/READY_FOR_WINDOWS.json").is_file():
            return root
    raise FileNotFoundError("Could not find work/project-research-v5/READY_FOR_WINDOWS.json")


def readiness(root: Path) -> dict:
    path = root / "work/project-research-v5/READY_FOR_WINDOWS.json"
    record = strict_load(path)
    if record.get("schema") != "nicla-v5-ready-for-windows-1" or not record.get("ready_for_windows"):
        raise ValueError("Readiness record is missing or does not allow the documented Windows scope")
    verify_bindings(root, record["bindings"])
    for row in record.get("row_prediction_bindings", []):
        path = (root / row["path"]).resolve()
        path.relative_to(root.resolve())
        if not path.is_file() or path.stat().st_size != row["bytes"] or sha(path) != row["sha256"]:
            raise ValueError("Prediction binding mismatch: " + row["path"])
        derive_metrics_from_predictions(root, row["path"])
    for row in record.get("wheelhouse", []):
        path = (root / row["path"]).resolve()
        path.relative_to(root.resolve())
        if not path.is_file() or path.stat().st_size != row["bytes"] or sha(path) != row["sha256"]:
            raise ValueError("Wheel binding mismatch: " + row["path"])
    return record


def doctor(root: Path, static: bool) -> int:
    record = readiness(root)
    report = {
        "schema": "v5-windows-doctor-1",
        "status": "pass",
        "workspace": str(root),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "python_bits": struct.calcsize("P") * 8,
        "static_only": static,
        "readiness_status": record["status"],
        "recognition_qualified": record["recognition_qualified"],
        "candidate_application_deployable": record["candidate_application_deployable"],
        "bound_files_verified": len(record["bindings"]),
        "row_prediction_files_verified": len(record["row_prediction_bindings"]),
        "wheel_files_verified": len(record["wheelhouse"]),
        "device_opened": False,
    }
    if not static:
        errors = []
        if os.name != "nt":
            errors.append("full doctor requires native Windows")
        if sys.version_info[:2] != (3, 12) or struct.calcsize("P") != 8:
            errors.append("full doctor requires 64-bit CPython 3.12")
        try:
            import hid  # noqa: F401
            import mpremote  # noqa: F401
            import numpy  # noqa: F401
            import serial  # noqa: F401
            from PIL import Image  # noqa: F401
            from nicla_study.instruments.windows_fnirsi import capture_windows, offline_audit, steady_average  # noqa: F401
        except Exception as exc:  # emit one actionable doctor result
            errors.append("instrument imports failed: %s: %s" % (type(exc).__name__, exc))
        if "tensorflow" in sys.modules or "h5py" in sys.modules:
            errors.append("instrument environment unexpectedly imported TensorFlow/h5py")
        report["device_opened"] = False
        report["errors"] = errors
        if errors:
            report["status"] = "fail"
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["status"] == "pass" else 2


def demo(root: Path, output: Path | None) -> int:
    import numpy as np
    record = readiness(root)
    base = root / "work/project-research-v5/reports/M5/rgb48_encoder_parity"
    board = np.load(base / "BOARD_OUTPUTS_I8.npy", allow_pickle=False)
    host = np.load(base / "HOST_OUTPUTS_I8.npy", allow_pickle=False)
    if board.shape != host.shape or not np.array_equal(board, host):
        raise ValueError("Saved 48px synthetic board/reference outputs no longer match")
    result = {
        "schema": "v5-parity-fixture-demo-1",
        "status": "pass_exact_synthetic_fixture_only",
        "model_sha256": record["encoder_parity_only"]["model_sha256"],
        "fixture_rows": int(board.shape[0]),
        "output_values": int(board.size),
        "exact_equal": True,
        "device_opened": False,
        "identity_output": False,
        "scope": "replay of saved synthetic 48px board/reference int8 output arrays; not a face-recognition demo",
    }
    if output:
        path = output if output.is_absolute() else root / output
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        result["output_json"] = str(path)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


def discover(root: Path) -> int:
    readiness(root)
    if os.name != "nt":
        print(json.dumps({"status": "not_run", "reason": "fresh HID/COM inventory requires native Windows", "device_opened": False}, indent=2))
        return 2
    import hid
    from serial.tools import list_ports
    from nicla_study.instruments.windows_fnirsi.instrument_adapter import describe
    from nicla_study.instruments.windows_fnirsi.upstream.power.fnirsi_protocol import PID, VID
    hids = [describe(row, include_path=False) for row in hid.enumerate(VID, PID)]
    ports = [{"device": item.device, "description": item.description, "manufacturer": item.manufacturer,
              "product": item.product, "vid": item.vid, "pid": item.pid, "serial_number_present": bool(item.serial_number)}
             for item in list_ports.comports()]
    print(json.dumps({"status": "inventory_only", "meter_hid_candidates": hids, "serial_candidates": ports,
                      "device_opened": False, "interpretation": "descriptor inventory only; confirm identities physically before selecting or opening any device"}, indent=2))
    return 0


def gated_operation(root: Path, action: str) -> int:
    record = readiness(root)
    message = {
        "schema": "v5-safe-operation-gate-1",
        "status": "refused",
        "action": action,
        "reason": "READY_FOR_WINDOWS has no useful camera-to-identity application or exact standalone deploy/rollback receipt",
        "ready_for_final_build": record["ready_for_final_build"],
        "recognition_qualified": record["recognition_qualified"],
        "board_opened": False,
        "board_modified": False,
    }
    print(json.dumps(message, indent=2, sort_keys=True))
    return 4


def readback(root: Path, args) -> int:
    record = readiness(root)
    if args.port:
        if not args.backup or not args.receipt:
            raise ValueError("device readback requires --port, --backup and --receipt")
        tool = root / "FaceRecognition/tools/research_v5/romfs_lab.py"
        command = [sys.executable, str(tool), "--port", args.port, "--backup", str(args.backup), "--receipt", str(args.receipt)]
        return subprocess.call(command, cwd=root)
    image = root / "work/project-research-v5/staging/romfs-fused48/romfs0.img"
    receipt = root / "work/project-research-v5/reports/M5/romfs_fused48_deploy.json"
    actual = sha(image)
    expected = strict_load(receipt).get("readback_sha256")
    report = {"status": "pass" if actual == expected else "fail", "scope": "offline verification of the Mac-readback image receipt",
              "image_sha256": actual, "receipt_readback_sha256": expected, "device_opened": False}
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["status"] == "pass" else 2


def capture(root: Path, passthrough: list[str]) -> int:
    readiness(root)
    if passthrough and passthrough[0] == "--":
        passthrough = passthrough[1:]
    tool = root / "FaceRecognition/nicla_study/instruments/windows_fnirsi/capture_windows.py"
    command = [sys.executable, str(tool), "--project-root", str(root / "work/project-research-v5/windows"), *passthrough]
    return subprocess.call(command, cwd=root)


def qc(root: Path, args) -> int:
    readiness(root)
    # offline_audit uses package-relative imports, so launch it as a module and
    # expose the relocated FaceRecognition package root explicitly.
    command = [sys.executable, "-m", "nicla_study.instruments.windows_fnirsi.offline_audit",
               "--run-dir", str(args.run_dir), "--output-json", str(args.output_json)]
    env = os.environ.copy()
    app_root = str(root / "FaceRecognition")
    env["PYTHONPATH"] = app_root + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    return subprocess.call(command, cwd=root, env=env)


def resume(root: Path, run_dir: Path) -> int:
    readiness(root)
    path = run_dir if run_dir.is_absolute() else root / run_dir
    if not path.is_dir():
        raise FileNotFoundError(path)
    meta_path = path / "run_metadata.json"
    meta = strict_load(meta_path) if meta_path.is_file() else {}
    original_id = meta.get("run_id", path.name)
    next_id = "rerun-%s-%s" % (datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"), original_id[:40])
    report = {"status": "resume_as_new_linked_run", "original_run_id": original_id, "new_run_id": next_id,
              "rerun_of": original_id, "append_to_original": False, "hardware_opened": False,
              "reason": "immutable captures are never appended or silently resumed; use capture with a new --run-id and --rerun-of"}
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


def package_return(root: Path, destination: Path) -> int:
    readiness(root)
    tool = root / "FaceRecognition/tools/research_v5/package_v5_return.py"
    output = destination if destination.is_absolute() else root / destination
    command = [sys.executable, str(tool), "--workspace", str(root), "--destination", str(output)]
    return subprocess.call(command, cwd=root)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", help="workspace root or a path inside the extracted transfer")
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("doctor"); p.add_argument("--static", action="store_true", help="verify bindings only; skip Windows runtime imports")
    sub.add_parser("discover")
    p = sub.add_parser("demo"); p.add_argument("--output-json", type=Path)
    sub.add_parser("deploy"); sub.add_parser("rollback")
    p = sub.add_parser("readback"); p.add_argument("--port"); p.add_argument("--backup", type=Path); p.add_argument("--receipt", type=Path)
    p = sub.add_parser("capture"); p.add_argument("args", nargs=argparse.REMAINDER)
    p = sub.add_parser("qc"); p.add_argument("--run-dir", type=Path, required=True); p.add_argument("--output-json", type=Path, required=True)
    p = sub.add_parser("resume"); p.add_argument("run_dir", type=Path)
    p = sub.add_parser("return"); p.add_argument("--destination", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        root = workspace_arg(args.workspace)
        if args.command == "doctor": return doctor(root, args.static)
        if args.command == "discover": return discover(root)
        if args.command == "demo": return demo(root, args.output_json)
        if args.command in {"deploy", "rollback"}: return gated_operation(root, args.command)
        if args.command == "readback": return readback(root, args)
        if args.command == "capture": return capture(root, args.args)
        if args.command == "qc": return qc(root, args)
        if args.command == "resume": return resume(root, args.run_dir)
        if args.command == "return": return package_return(root, args.destination)
        parser.error("unknown command")
    except (OSError, ValueError, KeyError, TypeError) as exc:
        print(json.dumps({"status": "failed_closed", "error": "%s: %s" % (type(exc).__name__, exc)}, indent=2), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
