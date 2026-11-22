#!/usr/bin/env python3
"""Read back and load-probe one exact I-matrix large-model variant on Nicla.

This does not run identity inference or qualify camera parity, standalone use,
or power. It verifies the exact deployed firmware/config/model identities and
asks the firmware's bounded preload command to load the large model so the
runtime metadata and available-heap receipt can be captured before I runs.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import sys
import time

APP = Path(__file__).resolve().parents[1]
WORKSPACE = APP.parent
sys.path.insert(0, str(APP / "host"))
from nicla_protocol import NiclaRawSerialCapture  # noqa: E402


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def append_qualification(path: Path, receipt: dict) -> None:
    lock_path = path.with_suffix(path.suffix + ".lock")
    try:
        from nicla_study.fs import FileLock, atomic_write_json
    except ImportError:
        sys.path.insert(0, str(APP))
        from nicla_study.fs import FileLock, atomic_write_json
    key = (receipt.get("artifact_sha256", "").lower(),
           receipt.get("device_config_sha256", "").lower())
    with FileLock(lock_path, timeout_s=10):
        if path.is_file():
            manifest = read_json(path)
            if manifest.get("schema") != "I-artifact-qualification-manifest-1":
                raise ValueError("unsupported I artifact qualification manifest")
        else:
            manifest = {"schema": "I-artifact-qualification-manifest-1",
                "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                "scope": "per-root model metadata/resource probes; not full H or recognition evidence",
                "variant_qualifications": []}
        rows = manifest.setdefault("variant_qualifications", [])
        old = [row for row in rows if (str(row.get("artifact_sha256", "")).lower(),
                                       str(row.get("device_config_sha256", "")).lower()) == key]
        if old:
            if any(row.get("receipt_sha256") == receipt.get("receipt_sha256") for row in old):
                return
            receipt = dict(receipt, attempt_index=1 + max(int(row.get("attempt_index", 1)) for row in old))
        rows.append(receipt)
        rows.sort(key=lambda row: (str(row.get("arm_id")), int(row.get("training_root_seed", -1))))
        manifest["updated_utc"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
        atomic_write_json(path, manifest)


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--port", required=True, help="freshly identified Nicla USB CDC or qualified UART port")
    p.add_argument("--variant-manifest", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True, help="new immutable probe directory")
    p.add_argument("--qualification-manifest", type=Path,
                   default=WORKSPACE / "work/project-finish-v4/handoffs/I/ARTIFACT_QUALIFICATION_MANIFEST.json")
    p.add_argument("--dtr", action="store_true", help="assert only when the current qualified mode requires it")
    p.add_argument("--rts", action="store_true")
    p.add_argument("--baudrate", type=int, default=115200)
    args = p.parse_args(argv)
    if not re.fullmatch(r"[A-Za-z0-9._-]{1,64}", args.output_dir.name):
        p.error("output directory name must be a short portable identifier")
    if not args.variant_manifest.is_file():
        p.error("variant manifest does not exist")
    bundle = read_json(args.variant_manifest)
    if bundle.get("schema") != "I-nicla-variant-overlay-1":
        p.error("unsupported variant bundle manifest")
    for key in ("artifact_sha256", "firmware_sha256", "device_runtime_config_sha256"):
        if not re.fullmatch(r"[0-9a-fA-F]{64}", str(bundle.get(key, ""))):
            p.error("variant manifest is missing a valid " + key)
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=False)
    run_id = "IVP-" + hashlib.sha256((bundle["bundle_id"] + "|" + str(time.time_ns())).encode()).hexdigest()[:20]
    capture = None
    error_text = None
    identity_ack = config_ack = ping_ack = stop_ack = preload_ack = None
    model_load = None
    try:
        capture = NiclaRawSerialCapture(args.port, output, run_id, baudrate=args.baudrate,
            dtr=args.dtr, rts=args.rts,
            expected_firmware_sha256=bundle["firmware_sha256"],
            expected_config_sha256=bundle["device_runtime_config_sha256"])
        identity_ack = capture.command({"cmd": "identity"}, wait_seconds=5)
        config_ack = capture.command({"cmd": "config"}, wait_seconds=5)
        ping_ack = capture.command({"cmd": "ping"}, wait_seconds=5)
        if identity_ack.get("device") != "Arduino Nicla Vision / OpenMV CSI":
            raise ValueError("live device identity is not Nicla Vision")
        if str(identity_ack.get("firmware_sha256", "")).lower() != bundle["firmware_sha256"].lower():
            raise ValueError("live firmware SHA-256 differs from variant manifest")
        if str(identity_ack.get("config_sha256", "")).lower() != bundle["device_runtime_config_sha256"].lower():
            raise ValueError("live configuration SHA-256 differs from variant manifest")
        if config_ack.get("large_model_id") != bundle.get("model_id"):
            raise ValueError("live large-model ID differs from the prepared variant")
        if ping_ack.get("pong") is not True:
            raise ValueError("Nicla ping failed")
        stop_ack = capture.command({"cmd": "stop"}, wait_seconds=5)
        if stop_ack.get("control_status", "ok") != "ok":
            raise ValueError("Nicla did not stop autonomous work before model probe")
        preload_ack = capture.command({"cmd": "preload_model", "model_slot": "large"}, wait_seconds=20)
        if preload_ack.get("control_status", "ok") != "ok":
            raise ValueError("large-model preload failed: " + str(preload_ack.get("reason_code")))
        if preload_ack.get("model_id") != bundle.get("model_id"):
            raise ValueError("preload acknowledgement identifies a different model")
        for row in capture.valid_records:
            if row.get("event") == "model_load" and row.get("model_id") == bundle.get("model_id"):
                model_load = row
        if not model_load:
            raise ValueError("no validated model_load telemetry record was received")
    except Exception as error:
        error_text = "%s: %s" % (type(error).__name__, error)
    finally:
        if capture is not None:
            try:
                if preload_ack is not None:
                    capture.command({"cmd": "stop"}, wait_seconds=5)
            except Exception:
                pass
            capture.close("complete" if error_text is None else "failed")

    raw_path = output / "nicla_serial_raw.bin"
    event_path = output / "nicla_serial_events.jsonl"
    rejected = malformed = 0
    if event_path.is_file():
        for line in event_path.read_text(encoding="utf-8").splitlines():
            try:
                row = json.loads(line)
            except ValueError:
                malformed += 1
                continue
            rejected += row.get("validation", {}).get("status") == "rejected"
    checks = {
        "device_identity_matches": bool(identity_ack and identity_ack.get("device") == "Arduino Nicla Vision / OpenMV CSI"),
        "exact_firmware_and_config_readback": bool(identity_ack and
            str(identity_ack.get("firmware_sha256", "")).lower() == bundle["firmware_sha256"].lower() and
            str(identity_ack.get("config_sha256", "")).lower() == bundle["device_runtime_config_sha256"].lower()),
        "live_config_advertises_variant_model_id": bool(config_ack and config_ack.get("large_model_id") == bundle.get("model_id")),
        "ping_and_stop_acknowledged": bool(ping_ack and ping_ack.get("pong") is True and stop_ack and stop_ack.get("control_status", "ok") == "ok"),
        "variant_model_preload_acknowledged": bool(preload_ack and preload_ack.get("control_status", "ok") == "ok" and preload_ack.get("model_id") == bundle.get("model_id")),
        "model_load_hash_matches_artifact": bool(model_load and str(model_load.get("model_sha256", "")).lower() == bundle["artifact_sha256"].lower()),
        "model_metadata_and_memory_reported": bool(model_load and preload_ack and model_load.get("metadata", {}).get("model_bytes") == bundle.get("artifact_bytes") and isinstance(model_load.get("metadata", {}).get("arena_bytes"), int) and isinstance(preload_ack.get("python_heap_free_bytes"), int)),
        "raw_and_validated_serial_logs_retained": raw_path.is_file() and raw_path.stat().st_size > 0 and event_path.is_file() and rejected == 0 and malformed == 0,
        "probe_error_absent": error_text is None,
    }
    status = "complete_verified" if all(checks.values()) else "failed_probe_or_incomplete"
    receipt = {
        "schema": "I-artifact-model-resource-qualification-1",
        "status": status,
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "evidence_tier": "physical_model_metadata_and_load_resource_probe",
        "physical_scope": "Exact model file readback, OpenMV load/metadata and heap/arena evidence only; not live-camera recognition, full H, standalone or power qualification.",
        "bundle_id": bundle.get("bundle_id"), "arm_id": bundle.get("arm_id"),
        "training_root_seed": bundle.get("training_root_seed"), "export_format": bundle.get("export_format"),
        "artifact_sha256": bundle.get("artifact_sha256"), "model_sha256": bundle.get("artifact_sha256"),
        "model_id": bundle.get("model_id"), "firmware_sha256": bundle.get("firmware_sha256"),
        "device_config_sha256": bundle.get("device_runtime_config_sha256"),
        "probe_run_id": run_id, "port_descriptor": capture.metadata.get("port_descriptor") if capture else None,
        "identity_ack": identity_ack, "config_ack": config_ack, "ping_ack": ping_ack,
        "stop_ack": stop_ack, "preload_ack": preload_ack, "model_load_record": model_load,
        "raw_serial_sha256": sha256(raw_path) if raw_path.is_file() else None,
        "serial_event_log_sha256": sha256(event_path) if event_path.is_file() else None,
        "checks": checks, "error": error_text,
        "output_dir_relative": output.relative_to(WORKSPACE).as_posix()
            if output.is_relative_to(WORKSPACE) else "external_output_dir",
    }
    receipt_path = output / "variant_qualification_receipt.json"
    receipt_path.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    receipt["receipt_path_relative"] = receipt_path.relative_to(WORKSPACE).as_posix() if receipt_path.is_relative_to(WORKSPACE) else str(receipt_path)
    receipt["receipt_sha256"] = sha256(receipt_path)
    # The central manifest stores the receipt file's byte hash; avoid a
    # self-referential hash inside the receipt itself.
    append_qualification(args.qualification_manifest.resolve(), receipt)
    print(json.dumps({"status": status, "receipt": str(receipt_path),
        "receipt_sha256": sha256(receipt_path), "qualification_manifest": str(args.qualification_manifest),
        "checks": checks, "error": error_text}, indent=2, sort_keys=True))
    return 0 if status == "complete_verified" else 4


if __name__ == "__main__":
    raise SystemExit(main())
