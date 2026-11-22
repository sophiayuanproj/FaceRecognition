#!/usr/bin/env python3
"""Create the current firmware/config/model source receipt for J and Sophia."""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
WORKSPACE = ROOT.parent
STUDY = WORKSPACE / "work/project-finish-v4"
FIRMWARE_PARTS = ("boot.py", "wearable_app.py", "identity_contracts.py",
    "runtime_support.py", "telemetry.py", "mpy/build_manifest.json",
    "mpy/wearable_app.mpy", "mpy/identity_contracts.mpy",
    "mpy/runtime_support.mpy", "mpy/telemetry.mpy")
SOURCES = {
    "boot.py": ROOT / "firmware/boot.py",
    "wearable_app.py": ROOT / "firmware/main.py",
    "identity_contracts.py": ROOT / "firmware/identity_contracts.py",
    "runtime_support.py": ROOT / "firmware/runtime_support.py",
    "telemetry.py": ROOT / "firmware/telemetry.py",
    "mpy/build_manifest.json": ROOT / "firmware/mpy/build_manifest.json",
    "mpy/wearable_app.mpy": ROOT / "firmware/mpy/wearable_app.mpy",
    "mpy/identity_contracts.mpy": ROOT / "firmware/mpy/identity_contracts.mpy",
    "mpy/runtime_support.mpy": ROOT / "firmware/mpy/runtime_support.mpy",
    "mpy/telemetry.mpy": ROOT / "firmware/mpy/telemetry.mpy",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    manifest_path = ROOT / "models/deployment-v1/integration_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    source_files = {name: {"path": path.relative_to(WORKSPACE).as_posix(),
                           "sha256": sha256(path), "bytes": path.stat().st_size}
                    for name, path in SOURCES.items()}
    firmware_digest = hashlib.sha256()
    for name in FIRMWARE_PARTS:
        firmware_digest.update((name + "\t" + source_files[name]["sha256"] + "\n").encode("utf-8"))
    runtime_path = ROOT / "firmware/runtime_config.json"
    runtime_sha = sha256(runtime_path)
    artifact_checks = {}
    failures = []
    for name, record in manifest.get("files", {}).items():
        path = ROOT / "models/deployment-v1" / name
        actual = sha256(path) if path.is_file() else None
        expected = record.get("sha256")
        artifact_checks[name] = {"expected_sha256": expected, "actual_sha256": actual,
            "exists": path.is_file(), "matches": actual == expected}
        if actual != expected:
            failures.append("artifact_hash_mismatch:" + name)
    gallery_path = ROOT / "models/deployment-v1/fixed_roster_gallery.json"
    gallery_expected = manifest.get("gallery", {}).get("sha256")
    gallery_actual = sha256(gallery_path) if gallery_path.is_file() else None
    if gallery_actual != gallery_expected:
        failures.append("gallery_hash_mismatch")
    if manifest.get("runtime_config", {}).get("sha256") != runtime_sha:
        failures.append("integration_manifest_runtime_config_hash_mismatch")
    receipt = {
        "schema": "J-firmware-source-receipt-1",
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "status": "pass" if not failures else "failed",
        "firmware_sha256": firmware_digest.hexdigest(),
        "firmware_hash_contract": "SHA-256 of ordered name<TAB>file_sha256<LF> rows for boot.py, source runtime modules, precompiled-runtime build manifest, and four matching .mpy modules",
        "firmware_sources": source_files,
        "boot_source": {"path": "FaceRecognition/firmware/boot.py",
            "sha256": sha256(ROOT / "firmware/boot.py"), "bytes": (ROOT / "firmware/boot.py").stat().st_size},
        "precompiled_runtime": json.loads((ROOT / "firmware/mpy/build_manifest.json").read_text(encoding="utf-8")),
        "runtime_config": {"path": "FaceRecognition/firmware/runtime_config.json", "sha256": runtime_sha},
        "integration_manifest": {"path": "FaceRecognition/models/deployment-v1/integration_manifest.json",
            "sha256": sha256(manifest_path), "runtime_config_hash_matches": manifest.get("runtime_config", {}).get("sha256") == runtime_sha},
        "deployment_artifacts": artifact_checks,
        "gallery": {"sha256": gallery_actual, "matches_manifest": gallery_actual == gallery_expected},
        "policy_ids": list(json.loads(runtime_path.read_text(encoding="utf-8")).get("policies", {})),
        "controller_calibration_status": json.loads(runtime_path.read_text(encoding="utf-8")).get("controller_calibration", {}).get("status"),
        "failures": failures,
        "hardware_status": "not_checked_from_this_source_receipt",
    }
    out = STUDY / "handoffs/J/SOURCE_RECEIPT.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    tmp.replace(out)
    print(json.dumps(receipt, indent=2))
    return 0 if receipt["status"] == "pass" else 2


if __name__ == "__main__":
    raise SystemExit(main())
