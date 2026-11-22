#!/usr/bin/env python3
"""Verify checked-in Nicla assets and run portable tests without opening hardware."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
APP = ROOT / "FaceRecognition"


def check_file(path: Path, expected: dict) -> None:
    if not path.is_file():
        raise ValueError(f"Missing versioned asset: {path.relative_to(ROOT)}")
    data = path.read_bytes()
    if "bytes" in expected and len(data) != expected["bytes"]:
        raise ValueError(f"Incorrect size: {path.relative_to(ROOT)}")
    if hashlib.sha256(data).hexdigest() != expected["sha256"]:
        raise ValueError(f"Incorrect SHA-256: {path.relative_to(ROOT)}")


def check_assets() -> None:
    fingerprints = json.loads((ROOT / "bench/RELEASE_HASHES.json").read_text())
    model_root = APP / "models/deployment-v1"
    for name, digest in fingerprints["model_asset_files"].items():
        check_file(model_root / name, {"sha256": digest})
    manifest_path = APP / "firmware/mpy/build_manifest.json"
    check_file(manifest_path, {"sha256": fingerprints["release_files"]["mpy/build_manifest.json"]})
    manifest = json.loads(manifest_path.read_text())
    for name, record in manifest["sources"].items():
        source_name = "main.py" if name == "wearable_app.py" else name
        check_file(APP / "firmware" / source_name, record)
    for name, record in manifest["modules"].items():
        check_file(manifest_path.parent / name, record)
    for name, expected_size in {
        "detector_gray_fomo_int8.tflite": 10160,
        "identity_embedding_a010_int8.tflite": 101576,
        "identity_classifier_a025_int8.tflite": 327072,
    }.items():
        if (model_root / name).stat().st_size != expected_size:
            raise ValueError(f"Truncated model: {name}")
    print("Full model files, firmware source and compiled modules verified.", flush=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--assets-only", action="store_true", help="Check assets without importing optional test dependencies.")
    args = parser.parse_args()
    try:
        check_assets()
    except (OSError, KeyError, ValueError) as error:
        print(f"Asset verification failed: {error}", file=sys.stderr)
        return 1
    if args.assets_only:
        return 0
    commands = [
        [sys.executable, "-B", "-m", "unittest", "discover", "-s", "host/tests"],
        [sys.executable, "-B", "-m", "unittest", "control_center.tests.test_control_center",
         "control_center.tests.test_launcher_acceptance", "control_center.tests.test_power_routes"],
        [sys.executable, "-B", "-m", "control_center.server", "--doctor", "--port", "0"],
    ]
    for command in commands:
        result = subprocess.run(command, cwd=APP, check=False)
        if result.returncode:
            return result.returncode
    node = shutil.which("node")
    if node is None:
        print("Node.js is required for dashboard JavaScript tests. Install it and rerun.", file=sys.stderr)
        return 2
    return subprocess.run([node, "--test", "control_center/tests/test_device_timeline.cjs"], cwd=APP, check=False).returncode


if __name__ == "__main__":
    raise SystemExit(main())
