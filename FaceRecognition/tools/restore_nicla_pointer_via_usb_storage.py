#!/usr/bin/env python3
"""Restore a prior hash-verified Nicla release pointer through OpenMV storage.

This is for cases where the running firmware cannot enter REPL. It verifies
both deployment receipts and both release trees, preserves the active pointer
as a timestamped history file, and atomically restores the recorded prior
pointer. It never edits or formats a release tree or startup file.
"""

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def flash_path(volume, path):
    if not isinstance(path, str) or not path.startswith("/flash/") or ".." in path.split("/"):
        raise ValueError("invalid_flash_path:" + repr(path))
    return volume / path.removeprefix("/flash/")


def verify_deployment(volume, manifest_path):
    manifest_path = Path(manifest_path)
    manifest = json.loads(manifest_path.read_text())
    release_root = manifest.get("release_root")
    release = flash_path(volume, release_root)
    errors = []
    for rel, row in manifest.get("installed_files", {}).items():
        if rel.startswith("/") or ".." in rel.split("/"):
            errors.append("invalid_relative_path:" + rel)
            continue
        path = release / rel
        if not path.is_file() or sha256(path) != row.get("sha256"):
            errors.append("installed_file_mismatch:" + rel)
    for key, row in manifest.get("reused_model_files", {}).items():
        path = flash_path(volume, row.get("path"))
        if not path.is_file() or sha256(path) != row.get("sha256"):
            errors.append("reused_model_mismatch:" + key)
    if errors:
        raise RuntimeError("release verification failed: " + ", ".join(errors))
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mount-root", type=Path, default=Path("/Volumes/NO NAME"))
    parser.add_argument("--failed-deployment", type=Path, required=True,
                        help="directory containing deployment-manifest.json for the active release")
    parser.add_argument("--restore-deployment", type=Path, required=True,
                        help="directory containing deployment-manifest.json for the prior release")
    args = parser.parse_args()
    volume = args.mount_root.resolve()
    if not (volume / ".openmv_disk").is_file():
        raise RuntimeError("Mount root is not the Nicla OpenMV storage volume")

    failed_path = args.failed_deployment / "deployment-manifest.json"
    prior_path = args.restore_deployment / "deployment-manifest.json"
    failed = verify_deployment(volume, failed_path)
    prior = verify_deployment(volume, prior_path)
    active_pointer = volume / "face_runtime.current"
    current_value = active_pointer.read_text().strip()
    expected_active = failed.get("release_root")
    prior_value = failed.get("prior_target", {}).get("face_runtime.current", {}).get("value")
    if current_value != expected_active:
        raise RuntimeError("active pointer changed since failed deployment: " + current_value)
    if prior.get("release_root") != prior_value:
        raise RuntimeError("prior deployment receipt does not match recorded rollback pointer")
    if (volume / "main.py").is_file():
        startup_expected = failed.get("installed_files", {}).get("boot.py", {}).get("sha256")
        if startup_expected and sha256(volume / "main.py") != startup_expected:
            raise RuntimeError("startup hash differs from failed release receipt")

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    local_active_copy = args.failed_deployment / ("face_runtime.current.before_rollback." + stamp)
    history = volume / ("face_runtime.history.rollback-" + stamp)
    if local_active_copy.exists() or history.exists():
        raise FileExistsError("rollback evidence path already exists")
    pointer_bytes = active_pointer.read_bytes()
    local_active_copy.write_bytes(pointer_bytes)
    if sha256(local_active_copy) != sha256(active_pointer):
        raise RuntimeError("local copy of failed active pointer did not verify")
    history.write_bytes(pointer_bytes)
    if sha256(history) != sha256(active_pointer):
        raise RuntimeError("on-device history pointer did not verify")

    next_pointer = volume / "face_runtime.current.next"
    if next_pointer.exists():
        raise FileExistsError("stale face_runtime.current.next exists; inspect before recovery")
    prior_bytes = (prior_value + "\n").encode("utf-8")
    next_pointer.write_bytes(prior_bytes)
    if next_pointer.read_text().strip() != prior_value:
        raise RuntimeError("staged rollback pointer did not verify")
    next_pointer.replace(active_pointer)
    if active_pointer.read_text().strip() != prior_value:
        raise RuntimeError("restored active pointer did not verify")

    receipt = {
        "schema": "nicla-usb-pointer-restore-1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "pointer_restored_restart_pending",
        "reason": "failed_new_release_handshake",
        "failed_release_root": expected_active,
        "restored_release_root": prior_value,
        "restored_release_receipt": str(prior_path.resolve()),
        "failed_pointer_backup": str(local_active_copy.resolve()),
        "failed_pointer_history_on_device": "/flash/" + history.name,
        "failed_pointer_sha256": sha256(local_active_copy),
        "restored_pointer_sha256": sha256(active_pointer),
        "startup_sha256_unchanged": sha256(volume / "main.py"),
        "release_trees_hash_verified_before_restore": True,
        "restart_performed": False,
    }
    receipt_path = args.failed_deployment / ("pointer-restore-receipt." + stamp + ".json")
    receipt_path.write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps({"status": receipt["status"], "failed_release": expected_active,
        "restored_release": prior_value, "failed_pointer_history": "/flash/" + history.name,
        "receipt": str(receipt_path.resolve()), "restart_performed": False}, indent=2))


if __name__ == "__main__":
    main()
