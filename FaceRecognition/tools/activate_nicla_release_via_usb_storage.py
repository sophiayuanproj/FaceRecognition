#!/usr/bin/env python3
"""Activate a previously installed, hash-verified Nicla release via storage.

Use when serial REPL deployment is unavailable. The release must already be
complete on the OpenMV volume. This changes only the active pointer and keeps
the prior pointer both locally and on-device as rollback evidence.
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


def verify_release(volume, manifest_path):
    manifest = json.loads(Path(manifest_path).read_text())
    release = flash_path(volume, manifest.get("release_root"))
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
    startup_sha = manifest.get("installed_files", {}).get("boot.py", {}).get("sha256")
    if not startup_sha or not (volume / "main.py").is_file() or sha256(volume / "main.py") != startup_sha:
        raise RuntimeError("startup hash does not match target release receipt")
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mount-root", type=Path, default=Path("/Volumes/NO NAME"))
    parser.add_argument("--deployment", type=Path, required=True,
                        help="directory containing the target deployment-manifest.json")
    parser.add_argument("--expected-current", required=True,
                        help="current /flash/releases/... pointer that must be active before promotion")
    args = parser.parse_args()
    volume = args.mount_root.resolve()
    if not (volume / ".openmv_disk").is_file():
        raise RuntimeError("Mount root is not the Nicla OpenMV storage volume")
    manifest_path = args.deployment / "deployment-manifest.json"
    manifest = verify_release(volume, manifest_path)
    target = manifest.get("release_root")
    if manifest.get("prior_target", {}).get("face_runtime.current", {}).get("value") != args.expected_current:
        raise RuntimeError("target deployment receipt does not record the expected prior pointer")
    pointer = volume / "face_runtime.current"
    before = pointer.read_text().strip()
    if before != args.expected_current:
        raise RuntimeError("active pointer changed: expected %s, found %s" % (args.expected_current, before))

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    local_before = args.deployment / ("face_runtime.current.before_activation." + stamp)
    history = volume / ("face_runtime.history.activation-" + stamp)
    next_pointer = volume / "face_runtime.current.next"
    if local_before.exists() or history.exists() or next_pointer.exists():
        raise FileExistsError("an activation evidence path already exists; inspect before retry")
    old_bytes = pointer.read_bytes()
    local_before.write_bytes(old_bytes)
    history.write_bytes(old_bytes)
    if sha256(local_before) != sha256(pointer) or sha256(history) != sha256(pointer):
        raise RuntimeError("prior pointer preservation failed")
    next_pointer.write_bytes((target + "\n").encode("utf-8"))
    if next_pointer.read_text().strip() != target:
        raise RuntimeError("staged release pointer did not verify")
    next_pointer.replace(pointer)
    if pointer.read_text().strip() != target:
        raise RuntimeError("active release pointer did not verify")

    receipt = {
        "schema": "nicla-usb-release-activation-1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "pointer_activated_restart_pending",
        "release_id": manifest.get("release_id"),
        "release_root": target,
        "prior_release_root": before,
        "release_tree_hash_verified": True,
        "startup_sha256": sha256(volume / "main.py"),
        "prior_pointer_backup": str(local_before.resolve()),
        "prior_pointer_history_on_device": "/flash/" + history.name,
        "restart_performed": False,
    }
    receipt_path = args.deployment / ("activation-receipt." + stamp + ".json")
    receipt_path.write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps({"status": receipt["status"], "release_id": receipt["release_id"],
        "active_pointer": pointer.read_text().strip(), "prior_pointer_history": "/flash/" + history.name,
        "receipt": str(receipt_path.resolve()), "restart_performed": False}, indent=2))


if __name__ == "__main__":
    main()
