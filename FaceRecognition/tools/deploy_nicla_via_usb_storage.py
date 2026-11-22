#!/usr/bin/env python3
"""Stage and activate a versioned Nicla release through its USB mass-storage volume.

Use this when the live firmware watchdog makes REPL-based file copies unreliable.
Only a fresh release directory and the two startup pointer files are changed.
"""

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import shutil
import sys
import tempfile
import time

ROOT = Path(__file__).resolve().parents[1]
PROJECT = ROOT.parent
sys.path.insert(0, str(ROOT / "tools"))
from deploy_nicla_runtime import MODEL_NAMES, PY_NAMES, make_payload, sha256


def release_id_for_sources():
    digest = hashlib.sha256()
    inputs = [ROOT / "firmware/main.py", ROOT / "firmware/boot.py",
              ROOT / "firmware/runtime_config.json"]
    inputs.extend(ROOT / "firmware" / name for name in PY_NAMES)
    inputs.extend(ROOT / "models/deployment-v1" / name for name in MODEL_NAMES)
    for path in sorted(inputs):
        digest.update(path.name.encode("utf-8") + b"\t")
        digest.update(sha256(path).encode("ascii") + b"\n")
    return "face-v1-" + digest.hexdigest()[:12] + "-usb"


def write_exact(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as stream:
        stream.write(data)
        stream.flush()
        try:
            import os
            os.fsync(stream.fileno())
        except (AttributeError, OSError):
            pass


def promote_file(source, destination, previous=None):
    if previous is not None and destination.exists():
        if previous.exists():
            previous.unlink()
        shutil.copyfile(destination, previous)
        if sha256(destination) != sha256(previous):
            raise RuntimeError("previous-file backup readback mismatch: " + destination.name)
    # Replace in one same-volume operation so a transient rename failure does
    # not leave the active startup file or pointer missing.
    source.replace(destination)


def verify_tree_with_retry(root, expected, attempts=10, delay_s=0.5):
    """Verify a renamed FAT directory before making it the active release.

    macOS may briefly return stale directory metadata immediately after a
    rename on the OpenMV mass-storage volume. Retry readback while the old
    release remains selected; never promote the pointer on a transient miss.
    """
    last_error = None
    for attempt in range(attempts):
        try:
            installed = {}
            for name, expected_sha in expected.items():
                actual_path = root / name
                actual_sha = sha256(actual_path)
                if actual_sha != expected_sha:
                    raise RuntimeError("final release readback hash mismatch: " + name)
                installed[name] = {"sha256": actual_sha,
                                   "bytes": actual_path.stat().st_size}
            return installed
        except (FileNotFoundError, RuntimeError, OSError) as exc:
            last_error = exc
            if attempt + 1 < attempts:
                try:
                    import os
                    os.sync()
                except (AttributeError, OSError):
                    pass
                time.sleep(delay_s)
    raise RuntimeError("final release readback failed before activation: " + str(last_error))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mount-root", type=Path, default=Path("/Volumes/NO NAME"),
                        help="mounted root of the Nicla's USB mass-storage volume")
    parser.add_argument("--release-id", help="version identifier; defaults to a content-derived ID")
    args = parser.parse_args()
    volume = args.mount_root.resolve()
    if not (volume / ".openmv_disk").is_file():
        raise RuntimeError("Mount root is not the Nicla OpenMV storage volume")
    release_id = args.release_id or release_id_for_sources()
    if not re.fullmatch(r"[A-Za-z0-9_-]{1,48}", release_id):
        raise ValueError("invalid release ID")
    release_root = "/flash/releases/" + release_id
    local_backup_root = PROJECT / "work/project-finish-v4/deployments/backups"
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    backup_dir = local_backup_root / (stamp + "-usb")
    backup_dir.mkdir(parents=True, exist_ok=False)

    current_main = volume / "main.py"
    current_pointer = volume / "face_runtime.current"
    if not current_main.is_file() or not current_pointer.is_file():
        raise RuntimeError("Nicla startup or active release pointer is missing")
    old_pointer = current_pointer.read_text().strip()
    if not old_pointer.startswith("/flash/releases/") or ".." in old_pointer.split("/"):
        raise RuntimeError("active release pointer is invalid")
    shutil.copyfile(current_main, backup_dir / "main.py")
    shutil.copyfile(current_pointer, backup_dir / "face_runtime.current")
    prior = {"main.py": {"sha256": sha256(backup_dir / "main.py"),
                         "bytes": (backup_dir / "main.py").stat().st_size},
             "face_runtime.current": {"sha256": sha256(backup_dir / "face_runtime.current"),
                                      "bytes": (backup_dir / "face_runtime.current").stat().st_size,
                                      "value": old_pointer}}

    temporary_root = volume / "releases" / (release_id + ".next")
    final_root = volume / "releases" / release_id
    if temporary_root.exists() or final_root.exists():
        raise FileExistsError("release or staging directory already exists on the Nicla")
    with tempfile.TemporaryDirectory(prefix="nicla-usb-release-") as generated:
        payload = make_payload(release_root, generated)
        expected = {name: sha256(path) for name, path in payload.items()}
        (backup_dir / "target-manifest.json").write_text(json.dumps({
            "release_id": release_id, "release_root": release_root,
            "payload": {name: {"sha256": expected[name], "bytes": path.stat().st_size}
                       for name, path in payload.items()}, "prior_target": prior}, indent=2) + "\n")

        temporary_root.mkdir(parents=True)
        for name, source in payload.items():
            destination = temporary_root / name
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, destination)
            if sha256(destination) != expected[name]:
                raise RuntimeError("USB-storage readback hash mismatch: " + name)
        temporary_root.rename(final_root)

        # Verify every file after the staging-directory rename, before
        # changing the active pointer or replacing the boot entry point.
        installed = verify_tree_with_retry(final_root, expected)

        pointer_next = volume / "face_runtime.current.next"
        pointer_previous = volume / "face_runtime.current.prev"
        write_exact(pointer_next, (release_root + "\n").encode("utf-8"))
        if pointer_next.read_text().strip() != release_root:
            raise RuntimeError("release pointer staging readback mismatch")
        promote_file(pointer_next, current_pointer, pointer_previous)
        if current_pointer.read_text().strip() != release_root:
            raise RuntimeError("active release pointer did not promote")

        # Startup entry point is deliberately installed after the complete,
        # hashed release and active pointer are already present.
        startup_next = volume / "main.py.next"
        startup_previous = volume / "main.py.prev"
        write_exact(startup_next, payload["boot.py"].read_bytes())
        if sha256(startup_next) != sha256(payload["boot.py"]):
            raise RuntimeError("startup staging readback mismatch")
        promote_file(startup_next, current_main, startup_previous)
        if sha256(current_main) != sha256(payload["boot.py"]):
            raise RuntimeError("final startup readback mismatch")

    manifest = {"schema": "nicla-versioned-deployment-1", "created_utc": stamp,
        "release_id": release_id, "release_root": release_root,
        "deployment_method": "USB mass-storage staging while watchdog-fed runtime remained active",
        "preflight": "all payload hashes read back from the mounted board volume; physical model/UART preflight follows reset",
        "installed_files": installed, "prior_target": prior,
        "rollback_pointer_backup": str(backup_dir / "face_runtime.current"),
        "startup_backup": str(backup_dir / "main.py"),
        "physical_camera_inference": "pending bounded board smoke"}
    (backup_dir / "deployment-manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({"status": "installed", "release_id": release_id,
        "release_root": release_root, "backup_dir": str(backup_dir),
        "installed_file_count": len(installed), "prior_release": old_pointer,
        "active_release": current_pointer.read_text().strip(),
        "startup_last": True, "reset": False}, indent=2))


if __name__ == "__main__":
    main()
