#!/usr/bin/env python3
"""Install the face pipeline as Nicla Vision's cold-boot /flash/main.py app."""

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
PROJECT = ROOT.parent
MPREMOTE = ROOT / ".venv-nicla" / "bin" / "mpremote"
BACKUP_ROOT = PROJECT / "work" / "model-training" / "deployment-backup" / "standalone"


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def invoke(port, *args, check=True, capture=False):
    if not MPREMOTE.is_file():
        raise RuntimeError("Missing Nicla host tooling: " + str(MPREMOTE))
    command = [str(MPREMOTE), "connect", port, *map(str, args)]
    result = subprocess.run(command, text=True,
                            stdout=subprocess.PIPE if capture else None,
                            stderr=subprocess.STDOUT if capture else None)
    if check and result.returncode:
        output = result.stdout if capture else ""
        raise RuntimeError("Command failed (%d): %s\n%s" %
                           (result.returncode, " ".join(command), output))
    return result


def local_payloads():
    return {
        "recognize_embedding.py": ROOT / "firmware" / "recognize_embedding.py",
        "embedding_config.py": ROOT / "firmware" / "embedding_config.py",
        "identity_embedding_int8.tflite": ROOT / "models" / "identity_embedding_int8.tflite",
        "embedding_calibration.json": PROJECT / "work" / "model-training" / "artifacts" / "embedding_calibration.json",
    }


def remote_hash(port, filename):
    result = invoke(port, "fs", "sha256sum", ":/flash/" + filename, capture=True)
    match = re.search(r"\b([0-9a-f]{64})\b", result.stdout or "")
    if not match:
        raise RuntimeError("Could not read remote SHA-256 for /flash/" + filename +
                           ": " + (result.stdout or "").strip())
    return match.group(1)


def preflight(port):
    runtime = invoke(port, "exec",
                     "import csi, ml, os; "
                     "assert os.stat('/rom/fomo_face_detection.tflite'); "
                     "print('NICLA_OPENMV_RUNTIME_READY')",
                     capture=True)
    if "NICLA_OPENMV_RUNTIME_READY" not in (runtime.stdout or ""):
        raise RuntimeError("The target did not confirm OpenMV CSI/ML and stock FOMO")
    return runtime.stdout.strip()


def backup_remote(port, backup_dir):
    # Preserve each file that this deployment could replace. Existing local
    # enrollment templates are intentionally left untouched and never copied.
    names = [
        "main.py", "recognize_embedding.py", "embedding_config.py",
        "identity_embedding_int8.tflite", "embedding_calibration.json",
    ]
    records = {}
    for name in names:
        destination = backup_dir / name
        result = invoke(port, "fs", "cp", ":/flash/" + name, str(destination),
                        check=False, capture=True)
        if result.returncode == 0 and destination.is_file():
            records[name] = {"present": True, "sha256": sha256(destination),
                             "bytes": destination.stat().st_size}
        else:
            destination.unlink(missing_ok=True)
            records[name] = {"present": False}
    if not records["main.py"]["present"]:
        raise RuntimeError("Could not back up existing /flash/main.py; no files were changed")
    return records


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", required=True, help="explicit /dev/cu.usbmodem... Nicla port")
    args = parser.parse_args()

    payloads = local_payloads()
    for path in [*payloads.values(), ROOT / "firmware" / "main.py"]:
        if not path.is_file():
            raise RuntimeError("Required deployment file is missing: " + str(path))
    calibration = json.loads(payloads["embedding_calibration.json"].read_text())
    threshold = float(calibration["threshold"])
    if not -1.0 <= threshold <= 1.0:
        raise RuntimeError("Calibration threshold is outside cosine-similarity range")

    preflight_output = preflight(args.port)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    backup_dir = BACKUP_ROOT / stamp
    backup_dir.mkdir(parents=True, exist_ok=False)
    previous = backup_remote(args.port, backup_dir)

    # Keep an on-board rollback copy of the original startup file. Do not
    # overwrite an earlier copy: it may be the only clean recovery path.
    preserve = (
        "import os\n"
        "created = False\n"
        "try:\n"
        "    os.stat('/flash/main.py.stock')\n"
        "except OSError:\n"
        "    with open('/flash/main.py', 'rb') as src:\n"
        "        data = src.read()\n"
        "    with open('/flash/main.py.stock', 'wb') as dst:\n"
        "        dst.write(data)\n"
        "    created = True\n"
        "print('NICLA_STARTUP_BACKUP_CREATED' if created else 'NICLA_STARTUP_BACKUP_EXISTS')\n"
    )
    preserve_result = invoke(args.port, "exec", preserve, capture=True)
    preserve_output = preserve_result.stdout or ""
    if not ("NICLA_STARTUP_BACKUP_CREATED" in preserve_output or
            "NICLA_STARTUP_BACKUP_EXISTS" in preserve_output):
        raise RuntimeError("Could not create on-board rollback copy of /flash/main.py")
    stock_digest = remote_hash(args.port, "main.py.stock")
    if ("NICLA_STARTUP_BACKUP_CREATED" in preserve_output and
            stock_digest != previous["main.py"]["sha256"]):
        raise RuntimeError("On-board rollback file does not match the backed-up startup script")

    installed = {}
    for name, source in payloads.items():
        invoke(args.port, "fs", "cp", "-f", str(source), ":/flash/" + name)
        actual = remote_hash(args.port, name)
        expected = sha256(source)
        if actual != expected:
            raise RuntimeError("Remote checksum mismatch for /flash/" + name)
        installed[name] = {"bytes": source.stat().st_size, "sha256": expected}

    # Install the entry point last, via a fully written temporary file.
    startup = ROOT / "firmware" / "main.py"
    invoke(args.port, "fs", "cp", "-f", str(startup), ":/flash/main.py.next")
    if remote_hash(args.port, "main.py.next") != sha256(startup):
        raise RuntimeError("Remote checksum mismatch for startup entry point")
    promote = (
        "import os\n"
        "try:\n"
        "    os.remove('/flash/main.py')\n"
        "except OSError:\n"
        "    pass\n"
        "os.rename('/flash/main.py.next', '/flash/main.py')\n"
        "print('NICLA_STANDALONE_MAIN_INSTALLED')\n"
    )
    result = invoke(args.port, "exec", promote, capture=True)
    if "NICLA_STANDALONE_MAIN_INSTALLED" not in (result.stdout or ""):
        raise RuntimeError("Could not promote the standalone main.py")
    installed["main.py"] = {"bytes": startup.stat().st_size, "sha256": sha256(startup)}
    if remote_hash(args.port, "main.py") != installed["main.py"]["sha256"]:
        raise RuntimeError("Final /flash/main.py checksum does not match local entry point")

    manifest = {
        "schema_version": 1,
        "created_utc": stamp,
        "port_used": args.port,
        "runtime_preflight": preflight_output,
        "boot_mode": "OpenMV MicroPython cold-boot /flash/main.py; no Arduino IDE or host process required after deployment",
        "stock_romfs_modified": False,
        "enrollment_templates_overwritten": False,
        "previous_flash_files": previous,
        "on_board_startup_backup_sha256": stock_digest,
        "installed_flash_files": installed,
        "calibration_threshold": threshold,
        "startup_rollback": "/flash/main.py.stock",
        "verification": "Remote file SHA-256 checks passed; a physical cold-boot camera run is still required.",
    }
    (backup_dir / "deployment-manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")

    # A hard reset follows the MicroPython cold-boot path and starts main.py.
    invoke(args.port, "reset")

    print("Standalone pipeline installed and checksummed.")
    print("Nicla hard-reset completed; the standalone main.py now owns the camera loop.")
    print("After the reset it runs without the laptop.")
    print("Stock FOMO ROMFS and existing enrollment templates were preserved.")
    print("Local rollback snapshot:", backup_dir)
    print("On-board original startup backup: /flash/main.py.stock")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, ValueError, KeyError, RuntimeError, subprocess.CalledProcessError) as error:
        print("Standalone install stopped:", error, file=sys.stderr)
        raise SystemExit(1)
