#!/usr/bin/env python3
"""Restore a prior Nicla startup and release pointer from a deploy backup."""

import argparse
from pathlib import Path
import re
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
MPREMOTE = ROOT / ".venv-nicla/bin/mpremote"


def sha256(path):
    import hashlib
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def invoke(port, *args, check=True):
    if not MPREMOTE.is_file():
        raise RuntimeError("Missing Nicla host tooling: " + str(MPREMOTE))
    result = subprocess.run([str(MPREMOTE), "connect", port, *map(str, args)],
        text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    if check and result.returncode:
        raise RuntimeError("mpremote failed: " + (result.stdout or ""))
    return result


def remote_sha(port, path):
    output = invoke(port, "fs", "sha256sum", ":" + path).stdout or ""
    match = re.search(r"\b([0-9a-f]{64})\b", output)
    if not match:
        raise RuntimeError("Could not read remote SHA-256: " + path)
    return match.group(1)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", required=True)
    parser.add_argument("--backup-dir", type=Path, required=True,
                        help="local deployment backup directory created by deploy_nicla_runtime.py")
    parser.add_argument("--stock-startup", action="store_true",
                        help="restore /flash/main.py.stock instead of the captured startup")
    parser.add_argument("--no-reset", action="store_true")
    args = parser.parse_args()
    backup = args.backup_dir.resolve()
    if not backup.is_dir():
        raise FileNotFoundError("backup directory not found")
    manifest = backup / "deployment-manifest.json"
    # A deployment may have failed before writing its final manifest. Its
    # initial target snapshots are still sufficient for exact startup recovery.
    startup = backup / "main.py"
    pointer = backup / "face_runtime.current"
    if args.stock_startup:
        restore = (
            "import os\n"
            "with open('/flash/main.py.stock','rb') as src: data=src.read()\n"
            "with open('/flash/main.py.restore','wb') as dst: dst.write(data)\n"
            "try:\n    os.remove('/flash/main.py')\nexcept OSError:\n    pass\n"
            "os.rename('/flash/main.py.restore','/flash/main.py')\n"
            "print('NICLA_STOCK_STARTUP_RESTORED')\n")
        invoke(args.port, "exec", restore)
        print("Restored /flash/main.py.stock; release data remains recoverable in /flash/releases.")
    else:
        if not startup.is_file():
            raise FileNotFoundError("backup has no captured startup: " + str(startup))
        invoke(args.port, "fs", "cp", str(startup), ":/flash/main.py.restore")
        if remote_sha(args.port, "/flash/main.py.restore") != sha256(startup):
            raise RuntimeError("restored startup readback mismatch")
        invoke(args.port, "exec", "import os\n"
            "try:\n    os.remove('/flash/main.py')\nexcept OSError:\n    pass\n"
            "os.rename('/flash/main.py.restore','/flash/main.py')\n"
            "print('PREVIOUS_STARTUP_RESTORED')\n")
        if remote_sha(args.port, "/flash/main.py") != sha256(startup):
            raise RuntimeError("final startup restore checksum mismatch")
        if pointer.is_file():
            invoke(args.port, "fs", "cp", str(pointer), ":/flash/face_runtime.current.restore")
            if remote_sha(args.port, "/flash/face_runtime.current.restore") != sha256(pointer):
                raise RuntimeError("previous release pointer readback mismatch")
            invoke(args.port, "exec", "import os\n"
                "try:\n    os.remove('/flash/face_runtime.current')\nexcept OSError:\n    pass\n"
                "os.rename('/flash/face_runtime.current.restore','/flash/face_runtime.current')\n"
                "print('PREVIOUS_RELEASE_POINTER_RESTORED')\n")
        else:
            invoke(args.port, "exec", "import os\n"
                "try:\n    os.remove('/flash/face_runtime.current')\nexcept OSError:\n    pass\n"
                "print('PREVIOUS_RELEASE_POINTER_ABSENT')\n")
        print("Restored captured startup and, when present, the previously active release pointer.")
    if manifest.is_file():
        print("Deployment record:", manifest)
    if not args.no_reset:
        invoke(args.port, "reset")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, RuntimeError, subprocess.CalledProcessError) as error:
        print("Nicla recovery stopped:", error, file=sys.stderr)
        raise SystemExit(1)
