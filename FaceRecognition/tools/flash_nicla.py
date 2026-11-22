#!/usr/bin/env python3
"""Explicit Nicla Vision DFU operations. No board operation occurs on import."""

import argparse
import hashlib
import json
from pathlib import Path
import re
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
DEVICE = "2341:035f"
ROMFS_BYTES = 4 * 1024 * 1024


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def check_romfs(path):
    if not path.is_file():
        raise ValueError("Missing ROMFS image: " + str(path))
    with path.open("rb") as stream:
        if stream.read(3) != b"\xd2\xcd\x31":
            raise ValueError("Image is not the expected MicroPython ROMFS format")
    if not 16 <= path.stat().st_size <= ROMFS_BYTES:
        raise ValueError("ROMFS image size is outside the Nicla partition budget")


def stock_files():
    manifest = json.loads((ROOT / "config/firmware-release.json").read_text())
    if manifest["board"] != "ARDUINO_NICLA_VISION" or manifest["version"] != "v5.0.1":
        raise ValueError("Unexpected board/release in firmware manifest")
    files = ROOT / "downloads/nicla-v5.0.1"
    entries = {entry["name"]: entry for entry in manifest["files"]}
    for name in ("firmware.bin", "romfs0.img"):
        path = files / name
        if not path.is_file() or digest(path) != entries[name]["sha256"]:
            raise ValueError("Missing or changed downloaded release file: " + str(path))
    check_romfs(files / "romfs0.img")
    return files / "firmware.bin", files / "romfs0.img"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--stock", action="store_true", help="install stock ROMFS and firmware")
    mode.add_argument("--romfs", type=Path, help="replace model ROMFS only, using an aligned built image")
    args = parser.parse_args()
    tool = shutil.which("dfu-util")
    if not tool:
        raise ValueError("dfu-util is not on PATH")
    firmware = None
    if args.stock:
        firmware, image = stock_files()
    else:
        image = args.romfs.resolve()
        check_romfs(image)
    result = subprocess.run([tool, "-d", DEVICE, "-l"], check=True,
                            text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    print(result.stdout, end="")
    expected = (0, 1) if args.stock else (1,)
    for alternate in expected:
        if not re.search(r"\[2341:035f\].*\balt=" + str(alternate) + r"\b", result.stdout, re.I):
            raise ValueError("Expected Nicla DFU alternate not found. Connect the board and double-press reset.")
    print("Writing Nicla model ROMFS:", image)
    subprocess.run([tool, "-d", DEVICE, "-a", "1", "-s", "0x90B00000", "-D", str(image)], check=True)
    if firmware:
        print("Writing Nicla application firmware:", firmware)
        subprocess.run([tool, "-d", DEVICE, "-a", "0", "-s", "0x08040000:leave", "-D", str(firmware)], check=True)
        print("Stock firmware and ROMFS transfer complete. Confirm the board boots after re-enumeration.")
    else:
        print("ROMFS transfer complete. Press reset once to restart the existing firmware.")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, ValueError, KeyError, subprocess.CalledProcessError) as error:
        print("Nicla upload stopped:", error, file=sys.stderr)
        raise SystemExit(1)

