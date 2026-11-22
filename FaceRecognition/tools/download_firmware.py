#!/usr/bin/env python3
"""Download the pinned official Nicla release files without flashing a board."""

import hashlib
import json
from pathlib import Path
import urllib.request
import zipfile

ROOT = Path(__file__).resolve().parents[1]


def main():
    manifest = json.loads((ROOT / "config/firmware-release.json").read_text())
    folder = ROOT / "downloads"
    folder.mkdir(exist_ok=True)
    archive = folder / "firmware_v5.0.1.zip"
    if not archive.is_file():
        partial = archive.with_suffix(".zip.part")
        print("Downloading", manifest["source"])
        with urllib.request.urlopen(manifest["source"], timeout=120) as response, partial.open("wb") as stream:
            while chunk := response.read(1024 * 1024):
                stream.write(chunk)
        partial.replace(archive)
    with archive.open("rb") as stream:
        checksum = hashlib.file_digest(stream, "sha256").hexdigest()
    if checksum != manifest["archive_sha256"]:
        raise ValueError("Downloaded archive checksum does not match the pinned release")
    output = folder / "nicla-v5.0.1"
    output.mkdir(exist_ok=True)
    with zipfile.ZipFile(archive) as bundle:
        members = {Path(name).name: name for name in bundle.namelist()
                   if "ARDUINO_NICLA_VISION" in Path(name).parts and not name.endswith("/")}
        for entry in manifest["files"]:
            data = bundle.read(members[entry["name"]])
            if len(data) != entry["bytes"] or hashlib.sha256(data).hexdigest() != entry["sha256"]:
                raise ValueError("Release member checksum mismatch: " + entry["name"])
            (output / entry["name"]).write_bytes(data)
            print("Ready:", output / entry["name"])
    print("Release files are ready; no device has been flashed.")


if __name__ == "__main__":
    main()

