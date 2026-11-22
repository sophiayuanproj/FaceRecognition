#!/usr/bin/env python3
"""Build an aligned OpenMV model image from real supplied exports."""

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "romfs/project.json")
    args = parser.parse_args()
    config_path = args.config.resolve()
    config = json.loads(config_path.read_text())
    partition = config.get("0")
    if partition is None or int(partition["size"], 16) != 4 * 1024 * 1024:
        raise ValueError("Use Nicla ROMFS partition 0 with size 0x400000")
    if set(config) != {"0"}:
        raise ValueError("This helper builds only Nicla partition 0")
    source_files = []
    names = set()
    for entry in partition["entries"]:
        path = Path(entry["path"].replace("{TOP}", str(ROOT)))
        if not path.is_absolute():
            path = ROOT / path
        if not path.is_file():
            raise ValueError("Missing export: " + str(path) + ". Supply trained models before building.")
        if path.name in names:
            raise ValueError("Duplicate ROMFS basename: " + path.name)
        names.add(path.name)
        source_files.append(path)
        if entry["type"] == "tflite":
            if entry.get("alignment") != 16:
                raise ValueError("TFLite entries must specify alignment: 16")
            labels = path.with_suffix(".txt")
            if not labels.is_file() or not labels.read_text().strip():
                raise ValueError("Missing nonempty matching labels: " + str(labels))
            if labels.name in names:
                raise ValueError("Duplicate ROMFS label basename: " + labels.name)
            names.add(labels.name)
            source_files.append(labels)
    output = ROOT / "build/romfs"
    output.mkdir(parents=True, exist_ok=True)
    subprocess.run([
        sys.executable, str(ROOT / "tools/vendor/openmv/mkromfs.py"),
        "--config", str(config_path), "--top-dir", str(ROOT),
        "--out-dir", str(output), "--build-dir", str(output / "intermediate"), "-j", "1",
    ], check=True)
    image = output / "romfs0.img"
    report = {
        "board": "ARDUINO_NICLA_VISION", "builder_release": "v5.0.1",
        "image": image.name, "bytes": image.stat().st_size, "sha256": digest(image),
        "sources": [{"name": p.name, "bytes": p.stat().st_size, "sha256": digest(p)} for p in source_files],
        "status": "built; not uploaded or hardware validated",
    }
    (output / "build.json").write_text(json.dumps(report, indent=2) + "\n")
    print("Aligned model image:", image)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, ValueError, KeyError, subprocess.CalledProcessError) as error:
        print("Model build stopped:", error, file=sys.stderr)
        raise SystemExit(1)

