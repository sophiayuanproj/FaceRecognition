#!/usr/bin/env python3
"""Build hash-closed MicroPython bytecode modules for the Nicla runtime."""

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import struct
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
MODULE_SOURCES = {
    "wearable_app": (ROOT / "firmware/main.py", "wearable_app.py"),
    "identity_contracts": (ROOT / "firmware/identity_contracts.py", "identity_contracts.py"),
    "runtime_support": (ROOT / "firmware/runtime_support.py", "runtime_support.py"),
    "telemetry": (ROOT / "firmware/telemetry.py", "telemetry.py"),
}


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--compiler", required=True, help="path to version-matched mpy-cross")
    parser.add_argument("--target-mpy", required=True, type=int,
        help="exact os._mpy value read from the target (e.g. 8966)")
    parser.add_argument("--package-version", required=True,
        help="installable compiler package/version recorded for reproduction")
    parser.add_argument("--out-dir", type=Path, default=ROOT / "firmware/mpy")
    parser.add_argument("--opt-level", type=int, choices=range(4), default=0)
    parser.add_argument("--include-offline-sightings", action="store_true",
                        help="compile the optional offline encounter JPEG spool")
    args = parser.parse_args(argv)
    compiler = Path(args.compiler).expanduser().resolve()
    if not compiler.is_file():
        raise FileNotFoundError("mpy-cross not found: " + str(compiler))
    version = subprocess.run([str(compiler), "--version"], check=True,
        text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT).stdout.strip()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    source_rows = {}
    module_rows = {}
    sources = dict(MODULE_SOURCES)
    if args.include_offline_sightings:
        sources["offline_sightings"] = (ROOT / "firmware/offline_sightings.py", "offline_sightings.py")
    for module, (source, release_source_name) in sources.items():
        if not source.is_file():
            raise FileNotFoundError("runtime source missing: " + str(source))
        output = args.out_dir / (module + ".mpy")
        subprocess.run([str(compiler), "-O%d" % args.opt_level, "-s", module,
                        "-o", str(output), str(source)], check=True)
        data = output.read_bytes()
        if len(data) < 4 or data[0:1] != b"M":
            raise RuntimeError("compiler did not emit a MicroPython module: " + str(output))
        if data[1] != (args.target_mpy & 255):
            raise RuntimeError("compiled format byte %d does not match target os._mpy low byte %d" %
                               (data[1], args.target_mpy & 255))
        source_rows[release_source_name] = {
            "sha256": sha256(source), "bytes": source.stat().st_size}
        module_rows[output.name] = {"module": module, "sha256": hashlib.sha256(data).hexdigest(),
            "bytes": len(data), "source": release_source_name}

    manifest = {
        "schema": "nicla-precompiled-runtime-1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "compiler": version,
        "compiler_package": args.package_version,
        "compile_opt_level": args.opt_level,
        "mpy_value": args.target_mpy,
        "format_low_byte": args.target_mpy & 255,
        "sources": source_rows,
        "modules": module_rows,
    }
    manifest_path = args.out_dir / "build_manifest.json"
    manifest_path.write_text(json.dumps(manifest, sort_keys=True, indent=2) + "\n")
    print(json.dumps({"status": "built", "out_dir": str(args.out_dir),
        "manifest": str(manifest_path), "manifest_sha256": sha256(manifest_path),
        "modules": module_rows}, indent=2))


if __name__ == "__main__":
    try:
        main()
    except (OSError, ValueError, RuntimeError, subprocess.CalledProcessError) as error:
        print("Nicla bytecode build stopped:", error, file=sys.stderr)
        raise SystemExit(1)
