#!/usr/bin/env python3
"""Copy C's exact raw-byte parity fixtures to a real Nicla without changing startup."""

import argparse
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
PROJECT = ROOT.parent
STUDY = PROJECT / "work/project-finish-v4"
FIXTURE_ROOT = STUDY / "fixtures/C/selected_challenger"
MODEL = STUDY / "models/C/cheap_sepconv_16/seed-20261102/detector_full_int8.tflite"

from mpremote_path import resolve_mpremote

MPREMOTE = resolve_mpremote(ROOT)

def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def invoke(port, *args, check=True):
    if not MPREMOTE.is_file():
        raise RuntimeError("Missing mpremote: " + str(MPREMOTE))
    result = subprocess.run([str(MPREMOTE), "connect", port, *map(str, args)],
        text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    if check and result.returncode:
        raise RuntimeError("mpremote failed: " + (result.stdout or ""))
    return result


def remote_sha(port, path):
    output = invoke(port, "fs", "sha256sum", ":" + path).stdout or ""
    match = re.search(r"\b([0-9a-f]{64})\b", output)
    if not match:
        raise RuntimeError("No remote hash for " + path)
    return match.group(1)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", required=True)
    args = parser.parse_args()
    model_metadata = json.loads((FIXTURE_ROOT / "fixture_manifest.json").read_text())
    if model_metadata["model_sha256"] != sha256(MODEL):
        raise ValueError("C fixture model hash differs from source model")
    payload = {REMOTE_MODEL: MODEL, REMOTE_MODEL_LABELS: LABELS}
    for fixture_id in FIXTURE_IDS:
        folder = FIXTURE_ROOT / fixture_id
        fixture = json.loads((folder / "fixture.json").read_text())
        names = ["fixture.json", "camera_window_gray240.u8", "camera_window_gray240.pgm",
                 "detector_input_gray96.u8", "detector_input_gray96.pgm",
                 "detector_input_int8.i8", "detector_output_int8.i8"]
        if fixture.get("exact_raw_files", {}).get("identity_input_gray96.u8"):
            names.extend(["identity_input_gray96.u8", "identity_input_gray96.pgm", "identity_input_int8.i8"])
        for name in names:
            local = folder / name
            if not local.is_file():
                raise FileNotFoundError("C fixture payload incomplete: " + str(local))
            payload[REMOTE_FIXTURE_ROOT + "/" + fixture_id + "/" + name] = local
    invoke(args.port, "exec", "import os\n"
        "for p in ('/flash/c_fixtures','/flash/c_fixtures/selected_candidate'):\n"
        "    try: os.mkdir(p)\n    except OSError: pass\n"
        "print('C_FIXTURE_ROOT_READY')\n")
    for remote, local in payload.items():
        if remote.startswith(REMOTE_FIXTURE_ROOT + "/"):
            parent = remote.rsplit("/", 1)[0]
            invoke(args.port, "exec", "import os\n"
                "p=%r\n"
                "try: os.mkdir(p)\nexcept OSError: pass\n" % parent)
        existing = invoke(args.port, "fs", "sha256sum", ":" + remote, check=False)
        if existing.returncode == 0:
            if remote_sha(args.port, remote) != sha256(local):
                raise FileExistsError("refusing to overwrite different on-board fixture data: " + remote)
            continue
        invoke(args.port, "fs", "cp", str(local), ":" + remote)
        if remote_sha(args.port, remote) != sha256(local):
            raise RuntimeError("readback checksum mismatch: " + remote)
    manifest = {"status": "fixtures_staged", "fixture_count": len(FIXTURE_IDS),
        "model_sha256": sha256(MODEL), "copied_file_count": len(payload),
        "remote_root": REMOTE_FIXTURE_ROOT,
        "probe_script": str(ROOT / "tools/nicla_fixture_parity.py"),
        "startup_modified": False}
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, ValueError, RuntimeError, subprocess.CalledProcessError) as error:
        print("Exact fixture staging stopped:", error, file=sys.stderr)
        raise SystemExit(1)
