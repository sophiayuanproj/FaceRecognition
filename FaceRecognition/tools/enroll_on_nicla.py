#!/usr/bin/env python3
"""Run explicit local embedding enrollment on a connected Nicla Vision."""

import argparse
import json
import re
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MPREMOTE = ROOT / ".venv-nicla" / "bin" / "mpremote"
ENROLLMENT_SCRIPT = ROOT / "firmware" / "enroll_identity.py"


def invoke(port, *args, check=True):
    command = [str(MPREMOTE), "connect", port, *map(str, args)]
    result = subprocess.run(command, text=True)
    if check and result.returncode:
        raise RuntimeError("Enrollment command failed: " + " ".join(command))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", required=True)
    parser.add_argument("--identity-id", required=True,
                        help="pseudonymous token only; do not use a person's name")
    parser.add_argument("--samples", type=int, default=5)
    parser.add_argument("--attempts", type=int, default=30)
    parser.add_argument("--replace", action="store_true",
                        help="replace an existing template with this identity ID")
    args = parser.parse_args()
    if not re.fullmatch(r"[A-Za-z0-9_-]{1,24}", args.identity_id):
        raise ValueError("identity ID must be a 1-24 character pseudonymous token")
    if not 3 <= args.samples <= 20 or not args.samples <= args.attempts <= 60:
        raise ValueError("Use 3-20 samples and at least that many, up to 60, attempts")
    if not MPREMOTE.is_file() or not ENROLLMENT_SCRIPT.is_file():
        raise RuntimeError("Install the Nicla host environment and firmware files first")

    request = {
        "version": 1,
        "identity_id": args.identity_id,
        "min_samples": args.samples,
        "max_attempts": args.attempts,
        "replace_existing": args.replace,
    }
    print("Enrollment uses the live camera. Show only one consenting person.")
    print("Keep the face centered and vary pose slightly; raw images are not saved.")
    print("Capturing enrollment samples now; press Ctrl-C to cancel.")
    remote_request = "/flash/enrollment_request.json"
    with tempfile.NamedTemporaryFile("w", suffix=".json", encoding="utf-8") as stream:
        json.dump(request, stream)
        stream.flush()
        try:
            invoke(args.port, "fs", "cp", "-f", stream.name, ":" + remote_request)
            invoke(args.port, "run", str(ENROLLMENT_SCRIPT))
        finally:
            # The device script also removes this file on normal completion.
            invoke(args.port, "fs", "rm", ":" + remote_request, check=False)
    print("Enrollment run ended. Hard-reset the Nicla to return to standalone recognition.")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, ValueError, RuntimeError, subprocess.CalledProcessError) as error:
        print("Enrollment stopped:", error, file=sys.stderr)
        raise SystemExit(1)
