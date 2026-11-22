#!/usr/bin/env python3
"""Download the exact public R19 power study from this repository's release."""
from __future__ import annotations

import hashlib
from pathlib import Path
import sys
import urllib.request

NAME = "Nicla_R19_Mac_Return_Belkin_10_Pairs_Final.zip"
SHA256 = "66f9d14dd4b75b87d79d974cd28dc17492c5f3625db6edbc3255911e347e2356"
SIZE = 41332455
URL = "https://github.com/sophiayuanproj/FaceRecognition/releases/download/research-assets-2026-11-22/" + NAME
ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    directory = ROOT / "assets"
    directory.mkdir(exist_ok=True)
    target = directory / NAME
    pending = target.with_suffix(".zip.part")
    if not target.exists():
        try:
            request = urllib.request.Request(URL, headers={"User-Agent": "Nicla-research-assets"})
            with urllib.request.urlopen(request, timeout=60) as source, pending.open("wb") as output:
                size = 0
                while block := source.read(1024 * 1024):
                    size += len(block)
                    if size > SIZE:
                        raise ValueError("Evidence download exceeded its expected size")
                    output.write(block)
            candidate = pending
        except (OSError, ValueError) as error:
            pending.unlink(missing_ok=True)
            print(f"Power evidence download failed: {error}", file=sys.stderr)
            return 1
    else:
        candidate = target
    digest = hashlib.sha256(candidate.read_bytes()).hexdigest()
    if candidate.stat().st_size != SIZE or digest != SHA256:
        if candidate == pending:
            pending.unlink(missing_ok=True)
        print("Power evidence failed size/SHA-256 verification", file=sys.stderr)
        return 1
    if candidate == pending:
        pending.replace(target)
    print(f"Verified power study: {target}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
