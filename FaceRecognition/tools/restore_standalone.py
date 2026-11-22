#!/usr/bin/env python3
"""Restore the on-board pre-project /flash/main.py and hard-reset the Nicla."""

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MPREMOTE = ROOT / ".venv-nicla" / "bin" / "mpremote"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", required=True)
    args = parser.parse_args()
    prefix = [str(MPREMOTE), "connect", args.port]
    restore = (
        "import os\n"
        "with open('/flash/main.py.stock', 'rb') as src:\n"
        "    data = src.read()\n"
        "with open('/flash/main.py.restore', 'wb') as dst:\n"
        "    dst.write(data)\n"
        "try:\n"
        "    os.remove('/flash/main.py')\n"
        "except OSError:\n"
        "    pass\n"
        "os.rename('/flash/main.py.restore', '/flash/main.py')\n"
        "print('NICLA_STOCK_STARTUP_RESTORED')\n"
    )
    result = subprocess.run([*prefix, "exec", restore], text=True,
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    if result.returncode or "NICLA_STOCK_STARTUP_RESTORED" not in result.stdout:
        raise RuntimeError("Could not restore /flash/main.py:\n" + result.stdout)
    print(result.stdout, end="")
    subprocess.run([*prefix, "reset"], check=True)
    print("Nicla hard-reset; stock startup restored. The model and scripts remain in /flash.")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, RuntimeError, subprocess.CalledProcessError) as error:
        print("Startup restore stopped:", error, file=sys.stderr)
        raise SystemExit(1)
