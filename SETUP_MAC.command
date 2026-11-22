#!/bin/sh
set -eu

workspace=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
app="$workspace/FaceRecognition"
python_bin=${NICLA_PYTHON:-python3.12}

if [ "${1:-}" = "--help" ]; then
    echo "Run SETUP_MAC.command to install the local camera dashboard and deployment tools."
    echo "Requires Python 3.12. Set NICLA_PYTHON to use a specific Python executable."
    echo "Run SETUP_MAC.command --check to check an existing installation."
    exit 0
fi
if [ "$#" -gt 1 ] || { [ "$#" -eq 1 ] && [ "$1" != "--check" ]; }; then
    echo "Usage: SETUP_MAC.command [--check]" >&2
    exit 2
fi

if [ "${1:-}" != "--check" ]; then
    if ! command -v "$python_bin" >/dev/null 2>&1; then
        echo "Python 3.12 was not found. Install it or set NICLA_PYTHON to its executable." >&2
        exit 2
    fi
    "$python_bin" -c 'import sys; assert sys.version_info[:2] == (3, 12), "Use Python 3.12."'
    "$python_bin" -m venv "$app/.venv-nicla"
    "$app/.venv-nicla/bin/python" -m pip install --disable-pip-version-check \
        -r "$app/requirements-control-center.txt"
fi

if [ ! -x "$app/.venv-nicla/bin/python" ]; then
    echo "Run SETUP_MAC.command first to create the local environment." >&2
    exit 2
fi
"$app/.venv-nicla/bin/python" -c 'import sys, serial, mpremote; from PIL import Image; assert sys.version_info[:2] == (3, 12); print("Camera and deployment dependencies are ready.")'
sh "$app/bin/nicla-control-center" --doctor --port 0
echo "Setup passed. Open FaceRecognition/bin/nicla-control-center.command to start the dashboard."
