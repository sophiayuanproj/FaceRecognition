#!/bin/sh
# Reproduce the native host environment. Does not connect to the board.
set -eu
PROJECT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
HOST_PYTHON=${NICLA_PYTHON:-python3}
"$HOST_PYTHON" -m venv "$PROJECT_DIR/.venv-nicla"
PIP_CACHE_DIR="$PROJECT_DIR/.cache/pip" "$PROJECT_DIR/.venv-nicla/bin/python" -m pip install \
    -r "$PROJECT_DIR/requirements-host.lock"
"$PROJECT_DIR/bin/nicla" arduino core update-index
"$PROJECT_DIR/bin/nicla" arduino core install arduino:mbed_nicla@4.6.0
"$PROJECT_DIR/.venv-nicla/bin/python" "$PROJECT_DIR/tools/download_firmware.py"
echo "Host dependencies and release files installed. Board setup awaits the Nicla."

