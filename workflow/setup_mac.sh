#!/bin/sh
set -eu
cd "$(dirname "$0")/.."
case "${1:-train}" in
  train)
    exec python3 workflow/setup_training.py --python "${NICLA_TRAIN_PYTHON:-python3.10}"
    ;;
  app)
    echo "For the current camera application, use the setup and launcher commands in START_HERE.md."
    ;;
  *) echo "Usage: sh workflow/setup_mac.sh train" >&2; exit 2 ;;
esac
