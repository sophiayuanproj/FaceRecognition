"""Create the separate, pinned TensorFlow training environment."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    default_python = sys.executable if sys.version_info[:2] in ((3, 9), (3, 10)) else "python3.10"
    parser.add_argument("--python", default=os.environ.get("NICLA_TRAIN_PYTHON", default_python),
                        help="Native Python 3.9 or 3.10 executable")
    args = parser.parse_args()
    probe = subprocess.run([args.python, "-c",
        "import json,platform,sys; print(json.dumps([list(sys.version_info[:2]),platform.system(),platform.machine()]))"],
        check=True, text=True, capture_output=True)
    version, system, machine = json.loads(probe.stdout)
    if version not in ([3, 9], [3, 10]):
        raise SystemExit("This recorded TensorFlow 2.10 training stack needs Python 3.9 or 3.10. Pass --python with that executable.")
    if system == "Darwin" and machine == "arm64":
        tensorflow = "tensorflow-macos==2.10.0"
    elif machine.lower() in {"x86_64", "amd64"}:
        tensorflow = "tensorflow==2.10.1"
    else:
        raise SystemExit("The recorded training stack supports Intel macOS/Linux/Windows and native Apple Silicon macOS.")
    environment = ROOT / "work/model-training/.venv"
    subprocess.run([args.python, "-m", "venv", str(environment)], check=True)
    python = environment / ("Scripts/python.exe" if system == "Windows" else "bin/python")
    subprocess.run([str(python), "-m", "pip", "install", "--only-binary=:all:",
                    tensorflow, "-r", str(ROOT / "workflow/requirements-training.txt")], check=True)
    subprocess.run([str(python), "-m", "pip", "check"], check=True)
    subprocess.run([str(python), "-c",
                    "import cv2,h5py,numpy,platform,tensorflow; print(platform.machine(),tensorflow.__version__,numpy.__version__,cv2.__version__)"],
                   check=True)
    print("Training Python:", python)


if __name__ == "__main__":
    main()
