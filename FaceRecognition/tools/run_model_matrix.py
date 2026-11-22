#!/usr/bin/env python3
"""Temporarily stage comparison exports, run the headless Nicla matrix, clean up."""

import argparse
from datetime import datetime
from pathlib import Path
import subprocess
import sys


PROJECT_DIR = Path(__file__).resolve().parents[1]
ARTIFACTS_DIR = PROJECT_DIR.parent / "work" / "model-training" / "artifacts"
MPREMOTE = PROJECT_DIR / ".venv-nicla" / "bin" / "mpremote"
BOARD_SCRIPT = PROJECT_DIR / "firmware" / "model_matrix.py"
REPORT_DIR = PROJECT_DIR / "reports"

STAGED_MODELS = (
    (ARTIFACTS_DIR / "tiny_int8.tflite", "/flash/qa_tiny_int8.tflite"),
    (ARTIFACTS_DIR / "identity_embedding_float.tflite", "/flash/qa_base_float.tflite"),
    (ARTIFACTS_DIR / "tiny_float.tflite", "/flash/qa_tiny_float.tflite"),
)


def run(command, *, capture=False):
    result = subprocess.run(
        command,
        cwd=PROJECT_DIR,
        text=True,
        stdout=subprocess.PIPE if capture else None,
        stderr=subprocess.STDOUT if capture else None,
        check=False,
    )
    return result.returncode, result.stdout if capture else ""


def mpremote(port, *args):
    return [str(MPREMOTE), "connect", port, *args]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", required=True, help="Explicit Nicla USB serial port")
    args = parser.parse_args()

    missing = [str(source) for source, _ in STAGED_MODELS if not source.is_file()]
    if missing:
        parser.error("missing comparison exports: " + ", ".join(missing))
    if not MPREMOTE.is_file():
        parser.error("missing project mpremote: " + str(MPREMOTE))

    report_path = REPORT_DIR / ("nicla_model_matrix_" + datetime.now().strftime("%Y%m%dT%H%M%S") + ".jsonl")
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    staged = []
    run_return_code = 1
    output = ""
    try:
        # These QA-only names were checked absent in the live /flash inventory.
        for source, remote_path in STAGED_MODELS:
            code, text = run(mpremote(args.port, "fs", "cp", str(source), ":" + remote_path), capture=True)
            output += text
            if code:
                raise RuntimeError("could not stage " + source.name + " (mpremote exit " + str(code) + ")")
            staged.append(remote_path)

        run_return_code, output = run(mpremote(args.port, "run", str(BOARD_SCRIPT)), capture=True)
    except Exception as error:
        output += "\nHOST_RUNNER_ERROR: " + str(error) + "\n"
    finally:
        for remote_path in reversed(staged):
            code, text = run(mpremote(args.port, "fs", "rm", ":" + remote_path), capture=True)
            output += text
            if code:
                output += "\nCLEANUP_WARNING: could not remove " + remote_path + " (mpremote exit " + str(code) + ")\n"

    report_path.write_text(output, encoding="utf-8")
    sys.stdout.write(output)
    print("Report saved: " + str(report_path))
    print("Temporary comparison models removed: " + str(len(staged)))
    return run_return_code


if __name__ == "__main__":
    raise SystemExit(main())
