#!/usr/bin/env python3
"""Safely stage and test the compression sweep models on the connected Nicla."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path


PROJECT = Path(__file__).resolve().parents[1]
ROOT = PROJECT.parent
ARTIFACTS = ROOT / "work/model-training/compression/strategy_sweep_20261022"
MPREMOTE = PROJECT / ".venv-nicla/bin/mpremote"
BOARD_SCRIPT = PROJECT / "firmware/compression_matrix_probe.py"
REMOTE_DIR = "/flash/qa_compression_20261022"
MINIMUM_HEADROOM = 2 * 1024 * 1024
FAMILIES = ("base", "tiny")
FORMATS = (
    "float32", "float16_weights", "dynamic_int8_weights", "full_int8_io",
)
EXTRA_SOURCES = (
    ("magnitude_prune_50_dense", ARTIFACTS.parent / "advanced_strategy_sweep_corrected_20261022/magnitude_prune_50/full_int8_dense_encoding.tflite"),
    ("magnitude_prune_50_sparse", ARTIFACTS.parent / "advanced_strategy_sweep_corrected_20261022/magnitude_prune_50/full_int8_sparse_encoding.tflite"),
    ("structured_prune_2of4_dense", ARTIFACTS.parent / "advanced_strategy_sweep_corrected_20261022/structured_prune_2of4/full_int8_dense_encoding.tflite"),
    ("structured_prune_2of4_sparse", ARTIFACTS.parent / "advanced_strategy_sweep_corrected_20261022/structured_prune_2of4/full_int8_sparse_encoding.tflite"),
    ("cluster_16_full_int8", ARTIFACTS.parent / "advanced_strategy_sweep_corrected_20261022/cluster_16/full_int8_dense_encoding.tflite"),
    ("qat_full_int8", ARTIFACTS.parent / "qat_strategy_sweep_20261022/qat_full_int8/full_int8_dense_encoding.tflite"),
)


def mp(port, *args):
    return subprocess.run(
        [str(MPREMOTE), "connect", port, *args],
        cwd=PROJECT, text=True, stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT, check=False,
    )


def events(output):
    for line in output.splitlines():
        line = line.strip()
        if line.startswith("{"):
            try:
                yield json.loads(line)
            except ValueError:
                pass


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", required=True)
    parser.add_argument("--output", type=Path, default=ARTIFACTS / "nicla_compression_matrix_20261022.jsonl")
    args = parser.parse_args()
    source_specs = [
        (family + "_" + fmt, ARTIFACTS / family / (family + "_" + fmt + ".tflite"))
        for family in FAMILIES for fmt in FORMATS
    ] + list(EXTRA_SOURCES)
    missing = [str(path) for _, path in source_specs if not path.is_file()]
    if missing:
        parser.error("compression artifacts are missing: " + ", ".join(missing))
    if not MPREMOTE.is_file() or not BOARD_SCRIPT.is_file():
        parser.error("mpremote or board probe script is missing")
    args.output.parent.mkdir(parents=True, exist_ok=True)

    before = mp(args.port, "run", str(PROJECT / "firmware/board_info.py"))
    if before.returncode:
        parser.error("read-only board inventory failed: " + before.stdout[-800:])
    before_events = list(events(before.stdout))
    storage = next((e for e in before_events if e.get("event") == "flash_storage"), None)
    entries = {e.get("name") for e in before_events if e.get("event") == "flash_entry"}
    if storage is None:
        parser.error("board did not report /flash capacity")
    if REMOTE_DIR.rsplit("/", 1)[1] in entries:
        parser.error("refusing to overwrite an existing temporary model directory")
    total_bytes = sum(path.stat().st_size for _, path in source_specs)
    if storage["available_bytes"] < total_bytes + MINIMUM_HEADROOM:
        parser.error("insufficient /flash headroom for QA staging")

    log = [
        json.dumps({"event": "host_compression_manifest",
                    "staged_models": [{"name": name, "bytes": path.stat().st_size} for name, path in source_specs],
                    "total_model_bytes": total_bytes,
                    "board_free_before_bytes": storage["available_bytes"]}, sort_keys=True),
    ]
    staged = False
    probe_code = 1
    after_storage = None
    try:
        result = mp(args.port, "fs", "mkdir", ":" + REMOTE_DIR)
        if result.returncode:
            raise RuntimeError("temporary board directory creation failed: " + result.stdout[-800:])
        staged = True

        command = [str(MPREMOTE), "connect", args.port]
        for index, (name, source) in enumerate(source_specs):
            if index:
                command.append("+")
            command.extend(("fs", "cp", str(source), ":" + REMOTE_DIR + "/" + name + ".tflite"))
        copied = subprocess.run(command, cwd=PROJECT, text=True, stdout=subprocess.PIPE,
                                stderr=subprocess.STDOUT, check=False)
        if copied.returncode:
            raise RuntimeError("QA model staging failed: " + copied.stdout[-1000:])
        probe = mp(args.port, "run", str(BOARD_SCRIPT))
        probe_code = probe.returncode
        log.extend(probe.stdout.splitlines())
    except Exception as error:
        log.append(json.dumps({"event": "host_compression_runner_error", "error": str(error)}))
    finally:
        if staged:
            removed = mp(args.port, "fs", "-r", "rm", ":" + REMOTE_DIR)
            log.extend(removed.stdout.splitlines())
            log.append(json.dumps({"event": "temporary_directory_cleanup",
                                   "path": REMOTE_DIR, "success": removed.returncode == 0}))
        try:
            after = mp(args.port, "run", str(PROJECT / "firmware/board_info.py"))
            log.extend(after.stdout.splitlines())
            if after.returncode == 0:
                after_events = list(events(after.stdout))
                after_storage = next((e for e in after_events if e.get("event") == "flash_storage"), None)
                remaining = {e.get("name") for e in after_events if e.get("event") == "flash_entry"}
                log.append(json.dumps({
                    "event": "post_cleanup_storage_check",
                    "qa_directory_present": REMOTE_DIR.rsplit("/", 1)[1] in remaining,
                    "free_bytes_after": after_storage.get("available_bytes") if after_storage else None,
                }))
        except Exception as error:
            log.append(json.dumps({"event": "post_cleanup_storage_check_failed", "error": str(error)}))
        # Reboot the verified standalone main.py after the REPL-based probe.
        reset = mp(args.port, "reset")
        log.extend(reset.stdout.splitlines())
        log.append(json.dumps({"event": "standalone_reboot_requested", "success": reset.returncode == 0}))

    args.output.write_text("\n".join(log) + "\n")
    rows = [e for e in events("\n".join(log)) if e.get("event") == "compression_model_probe"]
    result = {
        "probe_exit_code": probe_code,
        "models_passed": sum(e.get("status") == "passed" for e in rows),
        "models_failed": sum(e.get("status") == "failed" for e in rows),
        "models_expected": len(source_specs),
        "board_free_before_bytes": storage["available_bytes"],
        "board_free_after_bytes": after_storage.get("available_bytes") if after_storage else None,
        "report": str(args.output.resolve()),
    }
    print(json.dumps(result, sort_keys=True))
    return 0 if result["models_passed"] == len(source_specs) and probe_code == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())

