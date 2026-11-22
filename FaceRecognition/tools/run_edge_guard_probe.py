#!/usr/bin/env python3
"""Run edge guards on actual MicroPython using a temporary import-only module."""

from __future__ import annotations

import argparse
from datetime import datetime
import json
import subprocess
from pathlib import Path


PROJECT = Path(__file__).resolve().parents[1]
MPREMOTE = PROJECT / ".venv-nicla/bin/mpremote"
BOARD_INFO = PROJECT / "firmware/board_info.py"
BOARD_PROBE = PROJECT / "firmware/edge_guard_probe.py"
ENROLLMENT_SOURCE = PROJECT / "firmware/enroll_identity.py"
QA_MODULE = "/flash/qa_enroll_identity.py"
REPORT_DIR = PROJECT / "reports"


def mp(port, *args):
    return subprocess.run([str(MPREMOTE), "connect", port, *args],
                          cwd=PROJECT, text=True, stdout=subprocess.PIPE,
                          stderr=subprocess.STDOUT, check=False)


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
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    if not MPREMOTE.is_file() or not BOARD_INFO.is_file() or not BOARD_PROBE.is_file():
        parser.error("mpremote or board probe files are missing")
    output_path = args.output or REPORT_DIR / (
        "nicla_edge_guard_probe_" + datetime.now().strftime("%Y%m%dT%H%M%S") + ".jsonl"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)

    before = mp(args.port, "run", str(BOARD_INFO))
    if before.returncode:
        parser.error("read-only board inventory failed: " + before.stdout[-700:])
    before_events = list(events(before.stdout))
    entries = {event.get("name") for event in before_events if event.get("event") == "flash_entry"}
    if QA_MODULE.rsplit("/", 1)[1] in entries:
        parser.error("refusing to overwrite existing QA import module")
    log = [json.dumps({"event": "host_edge_probe_manifest",
                       "qa_module": QA_MODULE,
                       "camera_accessed": False,
                       "images_saved": 0}, sort_keys=True)]
    log.extend(before.stdout.splitlines())

    staged = False
    cleanup_ok = False
    probe_status = "not_run"
    try:
        copied = mp(args.port, "fs", "cp", str(ENROLLMENT_SOURCE), ":" + QA_MODULE)
        if copied.returncode:
            raise RuntimeError("could not stage QA import module: " + copied.stdout[-600:])
        staged = True
        probe = mp(args.port, "run", str(BOARD_PROBE))
        log.extend(probe.stdout.splitlines())
        probe_events = list(events(probe.stdout))
        probe_status = next((row.get("status") for row in probe_events
                             if row.get("event") == "edge_guard_probe"), "missing_result")
        if probe.returncode:
            log.append(json.dumps({"event": "board_probe_nonzero_exit", "code": probe.returncode}))
    except Exception as error:
        log.append(json.dumps({"event": "host_edge_probe_error", "error": str(error)}))
    finally:
        if staged:
            removed = mp(args.port, "fs", "rm", ":" + QA_MODULE)
            log.extend(removed.stdout.splitlines())
            cleanup_ok = removed.returncode == 0
            log.append(json.dumps({"event": "qa_module_cleanup",
                                   "success": cleanup_ok}))
        after = mp(args.port, "run", str(BOARD_INFO))
        log.extend(after.stdout.splitlines())
        if after.returncode == 0:
            after_events = list(events(after.stdout))
            remaining = {row.get("name") for row in after_events if row.get("event") == "flash_entry"}
            log.append(json.dumps({"event": "post_probe_cleanup",
                                   "qa_module_present": QA_MODULE.rsplit("/", 1)[1] in remaining}))
        reset = mp(args.port, "reset")
        log.append(json.dumps({"event": "standalone_reboot_requested",
                               "success": reset.returncode == 0}))

    output_path.write_text("\n".join(log) + "\n", encoding="utf-8")
    print(json.dumps({"board_probe_status": probe_status,
                      "report": str(output_path.resolve()),
                      "temporary_import_module_removed": cleanup_ok}, sort_keys=True))
    return 0 if probe_status == "passed" and cleanup_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())

