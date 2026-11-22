"""Ten-root real cross-process test for the Windows named-mutex lock."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time

from hardware_lock_windows import HardwareLock


def child(lock_path, run_id, timeout_s, ready_path=None, hold_s=0.0):
    try:
        with HardwareLock(lock_path, run_id, timeout_s):
            if ready_path:
                Path(ready_path).write_text("acquired\n", encoding="utf-8")
            print("ACQUIRED", flush=True)
            if hold_s:
                time.sleep(hold_s)
        return 0
    except TimeoutError:
        print("BUSY", flush=True)
        return 6


def run_ten_roots(project_root):
    if os.name != "nt":
        raise OSError("run this qualification on native Windows")
    base = Path(project_root) / "reports" / ("windows-lock-ten-roots-" + time.strftime("%Y%m%dT%H%M%S") + "-%09d" % (time.time_ns() % 1000000000))
    base.mkdir(parents=True, exist_ok=False)
    rows = []
    script = Path(__file__).resolve()
    for root_seed in range(1, 11):
        root = base / ("root-%02d-seed-%04d" % (root_seed, 5100 + root_seed))
        root.mkdir()
        lock_path = root / "hardware.lock"
        ready = root / "owner-acquired.txt"
        command = [sys.executable, str(script), "--child", str(lock_path),
                   "owner-%04d" % (5100 + root_seed), "0", str(ready), "0.45"]
        owner = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                  text=True, creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        deadline = time.monotonic() + 10
        while not ready.exists() and owner.poll() is None and time.monotonic() < deadline:
            time.sleep(0.02)
        if not ready.exists():
            out, err = owner.communicate(timeout=2)
            raise AssertionError("root %d owner did not acquire: %s %s %s" % (root_seed, out, err, owner.returncode))
        contender = subprocess.run(
            [sys.executable, str(script), "--child", str(lock_path),
             "contender-%04d" % (5100 + root_seed), "0", "", "0"],
            text=True, capture_output=True, timeout=10,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        owner_out, owner_err = owner.communicate(timeout=10)
        if contender.returncode != 6 or "BUSY" not in contender.stdout:
            raise AssertionError("root %d contender was not excluded: %r %r" %
                                 (root_seed, contender.stdout, contender.stderr))
        if owner.returncode != 0:
            raise AssertionError("root %d owner failed: %r %r" % (root_seed, owner_out, owner_err))
        successor = subprocess.run(
            [sys.executable, str(script), "--child", str(lock_path),
             "successor-%04d" % (5100 + root_seed), "2.0", "", "0"],
            text=True, capture_output=True, timeout=10,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        if successor.returncode != 0 or "ACQUIRED" not in successor.stdout:
            raise AssertionError("root %d did not release for successor: %r %r" %
                                 (root_seed, successor.stdout, successor.stderr))
        metadata = json.loads(lock_path.read_text(encoding="utf-8"))
        if metadata.get("state") != "released":
            raise AssertionError("root %d lock metadata did not finalize" % root_seed)
        rows.append({"root_seed": 5100 + root_seed, "status": "pass",
                     "owner_process": owner.pid, "contender_return_code": contender.returncode,
                     "successor_return_code": successor.returncode,
                     "evidence_dir": str(root)})
    report = {"suite": "windows_named_mutex_cross_process_10_roots",
              "backend": "Local Windows named mutex",
              "roots_expected": 10, "roots_passed": len(rows), "rows": rows,
              "claim": "cross-process exclusion and post-owner release demonstrated; no HID hardware claim"}
    report_path = base / "summary.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"report": str(report_path), "roots_passed": len(rows)}, indent=2))
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path)
    parser.add_argument("--child", nargs=5, metavar=("LOCK", "RUN", "TIMEOUT", "READY", "HOLD"))
    args = parser.parse_args()
    if args.child:
        lock_path, run_id, timeout_s, ready_path, hold_s = args.child
        return child(lock_path, run_id, float(timeout_s), ready_path or None, float(hold_s))
    if not args.project_root:
        parser.error("--project-root is required")
    run_ten_roots(args.project_root.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

