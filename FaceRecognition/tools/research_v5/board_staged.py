"""Run ordered source stages in one standard-raw-REPL soft-reset session."""

import argparse
import hashlib
import json
from pathlib import Path
import time

from board_lab import connect


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--port", required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--timeout", type=int, default=90)
    p.add_argument("--program", type=Path, nargs="+", required=True)
    args = p.parse_args()
    if not 1 <= args.timeout <= 600:
        raise ValueError("Bounded stage timeout required")
    out = args.output
    if out.exists() and any(out.iterdir()):
        raise FileExistsError("Refusing to overwrite staged-run evidence: " + str(out))
    out.mkdir(parents=True, exist_ok=True)

    target = None
    rows = []
    try:
        target = connect(args.port)
        target.enter_raw_repl(soft_reset=True)
        for index, source_path in enumerate(args.program, 1):
            source = source_path.read_bytes()
            name = "stage_%02d" % index
            (out / (name + ".py")).write_bytes(source)
            row = {
                "stage": index,
                "source": source_path.name,
                "source_sha256": hashlib.sha256(source).hexdigest(),
                "reset_kind": "same_session_after_one_independent_soft_reset",
            }
            started = time.monotonic()
            try:
                stdout, stderr = target.exec_raw(source, timeout=args.timeout)
                (out / (name + ".stdout.bin")).write_bytes(stdout)
                (out / (name + ".stderr.bin")).write_bytes(stderr)
                row.update(
                    status=("error" if stderr else ("target_error" if b"R48ALLOC_FAIL " in stdout else "executed")),
                    stdout_sha256=hashlib.sha256(stdout).hexdigest(),
                    stderr_sha256=hashlib.sha256(stderr).hexdigest(),
                    stderr=stderr.decode("utf-8", "replace"),
                )
            except Exception as error:
                row.update(status="transport_error", error=repr(error))
            row["host_elapsed_s"] = time.monotonic() - started
            rows.append(row)
            (out / "RESULT.json").write_text(json.dumps(rows, indent=2, sort_keys=True, allow_nan=False) + "\n")
            print(json.dumps(row, sort_keys=True, allow_nan=False), flush=True)
            if row["status"] != "executed":
                break
    finally:
        if target is not None:
            try:
                target.exit_raw_repl()
            finally:
                target.close()
    return 0 if len(rows) == len(args.program) and all(row["status"] == "executed" for row in rows) else 1


if __name__ == "__main__":
    raise SystemExit(main())
