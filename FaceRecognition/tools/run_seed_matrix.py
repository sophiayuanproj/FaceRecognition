#!/usr/bin/env python3
"""Run each unique trained embedding export on Nicla and remove staged files."""

import argparse
from collections import Counter, defaultdict
from datetime import datetime
import hashlib
import json
from pathlib import Path
import statistics
import subprocess
import sys


PROJECT_DIR = Path(__file__).resolve().parents[1]
TRAINING_DIR = PROJECT_DIR.parent / "work" / "model-training"
MPREMOTE = PROJECT_DIR / ".venv-nicla" / "bin" / "mpremote"
BOARD_INFO = PROJECT_DIR / "firmware" / "board_info.py"
BOARD_MATRIX = PROJECT_DIR / "firmware" / "seed_matrix.py"
REPORT_DIR = PROJECT_DIR / "reports"
REMOTE_DIR = "/flash/qa_seed_models"
MODEL_NAMES = {"identity_embedding_int8.tflite", "identity_embedding_float.tflite",
               "tiny_int8.tflite", "tiny_float.tflite"}
MINIMUM_FREE_MARGIN = 2 * 1024 * 1024


def command(args, capture=True):
    return subprocess.run(
        args,
        cwd=PROJECT_DIR,
        text=True,
        stdout=subprocess.PIPE if capture else None,
        stderr=subprocess.STDOUT if capture else None,
        check=False,
    )


def mpremote(port, *args):
    return [str(MPREMOTE), "connect", port, *args]


def json_events(output):
    events = []
    for line in output.splitlines():
        line = line.strip()
        if not line.startswith("{"):
            continue
        try:
            events.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return events


def discover_models():
    candidates = {}
    source_roots = (TRAINING_DIR, PROJECT_DIR / "models")
    for source_root in source_roots:
        for path in sorted(source_root.rglob("*.tflite")):
            if path.name not in MODEL_NAMES:
                continue
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            candidates.setdefault(digest, path)
    result = []
    destination_names = set()
    for digest, source in sorted(candidates.items()):
        name = "q" + digest[:16] + ".tflite"
        if name in destination_names:
            raise RuntimeError("truncated SHA-256 filename collision: " + name)
        destination_names.add(name)
        stem = source.stem
        architecture = "tiny" if (stem.startswith("tiny_") or
                                   any(part == "tiny" or part.startswith("tiny_")
                                       for part in source.parts)) else "base"
        quantization = "int8" if stem.endswith("_int8") else "float"
        variant = architecture + "_" + quantization
        result.append({
            "name": name,
            "board_path": REMOTE_DIR + "/" + name,
            "source": str(source.relative_to(PROJECT_DIR.parent)),
            "sha256": digest,
            "size_bytes": source.stat().st_size,
            "variant": variant,
            "local_path": source,
        })
    return result


def board_info(port):
    result = command(mpremote(port, "run", str(BOARD_INFO)))
    if result.returncode:
        raise RuntimeError("read-only board diagnostics failed: " + result.stdout[-800:])
    events = json_events(result.stdout)
    storage = next((event for event in events if event.get("event") == "flash_storage"), None)
    entries = [event.get("name") for event in events if event.get("event") == "flash_entry"]
    if storage is None:
        raise RuntimeError("board diagnostics did not return /flash capacity")
    return result.stdout, storage, entries


def remove_remote_dir(port):
    result = command(mpremote(port, "fs", "-r", "rm", ":" + REMOTE_DIR))
    return result.returncode, result.stdout or ""


def summarize_probe(output, manifest):
    by_name = {row["name"]: row for row in manifest}
    rows = [row for row in json_events(output) if row.get("event") == "seed_model"]
    totals = Counter(row.get("status", "unknown") for row in rows)
    grouped = defaultdict(list)
    for row in rows:
        model = by_name.get(row.get("name"), {})
        if row.get("inference_us_each"):
            grouped[model.get("variant", "unknown")].extend(row["inference_us_each"])
    return {
        "tested_rows": len(rows),
        "passed": totals.get("passed", 0),
        "failed": totals.get("failed", 0),
        "variant_latency_us_median": {
            variant: int(statistics.median(values)) for variant, values in sorted(grouped.items()) if values
        },
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", required=True, help="Explicit Nicla USB serial port")
    args = parser.parse_args()
    if not MPREMOTE.is_file() or not BOARD_INFO.is_file() or not BOARD_MATRIX.is_file():
        parser.error("the Nicla environment or board scripts are missing")

    models = discover_models()
    if not models:
        parser.error("no trained embedding TFLite exports found")
    staged_bytes = sum(row["size_bytes"] for row in models)
    variants = Counter(row["variant"] for row in models)
    report_path = REPORT_DIR / ("nicla_seed_matrix_" + datetime.now().strftime("%Y%m%dT%H%M%S") + ".jsonl")
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    print("Checking current /flash capacity and QA directory before staging.", flush=True)
    before_output, before_storage, before_entries = board_info(args.port)
    if REMOTE_DIR.rsplit("/", 1)[1] in before_entries:
        parser.error("refusing to overwrite an existing " + REMOTE_DIR)
    if before_storage["available_bytes"] < staged_bytes + MINIMUM_FREE_MARGIN:
        parser.error("insufficient /flash headroom for complete staged sweep")

    records = [json.dumps({
        "event": "host_candidate_manifest",
        "candidate_count": len(models),
        "unique_model_bytes": staged_bytes,
        "variant_counts": dict(variants),
        "models": [{key: value for key, value in row.items() if key != "local_path"} for row in models],
    }, sort_keys=True)]
    records.extend(line for line in before_output.splitlines() if line.strip())
    staged = False
    matrix_output = ""
    matrix_code = 1
    cleanup_notes = []
    after_output = ""
    after_storage = None
    try:
        print("Staging " + str(len(models)) + " unique TFLite candidates (" + str(staged_bytes) + " bytes total).", flush=True)
        mkdir_result = command(mpremote(args.port, "fs", "mkdir", ":" + REMOTE_DIR))
        if mkdir_result.returncode:
            raise RuntimeError("could not create temporary model directory: " + (mkdir_result.stdout or "")[-500:])
        staged = True

        copy_args = [str(MPREMOTE), "connect", args.port]
        for index, model in enumerate(models):
            if index:
                copy_args.append("+")
            copy_args.extend(("fs", "cp", str(model["local_path"]), ":" + model["board_path"]))
        copy_result = command(copy_args)
        if copy_result.returncode:
            raise RuntimeError("model upload stopped early: " + (copy_result.stdout or "")[-800:])

        print("Running three same-input inference passes for each model on one captured crop.", flush=True)
        matrix_result = command(mpremote(args.port, "run", str(BOARD_MATRIX)))
        matrix_code = matrix_result.returncode
        matrix_output = matrix_result.stdout or ""
    except Exception as error:
        cleanup_notes.append("HOST_RUNNER_ERROR: " + str(error))
    finally:
        if staged:
            print("Removing the temporary candidate directory.", flush=True)
            code, cleanup_output = remove_remote_dir(args.port)
            cleanup_notes.extend(line for line in cleanup_output.splitlines() if line.strip())
            if code:
                cleanup_notes.append("CLEANUP_WARNING: recursive removal failed (exit " + str(code) + ")")
            else:
                cleanup_notes.append("CLEANUP_COMPLETE: " + REMOTE_DIR + " removed")
        try:
            after_output, after_storage, after_entries = board_info(args.port)
            if REMOTE_DIR.rsplit("/", 1)[1] in after_entries:
                cleanup_notes.append("CLEANUP_WARNING: temporary directory remains on /flash")
        except Exception as error:
            cleanup_notes.append("POST_RUN_STORAGE_CHECK_FAILED: " + str(error))

    records.extend(line for line in matrix_output.splitlines() if line.strip())
    records.extend(cleanup_notes)
    if after_output:
        records.extend(line for line in after_output.splitlines() if line.strip())
    report_path.write_text("\n".join(records) + "\n", encoding="utf-8")

    summary = summarize_probe(matrix_output, models)
    final_event = next((row for row in json_events(matrix_output) if row.get("event") == "seed_matrix_complete"), {})
    print("Candidate sweep summary: " + json.dumps({
        "candidates_staged": len(models),
        "unique_model_bytes": staged_bytes,
        "variant_counts": dict(variants),
        **summary,
        "board_reported_completion": final_event,
        "free_bytes_before": before_storage["available_bytes"],
        "free_bytes_after": after_storage["available_bytes"] if after_storage else None,
        "temporary_directory_removed": REMOTE_DIR.rsplit("/", 1)[1] not in (after_entries if after_output else ["unknown"]),
        "report": str(report_path),
    }, sort_keys=True))
    return 0 if matrix_code == 0 and summary["tested_rows"] == len(models) else 1


if __name__ == "__main__":
    raise SystemExit(main())
