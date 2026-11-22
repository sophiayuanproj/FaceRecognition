#!/usr/bin/env python3
"""Convert and evaluate every completed campaign checkpoint in validation-only mode."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
TRAINING_ROOT = ROOT / "work/model-training"
EVAL_SCRIPT = TRAINING_ROOT / "scripts/convert_eval.py"
PYTHON = TRAINING_ROOT / ".venv/bin/python"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--max-parallel", type=int, default=None)
    args = parser.parse_args()
    plan_path = args.plan if args.plan.is_absolute() else ROOT / args.plan
    plan = json.loads(plan_path.read_text())
    cap = args.max_parallel or plan.get("evaluation", {}).get("parallel_evaluation_jobs", 1)
    if cap < 1:
        parser.error("max parallel jobs must be at least 1")
    campaign_root = TRAINING_ROOT / "runs" / plan["experiment_id"]
    logs_root = campaign_root / "logs"
    logs_root.mkdir(parents=True, exist_ok=True)
    status_path = campaign_root / "evaluation_matrix_status.json"
    status_path.unlink(missing_ok=True)
    jobs = []
    if "evaluation_jobs" in plan:
        for item in plan["evaluation_jobs"]:
            run_dir = Path(item["run_dir"])
            if not run_dir.is_absolute():
                run_dir = ROOT / run_dir
            jobs.append((item["model_variant"], int(item["split_seed"]),
                         int(item["training_seed"]), run_dir))
    else:
        groups = plan.get("job_groups") or [{
            "model_variant": plan["model_variant"],
            "split_seed": plan["split_seed"],
            "training_seeds": plan["training_seeds"],
        }]
        for group in groups:
            for seed in group["training_seeds"]:
                run_dir = campaign_root / group["model_variant"] / ("split-%d" % group["split_seed"]) / ("seed-%d" % seed)
                jobs.append((group["model_variant"], group["split_seed"], seed, run_dir))
    job_keys = [(variant, split_seed, seed) for variant, split_seed, seed, _ in jobs]
    if len(job_keys) != len(set(job_keys)):
        raise SystemExit("Evaluation plan contains duplicate model/split/training-seed jobs")
    failures = []
    completed = []
    for index in range(0, len(jobs), cap):
        batch = jobs[index:index + cap]
        active = []
        for variant, split_seed, seed, run_dir in batch:
            # Never let an old result make a failed or skipped evaluation look current.
            (run_dir / "float_int8_evaluation.json").unlink(missing_ok=True)
            (run_dir / "calibration_manifest.json").unlink(missing_ok=True)
            if not (run_dir / "best_embedding.h5").is_file():
                failures.append({"variant": variant, "split_seed": split_seed,
                                 "seed": seed, "error": "missing best_embedding.h5",
                                 "run_dir": str(run_dir)})
                continue
            artifact_dir = run_dir / "artifacts"
            artifact_dir.mkdir(parents=True, exist_ok=True)
            log_path = logs_root / ("%s-split-%d-seed-%d-eval.log" % (variant, split_seed, seed))
            command = [
                str(PYTHON), str(EVAL_SCRIPT),
                "--run-dir", str(run_dir),
                "--validation-only",
                "--bootstrap", str(plan["evaluation"]["bootstrap_replicates"]),
                "--host-threads", str(plan["evaluation"]["host_threads_per_process"]),
                "--float-output", str(artifact_dir / "identity_embedding_float.tflite"),
                "--int8-output", str(artifact_dir / "identity_embedding_int8.tflite"),
            ]
            log = log_path.open("w")
            process = subprocess.Popen(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
            active.append({"process": process, "log": log, "log_path": log_path,
                           "variant": variant, "split_seed": split_seed, "seed": seed,
                           "run_dir": run_dir, "start": time.time()})
            print("START-EVAL variant=%s split_seed=%d seed=%d pid=%d log=%s" % (
                variant, split_seed, seed, process.pid, log_path,
            ), flush=True)
        while active:
            remaining = []
            for record in active:
                code = record["process"].poll()
                if code is None:
                    remaining.append(record)
                    continue
                record["log"].close()
                print("DONE-EVAL variant=%s split_seed=%d seed=%d exit=%d minutes=%.1f log=%s" % (
                    record["variant"], record["split_seed"], record["seed"], code,
                    (time.time() - record["start"]) / 60.0, record["log_path"],
                ), flush=True)
                if code:
                    failures.append({"variant": record["variant"], "split_seed": record["split_seed"],
                                     "seed": record["seed"], "exit_code": code,
                                     "log": str(record["log_path"]), "run_dir": str(record["run_dir"])})
                else:
                    completed.append({"variant": record["variant"],
                                      "split_seed": record["split_seed"],
                                      "seed": record["seed"],
                                      "run_dir": str(record["run_dir"])})
            active = remaining
            if active:
                time.sleep(5)
    summary = {"experiment_id": plan["experiment_id"], "planned_evaluations": len(jobs),
               "completed_evaluations": len(completed), "completed": completed,
               "planned": [{"variant": variant, "split_seed": split_seed, "seed": seed,
                            "run_dir": str(run_dir)}
                           for variant, split_seed, seed, run_dir in jobs],
               "failures": failures}
    status_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print("EVALUATION-MATRIX " + json.dumps(summary, sort_keys=True), flush=True)
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
