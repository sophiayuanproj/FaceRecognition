#!/usr/bin/env python3
"""Run a predeclared local training matrix with bounded process parallelism."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
TRAINING_ROOT = ROOT / "work/model-training"
TRAIN_SCRIPT = TRAINING_ROOT / "scripts/train_embedder.py"
PYTHON = TRAINING_ROOT / ".venv/bin/python"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--max-parallel", type=int, default=None)
    args = parser.parse_args()
    plan_path = args.plan if args.plan.is_absolute() else ROOT / args.plan
    plan = json.loads(plan_path.read_text())
    cap = args.max_parallel or plan.get("parallel_training_jobs") or plan.get("evaluation", {}).get("parallel_training_jobs", 1)
    if cap < 1:
        parser.error("max parallel jobs must be at least 1")
    logs_root = TRAINING_ROOT / "runs" / plan["experiment_id"] / "logs"
    logs_root.mkdir(parents=True, exist_ok=True)
    base_root = TRAINING_ROOT / "runs" / plan["experiment_id"]
    active = {}
    failures = []
    jobs = []
    groups = plan.get("job_groups") or [{
        "model_variant": plan["model_variant"],
        "split_seed": plan["split_seed"],
        "training_seeds": plan["training_seeds"],
    }]
    for group in groups:
        for seed in group["training_seeds"]:
            variant = group["model_variant"]
            split_seed = group["split_seed"]
            output = base_root / variant / ("split-%d" % split_seed) / ("seed-%d" % seed)
            jobs.append({
                "variant": variant,
                "split_seed": split_seed,
                "seed": seed,
                "augmentation_seed": seed,
                "output": output,
            })
    pending = list(jobs)
    while pending or active:
        while pending and len(active) < cap:
            job = pending.pop(0)
            output = job["output"]
            output.parent.mkdir(parents=True, exist_ok=True)
            if output.exists() and any(output.iterdir()):
                raise SystemExit("Refusing to overwrite non-empty run directory; inspect or resume manually: " + str(output))
            log_path = logs_root / ("%s-split-%d-seed-%d.log" % (
                job["variant"], job["split_seed"], job["seed"],
            ))
            command = [
                str(PYTHON), str(TRAIN_SCRIPT),
                "--seed", str(job["seed"]),
                "--split-seed", str(job["split_seed"]),
                "--augmentation-seed", str(job["augmentation_seed"]),
                "--objective", "supcon",
                "--model-variant", job["variant"],
                "--epochs", str(plan["epochs_max"]),
                "--patience", str(plan["early_stopping_patience"]),
                "--max-images-per-train-identity", "24",
                "--identities-per-batch", "8",
                "--images-per-identity", "4",
                "--embedding-dim", "64",
                "--intra-threads", str(plan["per_job_host_threads"]["intra_op"]),
                "--inter-threads", str(plan["per_job_host_threads"]["inter_op"]),
                "--output", str(output),
            ]
            log = log_path.open("w")
            process = subprocess.Popen(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
            active[process] = {"job": job, "log": log, "log_path": log_path, "start": time.time()}
            print("START variant=%s split_seed=%d seed=%d pid=%d log=%s" % (
                job["variant"], job["split_seed"], job["seed"], process.pid, log_path,
            ), flush=True)
        completed = []
        for process, record in list(active.items()):
            code = process.poll()
            if code is None:
                continue
            record["log"].close()
            elapsed = time.time() - record["start"]
            job = record["job"]
            print("DONE variant=%s split_seed=%d seed=%d exit=%d minutes=%.1f log=%s" % (
                job["variant"], job["split_seed"], job["seed"], code,
                elapsed / 60.0, record["log_path"],
            ), flush=True)
            if code:
                failures.append({"variant": job["variant"], "split_seed": job["split_seed"],
                                 "seed": job["seed"], "exit_code": code,
                                 "log": str(record["log_path"]), "output": str(job["output"])})
            completed.append(process)
        for process in completed:
            del active[process]
        if active:
            time.sleep(5)
    summary = {
        "experiment_id": plan["experiment_id"],
        "planned_jobs": len(jobs),
        "completed_jobs": len(jobs) - len(failures),
        "failures": failures,
    }
    summary_path = base_root / "training_matrix_status.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print("MATRIX " + json.dumps(summary, sort_keys=True), flush=True)
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
