#!/usr/bin/env python3
"""Resume validation-improving runs that reached the predeclared epoch cap."""

from __future__ import annotations

import argparse
import json
import subprocess
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
TRAINING_ROOT = ROOT / "work/model-training"
PYTHON = TRAINING_ROOT / ".venv/bin/python"
TRAIN_SCRIPT = TRAINING_ROOT / "scripts/train_embedder.py"


def candidate_for(run_dir, cap_epochs):
    summary_path = run_dir / "run_summary.json"
    config_path = run_dir / "train_config.json"
    history_path = run_dir / "validation_history.jsonl"
    if not (summary_path.is_file() and config_path.is_file() and history_path.is_file()):
        return None
    summary = json.loads(summary_path.read_text())
    config = json.loads(config_path.read_text())
    history = [json.loads(line) for line in history_path.read_text().splitlines() if line.strip()]
    completed = int(summary.get("epochs_completed", 0))
    if completed != cap_epochs or not history:
        return None
    if int(config.get("epochs_requested", cap_epochs)) != cap_epochs:
        return None
    best = max(history, key=lambda row: row["validation_verification"]["roc_auc"])
    epochs_since_best = completed - int(best["epoch"])
    if epochs_since_best >= int(summary.get("early_stopping_patience", 5)):
        return None
    has_previous_resume = float(summary.get("resume_previous_runtime_seconds", 0.0)) > 0
    if cap_epochs == 30 and has_previous_resume:
        return None
    if cap_epochs > 30 and not has_previous_resume:
        return None
    return summary, config, int(best["epoch"]), epochs_since_best


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign-dir", type=Path, action="append", required=True)
    parser.add_argument("--max-parallel", type=int, default=2)
    parser.add_argument("--cap-epochs", type=int, default=30)
    parser.add_argument("--target-epochs", type=int, default=40)
    args = parser.parse_args()
    if args.max_parallel < 1 or args.cap_epochs < 1 or args.target_epochs <= args.cap_epochs:
        parser.error("require max-parallel >= 1 and target-epochs > cap-epochs >= 1")
    candidates = []
    for raw_dir in args.campaign_dir:
        campaign_dir = raw_dir if raw_dir.is_absolute() else ROOT / raw_dir
        if not campaign_dir.is_dir():
            raise SystemExit("Campaign directory does not exist: " + str(campaign_dir))
        for summary_path in sorted(campaign_dir.rglob("run_summary.json")):
            item = candidate_for(summary_path.parent, args.cap_epochs)
            if item is not None:
                summary, config, best_epoch, wait = item
                candidates.append((summary_path.parent, summary, config, best_epoch, wait))

    failures = []
    active = {}
    pending = list(candidates)
    while pending or active:
        while pending and len(active) < args.max_parallel:
            run_dir, summary, config, best_epoch, wait = pending.pop(0)
            log_path = run_dir.parent.parent.parent / "logs" / (
                "%s-split-%d-seed-%d-resume%d.log" % (
                    summary["model_variant"], summary["split_seed"], summary["training_seed"],
                    args.target_epochs,
                )
            )
            log_path.parent.mkdir(parents=True, exist_ok=True)
            command = [
                str(PYTHON), str(TRAIN_SCRIPT),
                "--seed", str(summary["training_seed"]),
                "--split-seed", str(summary["split_seed"]),
                "--augmentation-seed", str(summary["augmentation_seed"]),
                "--objective", summary["objective"],
                "--model-variant", summary["model_variant"],
                "--epochs", str(args.target_epochs),
                "--patience", str(summary["early_stopping_patience"]),
                "--max-images-per-train-identity", str(config["max_images_per_train_identity"]),
                "--identities-per-batch", str(config["identities_per_batch"]),
                "--images-per-identity", str(config["images_per_identity"]),
                "--embedding-dim", str(config["embedding_dim"]),
                "--intra-threads", str(config["intra_threads"]),
                "--inter-threads", str(config["inter_threads"]),
                "--output", str(run_dir), "--resume",
            ]
            log = log_path.open("w")
            process = subprocess.Popen(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
            active[process] = {"run_dir": run_dir, "summary": summary, "best_epoch": best_epoch,
                               "wait": wait, "log": log, "log_path": log_path, "start": time.time()}
            print("RESUME variant=%s split=%d seed=%d cap=%d target=%d best_epoch=%d no_improvement=%d pid=%d log=%s" % (
                summary["model_variant"], summary["split_seed"], summary["training_seed"],
                args.cap_epochs, args.target_epochs, best_epoch, wait, process.pid, log_path,
            ), flush=True)
        completed = []
        for process, record in list(active.items()):
            code = process.poll()
            if code is None:
                continue
            record["log"].close()
            print("RESUME-DONE variant=%s split=%d seed=%d exit=%d minutes=%.1f log=%s" % (
                record["summary"]["model_variant"], record["summary"]["split_seed"],
                record["summary"]["training_seed"], code,
                (time.time() - record["start"]) / 60.0, record["log_path"],
            ), flush=True)
            if code:
                failures.append({"run_dir": str(record["run_dir"]), "exit_code": code,
                                 "log": str(record["log_path"])})
            completed.append(process)
        for process in completed:
            del active[process]
        if active:
            time.sleep(5)

    print(json.dumps({"cap_epochs": args.cap_epochs, "target_epochs": args.target_epochs,
                      "planned_extensions": len(candidates), "failures": failures}, sort_keys=True), flush=True)
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
