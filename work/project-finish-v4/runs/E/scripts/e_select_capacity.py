#!/usr/bin/env python3
"""Apply the frozen development-only capacity and teacher gates."""
from __future__ import annotations

import json
import math
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[5]
STUDY = ROOT / "work/project-finish-v4"
RUN_ROOT = STUDY / "runs/E"
sys.path.insert(0, str(ROOT / "FaceRecognition"))

import numpy as np
from nicla_study.state import resource_lease
from e_pilot import digest_file, write_json

SEEDS = [20261101 + i for i in range(10)]


def mean_ci(values):
    values = np.asarray(values, dtype=np.float64)
    mean = float(np.mean(values))
    sd = float(np.std(values, ddof=1)) if len(values) > 1 else 0.0
    half = 2.262 * sd / math.sqrt(len(values)) if len(values) > 1 else None
    return {"n": int(len(values)), "mean": mean, "sd_root": sd,
            "ci95_t_low": mean - half if half is not None else None,
            "ci95_t_high": mean + half if half is not None else None}


def main():
    plan = json.loads((STUDY / "reports/E/PREDECLARATION.json").read_text(encoding="utf-8"))
    registry = {"schema": "job-e-development-capacity-selection-1", "status": "in_progress",
                "rule": plan["capacity_selection_rule"],
                "predeclaration_sha256": digest_file(STUDY / "reports/E/PREDECLARATION.json"),
                "final_role_accessed": False, "candidates": {}}
    for alpha in (0.10, 0.25, 0.50, 0.75):
        metrics, exports = [], []
        arm = "capacity_a%03d" % int(alpha * 100)
        for seed in SEEDS:
            root = RUN_ROOT / "roots" / arm / ("seed-%d" % seed)
            path = root / "EXPORT_RESULT.json"
            if not path.is_file():
                continue
            result = json.loads(path.read_text(encoding="utf-8"))
            export = result.get("exports", {}).get("full_int8_ptq_train_only_rep", {})
            if export.get("status") != "complete_verified" or export.get("host_inference_status") != "pass":
                continue
            dev = export["development"]
            metrics.append({"seed": seed, "utility": dev["development_macro_utility"],
                            "unknown_fpir": dev["unknown_false_identification_rate"],
                            "known_correct_acceptance": dev["known_identity_macro_correct_acceptance"],
                            "known_identification": dev["known_identity_macro_identification"],
                            "bytes": export["bytes"], "parameters": export["parameter_count"],
                            "threshold": export["calibration"]["threshold"]})
            exports.append(export)
        if metrics:
            candidate = {"arm_id": arm, "alpha": alpha, "successful_full_int8_roots": len(metrics),
                         "metrics_by_root": metrics, "means_and_root_uncertainty": {
                             key: mean_ci([row[key] for row in metrics])
                             for key in ("utility", "unknown_fpir", "known_correct_acceptance",
                                         "known_identification", "bytes", "parameters")},
                         "all_roots_full_int8_host_inferred": len(metrics) == len(SEEDS)}
        else:
            candidate = {"arm_id": arm, "alpha": alpha, "successful_full_int8_roots": 0,
                         "metrics_by_root": [], "all_roots_full_int8_host_inferred": False}
        registry["candidates"][str(alpha)] = candidate

    small = registry["candidates"]["0.1"]
    ordinary = [registry["candidates"][str(a)] for a in (0.25, 0.5, 0.75)]
    eligible = [candidate for candidate in ordinary
                if candidate["successful_full_int8_roots"] == len(SEEDS)
                and candidate["means_and_root_uncertainty"]["unknown_fpir"]["mean"] <= 0.05
                and candidate["means_and_root_uncertainty"]["known_correct_acceptance"]["mean"] >
                    small["means_and_root_uncertainty"]["known_correct_acceptance"]["mean"]]
    if eligible:
        chosen = max(eligible, key=lambda candidate: (
            candidate["means_and_root_uncertainty"]["utility"]["mean"],
            -candidate["means_and_root_uncertainty"]["parameters"]["mean"]))
        gate_met = True
    else:
        convertible = [candidate for candidate in ordinary if candidate["successful_full_int8_roots"] > 0]
        chosen = max(convertible, key=lambda candidate: candidate["alpha"]) if convertible else None
        gate_met = False
    registry["selected_large_family"] = None if chosen is None else chosen["arm_id"]
    registry["selected_large_alpha"] = None if chosen is None else chosen["alpha"]
    registry["quality_gate_met"] = gate_met
    teacher = registry["candidates"]["0.75"]
    registry["teacher_usefulness_gate"] = {
        "teacher_alpha": 0.75,
        "teacher_full_int8_host_inferred_roots": teacher["successful_full_int8_roots"],
        "teacher_mean_utility_exceeds_small_a010": bool(
            teacher.get("successful_full_int8_roots") == len(SEEDS)
            and teacher["means_and_root_uncertainty"]["utility"]["mean"] >
                small["means_and_root_uncertainty"]["utility"]["mean"]),
        "teacher_mean_unknown_fpir_at_most_0_05": bool(
            teacher.get("successful_full_int8_roots") == len(SEEDS)
            and teacher["means_and_root_uncertainty"]["unknown_fpir"]["mean"] <= 0.05),
        "distillation_claim_eligible": bool(
            teacher.get("successful_full_int8_roots") == len(SEEDS)
            and teacher["means_and_root_uncertainty"]["utility"]["mean"] >
                small["means_and_root_uncertainty"]["utility"]["mean"]
            and teacher["means_and_root_uncertainty"]["unknown_fpir"]["mean"] <= 0.05)}
    registry["status"] = "complete_verified" if all(
        row["successful_full_int8_roots"] == len(SEEDS) for row in registry["candidates"].values()) else "partial"
    registry["finished_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    path = STUDY / "reports/E/CAPACITY_SELECTION.json"
    write_json(path, registry)
    print(json.dumps({"event": "capacity_selected", "status": registry["status"],
                      "alpha": registry["selected_large_alpha"], "quality_gate_met": gate_met,
                      "teacher_usefulness_gate": registry["teacher_usefulness_gate"],
                      "report": str(path)}), flush=True)


if __name__ == "__main__":
    main()
