"""Recompute calibration-only risk/coverage curves for the 10-root head run.

Each root's operating threshold is selected from that root's known and unknown
calibration rows only. Development rows are then evaluated once per selected
calibration budget; no development-based threshold selection is performed.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
STUDY = ROOT / "work/project-research-v5"
sys.path.insert(0, str(ROOT / "FaceRecognition"))
from nicla_study.fs import sha256_file

RUNS = [STUDY / "runs" / ("camera_pretrained_embedding_head_v1_root_%d" % seed)
        for seed in range(56401, 56411)]
OUT = STUDY / "runs/camera_pretrained_embedding_head_v1_campaign"
CURVE_PATH = OUT / "OPERATING_CURVE.json"
CAPS = (0, 1, 2, 3)
KNOWN = ("K01", "K02", "K03", "K04", "K05")


def _usable(row: dict) -> bool:
    return row.get("usable") is True and isinstance(row.get("max_known_probability"), (int, float))


def _predicted_known(row: dict) -> bool:
    return row.get("predicted_label") in KNOWN


def _accept(row: dict, threshold: np.float32) -> bool:
    return (_usable(row) and _predicted_known(row) and
            np.float32(row["max_known_probability"]) >= threshold)


def _unknown_group_false(records: list[dict], threshold: np.float32, role: str) -> int:
    groups: dict[str, list[dict]] = {}
    for row in records:
        if row.get("role") != role:
            continue
        group = str(row.get("source_group") or row["capture_id"])
        groups.setdefault(group, []).append(row)
    return sum(any(_accept(row, threshold) for row in rows) for rows in groups.values())


def _calibration_counts(records: list[dict], threshold: np.float32) -> dict:
    known = [row for row in records if row.get("role") == "threshold_calibration" and row.get("label") in KNOWN]
    unknown = [row for row in records if row.get("role") == "unknown_calibration"]
    return {
        "known_rows": len(known),
        "known_valid_rows": sum(_usable(row) for row in known),
        "known_accepted_rows": int(sum(_accept(row, threshold) for row in known)),
        "known_correct_accepted_rows": int(sum(_accept(row, threshold) and
            row.get("predicted_label") == row.get("label") for row in known)),
        "known_wrong_accepted_rows": int(sum(_accept(row, threshold) and
            row.get("predicted_label") != row.get("label") for row in known)),
        "unknown_groups": len({str(row.get("source_group") or row["capture_id"]) for row in unknown}),
        "unknown_valid_rows": sum(_usable(row) for row in unknown),
        "unknown_false_groups": _unknown_group_false(records, threshold, "unknown_calibration"),
    }


def _select_threshold(records: list[dict], max_false_groups: int) -> tuple[np.float32, dict]:
    calibration = [row for row in records if row.get("role") in
                   ("threshold_calibration", "unknown_calibration") and _usable(row)]
    observed = {np.float32(row["max_known_probability"]) for row in calibration}
    candidates = {np.float32(0.0), np.float32(1.0)}
    for score in observed:
        candidates.add(score)
        candidates.add(np.nextafter(score, np.float32(np.inf), dtype=np.float32))
    eligible = []
    for threshold in candidates:
        counts = _calibration_counts(records, threshold)
        if counts["unknown_false_groups"] <= max_false_groups:
            key = (counts["known_correct_accepted_rows"],
                   -counts["unknown_false_groups"],
                   -counts["known_wrong_accepted_rows"],
                   -counts["known_accepted_rows"],
                   float(threshold))
            eligible.append((key, threshold, counts))
    if not eligible:
        raise ValueError("calibration had no finite threshold satisfying false-group budget")
    _key, selected, counts = max(eligible, key=lambda item: item[0])
    return np.float32(selected), counts


def _development(records: list[dict], threshold: np.float32) -> dict:
    known = [row for row in records if row.get("role") == "development" and row.get("label") in KNOWN]
    unknown = [row for row in records if row.get("role") == "unknown_development"]
    accepted = {row["capture_id"]: _accept(row, threshold) for row in known}
    per_person = {}
    for label in KNOWN:
        rows = [row for row in known if row["label"] == label]
        per_person[label] = {
            "rows": len(rows), "fomo_single_cluster_rows": sum(_usable(row) for row in rows),
            "six_way_top1_correct_all_rows": sum(_usable(row) and row.get("predicted_label") == label for row in rows),
            "correct_accepted": sum(accepted[row["capture_id"]] and row.get("predicted_label") == label for row in rows),
            "wrong_known_accepts": sum(accepted[row["capture_id"]] and row.get("predicted_label") != label for row in rows),
        }
    false_groups = _unknown_group_false(records, threshold, "unknown_development")
    false_rows = sum(_accept(row, threshold) for row in unknown)
    accepted_total = int(sum(accepted.values()))
    correct = int(sum(accepted[row["capture_id"]] and row.get("predicted_label") == row.get("label") for row in known))
    wrong = int(sum(accepted[row["capture_id"]] and row.get("predicted_label") != row.get("label") for row in known))
    return {
        "known_rows": len(known), "known_fomo_valid_rows": sum(_usable(row) for row in known),
        "known_raw_top1_all_rows": sum(_usable(row) and row.get("predicted_label") == row.get("label") for row in known) / max(1, len(known)),
        "known_correct_accepted": int(correct), "known_correct_acceptance_all_rows": correct / max(1, len(known)),
        "known_wrong_accepted": int(wrong), "known_coverage_all_rows": accepted_total / max(1, len(known)),
        "unknown_rows": len(unknown),
        "unknown_groups": len({str(row.get("source_group") or row["capture_id"]) for row in unknown}),
        "unknown_fomo_valid_rows": sum(_usable(row) for row in unknown),
        "unknown_false_accepted_rows": int(false_rows), "unknown_false_accepted_groups": int(false_groups),
        "unknown_zero_event_upper95_for_30": 1.0 - 0.05 ** (1.0 / 30.0),
        "per_person": per_person,
    }


def _json_scalar(value):
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError("unsupported JSON value: " + type(value).__name__)


def run() -> dict:
    if CURVE_PATH.exists():
        existing = json.loads(CURVE_PATH.read_text())
        for root in existing["roots"]:
            result_path = STUDY / "runs" / ("camera_pretrained_embedding_head_v1_root_%d" % root["root_seed"]) / "RESULT.json"
            predictions_path = STUDY / "private" / result_path.parent.name / "PREDICTIONS.jsonl"
            if (sha256_file(result_path) != root["result_sha256"] or
                    sha256_file(predictions_path) != root["prediction_sha256"]):
                raise ValueError("campaign parent changed after the curve was frozen")
        return existing
    roots = []
    for path in RUNS:
        result_path = path / "RESULT.json"
        predictions_path = STUDY / "private" / path.name / "PREDICTIONS.jsonl"
        result = json.loads(result_path.read_text())
        predictions = [json.loads(line) for line in predictions_path.read_text().splitlines() if line.strip()]
        if result.get("status") != "complete_single_gradient_pilot" or len(predictions) != 255:
            raise ValueError("missing complete root or 255 frozen predictions: " + path.name)
        if any(row.get("role") == "future_final" for row in predictions):
            raise ValueError("future-final role is prohibited")
        root = {"root_seed": result["training"]["seed"],
                "result_sha256": sha256_file(result_path),
                "prediction_sha256": sha256_file(predictions_path),
                "fit_rows": result["training"]["fit_rows"],
                "stopping_rows": result["training"]["stopping_rows"],
                "selected_epoch": result["training"]["selected_epoch"],
                "fit_stop_group_overlap": result["training"]["fit_stop_group_overlap"],
                "budgets": []}
        for cap in CAPS:
            threshold, calibration = _select_threshold(predictions, cap)
            root["budgets"].append({"max_unknown_calibration_false_groups": cap,
                "threshold_binary32": float(threshold), "calibration": calibration,
                "development": _development(predictions, threshold)})
        roots.append(root)
    aggregates = []
    for i, cap in enumerate(CAPS):
        metrics = [root["budgets"][i]["development"] for root in roots]
        fields = ("known_raw_top1_all_rows", "known_correct_acceptance_all_rows",
                  "known_coverage_all_rows", "known_wrong_accepted", "unknown_false_accepted_groups",
                  "unknown_false_accepted_rows")
        aggregate = {"max_unknown_calibration_false_groups": cap, "roots": len(roots), "metrics": {}}
        for field in fields:
            values = [metric[field] for metric in metrics]
            aggregate["metrics"][field] = {
                "mean": float(np.mean(values)), "min": float(np.min(values)), "max": float(np.max(values)),
            }
        aggregates.append(aggregate)
    OUT.mkdir(parents=True, exist_ok=True)
    report = {
        "schema": "v6-camera-pretrained-embedding-head-calibration-curves-1",
        "status": "complete_calibration_only_operating_curve",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "source_experiment": "camera_pretrained_embedding_head_v1_root_56401..56410",
        "input_parent": "work/project-research-v5/runs/camera_corrected_input_v2/INPUTS.json",
        "threshold_selection": "maximize correctly accepted known calibration rows subject to the listed unknown calibration false-group cap; ties minimize unknown false groups, wrong known accepts, total known accepts, then choose higher threshold",
        "precision": "all score comparisons use binary32 probabilities and thresholds; acceptance is >=",
        "denominators": {"known_development_rows": 50, "unknown_development_groups": 30,
                         "known_calibration_rows": 25, "unknown_calibration_groups": 30,
                         "known_development_fomo_single_cluster_rows": 35,
                         "unknown_development_fomo_single_cluster_rows": 20},
        "aggregate": aggregates, "roots": roots,
        "scope_limits": [
            "Ten roots vary initialization and balanced sampling only; they reuse one camera data set, one train/stopping split, one fixed 50-known/30-unknown development set, and one taped screen setup.",
            "The calibration cap is exploratory; it does not change the project 1% unknown-group target.",
            "Thirty development unknown identities cannot establish a 1% one-sided 95% upper risk (zero-event upper bound is 9.5%).",
            "The face-pretrained 112px MobileFaceNet encoder and this head remain host-only; they are not a deployable Nicla candidate.",
        ],
        "future_final_used": False, "ready_for_final_build": False,
    }
    CURVE_PATH.write_text(json.dumps(report, indent=2, sort_keys=True, allow_nan=False,
                                     default=_json_scalar) + "\n", encoding="utf-8")
    return report


if __name__ == "__main__":
    print(json.dumps(run(), indent=2, sort_keys=True, allow_nan=False, default=_json_scalar))
