"""Recompute task-classifier metrics with cached-detector misses in denominator."""

import hashlib
import json
import math
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
STUDY = ROOT / "work/project-research-v5"
RUNS = STUDY / "runs/task_classifier_48"


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def dump(path, value):
    Path(path).write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def upper95_zero(n):
    return 1.0 - 0.05 ** (1.0 / n) if n else None


def main():
    index_path = STUDY / "runs/host_reference/nonfinal_index.json"
    detection_path = STUDY / "runs/host_reference/nonfinal_features.npz"
    rows = json.loads(index_path.read_text())["rows"]
    with np.load(detection_path) as data:
        detected = np.asarray(data["detected"], dtype=bool)
    if len(rows) != len(detected):
        raise ValueError("frozen row index and detector vector differ in length")
    id_to_index = {}
    for i, row in enumerate(rows):
        rid = row["record_id"]
        if rid in id_to_index:
            raise ValueError("duplicate frozen record id: " + rid)
        id_to_index[rid] = i

    summary = {
        "schema": "v5-task-classifier-e2e-recompute-1",
        "source_scope": "LFW-v5 source images; cached host detector, MCU-matched48px INT8 encoder, and float32 774-parameter head",
        "models": [],
        "input_scope": "task-classifier predictions are recorded only when the cached source detector fired; all misses are added as NO_FACE failures",
        "unknown_unit": "source identity group; 30 development identities, not 300 independent people",
        "zero_false_accept_upper95_for_30_groups": upper95_zero(30),
        "future_final_used": False,
        "board_qualified": False,
        "readiness": "diagnostic only; no exact MCU identity path or actual-camera recognition claim",
        "analysis_code_sha256": sha256(Path(__file__)),
        "index_sha256": sha256(index_path),
        "detection_cache_sha256": sha256(detection_path),
        "classifier_protocol_sha256": sha256(RUNS / "PROTOCOL.json"),
    }

    for seed in range(53001, 53011):
        for backend in ("int8",):
            folder = RUNS / str(seed) / backend
            original = json.loads((folder / "RESULT.json").read_text())
            prediction_path = folder / "PREDICTIONS.jsonl"
            predictions = {}
            for line in prediction_path.read_text().splitlines():
                row = json.loads(line)
                rid = row["record_id"]
                if rid in predictions:
                    raise ValueError("duplicate classifier prediction: " + rid)
                p = row["probabilities"]
                if len(p) != 6 or not all(math.isfinite(float(x)) for x in p):
                    raise ValueError("invalid probability vector: " + rid)
                if abs(sum(p) - 1.0) > 1e-5:
                    raise ValueError("probability vector is not normalized: " + rid)
                if rid not in id_to_index:
                    raise ValueError("prediction absent from frozen source index: " + rid)
                if not detected[id_to_index[rid]]:
                    raise ValueError("classifier prediction exists for a cached detector miss: " + rid)
                predictions[rid] = row

            eval_ids = [
                i for i, row in enumerate(rows)
                if row["role"] in ("threshold_calibration", "unknown_calibration", "development", "unknown_development")
            ]
            expected_detected = {rows[i]["record_id"] for i in eval_ids if detected[i]}
            if set(predictions) != expected_detected:
                raise ValueError("prediction rows do not close over detected calibration/development rows")

            unknown_cal_ids = [
                row["record_id"] for row in rows
                if row["role"] == "unknown_calibration" and detected[id_to_index[row["record_id"]]]
            ]
            unknown_cal_scores = [
                max(predictions[rid]["probabilities"][:5]) for rid in unknown_cal_ids
            ]
            if not unknown_cal_scores:
                raise ValueError("unknown calibration has no detector-positive samples")
            threshold = float(np.nextafter(np.float64(max(unknown_cal_scores)), np.float64(np.inf)))
            if not math.isclose(threshold, float(original["threshold"]), rel_tol=0.0, abs_tol=1e-15):
                raise ValueError("raw calibration predictions do not reproduce stored threshold")

            samples = []
            for i in eval_ids:
                src = rows[i]
                pred = predictions.get(src["record_id"])
                if pred is None:
                    samples.append({
                        "record_id": src["record_id"],
                        "source_group": src.get("source_group"),
                        "source_identity": src.get("source_identity"),
                        "role": src["role"],
                        "true_label": src["label"],
                        "detected": False,
                        "predicted_label": "NO_FACE",
                        "accepted": False,
                    })
                else:
                    p = np.asarray(pred["probabilities"], dtype=np.float64)
                    cls = int(np.argmax(p))
                    accepted = cls < 5 and float(p[:5].max()) >= threshold
                    samples.append({
                        "record_id": src["record_id"],
                        "source_group": src.get("source_group"),
                        "source_identity": src.get("source_identity"),
                        "role": src["role"],
                        "true_label": src["label"],
                        "detected": True,
                        "predicted_label": "OTHER" if cls == 5 else "K%02d" % (cls + 1),
                        "accepted": bool(accepted),
                        "known_score": float(p[:5].max()),
                    })

            def group_risk(subset):
                groups = {}
                for row in subset:
                    group = row.get("source_identity") or row.get("source_group") or row["record_id"]
                    groups[group] = groups.get(group, False) or bool(row["accepted"])
                n = len(groups)
                false = sum(groups.values())
                return {
                    "n_independent_identity_groups": n,
                    "false_accept_identity_groups": int(false),
                    "zero_event_upper95": upper95_zero(n) if false == 0 else None,
                }

            def get_metrics(role, known):
                subset = [row for row in samples if row["role"] == role]
                hits = [row for row in subset if row["detected"]]
                if known:
                    correct = lambda row: row["predicted_label"] == row["true_label"]
                    return {
                        "total": len(subset),
                        "detector_hits": len(hits),
                        "detector_misses": len(subset) - len(hits),
                        "conditional_raw_top1": sum(correct(row) for row in hits) / len(hits) if hits else None,
                        "end_to_end_raw_top1": sum(correct(row) for row in hits) / len(subset) if subset else None,
                        "correct_accepted": sum(row["accepted"] and correct(row) for row in subset),
                        "wrong_known_accepted": sum(row["accepted"] and not correct(row) for row in subset),
                        "correct_acceptance_over_all": sum(row["accepted"] and correct(row) for row in subset) / len(subset) if subset else None,
                        "coverage_over_all": sum(row["accepted"] for row in subset) / len(subset) if subset else None,
                        "per_person": [
                            {
                                "label": "K%02d" % k,
                                "n": sum(row["true_label"] == "K%02d" % k for row in subset),
                                "detector_hits": sum(row["true_label"] == "K%02d" % k and row["detected"] for row in subset),
                                "correct_accepted": sum(row["true_label"] == "K%02d" % k and row["accepted"] and correct(row) for row in subset),
                            }
                            for k in range(1, 6)
                        ],
                    }
                return {
                    "total": len(subset),
                    "detector_hits": len(hits),
                    "detector_misses": len(subset) - len(hits),
                    "false_accepts": sum(row["accepted"] for row in subset),
                    "identity_group_risk": group_risk(subset),
                    "other_argmax": sum(row["detected"] and row["predicted_label"] == "OTHER" for row in subset),
                }

            record = {
                "seed": seed,
                "encoder_backend": backend,
                "task_head": "float32 host linear softmax; not included in the INT8 model",
                "threshold": threshold,
                "known_development": get_metrics("development", True),
                "unknown_development": get_metrics("unknown_development", False),
                "known_calibration": get_metrics("threshold_calibration", True),
                "unknown_calibration": get_metrics("unknown_calibration", False),
                "future_final_used": False,
                "board_qualified": False,
                "source_conditional_result_sha256": sha256(folder / "RESULT.json"),
                "prediction_rows_sha256": sha256(prediction_path),
            }
            dump(folder / "E2E_RESULT.json", record)
            summary["models"].append(record)

    by_backend = {}
    for backend in ("int8",):
        subset = [row for row in summary["models"] if row["encoder_backend"] == backend]
        vals = [row["known_development"]["correct_acceptance_over_all"] for row in subset]
        coverage = [row["known_development"]["coverage_over_all"] for row in subset]
        raw = [row["known_development"]["end_to_end_raw_top1"] for row in subset]
        fa_groups = [row["unknown_development"]["identity_group_risk"]["false_accept_identity_groups"] for row in subset]
        by_backend[backend] = {
            "gradient_roots": len(subset),
            "known_correct_acceptance_mean_over_all": float(np.mean(vals)),
            "known_correct_acceptance_min_max": [float(np.min(vals)), float(np.max(vals))],
            "known_coverage_mean_over_all": float(np.mean(coverage)),
            "known_coverage_min_max": [float(np.min(coverage)), float(np.max(coverage))],
            "end_to_end_raw_top1_mean": float(np.mean(raw)),
            "unknown_false_identity_groups_per_root": fa_groups,
            "unknown_group_count_per_root": 30,
            "scope": "host cached detector + float head; source-image experiment only",
        }
    summary["aggregate_by_encoder_backend"] = by_backend
    dump(RUNS / "E2E_RESULTS.json", summary)
    print(json.dumps(by_backend, allow_nan=False, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
