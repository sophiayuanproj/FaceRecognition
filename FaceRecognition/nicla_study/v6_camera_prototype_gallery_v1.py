"""Evaluate a group-balanced cosine gallery on corrected fused48 camera features.

This is a deterministic frozen-encoder baseline, not a gradient-training root.
Gallery construction uses train/other_train groups only; the threshold uses
unknown_calibration only; the development roles are evaluation-only. Raw
per-image scores and predictions remain private.
"""

from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
STUDY = ROOT / "work/project-research-v5"
INPUTS = STUDY / "runs/camera_corrected_input_v2"
FEATURES = STUDY / "private/camera_corrected_input_v2/features.npz"
OUT = STUDY / "runs/camera_prototype_gallery_v1"
PRIVATE = STUDY / "private/camera_prototype_gallery_v1"
KNOWN = ("K01", "K02", "K03", "K04", "K05")
CLASSES = KNOWN + ("OTHER",)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def unit(vector: np.ndarray) -> np.ndarray:
    value = np.asarray(vector, dtype=np.float32)
    norm = float(np.linalg.norm(value))
    if not np.isfinite(value).all() or not np.isfinite(norm) or norm <= 1e-8:
        raise ValueError("prototype or feature has invalid norm")
    return value / np.float32(norm)


def next_f32(value: float) -> float:
    return float(np.nextafter(np.float32(value), np.float32(np.inf), dtype=np.float32))


def threshold_for_unknown_group_cap(scores: list[float], groups: list[str], cap: int) -> float:
    by_group: dict[str, float] = {}
    for group, score in zip(groups, scores):
        by_group[group] = max(by_group.get(group, -np.inf), float(np.float32(score)))
    ordered = sorted(by_group.values(), reverse=True)
    if not ordered:
        raise ValueError("no usable unknown-calibration groups")
    if cap < 0 or cap >= len(ordered):
        raise ValueError("unknown false-group cap must be within observed groups")
    # To permit at most cap groups at or above threshold, step above rank cap+1.
    return next_f32(ordered[cap])


def evaluate(rows: list[dict], features: np.ndarray, scores: np.ndarray,
             predicted: np.ndarray, valid: np.ndarray, threshold: float) -> dict:
    decisions = []
    accepted_known = np.zeros(len(rows), dtype=bool)
    for i in range(len(rows)):
        if not valid[i]:
            decision = "NO_FACE"
        elif int(predicted[i]) == 5:
            decision = "OTHER"
        elif float(np.float32(scores[i, :5].max())) >= float(np.float32(threshold)):
            decision = CLASSES[int(predicted[i])]
            accepted_known[i] = True
        else:
            decision = "UNCERTAIN"
        decisions.append(decision)

    known_ids = [i for i, row in enumerate(rows)
                 if row["role"] == "development" and row["label"] in KNOWN]
    unknown_ids = [i for i, row in enumerate(rows) if row["role"] == "unknown_development"]
    unknown_groups: dict[str, list[int]] = defaultdict(list)
    for i in unknown_ids:
        unknown_groups[str(rows[i].get("source_group") or rows[i]["capture_id"])].append(i)
    false_groups = sum(any(accepted_known[i] for i in group_rows)
                       for group_rows in unknown_groups.values())
    per_person = {}
    for label in KNOWN:
        ids = [i for i in known_ids if rows[i]["label"] == label]
        per_person[label] = {
            "rows": len(ids),
            "single_cluster_rows": sum(bool(valid[i]) for i in ids),
            "raw_six_way_top1_correct": sum(bool(valid[i]) and CLASSES[int(predicted[i])] == label for i in ids),
            "correct_accepted": int(sum(bool(accepted_known[i]) and CLASSES[int(predicted[i])] == label for i in ids)),
            "wrong_known_accepted": int(sum(bool(accepted_known[i]) and CLASSES[int(predicted[i])] != label for i in ids)),
        }
    return {
        "threshold_binary32": float(np.float32(threshold)),
        "known_development_rows": len(known_ids),
        "known_single_cluster_rows": sum(bool(valid[i]) for i in known_ids),
        "known_raw_six_way_top1_correct": sum(bool(valid[i]) and CLASSES[int(predicted[i])] == rows[i]["label"] for i in known_ids),
        "known_correct_accepted": int(sum(bool(accepted_known[i]) and CLASSES[int(predicted[i])] == rows[i]["label"] for i in known_ids)),
        "known_wrong_accepted": int(sum(bool(accepted_known[i]) and CLASSES[int(predicted[i])] != rows[i]["label"] for i in known_ids)),
        "known_coverage_all_rows": float(sum(bool(accepted_known[i]) for i in known_ids) / max(1, len(known_ids))),
        "unknown_development_rows": len(unknown_ids),
        "unknown_development_groups": len(unknown_groups),
        "unknown_single_cluster_rows": sum(bool(valid[i]) for i in unknown_ids),
        "unknown_false_accepted_groups": int(false_groups),
        "unknown_zero_event_upper95_for_30": 0.09503385285530419 if len(unknown_groups) == 30 and false_groups == 0 else None,
        "per_person": per_person,
        "private_decisions": decisions,
    }


def run() -> dict:
    if OUT.exists() or PRIVATE.exists():
        raise FileExistsError("preserving existing frozen gallery outputs")
    index_path = INPUTS / "CAPTURE_INDEX.jsonl"
    input_path = INPUTS / "INPUTS.json"
    rows = [json.loads(line) for line in index_path.read_text(encoding="utf-8").splitlines()]
    with np.load(FEATURES) as archive:
        features = np.asarray(archive["features"], dtype=np.float32)
    if features.shape != (len(rows), 128):
        raise ValueError("feature/index shape mismatch")
    valid = np.asarray([row["usable_for_identity"] for row in rows], dtype=bool)
    for i, row in enumerate(rows):
        norm = float(np.linalg.norm(features[i]))
        if valid[i] and (not np.isfinite(features[i]).all() or abs(norm - 1.0) > 1e-4):
            raise ValueError("usable embedding is not finite and unit normalized")
        if not valid[i] and np.any(features[i]):
            raise ValueError("invalid/missed rows must stay zero-filled")

    grouped: dict[str, dict[str, list[np.ndarray]]] = {label: defaultdict(list) for label in CLASSES}
    for i, row in enumerate(rows):
        if not valid[i] or row["role"] not in ("train", "other_train"):
            continue
        label = row["label"]
        if label not in CLASSES or (row["role"] == "train" and label not in KNOWN) or \
                (row["role"] == "other_train" and label != "OTHER"):
            raise ValueError("training role/label mismatch")
        group = str(row.get("source_group") or row["capture_id"])
        grouped[label][group].append(features[i])

    prototypes = []
    group_counts = {}
    for label in CLASSES:
        group_vectors = [unit(np.mean(np.stack(vectors), axis=0))
                         for _, vectors in sorted(grouped[label].items())]
        if not group_vectors:
            raise ValueError("missing train-only source groups for " + label)
        prototypes.append(unit(np.mean(np.stack(group_vectors), axis=0)))
        group_counts[label] = len(group_vectors)
    prototypes = np.asarray(prototypes, dtype=np.float32)
    scores = np.zeros((len(rows), len(CLASSES)), dtype=np.float32)
    valid_ids = np.flatnonzero(valid)
    scores[valid_ids] = features[valid_ids] @ prototypes.T
    predicted = np.argmax(scores, axis=1)

    calibration_ids = [i for i, row in enumerate(rows)
                       if row["role"] == "unknown_calibration" and valid[i]]
    calibration_scores = [float(np.max(scores[i, :5])) for i in calibration_ids]
    calibration_groups = [str(rows[i].get("source_group") or rows[i]["capture_id"])
                          for i in calibration_ids]
    if len(set(calibration_groups)) < 4:
        raise ValueError("at least four usable unknown calibration groups are required for caps 0-3")
    curves = {}
    for cap in (0, 1, 2, 3):
        threshold = threshold_for_unknown_group_cap(calibration_scores, calibration_groups, cap)
        result = evaluate(rows, features, scores, predicted, valid, threshold)
        result.pop("private_decisions")
        result["unknown_calibration_false_accepted_groups"] = sum(
            any(calibration_scores[j] >= threshold for j, group in enumerate(calibration_groups)
                if group == group_id) for group_id in set(calibration_groups))
        curves[str(cap)] = result

    selected = curves["0"]
    detail = []
    for i, row in enumerate(rows):
        detail.append({"capture_id": row["capture_id"], "role": row["role"],
                       "label": row["label"], "source_group": row.get("source_group"),
                       "quality_status": row["quality_status"], "valid": bool(valid[i]),
                       "scores": scores[i].tolist() if valid[i] else None,
                       "predicted_class": CLASSES[int(predicted[i])] if valid[i] else None,
                       "decision": selected.get("private_decisions", [])[i]
                       if selected.get("private_decisions") else None})
    # Evaluate's private row decisions are intentionally stripped above; rebuild
    # only inside this private output file for the selected calibration point.
    threshold = selected["threshold_binary32"]
    for i, row in enumerate(rows):
        if not valid[i]:
            detail[i]["decision"] = "NO_FACE"
        elif int(predicted[i]) == 5:
            detail[i]["decision"] = "OTHER"
        elif float(np.float32(scores[i, :5].max())) >= threshold:
            detail[i]["decision"] = CLASSES[int(predicted[i])]
        else:
            detail[i]["decision"] = "UNCERTAIN"

    PRIVATE.mkdir(parents=True, exist_ok=False)
    OUT.mkdir(parents=True, exist_ok=False)
    prototype_path = PRIVATE / "prototypes.npz"
    np.savez_compressed(prototype_path, prototypes=prototypes)
    prediction_path = PRIVATE / "PREDICTIONS.jsonl"
    prediction_path.write_text("".join(json.dumps(row, sort_keys=True, allow_nan=False) + "\n"
                                               for row in detail), encoding="utf-8")
    result = {
        "schema": "v6-camera-frozen-cosine-gallery-v1",
        "status": "complete_deterministic_frozen_gallery_baseline",
        "model": {"encoder_sha256": json.loads(input_path.read_text())["model_sha256"],
                  "feature_dim": 128, "class_order": list(CLASSES),
                  "classifier": "source-group-balanced normalized cosine prototypes; no gradient training",
                  "training_roles": ["train", "other_train"],
                  "threshold_role": "unknown_calibration, grouped by source identity",
                  "threshold_rule": "max-known cosine after binary32 cast; threshold is next representable binary32 above the rank-(cap+1) unknown calibration group maximum; >= accepts",
                  "prototype_sha256": sha256(prototype_path)},
        "parent_inputs": {"input_index_sha256": sha256(index_path),
                          "features_sha256": sha256(FEATURES),
                          "input_contract_sha256": sha256(input_path),
                          "roles_sha256": sha256(STUDY / "data/manifests/optical_adaptation_parents.jsonl"),
                          "future_final_used": False},
        "training_group_counts": group_counts,
        "calibration": {"unknown_source_groups_total": len({str(row.get("source_group") or row["capture_id"])
                                                              for row in rows if row["role"] == "unknown_calibration"}),
                        "unknown_groups_usable": len(set(calibration_groups)),
                        "usable_rows": len(calibration_ids), "false_group_caps": [0, 1, 2, 3]},
        "development_operating_curve": curves,
        "private_predictions": str(prediction_path.relative_to(ROOT)),
        "limits": ["Single deterministic frozen encoder/gallery, no gradient-training roots are claimed.",
                   "FOMO misses and multiple clusters remain in the all-row known denominator.",
                   "Thirty unknown people cannot certify the one-sided 1% risk target.",
                   "Prototype scores are host evaluated; the actual board one-shot used the separate v2 linear head.",
                   "The fused48 encoder is device runnable; this gallery baseline is not yet in the standalone app."],
        "ready_for_final_build": False,
    }
    (OUT / "RESULT.json").write_text(json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n",
                                      encoding="utf-8")
    return result


if __name__ == "__main__":
    print(json.dumps(run(), indent=2, sort_keys=True, allow_nan=False))
