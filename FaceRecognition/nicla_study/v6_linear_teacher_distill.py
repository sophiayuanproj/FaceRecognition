"""One corrected-camera pilot mapping the compact encoder toward SFace.

The compact 48px encoder stays frozen. A regularized 128x128 output map is
fit from train-role camera crops to YuNet-aligned SFace teacher embeddings.
This tests whether the existing compact representation can absorb alignment
knowledge before any larger training campaign. The teacher and embeddings
remain host-only/private; the pilot is not a deployable release.
"""

from __future__ import annotations

import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "FaceRecognition/firmware"))
from open_set_decision import accepts, threshold_above_max


STUDY = ROOT / "work/project-research-v5"
INPUTS = STUDY / "runs/camera_corrected_input_v1"
PRIVATE_INPUT = STUDY / "private/camera_corrected_input_v1"
PRIVATE_TEACHER = STUDY / "private/camera_corrected_teacher_v1"
OUT = STUDY / "runs/camera_corrected_teacher_student_v1"
PRIVATE_OUT = STUDY / "private/camera_corrected_teacher_student_v1"
RESULT = OUT / "RESULT.json"
PREDICTIONS = PRIVATE_OUT / "PREDICTIONS.jsonl"
FEATURE_FIELD = 1  # YuNet-aligned SFace teacher, gated by one FOMO cluster.
ALPHAS = (0.01, 0.1, 1.0, 10.0)
SPLIT_SEED = 56002
KNOWN = ("K01", "K02", "K03", "K04", "K05")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def norm_rows(values: np.ndarray) -> np.ndarray:
    array = np.asarray(values, dtype=np.float32)
    lengths = np.linalg.norm(array, axis=1, keepdims=True)
    return array / np.maximum(lengths, 1e-8)


def fit_ridge(x: np.ndarray, y: np.ndarray, alpha: float) -> tuple[np.ndarray, np.ndarray]:
    x_mean = np.mean(x, axis=0)
    y_mean = np.mean(y, axis=0)
    xc = x - x_mean
    yc = y - y_mean
    gram = xc @ xc.T
    weights = xc.T @ np.linalg.solve(gram + float(alpha) * np.eye(len(xc), dtype=np.float32), yc)
    bias = y_mean - x_mean @ weights
    return weights.astype(np.float32), bias.astype(np.float32)


def cosine_pair_mse(prediction: np.ndarray, target: np.ndarray) -> float:
    p = norm_rows(prediction)
    t = norm_rows(target)
    mask = ~np.eye(len(p), dtype=bool)
    return float(np.mean((p @ p.T - t @ t.T)[mask] ** 2))


def make_group_holdout(rows: list[dict], eligible: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(SPLIT_SEED)
    fit, holdout = [], []
    for label in KNOWN:
        ids = [i for i, row in enumerate(rows) if eligible[i] and row["label"] == label]
        if len(ids) < 3:
            raise ValueError("not enough matched train-role groups for internal stop: " + label)
        rng.shuffle(ids)
        holdout.append(ids[0])
        fit.extend(ids[1:])
    other_ids = [i for i, row in enumerate(rows) if eligible[i] and row["role"] == "other_train"]
    if len(other_ids) < 10:
        raise ValueError("teacher distillation needs at least ten distinct OTHER train identities")
    rng.shuffle(other_ids)
    holdout.extend(other_ids[:3])
    fit.extend(other_ids[3:])
    return np.asarray(fit, dtype=np.int64), np.asarray(holdout, dtype=np.int64)


def make_gallery(records: list[dict], embeddings: np.ndarray) -> dict[str, np.ndarray]:
    gallery = {}
    for label in KNOWN:
        group_values = {}
        for index, row in enumerate(records):
            if row["role"] != "train" or row["label"] != label:
                continue
            group = row.get("source_group") or row["capture_id"]
            group_values.setdefault(group, []).append(embeddings[index])
        prototypes = [norm_rows(np.mean(np.stack(values), axis=0)[None])[0]
                      for values in group_values.values()]
        if not prototypes:
            raise ValueError("student route has no train-role gallery rows for " + label)
        gallery[label] = norm_rows(np.mean(np.stack(prototypes), axis=0)[None])[0]
    return gallery


def evaluate(records: list[dict], embeddings: np.ndarray, valid: np.ndarray) -> dict:
    train_records = records
    gallery = make_gallery(train_records, embeddings)
    scores = np.zeros((len(records), len(KNOWN)), dtype=np.float32)
    for i in np.flatnonzero(valid):
        vector = embeddings[i] / max(float(np.linalg.norm(embeddings[i])), 1e-8)
        scores[i] = np.asarray([float(np.dot(vector, gallery[label])) for label in KNOWN], dtype=np.float32)
    max_score = scores.max(axis=1)
    unknown_cal = [i for i, row in enumerate(records)
                   if row["role"] == "unknown_calibration" and valid[i]]
    if not unknown_cal:
        raise ValueError("no corrected unknown calibration feature rows")
    threshold = threshold_above_max(max_score[unknown_cal].tolist())

    known_dev = [i for i, row in enumerate(records)
                 if row["role"] == "development" and row["label"] in KNOWN]
    unknown_dev = [i for i, row in enumerate(records)
                   if row["role"] == "unknown_development"]
    accepted = valid & (max_score >= np.float32(threshold))
    prediction = np.asarray([KNOWN[int(i)] for i in scores.argmax(axis=1)], dtype=object)
    per_person = {}
    for label in KNOWN:
        ids = [i for i in known_dev if records[i]["label"] == label]
        per_person[label] = {
            "rows": len(ids),
            "fomo_valid": int(sum(valid[i] for i in ids)),
            "raw_top1_correct_all_rows": int(sum(valid[i] and prediction[i] == label for i in ids)),
            "correct_accepted": int(sum(accepted[i] and prediction[i] == label for i in ids)),
            "wrong_known_accepts": int(sum(accepted[i] and prediction[i] != label for i in ids)),
        }
    unknown_by_group = {}
    for i in unknown_dev:
        group = records[i].get("source_group") or records[i]["capture_id"]
        unknown_by_group.setdefault(group, []).append(i)
    return {
        "threshold_binary32": float(np.float32(threshold)),
        "threshold_rule": "one representable binary32 step above the maximum usable unknown-calibration max-known cosine; >= comparison",
        "unknown_calibration_groups": len({records[i].get("source_group") or records[i]["capture_id"]
                                            for i in range(len(records)) if records[i]["role"] == "unknown_calibration"}),
        "unknown_calibration_usable_rows": len(unknown_cal),
        "unknown_calibration_false_groups": 0,
        "known_development_rows": len(known_dev),
        "known_fomo_valid_rows": int(sum(valid[i] for i in known_dev)),
        "known_raw_top1_all_rows": float(sum(valid[i] and prediction[i] == records[i]["label"]
                                              for i in known_dev) / max(1, len(known_dev))),
        "known_correct_accepted": int(sum(accepted[i] and prediction[i] == records[i]["label"]
                                           for i in known_dev)),
        "known_correct_acceptance_all_rows": float(sum(accepted[i] and prediction[i] == records[i]["label"]
                                                       for i in known_dev) / max(1, len(known_dev))),
        "known_coverage_all_rows": float(sum(accepted[i] for i in known_dev) / max(1, len(known_dev))),
        "known_wrong_accepted": int(sum(accepted[i] and prediction[i] != records[i]["label"]
                                        for i in known_dev)),
        "unknown_development_rows": len(unknown_dev),
        "unknown_development_groups": len(unknown_by_group),
        "unknown_fomo_valid_rows": int(sum(valid[i] for i in unknown_dev)),
        "unknown_false_accepted_rows": int(sum(accepted[i] for i in unknown_dev)),
        "unknown_false_accepted_groups": int(sum(any(accepted[i] for i in ids)
                                                  for ids in unknown_by_group.values())),
        "unknown_zero_event_upper95_for_30": 1.0 - 0.05 ** (1.0 / 30.0),
        "per_person": per_person,
    }


def run() -> dict:
    if RESULT.exists() or PREDICTIONS.exists():
        raise FileExistsError("Teacher-student output exists; use a new versioned namespace")
    index_path = INPUTS / "CAPTURE_INDEX.jsonl"
    feature_path = PRIVATE_INPUT / "features.npz"
    teacher_path = PRIVATE_TEACHER / "RECORDS.jsonl"
    rows = [json.loads(line) for line in index_path.read_text().splitlines() if line.strip()]
    teacher_rows = [json.loads(line) for line in teacher_path.read_text().splitlines() if line.strip()]
    if len(rows) != 255 or len(teacher_rows) != 255:
        raise ValueError("teacher/input rows are not the frozen 255-frame camera set")
    if [r["capture_id"] for r in rows] != [r["capture_id"] for r in teacher_rows]:
        raise ValueError("camera input and private teacher rows are not in exact capture order")
    with np.load(feature_path) as cache:
        features = np.asarray(cache["features"], dtype=np.float32)
    if features.shape != (255, 128):
        raise ValueError("compact encoder feature cache has unexpected geometry")
    teacher = np.full((255, 128), np.nan, dtype=np.float32)
    for index, row in enumerate(teacher_rows):
        value = row["yu_net_aligned_embedding"]
        if value is not None:
            teacher[index] = np.asarray(value, dtype=np.float32)
    valid = np.asarray([bool(row["usable_for_identity"]) for row in rows], dtype=bool)
    train_roles = np.asarray([row["role"] in ("train", "other_train") for row in rows])
    eligible = valid & np.isfinite(teacher).all(axis=1) & train_roles
    fit_ids, stop_ids = make_group_holdout(rows, eligible)
    x, y = norm_rows(features), norm_rows(np.nan_to_num(teacher, nan=0.0))
    search = []
    for alpha in ALPHAS:
        weights, bias = fit_ridge(x[fit_ids], y[fit_ids], alpha)
        validation = x[stop_ids] @ weights + bias
        score = cosine_pair_mse(validation, y[stop_ids])
        mean_cosine = float(np.mean(np.sum(norm_rows(validation) * y[stop_ids], axis=1)))
        search.append({"alpha": alpha, "validation_pairwise_cosine_mse": score,
                       "validation_mean_teacher_cosine": mean_cosine})
    selected = min(search, key=lambda row: (row["validation_pairwise_cosine_mse"], row["alpha"]))
    weights, bias = fit_ridge(x[eligible], y[eligible], float(selected["alpha"]))
    projected = x @ weights + bias
    projected_norm = np.linalg.norm(projected, axis=1)
    student_valid = valid & np.isfinite(projected).all(axis=1) & (projected_norm > 1e-8)
    projected = norm_rows(projected)
    base = evaluate(rows, x, valid)
    distilled = evaluate(rows, projected, student_valid)

    PRIVATE_OUT.mkdir(parents=True, exist_ok=True)
    OUT.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(PRIVATE_OUT / "linear_projection.npz", weights=weights, bias=bias,
                        predictions=projected, teacher=teacher)
    prediction_rows = []
    for i, row in enumerate(rows):
        prediction_rows.append({"capture_id": row["capture_id"], "role": row["role"],
                                "label": row["label"], "source_group": row.get("source_group"),
                                "fomo_valid": bool(valid[i]), "teacher_valid": bool(np.isfinite(teacher[i]).all()),
                                "projected_valid": bool(student_valid[i]),
                                "teacher_cosine": (float(np.dot(projected[i], y[i]))
                                                   if np.isfinite(teacher[i]).all() else None)})
    PREDICTIONS.write_text("".join(json.dumps(row, sort_keys=True, allow_nan=False) + "\n"
                                         for row in prediction_rows), encoding="utf-8")
    summary = {
        "schema": "v6-camera-linear-sface-distillation-pilot-1",
        "status": "complete_single_exploratory_pilot",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "parent_input_experiment": "work/project-research-v5/runs/camera_corrected_input_v1/INPUTS.json",
        "parent_teacher_experiment": "work/project-research-v5/runs/camera_corrected_teacher_v1/RESULT.json",
        "input": {"model": "frozen 48x48 RGB888 compact encoder", "feature_dim": 128,
                  "capture_rows": 255, "feature_cache_sha256": sha256(feature_path),
                  "capture_index_sha256": sha256(index_path)},
        "teacher": {"model": "SFace 2021dec YuNet-aligned full-frame embedding",
                    "teacher_rows_sha256": sha256(teacher_path),
                    "deployment_eligible": False, "pretraining_overlap": "unresolved"},
        "fit": {"method": "ridge linear map to teacher embedding, then L2 normalize",
                "fit_roles": ["train", "other_train"], "fit_rows": int(eligible.sum()),
                "internal_group_holdout_rows": int(len(stop_ids)),
                "holdout_split_seed": SPLIT_SEED, "alpha_search": search,
                "selected_alpha": float(selected["alpha"]),
                "selected_stopping_pairwise_cosine_mse": selected["validation_pairwise_cosine_mse"],
                "parameters": int(weights.size + bias.size), "pilot_fits": 1,
                "independent_gradient_roots": 0,
                "learning_behavior": "one train-role group holdout and fixed regularization search; not a ten-root comparison"},
        "base_gallery_route": base,
        "teacher_projected_gallery_route": distilled,
        "improvement_over_base": {
            "known_correct_acceptance_delta": distilled["known_correct_acceptance_all_rows"] - base["known_correct_acceptance_all_rows"],
            "known_coverage_delta": distilled["known_coverage_all_rows"] - base["known_coverage_all_rows"],
            "unknown_false_group_delta": distilled["unknown_false_accepted_groups"] - base["unknown_false_accepted_groups"],
        },
        "private_artifacts": {"prediction_rows": str(PREDICTIONS.relative_to(ROOT)),
                              "projection": str((PRIVATE_OUT / "linear_projection.npz").relative_to(ROOT)),
                              "projection_sha256": sha256(PRIVATE_OUT / "linear_projection.npz")},
        "limits": [
            "Host-only projection diagnostic; no camera/ROI/device tensor parity for this model output.",
            "YuNet alignment is teacher supervision only; the deployable input remains the corrected FOMO crop.",
            "Only one training root was run; do not claim a learned comparison or final recognition result.",
            "Thresholds were derived only from unknown_calibration rows; 30 groups cannot certify a 1% upper risk.",
        ],
        "future_final_used": False,
        "ready_for_final_build": False,
    }
    RESULT.write_text(json.dumps(summary, indent=2, sort_keys=True, allow_nan=False) + "\n",
                      encoding="utf-8")
    return summary


if __name__ == "__main__":
    print(json.dumps(run(), indent=2, sort_keys=True, allow_nan=False))
