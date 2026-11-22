"""One camera-matched MobileNetV1 fine-tuning pilot with v2 SFace supervision.

Train-role FOMO crops are the student inputs; YuNet-aligned SFace embeddings
are targets only. An internal source-group holdout from the train role selects
the stopping epoch. Calibration, development and future-final rows are not used
for fitting or stopping.
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
import sys
import time

import cv2
import numpy as np


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "FaceRecognition"))
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "FaceRecognition/firmware"))
from open_set_decision import accepts, threshold_above_max
from nicla_study import d_campaign as d
from nicla_study import d_teacher
from nicla_study import v6_camera_crop_features48_v2 as replay
from nicla_study.v5_native_rgb import native_rgb
from nicla_study.fs import sha256_file
from nicla_study.state import resource_lease


STUDY = ROOT / "work/project-research-v5"
PRIVATE_TEACHER = STUDY / "private/camera_corrected_teacher_v2/RECORDS.jsonl"
INDEX_PATH = STUDY / "runs/camera_corrected_input_v2/CAPTURE_INDEX.jsonl"
OUT = STUDY / "runs/camera_corrected_deep_student_v2"
PRIVATE = STUDY / "private/camera_corrected_deep_student_v2"
RESULT = OUT / "RESULT.json"
MODEL_WEIGHTS = PRIVATE / "mobilenet_student_weights.npz"
PREDICTIONS = PRIVATE / "PREDICTIONS.jsonl"
ATTEMPTS = PRIVATE / "ATTEMPTS.jsonl"
SEED = 56101
ALPHA = 0.1
ROI_SIDE = 120
INPUT_SIDE = 96
KNOWN = tuple(d.KNOWN)
CONFIG = {"learning_rate": 0.0003, "unfreeze_top_fraction": 0.25,
          "augmentation": "mild_gray_v1", "epochs_max": 12}


def normalize(vector: np.ndarray) -> np.ndarray:
    value = np.asarray(vector, dtype=np.float32).reshape(-1)
    norm = float(np.linalg.norm(value))
    if not np.isfinite(value).all() or norm <= 1e-8:
        raise ValueError("student embedding is invalid")
    return value / norm


def _group_holdout(rows: list[dict]) -> tuple[list[int], list[int]]:
    rng = np.random.default_rng(SEED ^ 0x61502)
    fit, stop = [], []
    for label in KNOWN:
        groups: dict[str, list[int]] = {}
        for i, row in enumerate(rows):
            if row["label_id"] == label:
                groups.setdefault(str(row.get("source_group") or row["record_id"]), []).append(i)
        group_ids = sorted(groups)
        if len(group_ids) < 3:
            raise ValueError("too few FOMO+teacher train rows for " + label)
        rng.shuffle(group_ids)
        stop.extend(groups[group_ids[0]])
        for group_id in group_ids[1:]:
            fit.extend(groups[group_id])
    other_groups: dict[str, list[int]] = {}
    for i, row in enumerate(rows):
        if row["identity_partition"] == "unknown":
            other_groups.setdefault(str(row.get("source_group") or row["record_id"]), []).append(i)
    other_group_ids = sorted(other_groups)
    if len(other_group_ids) < 13:
        raise ValueError("need at least 13 distinct OTHER train groups for the balanced fit/stop split")
    rng.shuffle(other_group_ids)
    for group_id in other_group_ids[:3]:
        stop.extend(other_groups[group_id])
    for group_id in other_group_ids[3:]:
        fit.extend(other_groups[group_id])
    if set(fit) & set(stop):
        raise AssertionError("source-group holdout overlap")
    fit_groups = {(rows[i]["label_id"], str(rows[i].get("source_group") or rows[i]["record_id"]))
                  for i in fit}
    stop_groups = {(rows[i]["label_id"], str(rows[i].get("source_group") or rows[i]["record_id"]))
                   for i in stop}
    if fit_groups & stop_groups:
        raise AssertionError("a source group appears in both fit and stopping sets")
    return sorted(fit), sorted(stop)


def _predict_map(embedding_model, images: np.ndarray, batch_size: int = 16) -> np.ndarray:
    outputs = []
    for start in range(0, len(images), batch_size):
        output = embedding_model(d.gray_float(images[start:start + batch_size]), training=False).numpy()
        outputs.append(np.asarray(output, dtype=np.float32))
    result = np.concatenate(outputs, axis=0)
    if result.ndim != 2 or result.shape[1] != 64 or not np.isfinite(result).all():
        finite = int(np.isfinite(result).sum()) if result.size else 0
        raise ValueError("student embedding output failed the 64-D finite contract: "
                         "shape=%r dtype=%s finite=%d/%d" %
                         (result.shape, result.dtype, finite, result.size))
    return result


def _append_attempt(row: dict) -> None:
    PRIVATE.mkdir(parents=True, exist_ok=True)
    with ATTEMPTS.open("a", encoding="utf-8", newline="\n") as stream:
        stream.write(json.dumps(row, sort_keys=True, allow_nan=False) + "\n")
        stream.flush()


def pairwise_teacher_mse(rows: list[dict], embeddings: np.ndarray,
                         teacher: dict[str, np.ndarray]) -> float:
    target = np.stack([normalize(teacher[str(row["record_id"])]) for row in rows])
    predicted = np.stack([normalize(embeddings[i]) for i in range(len(rows))])
    target_similarity = target @ target.T
    predicted_similarity = predicted @ predicted.T
    off_diagonal = ~np.eye(len(rows), dtype=bool)
    if not np.any(off_diagonal):
        raise ValueError("pairwise diagnostic needs at least two independent rows")
    return float(np.mean(np.square(predicted_similarity[off_diagonal] -
                                   target_similarity[off_diagonal])))


def leave_group_out_accuracy(rows: list[dict], embeddings: np.ndarray) -> dict:
    """Train-role known-ID accuracy using only other source groups as prototypes."""
    known_rows = [(i, row) for i, row in enumerate(rows) if row["label_id"] in KNOWN]
    by_label: dict[str, dict[str, list[np.ndarray]]] = {label: {} for label in KNOWN}
    for i, row in known_rows:
        group = str(row.get("source_group") or row["record_id"])
        by_label[row["label_id"]].setdefault(group, []).append(normalize(embeddings[i]))
    correct = total = 0
    per_person = {}
    for i, row in known_rows:
        own_group = str(row.get("source_group") or row["record_id"])
        prototypes = {}
        for label in KNOWN:
            vectors = [vector for group, values in by_label[label].items() if group != own_group
                       for vector in values]
            if vectors:
                prototypes[label] = normalize(np.mean(np.stack(vectors), axis=0))
        if len(prototypes) != len(KNOWN):
            raise ValueError("leave-source-group-out fit diagnostic lacks a class prototype")
        prediction = max(KNOWN, key=lambda label: float(np.dot(normalize(embeddings[i]), prototypes[label])))
        match = prediction == row["label_id"]
        per_person.setdefault(row["label_id"], {"rows": 0, "correct": 0})
        per_person[row["label_id"]]["rows"] += 1
        per_person[row["label_id"]]["correct"] += int(match)
        total += 1
        correct += int(match)
    return {"correct": correct, "rows": total, "accuracy": correct / max(1, total),
            "per_person": per_person}


def evaluate(rows: list[dict], embeddings: np.ndarray, valid: np.ndarray) -> dict:
    group_values = {label: {} for label in KNOWN}
    for index, row in enumerate(rows):
        if row["role"] != "train" or row["label"] not in KNOWN or not valid[index]:
            continue
        group = row.get("source_group") or row["capture_id"]
        group_values[row["label"]].setdefault(group, []).append(normalize(embeddings[index]))
    gallery = {}
    for label in KNOWN:
        prototypes = [normalize(np.mean(np.stack(values), axis=0))
                      for values in group_values[label].values()]
        if not prototypes:
            raise ValueError("no student gallery rows for " + label)
        gallery[label] = normalize(np.mean(np.stack(prototypes), axis=0))
    scores = np.zeros((len(rows), len(KNOWN)), dtype=np.float32)
    for index in np.flatnonzero(valid):
        vector = normalize(embeddings[index])
        scores[index] = np.asarray([np.dot(vector, gallery[label]) for label in KNOWN], dtype=np.float32)
    max_scores = scores.max(axis=1)
    unknown_cal = [i for i, row in enumerate(rows)
                   if row["role"] == "unknown_calibration" and valid[i]]
    if not unknown_cal:
        raise ValueError("no FOMO-valid unknown calibration rows")
    threshold = threshold_above_max(max_scores[unknown_cal].tolist())
    pred = np.asarray([KNOWN[int(i)] for i in scores.argmax(axis=1)], dtype=object)
    accepted = valid & (max_scores >= np.float32(threshold))
    known_dev = [i for i, row in enumerate(rows)
                 if row["role"] == "development" and row["label"] in KNOWN]
    unknown_dev = [i for i, row in enumerate(rows) if row["role"] == "unknown_development"]
    unknown_groups = {}
    for i in unknown_dev:
        group = rows[i].get("source_group") or rows[i]["capture_id"]
        unknown_groups.setdefault(group, []).append(i)
    per_person = {}
    for label in KNOWN:
        ids = [i for i in known_dev if rows[i]["label"] == label]
        per_person[label] = {"rows": len(ids), "fomo_valid": int(sum(valid[i] for i in ids)),
            "raw_top1_correct_all_rows": int(sum(valid[i] and pred[i] == label for i in ids)),
            "correct_accepted": int(sum(accepted[i] and pred[i] == label for i in ids)),
            "wrong_known_accepts": int(sum(accepted[i] and pred[i] != label for i in ids))}
    return {
        "threshold_binary32": float(np.float32(threshold)),
        "threshold_rule": "one representable binary32 step above the maximum usable unknown-calibration max-known cosine; >= comparison",
        "unknown_calibration_rows": len([r for r in rows if r["role"] == "unknown_calibration"]),
        "unknown_calibration_usable_rows": len(unknown_cal),
        "unknown_calibration_groups": len({r.get("source_group") or r["capture_id"] for r in rows
                                            if r["role"] == "unknown_calibration"}),
        "unknown_calibration_false_groups": 0,
        "known_development_rows": len(known_dev),
        "known_fomo_valid_rows": int(sum(valid[i] for i in known_dev)),
        "known_raw_top1_all_rows": float(sum(valid[i] and pred[i] == rows[i]["label"] for i in known_dev)
                                             / max(1, len(known_dev))),
        "known_correct_accepted": int(sum(accepted[i] and pred[i] == rows[i]["label"] for i in known_dev)),
        "known_correct_acceptance_all_rows": float(sum(accepted[i] and pred[i] == rows[i]["label"] for i in known_dev)
                                                    / max(1, len(known_dev))),
        "known_coverage_all_rows": float(sum(accepted[i] for i in known_dev) / max(1, len(known_dev))),
        "known_wrong_accepted": int(sum(accepted[i] and pred[i] != rows[i]["label"] for i in known_dev)),
        "unknown_development_rows": len(unknown_dev),
        "unknown_development_groups": len(unknown_groups),
        "unknown_fomo_valid_rows": int(sum(valid[i] for i in unknown_dev)),
        "unknown_false_accepted_rows": int(sum(accepted[i] for i in unknown_dev)),
        "unknown_false_accepted_groups": int(sum(any(accepted[i] for i in ids)
                                                  for ids in unknown_groups.values())),
        "unknown_zero_event_upper95_for_30": 1.0 - 0.05 ** (1.0 / 30.0),
        "per_person": per_person,
    }


def run() -> dict:
    if RESULT.exists() or MODEL_WEIGHTS.exists() or PREDICTIONS.exists():
        raise FileExistsError("Camera deep-student output exists; use a new versioned namespace")
    cv2.setNumThreads(2)
    input_rows = [json.loads(line) for line in INDEX_PATH.read_text().splitlines() if line.strip()]
    teacher_rows = [json.loads(line) for line in PRIVATE_TEACHER.read_text().splitlines() if line.strip()]
    replay_rows = replay.load_sets()
    if not (len(input_rows) == len(teacher_rows) == len(replay_rows) == 255):
        raise ValueError("student inputs/teacher are not the frozen 255-frame set")
    if [row["capture_id"] for row in input_rows] != [row["capture_id"] for row in teacher_rows]:
        raise ValueError("corrected input and teacher record order mismatch")
    if [row["request"]["capture_id"] for row in replay_rows] != [row["capture_id"] for row in input_rows]:
        raise ValueError("camera raw-frame and capture-index order mismatch")

    metadata = []
    images = np.zeros((len(input_rows), INPUT_SIDE, INPUT_SIDE, 1), dtype=np.uint8)
    valid = np.zeros(len(input_rows), dtype=bool)
    teacher_targets = {}
    for i, (row, teacher_row, replay_row) in enumerate(zip(input_rows, teacher_rows, replay_rows)):
        if any(row.get(key) != teacher_row.get(key) for key in ("capture_id", "role", "label", "source_group")):
            raise ValueError("camera input and teacher metadata differ at row %d" % i)
        if teacher_row["yu_net_aligned_embedding"] is not None:
            target = np.asarray(teacher_row["yu_net_aligned_embedding"], dtype=np.float32).reshape(-1)
            if target.shape != (128,) or not np.isfinite(target).all() or np.linalg.norm(target) <= 1e-8:
                raise ValueError("teacher embedding failed finite 128-D contract: " + row["capture_id"])
            teacher_targets[row["capture_id"]] = target
        if not row["usable_for_identity"]:
            continue
        roi = tuple(int(v) for v in row["roi_xywh_camera_px"])
        pixels = replay.rgb565_array((replay_row["folder"] / "camera_rgb565.bin").read_bytes())
        rgb = native_rgb(pixels, roi=roi, width=INPUT_SIDE, height=INPUT_SIDE)
        gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
        images[i, :, :, 0] = gray
        valid[i] = True

    fit_rows, stop_rows = [], []
    fit_images, stop_images = [], []
    for i, row in enumerate(input_rows):
        if not valid[i] or row["role"] not in ("train", "other_train"):
            continue
        target = teacher_targets.get(row["capture_id"])
        if target is None:
            continue
        label = row["label"]
        if (row["role"] == "train" and label not in KNOWN) or \
                (row["role"] == "other_train" and label != "OTHER"):
            raise ValueError("train role and identity label disagree: " + row["capture_id"])
        if row["role"] not in ("train", "other_train"):
            raise AssertionError("non-train role entered the student fit pool")
        fit_rows.append({"record_id": row["capture_id"], "label_id": label,
                         "identity_partition": "unknown" if label == "OTHER" else "known",
                         "identity_id": row.get("source_group") or row["capture_id"],
                         "role": row["role"], "source_group": row.get("source_group")})
        fit_images.append(images[i])
    fit_ids, stop_ids = _group_holdout(fit_rows)
    model_rows = fit_rows
    all_fit_images = np.asarray(fit_images, dtype=np.uint8)
    train_rows = [model_rows[i] for i in fit_ids]
    stop_rows = [model_rows[i] for i in stop_ids]
    train_images = all_fit_images[fit_ids]
    stopping_images = all_fit_images[stop_ids]
    train_teacher = {str(row["record_id"]): teacher_targets[row["record_id"]] for row in train_rows}
    stop_teacher = {str(row["record_id"]): teacher_targets[row["record_id"]] for row in stop_rows}
    if len({row["identity_id"] for row in train_rows if row["identity_partition"] == "unknown"}) < 10:
        raise ValueError("internal stop split left fewer than ten OTHER train groups")

    # Establish the train-only control before unfreezing the backbone. This uses
    # the same seed/checkpoint and only the internally held-out train-role rows.
    control_model, control_embedding, _ = d.build_classifier(
        ROOT, ALPHA, True, 6, SEED, "v6_camera_student_unadapted_control_v2")
    initial_train_embeddings = _predict_map(control_embedding, train_images)
    initial_stop_embeddings = _predict_map(control_embedding, stopping_images)
    initial_fit_mse = pairwise_teacher_mse(train_rows, initial_train_embeddings, train_teacher)
    initial_stop_mse = pairwise_teacher_mse(stop_rows, initial_stop_embeddings, stop_teacher)
    initial_fit_accuracy = leave_group_out_accuracy(train_rows, initial_train_embeddings)
    del control_model, control_embedding
    import tensorflow as tf
    tf.keras.backend.clear_session()

    started = time.monotonic()
    # The existing shared training lease and RUN_STATE live with the canonical
    # finish-v4 training workspace; STUDY is the separate v5 evidence namespace.
    with resource_lease(d.study_paths(ROOT)["study"], "heavy_training",
                        owner="v6_camera_mobilenet_distill", timeout_s=0):
        student_model, history = d_teacher._train_distilled(
            ROOT, train_rows, train_images, train_teacher, stop_rows, stopping_images,
            stop_teacher, ALPHA, SEED, CONFIG, "v6_camera_sface_student_a010_seed56101_v2")
    embedding_model = student_model._d_embedding_model
    elapsed = time.monotonic() - started
    PRIVATE.mkdir(parents=True, exist_ok=True)
    OUT.mkdir(parents=True, exist_ok=True)
    attempt_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
    attempt_weights = PRIVATE / ("attempt_" + attempt_id + "_weights.npz")
    weight_arrays = {"weight_%03d" % i: np.asarray(value) for i, value in enumerate(embedding_model.get_weights())}
    np.savez_compressed(attempt_weights, **weight_arrays)
    _append_attempt({"attempt_id": attempt_id, "status": "training_complete_inference_pending",
                     "seed": SEED, "config": CONFIG, "elapsed_seconds": elapsed,
                     "selected_epoch": history["selected_epoch"], "steps_run": history["steps_run"],
                     "learning_curve": history["curve"], "checkpoint_path": str(attempt_weights.relative_to(ROOT)),
                     "checkpoint_sha256": sha256_file(attempt_weights)})
    # Invalid/no-face rows stay invalid. Never infer validity from the model's
    # bias response to the zero-filled storage placeholder.
    all_embeddings = np.zeros((len(input_rows), 64), dtype=np.float32)
    valid_indices = np.flatnonzero(valid)
    try:
        all_embeddings[valid_indices] = _predict_map(embedding_model, images[valid_indices])
    except Exception as exc:
        _append_attempt({"attempt_id": attempt_id, "status": "failed_selected_checkpoint_inference",
                         "error": repr(exc), "checkpoint_path": str(attempt_weights.relative_to(ROOT)),
                         "checkpoint_sha256": sha256_file(attempt_weights),
                         "selected_epoch": history["selected_epoch"], "steps_run": history["steps_run"],
                         "learning_curve": history["curve"]})
        raise
    valid[valid_indices] &= np.linalg.norm(all_embeddings[valid_indices], axis=1) > 1e-8
    metrics = evaluate(input_rows, all_embeddings, valid)
    fit_embeddings = _predict_map(embedding_model, train_images)
    stop_embeddings = _predict_map(embedding_model, stopping_images)
    trained_fit_mse = pairwise_teacher_mse(train_rows, fit_embeddings, train_teacher)
    trained_stop_mse = pairwise_teacher_mse(stop_rows, stop_embeddings, stop_teacher)

    np.savez_compressed(MODEL_WEIGHTS, **weight_arrays)
    prediction_rows = []
    for i, row in enumerate(input_rows):
        prediction_rows.append({"capture_id": row["capture_id"], "role": row["role"], "label": row["label"],
                                "source_group": row.get("source_group"), "fomo_valid": bool(valid[i]),
                                "student_embedding": normalize(all_embeddings[i]).tolist() if valid[i] else None})
    PREDICTIONS.write_text("".join(json.dumps(row, sort_keys=True, allow_nan=False) + "\n"
                                         for row in prediction_rows), encoding="utf-8")
    summary = {
        "schema": "v6-camera-sface-distilled-mobilenet-pilot-2",
        "status": "complete_single_gradient_pilot",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "parent_input_experiment": "work/project-research-v5/runs/camera_corrected_input_v2/INPUTS.json",
        "parent_teacher_experiment": "work/project-research-v5/runs/camera_corrected_teacher_v2/RESULT.json",
        "student": {"architecture": "MobileNetV1 alpha 0.1 grayscale96 + learned 64-D embedding",
                    "initial_checkpoint_sha256": d.EI_HASHES[ALPHA],
                    "input_contract": "RGB565->native RGB888 120px ROI resized 96x96->gray8->float32/255",
                    "teacher_target": "YuNet single-face aligned SFace 128-D embedding on the same frame; targets only, not a runtime detector",
                    "weights_path": str(MODEL_WEIGHTS.relative_to(ROOT)),
                    "weights_sha256": sha256_file(MODEL_WEIGHTS),
                    "parameter_count": int(embedding_model.count_params()),
                    "trainable_parameter_count": int(sum(np.prod(v.shape) for v in embedding_model.trainable_variables))},
        "training": {"seed": SEED, "alpha": ALPHA, "config": CONFIG,
                     "input_rows_with_teacher": len(fit_rows), "fit_rows": len(train_rows),
                     "internal_train_role_group_stopping_rows": len(stop_rows),
                     "fit_groups_by_label": {label: sum(row["label_id"] == label for row in train_rows) for label in KNOWN},
                     "other_fit_groups": len({row["identity_id"] for row in train_rows
                                               if row["identity_partition"] == "unknown"}),
                     "initial_checkpoint_sha256": d.EI_HASHES[ALPHA],
                     "learning_curve": history["curve"], "selected_epoch": history["selected_epoch"],
                     "selected_stopping_pairwise_cosine_mse": history["selected_stopping_pairwise_cosine_mse"],
                     "steps_run": history["steps_run"], "sample_presentations": history["sample_presentations"],
                     "max_gradient_global_norm": history["max_gradient_global_norm"],
                     "train_only_control": {
                         "initial_fit_pairwise_teacher_mse": initial_fit_mse,
                         "selected_fit_pairwise_teacher_mse": trained_fit_mse,
                         "initial_stopping_pairwise_teacher_mse": initial_stop_mse,
                         "selected_stopping_pairwise_teacher_mse": trained_stop_mse,
                         "initial_leave_source_group_out_known_top1": initial_fit_accuracy,
                         "selected_leave_source_group_out_known_top1": leave_group_out_accuracy(train_rows, fit_embeddings),
                         "fit_stop_source_group_overlap": 0,
                     },
                     "elapsed_seconds": elapsed, "gradient_training": True},
        "evaluation": metrics,
        "private_predictions": str(PREDICTIONS.relative_to(ROOT)),
        "limits": [
            "One gradient-training pilot only; no ten-root learned comparison has been reported.",
            "Host input generation and YuNet teacher targets are not exact Nicla tensor/device parity.",
            "FOMO detector misses and ambiguous outputs remain counted in end-to-end denominators.",
            "The unknown development/calibration sets contain 30 groups and cannot certify a 1% upper risk.",
            "No candidate is deployable until the Nicla FOMO allocation and full runtime resource gates pass.",
        ],
        "future_final_used": False,
        "ready_for_final_build": False,
    }
    RESULT.write_text(json.dumps(summary, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    return summary


if __name__ == "__main__":
    print(json.dumps(run(), indent=2, sort_keys=True, allow_nan=False))
