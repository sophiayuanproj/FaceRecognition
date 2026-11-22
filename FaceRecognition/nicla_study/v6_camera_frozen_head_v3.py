"""Train one balanced linear head on corrected camera-matched v3 embeddings.

The frozen 48px encoder features come only from strict-threshold FOMO crops in
the v2 replay. Fit and early stopping use train/other_train source groups;
threshold calibration and development rows remain sealed from optimization.
This is a one-root diagnostic pilot, not a deployment candidate by itself.
"""

from __future__ import annotations

import hashlib
import json
import math
from datetime import datetime, timezone
from pathlib import Path
import sys

import numpy as np
import tensorflow as tf

ROOT = Path(__file__).resolve().parents[2]
STUDY = ROOT / "work/project-research-v5"
FIRMWARE = ROOT / "FaceRecognition/firmware"
sys.path.insert(0, str(ROOT / "FaceRecognition"))
sys.path.insert(0, str(FIRMWARE))
from open_set_decision import threshold_above_max, accepts
from nicla_study.fs import sha256_file

INPUT_DIR = STUDY / "runs/camera_corrected_input_v3"
PRIVATE_INPUT_DIR = STUDY / "private/camera_corrected_input_v3"
OUT = STUDY / "runs/camera_corrected_head_v3"
PRIVATE = STUDY / "private/camera_corrected_head_v3"
RESULT = OUT / "RESULT.json"
PREDICTIONS = PRIVATE / "PREDICTIONS.jsonl"
WEIGHTS = PRIVATE / "linear_head.npz"
SEED = 56002
KNOWN = ("K01", "K02", "K03", "K04", "K05")
CLASSES = KNOWN + ("OTHER",)
PATIENCE = 12
MAX_EPOCHS = 120
LEARNING_RATE = 0.025
BALANCED_SAMPLES_PER_CLASS_PER_EPOCH = 32


def _softmax(logits: np.ndarray) -> np.ndarray:
    values = np.asarray(logits, dtype=np.float32)
    shifted = values - np.max(values, axis=1, keepdims=True)
    exp = np.exp(shifted.astype(np.float64))
    return (exp / exp.sum(axis=1, keepdims=True)).astype(np.float32)


def _group_split(rows: list[dict]) -> tuple[list[int], list[int]]:
    rng = np.random.default_rng(SEED ^ 0x51A7)
    by_class: dict[str, dict[str, list[int]]] = {label: {} for label in CLASSES}
    for i, row in enumerate(rows):
        by_class[row["label"]].setdefault(row["source_group"], []).append(i)
    fit, stopping = [], []
    for label in CLASSES:
        groups = sorted(by_class[label])
        if len(groups) < 3:
            raise ValueError("fewer than three train source groups for " + label)
        rng.shuffle(groups)
        stopping.extend(by_class[label][groups[0]])
        for group in groups[1:]:
            fit.extend(by_class[label][group])
    fit_groups = {(rows[i]["label"], rows[i]["source_group"]) for i in fit}
    stop_groups = {(rows[i]["label"], rows[i]["source_group"]) for i in stopping}
    if fit_groups & stop_groups:
        raise AssertionError("train and stopping groups overlap")
    return sorted(fit), sorted(stopping)


def _balanced_batches(features: np.ndarray, targets: np.ndarray, rng: np.random.Generator):
    members = [np.flatnonzero(targets == class_id) for class_id in range(len(CLASSES))]
    if any(len(ids) == 0 for ids in members):
        raise ValueError("every class needs at least one train-role fit row")
    samples = []
    for class_id, ids in enumerate(members):
        chosen = rng.choice(ids, BALANCED_SAMPLES_PER_CLASS_PER_EPOCH, replace=True)
        samples.extend((int(i), class_id) for i in chosen)
    rng.shuffle(samples)
    for start in range(0, len(samples), 12):
        batch = samples[start:start + 12]
        ids = np.asarray([item[0] for item in batch], dtype=np.int64)
        labels = np.asarray([item[1] for item in batch], dtype=np.int64)
        yield features[ids], labels


def _accuracy(logits: np.ndarray, target: np.ndarray) -> dict:
    pred = np.argmax(logits, axis=1)
    per_class = {}
    for class_id, label in enumerate(CLASSES):
        ids = np.flatnonzero(target == class_id)
        per_class[label] = {"rows": int(len(ids)), "correct": int(np.sum(pred[ids] == class_id))}
    return {"correct": int(np.sum(pred == target)), "rows": int(len(target)),
            "accuracy": float(np.mean(pred == target)) if len(target) else None,
            "per_class": per_class}


def _eval(rows: list[dict], probs: np.ndarray, valid: np.ndarray, threshold: float) -> dict:
    predicted = np.argmax(probs, axis=1)
    max_known = np.max(probs[:, :5], axis=1)
    accepted = valid & (predicted < 5) & np.asarray([accepts(float(v), threshold) for v in max_known])
    known_ids = [i for i, row in enumerate(rows) if row["role"] == "development" and row["label"] in KNOWN]
    unknown_ids = [i for i, row in enumerate(rows) if row["role"] == "unknown_development"]
    per_person = {}
    for label in KNOWN:
        ids = [i for i in known_ids if rows[i]["label"] == label]
        per_person[label] = {
            "rows": len(ids),
            "single_cluster_rows": sum(bool(valid[i]) for i in ids),
            "six_way_top1_correct_all_rows": sum(bool(valid[i]) and CLASSES[predicted[i]] == label for i in ids),
            "known_only_top1_correct_detected": sum(bool(valid[i]) and CLASSES[int(np.argmax(probs[i, :5]))] == label for i in ids),
            "correct_accepted": sum(bool(accepted[i]) and CLASSES[predicted[i]] == label for i in ids),
            "wrong_known_accepted": sum(bool(accepted[i]) and CLASSES[predicted[i]] != label for i in ids),
        }
    unknown_groups: dict[str, list[int]] = {}
    for i in unknown_ids:
        unknown_groups.setdefault(str(rows[i].get("source_group") or rows[i]["capture_id"]), []).append(i)
    false_groups = sum(any(bool(accepted[i]) for i in ids) for ids in unknown_groups.values())
    return {
        "threshold_binary32": float(np.float32(threshold)),
        "known_development_rows": len(known_ids),
        "known_single_cluster_rows": sum(bool(valid[i]) for i in known_ids),
        "known_six_way_top1_correct_all_rows": sum(bool(valid[i]) and CLASSES[predicted[i]] == rows[i]["label"] for i in known_ids),
        "known_only_top1_correct_detected": sum(bool(valid[i]) and CLASSES[int(np.argmax(probs[i, :5]))] == rows[i]["label"] for i in known_ids),
        "known_correct_accepted": sum(bool(accepted[i]) and CLASSES[predicted[i]] == rows[i]["label"] for i in known_ids),
        "known_coverage_all_rows": float(np.sum(accepted[known_ids]) / max(1, len(known_ids))),
        "known_wrong_accepted": sum(bool(accepted[i]) and CLASSES[predicted[i]] != rows[i]["label"] for i in known_ids),
        "unknown_development_rows": len(unknown_ids),
        "unknown_development_groups": len(unknown_groups),
        "unknown_single_cluster_rows": sum(bool(valid[i]) for i in unknown_ids),
        "unknown_false_accepted_rows": sum(bool(accepted[i]) for i in unknown_ids),
        "unknown_false_accepted_groups": false_groups,
        "unknown_zero_event_upper95_for_30": 1.0 - 0.05 ** (1.0 / 30.0),
        "per_person": per_person,
    }


def run() -> dict:
    for path in (RESULT, PREDICTIONS, WEIGHTS):
        if path.exists():
            raise FileExistsError("corrected v2 head result exists; use a new versioned namespace")
    index_path = INPUT_DIR / "CAPTURE_INDEX.jsonl"
    inputs_path = INPUT_DIR / "INPUTS.json"
    features_path = PRIVATE_INPUT_DIR / "features.npz"
    rows = [json.loads(line) for line in index_path.read_text().splitlines() if line.strip()]
    inputs = json.loads(inputs_path.read_text())
    features = np.load(features_path)["features"].astype(np.float32)
    if len(rows) != 255 or features.shape != (255, 128) or inputs.get("schema") != "v6-camera-crop-inputs-3":
        raise ValueError("v3 frozen inputs are incomplete or have the wrong schema")
    valid = np.asarray([bool(row["usable_for_identity"]) for row in rows], dtype=bool)
    for i, row in enumerate(rows):
        if valid[i] and (not np.isfinite(features[i]).all() or abs(float(np.linalg.norm(features[i])) - 1.0) > 2e-4):
            raise ValueError("usable encoder feature is not finite and unit-normalized: " + row["capture_id"])
        if not valid[i] and np.any(features[i]):
            raise ValueError("non-usable camera rows must remain zero-filled")

    train_ids = [i for i, row in enumerate(rows)
                 if valid[i] and row["role"] in ("train", "other_train")]
    train_rows = []
    train_features = []
    for i in train_ids:
        row = rows[i]
        if (row["role"] == "train" and row["label"] not in KNOWN) or \
                (row["role"] == "other_train" and row["label"] != "OTHER"):
            raise ValueError("role and identity label mismatch: " + row["capture_id"])
        group = row.get("source_group") or row["capture_id"]
        train_rows.append({"capture_id": row["capture_id"], "role": row["role"],
                           "label": row["label"], "source_group": group})
        train_features.append(features[i])
    train_features = np.asarray(train_features, dtype=np.float32)
    targets = np.asarray([CLASSES.index(row["label"]) for row in train_rows], dtype=np.int64)
    fit_ids, stop_ids = _group_split(train_rows)
    fit_x, fit_y = train_features[fit_ids], targets[fit_ids]
    stop_x, stop_y = train_features[stop_ids], targets[stop_ids]

    tf.keras.utils.set_random_seed(SEED)
    model = tf.keras.Sequential([tf.keras.layers.Input(shape=(128,)),
                                 tf.keras.layers.Dense(len(CLASSES), name="camera_identity_logits")])
    optimizer = tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE)
    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    initial_fit_logits = model(fit_x, training=False).numpy()
    initial_stop_logits = model(stop_x, training=False).numpy()
    initial_fit_loss = float(tf.reduce_mean(loss_fn(fit_y, initial_fit_logits)).numpy())
    initial_stop_loss = float(tf.reduce_mean(loss_fn(stop_y, initial_stop_logits)).numpy())
    history = []
    best_weights, best_stop_loss, best_epoch, stale = None, math.inf, 0, 0
    rng = np.random.default_rng(SEED)
    finite_gradient_batches = 0
    for epoch in range(MAX_EPOCHS):
        batch_losses = []
        grad_norms = []
        for batch_x, batch_y in _balanced_batches(fit_x, fit_y, rng):
            with tf.GradientTape() as tape:
                logits = model(batch_x, training=True)
                loss = tf.reduce_mean(loss_fn(batch_y, logits))
            grads = tape.gradient(loss, model.trainable_variables)
            if any(g is None or not bool(tf.reduce_all(tf.math.is_finite(g)).numpy()) for g in grads):
                raise ValueError("non-finite gradient in camera linear-head pilot")
            grad_norm = float(tf.linalg.global_norm(grads).numpy())
            if not math.isfinite(grad_norm):
                raise ValueError("non-finite gradient norm")
            finite_gradient_batches += 1
            grad_norms.append(grad_norm)
            optimizer.apply_gradients(zip(grads, model.trainable_variables))
            batch_losses.append(float(loss.numpy()))
        fit_logits = model(fit_x, training=False).numpy()
        stop_logits = model(stop_x, training=False).numpy()
        fit_loss = float(tf.reduce_mean(loss_fn(fit_y, fit_logits)).numpy())
        stop_loss = float(tf.reduce_mean(loss_fn(stop_y, stop_logits)).numpy())
        history.append({"epoch": epoch + 1, "balanced_batch_cross_entropy": float(np.mean(batch_losses)),
                        "fit_cross_entropy": fit_loss, "stop_cross_entropy": stop_loss,
                        "fit_accuracy": _accuracy(fit_logits, fit_y)["accuracy"],
                        "stop_accuracy": _accuracy(stop_logits, stop_y)["accuracy"],
                        "mean_gradient_norm": float(np.mean(grad_norms))})
        if stop_loss < best_stop_loss - 1e-6:
            best_stop_loss, best_epoch, stale = stop_loss, epoch + 1, 0
            best_weights = [np.array(w, copy=True) for w in model.get_weights()]
        else:
            stale += 1
        if stale >= PATIENCE:
            break
    if best_weights is None:
        raise ValueError("no train-only stopping checkpoint was selected")
    model.set_weights(best_weights)
    selected_fit_logits = model(fit_x, training=False).numpy()
    selected_stop_logits = model(stop_x, training=False).numpy()

    all_logits = np.zeros((len(rows), len(CLASSES)), dtype=np.float32)
    valid_ids = np.flatnonzero(valid)
    all_logits[valid_ids] = model(features[valid_ids], training=False).numpy()
    probabilities = np.zeros_like(all_logits)
    probabilities[valid_ids] = _softmax(all_logits[valid_ids])
    max_known = np.max(probabilities[:, :5], axis=1)
    unknown_cal_ids = [i for i, row in enumerate(rows)
                       if row["role"] == "unknown_calibration" and valid[i]]
    known_cal_ids = [i for i, row in enumerate(rows)
                     if row["role"] == "threshold_calibration" and valid[i]]
    if not unknown_cal_ids or not known_cal_ids:
        raise ValueError("known and unknown threshold calibration rows are required")
    threshold = threshold_above_max(max_known[unknown_cal_ids].tolist())
    evaluation = _eval(rows, probabilities, valid, threshold)
    evaluation["known_threshold_calibration_rows"] = sum(row["role"] == "threshold_calibration" for row in rows)
    evaluation["known_threshold_calibration_usable_rows"] = len(known_cal_ids)
    evaluation["unknown_threshold_calibration_rows"] = sum(row["role"] == "unknown_calibration" for row in rows)
    evaluation["unknown_threshold_calibration_usable_rows"] = len(unknown_cal_ids)
    evaluation["unknown_calibration_false_accepted_rows"] = int(np.sum(max_known[unknown_cal_ids] >= np.float32(threshold)))

    PRIVATE.mkdir(parents=True, exist_ok=True)
    OUT.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(WEIGHTS, kernel=model.get_weights()[0], bias=model.get_weights()[1])
    prediction_rows = []
    predicted = np.argmax(probabilities, axis=1)
    for i, row in enumerate(rows):
        prediction_rows.append({"capture_id": row["capture_id"], "role": row["role"],
                                "label": row["label"], "source_group": row.get("source_group"),
                                "quality_status": row["quality_status"], "valid": bool(valid[i]),
                                "logits": all_logits[i].tolist() if valid[i] else None,
                                "probabilities": probabilities[i].tolist() if valid[i] else None,
                                "predicted_class": CLASSES[int(predicted[i])] if valid[i] else None,
                                "decision": (CLASSES[int(predicted[i])] if valid[i] and predicted[i] == 5 else
                                             (CLASSES[int(predicted[i])] if valid[i] and predicted[i] < 5 and accepts(float(max_known[i]), threshold) else
                                              ("UNCERTAIN" if valid[i] else "NO_FACE")))}
        )
    PREDICTIONS.write_text("".join(json.dumps(row, sort_keys=True, allow_nan=False) + "\n"
                                               for row in prediction_rows), encoding="utf-8")
    result = {
        "schema": "v6-corrected-camera-frozen-head-pilot-3",
        "status": "complete_one_root_balanced_linear_head_pilot",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "parent_inputs": {"receipt": str(inputs_path.relative_to(ROOT)),
                          "receipt_sha256": sha256_file(inputs_path),
                          "capture_index_sha256": sha256_file(index_path),
                          "feature_archive": str(features_path.relative_to(ROOT)),
                          "feature_archive_sha256": sha256_file(features_path),
                          "encoder_model_sha256": inputs["model_sha256"],
                          "future_final_used": False},
        "model": {"architecture": "frozen operator_fused_encoder48 128-D L2 feature plus trainable 128x6 linear logits",
                  "input_contract": "v3 FOMO strict float32 >0.5; exact two-peak center span <=48px merged to highest-score peak and logged; camera_xywh ROI120; RGB888 to encoder48",
                  "output_classes": list(CLASSES), "weights_path": str(WEIGHTS.relative_to(ROOT)),
                  "weights_sha256": sha256_file(WEIGHTS), "feature_dimension": 128},
        "training": {"root_seed": SEED, "root_count": 1,
                     "fit_rows": int(len(fit_ids)), "stopping_rows": int(len(stop_ids)),
                     "fit_group_counts": {label: len({train_rows[i]["source_group"] for i in fit_ids if train_rows[i]["label"] == label}) for label in CLASSES},
                     "stopping_group_counts": {label: len({train_rows[i]["source_group"] for i in stop_ids if train_rows[i]["label"] == label}) for label in CLASSES},
                     "fit_stop_group_overlap": 0, "balanced_samples_per_class_per_epoch": BALANCED_SAMPLES_PER_CLASS_PER_EPOCH,
                     "epochs_max": MAX_EPOCHS, "epochs_run": len(history), "selected_epoch": best_epoch,
                     "patience": PATIENCE, "learning_rate": LEARNING_RATE,
                     "finite_gradient_batches": finite_gradient_batches,
                     "initial_fit_cross_entropy": initial_fit_loss,
                     "selected_fit_cross_entropy": float(tf.reduce_mean(loss_fn(fit_y, selected_fit_logits)).numpy()),
                     "initial_fit_accuracy": _accuracy(initial_fit_logits, fit_y),
                     "selected_fit_accuracy": _accuracy(selected_fit_logits, fit_y),
                     "initial_stop_cross_entropy": initial_stop_loss,
                     "selected_stop_cross_entropy": float(tf.reduce_mean(loss_fn(stop_y, selected_stop_logits)).numpy()),
                     "initial_stop_accuracy": _accuracy(initial_stop_logits, stop_y),
                     "selected_stop_accuracy": _accuracy(selected_stop_logits, stop_y),
                     "history": history},
        "evaluation": evaluation,
        "private_predictions": str(PREDICTIONS.relative_to(ROOT)),
        "limits": ["One initialization/sampling root only; no ten-root learned comparison is claimed.",
                   "Frozen encoder feature route has exact parity for one device frame only; head weights are host-only and not yet compiled or measured on Nicla.",
                   "Detection misses and multiple-cluster ambiguity remain in full development denominators.",
                   "Thirty unknown identities cannot certify the one-sided 1% risk target."],
        "ready_for_final_build": False,
    }
    RESULT.write_text(json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    return result


if __name__ == "__main__":
    print(json.dumps(run(), indent=2, sort_keys=True, allow_nan=False))
