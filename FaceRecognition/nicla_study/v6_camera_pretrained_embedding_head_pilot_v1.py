"""Fit one train-only six-way head on corrected camera MobileFaceNet features.

The pretrained face encoder stays frozen. The embedding classifier is an
upper-bound diagnostic for the corrected FOMO crop, not an MCU candidate: the
encoder's current size/input path have not passed device resource qualification.
"""

from __future__ import annotations

import argparse
import json
import math
from datetime import datetime, timezone
from pathlib import Path
import sys
import time

import numpy as np
import tensorflow as tf

ROOT = Path(__file__).resolve().parents[2]
STUDY = ROOT / "work/project-research-v5"
FIRMWARE = ROOT / "FaceRecognition/firmware"
sys.path.insert(0, str(ROOT / "FaceRecognition"))
sys.path.insert(0, str(FIRMWARE))
from open_set_decision import threshold_above_max, accepts
from nicla_study.fs import sha256_file

INPUT = STUDY / "private/camera_corrected_face_encoder_v2/FLOAT_RECORDS.jsonl"
INDEX = STUDY / "runs/camera_corrected_input_v2/CAPTURE_INDEX.jsonl"
DEFAULT_SEED = 56301
KNOWN = ("K01", "K02", "K03", "K04", "K05")
CLASSES = KNOWN + ("OTHER",)
MAX_EPOCHS = 120
PATIENCE = 12
LEARNING_RATE = 0.01
SAMPLES_PER_CLASS_PER_EPOCH = 32
BATCH_SIZE = 16


def _softmax(logits: np.ndarray) -> np.ndarray:
    values = np.asarray(logits, dtype=np.float32)
    shifted = values - np.max(values, axis=1, keepdims=True)
    exp = np.exp(shifted.astype(np.float64))
    return (exp / exp.sum(axis=1, keepdims=True)).astype(np.float32)


def _load_records() -> list[dict]:
    rows = [json.loads(line) for line in INPUT.read_text().splitlines() if line.strip()]
    index = [json.loads(line) for line in INDEX.read_text().splitlines() if line.strip()]
    if len(rows) != 255 or len(index) != len(rows):
        raise ValueError("expected exactly 255 frozen corrected camera rows")
    for row, meta in zip(rows, index):
        if row["capture_id"] != meta["capture_id"]:
            raise ValueError("feature and input row order differ")
        if row["role"] != meta["role"] or row["label"] != meta["label"]:
            raise ValueError("feature/input role or label mismatch")
    return rows


def _feature(row: dict) -> np.ndarray | None:
    values = row.get("embedding")
    if row.get("fomo_cluster_count") != 1 or not isinstance(values, list) or len(values) != 128:
        return None
    vector = np.asarray(values, dtype=np.float32)
    if vector.shape != (128,) or not np.isfinite(vector).all():
        raise ValueError("nonfinite or malformed pretrained face embedding")
    norm = float(np.linalg.norm(vector))
    if not math.isfinite(norm) or norm <= 1e-8:
        raise ValueError("invalid pretrained face embedding norm")
    return vector / np.float32(norm)


def _group_split(rows: list[dict], split_seed: int) -> tuple[list[int], list[int]]:
    rng = np.random.default_rng(split_seed ^ 0xFACE)
    by_class: dict[str, dict[str, list[int]]] = {label: {} for label in CLASSES}
    for i, row in enumerate(rows):
        by_class[row["label"]].setdefault(row["source_group"], []).append(i)
    fit, stop = [], []
    for label in CLASSES:
        groups = sorted(by_class[label])
        if len(groups) < 3:
            raise ValueError("fewer than three train source groups for " + label)
        rng.shuffle(groups)
        stop.extend(by_class[label][groups[0]])
        fit.extend(i for group in groups[1:] for i in by_class[label][group])
    fit_groups = {(rows[i]["label"], rows[i]["source_group"]) for i in fit}
    stop_groups = {(rows[i]["label"], rows[i]["source_group"]) for i in stop}
    if fit_groups & stop_groups:
        raise AssertionError("train/stopping source groups overlap")
    return sorted(fit), sorted(stop)


def _cross_entropy(model, x: np.ndarray, y: np.ndarray) -> float:
    loss = tf.keras.losses.sparse_categorical_crossentropy(y, model(x, training=False), from_logits=True)
    value = float(tf.reduce_mean(loss).numpy())
    if not math.isfinite(value):
        raise ValueError("nonfinite cross entropy")
    return value


def _accuracy(logits: np.ndarray, y: np.ndarray) -> float:
    return float(np.mean(np.argmax(logits, axis=1) == y)) if len(y) else 0.0


def _evaluate(rows: list[dict], probabilities: list[np.ndarray | None], threshold: float) -> dict:
    known_ids = [i for i, row in enumerate(rows) if row["role"] == "development" and row["label"] in KNOWN]
    unknown_ids = [i for i, row in enumerate(rows) if row["role"] == "unknown_development"]
    predictions, scores, accepted = [], [], []
    for prob in probabilities:
        if prob is None:
            predictions.append(None)
            scores.append(None)
            accepted.append(False)
        else:
            pred = int(np.argmax(prob))
            score = float(np.max(prob[:len(KNOWN)]))
            predictions.append(pred)
            scores.append(score)
            accepted.append(pred < len(KNOWN) and accepts(score, threshold))
    known_correct_top1 = 0
    known_identity_top1 = 0
    per_person = {}
    for label in KNOWN:
        ids = [i for i in known_ids if rows[i]["label"] == label]
        valid = [i for i in ids if predictions[i] is not None]
        k_pred = [int(np.argmax(probabilities[i][:len(KNOWN)])) for i in valid]
        raw = sum(CLASSES[predictions[i]] == label for i in valid)
        raw_known = sum(KNOWN[k_pred[j]] == label for j, i in enumerate(valid))
        right = sum(accepted[i] and CLASSES[predictions[i]] == label for i in ids)
        wrong = sum(accepted[i] and CLASSES[predictions[i]] != label for i in ids)
        known_correct_top1 += raw
        known_identity_top1 += raw_known
        per_person[label] = {
            "rows": len(ids), "fomo_single_cluster_rows": len(valid),
            "six_way_top1_correct": raw, "known_only_top1_correct": raw_known,
            "correct_accepted": int(right), "wrong_known_accepts": int(wrong),
        }
    groups: dict[str, list[int]] = {}
    for i in unknown_ids:
        groups.setdefault(rows[i].get("source_group") or rows[i]["capture_id"], []).append(i)
    false_groups = sum(any(accepted[i] for i in group) for group in groups.values())
    return {
        "known_development_rows": len(known_ids),
        "known_fomo_valid_rows": sum(predictions[i] is not None for i in known_ids),
        "known_raw_top1_all_rows": known_correct_top1 / max(1, len(known_ids)),
        "known_only_top1_all_rows": known_identity_top1 / max(1, len(known_ids)),
        "known_correct_accepted": sum(accepted[i] and CLASSES[predictions[i]] == rows[i]["label"] for i in known_ids),
        "known_correct_acceptance_all_rows": sum(accepted[i] and CLASSES[predictions[i]] == rows[i]["label"] for i in known_ids) / max(1, len(known_ids)),
        "known_coverage_all_rows": sum(accepted[i] for i in known_ids) / max(1, len(known_ids)),
        "known_wrong_accepted": sum(accepted[i] and CLASSES[predictions[i]] != rows[i]["label"] for i in known_ids),
        "per_person": per_person,
        "unknown_development_rows": len(unknown_ids),
        "unknown_development_groups": len(groups),
        "unknown_fomo_valid_rows": sum(predictions[i] is not None for i in unknown_ids),
        "unknown_false_accepted_rows": sum(accepted[i] for i in unknown_ids),
        "unknown_false_accepted_groups": false_groups,
        "unknown_zero_event_upper95_for_30": 1.0 - 0.05 ** (1.0 / 30.0),
    }


def run(seed: int = DEFAULT_SEED, split_seed: int | None = None,
        run_name: str = "camera_pretrained_embedding_head_v1") -> dict:
    if isinstance(seed, bool) or int(seed) != seed or seed <= 0:
        raise ValueError("seed must be a positive integer")
    seed = int(seed)
    if split_seed is None:
        split_seed = seed
    if isinstance(split_seed, bool) or int(split_seed) != split_seed or split_seed <= 0:
        raise ValueError("split_seed must be a positive integer")
    split_seed = int(split_seed)
    if not run_name or "/" in run_name or "\\" in run_name or run_name in (".", ".."):
        raise ValueError("run_name must be one path component")
    out_dir = STUDY / "runs" / run_name
    private_dir = STUDY / "private" / run_name
    result_path = out_dir / "RESULT.json"
    predictions_path = private_dir / "PREDICTIONS.jsonl"
    weights_path = private_dir / "linear_head.npz"
    if result_path.exists() or weights_path.exists() or predictions_path.exists():
        raise FileExistsError("output namespace already exists")
    out_dir.mkdir(parents=True)
    private_dir.mkdir(parents=True)
    tf.keras.utils.set_random_seed(seed)
    rows = _load_records()

    feature_map = {row["capture_id"]: _feature(row) for row in rows}
    train_rows = []
    for row in rows:
        if row["role"] not in ("train", "other_train"):
            continue
        if row["role"] == "train" and row["label"] not in KNOWN:
            raise ValueError("train role has a non-roster label")
        if row["role"] == "other_train" and row["label"] != "OTHER":
            raise ValueError("other_train role has a non-OTHER label")
        x = feature_map[row["capture_id"]]
        if x is None:
            continue
        train_rows.append({**row, "source_group": row.get("source_group") or row["capture_id"]})
    fit_ids, stop_ids = _group_split(train_rows, split_seed)
    fit_rows = [train_rows[i] for i in fit_ids]
    stop_rows = [train_rows[i] for i in stop_ids]
    x_fit = np.stack([feature_map[row["capture_id"]] for row in fit_rows]).astype(np.float32)
    y_fit = np.asarray([CLASSES.index(row["label"]) for row in fit_rows], dtype=np.int64)
    x_stop = np.stack([feature_map[row["capture_id"]] for row in stop_rows]).astype(np.float32)
    y_stop = np.asarray([CLASSES.index(row["label"]) for row in stop_rows], dtype=np.int64)
    if any(not np.any(y_fit == class_id) for class_id in range(len(CLASSES))):
        raise ValueError("balanced training requires all six classes")

    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(128,), dtype=tf.float32),
        tf.keras.layers.Dense(len(CLASSES), use_bias=True,
                              kernel_initializer=tf.keras.initializers.GlorotUniform(seed=seed),
                              bias_initializer="zeros"),
    ])
    optimizer = tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE)
    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    initial_fit_logits = model(x_fit, training=False).numpy()
    initial_stop_logits = model(x_stop, training=False).numpy()
    initial_fit_loss = _cross_entropy(model, x_fit, y_fit)
    initial_stop_loss = _cross_entropy(model, x_stop, y_stop)
    initial_fit_accuracy = _accuracy(initial_fit_logits, y_fit)
    initial_stop_accuracy = _accuracy(initial_stop_logits, y_stop)

    rng = np.random.default_rng(seed ^ 0xBADA55)
    per_class = [np.flatnonzero(y_fit == class_id) for class_id in range(len(CLASSES))]
    history, best_weights, best_loss, best_epoch, stale = [], None, float("inf"), 0, 0
    max_gradient = 0.0
    started = time.monotonic()
    for epoch in range(MAX_EPOCHS):
        sampled = []
        for class_id, members in enumerate(per_class):
            ids = rng.choice(members, SAMPLES_PER_CLASS_PER_EPOCH, replace=True)
            sampled.extend((int(i), class_id) for i in ids)
        rng.shuffle(sampled)
        finite_batches, batch_losses, grad_norms = 0, [], []
        for start in range(0, len(sampled), BATCH_SIZE):
            batch = sampled[start:start + BATCH_SIZE]
            bx = x_fit[[item[0] for item in batch]]
            by = np.asarray([item[1] for item in batch], dtype=np.int64)
            with tf.GradientTape() as tape:
                logits = model(bx, training=True)
                loss = tf.reduce_mean(loss_fn(by, logits))
            gradients = tape.gradient(loss, model.trainable_variables)
            if not math.isfinite(float(loss.numpy())) or any(g is None or not np.isfinite(g.numpy()).all() for g in gradients):
                raise ValueError("nonfinite training loss or gradient")
            norm = float(tf.linalg.global_norm(gradients).numpy())
            if not math.isfinite(norm):
                raise ValueError("nonfinite gradient norm")
            optimizer.apply_gradients(zip(gradients, model.trainable_variables))
            finite_batches += 1
            batch_losses.append(float(loss.numpy()))
            grad_norms.append(norm)
        max_gradient = max(max_gradient, max(grad_norms, default=0.0))
        fit_logits = model(x_fit, training=False).numpy()
        stop_logits = model(x_stop, training=False).numpy()
        fit_loss = _cross_entropy(model, x_fit, y_fit)
        stop_loss = _cross_entropy(model, x_stop, y_stop)
        history.append({
            "epoch": epoch + 1, "finite_gradient_batches": finite_batches,
            "balanced_batch_cross_entropy": float(np.mean(batch_losses)),
            "fit_cross_entropy": fit_loss, "stop_cross_entropy": stop_loss,
            "fit_accuracy": _accuracy(fit_logits, y_fit),
            "stop_accuracy": _accuracy(stop_logits, y_stop),
            "mean_gradient_global_norm": float(np.mean(grad_norms)),
        })
        if stop_loss < best_loss - 1e-6:
            best_loss = stop_loss
            best_epoch = epoch + 1
            best_weights = [np.array(value, copy=True) for value in model.get_weights()]
            stale = 0
        else:
            stale += 1
        if stale >= PATIENCE:
            break
    if best_weights is None:
        raise ValueError("no train-only stopping checkpoint selected")
    model.set_weights(best_weights)
    elapsed = time.monotonic() - started
    selected_fit_logits = model(x_fit, training=False).numpy()
    selected_stop_logits = model(x_stop, training=False).numpy()
    selected_fit_loss = _cross_entropy(model, x_fit, y_fit)
    selected_stop_loss = _cross_entropy(model, x_stop, y_stop)

    all_probs: list[np.ndarray | None] = []
    all_logits = {}
    for row in rows:
        feature = feature_map[row["capture_id"]]
        if feature is None:
            all_probs.append(None)
            continue
        logits = np.asarray(model(feature[None, :], training=False).numpy()[0], dtype=np.float32)
        all_logits[row["capture_id"]] = logits
        all_probs.append(_softmax(logits[None, :])[0])

    unknown_cal = [i for i, row in enumerate(rows)
                   if row["role"] == "unknown_calibration" and all_probs[i] is not None]
    if not unknown_cal:
        raise ValueError("no usable unknown calibration features")
    calibration_scores = [float(np.max(all_probs[i][:len(KNOWN)])) for i in unknown_cal]
    threshold = float(threshold_above_max(calibration_scores))
    metrics = _evaluate(rows, all_probs, threshold)

    # Publish calibration-only acceptance tradeoffs at thresholds induced by
    # observed binary32 confidence scores. Development never selects a rule.
    known_cal = [i for i, row in enumerate(rows)
                 if row["role"] == "threshold_calibration" and all_probs[i] is not None]
    cal_rows = known_cal + unknown_cal
    cal_scores = sorted({float(np.max(all_probs[i][:len(KNOWN)])) for i in cal_rows})
    curve = []
    for candidate in cal_scores:
        known_accepted = sum(accepts(float(np.max(all_probs[i][:len(KNOWN)])), candidate) and
                             int(np.argmax(all_probs[i])) < len(KNOWN) for i in known_cal)
        unknown_groups: dict[str, bool] = {}
        for i in unknown_cal:
            group = rows[i].get("source_group") or rows[i]["capture_id"]
            false = accepts(float(np.max(all_probs[i][:len(KNOWN)])), candidate) and \
                    int(np.argmax(all_probs[i])) < len(KNOWN)
            unknown_groups[group] = unknown_groups.get(group, False) or bool(false)
        curve.append({
            "threshold_binary32": float(np.float32(candidate)),
            "known_calibration_correct_or_known_accepts": int(known_accepted),
            "known_calibration_rows": len(known_cal),
            "unknown_calibration_false_groups": sum(unknown_groups.values()),
            "unknown_calibration_groups": len(unknown_groups),
        })

    predictions = []
    for i, row in enumerate(rows):
        prob = all_probs[i]
        prediction = {
            "capture_id": row["capture_id"], "capture_set": row["capture_set"],
            "role": row["role"], "label": row["label"],
            "source_group": row.get("source_group"),
            "fomo_cluster_count": row.get("fomo_cluster_count"),
            "usable": prob is not None,
        }
        if prob is not None:
            pred = int(np.argmax(prob))
            score = float(np.max(prob[:len(KNOWN)]))
            prediction.update({
                "class_probabilities": [float(v) for v in prob],
                "predicted_label": CLASSES[pred], "max_known_probability": score,
                "accepted_known": bool(pred < len(KNOWN) and accepts(score, threshold)),
            })
        predictions.append(prediction)
    predictions_path.write_text("".join(json.dumps(row, sort_keys=True, allow_nan=False) + "\n"
                                                for row in predictions), encoding="utf-8")
    np.savez_compressed(weights_path, kernel=np.asarray(model.get_weights()[0], dtype=np.float32),
                        bias=np.asarray(model.get_weights()[1], dtype=np.float32))

    report = {
        "schema": "v6-camera-pretrained-face-embedding-head-pilot-1",
        "status": "complete_single_gradient_pilot",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "future_final_used": False, "ready_for_final_build": False,
        "parent_input_experiment": "work/project-research-v5/runs/camera_corrected_input_v2/INPUTS.json",
        "parent_embedding_result": "work/project-research-v5/runs/camera_corrected_face_encoder_v2/RESULT.json",
        "input_contract": {
            "coordinate_space": "camera_xywh_px", "roi_side_camera_px": 120,
            "encoder_input_hwc": [112, 112, 3], "encoder_feature_dim": 128,
            "fomo_score_gate": "strict_gt_float32_0.5", "usable_cluster_count": 1,
            "embedding_normalization": "L2 before dense head",
        },
        "encoder": {
            "model_sha256": "",
            "status": "frozen_host_face_pretrained_MobileFaceNet_diagnostic",
        },
        "head": {
            "type": "single_dense_six_way_logits", "trainable_parameters": 128 * 6 + 6,
            "class_order": list(CLASSES), "weights_path": str(weights_path.relative_to(ROOT)),
            "weights_sha256": sha256_file(weights_path),
        },
        "training": {
            "seed": seed, "group_split_seed": split_seed,
            "fit_rows": len(fit_rows), "stopping_rows": len(stop_rows),
            "fit_groups_by_class": {label: len({r["source_group"] for r in fit_rows if r["label"] == label}) for label in CLASSES},
            "stopping_groups_by_class": {label: len({r["source_group"] for r in stop_rows if r["label"] == label}) for label in CLASSES},
            "fit_stop_group_overlap": 0, "balanced_samples_per_class_per_epoch": SAMPLES_PER_CLASS_PER_EPOCH,
            "batch_size": BATCH_SIZE, "epochs_run": len(history), "selected_epoch": best_epoch,
            "finite_gradient_batches": sum(r["finite_gradient_batches"] for r in history),
            "maximum_gradient_global_norm": max_gradient,
            "initial_fit_cross_entropy": initial_fit_loss, "selected_fit_cross_entropy": selected_fit_loss,
            "initial_stop_cross_entropy": initial_stop_loss, "selected_stop_cross_entropy": selected_stop_loss,
            "initial_fit_accuracy": initial_fit_accuracy, "selected_fit_accuracy": _accuracy(selected_fit_logits, y_fit),
            "initial_stop_accuracy": initial_stop_accuracy, "selected_stop_accuracy": _accuracy(selected_stop_logits, y_stop),
            "elapsed_seconds": elapsed, "learning_curve": history,
        },
        "calibration": {
            "threshold_rule": "one representable binary32 step above max unknown-calibration max-known probability; >=",
            "threshold_binary32": threshold,
            "known_calibration_usable_rows": len(known_cal),
            "unknown_calibration_usable_rows": len(unknown_cal),
            "unknown_calibration_groups": len({rows[i].get("source_group") or rows[i]["capture_id"] for i in unknown_cal}),
            "threshold_curve": curve,
        },
        "evaluation": metrics,
        "private_predictions": str(predictions_path.relative_to(ROOT)),
        "prediction_sha256": sha256_file(predictions_path),
        "pilot_useful": False,
        "pilot_improved_over_frozen_gallery": bool(
            metrics["known_raw_top1_all_rows"] > 0.30 and
            metrics["known_correct_acceptance_all_rows"] > 0.06),
        "candidate_application_deployable": False,
    }
    # Bind exact source model hash from its diagnostic result after the run.
    encoder_result = json.loads((STUDY / "runs/camera_corrected_face_encoder_v2/RESULT.json").read_text())
    report["encoder"]["model_sha256"] = encoder_result["models"]["float"]["model"]["sha256"]
    result_path.write_text(json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--split-seed", type=int)
    parser.add_argument("--run-name", default="camera_pretrained_embedding_head_v1")
    args = parser.parse_args()
    value = run(args.seed, args.split_seed, args.run_name)
    print(json.dumps({
        "status": value["status"], "seed": value["training"]["seed"],
        "split_seed": value["training"]["group_split_seed"],
        "training": {k: v for k, v in value["training"].items() if k != "learning_curve"},
        "calibration": {k: v for k, v in value["calibration"].items() if k != "threshold_curve"},
        "evaluation": value["evaluation"], "pilot_useful": value["pilot_useful"],
    }, sort_keys=True))
