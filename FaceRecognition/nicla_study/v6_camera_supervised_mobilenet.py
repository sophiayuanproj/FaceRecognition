"""Single camera-matched MobileNetV1 five-known-plus-OTHER pilot.

Uses the corrected FOMO-centered camera ROI and train-role source groups only
for fitting and internal stopping. Threshold calibration and development
roles remain evaluation-only. The classifier is a host research candidate,
not a Nicla-qualified application.
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
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "FaceRecognition/firmware"))
from nicla_study import d_campaign as d
from nicla_study import v6_camera_crop_features48 as replay
from nicla_study.fs import sha256_file
from nicla_study.state import resource_lease
from nicla_study.v5_native_rgb import native_rgb
from open_set_decision import threshold_above_max


STUDY = ROOT / "work/project-research-v5"
INDEX_PATH = STUDY / "runs/camera_corrected_input_v1/CAPTURE_INDEX.jsonl"
INPUT_RECEIPT = STUDY / "runs/camera_corrected_input_v1/INPUTS.json"
OUT = STUDY / "runs/camera_supervised_mobilenet_v1"
PRIVATE = STUDY / "private/camera_supervised_mobilenet_v1"
RESULT = OUT / "RESULT.json"
WEIGHTS = PRIVATE / "mobilenet_classifier_weights.npz"
PREDICTIONS = PRIVATE / "PREDICTIONS.jsonl"
SEED = 56102
ALPHA = 0.1
ROI_SIDE = 120
INPUT_SIDE = 96
LEARNING_RATE = 0.0003
UNFREEZE_TOP_FRACTION = 0.25
EPOCHS_MAX = 20
PATIENCE = 4
KNOWN = tuple(d.KNOWN)
SIX = tuple(d.SIX)


def _softmax(logits: np.ndarray) -> np.ndarray:
    values = np.asarray(logits, dtype=np.float32)
    if values.ndim != 2 or values.shape[1] != 6 or not np.isfinite(values).all():
        raise ValueError("classifier logits failed finite six-class contract")
    shifted = values - np.max(values, axis=1, keepdims=True)
    exp = np.exp(shifted.astype(np.float64))
    probabilities = exp / exp.sum(axis=1, keepdims=True)
    return probabilities.astype(np.float32)


def _group_split(rows: list[dict]) -> tuple[list[int], list[int]]:
    rng = np.random.default_rng(SEED ^ 0xC1A55)
    fit, stop = [], []
    for label in KNOWN:
        groups: dict[str, list[int]] = {}
        for i, row in enumerate(rows):
            if row["label_id"] == label:
                groups.setdefault(str(row["source_group"]), []).append(i)
        names = sorted(groups)
        if len(names) < 3:
            raise ValueError("too few distinct train source groups for " + label)
        rng.shuffle(names)
        stop.extend(groups[names[0]])
        for name in names[1:]:
            fit.extend(groups[name])
    unknown_groups: dict[str, list[int]] = {}
    for i, row in enumerate(rows):
        if row["identity_partition"] == "unknown":
            unknown_groups.setdefault(str(row["source_group"]), []).append(i)
    names = sorted(unknown_groups)
    if len(names) < 13:
        raise ValueError("OTHER fitting requires at least 13 distinct train groups")
    rng.shuffle(names)
    for name in names[:3]:
        stop.extend(unknown_groups[name])
    for name in names[3:]:
        fit.extend(unknown_groups[name])
    fit_groups = {(rows[i]["label_id"], rows[i]["source_group"]) for i in fit}
    stop_groups = {(rows[i]["label_id"], rows[i]["source_group"]) for i in stop}
    if set(fit) & set(stop) or fit_groups & stop_groups:
        raise AssertionError("train and stopping groups overlap")
    return sorted(fit), sorted(stop)


def _cross_entropy(logits: np.ndarray, rows: list[dict]) -> float:
    targets = np.asarray([SIX.index(row["label_id"]) for row in rows], dtype=np.int64)
    losses = d.tf.keras.losses.sparse_categorical_crossentropy(
        targets, logits, from_logits=True).numpy()
    return float(np.mean(losses))


def _accuracy(logits: np.ndarray, rows: list[dict]) -> dict:
    predicted = np.argmax(logits, axis=1)
    target = np.asarray([SIX.index(row["label_id"]) for row in rows], dtype=np.int64)
    per_class = {}
    for label in SIX:
        indices = [i for i, row in enumerate(rows) if row["label_id"] == label]
        if indices:
            per_class[label] = {"rows": len(indices),
                                "correct": int(np.sum(predicted[indices] == SIX.index(label)))}
    return {"correct": int(np.sum(predicted == target)), "rows": len(rows),
            "accuracy": float(np.mean(predicted == target)) if len(target) else None,
            "per_class": per_class}


def _threshold_curve(cal_known: np.ndarray, cal_unknown: np.ndarray) -> list[dict]:
    scores = np.concatenate([cal_known, cal_unknown]).astype(np.float32)
    thresholds = np.unique(np.quantile(scores, np.linspace(0.0, 1.0, 21)).astype(np.float32))
    return [{"threshold": float(threshold),
             "known_calibration_accepted": int(np.sum(cal_known >= threshold)),
             "known_calibration_rows": int(len(cal_known)),
             "unknown_calibration_accepted": int(np.sum(cal_unknown >= threshold)),
             "unknown_calibration_rows": int(len(cal_unknown))}
            for threshold in thresholds]


def _evaluate(rows: list[dict], probabilities: np.ndarray, valid: np.ndarray,
              threshold: float) -> dict:
    pred6 = np.argmax(probabilities, axis=1)
    known_score = np.max(probabilities[:, :5], axis=1)
    pred_known = np.argmax(probabilities[:, :5], axis=1)
    accepted = valid & (pred6 < 5) & (known_score >= np.float32(threshold))
    known_ids = [i for i, row in enumerate(rows) if row["label"] in KNOWN and row["role"] == "development"]
    unknown_ids = [i for i, row in enumerate(rows) if row["role"] == "unknown_development"]
    per_person = {}
    for label in KNOWN:
        ids = [i for i in known_ids if rows[i]["label"] == label]
        per_person[label] = {
            "rows": len(ids), "fomo_valid": int(sum(valid[i] for i in ids)),
            "raw_top1_correct_all_rows": int(sum(valid[i] and pred6[i] == SIX.index(label) for i in ids)),
            "known_identity_argmax_correct_all_rows": int(sum(valid[i] and pred_known[i] == KNOWN.index(label) for i in ids)),
            "correct_accepted": int(sum(accepted[i] and pred6[i] == SIX.index(label) for i in ids)),
            "wrong_known_accepts": int(sum(accepted[i] and pred6[i] != SIX.index(label) for i in ids)),
        }
    groups: dict[str, list[int]] = {}
    for i in unknown_ids:
        groups.setdefault(str(rows[i].get("source_group") or rows[i]["capture_id"]), []).append(i)
    false_unknown_groups = sum(any(accepted[i] for i in ids) for ids in groups.values())
    return {
        "threshold_binary32": float(np.float32(threshold)),
        "threshold_rule": "one representable binary32 step above the maximum unknown-calibration max-known probability; six-way OTHER prediction required for acceptance",
        "known_development_rows": len(known_ids),
        "known_fomo_valid_rows": int(sum(valid[i] for i in known_ids)),
        "known_raw_top1_all_rows": float(sum(valid[i] and pred6[i] == SIX.index(rows[i]["label"])
                                               for i in known_ids) / max(1, len(known_ids))),
        "known_identity_argmax_top1_all_rows": float(sum(valid[i] and pred_known[i] == KNOWN.index(rows[i]["label"])
                                                          for i in known_ids) / max(1, len(known_ids))),
        "known_correct_accepted": int(sum(accepted[i] and pred6[i] == SIX.index(rows[i]["label"])
                                          for i in known_ids)),
        "known_correct_acceptance_all_rows": float(sum(accepted[i] and pred6[i] == SIX.index(rows[i]["label"])
                                                        for i in known_ids) / max(1, len(known_ids))),
        "known_coverage_all_rows": float(sum(accepted[i] for i in known_ids) / max(1, len(known_ids))),
        "known_wrong_accepted": int(sum(accepted[i] and pred6[i] != SIX.index(rows[i]["label"])
                                         for i in known_ids)),
        "unknown_development_rows": len(unknown_ids),
        "unknown_development_groups": len(groups),
        "unknown_fomo_valid_rows": int(sum(valid[i] for i in unknown_ids)),
        "unknown_false_accepted_rows": int(sum(accepted[i] for i in unknown_ids)),
        "unknown_false_accepted_groups": int(false_unknown_groups),
        "unknown_zero_event_upper95_for_30": 1.0 - 0.05 ** (1.0 / 30.0),
        "per_person": per_person,
    }


def run() -> dict:
    if RESULT.exists() or WEIGHTS.exists() or PREDICTIONS.exists():
        raise FileExistsError("camera supervised MobileNet output exists; use a new versioned namespace")
    cv2.setNumThreads(2)
    rows = [json.loads(line) for line in INDEX_PATH.read_text().splitlines() if line.strip()]
    replay_rows = replay.load_sets()
    if len(rows) != 255 or len(replay_rows) != len(rows):
        raise ValueError("expected exactly 255 frozen raw camera frames")
    if [row["request"]["capture_id"] for row in replay_rows] != [row["capture_id"] for row in rows]:
        raise ValueError("raw frame and corrected input index order differ")

    images = np.zeros((len(rows), INPUT_SIDE, INPUT_SIDE, 1), dtype=np.uint8)
    valid = np.zeros(len(rows), dtype=bool)
    for i, (row, raw_row) in enumerate(zip(rows, replay_rows)):
        if not row["usable_for_identity"]:
            continue
        if int(row["roi_side_camera_px"]) != ROI_SIDE:
            raise ValueError("unexpected corrected camera ROI side")
        pixels = replay.rgb565_array((raw_row["folder"] / "camera_rgb565.bin").read_bytes())
        rgb = native_rgb(pixels, roi=tuple(row["roi_xywh_camera_px"]),
                         width=INPUT_SIDE, height=INPUT_SIDE)
        if rgb.shape != (INPUT_SIDE, INPUT_SIDE, 3) or not np.any(rgb):
            raise ValueError("usable corrected camera ROI failed RGB input quality check: " + row["capture_id"])
        images[i, :, :, 0] = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
        valid[i] = True

    candidate_rows, candidate_images = [], []
    for i, row in enumerate(rows):
        if not valid[i] or row["role"] not in ("train", "other_train"):
            continue
        label = row["label"]
        if (row["role"] == "train" and label not in KNOWN) or \
                (row["role"] == "other_train" and label != "OTHER"):
            raise ValueError("camera train role and label disagree: " + row["capture_id"])
        candidate_rows.append({"record_id": row["capture_id"], "capture_id": row["capture_id"],
                               "label": label, "label_id": label,
                               "identity_partition": "unknown" if label == "OTHER" else "known",
                               "identity_id": row.get("source_group") or row["capture_id"],
                               "source_group": row.get("source_group") or row["capture_id"],
                               "role": row["role"]})
        candidate_images.append(images[i])
    fit_ids, stop_ids = _group_split(candidate_rows)
    candidate_images = np.asarray(candidate_images, dtype=np.uint8)
    fit_rows = [candidate_rows[i] for i in fit_ids]
    stop_rows = [candidate_rows[i] for i in stop_ids]
    fit_images, stop_images = candidate_images[fit_ids], candidate_images[stop_ids]
    if len({row["identity_id"] for row in fit_rows if row["identity_partition"] == "unknown"}) < 10:
        raise ValueError("balanced six-way training requires ten distinct OTHER fit groups")

    model, embedding_model, model_info = d.build_classifier(
        ROOT, ALPHA, True, 6, SEED, "v6_camera_supervised_mobilenet_a010_seed56102")
    trainability = d.unfreeze_top_fraction(model, UNFREEZE_TOP_FRACTION)
    d.compile_classifier(model, LEARNING_RATE, 6, outlier_exposure=False)
    initial_fit_logits = d.evaluate_logits(model, fit_images)
    initial_stop_logits = d.evaluate_logits(model, stop_images)
    initial_fit_loss = _cross_entropy(initial_fit_logits, fit_rows)
    initial_stop_loss = _cross_entropy(initial_stop_logits, stop_rows)
    initial_fit_accuracy = _accuracy(initial_fit_logits, fit_rows)
    initial_stop_accuracy = _accuracy(initial_stop_logits, stop_rows)

    started = time.monotonic()
    curve, best_weights, best_loss, best_epoch, stale = [], None, float("inf"), 0, 0
    with resource_lease(d.study_paths(ROOT)["study"], "heavy_training",
                        owner="v6_camera_supervised_mobilenet", timeout_s=0):
        for epoch in range(EPOCHS_MAX):
            epoch_result = d.train_epoch(model, fit_rows, fit_images, "survface_96",
                                         SEED + epoch * 1009, True, 6, False)
            stop_logits = d.evaluate_logits(model, stop_images)
            stop_loss = _cross_entropy(stop_logits, stop_rows)
            row = {"epoch": epoch + 1, **epoch_result, "stopping_cross_entropy": stop_loss,
                   "stopping_six_way_accuracy": _accuracy(stop_logits, stop_rows)["accuracy"]}
            curve.append(row)
            if stop_loss < best_loss - 1e-7:
                best_loss, best_epoch, stale = stop_loss, epoch + 1, 0
                best_weights = [np.array(value, copy=True) for value in model.get_weights()]
            else:
                stale += 1
            if stale >= PATIENCE:
                break
    elapsed = time.monotonic() - started
    if best_weights is None:
        raise ValueError("supervised pilot produced no selectable checkpoint")
    model.set_weights(best_weights)

    selected_fit_logits = d.evaluate_logits(model, fit_images)
    selected_stop_logits = d.evaluate_logits(model, stop_images)
    selected_fit_loss = _cross_entropy(selected_fit_logits, fit_rows)
    selected_stop_loss = _cross_entropy(selected_stop_logits, stop_rows)
    selected_fit_accuracy = _accuracy(selected_fit_logits, fit_rows)
    selected_stop_accuracy = _accuracy(selected_stop_logits, stop_rows)

    all_logits = np.zeros((len(rows), 6), dtype=np.float32)
    valid_indices = np.flatnonzero(valid)
    all_logits[valid_indices] = d.evaluate_logits(model, images[valid_indices])
    probabilities = np.zeros_like(all_logits)
    probabilities[valid_indices] = _softmax(all_logits[valid_indices])
    max_known = np.max(probabilities[:, :5], axis=1)
    known_cal_ids = [i for i, row in enumerate(rows) if row["role"] == "threshold_calibration" and valid[i]]
    unknown_cal_ids = [i for i, row in enumerate(rows) if row["role"] == "unknown_calibration" and valid[i]]
    if not known_cal_ids or not unknown_cal_ids:
        raise ValueError("frozen known and unknown calibration rows are required")
    threshold = threshold_above_max(max_known[unknown_cal_ids].tolist())
    evaluation = _evaluate(rows, probabilities, valid, threshold)
    evaluation["known_threshold_calibration_usable_rows"] = len(known_cal_ids)
    evaluation["unknown_threshold_calibration_rows"] = sum(row["role"] == "unknown_calibration" for row in rows)
    evaluation["unknown_threshold_calibration_usable_rows"] = len(unknown_cal_ids)
    evaluation["unknown_calibration_false_accepts"] = int(np.sum(max_known[unknown_cal_ids] >= np.float32(threshold)))
    evaluation["threshold_calibration_tradeoff"] = _threshold_curve(
        max_known[known_cal_ids], max_known[unknown_cal_ids])

    PRIVATE.mkdir(parents=True, exist_ok=True)
    OUT.mkdir(parents=True, exist_ok=True)
    weight_arrays = {"weight_%03d" % i: np.asarray(value) for i, value in enumerate(model.get_weights())}
    np.savez_compressed(WEIGHTS, **weight_arrays)
    prediction_rows = []
    for i, row in enumerate(rows):
        prediction_rows.append({"capture_id": row["capture_id"], "role": row["role"],
                                "label": row["label"], "source_group": row.get("source_group"),
                                "fomo_valid": bool(valid[i]),
                                "logits": all_logits[i].tolist() if valid[i] else None,
                                "probabilities": probabilities[i].tolist() if valid[i] else None})
    PREDICTIONS.write_text("".join(json.dumps(row, sort_keys=True, allow_nan=False) + "\n"
                                             for row in prediction_rows), encoding="utf-8")
    train_rows_digest = hashlib.sha256("\n".join(row["capture_id"] for row in candidate_rows).encode()).hexdigest()
    result = {
        "schema": "v6-camera-supervised-mobilenet-pilot-1",
        "status": "complete_single_gradient_pilot",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "parent_input_experiment": str(INPUT_RECEIPT.relative_to(ROOT)),
        "parent_input_sha256": sha256_file(INPUT_RECEIPT),
        "source_script_sha256": sha256_file(Path(__file__)),
        "train_role_capture_list_sha256": train_rows_digest,
        "student": {"architecture": "MobileNetV1 alpha 0.1 grayscale96; pretrained EI backbone; 64-D embedding; six-way logits [K01..K05,OTHER]",
                    "initial_checkpoint_sha256": d.EI_HASHES[ALPHA],
                    "input_contract": "RGB565->camera_xywh corrected 120px FOMO ROI->native RGB888 resize96->gray8->float32/255",
                    "weights_path": str(WEIGHTS.relative_to(ROOT)),
                    "weights_sha256": sha256_file(WEIGHTS),
                    "parameter_count": int(model.count_params()),
                    "trainable_parameter_count": int(sum(np.prod(value.shape) for value in model.trainable_variables))},
        "training": {"seed": SEED, "alpha": ALPHA, "learning_rate": LEARNING_RATE,
                     "unfreeze_top_fraction": UNFREEZE_TOP_FRACTION,
                     "trainability": trainability, "epochs_max": EPOCHS_MAX,
                     "selected_epoch": best_epoch, "epochs_run": len(curve), "patience": PATIENCE,
                     "fit_rows": len(fit_rows), "stopping_rows": len(stop_rows),
                     "fit_groups_by_label": {label: sum(row["label_id"] == label for row in fit_rows) for label in SIX},
                     "other_fit_groups": len({row["identity_id"] for row in fit_rows if row["identity_partition"] == "unknown"}),
                     "fit_stop_group_overlap": 0,
                     "initial_train_cross_entropy": initial_fit_loss,
                     "selected_train_cross_entropy": selected_fit_loss,
                     "initial_stopping_cross_entropy": initial_stop_loss,
                     "selected_stopping_cross_entropy": selected_stop_loss,
                     "initial_train_six_way_accuracy": initial_fit_accuracy,
                     "selected_train_six_way_accuracy": selected_fit_accuracy,
                     "initial_stopping_six_way_accuracy": initial_stop_accuracy,
                     "selected_stopping_six_way_accuracy": selected_stop_accuracy,
                     "curve": curve, "elapsed_seconds": elapsed},
        "evaluation": evaluation,
        "private_predictions": str(PREDICTIONS.relative_to(ROOT)),
        "limits": ["One pilot seed only; no ten-root learned comparison is claimed.",
                   "Host-generated 96px grayscale tensors have not passed exact Nicla tensor/device parity.",
                   "FOMO detection misses and ambiguous crops remain in end-to-end denominators.",
                   "Thirty unknown development groups cannot certify a 1% upper risk.",
                   "No Nicla deployment, resource, startup, or cold-start qualification is claimed."],
        "future_final_used": False,
        "ready_for_final_build": False,
    }
    RESULT.write_text(json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    return result


if __name__ == "__main__":
    print(json.dumps(run(), indent=2, sort_keys=True, allow_nan=False))
