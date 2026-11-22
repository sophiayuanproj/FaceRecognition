"""Finite camera-ROI-size diagnostic for the host-only SFace teacher.

Only the camera-pixel crop side varies. FOMO gating, 112px SFace input,
train-group gallery construction, and calibration roles stay fixed.
"""

from __future__ import annotations

import json
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
import sys

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "FaceRecognition"))
sys.path.insert(0, str(ROOT / "FaceRecognition/firmware"))
from camera_geometry import centered_roi_clamped, fomo_center_camera_px
from identity_contracts import quality_decision
from nicla_study import v6_camera_crop_features48_v2 as replay
from nicla_study.fs import sha256_file
from nicla_study.v5_native_rgb import native_rgb
from open_set_decision import accepts, threshold_above_max

STUDY = ROOT / "work/project-research-v5"
MODEL_DIR = ROOT / "work/project-finish-v4/models/pretrained/sface"
SFACE_PATH = MODEL_DIR / "face_recognition_sface_2021dec.onnx"
SFACE_SHA256 = "0ba9fbfa01b5270c96627c4ef784da859931e02f04419c829e83484087c34e79"
INPUTS = STUDY / "runs/camera_corrected_input_v2/INPUTS.json"
OUT = STUDY / "runs/camera_sface_roi_sweep_v1"
PRIVATE = STUDY / "private/camera_sface_roi_sweep_v1"
KNOWN = ("K01", "K02", "K03", "K04", "K05")
ROI_SIDES = (48, 64, 80, 96, 120)
DETECTION_THRESHOLD = 0.5
TENSOR_SIDE = 112


def json_default(value):
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError("not JSON serializable: " + type(value).__name__)


def normalize(value: np.ndarray) -> np.ndarray:
    vector = np.asarray(value, dtype=np.float32).reshape(-1)
    length = float(np.linalg.norm(vector))
    if vector.shape != (128,) or not np.isfinite(vector).all() or length <= 1e-8:
        raise ValueError("SFace output is not finite, nonzero, and 128-D")
    return vector / length


def summarize(rows: list[dict], embeddings: np.ndarray, valid: np.ndarray, side: int) -> dict:
    grouped: dict[str, dict[str, list[np.ndarray]]] = defaultdict(lambda: defaultdict(list))
    for i, row in enumerate(rows):
        if row["role"] == "train" and row["label"] in KNOWN and valid[i]:
            grouped[row["label"]][row.get("source_group") or row["capture_id"]].append(embeddings[i])
    gallery = {}
    for label in KNOWN:
        prototypes = [normalize(np.mean(np.stack(values), axis=0)) for values in grouped[label].values()]
        if not prototypes:
            raise ValueError("no SFace camera training prototypes for " + label)
        gallery[label] = normalize(np.mean(np.stack(prototypes), axis=0))
    scores = np.zeros((len(rows), 5), dtype=np.float32)
    prediction = [None] * len(rows)
    for i in np.flatnonzero(valid):
        vector = normalize(embeddings[i])
        scores[i] = np.asarray([np.dot(vector, gallery[label]) for label in KNOWN], dtype=np.float32)
        top = int(np.argmax(scores[i]))
        prediction[i] = {"top1": KNOWN[top], "score": float(scores[i, top]), "scores": scores[i].tolist()}
    max_score = scores.max(axis=1)
    unknown_cal = [i for i, row in enumerate(rows) if row["role"] == "unknown_calibration" and valid[i]]
    if not unknown_cal:
        raise ValueError("no usable unknown calibration row for ROI side " + str(side))
    threshold = threshold_above_max(max_score[unknown_cal].tolist())
    accepted = np.asarray([bool(valid[i] and accepts(float(max_score[i]), threshold))
                           for i in range(len(rows))], dtype=bool)
    known_dev = [i for i, row in enumerate(rows) if row["role"] == "development" and row["label"] in KNOWN]
    known_cal = [i for i, row in enumerate(rows) if row["role"] == "threshold_calibration" and row["label"] in KNOWN]
    unknown_dev = [i for i, row in enumerate(rows) if row["role"] == "unknown_development"]
    groups: dict[str, list[int]] = defaultdict(list)
    for i in unknown_dev:
        groups[str(rows[i].get("source_group") or rows[i]["capture_id"])].append(i)
    per_person = {}
    for label in KNOWN:
        ids = [i for i in known_dev if rows[i]["label"] == label]
        per_person[label] = {"rows": len(ids), "fomo_valid": int(np.sum(valid[ids])),
            "raw_top1_correct_all_rows": sum(valid[i] and prediction[i]["top1"] == label for i in ids),
            "correct_accepted": sum(accepted[i] and prediction[i]["top1"] == label for i in ids),
            "wrong_known_accepted": sum(accepted[i] and prediction[i]["top1"] != label for i in ids)}
    correct = sum(valid[i] and prediction[i]["top1"] == rows[i]["label"] for i in known_dev)
    return {"roi_side_camera_px": side, "tensor_hwc": [TENSOR_SIDE, TENSOR_SIDE, 3],
        "train_groups_per_known": {label: len(grouped[label]) for label in KNOWN},
        "known_development_rows": len(known_dev), "known_fomo_valid_rows": int(np.sum(valid[known_dev])),
        "known_raw_top1_correct_all_rows": correct, "known_raw_top1_all_rows": correct / max(1, len(known_dev)),
        "known_correct_accepted": sum(accepted[i] and prediction[i]["top1"] == rows[i]["label"] for i in known_dev),
        "known_correct_acceptance_all_rows": sum(accepted[i] and prediction[i]["top1"] == rows[i]["label"] for i in known_dev) / max(1, len(known_dev)),
        "known_coverage_all_rows": float(np.sum(accepted[known_dev]) / max(1, len(known_dev))),
        "known_wrong_accepted": sum(accepted[i] and prediction[i]["top1"] != rows[i]["label"] for i in known_dev),
        "known_threshold_calibration_rows": len(known_cal), "known_threshold_calibration_usable_rows": int(np.sum(valid[known_cal])),
        "known_threshold_calibration_accepted": int(np.sum(accepted[known_cal])),
        "unknown_calibration_groups": len({row.get("source_group") or row["capture_id"] for row in rows if row["role"] == "unknown_calibration"}),
        "unknown_calibration_usable_rows": len(unknown_cal), "unknown_calibration_max_score": float(max(max_score[i] for i in unknown_cal)),
        "threshold_binary32": float(np.float32(threshold)),
        "unknown_development_rows": len(unknown_dev), "unknown_development_groups": len(groups),
        "unknown_fomo_valid_rows": int(np.sum(valid[unknown_dev])),
        "unknown_false_accepted_rows": int(np.sum(accepted[unknown_dev])),
        "unknown_false_accepted_groups": sum(any(accepted[i] for i in ids) for ids in groups.values()),
        "unknown_zero_event_upper95_for_30": 1.0 - 0.05 ** (1.0 / 30.0),
        "per_person": per_person, "_predictions": prediction}


def run() -> dict:
    if (OUT / "RESULT.json").exists():
        raise FileExistsError("SFace ROI sweep already exists; use a new versioned namespace")
    if sha256_file(SFACE_PATH) != SFACE_SHA256:
        raise ValueError("SFace checkpoint hash mismatch")
    input_receipt = json.loads(INPUTS.read_text())
    rows = [{**row, "capture_id": row["request"]["capture_id"],
             "source_group": row["source"].get("source_group")}
            for row in replay.load_sets()]
    if len(rows) != 255 or input_receipt.get("future_final_used") is not False:
        raise ValueError("frozen camera source or future-final provenance mismatch")
    recognizer = cv2.FaceRecognizerSF.create(str(SFACE_PATH), "")
    all_embeddings, all_valid, metrics, pred_rows, crop_records = {}, {}, {}, [], []
    for side in ROI_SIDES:
        embeddings = np.zeros((len(rows), 128), dtype=np.float32)
        valid = np.zeros(len(rows), dtype=bool)
        for i, row in enumerate(rows):
            if len(row["boxes"]) != 1:
                continue
            pixels = replay.rgb565_array((row["folder"] / "camera_rgb565.bin").read_bytes())
            rect, score = row["boxes"][0]
            center = fomo_center_camera_px(rect, 240, 240)
            roi = centered_roi_clamped(center, side, 240, 240)
            rgb = native_rgb(pixels, roi=roi, width=TENSOR_SIDE, height=TENSOR_SIDE)
            bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
            gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
            quality_ok, quality_reason = quality_decision(float(gray.mean()), float(gray.std()))
            crop_records.append({"capture_id": row["capture_id"], "roi_side_camera_px": side,
                "role": row["role"], "label": row["label"], "source_group": row["source_group"],
                "center_xy_camera_px": list(center), "roi_xywh_camera_px": list(roi),
                "fomo_score": float(score), "crop_mean_u8": float(gray.mean()),
                "crop_stdev_u8": float(gray.std()), "quality_ok": bool(quality_ok),
                "quality_reason": quality_reason})
            if quality_ok:
                embeddings[i] = normalize(recognizer.feature(bgr))
                valid[i] = True
        result = summarize(rows, embeddings, valid, side)
        predictions = result.pop("_predictions")
        metrics[str(side)] = result
        all_embeddings["roi_" + str(side)] = embeddings
        all_valid[str(side)] = valid
        for i, row in enumerate(rows):
            prediction = predictions[i]
            pred_rows.append({"capture_id": row["capture_id"], "roi_side_camera_px": side,
                "role": row["role"], "label": row["label"], "source_group": row["source_group"],
                "quality_status": "sface_valid" if valid[i] else
                    ("fomo_single_crop_quality_rejected" if len(row["boxes"]) == 1 else
                     ("multiple_clusters" if len(row["boxes"]) > 1 else "no_face")),
                "prediction": prediction})
    PRIVATE.mkdir(parents=True, exist_ok=True)
    OUT.mkdir(parents=True, exist_ok=True)
    emb_path = PRIVATE / "EMBEDDINGS.npz"
    np.savez_compressed(emb_path, **all_embeddings)
    records_path = PRIVATE / "ROI_RECORDS.jsonl"
    records_path.write_text("".join(json.dumps(row, sort_keys=True, allow_nan=False) + "\n"
                                               for row in crop_records), encoding="utf-8")
    predictions_path = PRIVATE / "PREDICTIONS.jsonl"
    predictions_path.write_text("".join(json.dumps(row, sort_keys=True, allow_nan=False) + "\n"
                                                   for row in pred_rows), encoding="utf-8")
    summary = {"schema": "v6-camera-sface-roi-sweep-1", "status": "complete_host_diagnostic",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "parent_input": "work/project-research-v5/runs/camera_corrected_input_v2/INPUTS.json",
        "parent_input_sha256": sha256_file(INPUTS), "capture_rows": len(rows),
        "teacher": {"model": "OpenCV Zoo SFace 2021dec", "sha256": SFACE_SHA256,
                    "pretraining_overlap": "unresolved", "host_only": True},
        "roi_sizes_are_camera_pixels_not_encoder_resolutions": True,
        "fomo_gate": {"threshold": DETECTION_THRESHOLD, "comparison": "strict_gt_float32",
                      "coordinate_space": "camera_xywh_px", "multiple_cluster_action": "UNCERTAIN"},
        "roi_sizes_camera_px": list(ROI_SIDES), "encoder_tensor_hwc": [112, 112, 3],
        "results": metrics, "private_embeddings": str(emb_path.relative_to(ROOT)),
        "private_roi_records": str(records_path.relative_to(ROOT)),
        "private_predictions": str(predictions_path.relative_to(ROOT)),
        "future_final_used": False, "ready_for_final_build": False,
        "limits": ["SFace is host-only; its aligned reference is not a Nicla candidate.",
                   "Each threshold uses unknown calibration only. Development rows were held out from crop selection.",
                   "The screen capture setup is one physical environment; unknown identity results do not establish the 1% risk target."]}
    (OUT / "RESULT.json").write_text(json.dumps(summary, indent=2, sort_keys=True,
                                                  allow_nan=False, default=json_default) + "\n", encoding="utf-8")
    return summary


if __name__ == "__main__":
    print(json.dumps(run(), indent=2, sort_keys=True, allow_nan=False, default=json_default))
