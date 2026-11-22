"""Finite corrected camera-ROI-size diagnostic for the frozen 48px encoder.

All variants share one 48x48 RGB encoder and the same strict float32 FOMO gate.
Only the fixed camera-pixel ROI side changes. Gallery fitting uses train-role
known rows; thresholds use unknown_calibration; development roles are held out.
"""

from __future__ import annotations

import json
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
import sys

import numpy as np
import tensorflow as tf

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "FaceRecognition"))
sys.path.insert(0, str(ROOT / "FaceRecognition/firmware"))
from camera_geometry import centered_roi_clamped, fomo_center_camera_px
from identity_contracts import strict_score_pass
from nicla_study import v6_camera_crop_features48_v2 as replay
from nicla_study.fs import sha256_file
from nicla_study.v5_native_rgb import native_rgb
from open_set_decision import accepts, threshold_above_max

STUDY = ROOT / "work/project-research-v5"
MODEL = STUDY / "runs/operator_fused_encoder48/int8.tflite"
INPUTS = STUDY / "runs/camera_corrected_input_v2/INPUTS.json"
OUT = STUDY / "runs/camera_roi_sweep_v2"
PRIVATE = STUDY / "private/camera_roi_sweep_v2"
KNOWN = ("K01", "K02", "K03", "K04", "K05")
ROI_SIDES = (48, 64, 80, 96, 120)
TENSOR_SIDE = 48
DETECTION_THRESHOLD = 0.5


def _json_default(value):
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError("not JSON serializable: " + type(value).__name__)


def _normalize(values: np.ndarray) -> np.ndarray:
    array = np.asarray(values, dtype=np.float32).reshape(-1)
    norm = float(np.linalg.norm(array))
    if array.shape != (128,) or not np.isfinite(array).all() or norm <= 1e-8:
        raise ValueError("encoder output is not finite, nonzero, and 128-D")
    return array / norm


def _embedding_model():
    it = tf.lite.Interpreter(model_path=str(MODEL), num_threads=2,
        experimental_op_resolver_type=tf.lite.experimental.OpResolverType.BUILTIN_REF)
    it.allocate_tensors()
    inp, out = it.get_input_details()[0], it.get_output_details()[0]
    if inp["shape"].tolist() != [1, 48, 48, 3] or inp["dtype"] != np.int8:
        raise ValueError("encoder input contract changed")
    if out["shape"].tolist() != [1, 128] or out["dtype"] != np.int8:
        raise ValueError("encoder output contract changed")
    in_scale, in_zero = inp["quantization"]
    if not in_scale or in_scale <= 0:
        raise ValueError("invalid encoder input quantization")
    out_scale, out_zero = out["quantization"]
    if not out_scale or out_scale <= 0:
        raise ValueError("invalid encoder output quantization")
    lut = np.clip(np.rint(((np.arange(256, dtype=np.float32) - 127.5) / 128.0) /
                          float(in_scale)) + int(in_zero), -128, 127).astype(np.int8)
    return it, inp, out, lut


def _summarize(rows: list[dict], embeddings: np.ndarray, valid: np.ndarray, side: int) -> dict:
    grouped: dict[str, dict[str, list[np.ndarray]]] = defaultdict(lambda: defaultdict(list))
    for i, row in enumerate(rows):
        if row["role"] == "train" and row["label"] in KNOWN and valid[i]:
            group = row.get("source_group") or row["capture_id"]
            grouped[row["label"]][group].append(embeddings[i])
    gallery = {}
    for label in KNOWN:
        prototypes = [_normalize(np.mean(np.stack(v), axis=0)) for v in grouped[label].values()]
        if not prototypes:
            raise ValueError("no train-role prototype for " + label)
        gallery[label] = _normalize(np.mean(np.stack(prototypes), axis=0))

    prediction_rows = []
    max_scores = np.zeros(len(rows), dtype=np.float32)
    for i, row in enumerate(rows):
        if not valid[i]:
            prediction_rows.append(None)
            continue
        vector = _normalize(embeddings[i])
        scores = np.asarray([np.dot(vector, gallery[label]) for label in KNOWN], dtype=np.float32)
        best = int(np.argmax(scores))
        max_scores[i] = scores[best]
        prediction_rows.append({"top1": KNOWN[best], "score": float(scores[best]),
                                "class_scores": scores.tolist()})

    unknown_cal = [i for i, row in enumerate(rows)
                   if row["role"] == "unknown_calibration" and valid[i]]
    if not unknown_cal:
        raise ValueError("unknown-calibration rows did not produce embeddings")
    threshold = threshold_above_max(max_scores[unknown_cal].tolist())
    accepted = np.asarray([bool(valid[i] and accepts(float(max_scores[i]), threshold))
                           for i in range(len(rows))], dtype=bool)
    known_dev = [i for i, row in enumerate(rows)
                 if row["role"] == "development" and row["label"] in KNOWN]
    known_cal = [i for i, row in enumerate(rows)
                 if row["role"] == "threshold_calibration" and row["label"] in KNOWN]
    unknown_dev = [i for i, row in enumerate(rows) if row["role"] == "unknown_development"]
    unknown_groups: dict[str, list[int]] = defaultdict(list)
    for i in unknown_dev:
        unknown_groups[str(rows[i].get("source_group") or rows[i]["capture_id"])].append(i)
    per_person = {}
    for label in KNOWN:
        ids = [i for i in known_dev if rows[i]["label"] == label]
        per_person[label] = {
            "rows": len(ids), "fomo_valid": sum(bool(valid[i]) for i in ids),
            "raw_top1_correct_all_rows": sum(valid[i] and prediction_rows[i]["top1"] == label for i in ids),
            "correct_accepted": sum(accepted[i] and prediction_rows[i]["top1"] == label for i in ids),
            "wrong_known_accepts": sum(accepted[i] and prediction_rows[i]["top1"] != label for i in ids),
        }
    known_correct = sum(valid[i] and prediction_rows[i]["top1"] == rows[i]["label"] for i in known_dev)
    unknown_false_rows = sum(bool(accepted[i]) for i in unknown_dev)
    return {
        "roi_side_camera_px": side, "tensor_hwc": [TENSOR_SIDE, TENSOR_SIDE, 3],
        "fit_group_counts": {label: len(grouped[label]) for label in KNOWN},
        "known_development_rows": len(known_dev), "known_fomo_valid_rows": sum(bool(valid[i]) for i in known_dev),
        "known_raw_top1_correct_all_rows": known_correct,
        "known_raw_top1_all_rows": known_correct / max(1, len(known_dev)),
        "known_correct_accepted": sum(accepted[i] and prediction_rows[i]["top1"] == rows[i]["label"] for i in known_dev),
        "known_correct_acceptance_all_rows": sum(accepted[i] and prediction_rows[i]["top1"] == rows[i]["label"] for i in known_dev) / max(1, len(known_dev)),
        "known_coverage_all_rows": sum(bool(accepted[i]) for i in known_dev) / max(1, len(known_dev)),
        "known_wrong_accepted": sum(accepted[i] and prediction_rows[i]["top1"] != rows[i]["label"] for i in known_dev),
        "unknown_calibration_rows": sum(row["role"] == "unknown_calibration" for row in rows),
        "unknown_calibration_groups": len({row.get("source_group") or row["capture_id"] for row in rows if row["role"] == "unknown_calibration"}),
        "unknown_calibration_usable_rows": len(unknown_cal),
        "unknown_calibration_max_score": float(max(max_scores[i] for i in unknown_cal)),
        "unknown_calibration_threshold_binary32": float(np.float32(threshold)),
        "known_threshold_calibration_rows": len(known_cal),
        "known_threshold_calibration_usable_rows": sum(bool(valid[i]) for i in known_cal),
        "known_threshold_calibration_accepted": sum(bool(accepted[i]) for i in known_cal),
        "threshold_rule": "one representable binary32 step above max unknown-calibration cosine; >= comparison",
        "unknown_development_rows": len(unknown_dev), "unknown_development_groups": len(unknown_groups),
        "unknown_fomo_valid_rows": sum(bool(valid[i]) for i in unknown_dev),
        "unknown_false_accepted_rows": int(unknown_false_rows),
        "unknown_false_accepted_groups": sum(any(bool(accepted[i]) for i in ids) for ids in unknown_groups.values()),
        "unknown_zero_event_upper95_for_30": 1.0 - 0.05 ** (1.0 / 30.0),
        "per_person": per_person,
        "_prediction_rows": prediction_rows,
    }


def run() -> dict:
    if (OUT / "RESULT.json").exists() or (PRIVATE / "PREDICTIONS.jsonl").exists():
        raise FileExistsError("ROI sweep exists; choose a new versioned namespace")
    input_receipt = json.loads(INPUTS.read_text())
    if input_receipt.get("future_final_used") is not False or input_receipt.get("capture_rows") != 255:
        raise ValueError("v2 input provenance mismatch")
    rows = [{**row, "capture_id": row["request"]["capture_id"],
             "source_group": row["source"].get("source_group")}
            for row in replay.load_sets()]
    it, inp, out, lut = _embedding_model()
    results = {}
    all_embeddings = {}
    private_prediction_rows = []
    roi_rows = []
    for side in ROI_SIDES:
        embeddings = np.zeros((len(rows), 128), dtype=np.float32)
        valid = np.zeros(len(rows), dtype=bool)
        for i, row in enumerate(rows):
            if len(row["boxes"]) != 1:
                continue
            payload = (row["folder"] / "camera_rgb565.bin").read_bytes()
            pixels = replay.rgb565_array(payload)
            rect, score = row["boxes"][0]
            if not strict_score_pass(score, DETECTION_THRESHOLD):
                raise AssertionError("replay accepted a detector score rejected by the board rule")
            center = fomo_center_camera_px(rect, 240, 240)
            roi = centered_roi_clamped(center, side, 240, 240)
            rgb = native_rgb(pixels, roi=roi, width=TENSOR_SIDE, height=TENSOR_SIDE)
            values = (rgb.astype(np.float32) - 127.5) / 128.0
            q = np.clip(np.rint(values / float(inp["quantization"][0])) + int(inp["quantization"][1]), -128, 127).astype(np.int8)
            it.set_tensor(inp["index"], q[None])
            it.invoke()
            vector = it.get_tensor(out["index"])[0]
            scale, zero = out["quantization"]
            embeddings[i] = _normalize((vector.astype(np.float32) - int(zero)) * float(scale))
            valid[i] = True
            roi_rows.append({"capture_id": row["request"]["capture_id"], "roi_side_camera_px": side,
                             "center_xy_camera_px": list(center), "roi_xywh_camera_px": list(roi),
                             "fomo_score": float(score), "camera_frame_sha256": row["camera_frame_sha256"],
                             "tensor_hwc": [48, 48, 3]})
        summary = _summarize(rows, embeddings, valid, side)
        private_rows = summary.pop("_prediction_rows")
        for i, row in enumerate(rows):
            prediction = private_rows[i]
            private_prediction_rows.append({"capture_id": row["request"]["capture_id"],
                "roi_side_camera_px": side, "role": row["role"], "label": row["label"],
                "source_group": row["source"].get("source_group"),
                "quality_status": "fomo_single_cluster" if valid[i] else
                    ("multiple_clusters" if len(row["boxes"]) > 1 else "no_face"),
                "valid": bool(valid[i]), "top1": prediction.get("top1") if prediction else None,
                "score": prediction.get("score") if prediction else None,
                "class_scores": prediction.get("class_scores") if prediction else None})
        all_embeddings[str(side)] = embeddings
        results[str(side)] = summary

    # Restrict all raw scores and embeddings to the private file.
    PRIVATE.mkdir(parents=True, exist_ok=True)
    OUT.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(PRIVATE / "EMBEDDINGS.npz", **{
        "roi_" + key: value for key, value in all_embeddings.items()})
    (PRIVATE / "ROI_RECORDS.jsonl").write_text(
        "".join(json.dumps(row, sort_keys=True, allow_nan=False) + "\n" for row in roi_rows), encoding="utf-8")
    (PRIVATE / "PREDICTIONS.jsonl").write_text(
        "".join(json.dumps(row, sort_keys=True, allow_nan=False) + "\n" for row in private_prediction_rows), encoding="utf-8")
    summary = {
        "schema": "v6-camera-roi-sweep-2", "status": "complete_finite_host_diagnostic",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "parent_input": "work/project-research-v5/runs/camera_corrected_input_v2/INPUTS.json",
        "parent_input_sha256": sha256_file(INPUTS), "capture_rows": len(rows),
        "model_sha256": sha256_file(MODEL), "model_input_hwc": [48, 48, 3],
        "crop_sizes_are_camera_roi_pixels_not_encoder_resolutions": True,
        "fomo_gate": {"threshold": DETECTION_THRESHOLD, "comparison": "strict_gt_float32",
                      "coordinate_space": "camera_xywh_px", "multiple_cluster_action": "UNCERTAIN"},
        "roi_sizes_camera_px": list(ROI_SIDES), "results": results,
        "private_embeddings": str((PRIVATE / "EMBEDDINGS.npz").relative_to(ROOT)),
        "private_roi_records": str((PRIVATE / "ROI_RECORDS.jsonl").relative_to(ROOT)),
        "future_final_used": False, "ready_for_final_build": False,
        "limits": ["One fixed 48px encoder and one camera experiment; only the crop ROI side varied.",
                   "Thresholds are calibrated on unknown_calibration only; development metrics are held out and must not be used to pick the deployment threshold.",
                   "Host tensor replay is not exact device parity for ROI sizes other than the V6 120px demonstration.",
                   "No gradient-trained model or deployment claim is made."],
    }
    (OUT / "RESULT.json").write_text(json.dumps(summary, indent=2, sort_keys=True, allow_nan=False,
                                                  default=_json_default) + "\n", encoding="utf-8")
    return summary


if __name__ == "__main__":
    print(json.dumps(run(), indent=2, sort_keys=True, allow_nan=False, default=_json_default))
