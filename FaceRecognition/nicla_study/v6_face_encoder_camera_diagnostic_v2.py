"""Replay the existing 112px face-pretrained encoder on corrected camera crops.

This tests whether a larger face-aware model can carry recognition before
starting a new learned campaign. Float and INT8 use identical corrected
FOMO crops and the reference TFLite interpreter. This remains host-only.
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
from open_set_decision import accepts, threshold_above_max
from camera_geometry import centered_roi_clamped, fomo_center_camera_px
from nicla_study import v6_camera_crop_features48_v2 as replay
from nicla_study.v5_native_rgb import native_rgb
from nicla_study.fs import sha256_file
from identity_contracts import strict_score_pass


STUDY = ROOT / "work/project-research-v5"
MODEL_DIR = STUDY / "runs/face_encoder"
MODELS = {"float": MODEL_DIR / "float.tflite", "int8": MODEL_DIR / "int8.tflite"}
OUT = STUDY / "runs/camera_corrected_face_encoder_v2"
PRIVATE = STUDY / "private/camera_corrected_face_encoder_v2"
ROI_SIDE = 120
TENSOR_SIDE = 112
FOMO_THRESHOLD = 0.5
KNOWN = ("K01", "K02", "K03", "K04", "K05")


def norm(vector: np.ndarray) -> np.ndarray:
    value = np.asarray(vector, dtype=np.float32).reshape(-1)
    length = float(np.linalg.norm(value))
    if value.shape != (128,) or not np.isfinite(value).all() or length <= 1e-8:
        raise ValueError("face encoder output is not finite nonzero 128-D")
    return value / length


def fomo_rows(metadata: dict) -> list[tuple[tuple[float, ...], float]]:
    return [(tuple(float(v) for v in item[0]), float(item[1]))
            for item in replay.clusters_for(metadata)
            if strict_score_pass(item[1], FOMO_THRESHOLD)]


def _input_tensor(rgb: np.ndarray, details: dict) -> np.ndarray:
    if details["dtype"] == np.float32:
        return ((rgb.astype(np.float32) - 127.5) / 128.0)[None]
    scale, zero = details["quantization"]
    if not scale or scale <= 0:
        raise ValueError("invalid quantized MobileFaceNet input scale")
    values = (rgb.astype(np.float32) - 127.5) / 128.0
    quantized = np.clip(np.rint(values / float(scale)) + int(zero), -128, 127).astype(np.int8)
    return quantized[None]


def _extract(model_path: Path, rows: list[dict]) -> tuple[list[dict], dict]:
    interpreter = tf.lite.Interpreter(
        model_path=str(model_path), num_threads=2,
        experimental_op_resolver_type=tf.lite.experimental.OpResolverType.BUILTIN_REF,
    )
    interpreter.allocate_tensors()
    input_detail = interpreter.get_input_details()[0]
    output_detail = interpreter.get_output_details()[0]
    if input_detail["shape"].tolist() != [1, 112, 112, 3]:
        raise ValueError("unexpected face-pretrained model input geometry")
    if output_detail["shape"].tolist() != [1, 128]:
        raise ValueError("unexpected face-pretrained model output geometry")
    records = []
    for row in rows:
        raw = (row["folder"] / "camera_rgb565.bin").read_bytes()
        pixels = replay.rgb565_array(raw)
        detections = fomo_rows(row["metadata"])
        record = {"capture_id": row["request"]["capture_id"], "capture_set": row["capture_set"],
                  "role": row["role"], "phase": row["phase"], "label": row["label"],
                  "source_group": row["source"].get("source_group"), "fomo_cluster_count": len(detections),
                  "embedding": None}
        if len(detections) == 1 and detections[0][1] >= FOMO_THRESHOLD:
            center = fomo_center_camera_px(detections[0][0], 240, 240)
            roi = centered_roi_clamped(center, ROI_SIDE, 240, 240)
            rgb = native_rgb(pixels, roi=roi, width=TENSOR_SIDE, height=TENSOR_SIDE)
            interpreter.set_tensor(input_detail["index"], _input_tensor(rgb, input_detail))
            interpreter.invoke()
            values = interpreter.get_tensor(output_detail["index"])[0]
            if output_detail["dtype"] == np.int8:
                scale, zero = output_detail["quantization"]
                if not scale or scale <= 0:
                    raise ValueError("invalid quantized MobileFaceNet output scale")
                values = (values.astype(np.float32) - int(zero)) * float(scale)
            record.update({"embedding": norm(values).tolist(), "roi_xywh_camera_px": list(roi),
                           "fomo_score": detections[0][1]})
        records.append(record)
    metadata = {"path": model_path.relative_to(ROOT).as_posix(), "sha256": sha256_file(model_path),
                "bytes": model_path.stat().st_size, "input_shape": input_detail["shape"].tolist(),
                "input_dtype": str(input_detail["dtype"]),
                "input_quantization": list(input_detail["quantization"]),
                "output_shape": output_detail["shape"].tolist(),
                "output_dtype": str(output_detail["dtype"]),
                "output_quantization": list(output_detail["quantization"]),
                "operators": sorted({row["op_name"] for row in interpreter._get_ops_details()})}
    return records, metadata


def summarize(records: list[dict]) -> dict:
    known_train = [row for row in records if row["role"] == "train" and row["label"] in KNOWN
                   and row["embedding"] is not None]
    groups: dict[str, dict[str, list[np.ndarray]]] = defaultdict(lambda: defaultdict(list))
    for row in known_train:
        groups[row["label"]][row.get("source_group") or row["capture_id"]].append(
            norm(np.asarray(row["embedding"], dtype=np.float32)))
    gallery = {}
    for label in KNOWN:
        prototypes = [norm(np.mean(np.stack(values), axis=0)) for values in groups[label].values()]
        if not prototypes:
            raise ValueError("no train-role camera gallery rows for " + label)
        gallery[label] = norm(np.mean(np.stack(prototypes), axis=0))

    score_rows = []
    for row in records:
        if row["embedding"] is None:
            score_rows.append(None)
        else:
            vector = norm(np.asarray(row["embedding"], dtype=np.float32))
            scores = np.asarray([float(np.dot(vector, gallery[label])) for label in KNOWN], np.float32)
            index = int(scores.argmax())
            score_rows.append({"top1": KNOWN[index], "score": float(scores[index])})
    cal_unknown = [score_rows[i]["score"] for i, row in enumerate(records)
                   if row["role"] == "unknown_calibration" and score_rows[i] is not None]
    if not cal_unknown:
        raise ValueError("no usable unknown calibration embeddings")
    threshold = threshold_above_max(cal_unknown)
    known_ids = [i for i, row in enumerate(records)
                 if row["role"] == "development" and row["label"] in KNOWN]
    unknown_ids = [i for i, row in enumerate(records) if row["role"] == "unknown_development"]
    accepted = {i: bool(score_rows[i] is not None and accepts(score_rows[i]["score"], threshold))
                for i in range(len(records))}
    per_person = {}
    for label in KNOWN:
        ids = [i for i in known_ids if records[i]["label"] == label]
        per_person[label] = {"rows": len(ids),
            "fomo_valid": sum(score_rows[i] is not None for i in ids),
            "raw_top1_correct_all_rows": sum(score_rows[i] is not None and score_rows[i]["top1"] == label for i in ids),
            "correct_accepted": sum(accepted[i] and score_rows[i]["top1"] == label for i in ids),
            "wrong_known_accepts": sum(accepted[i] and score_rows[i]["top1"] != label for i in ids)}
    unknown_groups: dict[str, list[int]] = defaultdict(list)
    for i in unknown_ids:
        unknown_groups[records[i].get("source_group") or records[i]["capture_id"]].append(i)
    return {
        "threshold_binary32": float(np.float32(threshold)),
        "threshold_rule": "one representable binary32 step above the maximum usable unknown-calibration max-known cosine; >= comparison",
        "unknown_calibration_groups": len({row.get("source_group") or row["capture_id"] for row in records
                                            if row["role"] == "unknown_calibration"}),
        "unknown_calibration_usable_rows": len(cal_unknown),
        "unknown_calibration_false_groups": 0,
        "known_development_rows": len(known_ids),
        "known_fomo_valid_rows": sum(score_rows[i] is not None for i in known_ids),
        "known_raw_top1_all_rows": sum(score_rows[i] is not None and score_rows[i]["top1"] == records[i]["label"]
                                       for i in known_ids) / max(1, len(known_ids)),
        "known_correct_accepted": sum(accepted[i] and score_rows[i]["top1"] == records[i]["label"]
                                      for i in known_ids),
        "known_correct_acceptance_all_rows": sum(accepted[i] and score_rows[i]["top1"] == records[i]["label"]
                                                 for i in known_ids) / max(1, len(known_ids)),
        "known_coverage_all_rows": sum(accepted[i] for i in known_ids) / max(1, len(known_ids)),
        "known_wrong_accepted": sum(accepted[i] and score_rows[i]["top1"] != records[i]["label"]
                                    for i in known_ids),
        "unknown_development_rows": len(unknown_ids),
        "unknown_development_groups": len(unknown_groups),
        "unknown_fomo_valid_rows": sum(score_rows[i] is not None for i in unknown_ids),
        "unknown_false_accepted_rows": sum(accepted[i] for i in unknown_ids),
        "unknown_false_accepted_groups": sum(any(accepted[i] for i in values)
                                             for values in unknown_groups.values()),
        "unknown_zero_event_upper95_for_30": 1.0 - 0.05 ** (1.0 / 30.0),
        "per_person": per_person,
    }


def run() -> dict:
    if (OUT / "RESULT.json").exists() or (PRIVATE / "RECORDS.jsonl").exists():
        raise FileExistsError("Face encoder camera diagnostic exists; use a new versioned namespace")
    rows = replay.load_sets()
    if len(rows) != 255:
        raise ValueError("expected the frozen 255-frame camera set")
    model_results = {}
    model_rows = {}
    for kind, path in MODELS.items():
        extracted, metadata = _extract(path, rows)
        model_rows[kind] = extracted
        model_results[kind] = {"model": metadata, "metrics": summarize(extracted)}
    float_by_id = {row["capture_id"]: row for row in model_rows["float"]}
    int8_cosines = []
    for row in model_rows["int8"]:
        other = float_by_id[row["capture_id"]]
        if row["embedding"] is not None and other["embedding"] is not None:
            int8_cosines.append(float(np.dot(np.asarray(row["embedding"], np.float32),
                                             np.asarray(other["embedding"], np.float32))))
    summary = {
        "schema": "v6-corrected-camera-face-encoder-diagnostic-2",
        "status": "complete_host_diagnostic",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "parent_input_experiment": "work/project-research-v5/runs/camera_corrected_input_v2/INPUTS.json",
        "input": {"coordinate_space": "camera_xywh_px", "roi_side_camera_px": ROI_SIDE,
                  "tensor_shape_hwc": [TENSOR_SIDE, TENSOR_SIDE, 3],
                  "resize": "OpenMV RGB565-native 16.16 interpolation; RGB888; MobileFaceNet normalization (pixel-127.5)/128",
                  "fomo_single_cluster_score_min": FOMO_THRESHOLD,
                  "fomo_threshold_comparison": "strict_gt_float32"},
        "models": model_results,
        "float_int8_same_input_cosine": {"n": len(int8_cosines),
            "mean": float(np.mean(int8_cosines)) if int8_cosines else None,
            "p05": float(np.quantile(int8_cosines, 0.05)) if int8_cosines else None,
            "minimum": float(np.min(int8_cosines)) if int8_cosines else None},
        "deployment_limits": [
            "112x112 MobileFaceNet is host-only here; prior Nicla resource blocks did not qualify the larger input/model allocation.",
            "The source checkpoint training-data overlap is unresolved and deployment eligibility is false.",
            "The device failed to allocate stock FOMO before exact camera-to-model tensor parity could be checked.",
            "Thresholds use camera unknown-calibration rows only; 30 groups cannot establish a 1% upper risk.",
        ],
        "future_final_used": False,
        "ready_for_final_build": False,
    }
    PRIVATE.mkdir(parents=True, exist_ok=True)
    OUT.mkdir(parents=True, exist_ok=True)
    for kind, records in model_rows.items():
        (PRIVATE / (kind.upper() + "_RECORDS.jsonl")).write_text(
            "".join(json.dumps(row, sort_keys=True, allow_nan=False) + "\n" for row in records),
            encoding="utf-8")
    (OUT / "RESULT.json").write_text(json.dumps(summary, indent=2, sort_keys=True, allow_nan=False) + "\n",
                                     encoding="utf-8")
    return summary


if __name__ == "__main__":
    print(json.dumps(run(), indent=2, sort_keys=True, allow_nan=False))
