"""Host-only SFace diagnostic using strict-threshold corrected v2 camera crops.

This probes the useful-teacher branch on the existing 255 captured RGB565
frames. It never uses future-final data and never writes raw frames or feature
vectors outside the private research directory.
"""

from __future__ import annotations

import hashlib
import json
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
import sys

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "FaceRecognition"))
sys.path.insert(0, str(ROOT / "FaceRecognition/firmware"))
from nicla_study import v6_camera_crop_features48_v2 as replay
from nicla_study.v5_native_rgb import native_rgb, unpack
from nicla_study.fs import sha256_file
from identity_contracts import quality_decision
from identity_contracts import strict_score_pass
from open_set_decision import accepts, threshold_above_max


STUDY = ROOT / "work/project-research-v5"
MODEL_DIR = ROOT / "work/project-finish-v4/models/pretrained/sface"
SFACE_PATH = MODEL_DIR / "face_recognition_sface_2021dec.onnx"
YUNET_PATH = MODEL_DIR / "face_detection_yunet_2023mar.onnx"
SFACE_SHA256 = "0ba9fbfa01b5270c96627c4ef784da859931e02f04419c829e83484087c34e79"
YUNET_SHA256 = "8f2383e4dd3cfbb4553ea8718107fc0423210dc964f9f4280604804ed2552fa4"
ROI_SIDE = 120
FOMO_THRESHOLD = 0.5
KNOWN = ("K01", "K02", "K03", "K04", "K05")
OUT = STUDY / "runs/camera_corrected_teacher_v2"
PRIVATE = STUDY / "private/camera_corrected_teacher_v2"
RESULT_PATH = OUT / "RESULT.json"
RECORD_PATH = PRIVATE / "RECORDS.jsonl"
CACHE_PATH = PRIVATE / "sface_features.npz"


def normalize(vector: np.ndarray) -> np.ndarray:
    value = np.asarray(vector, dtype=np.float32).reshape(-1)
    norm = float(np.linalg.norm(value))
    if value.shape != (128,) or not np.isfinite(value).all() or norm <= 1e-8:
        raise ValueError("SFace embedding must be finite, nonzero, and 128-D")
    return value / norm


def fomo_rectangles(metadata: dict) -> list[tuple[tuple[float, ...], float]]:
    clusters = replay.clusters_for(metadata)
    return [(tuple(float(value) for value in row[0]), float(row[1])) for row in clusters
            if strict_score_pass(row[1], FOMO_THRESHOLD)]


def _features_for_row(row: dict, sface, yunet) -> dict:
    raw = (row["folder"] / "camera_rgb565.bin").read_bytes()
    pixels = replay.rgb565_array(raw)
    detections = fomo_rectangles(row["metadata"])
    record = {
        "capture_id": row["request"]["capture_id"],
        "capture_set": row["capture_set"],
        "role": row["role"],
        "phase": row["phase"],
        "label": row["label"],
        "source_group": row["source"].get("source_group"),
        "fomo_cluster_count": len(detections),
        "fomo_scores": [score for _rect, score in detections],
        "fomo_status": "NO_FACE" if not detections else (
            "MULTIPLE_FACES" if len(detections) > 1 else "ONE_FACE"),
        "camera_frame_sha256": row["camera_frame_sha256"],
        "fomo_crop_embedding": None,
        "yu_net_aligned_embedding": None,
        "crop_quality_status": None,
        "yunet_detection_count": None,
    }
    if len(detections) == 1 and detections[0][1] >= FOMO_THRESHOLD:
        center = replay.fomo_center_camera_px(detections[0][0], 240, 240)
        roi = replay.centered_roi_clamped(center, ROI_SIDE, 240, 240)
        rgb = native_rgb(pixels, roi=roi, width=112, height=112)
        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        quality_ok, quality_reason = quality_decision(float(gray.mean()), float(gray.std()))
        record.update({"center_xy_camera_px": list(center), "roi_xywh_camera_px": list(roi),
                       "crop_mean_u8": float(gray.mean()), "crop_stdev_u8": float(gray.std()),
                       "crop_quality_status": quality_reason})
        if quality_ok:
            record["fomo_crop_embedding"] = normalize(sface.feature(bgr)).tolist()

        # This is a host-alignment diagnostic, gated by the same FOMO single
        # cluster so misses and ambiguous FOMO outputs remain end-to-end misses.
        frame_rgb = unpack(pixels)
        frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
        _ok, faces = yunet.detect(frame_bgr)
        count = 0 if faces is None else int(len(faces))
        record["yunet_detection_count"] = count
        if count == 1:
            aligned = sface.alignCrop(frame_bgr, faces[0])
            record["yu_net_aligned_embedding"] = normalize(sface.feature(aligned)).tolist()
    else:
        record["crop_quality_status"] = "detector_not_single_above_threshold"
    return record


def _gallery(records: list[dict], field: str) -> dict[str, np.ndarray]:
    grouped: dict[str, dict[str, list[np.ndarray]]] = defaultdict(lambda: defaultdict(list))
    for row in records:
        if row["role"] != "train" or row["label"] not in KNOWN or row[field] is None:
            continue
        group = row.get("source_group")
        if not isinstance(group, str) or not group:
            raise ValueError("known training row has no frozen source group")
        grouped[row["label"]][group].append(normalize(np.asarray(row[field], np.float32)))
    result = {}
    for label in KNOWN:
        group_vectors = []
        for values in grouped[label].values():
            group_vectors.append(normalize(np.mean(np.stack(values), axis=0)))
        if not group_vectors:
            raise ValueError("no valid camera-training source groups for " + label)
        result[label] = normalize(np.mean(np.stack(group_vectors), axis=0))
    return result


def _score(row: dict, gallery: dict[str, np.ndarray], field: str) -> dict:
    value = row[field]
    if value is None:
        return {"detected": False, "top1": None, "score": None}
    vector = normalize(np.asarray(value, dtype=np.float32))
    scores = {label: float(np.dot(vector, gallery[label])) for label in KNOWN}
    top = max(KNOWN, key=scores.get)
    return {"detected": True, "top1": top, "score": scores[top], "scores": scores}


def _summarize(records: list[dict], field: str, gallery: dict[str, np.ndarray]) -> dict:
    unknown_cal = [row for row in records
                   if row["role"] == "unknown_calibration" and row[field] is not None]
    if not unknown_cal:
        raise ValueError("no usable unknown calibration rows for " + field)
    cal_scores = [_score(row, gallery, field)["score"] for row in unknown_cal]
    threshold = threshold_above_max(cal_scores)

    known_dev = [row for row in records if row["role"] == "development" and row["label"] in KNOWN]
    unknown_dev = [row for row in records if row["role"] == "unknown_development"]
    cal_unknown_groups = defaultdict(list)
    for row in records:
        if row["role"] == "unknown_calibration":
            cal_unknown_groups[row.get("source_group") or row["capture_id"]].append(row)

    def annotate(rows):
        output = []
        for row in rows:
            scored = _score(row, gallery, field)
            accepted = bool(scored["detected"] and accepts(scored["score"], threshold))
            output.append((row, scored, accepted))
        return output

    known_scored = annotate(known_dev)
    unknown_scored = annotate(unknown_dev)
    cal_false_groups = 0
    for group_rows in cal_unknown_groups.values():
        if any((_score(row, gallery, field)["detected"] and
                accepts(_score(row, gallery, field)["score"], threshold)) for row in group_rows):
            cal_false_groups += 1
    unknown_groups = defaultdict(list)
    for row, scored, accepted in unknown_scored:
        unknown_groups[row.get("source_group") or row["capture_id"]].append((scored, accepted))
    dev_false_groups = sum(any(accepted for _scored, accepted in group_rows)
                           for group_rows in unknown_groups.values())
    per_person = {}
    for label in KNOWN:
        subset = [(row, scored, accepted) for row, scored, accepted in known_scored
                  if row["label"] == label]
        per_person[label] = {
            "rows": len(subset),
            "fomo_valid": sum(scored["detected"] for _row, scored, _accepted in subset),
            "raw_top1_correct_all_rows": sum(scored["detected"] and scored["top1"] == label
                                              for _row, scored, _accepted in subset),
            "correct_accepted": sum(accepted and scored["top1"] == label
                                    for _row, scored, accepted in subset),
            "wrong_known_accepts": sum(accepted and scored["top1"] != label
                                       for _row, scored, accepted in subset),
        }
    n_known = len(known_scored)
    return {
        "feature_route": field,
        "threshold_binary32": threshold,
        "threshold_rule": "one representable binary32 step above maximum usable unknown_calibration max-known cosine; >= comparison",
        "unknown_calibration_rows": len([row for row in records if row["role"] == "unknown_calibration"]),
        "unknown_calibration_groups": len(cal_unknown_groups),
        "unknown_calibration_usable_rows": len(unknown_cal),
        "unknown_calibration_false_groups": cal_false_groups,
        "known_development_rows": n_known,
        "known_fomo_valid_rows": sum(scored["detected"] for _row, scored, _accepted in known_scored),
        "known_raw_top1_all_rows": sum(scored["detected"] and scored["top1"] == row["label"]
                                       for row, scored, _accepted in known_scored) / max(1, n_known),
        "known_correct_accepted_all_rows": sum(accepted and scored["top1"] == row["label"]
                                               for row, scored, accepted in known_scored),
        "known_coverage_all_rows": sum(accepted for _row, _scored, accepted in known_scored) / max(1, n_known),
        "known_wrong_accepted": sum(accepted and scored["top1"] != row["label"]
                                    for row, scored, accepted in known_scored),
        "known_correct_acceptance_all_rows": sum(accepted and scored["top1"] == row["label"]
                                                  for row, scored, accepted in known_scored) / max(1, n_known),
        "unknown_development_rows": len(unknown_scored),
        "unknown_development_groups": len(unknown_groups),
        "unknown_fomo_valid_rows": sum(scored["detected"] for _row, scored, _accepted in unknown_scored),
        "unknown_false_accepted_rows": sum(accepted for _row, _scored, accepted in unknown_scored),
        "unknown_false_accepted_groups": int(dev_false_groups),
        "unknown_zero_event_upper95_for_30": 1.0 - 0.05 ** (1.0 / 30.0),
        "per_person": per_person,
        "calibration_max_unknown_score": float(max(cal_scores)),
        "calibration_only_threshold_selection": True,
        "host_only": True,
        "future_final_used": False,
    }


def run() -> dict:
    if RESULT_PATH.exists() or RECORD_PATH.exists() or CACHE_PATH.exists():
        raise FileExistsError("Teacher diagnostic output exists; choose a new versioned namespace")
    actual_sface = sha256_file(SFACE_PATH)
    actual_yunet = sha256_file(YUNET_PATH)
    if actual_sface != SFACE_SHA256 or actual_yunet != YUNET_SHA256:
        raise ValueError("SFace or YuNet checkpoint hash mismatch")
    cv2.setNumThreads(2)
    sface = cv2.FaceRecognizerSF.create(str(SFACE_PATH), "")
    yunet = cv2.FaceDetectorYN.create(str(YUNET_PATH), "", (240, 240), 0.85, 0.3, 5000)

    rows = replay.load_sets()
    if len(rows) != 255:
        raise ValueError("expected the frozen 255-frame camera set")
    records = [_features_for_row(row, sface, yunet) for row in rows]
    route_summaries = {}
    route_galleries = {}
    for field in ("fomo_crop_embedding", "yu_net_aligned_embedding"):
        gallery = _gallery(records, field)
        route_galleries[field] = gallery
        route_summaries[field] = _summarize(records, field, gallery)

    raw_features = np.full((len(records), 2, 128), np.nan, dtype=np.float32)
    for index, row in enumerate(records):
        for column, field in enumerate(("fomo_crop_embedding", "yu_net_aligned_embedding")):
            if row[field] is not None:
                raw_features[index, column] = np.asarray(row[field], dtype=np.float32)
    PRIVATE.mkdir(parents=True, exist_ok=True)
    OUT.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(CACHE_PATH, embeddings=raw_features)
    RECORD_PATH.write_text("".join(json.dumps(row, allow_nan=False, sort_keys=True) + "\n"
                                         for row in records), encoding="utf-8")
    summary = {
        "schema": "v6-corrected-camera-sface-diagnostic-2",
        "status": "complete_host_diagnostic",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "parent_input_experiment": "work/project-research-v5/runs/camera_corrected_input_v2/INPUTS.json",
        "capture_rows": len(rows),
        "fomo_rule": {"coordinate_space": "camera_xywh_px", "single_cluster_score_min": FOMO_THRESHOLD,
                      "comparison": "strict_gt_float32",
                      "roi_side_camera_px": ROI_SIDE, "boundary_policy": "slide_inward_fixed_size_no_padding"},
        "teacher": {"model": "OpenCV Zoo SFace 2021dec", "sha256": actual_sface,
                    "yu_net_sha256": actual_yunet, "pretraining_overlap": "unresolved",
                    "deployment_eligible": False},
        "feature_methods": {
            "fomo_crop_embedding": "one corrected 120px FOMO-centred camera ROI, native RGB565 resize to112x112, BGR triplication and SFace feature without landmarks",
            "yu_net_aligned_embedding": "same FOMO single-cluster gate, full-frame YuNet single detection and SFace alignCrop; host diagnostic only",
        },
        "routes": route_summaries,
        "per_capture_embeddings_private": str(RECORD_PATH.relative_to(ROOT)),
        "private_feature_cache": str(CACHE_PATH.relative_to(ROOT)),
        "private_cache_sha256": sha256_file(CACHE_PATH),
        "grouping": "one train prototype per known source group, then equal-weight class prototype",
        "limits": [
            "Host SFace is a diagnostic teacher, not an MCU candidate or a device parity result.",
            "FOMO single-cluster, crop quality and YuNet alignment misses remain in all end-to-end denominators.",
            "Threshold is selected from camera unknown-calibration groups only; the 30-group set cannot certify a 1% upper risk.",
            "SFace pretraining exposure remains unresolved; all outputs and embeddings remain private.",
        ],
        "future_final_used": False,
        "ready_for_final_build": False,
    }
    RESULT_PATH.write_text(json.dumps(summary, indent=2, sort_keys=True, allow_nan=False) + "\n",
                           encoding="utf-8")
    return summary


if __name__ == "__main__":
    print(json.dumps(run(), indent=2, sort_keys=True, allow_nan=False))
