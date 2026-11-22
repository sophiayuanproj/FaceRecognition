"""Test fixed crops around the Nicla Haar detector's saved camera boxes.

This is a finite host diagnostic using the real 240x240 RGB565 captures and
the Haar boxes produced on-device in the same capture blocks. Haar coordinates
are explicitly mapped from its 160x160 gray input to camera pixels. Two square
context factors are selected using calibration roles only; development remains
evaluation-only. SFace is a host-only reference with unresolved pretraining
exposure, not an MCU candidate.
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
from camera_geometry import centered_roi_clamped
from identity_contracts import quality_decision
from nicla_study import v6_camera_crop_features48_v2 as replay
from nicla_study.fs import sha256_file
from nicla_study.v5_native_rgb import native_rgb
from open_set_decision import accepts, threshold_above_max

STUDY = ROOT / "work/project-research-v5"
SFACE_PATH = ROOT / "work/project-finish-v4/models/pretrained/sface/face_recognition_sface_2021dec.onnx"
YUNET_PATH = ROOT / "work/project-finish-v4/models/pretrained/sface/face_detection_yunet_2023mar.onnx"
SFACE_SHA256 = "0ba9fbfa01b5270c96627c4ef784da859931e02f04419c829e83484087c34e79"
YUNET_SHA256 = "8f2383e4dd3cfbb4553ea8718107fc0423210dc964f9f4280604804ed2552fa4"
OUT = STUDY / "runs/camera_haar_crop_sface_v1"
PRIVATE = STUDY / "private/camera_haar_crop_sface_v1"
KNOWN = ("K01", "K02", "K03", "K04", "K05")
FACTORS = (1.0, 1.4)
CAMERA_PER_HAAR_PIXEL = 240.0 / 160.0


def normalize(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32).reshape(-1)
    norm = float(np.linalg.norm(values))
    if values.shape != (128,) or not np.isfinite(values).all() or norm <= 1e-8:
        raise ValueError("SFace output must be finite nonzero 128-D")
    return values / np.float32(norm)


def haar_boxes_160(value) -> list[tuple[float, float, float, float]]:
    if value in (None, ""):
        return []
    if isinstance(value, str):
        import ast
        value = ast.literal_eval(value)
    result = []
    for item in value:
        if not isinstance(item, (list, tuple)) or len(item) != 4:
            raise ValueError("malformed Haar gray160 rectangle")
        box = tuple(float(part) for part in item)
        if not np.isfinite(box).all() or box[2] <= 0 or box[3] <= 0:
            raise ValueError("invalid Haar rectangle")
        result.append(box)
    return result


def extract(rows: list[dict], sface, context_factor: float) -> list[dict]:
    records = []
    for row in rows:
        base = {
            "capture_id": row["request"]["capture_id"], "capture_set": row["capture_set"],
            "role": row["role"], "phase": row["phase"], "label": row["label"],
            "source_group": row["source"].get("source_group"),
            "source_record_id": row["source"].get("record_id"),
            "source_path": row["source"].get("path"),
            "source_sha256": row["source"].get("sha256"),
            "camera_frame_sha256": row["camera_frame_sha256"],
            "haar_box_count": len(haar_boxes_160(row["metadata"].get("haar_boxes_160"))),
            "embedding": None, "crop_xywh_camera_px": None,
            "quality_status": "haar_not_single",
        }
        boxes = haar_boxes_160(row["metadata"].get("haar_boxes_160"))
        if len(boxes) == 1:
            x, y, width, height = boxes[0]
            cx = (x + width / 2.0) * CAMERA_PER_HAAR_PIXEL
            cy = (y + height / 2.0) * CAMERA_PER_HAAR_PIXEL
            side = max(1, int(round(context_factor * max(width, height) * CAMERA_PER_HAAR_PIXEL)))
            roi = centered_roi_clamped((cx, cy), side, 240, 240)
            raw = (row["folder"] / "camera_rgb565.bin").read_bytes()
            pixels = replay.rgb565_array(raw)
            rgb = native_rgb(pixels, roi=roi, width=112, height=112)
            gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
            quality, reason = quality_decision(float(gray.mean()), float(gray.std()))
            base.update({"haar_rect_gray160": [x, y, width, height],
                         "haar_box_camera_xywh_px": [x * CAMERA_PER_HAAR_PIXEL,
                             y * CAMERA_PER_HAAR_PIXEL, width * CAMERA_PER_HAAR_PIXEL,
                             height * CAMERA_PER_HAAR_PIXEL],
                         "context_factor": context_factor, "crop_xywh_camera_px": list(roi),
                         "quality_status": reason})
            if quality:
                bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
                base["embedding"] = normalize(sface.feature(bgr)).tolist()
        records.append(base)
    return records


def source_gallery(rows: list[dict], sface, yunet) -> tuple[dict[str, np.ndarray], dict]:
    """Build an aligned source-face gallery from frozen known train records."""
    by_group: dict[str, tuple[str, str, str]] = {}
    for row in rows:
        if row["role"] != "train" or row["label"] not in KNOWN:
            continue
        source = row["source"]
        group = source.get("source_group") or source["record_id"]
        value = (row["label"], source["path"], source["sha256"])
        if group in by_group and by_group[group] != value:
            raise ValueError("source group maps to inconsistent train image metadata")
        by_group[group] = value
    grouped: dict[str, dict[str, list[np.ndarray]]] = defaultdict(lambda: defaultdict(list))
    bindings = []
    for group, (label, relative_path, expected_sha) in sorted(by_group.items()):
        path = ROOT / relative_path
        if sha256_file(path) != expected_sha:
            raise ValueError("known train source image hash mismatch")
        image = cv2.imread(str(path))
        if image is None:
            raise ValueError("cannot read known train source image")
        height, width = image.shape[:2]
        yunet.setInputSize((width, height))
        _ok, boxes = yunet.detect(image)
        if boxes is None or len(boxes) != 1:
            bindings.append({"source_group_sha256": replay.sha256(group.encode()),
                             "source_sha256": expected_sha, "label": label,
                             "status": "no_single_yunet_face"})
            continue
        aligned = sface.alignCrop(image, boxes[0])
        if aligned is None or aligned.size == 0:
            raise ValueError("SFace source alignCrop returned an empty image")
        grouped[label][group].append(normalize(sface.feature(aligned)))
        bindings.append({"source_group_sha256": replay.sha256(group.encode()),
                         "source_sha256": expected_sha, "label": label,
                         "status": "aligned_source_train"})
    gallery = {}
    counts = {}
    for label in KNOWN:
        prototypes = [normalize(np.mean(np.stack(vectors), axis=0))
                      for vectors in grouped[label].values()]
        if not prototypes:
            raise ValueError("source train gallery lacks " + label)
        gallery[label] = normalize(np.mean(np.stack(prototypes), axis=0))
        counts[label] = len(prototypes)
    binding_bytes = json.dumps(bindings, sort_keys=True, separators=(",", ":")).encode()
    return gallery, {"source_groups_by_label": counts,
                     "eligible_source_groups": len(by_group),
                     "aligned_source_groups": sum(row["status"] == "aligned_source_train" for row in bindings),
                     "binding_sha256": replay.sha256(binding_bytes),
                     "source_role": "frozen lfw-v5 train only"}


def score_records(records: list[dict], gallery: dict[str, np.ndarray]) -> dict[str, dict]:

    scored = {}
    for row in records:
        if row["embedding"] is None:
            scored[row["capture_id"]] = {"top1": None, "score": None}
            continue
        vector = normalize(np.asarray(row["embedding"], np.float32))
        values = np.asarray([np.dot(vector, gallery[label]) for label in KNOWN], np.float32)
        index = int(np.argmax(values))
        scored[row["capture_id"]] = {"top1": KNOWN[index], "score": float(values[index])}
    return scored


def _calibration(records: list[dict], scored: dict[str, dict]) -> dict:
    known = [row for row in records if row["role"] == "threshold_calibration" and row["label"] in KNOWN]
    unknown = [row for row in records if row["role"] == "unknown_calibration"]
    unknown_valid = [row for row in unknown if scored[row["capture_id"]]["score"] is not None]
    if not unknown_valid:
        raise ValueError("no valid unknown-calibration Haar crops")
    threshold = float(threshold_above_max([scored[row["capture_id"]]["score"] for row in unknown_valid]))
    known_valid = [row for row in known if scored[row["capture_id"]]["score"] is not None]
    known_correct = sum(
        scored[row["capture_id"]]["top1"] == row["label"] and
        accepts(scored[row["capture_id"]]["score"], threshold) for row in known
    )
    unknown_groups: dict[str, list[dict]] = defaultdict(list)
    for row in unknown_valid:
        unknown_groups[row.get("source_group") or row["capture_id"]].append(row)
    false_groups = sum(any(accepts(scored[row["capture_id"]]["score"], threshold) for row in values)
                       for values in unknown_groups.values())
    return {"threshold_binary32": threshold,
            "threshold_rule": "one representable binary32 step above max unknown-calibration cosine; >=",
            "known_rows": len(known), "known_valid_rows": len(known_valid),
            "known_correct_accepted": int(known_correct),
            "known_correct_acceptance_all_rows": known_correct / max(1, len(known)),
            "unknown_groups": len(unknown_groups), "unknown_valid_rows": len(unknown_valid),
            "unknown_false_groups": int(false_groups)}


def _development(records: list[dict], scored: dict[str, dict], threshold: float) -> dict:
    known = [row for row in records if row["role"] == "development" and row["label"] in KNOWN]
    unknown = [row for row in records if row["role"] == "unknown_development"]
    accepted = {}
    per_person = {}
    for row in known:
        pred = scored[row["capture_id"]]
        accepted[row["capture_id"]] = pred["score"] is not None and accepts(pred["score"], threshold)
    for label in KNOWN:
        part = [row for row in known if row["label"] == label]
        per_person[label] = {
            "rows": len(part),
            "haar_single_box_rows": sum(scored[row["capture_id"]]["score"] is not None for row in part),
            "top1_correct_all_rows": sum(scored[row["capture_id"]]["top1"] == label for row in part),
            "correct_accepted": sum(accepted[row["capture_id"]] and
                scored[row["capture_id"]]["top1"] == label for row in part),
            "wrong_known_accepts": sum(accepted[row["capture_id"]] and
                scored[row["capture_id"]]["top1"] != label for row in part),
        }
    unknown_groups: dict[str, list[dict]] = defaultdict(list)
    for row in unknown:
        unknown_groups[row.get("source_group") or row["capture_id"]].append(row)
    false_groups = sum(any(
        scored[row["capture_id"]]["score"] is not None and
        accepts(scored[row["capture_id"]]["score"], threshold) for row in part)
        for part in unknown_groups.values())
    unknown_accepted_rows = sum(
        scored[row["capture_id"]]["score"] is not None and
        accepts(scored[row["capture_id"]]["score"], threshold) for row in unknown)
    return {
        "known_rows": len(known), "known_haar_valid_rows": sum(scored[row["capture_id"]]["score"] is not None for row in known),
        "known_raw_top1_all_rows": sum(scored[row["capture_id"]]["top1"] == row["label"] for row in known) / max(1, len(known)),
        "known_correct_accepted": int(sum(accepted[row["capture_id"]] and
            scored[row["capture_id"]]["top1"] == row["label"] for row in known)),
        "known_correct_acceptance_all_rows": sum(accepted[row["capture_id"]] and
            scored[row["capture_id"]]["top1"] == row["label"] for row in known) / max(1, len(known)),
        "known_wrong_accepted": sum(accepted[row["capture_id"]] and
            scored[row["capture_id"]]["top1"] != row["label"] for row in known),
        "known_coverage_all_rows": sum(accepted.values()) / max(1, len(known)),
        "unknown_rows": len(unknown), "unknown_groups": len(unknown_groups),
        "unknown_haar_valid_rows": sum(scored[row["capture_id"]]["score"] is not None for row in unknown),
        "unknown_false_accepted_rows": int(unknown_accepted_rows),
        "unknown_false_accepted_groups": int(false_groups),
        "unknown_zero_event_upper95_for_30": 1.0 - 0.05 ** (1.0 / 30.0),
        "per_person": per_person,
    }


def run() -> dict:
    if (OUT / "RESULT.json").exists() or (PRIVATE / "RECORDS.jsonl").exists():
        raise FileExistsError("preserving the v1 result namespace; create a new version")
    if sha256_file(SFACE_PATH) != SFACE_SHA256 or sha256_file(YUNET_PATH) != YUNET_SHA256:
        raise ValueError("unexpected SFace or YuNet checkpoint hash")
    cv2.setNumThreads(2)
    sface = cv2.FaceRecognizerSF.create(str(SFACE_PATH), "")
    yunet = cv2.FaceDetectorYN.create(str(YUNET_PATH), "", (250, 250), .85, .3, 5000)
    rows = replay.load_sets()
    if len(rows) != 255:
        raise ValueError("expected all 255 frozen RGB565 captures")
    gallery, gallery_provenance = source_gallery(rows, sface, yunet)
    candidates = []
    records_by_factor = {}
    predictions_by_factor = {}
    for factor in FACTORS:
        records = extract(rows, sface, factor)
        scored = score_records(records, gallery)
        calibration = _calibration(records, scored)
        candidates.append({"context_factor": factor, "calibration": calibration})
        records_by_factor[factor] = records
        predictions_by_factor[factor] = scored
    # Choose the crop with the most correctly accepted known calibration rows.
    # Development rows are never consulted by this selection.
    selected = max(candidates, key=lambda item: (
        item["calibration"]["known_correct_accepted"],
        -item["calibration"]["unknown_false_groups"],
        item["calibration"]["known_valid_rows"],
        -item["context_factor"],
    ))
    factor = selected["context_factor"]
    calibration = selected["calibration"]
    records = records_by_factor[factor]
    predictions = predictions_by_factor[factor]
    for row in records:
        pred = predictions[row["capture_id"]]
        row["predicted_label"] = pred["top1"]
        row["max_known_cosine"] = pred["score"]
        row["accepted_known"] = pred["score"] is not None and accepts(pred["score"], calibration["threshold_binary32"])
    PRIVATE.mkdir(parents=True, exist_ok=True)
    OUT.mkdir(parents=True, exist_ok=True)
    records_path = PRIVATE / "RECORDS.jsonl"
    records_path.write_text("".join(json.dumps(row, sort_keys=True, allow_nan=False) + "\n" for row in records), encoding="utf-8")
    development = _development(records, predictions, calibration["threshold_binary32"])
    report = {
        "schema": "v6-camera-haar-crop-sface-diagnostic-1",
        "status": "complete_host_diagnostic",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "parent_input_experiment": "work/project-research-v5/runs/camera_corrected_input_v2/INPUTS.json",
        "capture_rows": len(rows), "camera_frame_geometry": [240, 240],
        "detector": {"name": "on-device OpenMV HaarCascade frontalface", "input_geometry": [160, 160],
                     "saved_coordinate_space": "gray160_xywh", "camera_scale": CAMERA_PER_HAAR_PIXEL,
                     "boxes_are_face_bounds": True, "multiple_box_policy": "UNCERTAIN/no identity"},
        "crop_candidates": candidates, "selected_context_factor": factor,
        "selection_rule": "max known-calibration correct acceptance, then fewer unknown false groups, then more known valid crops; development excluded",
        "selected_calibration": calibration, "development": development,
        "gallery": gallery_provenance,
        "sface": {"model": "OpenCV Zoo SFace 2021dec", "sha256": SFACE_SHA256,
                  "yunet_sha256": YUNET_SHA256,
                  "host_only": True, "pretraining_overlap": "unresolved", "deployment_eligible": False},
        "records_path": records_path.relative_to(ROOT).as_posix(),
        "records_sha256": sha256_file(records_path), "future_final_used": False,
        "limits": [
            "This is a host replay of saved on-device Haar boxes, not a live device recognition result.",
            "The box mapping and crop resize are coordinate-compatible diagnostics; they do not establish Haar box recall or full deployment parity.",
            "The same single taped screen setup is used; unknown rows are 30 identities and cannot certify a 1% upper risk.",
            "SFace checkpoint exposure is unresolved and this model has no MCU resource or license qualification.",
        ],
        "ready_for_final_build": False,
    }
    result_path = OUT / "RESULT.json"
    result_path.write_text(json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    return report


if __name__ == "__main__":
    print(json.dumps(run(), indent=2, sort_keys=True, allow_nan=False))
