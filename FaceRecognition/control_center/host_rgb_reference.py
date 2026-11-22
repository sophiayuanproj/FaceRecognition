"""Opt-in host-only YuNet + aligned SFace reference on a source JPEG.

This is deliberately a separate path from the grayscale FOMO reconstruction.
It consumes the same lossy 240x240 color preview JPEG but never claims to be
Nicla inference or an identity qualification.
"""

from __future__ import annotations

import base64
import hashlib
import json
import math
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np


WORKSPACE = Path(__file__).resolve().parents[2]
STUDY = WORKSPACE / "work" / "project-research-v5"
RESULT_PATH = STUDY / "runs" / "optical_sface_rgb_yunet_v1" / "RESULT.json"
FRAME_RECORDS_PATH = STUDY / "private" / "optical_sface_rgb_yunet_v1" / "FRAME_RECORDS.jsonl"
SOURCE_GROUPS_PATH = STUDY / "private" / "optical_sface_rgb_yunet_v1" / "SOURCE_GROUP_RECORDS.jsonl"
MODEL_DIR = WORKSPACE / "work" / "project-finish-v4" / "models" / "pretrained" / "sface"
YUNET_PATH = MODEL_DIR / "face_detection_yunet_2023mar.onnx"
SFACE_PATH = MODEL_DIR / "face_recognition_sface_2021dec.onnx"

RESULT_SHA256 = "3668aeb8cd48ba5471ce65738a268d0b9a9275a76d020bc0848245d892f033d0"
FRAME_RECORDS_SHA256 = "8ac3c8faa3b9e3ee9b227cc5f2da8dd43c4f9c4479ca359451a8c0a013e8268f"
SOURCE_GROUPS_SHA256 = "c278924c0656f922d8eef4022af4dc0178ed592647e4dc76b7ac46657e6b6071"
YUNET_SHA256 = "8f2383e4dd3cfbb4553ea8718107fc0423210dc964f9f4280604804ed2552fa4"
SFACE_SHA256 = "0ba9fbfa01b5270c96627c4ef784da859931e02f04419c829e83484087c34e79"
KNOWN_LABELS = ("K01", "K02", "K03", "K04", "K05")
ALL_LABELS = (*KNOWN_LABELS, "OTHER")
MAX_JPEG_BYTES = 1_000_000
EXPECTED_KNOWN_THRESHOLD = 0.3794962465763092
EXPECTED_MARGIN_THRESHOLD = 0.2637098431587219
EXPECTED_OPENCV_VERSION = "4.10.0"
EXPECTED_NUMPY_VERSION = "1.23.5"


def _sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _b64(value: bytes) -> str:
    return base64.b64encode(value).decode("ascii")


def _normalize(value: Any) -> np.ndarray:
    vector = np.asarray(value, dtype=np.float32).reshape(-1)
    if vector.shape != (128,) or not np.isfinite(vector).all():
        raise ValueError("sface_embedding_not_finite_128d")
    norm = np.float32(np.linalg.norm(vector))
    if not np.isfinite(norm) or norm <= np.float32(1e-8):
        raise ValueError("sface_embedding_zero_norm")
    return vector / norm


def _cosine(left: np.ndarray, right: np.ndarray) -> np.float32:
    return np.float32(np.dot(left, right))


def _threshold_above_max(values: list[np.float32]) -> float:
    if not values:
        raise ValueError("unknown_calibration_has_no_scoreable_people")
    maximum = np.float32(max(values))
    return float(np.nextafter(maximum, np.float32(np.inf)))


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError as error:
            raise ValueError(f"invalid_jsonl_record:{path.name}:{line_number}") from error
        if not isinstance(value, dict):
            raise ValueError(f"jsonl_record_not_object:{path.name}:{line_number}")
        records.append(value)
    return records


def _make_gallery(
    groups: list[dict[str, Any]],
) -> tuple[dict[str, np.ndarray], np.ndarray, dict[str, int]]:
    by_label: dict[str, dict[str, list[np.ndarray]]] = defaultdict(lambda: defaultdict(list))
    other_by_person: dict[str, list[np.ndarray]] = defaultdict(list)
    allowed_roles = {
        "train",
        "other_train",
        "threshold_calibration",
        "unknown_calibration",
        "development",
        "unknown_development",
    }
    for row in groups:
        role = row.get("role")
        label = row.get("label")
        if role not in allowed_roles:
            raise ValueError("rgb_reference_role_outside_pinned_run")
        embedding = row.get("embedding")
        if embedding is None:
            continue
        vector = _normalize(embedding)
        if role == "train" and label in KNOWN_LABELS:
            by_label[label][str(row["record_id"])].append(vector)
        elif role == "other_train" and label == "OTHER":
            other_by_person[str(row["source_identity"])].append(vector)

    gallery: dict[str, np.ndarray] = {}
    counts: dict[str, int] = {}
    for label in KNOWN_LABELS:
        if not by_label[label]:
            raise ValueError("rgb_reference_known_training_gallery_missing:" + label)
        source_prototypes = [
            _normalize(np.mean(np.stack(vectors), axis=0))
            for vectors in by_label[label].values()
        ]
        gallery[label] = _normalize(np.mean(np.stack(source_prototypes), axis=0))
        counts[label] = len(source_prototypes)

    if not other_by_person:
        raise ValueError("rgb_reference_other_training_gallery_missing")
    other_prototypes = [
        _normalize(np.mean(np.stack(vectors), axis=0))
        for vectors in other_by_person.values()
    ]
    other = _normalize(np.mean(np.stack(other_prototypes), axis=0))
    counts["OTHER_train_identities"] = len(other_prototypes)
    return gallery, other, counts


def _calibrate_from_unknown_only(
    groups: list[dict[str, Any]],
    gallery: dict[str, np.ndarray],
    other: np.ndarray,
) -> tuple[float, float, int]:
    known_scores: list[np.float32] = []
    margins: list[np.float32] = []
    usable_people: set[str] = set()
    for row in groups:
        if row.get("role") != "unknown_calibration" or row.get("embedding") is None:
            continue
        vector = _normalize(row["embedding"])
        best_known = max(_cosine(vector, gallery[label]) for label in KNOWN_LABELS)
        other_score = _cosine(vector, other)
        known_scores.append(best_known)
        margins.append(np.float32(best_known - other_score))
        usable_people.add(str(row.get("source_identity", row.get("record_id"))))
    return (
        _threshold_above_max(known_scores),
        _threshold_above_max(margins),
        len(usable_people),
    )


def load_models() -> dict[str, Any]:
    """Validate pinned local model/data artifacts and load host OpenCV models."""
    started = time.perf_counter()
    if np.__version__ != EXPECTED_NUMPY_VERSION:
        raise RuntimeError("rgb_reference_numpy_version_unpinned:" + np.__version__)
    for path, expected, name in (
        (RESULT_PATH, RESULT_SHA256, "rgb_reference_result"),
        (FRAME_RECORDS_PATH, FRAME_RECORDS_SHA256, "rgb_reference_frame_records"),
        (SOURCE_GROUPS_PATH, SOURCE_GROUPS_SHA256, "rgb_reference_source_groups"),
        (YUNET_PATH, YUNET_SHA256, "yunet_model"),
        (SFACE_PATH, SFACE_SHA256, "sface_model"),
    ):
        if not path.is_file() or _file_sha256(path) != expected:
            raise ValueError(name + "_hash_mismatch")

    result = json.loads(RESULT_PATH.read_text(encoding="utf-8"))
    if result.get("schema") != "optical-sface-yunet-rgb-result-v1":
        raise ValueError("rgb_reference_result_schema_mismatch")
    if result.get("status") != "complete_host_reference_only":
        raise ValueError("rgb_reference_result_not_complete")
    if result.get("deployment_scope") != "host-only ONNX detector/recognizer; no Nicla inference, runtime, memory, power, or ACK qualification":
        raise ValueError("rgb_reference_result_scope_mismatch")
    if result.get("yunet_model_sha256") != YUNET_SHA256 or result.get("sface_model_sha256") != SFACE_SHA256:
        raise ValueError("rgb_reference_result_model_hash_mismatch")
    if result.get("yunet_score_threshold") != 0.50 or result.get("yunet_nms_threshold") != 0.30:
        raise ValueError("rgb_reference_detector_threshold_mismatch")

    groups = _read_jsonl(SOURCE_GROUPS_PATH)
    gallery, other, gallery_counts = _make_gallery(groups)
    if gallery_counts != result.get("gallery_source_groups"):
        raise ValueError("rgb_reference_gallery_counts_mismatch")
    known_threshold, margin_threshold, calibration_people = _calibrate_from_unknown_only(
        groups, gallery, other
    )
    if known_threshold != result.get("threshold_known_max_cosine"):
        raise ValueError("rgb_reference_known_threshold_mismatch")
    if margin_threshold != result.get("threshold_known_other_margin"):
        raise ValueError("rgb_reference_margin_threshold_mismatch")
    if known_threshold != EXPECTED_KNOWN_THRESHOLD or margin_threshold != EXPECTED_MARGIN_THRESHOLD:
        raise ValueError("rgb_reference_threshold_pin_mismatch")
    if calibration_people != result["strategies"]["known_vs_other_margin"]["unknown_calibration_usable_people"]:
        raise ValueError("rgb_reference_unknown_calibration_count_mismatch")

    import cv2

    if cv2.__version__ != EXPECTED_OPENCV_VERSION:
        raise RuntimeError("rgb_reference_opencv_version_unpinned:" + cv2.__version__)
    if not hasattr(cv2, "FaceDetectorYN") or not hasattr(cv2, "FaceRecognizerSF_create"):
        raise RuntimeError("opencv_contrib_face_apis_missing")
    detector = cv2.FaceDetectorYN.create(
        str(YUNET_PATH),
        "",
        (240, 240),
        float(result["yunet_score_threshold"]),
        float(result["yunet_nms_threshold"]),
        5000,
    )
    recognizer = cv2.FaceRecognizerSF_create(str(SFACE_PATH), "")
    return {
        "detector": detector,
        "recognizer": recognizer,
        "gallery": gallery,
        "other_prototype": other,
        "result": result,
        "opencv_version": cv2.__version__,
        "numpy_version": np.__version__,
        "hashes": {
            "yunet_model": YUNET_SHA256,
            "sface_model": SFACE_SHA256,
            "run_result": RESULT_SHA256,
            "frame_records": FRAME_RECORDS_SHA256,
            "source_group_records": SOURCE_GROUPS_SHA256,
        },
        "threshold_known_max_cosine": known_threshold,
        "threshold_known_other_margin": margin_threshold,
        "gallery_source_groups": gallery_counts,
        "unknown_calibration_people": calibration_people,
        "load_ms": round((time.perf_counter() - started) * 1000.0, 3),
    }


def analyze_jpeg(jpeg: bytes, models: dict[str, Any]) -> dict[str, Any]:
    """Run pinned YuNet/SFace host reference against one decoded preview JPEG."""
    started = time.perf_counter()
    if not isinstance(jpeg, (bytes, bytearray)) or not 100 < len(jpeg) <= MAX_JPEG_BYTES:
        raise ValueError("rgb_reference_jpeg_missing_or_oversized")
    jpeg = bytes(jpeg)
    jpeg_sha = _sha256(jpeg)

    import cv2

    decode_started = time.perf_counter()
    image = cv2.imdecode(np.frombuffer(jpeg, dtype=np.uint8), cv2.IMREAD_COLOR)
    decode_ms = (time.perf_counter() - decode_started) * 1000.0
    if image is None:
        raise ValueError("rgb_reference_jpeg_decode_failed")
    if image.shape != (240, 240, 3):
        raise ValueError("rgb_reference_expected_240x240_color_jpeg")

    detect_started = time.perf_counter()
    _, faces = models["detector"].detect(image)
    detect_ms = (time.perf_counter() - detect_started) * 1000.0
    face_count = 0 if faces is None else int(len(faces))
    result: dict[str, Any] = {
        "schema": "host-rgb-reference-v1",
        "evidence_scope": "HOST RGB REFERENCE FROM LOSSY PREVIEW JPEG; NOT ON NICLA; NOT DEVICE-EQUIVALENT",
        "execution_location": "host_reference",
        "experimental": True,
        "recognition_qualified": False,
        "qualified_identity": None,
        "pretraining_caveat": models["result"]["pretraining_caveat"],
        "source": {
            "jpeg_sha256": jpeg_sha,
            "jpeg_bytes": len(jpeg),
            "width": 240,
            "height": 240,
            "color_decode": "OpenCV IMREAD_COLOR BGR8 from exact source JPEG",
            "device_frame_id": None,
        },
        "models": {
            "opencv_version": models["opencv_version"],
            "numpy_version": models["numpy_version"],
            **models["hashes"],
        },
        "detector": {
            "name": "YuNet",
            "score_threshold": 0.50,
            "nms_threshold": 0.30,
            "face_count": face_count,
            "status": "one_face" if face_count == 1 else "no_face" if face_count == 0 else "multiple_faces",
            "bbox_xywh_240_px": None,
            "score": None,
        },
        "recognition": {
            "name": "SFace 2021dec; YuNet five-landmark alignCrop; 112x112 BGR aligned crop",
            "status": "not_run",
            "reason": "recognition_requires_exactly_one_yunet_face",
            "raw_top1_known_class": None,
            "known_cosines": None,
            "other_cosine": None,
            "known_other_margin": None,
            "threshold_known_max_cosine": models["threshold_known_max_cosine"],
            "threshold_known_other_margin": models["threshold_known_other_margin"],
            "known_cosine_gate": {"accepted": None, "reason": "not_run"},
            "known_vs_other_margin_gate": {"accepted": None, "reason": "not_run"},
            "aligned_crop": None,
        },
        "timings_ms": {
            "jpeg_decode_bgr": round(decode_ms, 3),
            "yunet_detect": round(detect_ms, 3),
        },
        "calibration": {
            "source": "pinned optical-sface-yunet-rgb-result-v1; thresholds recomputed from unknown_calibration groups only",
            "unknown_calibration_usable_people": models["unknown_calibration_people"],
            "gallery_source_groups": models["gallery_source_groups"],
            "pretraining_exposure": "unresolved",
        },
    }
    if face_count == 0:
        result["recognition"]["reason"] = "yunet_detected_no_face_at_score_threshold_0.50"
        result["timings_ms"]["total"] = round((time.perf_counter() - started) * 1000.0, 3)
        return result
    if face_count != 1:
        result["recognition"]["reason"] = "multiple_yunet_faces_suppress_sface"
        result["timings_ms"]["total"] = round((time.perf_counter() - started) * 1000.0, 3)
        return result

    face = np.asarray(faces[0], dtype=np.float32)
    if face.size < 15 or not np.isfinite(face).all():
        raise ValueError("rgb_reference_yunet_face_row_invalid")
    result["detector"]["bbox_xywh_240_px"] = [float(value) for value in face[:4]]
    result["detector"]["score"] = float(face[-1])
    result["detector"]["landmarks_xy_240_px"] = [
        [float(face[4 + 2 * index]), float(face[5 + 2 * index])]
        for index in range(5)
    ]

    align_started = time.perf_counter()
    aligned = models["recognizer"].alignCrop(image, face)
    align_ms = (time.perf_counter() - align_started) * 1000.0
    if aligned is None or aligned.size == 0 or aligned.shape != (112, 112, 3):
        raise ValueError("rgb_reference_sface_aligned_crop_invalid")
    ok, encoded = cv2.imencode(".png", aligned)
    if not ok:
        raise ValueError("rgb_reference_aligned_crop_png_encode_failed")
    crop_bytes = encoded.tobytes()

    feature_started = time.perf_counter()
    feature = _normalize(models["recognizer"].feature(aligned))
    feature_ms = (time.perf_counter() - feature_started) * 1000.0
    known_cosines = {
        label: float(_cosine(feature, models["gallery"][label]))
        for label in KNOWN_LABELS
    }
    raw_top1 = max(KNOWN_LABELS, key=lambda label: known_cosines[label])
    max_known = np.float32(known_cosines[raw_top1])
    other_cosine = _cosine(feature, models["other_prototype"])
    margin = np.float32(max_known - other_cosine)
    known_accept = float(max_known) >= models["threshold_known_max_cosine"]
    margin_accept = float(margin) >= models["threshold_known_other_margin"]
    result["recognition"].update({
        "status": "scored",
        "reason": "raw_candidate_is_not_qualified; apply each calibrated host gate separately",
        "raw_top1_known_class": raw_top1,
        "known_cosines": known_cosines,
        "other_cosine": float(other_cosine),
        "known_other_margin": float(margin),
        "known_cosine_gate": {
            "accepted": known_accept,
            "observed": float(max_known),
            "operator": ">=",
            "threshold": models["threshold_known_max_cosine"],
            "reason": "max_known_cosine_at_or_above_threshold" if known_accept else "max_known_cosine_below_threshold",
        },
        "known_vs_other_margin_gate": {
            "accepted": margin_accept,
            "observed": float(margin),
            "operator": ">=",
            "threshold": models["threshold_known_other_margin"],
            "reason": "known_minus_other_margin_at_or_above_threshold" if margin_accept else "known_minus_other_margin_below_threshold",
        },
        "aligned_crop": {
            "shape": [112, 112, 3],
            "color_order": "BGR",
            "encoding": "PNG",
            "sha256": _sha256(crop_bytes),
            "bytes_base64": _b64(crop_bytes),
            "provenance": "SFace alignCrop of exactly this JPEG's one YuNet detection",
        },
    })
    result["timings_ms"].update({
        "yunet_align_crop": round(align_ms, 3),
        "sface_feature": round(feature_ms, 3),
        "score_and_gates": 0.0,
        "total": round((time.perf_counter() - started) * 1000.0, 3),
    })
    return result
