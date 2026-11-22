"""Host-only reconstruction of the experimental classifier path from a JPEG.

All pixels here are decoded from the lossy 8766 preview JPEG. Nothing in this
module can establish that these were the board's inference pixels or outputs.
This module is imported by the optional TF2.10 worker, never by the default UI.
"""

from __future__ import annotations

import base64
import hashlib
import io
import json
import math
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image


WORKSPACE = Path(__file__).resolve().parents[2]
FACE_ROOT = WORKSPACE / "FaceRecognition"
MODEL_ROOT = FACE_ROOT / "models" / "deployment-v1"
RUNTIME_CONFIG_PATH = FACE_ROOT / "firmware" / "runtime_config.json"
MANIFEST_PATH = MODEL_ROOT / "integration_manifest_host_reconstruction_v4.json"
DETECTOR_NAME = "detector_gray_fomo_int8.tflite"
CLASSIFIER_NAME = "identity_classifier_a025_int8.tflite"
DETECTOR_SHA256 = "4a484c9b56f806b4d13b603e66c3f7df60012ac74149e7a28655ddd497e5195c"
CLASSIFIER_SHA256 = "8be1d461ae1606fd7d6be4295049e7e47f0c7516323cb9f5889e7346b9b9ca55"
# Pin the current runtime config through the versioned v2 host-reconstruction
# manifest.  The original v1 manifest remains unchanged for provenance.
RUNTIME_CONFIG_SHA256 = "de8d3110d5c25b98fbc3341f8936b3e4224b1535fece7ba98a53708ea583dd2b"
MANIFEST_SHA256 = "0cf2f14c9dd9e01b980d8ddd777252d5ed32dd6bdabe3413b898654114d295a8"
DETECTOR_LABELS = ["background", "face"]
CLASSIFIER_LABELS = ["K01", "K02", "K03", "K04", "K05", "OTHER"]
MAX_JPEG_BYTES = 1_000_000

# The source file carries an OpenMV BSD-3-Clause notice and is used only as a
# local host-reference port of OpenMV 5.0.1 FOMO postprocessing.
sys.path.insert(0, str(FACE_ROOT / "nicla_study" / "c_reference"))
sys.path.insert(0, str(FACE_ROOT / "firmware"))
from common import openmv_fomo_detections  # noqa: E402
from identity_contracts import (  # noqa: E402
    classifier_decision,
    pillow_bilinear_grayscale_u8,
    quality_decision,
)


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _b64(data: bytes) -> str:
    return base64.b64encode(data).decode("ascii")


def _shape(detail: dict[str, Any]) -> tuple[int, ...]:
    return tuple(int(value) for value in detail["shape"])


def _check_tensor(detail: dict[str, Any], shape: tuple[int, ...], dtype: Any,
                  scale: float, zero: int, name: str) -> None:
    if _shape(detail) != shape:
        raise ValueError(name + "_shape_mismatch")
    if detail["dtype"] != dtype:
        raise ValueError(name + "_dtype_mismatch")
    actual_scale, actual_zero = detail["quantization"]
    if not math.isclose(float(actual_scale), scale, rel_tol=0.0, abs_tol=1e-8):
        raise ValueError(name + "_scale_mismatch")
    if int(actual_zero) != zero:
        raise ValueError(name + "_zero_point_mismatch")


def load_models() -> dict[str, Any]:
    """Load only manifest-pinned local artifacts in the TF2.10 worker."""
    started = time.perf_counter()
    if not RUNTIME_CONFIG_PATH.is_file() or not MANIFEST_PATH.is_file():
        raise FileNotFoundError("workspace_model_manifest_missing")
    runtime_bytes = RUNTIME_CONFIG_PATH.read_bytes()
    runtime = json.loads(runtime_bytes.decode("utf-8"))
    manifest_bytes = MANIFEST_PATH.read_bytes()
    if _sha256(manifest_bytes) != MANIFEST_SHA256:
        raise ValueError("host_manifest_hash_mismatch")
    manifest = json.loads(manifest_bytes.decode("utf-8"))
    runtime_sha = _sha256(runtime_bytes)
    if runtime_sha != RUNTIME_CONFIG_SHA256:
        raise ValueError("runtime_config_hash_mismatch")
    manifest_runtime_sha = manifest.get("runtime_config", {}).get("sha256")

    artifacts = {
        DETECTOR_NAME: (DETECTOR_SHA256, "detector"),
        CLASSIFIER_NAME: (CLASSIFIER_SHA256, "classifier"),
    }
    model_hashes: dict[str, str] = {}
    for name, (pinned_hash, role) in artifacts.items():
        path = MODEL_ROOT / name
        entry = manifest.get("files", {}).get(name, {})
        if not path.is_file() or _sha256_file(path) != pinned_hash or entry.get("sha256") != pinned_hash:
            raise ValueError(role + "_model_hash_mismatch")
        model_hashes[name] = pinned_hash

    label_entries = manifest.get("labels", {})
    detector_label_path = MODEL_ROOT / (DETECTOR_NAME.removesuffix(".tflite") + ".txt")
    classifier_label_path = MODEL_ROOT / (CLASSIFIER_NAME.removesuffix(".tflite") + ".txt")
    if _sha256_file(detector_label_path) != label_entries.get("detector_sha256"):
        raise ValueError("detector_labels_hash_mismatch")
    if _sha256_file(classifier_label_path) != label_entries.get("classifier_sha256"):
        raise ValueError("classifier_labels_hash_mismatch")
    detector_labels = detector_label_path.read_text(encoding="utf-8").splitlines()
    classifier_labels = classifier_label_path.read_text(encoding="utf-8").splitlines()
    if detector_labels != DETECTOR_LABELS or classifier_labels != CLASSIFIER_LABELS:
        raise ValueError("model_label_order_mismatch")
    if label_entries.get("class_order") != CLASSIFIER_LABELS:
        raise ValueError("manifest_class_order_mismatch")

    detection = runtime.get("detection", {})
    large = runtime.get("identity", {}).get("large", {})
    crop_contract = runtime.get("identity", {}).get("crop", {})
    quality_contract = runtime.get("identity", {}).get("quality", {})
    if detection.get("backend") != "trained_gray_fomo":
        raise ValueError("runtime_detector_backend_mismatch")
    if detection.get("model_sha256") != DETECTOR_SHA256 or detection.get("threshold") != 0.7:
        raise ValueError("runtime_detector_contract_mismatch")
    if (
        detection.get("coordinate_contract") != "openmv-v5.0.1-fomo-240-window-v1"
        or detection.get("resize_contract") != "grayscale 240x240 window to 96x96 bilinear; no padding"
        or detection.get("input_shape") != [1, 96, 96, 1]
        or detection.get("input_dtype") != "int8"
        or detection.get("input_zero_point") != -128
        or not math.isclose(float(detection.get("input_scale", -1)), 1.0 / 255.0, rel_tol=0.0, abs_tol=1e-8)
        or detection.get("output_shape") != [1, 12, 12, 2]
        or detection.get("output_dtype") != "int8"
        or detection.get("output_scale") != 1.0 / 256.0
        or detection.get("output_zero_point") != -128
        or detection.get("threshold_comparison") != "strict_gt"
    ):
        raise ValueError("runtime_detector_preprocess_or_coordinate_contract_mismatch")
    if large.get("model_sha256") != CLASSIFIER_SHA256 or large.get("class_order") != CLASSIFIER_LABELS:
        raise ValueError("runtime_classifier_contract_mismatch")
    if (
        large.get("backend") != "classifier_logits"
        or large.get("threshold") != 0.505630741905405
        or large.get("input_shape") != [1, 96, 96, 1]
        or large.get("input_dtype") != "int8"
        or large.get("input_zero_point") != -128
        or not math.isclose(float(large.get("input_scale", -1)), 1.0 / 255.0, rel_tol=0.0, abs_tol=1e-8)
        or large.get("output_shape") != [1, 6]
        or large.get("output_dtype") != "int8"
        or not math.isclose(float(large.get("output_scale", -1)), 0.026111828163266182, rel_tol=0.0, abs_tol=1e-8)
        or large.get("output_zero_point") != -10
        or large.get("output_activation") != "logits"
    ):
        raise ValueError("runtime_classifier_threshold_or_tensor_contract_mismatch")
    if (
        crop_contract.get("side_window_px") != 96
        or crop_contract.get("rounding") != "floor(center-side/2+0.5)"
        or crop_contract.get("boundary_policy") != "reject_without_clamp_or_padding"
        or quality_contract.get("mean_min_u8") != 12
        or quality_contract.get("mean_max_u8") != 243
        or quality_contract.get("stdev_min_u8") != 5
        or runtime.get("identity", {}).get("uncertainty_band") != 0.025
    ):
        raise ValueError("runtime_identity_crop_or_quality_contract_mismatch")

    import tensorflow as tf

    detector = tf.lite.Interpreter(model_path=str(MODEL_ROOT / DETECTOR_NAME), num_threads=2)
    detector.allocate_tensors()
    classifier = tf.lite.Interpreter(model_path=str(MODEL_ROOT / CLASSIFIER_NAME), num_threads=2)
    classifier.allocate_tensors()
    d_in = detector.get_input_details()[0]
    d_out = detector.get_output_details()[0]
    c_in = classifier.get_input_details()[0]
    c_out = classifier.get_output_details()[0]
    _check_tensor(d_in, (1, 96, 96, 1), np.int8, 1.0 / 255.0, -128, "detector_input")
    _check_tensor(d_out, (1, 12, 12, 2), np.int8, 1.0 / 256.0, -128, "detector_output")
    _check_tensor(c_in, (1, 96, 96, 1), np.int8, 1.0 / 255.0, -128, "classifier_input")
    _check_tensor(c_out, (1, 6), np.int8, 0.026111828163266182, -10, "classifier_output")
    return {
        "detector": detector,
        "classifier": classifier,
        "detector_input": d_in,
        "detector_output": d_out,
        "classifier_input": c_in,
        "classifier_output": c_out,
        "detector_threshold": float(detection["threshold"]),
        "classifier_threshold": float(large["threshold"]),
        "uncertainty_band": float(runtime["identity"]["uncertainty_band"]),
        "class_order": classifier_labels,
        "hashes": model_hashes,
        "runtime": tf.__version__,
        "runtime_config_sha256": runtime_sha,
        "manifest_runtime_config_sha256": manifest_runtime_sha,
        "manifest_runtime_config_matches": runtime_sha == manifest_runtime_sha,
        "integration_manifest_sha256": MANIFEST_SHA256,
        "model_load_ms": round((time.perf_counter() - started) * 1000.0, 3),
    }


def analyze_jpeg(jpeg: bytes, models: dict[str, Any]) -> dict[str, Any]:
    """Reconstruct a host tensor/crop from one JPEG; never writes pixels."""
    started = time.perf_counter()
    if not isinstance(jpeg, (bytes, bytearray)) or not 100 < len(jpeg) <= MAX_JPEG_BYTES:
        raise ValueError("jpeg_missing_or_oversized")
    jpeg = bytes(jpeg)
    jpeg_sha = _sha256(jpeg)

    decode_start = time.perf_counter()
    try:
        with Image.open(io.BytesIO(jpeg)) as image:
            if image.format != "JPEG":
                raise ValueError("preview_not_jpeg")
            if image.size != (240, 240):
                raise ValueError("preview_geometry_mismatch_expected_240x240")
            image.load()
            gray = image.convert("L")
    except (OSError, ValueError) as error:
        raise ValueError("jpeg_decode_failed:" + str(error)) from error
    gray_bytes = gray.tobytes()
    decode_ms = (time.perf_counter() - decode_start) * 1000.0

    prep_start = time.perf_counter()
    detector_u8 = bytearray(96 * 96)
    pillow_bilinear_grayscale_u8(gray_bytes, 240, 240, (0, 0, 240, 240), detector_u8)
    detector_input = (np.frombuffer(detector_u8, dtype=np.uint8).astype(np.int16) - 128).astype(np.int8)
    detector_tensor = detector_input.reshape((1, 96, 96, 1))
    prep_ms = (time.perf_counter() - prep_start) * 1000.0

    detector_start = time.perf_counter()
    models["detector"].set_tensor(models["detector_input"]["index"], detector_tensor)
    models["detector"].invoke()
    detector_raw = np.asarray(models["detector"].get_tensor(models["detector_output"]["index"]), dtype=np.int8)
    detector_ms = (time.perf_counter() - detector_start) * 1000.0
    if detector_raw.shape != (1, 12, 12, 2):
        raise ValueError("detector_output_geometry_mismatch")
    d_scale, d_zero = models["detector_output"]["quantization"]
    detector_probabilities = (detector_raw.astype(np.float32) - np.float32(d_zero)) * np.float32(d_scale)

    post_start = time.perf_counter()
    regions = openmv_fomo_detections(detector_probabilities, models["detector_threshold"])
    postprocess_ms = (time.perf_counter() - post_start) * 1000.0
    detector_tensor_bytes = detector_tensor.tobytes(order="C")
    detector_raw_bytes = detector_raw.tobytes(order="C")
    result: dict[str, Any] = {
        "schema": "host-jpeg-reconstruction-v1",
        "evidence_scope": "HOST RECONSTRUCTION FROM LOSSY PREVIEW JPEG; NOT FRAME-IDENTICAL OR DEVICE-EQUIVALENT",
        "experimental": True,
        "qualified_identity": None,
        "source": {
            "jpeg_sha256": jpeg_sha,
            "jpeg_bytes": len(jpeg),
            "width": 240,
            "height": 240,
            "pixel_source": "OpenMV RGB565 preview JPEG quality 55 decoded to host RGB and Pillow L grayscale",
        },
        "models": dict(models["hashes"]),
        "runtime": models["runtime"],
        "runtime_config_sha256": models["runtime_config_sha256"],
        "manifest_runtime_config_sha256": models["manifest_runtime_config_sha256"],
        "manifest_runtime_config_matches": models["manifest_runtime_config_matches"],
        "integration_manifest_sha256": models["integration_manifest_sha256"],
        "warnings": ([] if models["manifest_runtime_config_matches"] else [
            "manifest_runtime_config_sha256_differs_from_the_pinned_current_config"
        ]),
        "model_use_scope": "hash-pinned local workspace artifacts used only for this requested local experimental diagnostic; no weights or training data are redistributed and no third-party training-data license is asserted",
        "detector": {
            "status": "regions_found" if regions else "no_face_regions",
            "threshold": models["detector_threshold"],
            "threshold_comparison": "strict_gt_float32",
            "input_tensor": {
                "shape": [1, 96, 96, 1],
                "dtype": "int8",
                "scale": 1.0 / 255.0,
                "zero_point": -128,
                "sha256": _sha256(detector_tensor_bytes),
                "bytes_base64": _b64(detector_tensor_bytes),
                "provenance": "exact host tensor reconstructed from this JPEG; not the board tensor",
            },
            "raw_output_tensor": {
                "shape": [1, 12, 12, 2],
                "dtype": "int8",
                "scale": float(d_scale),
                "zero_point": int(d_zero),
                "sha256": _sha256(detector_raw_bytes),
                "bytes_base64": _b64(detector_raw_bytes),
            },
            "regions": [
                {
                    "region_xywh_window_px": row["nms_rect_window_xywh"],
                    "center_xy_window_px": row["center_frame_xy"],
                    "grid_xy": row["grid_xy"],
                    "score": row["score"],
                    "note": "OpenMV 5.0.1 FOMO NMS region; not a calibrated face box",
                }
                for row in regions
            ],
        },
        "identity": {
            "status": "not_run",
            "reason": "identity_stage_not_reached",
            "class_order": list(models["class_order"]),
            "candidate_class": None,
            "raw_scores_int8": None,
            "raw_logits": None,
            "class_probabilities": None,
            "threshold": models["classifier_threshold"],
            "uncertainty_band": models["uncertainty_band"],
            "threshold_semantics": "max known-class softmax probability; historical alpha0.25 calibration; experimental only",
            "crop": None,
            "input_tensor": None,
        },
        "timings_ms": {
            "jpeg_decode_to_grayscale": round(decode_ms, 3),
            "detector_resize_and_quantize": round(prep_ms, 3),
            "detector_inference": round(detector_ms, 3),
            "fomo_postprocess": round(postprocess_ms, 3),
        },
    }
    if not regions:
        result["identity"].update({"status": "not_run", "reason": "no_fomo_regions_above_threshold"})
        result["timings_ms"]["total"] = round((time.perf_counter() - started) * 1000.0, 3)
        return result
    if len(regions) != 1:
        result["identity"].update({"status": "not_run", "reason": "multiple_fomo_regions"})
        result["timings_ms"]["total"] = round((time.perf_counter() - started) * 1000.0, 3)
        return result

    crop_start = time.perf_counter()
    cx, cy = regions[0]["center_frame_xy"]
    side = 96
    x = math.floor(float(cx) - side / 2.0 + 0.5)
    y = math.floor(float(cy) - side / 2.0 + 0.5)
    if x < 0 or y < 0 or x + side > 240 or y + side > 240:
        result["identity"].update({
            "status": "not_run",
            "reason": "crop_outside_frame",
            "crop": {"xywh_window_px": [x, y, side, side], "sha256": None, "bytes_base64": None},
        })
        result["timings_ms"]["crop_and_quality"] = round((time.perf_counter() - crop_start) * 1000.0, 3)
        result["timings_ms"]["total"] = round((time.perf_counter() - started) * 1000.0, 3)
        return result
    crop = np.frombuffer(gray_bytes, dtype=np.uint8).reshape((240, 240))[y:y + side, x:x + side].copy()
    crop_bytes = crop.tobytes(order="C")
    crop_mean = float(crop.mean())
    crop_stdev = float(crop.std())
    quality_ok, quality_reason = quality_decision(
        crop_mean,
        crop_stdev,
        12,
        243,
        5,
    )
    identity_u8 = bytearray(96 * 96)
    pillow_bilinear_grayscale_u8(crop_bytes, 96, 96, (0, 0, 96, 96), identity_u8)
    identity_input = (np.frombuffer(identity_u8, dtype=np.uint8).astype(np.int16) - 128).astype(np.int8)
    identity_bytes = identity_input.tobytes(order="C")
    result["identity"]["crop"] = {
        "xywh_window_px": [x, y, side, side],
        "shape": [96, 96],
        "dtype": "uint8 grayscale",
        "sha256": _sha256(crop_bytes),
        "bytes_base64": _b64(crop_bytes),
        "quality_mean_u8_host": crop_mean,
        "quality_stdev_u8_host": crop_stdev,
        "quality_gate": "pass" if quality_ok else "reject",
        "quality_reason": quality_reason,
    }
    result["timings_ms"]["crop_and_quality"] = round((time.perf_counter() - crop_start) * 1000.0, 3)
    if not quality_ok:
        result["identity"].update({"status": "rejected", "reason": "crop_quality_" + quality_reason})
        result["timings_ms"]["total"] = round((time.perf_counter() - started) * 1000.0, 3)
        return result

    classifier_start = time.perf_counter()
    identity_tensor = identity_input.reshape((1, 96, 96, 1))
    models["classifier"].set_tensor(models["classifier_input"]["index"], identity_tensor)
    models["classifier"].invoke()
    classifier_raw = np.asarray(
        models["classifier"].get_tensor(models["classifier_output"]["index"]), dtype=np.int8
    )
    classifier_ms = (time.perf_counter() - classifier_start) * 1000.0
    if classifier_raw.shape != (1, 6):
        raise ValueError("classifier_output_geometry_mismatch")
    c_scale, c_zero = models["classifier_output"]["quantization"]
    logits = ((classifier_raw.astype(np.float32) - np.float32(c_zero)) * np.float32(c_scale))[0]
    gate = classifier_decision(
        [float(value) for value in logits],
        models["class_order"],
        models["classifier_threshold"],
        models["uncertainty_band"],
    )
    gate_to_host = {
        "KNOWN_IDENTITY": ("experimental_acceptance_gate_passed", "above_historical_threshold_band"),
        "OTHER": ("experimental_reject", gate.get("reason", "historical_other_gate")),
        "UNCERTAIN": ("experimental_review", gate.get("reason", "historical_uncertainty_band")),
    }
    decision_status, decision_reason = gate_to_host.get(
        gate.get("status"), ("experimental_review", gate.get("reason", "invalid_decision_state"))
    )
    classifier_raw_bytes = classifier_raw.tobytes(order="C")
    result["identity"].update({
        "status": decision_status,
        "reason": decision_reason,
        "candidate_class": gate.get("candidate_identity_id"),
        "raw_scores_int8": [int(value) for value in classifier_raw.reshape(-1)],
        "raw_logits": [float(value) for value in logits],
        "class_probabilities": gate.get("class_probabilities"),
        "threshold": models["classifier_threshold"],
        "uncertainty_band": models["uncertainty_band"],
        "threshold_comparison": "max_known_probability against threshold +/- uncertainty band; experimental only",
        "raw_output_tensor": {
            "shape": [1, 6],
            "dtype": "int8 logits",
            "scale": float(c_scale),
            "zero_point": int(c_zero),
            "sha256": _sha256(classifier_raw_bytes),
            "bytes_base64": _b64(classifier_raw_bytes),
        },
        "input_tensor": {
            "shape": [1, 96, 96, 1],
            "dtype": "int8",
            "scale": 1.0 / 255.0,
            "zero_point": -128,
            "sha256": _sha256(identity_bytes),
            "bytes_base64": _b64(identity_bytes),
            "provenance": "exact host tensor reconstructed from this JPEG crop; not the board tensor",
        },
    })
    result["timings_ms"]["classifier_inference"] = round(classifier_ms, 3)
    result["timings_ms"]["decision"] = 0.0
    result["timings_ms"]["total"] = round((time.perf_counter() - started) * 1000.0, 3)
    return result
