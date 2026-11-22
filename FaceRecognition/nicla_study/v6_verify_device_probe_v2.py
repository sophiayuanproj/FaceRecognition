"""Verify one real Nicla FOMO capture against its host input/reference.

This single diagnostic frame is isolated from the frozen training,
calibration, development, and future-final datasets. It checks exact camera
RGB565 -> device FOMO input bytes, reference-kernel FOMO output, and computes
the corrected identity ROI/host tensor for a later device parity run.
"""

from __future__ import annotations

import ast
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import tensorflow as tf

ROOT = Path(__file__).resolve().parents[2]
STUDY = ROOT / "work/project-research-v5"
CAPTURE_ID = "v6-corrected-v2-allocation-retry-20261102-01"
CAPTURE = STUDY / "private/optical-captures" / CAPTURE_ID
OUT = STUDY / "runs/camera_corrected_device_probe_v2"
PRIVATE = STUDY / "private/camera_corrected_device_probe_v2"
FOMO_MODEL = ROOT / "work/model-training/deployment-backup/fomo_face_detection.tflite"
IDENTITY_MODEL = STUDY / "runs/operator_fused_encoder48/int8.tflite"

sys.path.insert(0, str(ROOT / "FaceRecognition/firmware"))
sys.path.insert(0, str(ROOT / "FaceRecognition"))
from identity_contracts import centered_roi_clamped, fomo_center_camera_px
from nicla_study.v5_native_rgb import native_rgb


def sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def verify() -> dict:
    receipt_path = CAPTURE / "RECEIPT.json"
    receipt = json.loads(receipt_path.read_text())
    if receipt["request"].get("capture_id") != CAPTURE_ID:
        raise ValueError("capture receipt ID mismatch")
    if receipt["request"].get("future_final_used") is not False:
        raise ValueError("capture is not explicitly outside future-final")
    metadata = receipt["metadata"]
    if [metadata.get("width"), metadata.get("height")] != [240, 240]:
        raise ValueError("unexpected camera frame geometry")

    frame_bytes = (CAPTURE / "camera_rgb565.bin").read_bytes()
    frame_sha = sha256(frame_bytes)
    frame_row = next(row for row in receipt["files"] if row["name"] == "camera_rgb565.bin")
    if len(frame_bytes) != frame_row["bytes"] or frame_sha != frame_row["sha256"]:
        raise ValueError("raw camera frame hash differs from capture receipt")
    pixels = np.frombuffer(frame_bytes, dtype="<u2").reshape(240, 240)

    fomo_model = tf.lite.Interpreter(
        model_path=str(FOMO_MODEL), num_threads=2,
        experimental_op_resolver_type=tf.lite.experimental.OpResolverType.BUILTIN_REF,
    )
    fomo_model.allocate_tensors()
    fomo_input = fomo_model.get_input_details()[0]
    fomo_output = fomo_model.get_output_details()[0]
    if fomo_input["shape"].tolist() != [1, 96, 96, 3] or fomo_input["dtype"] != np.int8:
        raise ValueError("host FOMO model input contract differs from board")
    if fomo_output["shape"].tolist() != [1, 12, 12, 2] or fomo_output["dtype"] != np.int8:
        raise ValueError("host FOMO model output contract differs from board")
    model_bytes = FOMO_MODEL.read_bytes()
    board_model_text = metadata.get("fomo_model", "")
    board_model_bytes = int(board_model_text.split("model_size: ", 1)[1].split(",", 1)[0])
    rgb = native_rgb(pixels, width=96, height=96)
    scale, zero_point = fomo_input["quantization"]
    host_fomo_input = np.clip(
        np.rint(rgb.astype(np.float64) / 255.0 / float(scale)) + int(zero_point),
        -128, 127,
    ).astype(np.int8)
    device_fomo_input = np.frombuffer((CAPTURE / "fomo_input_i8.bin").read_bytes(), dtype=np.int8)
    if host_fomo_input.size != device_fomo_input.size:
        raise ValueError("host/device FOMO input lengths differ")
    input_exact = bool(np.array_equal(host_fomo_input.reshape(-1), device_fomo_input))
    PRIVATE.mkdir(parents=True, exist_ok=True)
    host_fomo_input.reshape(-1).tofile(PRIVATE / "host_fomo_input_i8.bin")

    fomo_model.set_tensor(fomo_input["index"], host_fomo_input[None, ...])
    fomo_model.invoke()
    host_fomo_output = fomo_model.get_tensor(fomo_output["index"])
    device_fomo_output = np.frombuffer((CAPTURE / "fomo_output_i8.bin").read_bytes(), dtype=np.int8)
    if host_fomo_output.size != device_fomo_output.size:
        raise ValueError("host/device FOMO output lengths differ")
    delta = np.abs(host_fomo_output.reshape(-1).astype(np.int16) - device_fomo_output.astype(np.int16))
    output_exact = bool(np.array_equal(host_fomo_output.reshape(-1), device_fomo_output))

    fomo_groups = ast.literal_eval(metadata["fomo"])
    detections = [row for group in fomo_groups for row in group]
    if len(detections) != 1:
        raise ValueError("expected this diagnostic to contain exactly one FOMO cluster")
    rect, score = detections[0]
    rect = tuple(float(value) for value in rect)
    center = fomo_center_camera_px(rect, 240, 240)
    roi = centered_roi_clamped(center, 120, 240, 240)
    identity_rgb = native_rgb(pixels, roi=roi, width=48, height=48)

    identity_model = tf.lite.Interpreter(
        model_path=str(IDENTITY_MODEL), num_threads=2,
        experimental_op_resolver_type=tf.lite.experimental.OpResolverType.BUILTIN_REF,
    )
    identity_model.allocate_tensors()
    identity_input = identity_model.get_input_details()[0]
    if identity_input["shape"].tolist() != [1, 48, 48, 3] or identity_input["dtype"] != np.int8:
        raise ValueError("selected identity encoder input contract changed")
    identity_scale, identity_zero = identity_input["quantization"]
    normalized = (identity_rgb.astype(np.float32) - 127.5) / 128.0
    identity_tensor = np.clip(
        np.rint(normalized / float(identity_scale)) + int(identity_zero), -128, 127,
    ).astype(np.int8)
    identity_tensor.reshape(-1).tofile(PRIVATE / "host_identity_tensor_i8.bin")

    result = {
        "schema": "v6-real-camera-device-input-probe-2",
        "status": "fomo_device_detected_host_input_exact" if input_exact else "fomo_input_mismatch",
        "capture_id": CAPTURE_ID,
        "capture_receipt": {
            "path": CAPTURE.relative_to(ROOT).as_posix(),
            "sha256": sha256(receipt_path.read_bytes()),
            "program_sha256": receipt.get("program_sha256"),
            "future_final_used": False,
            "evaluation_role": "non_evaluation",
        },
        "camera_frame": {"width": 240, "height": 240, "pixel_format": "RGB565",
                         "bytes": len(frame_bytes), "sha256": frame_sha},
        "device_fomo": {
            "status": "one_face_cluster",
            "rect_camera_xywh_px": list(rect),
            "center_camera_xy_px": list(center),
            "score": float(score),
            "model_bytes_reported_on_device": board_model_bytes,
            "model_bytes_host_reference": len(model_bytes),
            "model_sha256_host_reference": sha256(model_bytes),
            "arena_bytes_reported_on_device": int(board_model_text.split("ram_size: ", 1)[1].split(",", 1)[0]),
            "load_us": int(metadata["fomo_load_us"]),
            "predict_us": int(metadata["fomo_predict_us"]),
            "haar_boxes_160": metadata.get("haar_boxes_160"),
        },
        "fomo_input_parity": {
            "shape": [1, 96, 96, 3], "dtype": "int8",
            "input_scale": float(scale), "input_zero_point": int(zero_point),
            "device_tensor_sha256": sha256((CAPTURE / "fomo_input_i8.bin").read_bytes()),
            "host_tensor_sha256": sha256(host_fomo_input.tobytes()),
            "exact": input_exact,
            "different_values": int(np.count_nonzero(host_fomo_input.reshape(-1) != device_fomo_input)),
        },
        "fomo_reference_output": {
            "shape": [1, 12, 12, 2], "dtype": "int8",
            "device_tensor_sha256": sha256((CAPTURE / "fomo_output_i8.bin").read_bytes()),
            "host_tensor_sha256": sha256(host_fomo_output.tobytes()),
            "exact": output_exact,
            "different_values": int(np.count_nonzero(delta)),
            "max_abs_difference": int(delta.max()) if delta.size else 0,
            "host_model_size_matches_device_report": len(model_bytes) == board_model_bytes,
        },
        "corrected_identity_roi": {
            "policy": "centered 120x120 camera ROI shifted inward at frame edges; no padding",
            "roi_xywh_camera_px": list(roi),
            "tensor_shape": [1, 48, 48, 3],
            "tensor_dtype": "int8",
            "host_tensor_path": "RGB565 -> exact native OpenMV RGB565/RGB888 resize -> model int8 affine quantization",
            "host_tensor_bytes": int(identity_tensor.size),
            "host_tensor_sha256": sha256(identity_tensor.tobytes()),
            "device_identity_tensor_parity": "not_captured_in_this_probe",
            "identity_model_sha256": sha256(IDENTITY_MODEL.read_bytes()),
            "identity_model_input_scale": float(identity_scale),
            "identity_model_input_zero_point": int(identity_zero),
        },
        "claims": {
            "identity_result": "not_run",
            "recognition_accuracy": "not_measured",
            "device_camera_crop_identity_tensor_parity": False,
            "candidate_deployable": False,
            "final_confirmatory_data_opened": False,
        },
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "RESULT.json").write_text(
        json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8"
    )
    return result


if __name__ == "__main__":
    value = verify()
    print(json.dumps(value, indent=2, sort_keys=True, allow_nan=False))
    if value["status"] != "fomo_device_detected_host_input_exact":
        raise SystemExit(1)
