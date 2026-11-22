"""Verify one V4 real-frame detector result with the matched host reference."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import tensorflow as tf

ROOT = Path(__file__).resolve().parents[2]
STUDY = ROOT / "work/project-research-v5"
FOMO_MODEL = ROOT / "work/model-training/deployment-backup/fomo_face_detection.tflite"
sys.path.insert(0, str(ROOT / "FaceRecognition"))
from nicla_study.v5_native_rgb import native_rgb  # noqa: E402
from nicla_study.v6_actual_tensor_parity import _read_blobs  # noqa: E402


def sha(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def verify(transcript: Path, output: Path) -> dict:
    metadata, blobs = _read_blobs(transcript)
    raw = blobs.get("camera_rgb565")
    if raw is None or len(raw) != 240 * 240 * 2:
        raise ValueError("V4 transcript lacks one complete 240x240 RGB565 frame")
    if sha(raw) != metadata.get("frame_sha256"):
        raise ValueError("V4 camera frame does not match its device hash")
    if metadata.get("runtime") != "OpenMV 5.0.1 / MicroPython 1.28.0-64":
        raise ValueError("unexpected device runtime")

    interpreter = tf.lite.Interpreter(
        model_path=str(FOMO_MODEL), num_threads=2,
        experimental_op_resolver_type=tf.lite.experimental.OpResolverType.BUILTIN_REF,
    )
    interpreter.allocate_tensors()
    in_detail = interpreter.get_input_details()[0]
    out_detail = interpreter.get_output_details()[0]
    pixels = np.frombuffer(raw, dtype="<u2").reshape(240, 240)
    rgb = native_rgb(pixels, width=96, height=96)
    scale, zero = in_detail["quantization"]
    host_input = np.clip(
        np.rint(rgb.astype(np.float64) / 255.0 / float(scale)) + int(zero),
        -128, 127,
    ).astype(np.int8)
    interpreter.set_tensor(in_detail["index"], host_input[None, ...])
    interpreter.invoke()
    host_output = interpreter.get_tensor(out_detail["index"])[0]
    out_scale = float(out_detail["quantization_parameters"]["scales"][0])
    out_zero = int(out_detail["quantization_parameters"]["zero_points"][0])
    scores = (host_output.astype(np.int16) - out_zero) * out_scale
    face_scores = scores[:, :, 1]
    threshold = 0.5

    output = output.resolve()
    output.relative_to(STUDY.resolve())
    private = STUDY / "private/device_parity_runs" / output.stem
    private.mkdir(parents=True, exist_ok=True)
    (private / "real_frame_rgb565.bin").write_bytes(raw)
    (private / "host_fomo_input_i8.bin").write_bytes(host_input.tobytes())
    (private / "host_fomo_output_i8.bin").write_bytes(host_output.tobytes())

    device_no_face = metadata.get("crop_status") == "no_face" and metadata.get("fomo_face_count") == 0
    device_has_candidate = isinstance(metadata.get("fomo_face_count"), int) and metadata["fomo_face_count"] > 0
    host_no_face = float(face_scores.max()) < threshold
    host_has_candidate = bool((face_scores >= threshold).any())
    if device_no_face and host_no_face:
        status = "reference_agrees_no_face_at_threshold"
    elif device_has_candidate and host_has_candidate:
        status = "device_and_reference_have_face_candidates"
    else:
        status = "device_host_detector_disagreement"
    program_path = (STUDY / "reports/M6/corrected_saved_frame_parity_v1/DEVICE_PROGRAM_V4.py")
    report = {
        "schema": "v6-real-camera-fomo-reference-check-1",
        "status": status,
        "transcript_path": transcript.resolve().relative_to(ROOT).as_posix(),
        "device_program_sha256": sha(program_path.read_bytes()),
        "camera_frame": {"geometry": [240, 240], "pixel_format": "RGB565",
                         "bytes": len(raw), "sha256": sha(raw)},
        "device": {
            "runtime": metadata.get("runtime"),
            "camera_id": metadata.get("camera_id"),
            "fomo_result": metadata.get("fomo_result"),
            "fomo_face_count": metadata.get("fomo_face_count"),
            "crop_status": metadata.get("crop_status"),
            "fomo_error": metadata.get("fomo_error"),
            "encoder_attempted": metadata.get("encoder_status") is not None,
        },
        "host_reference": {
            "backend": "TensorFlow 2.10.1 BUILTIN_REF",
            "model_sha256": sha(FOMO_MODEL.read_bytes()),
            "input_shape": [1, 96, 96, 3],
            "input_sha256": sha(host_input.tobytes()),
            "output_shape": [1, 12, 12, 2],
            "output_sha256": sha(host_output.tobytes()),
            "face_output_channel": 1,
            "max_face_score": float(face_scores.max()),
            "cells_at_or_above_0_5": int((face_scores >= threshold).sum()),
            "threshold": threshold,
            "has_face_candidate_at_threshold": host_has_candidate,
            "matches_device_detection_presence": bool(device_has_candidate == host_has_candidate and
                                                       metadata.get("fomo_error") is None),
        },
        "private_artifacts": private.relative_to(ROOT).as_posix(),
        "source_role": "development K01 screen target, diagnostic exposure only",
        "future_final_used": False,
        "identity_input_captured": "tensor_int8" in blobs,
        "identity_inference_performed": "embedding_int8" in blobs,
        "candidate_deployable": False,
        "ready_for_final_build": False,
        "limitations": [
            "The FOMO input/output tensors were not captured on the device in this V4 attempt.",
            "This reference check is not exact device FOMO parity or an accuracy result.",
            "FOMO clusters were retained as returned; no cluster was merged.",
        ],
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n")
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--transcript", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = verify(args.transcript, args.output)
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
