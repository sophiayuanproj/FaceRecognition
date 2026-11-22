"""Reproduce host raw-byte transforms, INT8 inputs, FOMO output, and identity crops on Nicla.

Copy the frozen model to /flash/c_face_detector_seed-20261102.tflite and the
outputs/C/fixtures/selected_challenger tree to /flash/c_fixtures first. Run on
the actual Nicla Vision with OpenMV 5.0.1; host/JPEG previews are never inputs.
"""
import gc
import hashlib
import json
import math
import time
import image
import ml
import omv
from ml.postprocessing.edgeimpulse import Fomo
from ml.preprocessing import Normalization


MODEL_PATH = "/flash/c_face_detector_seed-20261102.tflite"
FIXTURE_ROOT = "/flash/c_fixtures/selected_candidate"
FIXTURE_IDS = (
    "seed-20261501-baseline",
    "seed-20261502-mirror_x",
    "seed-20261503-flip_y",
    "seed-20261504-rotate_90_ccw",
    "seed-20261505-center_jitter",
    "seed-20261506-low_light",
    "seed-20261507-blur",
    "seed-20261508-small_face",
    "seed-20261509-edge_clip",
    "seed-20261510-multiple_faces",
    "seed-20261601-real_pass_no_face",
    "seed-20261602-real_pass_no_face",
    "seed-20261603-real_pass_no_face",
)


def emit(event, fields):
    fields["schema"] = "job-c-nicla-fixture-parity-2"
    fields["event"] = event
    fields["ticks_ms"] = time.ticks_ms()
    print(json.dumps(fields))


def digest_bytes(value):
    return hashlib.sha256(value).hexdigest()


def flatten_bytes(value):
    out = bytearray()
    stack = [value]
    while stack:
        item = stack.pop()
        if hasattr(item, "shape") and len(item.shape) > 0:
            for index in range(item.shape[0] - 1, -1, -1):
                stack.append(item[index])
        elif hasattr(item, "__iter__") and not isinstance(item, (bytes, bytearray, str)):
            values = list(item)
            for index in range(len(values) - 1, -1, -1):
                stack.append(values[index])
        else:
            out.append(int(item) & 255)
    return bytes(out)


def read_bytes(path):
    with open(path, "rb") as stream:
        return stream.read()


def read_json(path):
    with open(path, "r") as stream:
        return json.load(stream)


def round_origin(center, side):
    return math.floor(center - side / 2.0 + 0.5)


def fomo_regions(outputs, labels):
    found = []
    for class_index, items in enumerate(outputs):
        if class_index == 0 or not items:
            continue
        label = labels[class_index] if labels and class_index < len(labels) else "class_" + str(class_index)
        if label.strip().lower() != "face":
            continue
        for rect, score in items:
            x, y, width, height = rect
            found.append({"class_index": class_index, "label": label,
                          "region_xywh_window": [int(x), int(y), int(width), int(height)],
                          "centroid_window_xy": [float(x + width / 2.0), float(y + height / 2.0)],
                          "score": float(score)})
    return found


def normalization_u8(frame, roi):
    bound = Normalization(roi=roi)(frame)
    raw = bytearray(96 * 96)
    bound(raw, (1, 96, 96, 1), ord("B"))
    return bound, bytes(raw)


def capture_native_outputs(model, inputs, outputs):
    # An active callback receives native tensor dtype/bytes; default predict()
    # dequantizes integer outputs to float32 and cannot support byte parity.
    return outputs


def main():
    gc.collect()
    model = ml.Model(MODEL_PATH)
    emit("model_loaded", {"openmv_version": omv.version_string(), "model_path": MODEL_PATH,
        "model_file_bytes": int(model.len), "tensor_arena_bytes": int(model.ram),
        "input_shape": model.input_shape, "input_dtype": model.input_dtype,
        "input_scale": model.input_scale, "input_zero_point": model.input_zero_point,
        "output_shape": model.output_shape, "output_dtype": model.output_dtype,
        "output_scale": model.output_scale, "output_zero_point": model.output_zero_point,
        "labels": model.labels})
    failures = 0
    for fixture_id in FIXTURE_IDS:
        folder = FIXTURE_ROOT + "/" + fixture_id
        fixture = read_json(folder + "/fixture.json")
        frame = image.Image(folder + "/camera_window_gray240.pgm", copy_to_fb=False)
        raw_frame = bytes(frame.bytearray())
        expected_frame = read_bytes(folder + "/camera_window_gray240.u8")
        frame_exact = raw_frame == expected_frame and frame.width() == 240 and frame.height() == 240 and frame.format() == image.GRAYSCALE

        detector_bound, detector_u8 = normalization_u8(frame, (0, 0, 240, 240))
        expected_detector_u8 = read_bytes(folder + "/detector_input_gray96.u8")
        detector_transform_exact = detector_u8 == expected_detector_u8
        expected_detector_pgm = image.Image(folder + "/detector_input_gray96.pgm", copy_to_fb=False)
        decoded_detector_pgm = bytes(expected_detector_pgm.bytearray())
        detector_preview_is_raw = decoded_detector_pgm == expected_detector_u8
        expected_quantized = read_bytes(folder + "/detector_input_int8.i8")
        quantized = bytes((value - 128) & 255 for value in detector_u8)
        quantized_exact = quantized == expected_quantized and abs(float(model.input_scale[0]) - 1.0 / 255.0) < 0.00001 and int(model.input_zero_point[0]) == -128

        start = time.ticks_us()
        raw_outputs = model.predict([detector_bound], callback=capture_native_outputs)
        inference_us = time.ticks_diff(time.ticks_us(), start)
        actual_output = flatten_bytes(raw_outputs[0])
        expected_output = read_bytes(folder + "/detector_output_int8.i8")
        output_exact = actual_output == expected_output
        differing = 0
        max_abs_delta = 0
        if len(actual_output) == len(expected_output):
            for index in range(len(actual_output)):
                delta = abs(actual_output[index] - expected_output[index])
                if delta:
                    differing += 1
                    if delta > max_abs_delta:
                        max_abs_delta = delta

        threshold = float(fixture["expected_detector_output"]["threshold"])
        postprocessed = Fomo(threshold=threshold)(model, [detector_bound], raw_outputs)
        regions = fomo_regions(postprocessed, model.labels)
        expected_detections = fixture["expected_detector_output"]["detections"]
        expected_rects = [row["nms_rect_window_xywh"] for row in expected_detections]
        actual_rects = [row["region_xywh_window"] for row in regions]
        fomo_geometry_exact = actual_rects == expected_rects

        expected_crop = fixture["expected_crop_geometry"].get("crop_xywh_window")
        identity_exact = None
        identity_sha = None
        identity_quality = None
        if expected_crop is not None and len(regions) == 1:
            x, y, side, _ = expected_crop
            identity_bound, identity_raw = normalization_u8(frame, (x, y, side, side))
            identity_path = folder + "/identity_input_gray96.u8"
            expected_identity = read_bytes(identity_path)
            identity_exact = identity_raw == expected_identity
            identity_sha = digest_bytes(identity_raw)
            total = total_sq = 0
            for value in identity_raw:
                total += value
                total_sq += value * value
            mean = total / float(len(identity_raw))
            stdev = math.sqrt(max(0.0, total_sq / float(len(identity_raw)) - mean * mean))
            identity_quality = {"mean_u8": mean, "stdev_u8": stdev,
                                "identity_input_quality_gate_pass": 12 <= mean <= 243 and stdev >= 5}

        passed = (frame_exact and detector_transform_exact and detector_preview_is_raw and quantized_exact and
                  output_exact and fomo_geometry_exact and identity_exact is not False)
        if not passed:
            failures += 1
        emit("fixture_result", {"fixture_id": fixture_id,
            "camera_window_raw_exact": frame_exact, "camera_window_sha256": digest_bytes(raw_frame),
            "detector_normalization_raw_exact": detector_transform_exact,
            "detector_pgm_is_lossless_preview": detector_preview_is_raw,
            "detector_u8_sha256": digest_bytes(detector_u8), "detector_int8_quantization_exact": quantized_exact,
            "host_output_sha256": digest_bytes(expected_output), "board_output_sha256": digest_bytes(actual_output),
            "output_exact_match": output_exact, "output_differing_byte_count": differing,
            "output_max_abs_int8_byte_delta": max_abs_delta, "openmv_fomo_rectangles_exact": fomo_geometry_exact,
            "expected_rects": expected_rects, "board_rects": actual_rects,
            "identity_input_raw_exact": identity_exact, "identity_input_sha256": identity_sha,
            "identity_input_quality": identity_quality, "inference_us": inference_us,
            "status": "pass" if passed else "raw_transform_model_or_crop_mismatch"})
        frame = None; expected_detector_pgm = None; detector_bound = None
        raw_outputs = None; postprocessed = None; regions = None
        gc.collect()
    emit("complete", {"fixture_count": len(FIXTURE_IDS), "failures": failures,
                       "exact_camera_to_detector_and_identity_transforms_required": True,
                       "output_tolerance_policy": "raw bytes, FOMO rectangles, and crop bytes are reported exactly; no mismatches are hidden"})
    if failures:
        raise RuntimeError("Nicla host-fixture parity failed")


try:
    main()
except Exception as error:
    emit("fatal_error", {"error": str(error)[:180]})
    raise
