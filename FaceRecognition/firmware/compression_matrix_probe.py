"""Headless board validation of the TFLite compression-format matrix.

Reads only temporary /flash/qa_compression_20261022 model files and one
in-memory camera crop. It never writes image data or modifies model files.
"""

import gc
import json
import math
import time

import csi
import ml
import omv
from ml.preprocessing import Normalization

STAGING = "/flash/qa_compression_20261022"
FRAME_SIZE = 240
SETTLE_MS = 1200
REPEATS = 5
MODELS = (
    ("base_float32", STAGING + "/base_float32.tflite"),
    ("base_float16_weights", STAGING + "/base_float16_weights.tflite"),
    ("base_dynamic_int8_weights", STAGING + "/base_dynamic_int8_weights.tflite"),
    ("base_full_int8_io", STAGING + "/base_full_int8_io.tflite"),
    ("tiny_float32", STAGING + "/tiny_float32.tflite"),
    ("tiny_float16_weights", STAGING + "/tiny_float16_weights.tflite"),
    ("tiny_dynamic_int8_weights", STAGING + "/tiny_dynamic_int8_weights.tflite"),
    ("tiny_full_int8_io", STAGING + "/tiny_full_int8_io.tflite"),
    ("magnitude_prune_50_dense", STAGING + "/magnitude_prune_50_dense.tflite"),
    ("magnitude_prune_50_sparse", STAGING + "/magnitude_prune_50_sparse.tflite"),
    ("structured_prune_2of4_dense", STAGING + "/structured_prune_2of4_dense.tflite"),
    ("structured_prune_2of4_sparse", STAGING + "/structured_prune_2of4_sparse.tflite"),
    ("cluster_16_full_int8", STAGING + "/cluster_16_full_int8.tflite"),
    ("qat_full_int8", STAGING + "/qat_full_int8.tflite"),
)


def finite(value):
    value = float(value)
    return value == value and value != float("inf") and value != -float("inf")


def safe(value):
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, bytes):
        try:
            return value.decode()
        except Exception:
            return repr(value)
    if isinstance(value, (tuple, list)):
        return [safe(item) for item in value]
    return str(value)


def shape_tuple(shape):
    while (isinstance(shape, (tuple, list)) and len(shape) == 1 and
           isinstance(shape[0], (tuple, list))):
        shape = shape[0]
    return tuple(int(value) for value in shape)


def emit(event, fields=None):
    row = {"schema_version": 1, "event": event, "ticks_ms": time.ticks_ms()}
    if fields:
        for key, value in fields.items():
            row[key] = safe(value)
    print(json.dumps(row))


def probe_one(name, path, crop, crop_stats):
    gc.collect()
    free_before = gc.mem_free()
    model = None
    model_input = None
    outputs = None
    previous = None
    max_delta = 0.0
    timings = []
    min_heap = free_before
    try:
        start = time.ticks_us()
        model = ml.Model(path)
        load_us = time.ticks_diff(time.ticks_us(), start)
        input_shape = shape_tuple(model.input_shape)
        output_shape = shape_tuple(model.output_shape)
        metadata = {
            "name": name,
            "path": path,
            "model_file_bytes": int(model.len),
            "tensor_arena_bytes": int(model.ram),
            "input_shape": safe(model.input_shape),
            "input_dtype": safe(model.input_dtype),
            "input_scale": safe(model.input_scale),
            "input_zero_point": safe(model.input_zero_point),
            "output_shape": safe(model.output_shape),
            "output_dtype": safe(model.output_dtype),
            "output_scale": safe(model.output_scale),
            "output_zero_point": safe(model.output_zero_point),
            "model_load_us": load_us,
            "python_heap_free_before_load_bytes": free_before,
            "python_heap_free_after_load_bytes": gc.mem_free(),
            "crop_mean_u8": int(crop_stats.mean),
            "crop_stdev_u8": int(crop_stats.stdev),
        }
        if input_shape != (1, 96, 96, 1) or output_shape != (1, 64):
            raise ValueError("unexpected embedding tensor shape")
        model_input = Normalization()(crop)
        for _ in range(REPEATS):
            start = time.ticks_us()
            outputs = model.predict([model_input])
            timings.append(time.ticks_diff(time.ticks_us(), start))
            vector = [float(value) for value in outputs[0][0]]
            outputs = None
            if len(vector) != 64 or not all(finite(value) for value in vector):
                raise ValueError("output is not a finite 64-value embedding")
            norm = sum(value * value for value in vector) ** 0.5
            if not finite(norm) or norm <= 0.0:
                raise ValueError("output embedding has zero or invalid norm")
            if previous is not None:
                delta = max(abs(vector[index] - previous[index]) for index in range(64))
                max_delta = max(max_delta, delta)
                if delta > 1e-6:
                    raise ValueError("same-input inference was not deterministic")
            previous = vector
            min_heap = min(min_heap, gc.mem_free())
        metadata.update({
            "status": "passed",
            "repeat_count": REPEATS,
            "inference_us_each": timings,
            "inference_us_median": sorted(timings)[len(timings) // 2],
            "embedding_norm": norm,
            "same_input_max_abs_delta": max_delta,
            "python_heap_min_during_inference_bytes": min_heap,
            "evidence_scope": "board format-load and numeric inference smoke; not accuracy or identity",
        })
        emit("compression_model_probe", metadata)
        return True
    except Exception as error:
        emit("compression_model_probe", {
            "name": name,
            "path": path,
            "status": "failed",
            "error": str(error)[:180],
            "python_heap_free_before_load_bytes": free_before,
            "python_heap_free_at_failure_bytes": gc.mem_free(),
            "evidence_scope": "board format-load and numeric inference smoke; not accuracy or identity",
        })
        return False
    finally:
        previous = None
        outputs = None
        model_input = None
        model = None
        gc.collect()


def main():
    gc.collect()
    import os
    names = sorted(os.listdir(STAGING))
    expected = sorted(path.rsplit("/", 1)[1] for name, path in MODELS)
    emit("compression_matrix_start", {
        "openmv_version": omv.version_string(),
        "candidate_count_expected": len(MODELS),
        "candidate_names_present": names,
        "repeat_count": REPEATS,
        "python_heap_free_bytes": gc.mem_free(),
        "images_saved": 0,
    })
    missing = [name for name in expected if name not in names]
    if missing:
        raise RuntimeError("missing staged models: " + repr(missing))

    cam = csi.CSI()
    cam.reset()
    cam.pixformat(csi.RGB565)
    cam.framesize(csi.QVGA)
    cam.window((FRAME_SIZE, FRAME_SIZE))
    cam.auto_rotation(False)
    cam.framebuffers(1)
    cam.snapshot(time=SETTLE_MS)
    frame = cam.snapshot()
    crop = frame.to_grayscale(roi=(72, 72, 96, 96), copy=True)
    stats = crop.get_statistics()
    passed = 0
    failed = 0
    for name, path in MODELS:
        if probe_one(name, path, crop, stats):
            passed += 1
        else:
            failed += 1
    crop = None
    frame = None
    gc.collect()
    emit("compression_matrix_complete", {
        "models": len(MODELS),
        "passed": passed,
        "failed": failed,
        "python_heap_free_bytes": gc.mem_free(),
        "images_saved": 0,
    })


try:
    main()
except Exception as error:
    emit("compression_matrix_fatal_error", {"error": str(error)[:180]})
    raise

