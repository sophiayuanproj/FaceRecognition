"""Smoke-run every unique trained embedding export staged under /flash/qa_seed_.

One live grayscale crop is used for all candidates. Three identical-input
timings check runtime stability only; they are not a validation-accuracy test.
The host runner removes all staged files after this script exits.
"""

import gc
import json
import os
import time

import csi
import ml
import omv
from ml.preprocessing import Normalization

STAGING_DIRECTORY = "/flash/qa_seed_models"
FRAME_SIZE = 240
SETTLE_MS = 1500
REPEATS = 3


def emit(event, fields=None):
    row = {"schema_version": 1, "event": event, "ticks_ms": time.ticks_ms()}
    if fields:
        for key, value in fields.items():
            row[key] = value
    print(json.dumps(row))


def to_json(value):
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, bytes):
        try:
            return value.decode()
        except Exception:
            return repr(value)
    if isinstance(value, (tuple, list)):
        return [to_json(item) for item in value]
    return str(value)


def is_embedding(model):
    return (to_json(model.input_shape) == [[1, 96, 96, 1]] and
            to_json(model.output_shape) == [[1, 64]])


def probe_one(name, path, frame_crop):
    gc.collect()
    heap_before = gc.mem_free()
    model = None
    outputs = None
    model_input = None
    try:
        load_start = time.ticks_us()
        model = ml.Model(path)
        load_us = time.ticks_diff(time.ticks_us(), load_start)
        details = {
            "name": name,
            "path": path,
            "model_file_bytes": model.len,
            "tensor_arena_bytes": model.ram,
            "input_shape": to_json(model.input_shape),
            "input_dtype": to_json(model.input_dtype),
            "output_shape": to_json(model.output_shape),
            "output_dtype": to_json(model.output_dtype),
            "load_us": load_us,
            "gc_free_before_load_bytes": heap_before,
            "gc_free_after_load_bytes": gc.mem_free(),
        }
        if not is_embedding(model):
            details["status"] = "failed"
            details["error"] = "unexpected embedding tensor shape"
            emit("seed_model", details)
            return False

        model_input = Normalization()(frame_crop)
        timings = []
        norm = 0.0
        for _ in range(REPEATS):
            start = time.ticks_us()
            outputs = model.predict([model_input])
            timings.append(time.ticks_diff(time.ticks_us(), start))
            vector = outputs[0][0]
            norm = sum(float(vector[i]) * float(vector[i]) for i in range(64)) ** 0.5
            outputs = None
        details.update({
            "status": "passed" if norm > 0.0 else "failed",
            "inference_us_each": timings,
            "embedding_norm": norm,
            "repeats": REPEATS,
            "decision": "runtime_smoke_only_no_identity_or_accuracy_claim",
        })
        if norm <= 0.0:
            details["error"] = "all-zero embedding output"
        emit("seed_model", details)
        return norm > 0.0
    except Exception as error:
        emit("seed_model", {
            "name": name,
            "path": path,
            "status": "failed",
            "error": str(error)[:180],
            "gc_free_before_load_bytes": heap_before,
        })
        return False
    finally:
        outputs = None
        model_input = None
        model = None
        gc.collect()


def main():
    gc.collect()
    candidates = sorted(name for name in os.listdir(STAGING_DIRECTORY)
                        if name.endswith(".tflite"))
    emit("seed_matrix_start", {
        "openmv_version": omv.version_string(),
        "candidate_count": len(candidates),
        "repeats_per_candidate": REPEATS,
        "gc_free_bytes": gc.mem_free(),
    })
    if not candidates:
        raise RuntimeError("no staged qa_seed_ model files found")

    cam = csi.CSI()
    cam.reset()
    cam.pixformat(csi.RGB565)
    cam.framesize(csi.QVGA)
    cam.window((FRAME_SIZE, FRAME_SIZE))
    cam.auto_rotation(False)
    cam.framebuffers(1)
    cam.snapshot(time=SETTLE_MS)
    frame = cam.snapshot()
    frame_crop = frame.to_grayscale(roi=(72, 72, 96, 96), copy=True)

    passed = 0
    failed = 0
    for name in candidates:
        if probe_one(name, STAGING_DIRECTORY + "/" + name, frame_crop):
            passed += 1
        else:
            failed += 1

    frame_crop = None
    frame = None
    gc.collect()
    emit("seed_matrix_complete", {
        "candidates": len(candidates),
        "passed": passed,
        "failed": failed,
        "gc_free_bytes": gc.mem_free(),
        "images_written": 0,
    })


try:
    main()
except Exception as error:
    emit("seed_matrix_fatal_error", {"error": str(error)[:180]})
    raise
