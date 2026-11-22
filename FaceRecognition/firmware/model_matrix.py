"""Bounded, headless on-board model and cascade smoke matrix.

Run with `bin/nicla model-matrix PORT`. It captures one in-memory frame and
tests one model at a time. No image is saved, and no board files are changed.
The optional /flash test models are staged by the host and cleaned up after the
run; this script only reads them.
"""

import gc
import json
import time

import csi
import image
import ml
import omv
from ml.preprocessing import Normalization

FRAME_SIZE = 240
SETTLE_MS = 1500
MAX_DETECTIONS_REPORTED = 8

MODELS = (
    ("base_int8", "/flash/identity_embedding_int8.tflite", "embedding"),
    ("tiny_int8", "/flash/qa_tiny_int8.tflite", "embedding"),
    ("base_float", "/flash/qa_base_float.tflite", "embedding"),
    ("tiny_float", "/flash/qa_tiny_float.tflite", "embedding"),
    ("stock_fomo_face", "/rom/fomo_face_detection.tflite", "fomo"),
    ("stock_blazeface", "/rom/blazeface_front_128.tflite", "blazeface"),
    ("stock_yolo_lc", "/rom/yolo_lc_192.tflite", "yolo_lc"),
    ("stock_person_detect", "/rom/person_detect.tflite", "image_raw"),
    ("stock_force_int_quant", "/rom/force_int_quant.tflite", "tensor_float_36"),
    ("stock_audio_preprocessor", "/rom/audio_preprocessor.tflite", "tensor_pcm16_480"),
    ("stock_micro_speech", "/rom/micro_speech.tflite", "tensor_int8_1960"),
)

CASCADES = (
    ("frontalface", "/rom/haarcascade_frontalface.cascade"),
    ("eye", "/rom/haarcascade_eye.cascade"),
    ("smile", "/rom/haarcascade_smile.cascade"),
)


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


def emit(event, fields=None):
    row = {"schema_version": 1, "event": event, "ticks_ms": time.ticks_ms()}
    if fields:
        for key, value in fields.items():
            row[key] = safe(value)
    print(json.dumps(row))


def model_metadata(model):
    return {
        "model_file_bytes": model.len,
        "tensor_arena_bytes": model.ram,
        "input_shape": model.input_shape,
        "input_dtype": model.input_dtype,
        "input_scale": model.input_scale,
        "input_zero_point": model.input_zero_point,
        "output_shape": model.output_shape,
        "output_dtype": model.output_dtype,
        "output_scale": model.output_scale,
        "output_zero_point": model.output_zero_point,
        "labels": model.labels,
    }


def output_shape_summary(value, depth=0):
    if hasattr(value, "shape"):
        return safe(value.shape)
    if isinstance(value, (tuple, list)):
        if depth >= 3:
            return {"container": type(value).__name__, "length": len(value)}
        return [output_shape_summary(item, depth + 1) for item in value[:3]]
    return type(value).__name__


def detection_count(outputs, kind):
    if kind == "blazeface":
        return len(outputs)
    if kind in ("fomo", "yolo_lc"):
        total = 0
        for class_detections in outputs:
            try:
                total += len(class_detections)
            except Exception:
                pass
        return total
    return None


def postprocessor(kind):
    if kind == "fomo":
        from ml.postprocessing.edgeimpulse import Fomo
        return Fomo(threshold=0.4)
    if kind == "blazeface":
        from ml.postprocessing.mediapipe import BlazeFace
        return BlazeFace(threshold=0.4)
    if kind == "yolo_lc":
        from ml.postprocessing.darknet import YoloLC
        return YoloLC(threshold=0.4)
    return None


def run_model(name, path, kind, frame, gray_crop):
    gc.collect()
    heap_before = gc.mem_free()
    start = time.ticks_us()
    model = None
    try:
        processor = postprocessor(kind)
        if processor is None:
            model = ml.Model(path)
        else:
            model = ml.Model(path, postprocess=processor)
        load_us = time.ticks_diff(time.ticks_us(), start)
        fields = model_metadata(model)
        fields.update({
            "name": name,
            "path": path,
            "load_us": load_us,
            "gc_free_before_load_bytes": heap_before,
            "gc_free_after_load_bytes": gc.mem_free(),
        })
        emit("model_loaded", fields)
        if kind == "embedding":
            model_input = Normalization()(gray_crop)
        elif kind == "tensor_float_36":
            from ulab import numpy as np
            model_input = np.zeros((1, 36), dtype=np.float)
        elif kind == "tensor_pcm16_480":
            from ulab import numpy as np
            model_input = np.zeros((1, 480), dtype=np.int16)
        elif kind == "tensor_int8_1960":
            from ulab import numpy as np
            # Quantized zero feature values map to the model's INT8 zero point.
            model_input = np.zeros((1, 1960), dtype=np.int8)
            for index in range(1960):
                model_input[0, index] = -128
        else:
            model_input = frame
        start = time.ticks_us()
        outputs = model.predict([model_input])
        inference_us = time.ticks_diff(time.ticks_us(), start)
        emit("model_inference", {
            "name": name,
            "inference_us": inference_us,
            "detection_count": detection_count(outputs, kind),
            "output_summary": output_shape_summary(outputs),
            "decision": "smoke_only_no_accuracy_or_identity_claim",
        })
        outputs = None
        model_input = None
    except Exception as error:
        emit("model_probe_failed", {"name": name, "path": path, "error": str(error)[:180]})
    finally:
        model = None
        gc.collect()
        emit("model_released", {"name": name, "gc_free_bytes": gc.mem_free()})


def run_cascade(name, path, gray):
    gc.collect()
    start = time.ticks_us()
    cascade = None
    try:
        cascade = image.HaarCascade(path)
        load_us = time.ticks_diff(time.ticks_us(), start)
        start = time.ticks_us()
        boxes = gray.find_features(cascade, threshold=0.5, scale=1.5)
        inference_us = time.ticks_diff(time.ticks_us(), start)
        emit("cascade_probe", {
            "name": name,
            "path": path,
            "load_us": load_us,
            "inference_us": inference_us,
            "detection_count": len(boxes),
            "detections": boxes[:MAX_DETECTIONS_REPORTED],
            "decision": "smoke_only_no_accuracy_claim",
        })
    except Exception as error:
        emit("cascade_probe_failed", {"name": name, "path": path, "error": str(error)[:180]})
    finally:
        cascade = None
        gc.collect()
        emit("cascade_released", {"name": name, "gc_free_bytes": gc.mem_free()})


def main():
    gc.collect()
    emit("matrix_start", {
        "openmv_version": omv.version_string(),
        "frame_size": [FRAME_SIZE, FRAME_SIZE],
        "gc_free_bytes": gc.mem_free(),
        "matrix_models": len(MODELS),
        "matrix_cascades": len(CASCADES),
        "image_storage": "one live camera frame only; no image files written",
        "noncamera_test_inputs": "zero feature tensors; smoke-only, not an audio benchmark",
    })

    cam = csi.CSI()
    cam.reset()
    cam.pixformat(csi.RGB565)
    cam.framesize(csi.QVGA)
    cam.window((FRAME_SIZE, FRAME_SIZE))
    cam.auto_rotation(False)
    cam.framebuffers(1)
    cam.snapshot(time=SETTLE_MS)
    frame = cam.snapshot()
    gray = frame.to_grayscale(copy=True)
    gray_crop = frame.to_grayscale(roi=(72, 72, 96, 96), copy=True)
    emit("camera_ready", {
        "frame_size": [frame.width(), frame.height()],
        "gray_crop": [72, 72, 96, 96],
        "gc_free_bytes": gc.mem_free(),
    })

    for name, path, kind in MODELS:
        run_model(name, path, kind, frame, gray_crop)
    gray_crop = None

    for name, path in CASCADES:
        run_cascade(name, path, gray)

    frame = None
    gray = None
    gc.collect()
    emit("matrix_complete", {"gc_free_bytes": gc.mem_free()})


try:
    main()
except Exception as error:
    emit("matrix_fatal_error", {"error": str(error)[:180]})
    raise
