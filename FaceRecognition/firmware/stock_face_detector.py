"""Stock RGB FOMO face-presence probe for Nicla Vision / OpenMV 5.0.1.

This is a bring-up baseline, not a trained project model or face recognizer.
FOMO output regions are heatmap regions, not calibrated full-face crop boxes.
"""

import gc
import json
import time
import csi
import ml
import omv
from ml.postprocessing.edgeimpulse import Fomo

MODEL_PATH = "/rom/fomo_face_detection.tflite"
DETECTION_THRESHOLD = 0.4
SAMPLE_PERIOD_MS = 500
SETTLE_MS = 2000
MAX_FRAMES = 0
MAX_REPORTED_DETECTIONS = 8
HMIRROR = False
VFLIP = False


def emit(event, fields):
    fields["schema_version"] = 1
    fields["event"] = event
    fields["ticks_ms"] = time.ticks_ms()
    print(json.dumps(fields))


def summarize_detections(model, outputs):
    records = []
    detection_count = 0
    labels = model.labels
    for class_index, detection_list in enumerate(outputs):
        if class_index == 0 or not detection_list:
            continue
        for rect, score in detection_list:
            detection_count += 1
            if len(records) >= MAX_REPORTED_DETECTIONS:
                continue
            x, y, width, height = rect
            label = "class_" + str(class_index)
            if labels and class_index < len(labels):
                label = labels[class_index][:48]
            records.append({
                "class_index": class_index,
                "label": label,
                "region_xywh": [x, y, width, height],
                "centroid_xy": [x + width / 2, y + height / 2],
                "score": float(score),
            })
    return records, detection_count


def main():
    gc.collect()
    emit("stock_detector_start", {
        "openmv_version": omv.version_string(),
        "model_path": MODEL_PATH,
        "threshold": DETECTION_THRESHOLD,
        "pixel_format": "RGB565",
        "sensor_frame_size": [320, 240],
        "centered_window": [240, 240],
        "framebuffer_count": 1,
        "sample_period_ms": SAMPLE_PERIOD_MS,
    })

    cam = csi.CSI()
    cam.reset()
    cam.pixformat(csi.RGB565)
    cam.framesize(csi.QVGA)
    cam.window((240, 240))
    cam.auto_rotation(False)
    cam.hmirror(HMIRROR)
    cam.vflip(VFLIP)
    cam.framebuffers(1)
    cam.snapshot(time=SETTLE_MS)

    load_start_us = time.ticks_us()
    model = ml.Model(MODEL_PATH, postprocess=Fomo(threshold=DETECTION_THRESHOLD))
    load_us = time.ticks_diff(time.ticks_us(), load_start_us)
    emit("stock_model_loaded", {
        "load_us": load_us,
        "model_file_bytes": model.len,
        "tensor_arena_bytes": model.ram,
        "input_shape": model.input_shape,
        "input_dtype": model.input_dtype,
        "input_scale": model.input_scale,
        "input_zero_point": model.input_zero_point,
        "output_shape": model.output_shape,
        "output_dtype": model.output_dtype,
        "gc_free_bytes": gc.mem_free(),
    })

    frame_index = 0
    while MAX_FRAMES == 0 or frame_index < MAX_FRAMES:
        cycle_start_ms = time.ticks_ms()
        capture_start_us = time.ticks_us()
        img = cam.snapshot()
        capture_us = time.ticks_diff(time.ticks_us(), capture_start_us)
        frame_ready_ticks_ms = time.ticks_ms()

        # No drawing, filtering, or mutation before prediction. With one buffer,
        # finish this prediction before taking another snapshot.
        inference_start_us = time.ticks_us()
        outputs = model.predict([img])
        inference_us = time.ticks_diff(time.ticks_us(), inference_start_us)
        records, detection_count = summarize_detections(model, outputs)
        emit("stock_face_detections", {
            "frame_index": frame_index,
            "frame_ready_ticks_ms": frame_ready_ticks_ms,
            "capture_us": capture_us,
            "inference_us": inference_us,
            "detection_count": detection_count,
            "detections": records,
            "detections_truncated": detection_count > len(records),
            "gc_free_bytes": gc.mem_free(),
        })
        # Retain no frame or inference history between iterations.
        outputs = None
        records = None
        frame_index += 1

        elapsed_ms = time.ticks_diff(time.ticks_ms(), cycle_start_ms)
        remaining_ms = SAMPLE_PERIOD_MS - elapsed_ms
        if remaining_ms > 0:
            time.sleep_ms(remaining_ms)

    emit("stock_detector_complete", {"frames": frame_index})


try:
    main()
except Exception as error:
    emit("fatal_error", {
        "component": "stock_face_detector",
        "model_path": MODEL_PATH,
        "error": str(error)[:160],
    })
    raise
