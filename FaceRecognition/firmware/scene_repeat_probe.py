"""Ten-frame headless scene stability check for stock FOMO and person models."""

import gc
import json
import time

import csi
import ml
import omv
from ml.postprocessing.edgeimpulse import Fomo

FRAME_SIZE = 240
SETTLE_MS = 1500
FRAMES = 10
SAMPLE_PERIOD_MS = 200


def emit(event, fields=None):
    row = {"schema_version": 1, "event": event, "ticks_ms": time.ticks_ms()}
    if fields:
        row.update(fields)
    print(json.dumps(row))


def median(values):
    return sorted(values)[len(values) // 2]


def run_fomo(cam):
    gc.collect()
    model = ml.Model("/rom/fomo_face_detection.tflite",
                     postprocess=Fomo(threshold=0.4))
    counts = []
    latencies = []
    max_scores = []
    capture_times = []
    for _ in range(FRAMES):
        cycle_started_ms = time.ticks_ms()
        frame_started_us = time.ticks_us()
        frame = cam.snapshot()
        capture_times.append(time.ticks_diff(time.ticks_us(), frame_started_us))
        started = time.ticks_us()
        outputs = model.predict([frame])
        latencies.append(time.ticks_diff(time.ticks_us(), started))
        count = 0
        frame_max = 0.0
        for class_index, detections in enumerate(outputs):
            if class_index == 0:
                continue
            for _, score in detections:
                count += 1
                frame_max = max(frame_max, float(score))
        counts.append(count)
        max_scores.append(frame_max)
        outputs = None
        frame = None
        elapsed_ms = time.ticks_diff(time.ticks_ms(), cycle_started_ms)
        remaining_ms = SAMPLE_PERIOD_MS - elapsed_ms
        if remaining_ms > 0:
            time.sleep_ms(remaining_ms)
    model = None
    gc.collect()
    emit("fomo_repeat_summary", {
        "frames": FRAMES,
        "frames_with_detections": sum(1 for value in counts if value > 0),
        "detection_counts": counts,
        "max_scores": max_scores,
        "median_inference_us": median(latencies),
        "median_capture_us": median(capture_times),
        "gc_free_after_unload_bytes": gc.mem_free(),
        "decision": "live_scene_probe_not_accuracy_validation",
    })


def run_person(cam):
    gc.collect()
    model = ml.Model("/rom/person_detect.tflite")
    labels = model.labels or ["class_" + str(index) for index in range(2)]
    predictions = []
    latencies = []
    capture_times = []
    for _ in range(FRAMES):
        cycle_started_ms = time.ticks_ms()
        frame_started_us = time.ticks_us()
        frame = cam.snapshot()
        capture_times.append(time.ticks_diff(time.ticks_us(), frame_started_us))
        started = time.ticks_us()
        output = model.predict([frame])
        latencies.append(time.ticks_diff(time.ticks_us(), started))
        scores = [float(output[0][0][index]) for index in range(2)]
        predictions.append({
            "scores": scores,
            "top_label": labels[scores.index(max(scores))],
        })
        output = None
        frame = None
        elapsed_ms = time.ticks_diff(time.ticks_ms(), cycle_started_ms)
        remaining_ms = SAMPLE_PERIOD_MS - elapsed_ms
        if remaining_ms > 0:
            time.sleep_ms(remaining_ms)
    model = None
    gc.collect()
    emit("person_repeat_summary", {
        "frames": FRAMES,
        "labels": labels,
        "predictions": predictions,
        "median_inference_us": median(latencies),
        "median_capture_us": median(capture_times),
        "gc_free_after_unload_bytes": gc.mem_free(),
        "decision": "stock_person_classifier_probe_no_threshold_claim",
    })


def main():
    emit("scene_repeat_start", {
        "openmv_version": omv.version_string(),
        "frames_per_model": FRAMES,
        "sample_period_ms": SAMPLE_PERIOD_MS,
        "frame_size": [FRAME_SIZE, FRAME_SIZE],
        "images_written": 0,
    })
    cam = csi.CSI()
    cam.reset()
    cam.pixformat(csi.RGB565)
    cam.framesize(csi.QVGA)
    cam.window((FRAME_SIZE, FRAME_SIZE))
    cam.auto_rotation(False)
    cam.framebuffers(1)
    cam.snapshot(time=SETTLE_MS)
    run_fomo(cam)
    run_person(cam)
    emit("scene_repeat_complete", {"gc_free_bytes": gc.mem_free()})


try:
    main()
except Exception as error:
    emit("scene_repeat_failed", {"error": str(error)[:180]})
    raise
