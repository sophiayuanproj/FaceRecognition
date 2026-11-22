"""Run repeated actual on-device INT8 embedding inference on a neutral scene.

This diagnostic forces a centered 96x96 crop through the model even when no
face is present. It proves model loading/inference, not a face or identity
result. It stores no image and retains no frame or embedding history.
"""

import gc
import json
import time

import csi
import ml
import omv
from ml.preprocessing import Normalization

import embedding_config as cfg

MAX_FRAMES = 100


def emit(event, fields=None):
    row = {"schema_version": 1, "event": event, "ticks_ms": time.ticks_ms()}
    if fields:
        row.update(fields)
    print(json.dumps(row))


def main():
    gc.collect()
    load_start = time.ticks_us()
    model = ml.Model(cfg.EMBEDDING_MODEL)
    load_us = time.ticks_diff(time.ticks_us(), load_start)
    emit("embedding_model_loaded", {
        "openmv_version": omv.version_string(),
        "model_path": cfg.EMBEDDING_MODEL,
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
        "load_us": load_us,
        "python_heap_free_bytes": gc.mem_free(),
        "measurement_scope": "tensor arena and MicroPython heap only",
    })
    cam = csi.CSI()
    cam.reset()
    cam.pixformat(csi.GRAYSCALE)
    cam.framesize(getattr(csi, cfg.FRAME_SIZE))
    cam.auto_rotation(False)
    cam.hmirror(cfg.HMIRROR)
    cam.vflip(cfg.VFLIP)
    cam.framebuffers(1)
    cam.snapshot(time=cfg.SETTLE_MS)
    roi = (112, 72, 96, 96)
    minimum_heap = gc.mem_free()
    embed_times = []
    capture_times = []
    for frame_id in range(MAX_FRAMES):
        gc.collect()
        started = time.ticks_us()
        frame = cam.snapshot()
        capture_us = time.ticks_diff(time.ticks_us(), started)
        capture_times.append(capture_us)
        started = time.ticks_us()
        output = model.predict([Normalization(roi=roi)(frame)])
        inference_us = time.ticks_diff(time.ticks_us(), started)
        embed_times.append(inference_us)
        vector = [float(output[0][0][i]) for i in range(len(output[0][0]))]
        norm = sum(value * value for value in vector) ** 0.5
        output = None
        vector = None
        frame = None
        gc.collect()
        minimum_heap = min(minimum_heap, gc.mem_free())
        emit("embedding_smoke_frame", {
            "frame_id": frame_id,
            "capture_us": capture_us,
            "embedding_inference_us": inference_us,
            "embedding_dimensions": len(output[0][0]) if output else model.output_shape[0][1],
            "embedding_norm": norm,
            "min_observed_python_heap_free_bytes": minimum_heap,
            "input_region": roi,
            "decision": "diagnostic_only_no_identity_claim",
        })
    emit("embedding_smoke_complete", {
        "frames": MAX_FRAMES,
        "mean_capture_us": sum(capture_times) // len(capture_times),
        "mean_embedding_inference_us": sum(embed_times) // len(embed_times),
        "min_observed_python_heap_free_bytes": minimum_heap,
    })


try:
    main()
except Exception as error:
    emit("embedding_smoke_failed", {"error": str(error)[:160]})
    raise
