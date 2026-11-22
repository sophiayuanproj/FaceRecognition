"""First detector -> fixed crop -> identity classifier pipeline for Nicla Vision.

Run this file on the board, with recognition_config.py installed in /flash or
with configuration embedded by the host launcher. Sending this script alone
does not also transfer its imported configuration file.
It uses OpenMV 5.0.1 APIs, not desktop Python modules. No identity is produced
until real exported models and their label files are present and validated.
"""

import csi
import gc
import json
import os
import time

import ml
from ml.postprocessing.edgeimpulse import Fomo
from ml.preprocessing import Normalization

import recognition_config as cfg


def emit(event, fields=None):
    record = {"schema_version": 1, "event": event, "ticks_ms": time.ticks_ms()}
    if fields:
        record.update(fields)
    print(json.dumps(record))


def elapsed(start):
    return time.ticks_diff(time.ticks_ms(), start)


def sample_heap(stats):
    # gc.mem_free() describes the Python heap, not every SRAM allocation on the
    # board. Model arenas and captured frames are reported separately.
    free = gc.mem_free()
    if "min_observed_python_heap_free_bytes" not in stats:
        stats["min_observed_python_heap_free_bytes"] = free
    else:
        stats["min_observed_python_heap_free_bytes"] = min(
            stats["min_observed_python_heap_free_bytes"], free
        )
    return free


def require_model_files(path):
    if not path.endswith(".tflite") or "." in path[:-7]:
        raise ValueError("Use a .tflite path with no other dots: " + path)
    label_path = path[:-7] + ".txt"
    for required in (path, label_path):
        try:
            os.stat(required)
        except OSError:
            raise OSError(
                "Missing " + required + ". Train and export the Edge Impulse "
                "OpenMV INT8 model, then put its .tflite and .txt in /rom."
            )


def validate_config():
    for name in ("DETECTION_THRESHOLD", "IDENTITY_MIN_SCORE", "IDENTITY_MIN_MARGIN"):
        value = getattr(cfg, name)
        if value != value or value < 0.0 or value > 1.0:
            raise ValueError(name + " must be between 0 and 1")
    if not isinstance(cfg.IDENTITY_CROP_SIDE_PIXELS, int) or cfg.IDENTITY_CROP_SIDE_PIXELS < 1:
        raise ValueError("IDENTITY_CROP_SIDE_PIXELS must be a positive integer")
    if cfg.FRAME_PERIOD_MS < 0:
        raise ValueError("FRAME_PERIOD_MS cannot be negative")
    if not isinstance(cfg.EXPOSURE_SETTLE_MS, int) or cfg.EXPOSURE_SETTLE_MS < 0:
        raise ValueError("EXPOSURE_SETTLE_MS must be a nonnegative integer")
    if cfg.FRAMEBUFFER_COUNT < 1:
        raise ValueError("FRAMEBUFFER_COUNT must be positive")


def model_info(model, path, role):
    if len(model.input_shape) != 1 or tuple(model.input_shape[0]) != (1, 96, 96, 1):
        raise ValueError(role + " must have one (1,96,96,1) grayscale input")
    if model.input_dtype[0] != "b":
        raise ValueError(role + " input must be INT8 (OpenMV dtype 'b')")
    scale = float(model.input_scale[0])
    zero = int(model.input_zero_point[0])
    if scale != scale or scale <= 0.0 or abs(scale - cfg.EXPECTED_INPUT_SCALE) > cfg.INPUT_SCALE_TOLERANCE:
        raise ValueError(
            role + " input scale is incompatible with stock OpenMV grayscale "
            "conversion; expected approximately 1/255, got " + str(scale)
        )
    if zero != cfg.EXPECTED_INPUT_ZERO_POINT:
        raise ValueError(role + " input zero point must be -128, got " + str(zero))
    if len(model.output_shape) != 1:
        raise ValueError(role + " must have exactly one output tensor")
    if model.output_dtype[0] not in ("b", "B", "f"):
        raise ValueError(role + " has an unsupported output dtype")
    labels = [label.strip() for label in (model.labels or [])]
    if not labels or any(not label.strip() for label in labels):
        raise ValueError(role + " requires a nonempty sibling .txt label file")
    if len(set(labels)) != len(labels):
        raise ValueError(role + " label file contains duplicate labels")
    shape = tuple(model.output_shape[0])
    if role == "detector":
        if len(shape) != 4 or shape[0] != 1 or shape[1] < 1 or shape[2] < 1 or shape[3] < 2:
            raise ValueError("Detector output must be (1, grid_h, grid_w, classes+background)")
        # Exports may list the background explicitly or only foreground labels.
        if len(labels) == shape[3] - 1:
            labels = ["background"] + labels
        elif len(labels) != shape[3]:
            raise ValueError("Detector labels do not match its output channels")
    else:
        if len(shape) != 2 or shape[0] != 1 or shape[1] < 2 or shape[1] != len(labels):
            raise ValueError("Identity output must be (1, number_of_labels), with at least two labels")
    return {
        "role": role,
        "path": path,
        "model_file_bytes": int(model.len),
        "tensor_arena_bytes": int(model.ram),
        "input_shape": list(model.input_shape[0]),
        "input_dtype": model.input_dtype[0],
        "input_scale": scale,
        "input_zero_point": zero,
        "output_shape": list(shape),
        "output_dtype": model.output_dtype[0],
        "output_scale": float(model.output_scale[0]),
        "output_zero_point": int(model.output_zero_point[0]),
        "labels": labels,
    }


def face_class_index(info):
    labels = info["labels"]
    if cfg.DETECTOR_FACE_CLASS_INDEX is not None:
        index = cfg.DETECTOR_FACE_CLASS_INDEX
        if index < 1 or index >= len(labels):
            raise ValueError("DETECTOR_FACE_CLASS_INDEX must select a foreground output channel")
        return index
    matches = [i for i in range(1, len(labels)) if labels[i].strip().lower() == "face"]
    if len(matches) == 1:
        return matches[0]
    if len(labels) == 2:
        return 1
    raise ValueError("Cannot select the face class; set DETECTOR_FACE_CLASS_INDEX explicitly")


def square_scene_roi(img):
    width, height = img.width(), img.height()
    if cfg.SCENE_ROI is None:
        side = min(width, height)
        return ((width - side) // 2, (height - side) // 2, side, side)
    if len(cfg.SCENE_ROI) != 4:
        raise ValueError("SCENE_ROI must be (x,y,width,height)")
    x, y, w, h = cfg.SCENE_ROI
    if any(not isinstance(value, int) for value in cfg.SCENE_ROI):
        raise ValueError("SCENE_ROI coordinates must be integers")
    if w < 1 or w != h or x < 0 or y < 0 or x + w > width or y + h > height:
        raise ValueError("SCENE_ROI must be a square entirely inside the captured frame")
    return tuple(cfg.SCENE_ROI)


def identity_roi(img, center):
    side = cfg.IDENTITY_CROP_SIDE_PIXELS
    x = int(round(center[0] - side / 2.0))
    y = int(round(center[1] - side / 2.0))
    if x < 0 or y < 0 or x + side > img.width() or y + side > img.height():
        return None
    return (x, y, side, side)


class Models:
    def __init__(self):
        self.detector = None
        self.identity = None
        self.logged_roles = []

    def load(self, role, stats):
        model = self.detector if role == "detector" else self.identity
        start = time.ticks_ms()
        if model is None:
            gc.collect()
            sample_heap(stats)
            if role == "detector":
                path = cfg.DETECTOR_MODEL
                model = ml.Model(path, postprocess=Fomo(threshold=cfg.DETECTION_THRESHOLD))
            else:
                path = cfg.IDENTITY_MODEL
                model = ml.Model(path)
            info = model_info(model, path, role)
            if cfg.KEEP_MODELS_RESIDENT:
                if role == "detector":
                    self.detector = model
                else:
                    self.identity = model
        else:
            path = cfg.DETECTOR_MODEL if role == "detector" else cfg.IDENTITY_MODEL
            info = model_info(model, path, role)
        load_ms = elapsed(start)
        sample_heap(stats)
        if role not in self.logged_roles:
            emit("model_metadata", info)
            self.logged_roles.append(role)
        return model, info, load_ms

    def detect(self, img, roi, stats):
        model, info, load_ms = self.load("detector", stats)
        index = face_class_index(info)
        start = time.ticks_ms()
        output = model.predict([Normalization(roi=roi)(img)])
        inference_ms = elapsed(start)
        faces = []
        face_count = 0
        # Stock OpenMV Fomo/NMS has ALREADY mapped rectangles to source-frame
        # ROI coordinates. Their sizes are synthetic heat-map marker sizes.
        if output and index < len(output):
            for rect, score in output[index]:
                confidence = float(score)
                if confidence >= cfg.DETECTION_THRESHOLD:
                    face_count += 1
                    # Two records distinguish the supported single-face case
                    # from multiple faces without retaining an unbounded list.
                    if len(faces) < 2:
                        x, y, w, h = [int(v) for v in rect]
                        faces.append({"center": (x + w / 2.0, y + h / 2.0), "score": confidence})
        output = None
        sample_heap(stats)
        model = None
        if not cfg.KEEP_MODELS_RESIDENT:
            gc.collect()
        stats["detector_load_validate_ms"] = load_ms
        stats["detector_inference_ms"] = inference_ms
        stats["detector_tensor_arena_bytes"] = info["tensor_arena_bytes"]
        stats["face_count"] = face_count
        stats["face_records_truncated"] = face_count > len(faces)
        return faces

    def classify(self, img, roi, stats):
        model, info, load_ms = self.load("identity", stats)
        start = time.ticks_ms()
        output = model.predict([Normalization(roi=roi)(img)])
        inference_ms = elapsed(start)
        # With no callback/postprocessor, OpenMV predict() returns copied FLOAT
        # arrays, dequantizing integer outputs itself. Do not dequantize again.
        scores = [float(output[0][0][i]) for i in range(len(info["labels"]))]
        output = None
        sample_heap(stats)
        model = None
        if not cfg.KEEP_MODELS_RESIDENT:
            gc.collect()
        stats["identity_load_validate_ms"] = load_ms
        stats["identity_inference_ms"] = inference_ms
        stats["identity_tensor_arena_bytes"] = info["tensor_arena_bytes"]
        if any(value != value or value < -0.01 or value > 1.01 for value in scores):
            raise ValueError("Identity output does not contain finite class scores in [0,1]")
        if abs(sum(scores) - 1.0) > cfg.SOFTMAX_SUM_TOLERANCE:
            raise ValueError("Identity output does not resemble softmax; inspect the exported model")
        ranking = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
        best, second = ranking[0], ranking[1]
        label = info["labels"][best]
        margin = scores[best] - scores[second]
        result = {
            "status": "uncertain",
            "identity": None,
            "candidate_label": label,
            "score": scores[best],
            "margin": margin,
            "class_scores": dict(zip(info["labels"], scores)),
        }
        if scores[best] >= cfg.IDENTITY_MIN_SCORE and margin >= cfg.IDENTITY_MIN_MARGIN:
            if label.strip().lower() in tuple(value.strip().lower() for value in cfg.OTHER_LABELS):
                result["status"] = "unknown"
            else:
                result["status"] = "known"
                result["identity"] = label
        return result


def main():
    validate_config()
    require_model_files(cfg.DETECTOR_MODEL)
    require_model_files(cfg.IDENTITY_MODEL)
    camera = csi.CSI()
    camera.reset()
    camera.pixformat(csi.GRAYSCALE)
    camera.framesize(getattr(csi, cfg.CAMERA_FRAME_SIZE))
    camera.hmirror(cfg.HMIRROR)
    camera.vflip(cfg.VFLIP)
    camera.auto_rotation(False)
    camera.framebuffers(cfg.FRAMEBUFFER_COUNT)
    # Capture/discard warm-up frames. This argument is a settling interval,
    # not a camera exposure duration; snapshot(time=...) returns None.
    camera.snapshot(time=cfg.EXPOSURE_SETTLE_MS)
    models = Models()
    emit("recognition_started", {
        "model_loading": "resident" if cfg.KEEP_MODELS_RESIDENT else "sequential_reload",
        "frame_period_ms": cfg.FRAME_PERIOD_MS,
        "identity_crop_side_pixels": cfg.IDENTITY_CROP_SIDE_PIXELS,
        "thresholds_are_provisional": True,
        "heap_metric_is_not_total_board_ram": True,
    })
    frame_id = 0
    while True:
        started = time.ticks_ms()
        frame_id += 1
        gc.collect()
        stats = {"frame_id": frame_id, "python_heap_free_before_bytes": gc.mem_free()}
        sample_heap(stats)
        capture_started = time.ticks_ms()
        img = camera.snapshot()
        stats["capture_ms"] = elapsed(capture_started)
        stats["frame_width"] = img.width()
        stats["frame_height"] = img.height()
        stats["grayscale_frame_pixel_bytes"] = img.width() * img.height()
        scene_roi = square_scene_roi(img)
        stats["scene_roi"] = scene_roi
        faces = models.detect(img, scene_roi, stats)
        result = {"status": "no_face", "identity": None}
        crop = None
        if len(faces) > 1:
            result["status"] = "multiple_faces"
        elif len(faces) == 1:
            crop_started = time.ticks_ms()
            crop = identity_roi(img, faces[0]["center"])
            stats["crop_selection_ms"] = elapsed(crop_started)
            stats["face_center"] = faces[0]["center"]
            stats["detection_score"] = faces[0]["score"]
            stats["identity_roi"] = crop
            if crop is None:
                result["status"] = "crop_outside_frame"
            else:
                result = models.classify(img, crop, stats)
        stats.update(result)
        # Classification must consume the clean source frame. Drawing before
        # classify() would feed the overlay itself into the identity model.
        if cfg.DRAW_DEBUG_OVERLAY:
            draw_started = time.ticks_ms()
            img.draw_rectangle(scene_roi, color=128)
            for face in faces:
                cx, cy = face["center"]
                img.draw_circle((int(cx), int(cy), 4), color=255)
            if crop is not None:
                img.draw_rectangle(crop, color=255)
            img.draw_string((0, 0), result["status"], color=255)
            stats["debug_overlay_ms"] = elapsed(draw_started)
        sample_heap(stats)
        gc.collect()
        stats["python_heap_free_after_bytes"] = gc.mem_free()
        # Includes capture, loading, inference, decisions and debug drawing;
        # excludes the following serial print and pacing delay.
        stats["pipeline_ms_before_serial_log"] = elapsed(started)
        emit("recognition_frame", stats)
        remaining = cfg.FRAME_PERIOD_MS - elapsed(started)
        img = None
        if remaining > 0:
            time.sleep_ms(remaining)


try:
    main()
except Exception as error:
    emit("recognition_halted", {
        "error": str(error)[:160],
        "identity": None,
        "next_step": "Correct the configuration/model issue and rerun on the board.",
    })
    raise
