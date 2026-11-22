"""Bounded Nicla Vision capture/detector/crop smoke for Job C -> H.

Run on the actual Arduino Nicla Vision with OpenMV 5.0.1. This logs raw frame
and identity crop hashes but never saves camera images or invokes identity.
"""
import gc
import hashlib
import json
import math
import time
import csi
import ml
import omv
from ml.postprocessing.edgeimpulse import Fomo
from ml.preprocessing import Normalization


STOCK_MODEL_PATH = "/rom/fomo_face_detection.tflite"
CANDIDATE_MODEL_PATH = "/flash/c_face_detector_seed-20261102.tflite"
STOCK_THRESHOLD = 0.40
CHALLENGER_THRESHOLD = 0.70  # Experimental challenger; not cleared for app deployment.
STOCK_CROP_SIDE = 96
CHALLENGER_CROP_SIDE = 96
STOCK_CROP_POLICY = "legacy_current"
CHALLENGER_CROP_POLICY = "candidate"
MAX_FRAMES_PER_STAGE = 20
SAMPLE_PERIOD_MS = 300
SETTLE_MS = 1500
HMIRROR = False
VFLIP = False


def emit(event, fields):
    fields["schema"] = "job-c-nicla-scene-crop-smoke-1"
    fields["event"] = event
    fields["ticks_ms"] = time.ticks_ms()
    print(json.dumps(fields))


def digest_bytes(value):
    return hashlib.sha256(value).hexdigest()


def detector_regions(outputs, labels):
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


def prepare_camera(cam, pixel_format):
    cam.reset()
    cam.pixformat(pixel_format)
    cam.framesize(csi.QVGA)
    cam.window((240, 240))
    cam.auto_rotation(False)
    cam.hmirror(HMIRROR)
    cam.vflip(VFLIP)
    cam.framebuffers(1)
    cam.snapshot(time=SETTLE_MS)


def run_stage(cam, name, pixel_format, model_path, threshold, is_rgb, crop_side, crop_policy):
    if crop_policy not in ("legacy_current", "candidate"):
        raise ValueError("Unsupported frozen crop policy")
    gc.collect()
    prepare_camera(cam, pixel_format)
    start = time.ticks_us()
    model = ml.Model(model_path, postprocess=Fomo(threshold=threshold))
    load_us = time.ticks_diff(time.ticks_us(), start)
    emit("model_loaded", {"stage": name, "path": model_path, "threshold": threshold,
        "model_load_us": load_us, "model_file_bytes": int(model.len),
        "tensor_arena_bytes": int(model.ram), "input_shape": model.input_shape,
        "input_dtype": model.input_dtype, "input_scale": model.input_scale,
        "input_zero_point": model.input_zero_point, "output_shape": model.output_shape,
        "output_dtype": model.output_dtype, "output_scale": model.output_scale,
        "output_zero_point": model.output_zero_point, "labels": model.labels,
        "camera_pixel_format": "RGB565" if is_rgb else "GRAYSCALE",
        "camera_window": [240, 240], "hmirror": HMIRROR, "vflip": VFLIP,
        "crop_side_window_px": crop_side, "crop_policy": crop_policy,
        "openmv_version": omv.version_string()})
    frame_index = 0
    while frame_index < MAX_FRAMES_PER_STAGE:
        cycle_start = time.ticks_ms()
        frame_start = time.ticks_us()
        frame = cam.snapshot()
        capture_us = time.ticks_diff(time.ticks_us(), frame_start)
        if frame.width() != 240 or frame.height() != 240:
            raise ValueError("Camera window did not produce a 240x240 frame")
        if is_rgb:
            input_image = frame
        else:
            input_image = frame
        norm = Normalization(roi=(0, 0, frame.width(), frame.height()))(input_image)
        infer_start = time.ticks_us()
        outputs = model.predict([norm])
        inference_us = time.ticks_diff(time.ticks_us(), infer_start)
        records = detector_regions(outputs, model.labels)
        raw_frame_bytes = bytes(frame.bytearray())
        raw_frame_sha = digest_bytes(raw_frame_bytes)
        status = "NO_FACE" if len(records) == 0 else "UNCERTAIN_MULTIPLE_FACES" if len(records) != 1 else "CROP_READY"
        crop_fields = {"crop_xywh_window": None, "identity_crop_sha256": None,
                       "identity_input_shape": None, "identity_input_mean_u8": None,
                       "identity_input_stdev_u8": None, "identity_input_quality_gate_pass": None,
                       "native_crop_mean_u8": None, "native_crop_stdev_u8": None,
                       "native_crop_quality_gate_pass": None}
        if len(records) == 1:
            rect_x, rect_y, rect_w, rect_h = records[0]["region_xywh_window"]
            if crop_policy == "legacy_current":
                cx, cy = rect_x + rect_w // 2, rect_y + rect_h // 2
                x, y = cx - crop_side // 2, cy - crop_side // 2
            else:
                cx, cy = records[0]["centroid_window_xy"]
                x, y = math.floor(cx - crop_side / 2.0 + 0.5), math.floor(cy - crop_side / 2.0 + 0.5)
            crop_fields["crop_xywh_window"] = [x, y, crop_side, crop_side]
            if x < 0 or y < 0 or x + crop_side > 240 or y + crop_side > 240:
                status = "UNCERTAIN_CROP_OUTSIDE"
            else:
                gray_crop = frame.to_grayscale(roi=(x, y, crop_side, crop_side), copy=True)
                native_stats = gray_crop.get_statistics()
                native_mean, native_stdev = float(native_stats.mean), float(native_stats.stdev)
                bound = Normalization()(gray_crop)
                identity_raw = bytearray(96 * 96)
                bound(identity_raw, (1, 96, 96, 1), ord("B"))
                total = total_sq = 0
                for value in identity_raw:
                    total += value
                    total_sq += value * value
                mean = total / float(len(identity_raw))
                stdev = math.sqrt(max(0.0, total_sq / float(len(identity_raw)) - mean * mean))
                native_quality_pass = 12 <= native_mean <= 243 and native_stdev >= 5
                input_quality_pass = 12 <= mean <= 243 and stdev >= 5
                crop_fields["identity_crop_sha256"] = digest_bytes(identity_raw)
                crop_fields["identity_input_shape"] = [96, 96]
                crop_fields["identity_input_mean_u8"] = mean
                crop_fields["identity_input_stdev_u8"] = stdev
                crop_fields["native_crop_mean_u8"] = native_mean
                crop_fields["native_crop_stdev_u8"] = native_stdev
                crop_fields["native_crop_quality_gate_pass"] = native_quality_pass
                crop_fields["identity_input_quality_gate_pass"] = input_quality_pass
                if not native_quality_pass or not input_quality_pass:
                    status = "UNCERTAIN_LOW_QUALITY_CROP"
                gray_crop = None
        emit("scene_crop_result", {"stage": name, "frame_index": frame_index,
            "camera_frame_window": [frame.width(), frame.height()], "capture_us": capture_us,
            "inference_us": inference_us, "raw_frame_sha256": raw_frame_sha,
            "detection_count": len(records), "detections": records,
            "status": status, "identity_model_invoked": False, **crop_fields,
            "gc_free_bytes": gc.mem_free()})
        frame_index += 1
        outputs = None; records = None; norm = None; input_image = None; frame = None
        gc.collect()
        delay = SAMPLE_PERIOD_MS - time.ticks_diff(time.ticks_ms(), cycle_start)
        if delay > 0:
            time.sleep_ms(delay)
    model = None
    gc.collect()
    emit("stage_complete", {"stage": name, "frames": frame_index,
                             "gc_free_bytes_after_unload": gc.mem_free()})


def main():
    cam = csi.CSI()
    # Follow the current recognizer's grayscale QVGA capture path. The stock
    # FOMO artifact itself still has a 3-channel RGB input contract.
    run_stage(cam, "stock_rgb_fomo_on_current_grayscale_capture", csi.GRAYSCALE,
              STOCK_MODEL_PATH, STOCK_THRESHOLD, False,
              STOCK_CROP_SIDE, STOCK_CROP_POLICY)
    run_stage(cam, "development_gray_challenger", csi.GRAYSCALE, CANDIDATE_MODEL_PATH,
              CHALLENGER_THRESHOLD, False, CHALLENGER_CROP_SIDE, CHALLENGER_CROP_POLICY)
    emit("complete", {"bounded_frames_per_stage": MAX_FRAMES_PER_STAGE,
                      "one_face_scope": True, "tracking": False})


try:
    main()
except Exception as error:
    emit("fatal_error", {"error": str(error)[:180]})
    raise
