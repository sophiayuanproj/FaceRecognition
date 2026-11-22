"""Stock FOMO face presence followed by local INT8 embedding verification.

No raw frames are saved. The stock detector and custom embedder load one at a
time to limit peak memory. FOMO centroids are not calibrated face boxes; the
fixed crop is a course-project approximation that requires camera-domain
validation before it can support real-person recognition claims.
"""

import csi
import gc
import json
import pyb
import time

import ml
from machine import LED
from ml.postprocessing.edgeimpulse import Fomo
from ml.preprocessing import Normalization

import embedding_config as cfg
from identity_contracts import binary32

DETECTOR_METADATA_LOGGED = False
EMBEDDING_METADATA_LOGGED = False
USB_VCP = pyb.USB_VCP()
LED_RED = LED("LED_RED")
LED_GREEN = LED("LED_GREEN")
LED_BLUE = LED("LED_BLUE")


def emit(event, fields=None):
    # Never let an unattended board block on an undrained USB log buffer.
    if not cfg.SERIAL_LOGGING or not USB_VCP.isconnected():
        return
    record = {"schema_version": 1, "event": event, "ticks_ms": time.ticks_ms()}
    if fields:
        record.update(fields)
    print(json.dumps(record))


def elapsed_us(start):
    return time.ticks_diff(time.ticks_us(), start)


def model_metadata(model, role, path):
    return {
        "role": role,
        "path": path,
        "model_file_bytes": int(model.len),
        "tensor_arena_bytes": int(model.ram),
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


def shape_tuple(shape):
    while (isinstance(shape, (list, tuple)) and len(shape) == 1 and
           isinstance(shape[0], (list, tuple))):
        shape = shape[0]
    return tuple(int(value) for value in shape)


def face_output_index(model, output_count):
    labels = model.labels or []
    for index in range(min(len(labels), output_count)):
        if labels[index].strip().lower() == "face":
            return index
    # Some OpenMV FOMO postprocessors omit the background output while
    # model.labels still includes it. In that form, output 0 maps to label 1.
    for output_index in range(output_count):
        label_index = output_index + 1
        if (label_index < len(labels) and
                labels[label_index].strip().lower() == "face"):
            return output_index
    if output_count == 2:
        return 1
    raise ValueError("Cannot map stock FOMO outputs to labels: labels=" + repr(labels) +
                     " output_count=" + str(output_count))


def detect_faces(frame, stats):
    global DETECTOR_METADATA_LOGGED
    gc.collect()
    before = gc.mem_free()
    start = time.ticks_us()
    detector = ml.Model(cfg.DETECTOR_MODEL, postprocess=Fomo(threshold=cfg.DETECTION_THRESHOLD))
    load_us = elapsed_us(start)
    meta = model_metadata(detector, "stock_fomo_face_presence", cfg.DETECTOR_MODEL)
    stats["detector_heap_free_after_load_bytes"] = gc.mem_free()
    if shape_tuple(meta["input_shape"]) != (1, 96, 96, 3):
        raise ValueError("Stock detector input shape is incompatible: " + repr(meta["input_shape"]))
    stats["detector_load_us"] = load_us
    stats["detector_tensor_arena_bytes"] = meta["tensor_arena_bytes"]
    if not DETECTOR_METADATA_LOGGED:
        emit("model_metadata", meta)
        DETECTOR_METADATA_LOGGED = True
    roi = (0, 0, frame.width(), frame.height())
    start = time.ticks_us()
    output = detector.predict([Normalization(roi=roi)(frame)])
    stats["detector_inference_us"] = elapsed_us(start)
    index = face_output_index(detector, len(output)) if output else None
    faces = []
    total = 0
    if output and index is not None and index < len(output):
        for rect, score in output[index]:
            value = float(score)
            if value >= cfg.DETECTION_THRESHOLD:
                total += 1
                if len(faces) < 2:
                    x, y, width, height = [int(v) for v in rect]
                    faces.append({"center": (x + width // 2, y + height // 2), "score": value})
    output = None
    detector = None
    gc.collect()
    stats["detector_heap_free_before_load_bytes"] = before
    stats["detector_heap_free_after_unload_bytes"] = gc.mem_free()
    stats["face_count"] = total
    stats["face_records_truncated"] = total > len(faces)
    return faces


def valid_crop(frame, center):
    side = cfg.IDENTITY_CROP_SIDE_PIXELS
    x = int(center[0] - side // 2)
    y = int(center[1] - side // 2)
    if x < 0 or y < 0 or x + side > frame.width() or y + side > frame.height():
        return None
    return (x, y, side, side)


def valid_identity_id(value):
    if not isinstance(value, str) or not value or len(value) > 24:
        return False
    for ch in value:
        if not ("a" <= ch <= "z" or "A" <= ch <= "Z" or
                "0" <= ch <= "9" or ch in "_-"):
            return False
    return True


def finite_number(value):
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if number != number or number == float("inf") or number == -float("inf"):
        return None
    return number


def load_template_ids():
    try:
        with open(cfg.TEMPLATE_FILE, "r") as stream:
            data = json.load(stream)
    except OSError:
        return []
    if not isinstance(data, dict) or data.get("version") != 1 or not isinstance(data.get("entries"), list):
        raise ValueError("Unsupported enrollment template file")
    if len(data["entries"]) > 8:
        raise ValueError("Template roster exceeds the supported maximum of 8")
    ids = []
    for entry in data["entries"]:
        if not isinstance(entry, dict) or not valid_identity_id(entry.get("identity_id")):
            raise ValueError("Every template needs a pseudonymous identity_id")
        ids.append(entry["identity_id"])
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate identity_id in enrollment template file")
    return ids


def load_templates(model_id=None):
    try:
        with open(cfg.TEMPLATE_FILE, "r") as stream:
            data = json.load(stream)
    except OSError:
        return []
    if not isinstance(data, dict):
        raise ValueError("Enrollment template file must contain a JSON object")
    entries = data.get("entries")
    if data.get("version") != 1 or not isinstance(entries, list):
        raise ValueError("Unsupported enrollment template file")
    if len(entries) > 8:
        raise ValueError("Template roster exceeds the supported maximum of 8")
    normalized = []
    seen_ids = set()
    for entry in entries:
        if not isinstance(entry, dict):
            raise ValueError("Every template entry must be a JSON object")
        identity_id = entry.get("identity_id")
        if not valid_identity_id(identity_id):
            raise ValueError("Template identity_id must be a pseudonymous 1-24 character ASCII token")
        if identity_id in seen_ids:
            raise ValueError("Duplicate identity_id in enrollment template file")
        seen_ids.add(identity_id)
        vector = entry.get("vector")
        if model_id is not None:
            model_vectors = entry.get("vectors", {})
            if not isinstance(model_vectors, dict) or model_id not in model_vectors:
                raise ValueError("Enrollment template is missing a vector for model " + str(model_id))
            vector = model_vectors[model_id]
        if not isinstance(vector, list) or len(vector) != 64:
            raise ValueError("Every template must contain one 64-value embedding")
        values = []
        for value in vector:
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValueError("Template embedding values must be finite numbers")
            number = finite_number(value)
            if number is None:
                raise ValueError("Template embedding values must be finite numbers")
            values.append(number)
        norm = sum(value * value for value in values) ** 0.5
        if finite_number(norm) is None or norm <= 0.0:
            raise ValueError("Enrollment template has zero or invalid norm")
        normalized.append({"identity_id": identity_id,
                           "vector": [value / norm for value in values]})
    return normalized


def load_calibration():
    try:
        with open(cfg.CALIBRATION_FILE, "r") as stream:
            data = json.load(stream)
    except OSError:
        return None
    if not isinstance(data, dict):
        raise ValueError("Calibration file must contain a JSON object")
    return data


def calibration_threshold(model_id=None):
    data = load_calibration()
    if data is None:
        return None
    raw_value = data.get("threshold")
    if model_id is not None:
        thresholds = data.get("max_gallery_thresholds", {})
        if not isinstance(thresholds, dict) or model_id not in thresholds:
            return None
        raw_value = thresholds.get(model_id)
    if isinstance(raw_value, bool) or not isinstance(raw_value, (int, float)):
        raise ValueError("Calibrated cosine threshold must be a finite number in [-1,1]")
    value = finite_number(raw_value)
    if value is None or value < -1.0 or value > 1.0:
        raise ValueError("Calibrated cosine threshold must be a finite number in [-1,1]")
    return value


def score_embedding_vector(vector, model_id="large"):
    """Apply a per-model, maximum-over-roster calibration; no pair-threshold fallback."""
    if not isinstance(vector, (list, tuple)) or len(vector) != 64:
        return {"status": "uncertain", "identity_id": None,
                "decision_reason": "embedding_dimension_mismatch"}
    values = []
    for value in vector:
        number = finite_number(value)
        if number is None:
            return {"status": "uncertain", "identity_id": None,
                    "decision_reason": "nonfinite_embedding"}
        values.append(number)
    norm = sum(value * value for value in values) ** 0.5
    if finite_number(norm) is None or norm <= 0.0:
        return {"status": "uncertain", "identity_id": None,
                "decision_reason": "zero_embedding"}
    values = [value / norm for value in values]
    try:
        templates = load_templates(model_id)
    except ValueError as error:
        return {"status": "uncertain", "identity_id": None,
                "decision_reason": "invalid_or_missing_model_template", "error": str(error)[:120],
                "embedding_norm": norm}
    if not templates:
        return {"status": "no_enrollment", "identity_id": None,
                "decision_reason": "no_local_templates", "embedding_norm": norm}
    data = load_calibration()
    if data is None:
        return {"status": "uncertain", "identity_id": None,
                "decision_reason": "no_calibration_file", "embedding_norm": norm}
    expected_config_hash = getattr(cfg, "MODEL_CONFIG_SHA256", "")
    if expected_config_hash and data.get("model_config_sha256") != expected_config_hash:
        return {"status": "uncertain", "identity_id": None,
                "decision_reason": "stale_model_config_calibration_hash", "embedding_norm": norm}
    expected_models = getattr(cfg, "MODEL_SHA256", {})
    model_hashes = data.get("model_sha256", {})
    expected_hash = expected_models.get(model_id) if isinstance(expected_models, dict) else None
    if expected_hash and (not isinstance(model_hashes, dict) or model_hashes.get(model_id) != expected_hash):
        return {"status": "uncertain", "identity_id": None,
                "decision_reason": "stale_model_calibration_hash", "embedding_norm": norm}
    expected_roster = data.get("roster_identity_ids")
    actual_roster = sorted(entry["identity_id"] for entry in templates)
    if expected_roster is not None and sorted(expected_roster) != actual_roster:
        return {"status": "uncertain", "identity_id": None,
                "decision_reason": "stale_roster_calibration", "embedding_norm": norm}
    try:
        with open(cfg.TEMPLATE_FILE, "r") as stream:
            template_data = json.load(stream)
    except OSError:
        template_data = None
    if not isinstance(template_data, dict):
        return {"status": "uncertain", "identity_id": None,
                "decision_reason": "invalid_template_metadata", "embedding_norm": norm}
    template_model_hashes = template_data.get("model_sha256", {})
    if not isinstance(template_model_hashes, dict) or template_model_hashes.get(model_id) != expected_hash:
        return {"status": "uncertain", "identity_id": None,
                "decision_reason": "stale_template_model_hash", "embedding_norm": norm}
    if expected_config_hash and template_data.get("model_config_sha256") != expected_config_hash:
        return {"status": "uncertain", "identity_id": None,
                "decision_reason": "stale_template_config_hash", "embedding_norm": norm}
    threshold = calibration_threshold(model_id)
    if threshold is None:
        return {"status": "uncertain", "identity_id": None,
                "decision_reason": "no_calibrated_max_gallery_threshold", "embedding_norm": norm}
    threshold = binary32(threshold)
    if threshold is None:
        return {"status": "uncertain", "identity_id": None,
                "decision_reason": "invalid_binary32_calibrated_threshold", "embedding_norm": norm}
    candidates = []
    for entry in templates:
        similarity = binary32(sum(values[i] * entry["vector"][i] for i in range(64)))
        if similarity is None:
            return {"status": "uncertain", "identity_id": None,
                    "decision_reason": "nonfinite_template_similarity", "embedding_norm": norm}
        candidates.append((similarity, entry["identity_id"]))
    candidates.sort(reverse=True)
    best_score, best_id = candidates[0]
    competitor_score = candidates[1][0] if len(candidates) > 1 else None
    competitor_margin = binary32(best_score - competitor_score) if competitor_score is not None else None
    status = "known" if best_score >= threshold else "unknown"
    reason = "above_max_gallery_threshold" if status == "known" else "below_max_gallery_threshold"
    band = binary32(getattr(cfg, "UNCERTAIN_BAND", 0.025))
    gap = binary32(abs(binary32(best_score - threshold)))
    if band is None or gap is None:
        return {"status": "uncertain", "identity_id": None,
                "decision_reason": "invalid_binary32_uncertainty_band", "embedding_norm": norm}
    if gap <= band:
        status = "uncertain"
        reason = "inside_max_gallery_uncertain_band"
    return {"status": status, "identity_id": best_id if status == "known" else None,
            "candidate_identity_id": best_id, "cosine_similarity": best_score,
            "nearest_competitor_similarity": competitor_score,
            "nearest_competitor_margin": competitor_margin,
            "decision_threshold": threshold, "decision_reason": reason,
            "embedding_norm": norm, "model_id": model_id,
            "gallery_size": len(templates)}

def local_file_exists(path):
    try:
        import os
        os.stat(path)
        return True
    except OSError:
        return False


def embed_crop(frame, roi, stats, return_vector=False, model_path=None):
    global EMBEDDING_METADATA_LOGGED
    start = time.ticks_us()
    # On Nicla Vision/OpenMV 5.0.1, Image.copy(roi=...) returned a black
    # image for RGB565 CSI frames. The ROI conversion path preserves pixels
    # and creates the grayscale crop needed by the embedder.
    crop = frame.to_grayscale(roi=roi, copy=True)
    crop_stats = crop.get_statistics()
    stats["crop_copy_grayscale_us"] = elapsed_us(start)
    stats["crop_mean_u8"] = int(crop_stats.mean)
    stats["crop_stdev_u8"] = int(crop_stats.stdev)
    if (crop_stats.mean < cfg.LOW_QUALITY_MIN_MEAN or
            crop_stats.mean > cfg.LOW_QUALITY_MAX_MEAN or
            crop_stats.stdev < cfg.LOW_QUALITY_MIN_STDEV):
        return {"status": "uncertain", "decision_reason": "low_quality_crop", "identity_id": None}

    gc.collect()
    heap_before = gc.mem_free()
    start = time.ticks_us()
    model_path = model_path or cfg.EMBEDDING_MODEL
    model = ml.Model(model_path)
    load_us = elapsed_us(start)
    meta = model_metadata(model, "identity_embedding", model_path)
    stats["embedding_heap_free_after_load_bytes"] = gc.mem_free()
    if shape_tuple(meta["input_shape"]) != (1, 96, 96, 1):
        raise ValueError("Embedding model input shape is incompatible: " + repr(meta["input_shape"]))
    if not meta["input_dtype"] or meta["input_dtype"][0] != "b":
        raise ValueError("Embedding model must accept signed INT8 input")
    if abs(float(meta["input_scale"][0]) - cfg.EXPECTED_INPUT_SCALE) > cfg.INPUT_SCALE_TOLERANCE:
        raise ValueError("Embedding input scale is incompatible with OpenMV grayscale preprocessing")
    if int(meta["input_zero_point"][0]) != cfg.EXPECTED_INPUT_ZERO_POINT:
        raise ValueError("Embedding input zero point is incompatible with OpenMV grayscale preprocessing")
    if shape_tuple(meta["output_shape"]) != (1, 64):
        raise ValueError("Embedding output must contain 64 values")
    stats["embedding_load_us"] = load_us
    stats["embedding_tensor_arena_bytes"] = meta["tensor_arena_bytes"]
    if not EMBEDDING_METADATA_LOGGED:
        emit("model_metadata", meta)
        EMBEDDING_METADATA_LOGGED = True
    start = time.ticks_us()
    output = model.predict([Normalization()(crop)])
    stats["embedding_inference_us"] = elapsed_us(start)
    raw = [float(output[0][0][i]) for i in range(64)]
    output = None
    if len(raw) != 64:
        raise ValueError("Embedding model returned an unexpected vector length")
    if any(finite_number(value) is None for value in raw):
        raise ValueError("Embedding model returned a non-finite vector")
    norm = sum(value * value for value in raw) ** 0.5
    if finite_number(norm) is None or norm <= 0.0:
        raise ValueError("Embedding model returned a zero or invalid vector")
    vector = [value / norm for value in raw]
    model = None
    crop = None
    gc.collect()
    stats["embedding_heap_free_before_load_bytes"] = heap_before
    stats["embedding_heap_free_after_unload_bytes"] = gc.mem_free()
    if return_vector:
        return {"status": "embedded", "identity_id": None,
                "embedding_norm": norm, "embedding_vector": vector}
    templates = load_templates()
    threshold = calibration_threshold()
    if not templates:
        return {"status": "no_enrollment", "identity_id": None,
                "decision_reason": "no_local_templates", "embedding_norm": norm}
    if threshold is None:
        return {"status": "uncertain", "identity_id": None,
                "decision_reason": "no_calibrated_threshold", "embedding_norm": norm}
    threshold = binary32(threshold)
    if threshold is None:
        return {"status": "uncertain", "decision_reason": "invalid_binary32_calibrated_threshold", "identity_id": None}
    candidates = []
    for entry in templates:
        similarity = binary32(sum(vector[i] * entry["vector"][i] for i in range(64)))
        if similarity is None:
            raise ValueError("Embedding comparison returned a non-finite similarity")
        candidates.append((similarity, entry["identity_id"]))
    candidates.sort(reverse=True)
    best_score, best_id = candidates[0]
    competitor_score = candidates[1][0] if len(candidates) > 1 else None
    competitor_margin = binary32(best_score - competitor_score) if competitor_score is not None else None
    status = "known" if best_score >= threshold else "unknown"
    reason = "above_calibrated_validation_threshold" if status == "known" else "below_calibrated_validation_threshold"
    band = binary32(cfg.UNCERTAIN_BAND)
    lower = binary32(threshold - band) if band is not None else None
    if lower is None:
        return {"status": "uncertain", "decision_reason": "invalid_binary32_uncertainty_band", "identity_id": None}
    if lower <= best_score < threshold:
        status = "uncertain"
        reason = "inside_uncertain_band"
    return {
        "status": status,
        "identity_id": best_id if status == "known" else None,
        "candidate_identity_id": best_id,
        "cosine_similarity": best_score,
        "nearest_competitor_similarity": competitor_score,
        "nearest_competitor_margin": competitor_margin,
        "decision_threshold": threshold,
        "decision_reason": reason,
        "embedding_norm": norm,
    }



def process_frame(frame, stats):
    """Classify one frame into explicit no-face/multiple/crop/model states."""
    try:
        faces = detect_faces(frame, stats)
        result = {"status": "no_face", "identity_id": None,
                  "decision_reason": "no_detector_face"}
        if len(faces) > 1:
            return {"status": "multiple_faces", "identity_id": None,
                    "decision_reason": "more_than_one_detector_region"}
        if len(faces) == 1:
            roi = valid_crop(frame, faces[0]["center"])
            stats["detection_score"] = faces[0]["score"]
            if roi is None:
                return {"status": "uncertain", "identity_id": None,
                        "decision_reason": "crop_clipped_by_frame_edge"}
            stats["identity_roi"] = roi
            result = embed_crop(frame, roi, stats)
        return result
    except Exception as error:
        return {"status": "pipeline_error", "identity_id": None,
                "error": str(error)[:160]}

def set_status_led(status):
    if not cfg.LED_FEEDBACK:
        return
    LED_RED.off()
    LED_GREEN.off()
    LED_BLUE.off()
    if status == "known":
        LED_GREEN.on()
    elif status == "unknown":
        LED_RED.on()
    elif status == "multiple_faces" or status == "uncertain":
        LED_RED.on()
        LED_GREEN.on()
    elif status == "pipeline_error":
        LED_RED.on()
        LED_BLUE.on()
    else:
        LED_BLUE.on()


def main():
    gc.collect()
    cam = csi.CSI()
    cam.reset()
    cam.pixformat(csi.RGB565)
    cam.framesize(getattr(csi, cfg.FRAME_SIZE))
    cam.window((240, 240))
    cam.auto_rotation(False)
    cam.hmirror(cfg.HMIRROR)
    cam.vflip(cfg.VFLIP)
    cam.framebuffers(cfg.FRAMEBUFFER_COUNT)
    cam.snapshot(time=cfg.SETTLE_MS)
    emit("recognition_started", {
        "openmv_version": "5.0.1",
        "detector_model": cfg.DETECTOR_MODEL,
        "embedding_model": cfg.EMBEDDING_MODEL,
        "template_file": cfg.TEMPLATE_FILE,
        "camera_pixel_format": "RGB565",
        "camera_frame_size": [240, 240],
        "identity_crop_side_pixels": cfg.IDENTITY_CROP_SIDE_PIXELS,
        "sample_period_ms": cfg.SAMPLE_PERIOD_MS,
        "identity_enrollment_present": local_file_exists(cfg.TEMPLATE_FILE),
    })
    frame_index = 0
    min_heap = gc.mem_free()
    while cfg.MAX_FRAMES == 0 or frame_index < cfg.MAX_FRAMES:
        cycle_start = time.ticks_us()
        gc.collect()
        stats = {"frame_index": frame_index, "python_heap_free_before_bytes": gc.mem_free()}
        capture_start = time.ticks_us()
        frame = cam.snapshot()
        stats["capture_us"] = elapsed_us(capture_start)
        stats["frame_width"] = frame.width()
        stats["frame_height"] = frame.height()
        result = process_frame(frame, stats)
        set_status_led(result["status"])
        stats.update(result)
        stats["pipeline_us_before_serial_log"] = elapsed_us(cycle_start)
        min_heap = min(min_heap, gc.mem_free())
        stats["min_observed_python_heap_free_bytes"] = min_heap
        emit("recognition_frame", stats)
        frame = None
        gc.collect()
        frame_index += 1
        elapsed_ms = elapsed_us(cycle_start) // 1000
        if cfg.SAMPLE_PERIOD_MS > elapsed_ms:
            time.sleep_ms(cfg.SAMPLE_PERIOD_MS - elapsed_ms)
    emit("recognition_complete", {"frames": frame_index, "min_python_heap_free_bytes": min_heap})


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        emit("recognition_halted", {"error": str(error)[:160], "identity_id": None})
        raise
