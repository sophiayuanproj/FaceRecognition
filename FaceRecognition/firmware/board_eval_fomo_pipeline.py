"""Run stock FOMO, the production fixed crop, and INT8 verification on held-out scenes.

The host places six temporary held-out grayscale inputs on /flash/fomo_eval/.
The default is a 160x160 neutral scene with a centered 96x96 pseudonymous
validation face crop, so the actual Nicla runs stock FOMO before the INT8
embedder without saving camera frames or installing enrollment templates. A
96x96 input is also accepted as an already-cropped board-input fallback.
"""

import gc
import image
import json
import ml
import time

from ml.postprocessing.edgeimpulse import Fomo
from ml.preprocessing import Normalization


DETECTOR_PATH = "/rom/fomo_face_detection.tflite"
EMBEDDER_PATH = "/flash/identity_embedding_int8.tflite"
MANIFEST_PATH = "/flash/fomo_eval/fomo-board-eval-manifest.json"
SAMPLE_ROOT = "/flash/fomo_eval/"
CALIBRATION_PATH = "/flash/embedding_calibration.json"
DETECTION_THRESHOLD = 0.4  # Production threshold; adjust only for a documented diagnostic sweep.
LOW_QUALITY_MIN_MEAN = 12
LOW_QUALITY_MAX_MEAN = 243
LOW_QUALITY_MIN_STDEV = 5
UNCERTAIN_BAND = 0.025


def emit(event, fields=None):
    row = {"schema_version": 1, "event": event, "ticks_ms": time.ticks_ms()}
    if fields:
        row.update(fields)
    print(json.dumps(row))


def elapsed_us(start):
    return time.ticks_diff(time.ticks_us(), start)


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
    for output_index in range(output_count):
        label_index = output_index + 1
        if (label_index < len(labels) and
                labels[label_index].strip().lower() == "face"):
            return output_index
    if output_count == 2:
        return 1
    raise ValueError("Cannot map stock FOMO outputs to labels")


def detect(frame):
    gc.collect()
    started = time.ticks_us()
    model = ml.Model(DETECTOR_PATH, postprocess=Fomo(threshold=DETECTION_THRESHOLD))
    load_us = elapsed_us(started)
    if shape_tuple(model.input_shape) != (1, 96, 96, 3):
        raise ValueError("Unexpected stock FOMO input shape: " + repr(model.input_shape))
    # PPM inputs are already RGB565. Copying those images into the CSI
    # framebuffer with to_rgb565(copy_to_fb=True) changed their pixel values
    # on OpenMV 5.0.1, so preserve RGB565 inputs and convert only other formats.
    if frame.format() != image.RGB565:
        frame = frame.to_rgb565(copy=True)
    roi = (0, 0, frame.width(), frame.height())
    started = time.ticks_us()
    output = model.predict([Normalization(roi=roi)(frame)])
    inference_us = elapsed_us(started)
    detections = []
    if output:
        index = face_output_index(model, len(output))
        if index < len(output):
            for rect, score in output[index]:
                score = float(score)
                if score >= DETECTION_THRESHOLD:
                    x, y, width, height = [int(value) for value in rect]
                    detections.append({
                        "center": [x + width // 2, y + height // 2],
                        "rect": [x, y, width, height],
                        "score": score,
                    })
    model_bytes = int(model.len)
    arena_bytes = int(model.ram)
    output = None
    model = None
    gc.collect()
    return detections, load_us, inference_us, model_bytes, arena_bytes, gc.mem_free()


def embed_crop(frame, roi):
    # Nicla Vision/OpenMV 5.0.1 returned a black image for copy(roi=...) on
    # RGB565 CSI frames. Convert just the selected ROI into a grayscale copy.
    crop = frame.to_grayscale(roi=roi, copy=True)
    quality = crop.get_statistics()
    quality_fields = {"crop_mean_u8": int(quality.mean), "crop_stdev_u8": int(quality.stdev)}
    if (quality.mean < LOW_QUALITY_MIN_MEAN or quality.mean > LOW_QUALITY_MAX_MEAN or
            quality.stdev < LOW_QUALITY_MIN_STDEV):
        crop = None
        gc.collect()
        return None, quality_fields, "low_quality_crop", None, None
    gc.collect()
    started = time.ticks_us()
    model = ml.Model(EMBEDDER_PATH)
    load_us = elapsed_us(started)
    if shape_tuple(model.input_shape) != (1, 96, 96, 1) or model.input_dtype[0] != "b":
        raise ValueError("Unexpected embedder interface: shape=" + repr(model.input_shape) +
                         " dtype=" + repr(model.input_dtype))
    started = time.ticks_us()
    output = model.predict([Normalization()(crop)])
    inference_us = elapsed_us(started)
    raw = [float(output[0][0][index]) for index in range(64)]
    norm = sum(value * value for value in raw) ** 0.5
    if norm <= 0.0:
        raise ValueError("Embedder returned a zero vector")
    vector = [value / norm for value in raw]
    output = None
    raw = None
    model = None
    crop = None
    gc.collect()
    return vector, quality_fields, None, load_us, inference_us


def main():
    with open(MANIFEST_PATH, "r") as stream:
        manifest = json.load(stream)
    with open(CALIBRATION_PATH, "r") as stream:
        threshold = float(json.load(stream)["threshold"])
    emit("fomo_board_eval_started", {
        "images": len(manifest["images"]), "detector": DETECTOR_PATH,
        "embedder": EMBEDDER_PATH, "threshold": threshold,
        "detection_threshold": DETECTION_THRESHOLD,
        "templates_storage": "temporary RAM only",
    })
    embeddings = {}
    metadata_logged = False
    min_heap = gc.mem_free()
    for entry in manifest["images"]:
        gc.collect()
        frame = image.Image(SAMPLE_ROOT + entry["file"])
        if frame.width() not in (96, 160) or frame.height() != frame.width():
            raise ValueError("FOMO eval input must be a 96x96 crop or 160x160 scene")
        detections, detector_load_us, detector_inference_us, detector_bytes, detector_arena, free_after_fomo = detect(frame)
        min_heap = min(min_heap, free_after_fomo)
        event = {
            "record_id": entry["record_id"], "face_count": len(detections),
            "detection_scores": [item["score"] for item in detections],
            "detector_load_us": detector_load_us,
            "detector_inference_us": detector_inference_us,
            "detector_model_file_bytes": detector_bytes,
            "detector_tensor_arena_bytes": detector_arena,
            "python_heap_free_after_detector_unload_bytes": free_after_fomo,
        }
        if not metadata_logged:
            emit("fomo_board_eval_metadata", {
                "detector_model_file_bytes": detector_bytes,
                "detector_tensor_arena_bytes": detector_arena,
                "detector_input_shape": [1, 96, 96, 3],
                "embedding_model_file_bytes": 26872,
                "embedding_tensor_arena_bytes": 51040,
            })
            metadata_logged = True
        if len(detections) != 1:
            event["status"] = "no_face" if not detections else "multiple_faces"
            event["decision_reason"] = "detector_count_not_one"
            emit("fomo_board_eval_detection", event)
            frame = None
            gc.collect()
            min_heap = min(min_heap, gc.mem_free())
            continue
        center = detections[0]["center"]
        side = 96
        if frame.width() == side and frame.height() == side:
            # Board input is already a face crop; use it whole after FOMO gates it.
            roi = (0, 0, side, side)
        else:
            roi = (center[0] - side // 2, center[1] - side // 2, side, side)
        if roi[0] < 0 or roi[1] < 0 or roi[0] + side > frame.width() or roi[1] + side > frame.height():
            event["status"] = "uncertain"
            event["decision_reason"] = "crop_clipped_by_frame_edge"
            event["identity_roi"] = list(roi)
            emit("fomo_board_eval_detection", event)
            frame = None
            gc.collect()
            continue
        vector, quality, failure, model_load_us, inference_us = embed_crop(frame, roi)
        event.update(quality)
        event["identity_roi"] = list(roi)
        event["embedding_load_us"] = model_load_us
        event["embedding_inference_us"] = inference_us
        if failure:
            event["status"] = "uncertain"
            event["decision_reason"] = failure
            emit("fomo_board_eval_detection", event)
            frame = None
            gc.collect()
            continue
        event["status"] = "embedded"
        emit("fomo_board_eval_detection", event)
        emit("fomo_board_eval_embedding", {
            "record_id": entry["record_id"], "identity_id": entry["identity_id"],
            "identity_roi": list(roi), "vector": vector,
            "embedding_load_us": model_load_us, "embedding_inference_us": inference_us,
        })
        embeddings[entry["record_id"]] = {
            "identity_id": entry["identity_id"], "vector": vector,
        }
        frame = None
        vector = None
        gc.collect()
        min_heap = min(min_heap, gc.mem_free())

    templates = {}
    for entry in manifest["images"]:
        row = embeddings.get(entry["record_id"])
        if row and row["identity_id"] not in templates:
            templates[row["identity_id"]] = (entry["record_id"], row["vector"])
            emit("fomo_board_eval_template_created", {
                "identity_id": row["identity_id"], "record_id": entry["record_id"],
                "storage": "temporary RAM only",
            })
    decision_count = 0
    for entry in manifest["images"]:
        query = embeddings.get(entry["record_id"])
        if not query or any(entry["record_id"] == template[0] for template in templates.values()):
            continue
        for template_identity, (template_record, template_vector) in templates.items():
            score = sum(query["vector"][index] * template_vector[index] for index in range(64))
            if score >= threshold:
                decision = "known"
            elif score >= threshold - UNCERTAIN_BAND:
                decision = "uncertain"
            else:
                decision = "unknown"
            same_identity = query["identity_id"] == template_identity
            emit("fomo_board_eval_pair_decision", {
                "query_record_id": entry["record_id"], "template_record_id": template_record,
                "verification_pair_type": "positive" if same_identity else "negative",
                "cosine_similarity": score, "threshold": threshold,
                "decision": decision,
                "decision_correct": decision == ("known" if same_identity else "unknown"),
            })
            decision_count += 1
    emit("fomo_board_eval_complete", {
        "input_images": len(manifest["images"]), "embedded_images": len(embeddings),
        "temporary_templates": len(templates), "template_decisions": decision_count,
        "minimum_python_heap_free_bytes": min_heap,
    })


try:
    main()
except Exception as error:
    emit("fomo_board_eval_failed", {"error": str(error)[:160]})
    raise
