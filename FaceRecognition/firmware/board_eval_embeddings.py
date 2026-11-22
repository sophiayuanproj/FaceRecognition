"""Compare the deployed model against temporary held-out PGM crops.

The host copies a small set to /flash/board_eval/ only for this check, then
deletes every PGM and the manifest. Logs use record IDs and vectors only.
"""

import gc
import image
import json
import ml
import time

from ml.preprocessing import Normalization

MODEL_PATH = "/flash/identity_embedding_int8.tflite"
MANIFEST_PATH = "/flash/board_eval/board-eval-manifest.json"
CALIBRATION_PATH = "/flash/embedding_calibration.json"
SAMPLE_ROOT = "/flash/board_eval/"
UNCERTAIN_BAND = 0.025


def emit(event, fields=None):
    row = {"schema_version": 1, "event": event, "ticks_ms": time.ticks_ms()}
    if fields:
        row.update(fields)
    print(json.dumps(row))


def main():
    gc.collect()
    started = time.ticks_us()
    model = ml.Model(MODEL_PATH)
    load_us = time.ticks_diff(time.ticks_us(), started)
    emit("board_eval_model_loaded", {
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
    })
    with open(MANIFEST_PATH, "r") as stream:
        manifest = json.load(stream)
    with open(CALIBRATION_PATH, "r") as stream:
        threshold = float(json.load(stream)["threshold"])
    minimum_heap = gc.mem_free()
    vectors = {}
    identities = {}
    for entry in manifest["images"]:
        gc.collect()
        frame = image.Image(SAMPLE_ROOT + entry["file"])
        frame.to_grayscale()
        if frame.width() != 96 or frame.height() != 96:
            raise ValueError("Board evaluation image must be 96x96")
        started = time.ticks_us()
        output = model.predict([Normalization()(frame)])
        inference_us = time.ticks_diff(time.ticks_us(), started)
        raw = [float(output[0][0][i]) for i in range(64)]
        norm = sum(value * value for value in raw) ** 0.5
        vector = [value / norm for value in raw]
        vectors[entry["record_id"]] = vector
        identities[entry["record_id"]] = entry["identity_id"]
        emit("board_eval_embedding", {
            "record_id": entry["record_id"],
            "inference_us": inference_us,
            "embedding_norm_before_l2": norm,
            "vector": vector,
            "grayscale_dimensions": [frame.width(), frame.height()],
        })
        output = None
        vector = None
        raw = None
        frame = None
        gc.collect()
        minimum_heap = min(minimum_heap, gc.mem_free())
    # Use the first unseen crop per pseudonymous validation identity as a
    # temporary in-memory enrollment template. No template file or photo is
    # written to /flash. Remaining crops are verification queries.
    templates = {}
    for entry in manifest["images"]:
        identity_id = entry["identity_id"]
        record_id = entry["record_id"]
        if identity_id not in templates:
            templates[identity_id] = (record_id, vectors[record_id])
            emit("board_eval_template_created", {
                "identity_id": identity_id,
                "template_record_id": record_id,
                "template_storage": "temporary RAM only",
            })
    for query_record_id, query_vector in vectors.items():
        if any(query_record_id == template_record_id for template_record_id, _ in templates.values()):
            continue
        query_identity_id = identities[query_record_id]
        for template_identity_id, (template_record_id, template_vector) in templates.items():
            score = sum(query_vector[i] * template_vector[i] for i in range(64))
            if score >= threshold:
                decision = "known"
            elif score >= threshold - UNCERTAIN_BAND:
                decision = "uncertain"
            else:
                decision = "unknown"
            same_identity = query_identity_id == template_identity_id
            emit("board_eval_pair_decision", {
                "query_record_id": query_record_id,
                "template_record_id": template_record_id,
                "verification_pair_type": "positive" if same_identity else "negative",
                "cosine_similarity": score,
                "threshold": threshold,
                "decision": decision,
                "decision_correct": decision == ("known" if same_identity else "unknown"),
            })
    emit("board_eval_complete", {
        "images": len(manifest["images"]),
        "minimum_python_heap_free_bytes": minimum_heap,
    })


try:
    main()
except Exception as error:
    emit("board_eval_failed", {"error": str(error)[:160]})
    raise
