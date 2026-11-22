"""Explicit, local enrollment for one consenting person in front of the Nicla.

The host helper writes a bounded request to /flash, then runs this script over
USB. This script saves only a normalized mean embedding, never camera frames.
"""

import csi
import gc
import json
import os
import time

import embedding_config as cfg
import recognize_embedding as runtime

REQUEST_FILE = "/flash/enrollment_request.json"
TEMPLATE_BACKUP = cfg.TEMPLATE_FILE + ".bak"
MAX_IDENTITIES = 8


def emit(event, fields=None):
    if not cfg.SERIAL_LOGGING or not runtime.USB_VCP.isconnected():
        return
    record = {"schema_version": 1, "event": event, "ticks_ms": time.ticks_ms()}
    if fields:
        record.update(fields)
    print(json.dumps(record))


def read_request():
    with open(REQUEST_FILE, "r") as stream:
        request = json.load(stream)
    if not isinstance(request, dict) or request.get("version") != 1:
        raise ValueError("Enrollment request must be a version 1 JSON object")
    identity_id = request.get("identity_id")
    if (not isinstance(identity_id, str) or not identity_id or len(identity_id) > 24 or
            any(not ("a" <= ch <= "z" or "A" <= ch <= "Z" or
                     "0" <= ch <= "9" or ch in "_-") for ch in identity_id)):
        raise ValueError("identity_id must be a pseudonymous 1-24 character ASCII token")
    min_samples = request.get("min_samples", 5)
    max_attempts = request.get("max_attempts", 30)
    if (isinstance(min_samples, bool) or not isinstance(min_samples, int) or
            isinstance(max_attempts, bool) or not isinstance(max_attempts, int)):
        raise ValueError("Enrollment sample and attempt limits must be integer values")
    if min_samples < 3 or min_samples > 20 or max_attempts < min_samples or max_attempts > 60:
        raise ValueError("Enrollment sample/attempt limits are outside the supported range")
    replace_existing = request.get("replace_existing", False)
    if not isinstance(replace_existing, bool):
        raise ValueError("replace_existing must be a boolean")
    return identity_id, min_samples, max_attempts, replace_existing

def raw_templates():
    try:
        with open(cfg.TEMPLATE_FILE, "r") as stream:
            data = json.load(stream)
    except OSError:
        return []
    if (not isinstance(data, dict) or data.get("version") != 1 or
            not isinstance(data.get("entries"), list)):
        raise ValueError("Existing template file has an unsupported format")
    if len(data["entries"]) > MAX_IDENTITIES:
        raise ValueError("Existing template roster exceeds the supported maximum")
    return data["entries"]


def normalized_mean(vectors):
    if not vectors:
        raise ValueError("At least one enrollment vector is required")
    for vector in vectors:
        if not isinstance(vector, list) or len(vector) != 64:
            raise ValueError("Every enrollment vector must contain 64 values")
        for value in vector:
            number = runtime.finite_number(value)
            if number is None:
                raise ValueError("Enrollment vectors must contain finite numbers")
    mean = [sum(float(vector[i]) for vector in vectors) / len(vectors) for i in range(64)]
    norm = sum(value * value for value in mean) ** 0.5
    if runtime.finite_number(norm) is None or norm <= 0.0:
        raise ValueError("Enrollment vectors average to zero or an invalid norm")
    return [value / norm for value in mean]


def write_templates(entries):
    temp_path = cfg.TEMPLATE_FILE + ".tmp"
    payload = {"version": 1, "entries": entries}
    with open(temp_path, "w") as stream:
        json.dump(payload, stream)
        stream.flush()

    # Keep one on-device prior version so power loss during replacement is
    # recoverable. os.rename is used only after a complete temporary write.
    if runtime.local_file_exists(cfg.TEMPLATE_FILE):
        try:
            os.remove(TEMPLATE_BACKUP)
        except OSError:
            pass
        os.rename(cfg.TEMPLATE_FILE, TEMPLATE_BACKUP)
    try:
        os.rename(temp_path, cfg.TEMPLATE_FILE)
    except Exception:
        if (not runtime.local_file_exists(cfg.TEMPLATE_FILE) and
                runtime.local_file_exists(TEMPLATE_BACKUP)):
            os.rename(TEMPLATE_BACKUP, cfg.TEMPLATE_FILE)
        raise


def run_enrollment(identity_id, min_samples, max_attempts, replace_existing):
    # Validate every previously saved record before modifying the roster.
    runtime.load_templates()
    entries = raw_templates()
    matches = [entry for entry in entries if entry.get("identity_id") == identity_id]
    if matches and not replace_existing:
        raise ValueError("identity_id already exists; rerun with --replace to overwrite it")
    if not matches and len(entries) >= MAX_IDENTITIES:
        raise ValueError("Template roster is full (maximum 8 identities)")

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
    emit("enrollment_ready", {
        "identity_id": identity_id,
        "required_samples": min_samples,
        "max_attempts": max_attempts,
        "raw_images_saved": False,
        "instructions": "Show one consenting person; keep face centered and vary pose slightly between accepted samples",
    })
    time.sleep_ms(3000)

    vectors = []
    min_heap = gc.mem_free()
    for attempt in range(max_attempts):
        gc.collect()
        frame = cam.snapshot()
        stats = {"enrollment_attempt": attempt + 1}
        try:
            faces = runtime.detect_faces(frame, stats)
            if len(faces) != 1:
                emit("enrollment_sample_skipped", {
                    "identity_id": identity_id,
                    "attempt": attempt + 1,
                    "reason": "no_face" if len(faces) == 0 else "multiple_faces",
                })
            else:
                roi = runtime.valid_crop(frame, faces[0]["center"])
                if roi is None:
                    emit("enrollment_sample_skipped", {
                        "identity_id": identity_id,
                        "attempt": attempt + 1,
                        "reason": "crop_clipped_by_frame_edge",
                    })
                else:
                    stats["detection_score"] = faces[0]["score"]
                    result = runtime.embed_crop(frame, roi, stats, return_vector=True)
                    if result["status"] == "embedded":
                        vectors.append(result["embedding_vector"])
                        emit("enrollment_sample_accepted", {
                            "identity_id": identity_id,
                            "attempt": attempt + 1,
                            "accepted_samples": len(vectors),
                            "required_samples": min_samples,
                        })
                    else:
                        emit("enrollment_sample_skipped", {
                            "identity_id": identity_id,
                            "attempt": attempt + 1,
                            "reason": result.get("decision_reason", result.get("status")),
                        })
        finally:
            frame = None
            gc.collect()
            min_heap = min(min_heap, gc.mem_free())
        if len(vectors) >= min_samples:
            break
        time.sleep_ms(700)

    if len(vectors) < min_samples:
        raise RuntimeError("Not enough valid face crops; accepted %d of %d required" %
                           (len(vectors), min_samples))

    vector = normalized_mean(vectors)
    new_entry = {"identity_id": identity_id, "vector": vector}
    updated = [entry for entry in entries if entry.get("identity_id") != identity_id]
    updated.append(new_entry)
    write_templates(updated)
    emit("enrollment_complete", {
        "identity_id": identity_id,
        "accepted_samples": len(vectors),
        "template_count": len(updated),
        "raw_images_saved": False,
        "min_python_heap_free_bytes": min_heap,
        "template_file": cfg.TEMPLATE_FILE,
    })


def main():
    identity_id, min_samples, max_attempts, replace_existing = read_request()
    run_enrollment(identity_id, min_samples, max_attempts, replace_existing)


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        emit("enrollment_failed", {"error": str(error)[:160], "raw_images_saved": False})
        raise
    finally:
        try:
            os.remove(REQUEST_FILE)
        except OSError:
            pass
