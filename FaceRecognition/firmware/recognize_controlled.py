"""Eight-policy Nicla controller adapter; emits status only, never stores frames."""

import csi
import binascii
import gc
import hashlib
import imu
import json
import os
import pyb
import time

import embedding_config as cfg
import recognize_embedding as recognition
from policy_controller import PolicyController, reason_coded_quality

USB_VCP = pyb.USB_VCP()
SEQUENCE = 0
RUN_ID = None
VERIFIED_MODELS = {}


def emit(event, fields=None):
    global SEQUENCE
    if not cfg.SERIAL_LOGGING or not USB_VCP.isconnected():
        return
    run_id = RUN_ID or "boot-%08d" % time.ticks_ms()
    policy_id = getattr(cfg, "POLICY_ID", "G")
    if policy_id not in ("H", "L", "S", "G", "A-acquire", "A-sense", "A-model", "A-full"):
        policy_id = "G"
    row = {
        "schema_version": 1,
        "event": event,
        "run_id": run_id,
        "policy_id": policy_id,
        "sequence": SEQUENCE,
        "ticks_ms": time.ticks_ms(),
        "host_received_monotonic_ns": None,
        "host_received_utc": None,
        "timing_uncertainty_ms": None,
        "model_config_sha256": getattr(cfg, "MODEL_CONFIG_SHA256", "") or None,
        "firmware_sha256": getattr(cfg, "FIRMWARE_SHA256", "") or None,
    }
    SEQUENCE += 1
    if fields:
        row.update(fields)
    print(json.dumps(row))


# Detector/embedder helpers share this envelope while called by the controlled
# adapter. Standalone recognition retains its legacy logger.
recognition.emit = emit


def elapsed_us(start):
    return time.ticks_diff(time.ticks_us(), start)


def _write_qa_boot_marker(path, payload):
    """Atomically write a minimal test-only startup marker when explicitly configured."""
    if not isinstance(path, str) or not path.startswith("/flash/") or ".." in path.split("/"):
        raise ValueError("invalid_qa_boot_marker_path")
    temporary = path + ".tmp"
    try:
        os.remove(temporary)
    except OSError:
        pass
    with open(temporary, "w") as stream:
        stream.write(json.dumps(payload))
    try:
        os.remove(path)
    except OSError:
        pass
    os.rename(temporary, path)


def _file_sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        while True:
            block = stream.read(1024)
            if not block:
                break
            digest.update(block)
    return binascii.hexlify(digest.digest()).decode("ascii")


def _release_path(name):
    root = getattr(cfg, "RELEASE_ROOT", "/flash") or "/flash"
    if not isinstance(root, str) or (root != "/flash" and not root.startswith("/flash/")):
        raise ValueError("invalid_release_root")
    if ".." in root.split("/"):
        raise ValueError("invalid_release_root")
    return root.rstrip("/") + "/" + name


def _runtime_config_sha256():
    with open(_release_path("embedding_config.py"), "rb") as stream:
        lines = stream.read().splitlines()
    normalized = []
    for line in lines:
        if line.strip().startswith(b"MODEL_CONFIG_SHA256"):
            normalized.append(b'MODEL_CONFIG_SHA256 = ""')
        else:
            normalized.append(line)
    payload = b"\n".join(normalized) + b"\n"
    policy_hash = _file_sha256(cfg.POLICY_CONFIG_FILE)
    digest = hashlib.sha256(b"nicla-model-config-v1\0" + payload + b"\n" + policy_hash.encode("ascii"))
    return binascii.hexlify(digest.digest()).decode("ascii"), policy_hash


def _runtime_firmware_sha256():
    bundle = (("main.py", _release_path("main.py")),
              ("policy_controller.py", _release_path("policy_controller.py")),
              ("recognize_controlled.py", _release_path("recognize_controlled.py")),
              ("recognize_embedding.py", _release_path("recognize_embedding.py")))
    payload = "".join("%s\t%s\n" % (name, _file_sha256(path)) for name, path in bundle)
    return binascii.hexlify(hashlib.sha256(payload.encode("utf-8")).digest()).decode("ascii")


def _validate_release(policy_id):
    actual_config, policy_file_hash = _runtime_config_sha256()
    expected_config = getattr(cfg, "MODEL_CONFIG_SHA256", "")
    if not expected_config or actual_config != expected_config:
        raise ValueError("stale_or_unstamped_model_config")
    actual_firmware = _runtime_firmware_sha256()
    expected_firmware = getattr(cfg, "FIRMWARE_SHA256", "")
    if not expected_firmware or actual_firmware != expected_firmware:
        raise ValueError("stale_or_unstamped_firmware_bundle")
    if policy_id == "A-acquire":
        model_ids = ()
    else:
        model_ids = ("small", "large") if policy_id in ("A-model", "A-full") else (("small",) if policy_id == "S" else ("large",))
    verified = {}
    hashes = getattr(cfg, "MODEL_SHA256", {})
    for model_id in model_ids:
        model_path = _model_for(model_id)
        expected = hashes.get(model_id) if isinstance(hashes, dict) else None
        if not expected:
            raise ValueError("model_hash_not_configured:" + model_id)
        actual = _file_sha256(model_path)
        if actual != expected:
            raise ValueError("model_hash_mismatch:" + model_id)
        verified[model_id] = actual
    return {"model_config_sha256": actual_config, "firmware_sha256": actual_firmware,
            "policy_file_sha256": policy_file_hash, "model_sha256": verified}


def _get_imu():
    try:
        return imu.acceleration_mg(), imu.angular_rate_mdps(), None
    except Exception as error:
        return None, None, str(error)[:120]


def _quality_stats(image):
    try:
        stats = image.get_statistics()
        return int(stats.mean), int(stats.stdev)
    except Exception:
        return None, None


def _model_for(model_id):
    if model_id == "small":
        return getattr(cfg, "SMALL_EMBEDDING_MODEL", "/flash/identity_embedding_small_int8.tflite")
    return cfg.EMBEDDING_MODEL


def _recognize_vector(frame, roi, model_id, fields):
    model_path = _model_for(model_id)
    start = time.ticks_us()
    result = recognition.embed_crop(frame, roi, fields, return_vector=True, model_path=model_path)
    latency = elapsed_us(start)
    fields[model_id + "_model_latency_us"] = latency
    fields["model_id"] = model_id
    if result.get("status") != "embedded":
        return result
    return recognition.score_embedding_vector(result["embedding_vector"], model_id)


def _run_recognition(frame, roi, controller, now_ms, fields):
    # With no local gallery there is no identity decision to make; leave the
    # detector and sensing path available, but avoid an unnecessary embedder.
    templates = recognition.load_template_ids()
    if not templates:
        return {"status": "no_enrollment", "identity_id": None,
                "decision_reason": "no_local_templates"}, False

    policy = controller.policy_id
    if policy not in ("A-model", "A-full"):
        model_id = "small" if policy == "S" else "large"
        controller.counters[model_id + "_model_calls"] += 1
        start_event = model_id + "_model_start"
        end_event = model_id + "_model_end"
        emit(start_event, {"model_id": model_id})
        result = _recognize_vector(frame, roi, model_id, fields)
        emit(end_event, {"model_id": model_id, "latency_us": fields.get(model_id + "_model_latency_us")})
        return result, result.get("status") in ("known", "unknown")

    small_id = "small"
    small_threshold = recognition.calibration_threshold("small")
    model_plan, reason = controller.select_models(True, now_ms=now_ms)
    if not model_plan:
        return {"status": "uncertain", "identity_id": None, "decision_reason": reason}, False
    controller.counters["small_model_calls"] += 1
    emit("small_model_start", {"model_id": small_id})
    small_result = _recognize_vector(frame, roi, small_id, fields)
    emit("small_model_end", {"model_id": small_id, "latency_us": fields.get("small_model_latency_us"),
                              "status": small_result.get("status")})
    if small_result.get("status") == "uncertain":
        similarity = small_result.get("cosine_similarity")
        threshold = small_result.get("decision_threshold", small_threshold)
    else:
        similarity = small_result.get("cosine_similarity")
        threshold = small_result.get("decision_threshold", small_threshold)
    calls, escalation_reason = controller.select_models(
        True, small_similarity=similarity, small_threshold=threshold,
        uncertain_band=getattr(cfg, "UNCERTAIN_BAND", 0.025), now_ms=now_ms,
    )
    escalated = len(calls) == 2 and calls[1] == "large"
    emit("escalation_decision", {"reason_code": escalation_reason,
                                 "escalated": escalated,
                                 "small_score": similarity,
                                 "small_threshold": threshold})
    if not escalated:
        return small_result, small_result.get("status") in ("known", "unknown")
    controller.counters["large_model_calls"] += 1
    emit("large_model_start", {"model_id": "large"})
    large_result = _recognize_vector(frame, roi, "large", fields)
    emit("large_model_end", {"model_id": "large", "latency_us": fields.get("large_model_latency_us"),
                              "status": large_result.get("status")})
    return large_result, large_result.get("status") in ("known", "unknown")


def process_one_frame(frame, controller, accel_mg, gyro_mdps, now_ms):
    fields = {"frame_width": frame.width(), "frame_height": frame.height(), "accel_mg": accel_mg,
              "gyro_mdps": gyro_mdps, "identity_id": None}
    if controller.policy_id == "A-acquire":
        interval, interval_reason = controller.next_interval(True, 0)
        fields.update({"interval_ms": interval, "interval_reason": interval_reason,
                       "detector_calls": 0, "identity_inference_skipped": True,
                       "camera_sleep_request": False,
                       "sleep_reason": "acquisition_only_policy"})
        return ({"status": "acquisition_only", "identity_id": None,
                 "decision_reason": "acquisition_only_policy"}, fields, interval, None, False)
    motion_score = controller.motion_score(accel_mg, gyro_mdps)
    motion = controller.update_motion(motion_score)
    emit("imu_gate", {"accel_mg": accel_mg, "gyro_mdps": gyro_mdps,
                       "motion_score_mg_equivalent": motion_score,
                       "motion_active": motion["motion_active"], "reason_code": motion["reason"]})
    mean, stdev = _quality_stats(frame)
    quality_ok, quality_reason = reason_coded_quality(
        mean, stdev,
        getattr(cfg, "LOW_QUALITY_MIN_MEAN", 12),
        getattr(cfg, "LOW_QUALITY_MAX_MEAN", 243),
        getattr(cfg, "LOW_QUALITY_MIN_STDEV", 5),
    )
    quality_gate_enabled = controller.features["quality_adaptation"]
    emit("quality_gate", {"quality_mean_u8": mean, "quality_stdev_u8": stdev,
                           "passed": quality_ok, "applied": quality_gate_enabled,
                           "reason_code": quality_reason,
                           "brightness_is_lux": False})
    if not quality_ok and quality_gate_enabled:
        controller.clear_temporal()
        controller.counters["discarded_quality_frames"] += 1
        result = {"status": "uncertain", "identity_id": None,
                  "decision_reason": "scene_quality_" + quality_reason}
        interval, interval_reason = controller.next_interval(False, 0)
        sleep_request, sleep_reason = controller.should_sleep_camera(False, 0)
        fields.update({"scene_mean_u8": mean, "scene_stdev_u8": stdev,
                       "interval_ms": interval, "interval_reason": interval_reason,
                       "camera_sleep_request": sleep_request, "sleep_reason": sleep_reason})
        return result, fields, interval, 0, False

    controller.counters["detector_calls"] += 1
    emit("detector_start", {"model_id": "stock_fomo"})
    detector_start = time.ticks_us()
    detector_stats = {}
    faces = recognition.detect_faces(frame, detector_stats)
    fields["detection_latency_us"] = elapsed_us(detector_start)
    fields["face_count"] = detector_stats.get("face_count", len(faces))
    fields.update(detector_stats)
    emit("detector_end", {"face_count": fields["face_count"],
                           "detection_latency_us": fields["detection_latency_us"],
                           "detector_load_us": fields.get("detector_load_us"),
                           "detector_inference_us": fields.get("detector_inference_us"),
                           "detector_tensor_arena_bytes": fields.get("detector_tensor_arena_bytes"),
                           "detector_heap_free_before_load_bytes": fields.get("detector_heap_free_before_load_bytes"),
                           "detector_heap_free_after_load_bytes": fields.get("detector_heap_free_after_load_bytes"),
                           "detector_heap_free_after_unload_bytes": fields.get("detector_heap_free_after_unload_bytes")})
    if not faces:
        controller.clear_temporal()
        result = {"status": "no_face", "identity_id": None,
                  "decision_reason": "no_detector_face"}
        interval, interval_reason = controller.next_interval(True, 0)
        sleep_request, sleep_reason = controller.should_sleep_camera(True, 0)
        fields.update({"interval_ms": interval, "interval_reason": interval_reason,
                       "camera_sleep_request": sleep_request, "sleep_reason": sleep_reason})
        return result, fields, interval, 0, True
    if fields["face_count"] > 1 or len(faces) > 1:
        controller.clear_temporal()
        result = {"status": "multiple_faces", "identity_id": None,
                  "decision_reason": "more_than_one_detector_region"}
        interval, interval_reason = controller.next_interval(True, fields["face_count"])
        sleep_request, sleep_reason = controller.should_sleep_camera(True, fields["face_count"])
        fields.update({"interval_ms": interval, "interval_reason": interval_reason,
                       "camera_sleep_request": sleep_request, "sleep_reason": sleep_reason})
        return result, fields, interval, fields["face_count"], True

    identity_allowed, identity_gate_reason = controller.should_run_identity(quality_ok)
    emit("identity_gate", {"allowed": identity_allowed,
                            "reason_code": identity_gate_reason,
                            "wearer_motion_active": controller.motion_active})
    if not identity_allowed:
        controller.clear_temporal()
        result = {"status": "uncertain", "identity_id": None,
                  "decision_reason": identity_gate_reason}
        interval, interval_reason = controller.next_interval(True, fields["face_count"])
        sleep_request, sleep_reason = controller.should_sleep_camera(True, fields["face_count"])
        fields.update({"interval_ms": interval, "interval_reason": interval_reason,
                       "camera_sleep_request": sleep_request, "sleep_reason": sleep_reason,
                       "identity_inference_skipped": True})
        return result, fields, interval, fields["face_count"], False

    roi = recognition.valid_crop(frame, faces[0]["center"])
    if roi is None:
        controller.clear_temporal()
        emit("crop_status", {"status": "uncertain", "reason_code": "crop_clipped_by_frame_edge"})
        result = {"status": "uncertain", "identity_id": None,
                  "decision_reason": "crop_clipped_by_frame_edge"}
        interval, interval_reason = controller.next_interval(False, 1)
        fields.update({"interval_ms": interval, "interval_reason": interval_reason,
                       "crop_status": "clipped"})
        return result, fields, interval, 1, False
    fields["identity_roi"] = roi
    emit("crop_status", {"status": "valid", "identity_roi": roi,
                         "source":"stock_fomo_centroid_fixed_crop; not a calibrated face box"})
    result, valid_vote = _run_recognition(frame, roi, controller, now_ms, fields)
    if result.get("decision_reason") == "low_quality_crop":
        controller.counters["discarded_quality_frames"] += 1
    if controller.policy_id in ("G", "A-full"):
        before = result["status"]
        if before in ("known", "unknown"):
            status, confirmation_reason = controller.temporal_decision(
                before, result.get("identity_id"), valid_evidence=valid_vote,
                now_ms=now_ms)
        else:
            controller.clear_temporal()
            status, confirmation_reason = before, result.get("decision_reason", "no_valid_decision")
        result = dict(result)
        result["status"] = status
        result["identity_id"] = result.get("identity_id") if status == "known" else None
        result["decision_reason"] = confirmation_reason
    interval, interval_reason = controller.next_interval(True, 1)
    sleep_request, sleep_reason = controller.should_sleep_camera(True, 1)
    fields.update({"interval_ms": interval, "interval_reason": interval_reason,
                   "camera_sleep_request": sleep_request, "sleep_reason": sleep_reason})
    return result, fields, interval, 1, valid_vote


def main():
    global RUN_ID, SEQUENCE, VERIFIED_MODELS
    qa_soak_duration_ms = getattr(cfg, "QA_SOAK_DURATION_MS", None)
    if type(qa_soak_duration_ms) is not int or qa_soak_duration_ms <= 0:
        qa_soak_duration_ms = None
    qa_sampling_schedule = getattr(cfg, "QA_SAMPLING_SCHEDULE", None)
    policy_id = getattr(cfg, "POLICY_ID", "G")
    sensor_sleep_enabled = bool(getattr(cfg, "CAMERA_SLEEP_ENABLED", False))
    controller = PolicyController(policy_id, ticks_diff=time.ticks_diff,
                                  allow_camera_sleep=sensor_sleep_enabled)
    SEQUENCE = 0
    RUN_ID = "%s-%08d" % (policy_id.replace("-", "_"), time.ticks_ms())
    release_started = time.ticks_us()
    VERIFIED_MODELS = _validate_release(policy_id)["model_sha256"]
    emit("release_validation", {"model_config_sha256": getattr(cfg, "MODEL_CONFIG_SHA256", ""),
                                "firmware_sha256": getattr(cfg, "FIRMWARE_SHA256", ""),
                                "verified_model_sha256": VERIFIED_MODELS,
                                "validation_latency_us": elapsed_us(release_started),
                                "status": "passed"})
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
    warm_start = time.ticks_us()
    cam.snapshot(time=cfg.SETTLE_MS)
    controller.note_wake(time.ticks_ms())
    emit("camera_wake", {"reason_code": "initial_camera_startup",
                          "wake_to_settled_us": elapsed_us(warm_start),
                          "camera_state": "awake"})
    emit("run_start", {"policy_id": policy_id, "camera_warm_start_us": elapsed_us(warm_start),
                        "sample_period_ms": cfg.SAMPLE_PERIOD_MS,
                        "camera_sleep_enabled": sensor_sleep_enabled,
                        "small_model_path": getattr(cfg, "SMALL_EMBEDDING_MODEL", None),
                        "large_model_path": cfg.EMBEDDING_MODEL,
                        "identity_enrollment_present": recognition.local_file_exists(cfg.TEMPLATE_FILE),
                        "camera_pixel_format": "RGB565", "camera_frame_size": [240, 240]})
    frame_index = 0
    camera_sleeping = False
    min_heap = gc.mem_free()
    previous_start = time.ticks_ms()
    soak_start_ms = time.ticks_ms()
    qa_boot_marker_written = False
    while cfg.MAX_FRAMES == 0 or frame_index < cfg.MAX_FRAMES:
        if qa_soak_duration_ms is not None and time.ticks_diff(time.ticks_ms(), soak_start_ms) >= qa_soak_duration_ms:
            break
        cycle_start_us = time.ticks_us()
        cycle_start_ms = time.ticks_ms()
        if camera_sleeping:
            wake_start = time.ticks_us()
            try:
                cam.sleep(False)
                cam.snapshot(time=cfg.SETTLE_MS)
                camera_sleeping = False
                controller.note_wake(time.ticks_ms())
                emit("camera_wake", {"reason_code": "scheduled_periodic_check",
                                      "wake_to_settled_us": elapsed_us(wake_start),
                                      "camera_state": "awake"})
            except Exception as error:
                emit("error", {"error_code": "camera_wake_failed", "message": str(error)[:120]})
                camera_sleeping = False
        gc.collect()
        fields = {"frame_index": frame_index, "python_heap_free_before_bytes": gc.mem_free()}
        emit("frame_request", {"frame_index": frame_index})
        capture_start = time.ticks_us()
        try:
            frame = cam.snapshot()
            if frame is None:
                raise ValueError("camera returned no frame")
        except Exception as error:
            controller.clear_temporal()
            frame = None
            result = {"status": "uncertain", "identity_id": None,
                      "decision_reason": "camera_capture_failed", "error": str(error)[:120]}
            heartbeat = controller.heartbeat(time.ticks_ms())
            if heartbeat:
                emit("timeout", {"reason_code": heartbeat["reason"], "age_ms": heartbeat["age_ms"]})
                controller.interval_ms = 250
            controller.note_result(time.ticks_ms(), result["status"])
            emit("error", {"error_code": "camera_capture_failed", "message": str(error)[:120]})
            emit("recognition_decision", result)
            frame_index += 1
            time.sleep_ms(min(1000, controller.interval_ms))
            continue
        capture_us = elapsed_us(capture_start)
        controller.note_capture(time.ticks_ms())
        if policy_id == "A-acquire":
            accel, gyro, imu_error = None, None, None
        else:
            accel, gyro, imu_error = _get_imu()
        if imu_error:
            emit("error", {"error_code": "imu_read_failed", "message": imu_error})
        try:
            result, extra, interval, face_count, quality_valid = process_one_frame(
                frame, controller, accel, gyro, time.ticks_ms())
        except Exception as error:
            controller.clear_temporal()
            result = {"status": "pipeline_error", "identity_id": None,
                      "decision_reason": "recognition_pipeline_exception", "error": str(error)[:120]}
            extra = {}
            interval, _ = controller.next_interval(False, 0)
            face_count = 0
            quality_valid = False
            emit("error", {"error_code": "recognition_pipeline_error", "message": str(error)[:120]})
        fields.update(extra)
        fields["capture_latency_us"] = capture_us
        fields["frame_ready_ticks_ms"] = time.ticks_ms()
        fields["pipeline_latency_us_before_log"] = elapsed_us(cycle_start_us)
        fields["status"] = result.get("status")
        fields["reason_code"] = result.get("decision_reason")
        fields["identity_id"] = result.get("identity_id")
        fields["model_id"] = result.get("model_id")
        result.update({k: v for k, v in fields.items() if k not in ("identity_roi",)})
        result, heartbeat = controller.apply_response_deadline(result, time.ticks_ms())
        if heartbeat:
            emit("timeout", {"reason_code": heartbeat["reason"], "age_ms": heartbeat["age_ms"]})
            controller.interval_ms = 250
            interval = 250
            fields["interval_ms"] = 250
            fields["interval_reason"] = "response_timeout_recovery"
            fields["status"] = "uncertain"
            fields["reason_code"] = heartbeat["reason"]
            fields["identity_id"] = None
            result.update({k: v for k, v in fields.items() if k != "identity_roi"})
        if qa_soak_duration_ms is not None and isinstance(qa_sampling_schedule, list) and qa_sampling_schedule:
            schedule_elapsed_ms = time.ticks_diff(time.ticks_ms(), soak_start_ms)
            for schedule_item in qa_sampling_schedule:
                if (isinstance(schedule_item, dict) and type(schedule_item.get("offset_ms")) is int and
                        type(schedule_item.get("interval_ms")) is int and
                        schedule_elapsed_ms >= schedule_item["offset_ms"]):
                    interval = schedule_item["interval_ms"]
                elif isinstance(schedule_item, dict) and type(schedule_item.get("offset_ms")) is int:
                    if schedule_elapsed_ms < schedule_item["offset_ms"]:
                        break
            fields["interval_ms"] = interval
            fields["interval_reason"] = "qa_randomized_soak_schedule"
            result["interval_ms"] = interval
            result["interval_reason"] = "qa_randomized_soak_schedule"
        controller.note_result(time.ticks_ms(), result.get("status"))
        min_heap = min(min_heap, gc.mem_free())
        result["gc_free_bytes"] = gc.mem_free()
        result["min_gc_free_bytes"] = min_heap
        emit("frame_ready", {"frame_index": frame_index, "capture_latency_us": capture_us})
        emit("recognition_decision", result)
        marker_path = getattr(cfg, "QA_BOOT_MARKER", "")
        if marker_path and not qa_boot_marker_written:
            _write_qa_boot_marker(marker_path, {
                "event": "qa_candidate_boot_health",
                "release_validation": "passed",
                "status": result.get("status"),
                "identity_present": bool(result.get("identity_id")),
                "frame_index": frame_index,
                "frames_expected": cfg.MAX_FRAMES,
                "model_sha256": VERIFIED_MODELS.get("large"),
                "model_config_sha256": getattr(cfg, "MODEL_CONFIG_SHA256", ""),
                "firmware_sha256": getattr(cfg, "FIRMWARE_SHA256", ""),
                "ticks_ms": time.ticks_ms(),
                "scope": "single-frame no-enrollment warm-reset startup check",
            })
            qa_boot_marker_written = True
        if result.get("status") == "pipeline_error":
            emit("error", {"error_code": "recognition_pipeline_error", "message": result.get("error")})
        if result.get("camera_sleep_request") and sensor_sleep_enabled and not camera_sleeping:
            sleep_start = time.ticks_us()
            try:
                cam.sleep(True)
                camera_sleeping = True
                emit("camera_sleep", {"reason_code": result.get("sleep_reason"),
                                       "api_call_us": elapsed_us(sleep_start),
                                       "camera_state": "sleeping"})
            except Exception as error:
                emit("error", {"error_code": "camera_sleep_failed", "message": str(error)[:120],
                               "reason_code": "sensor_sleep_api_or_driver_failure"})
        frame = None
        gc.collect()
        frame_index += 1
        elapsed_ms = time.ticks_diff(time.ticks_ms(), cycle_start_ms)
        remaining_ms = interval - elapsed_ms
        if remaining_ms > 0:
            time.sleep_ms(remaining_ms)
        previous_start = cycle_start_ms
    emit("run_end", {"frames": frame_index, "min_gc_free_bytes": min_heap,
                      "counters": controller.counters,
                      "camera_sleep_enabled": sensor_sleep_enabled})


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        emit("startup_error", {"error_code": "controller_halted",
                                "reason_code": "release_validation_or_startup_failure",
                                "status": "halted", "message": str(error)[:160]})
        raise
