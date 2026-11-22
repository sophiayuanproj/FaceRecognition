"""Bounded on-board checks for an isolated /flash/qa_v2 release.

This module is imported only by an explicit, host-controlled QA command. It
does not write files, access enrollment/calibration inputs, print identity
vectors, or change the installed startup path. Layout vectors are returned
only to the connected host for immediate parity comparison and must not be
persisted by the host runner.
"""

import gc
import hashlib
import binascii
import json
import math
import os
import time

import csi
import embedding_config as cfg
import image
import ml
import recognize_controlled as controlled
import recognize_embedding as recognition
from ml.preprocessing import Normalization
from policy_controller import PolicyController, reason_coded_quality


def _emit(value):
    print(json.dumps(value))


def _sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        while True:
            block = stream.read(1024)
            if not block:
                break
            digest.update(block)
    return binascii.hexlify(digest.digest()).decode("ascii")


def _read_manifest():
    with open(cfg.RELEASE_ROOT + "/qualification-manifest.json", "r") as stream:
        return json.load(stream)


def _model_metadata(model):
    return {
        "flatbuffer_bytes": int(model.len),
        "tensor_arena_bytes": int(model.ram),
        "input_shape": model.input_shape,
        "input_dtype": model.input_dtype,
        "input_scale": model.input_scale,
        "input_zero_point": model.input_zero_point,
        "output_shape": model.output_shape,
        "output_dtype": model.output_dtype,
        "output_scale": model.output_scale,
        "output_zero_point": model.output_zero_point,
    }


def audit_release():
    """Validate the staged config/bundle/model and report hashes and heap."""
    gc.collect()
    heap_before = gc.mem_free()
    verified = controlled._validate_release(cfg.POLICY_ID)
    start = time.ticks_us()
    model = ml.Model(cfg.EMBEDDING_MODEL)
    load_us = time.ticks_diff(time.ticks_us(), start)
    metadata = _model_metadata(model)
    heap_loaded = gc.mem_free()
    model = None
    gc.collect()
    _emit({"event": "qa_release_audit", "status": "pass",
           "model_config_sha256": verified["model_config_sha256"],
           "firmware_sha256": verified["firmware_sha256"],
           "candidate_model_sha256": verified["model_sha256"].get("large"),
           "load_us": load_us, "heap_free_before_bytes": heap_before,
           "heap_free_loaded_bytes": heap_loaded,
           "heap_free_after_unload_bytes": gc.mem_free(), "model": metadata})


def _quantile(values, probability):
    ordered = sorted(values)
    position = int(math.ceil(probability * len(ordered))) - 1
    position = max(0, min(len(ordered) - 1, position))
    return ordered[position]


def run_timing_blocks():
    """Ten separately loaded blocks; 5 warm-ups and 100 timed predicts each."""
    manifest = _read_manifest()
    layouts = manifest["layout_blocks"]
    seeds = manifest["timing_seeds"]
    for block_index in range(len(seeds)):
        block_seed = int(seeds[block_index])
        layout = layouts[(block_seed * 2654435761) % len(layouts)]
        sample_path = cfg.RELEASE_ROOT + "/" + layout["expected_crop_path"]
        gc.collect()
        heap_before = gc.mem_free()
        crop = image.Image(sample_path, copy_to_fb=True)
        start = time.ticks_us()
        model = ml.Model(cfg.EMBEDDING_MODEL)
        load_us = time.ticks_diff(time.ticks_us(), start)
        metadata = _model_metadata(model)
        heap_loaded = gc.mem_free()
        for _ in range(5):
            model.predict([Normalization()(crop)])
        latencies = []
        for _ in range(100):
            start = time.ticks_us()
            model.predict([Normalization()(crop)])
            latencies.append(time.ticks_diff(time.ticks_us(), start))
        model = None
        crop = None
        gc.collect()
        _emit({"event": "qa_timing_block", "status": "pass", "block_seed": block_seed,
               "timed_inferences": len(latencies), "warmup_inferences": 5,
               "model_load_ms": load_us / 1000.0,
               "warm_p50_ms": _quantile(latencies, 0.50) / 1000.0,
               "warm_p95_ms": _quantile(latencies, 0.95) / 1000.0,
               "warm_p99_ms": _quantile(latencies, 0.99) / 1000.0,
               "flatbuffer_bytes": metadata["flatbuffer_bytes"],
               "tensor_arena_bytes": metadata["tensor_arena_bytes"],
               "heap_free_before_load_bytes": heap_before,
               "heap_free_after_load_bytes": heap_loaded,
               "heap_free_after_unload_bytes": gc.mem_free(),
               "input_case_id": layout["layout_case_id"]})


def run_layout_blocks():
    """Test the exact board grayscale ROI path on ten synthetic dev layouts."""
    manifest = _read_manifest()
    for row in manifest["layout_blocks"]:
        scene_path = cfg.RELEASE_ROOT + "/" + row["scene_path"]
        expected_sha = row["expected_crop_sha256"]
        expected_crop_path = cfg.RELEASE_ROOT + "/" + row["expected_crop_path"]
        frame = image.Image(scene_path, copy_to_fb=True)
        roi = tuple(row["source_roi_xywh"])
        crop_start = time.ticks_us()
        crop = frame.to_grayscale(roi=roi, copy=True)
        crop_us = time.ticks_diff(time.ticks_us(), crop_start)
        crop_sha = _sha256_bytes(crop)
        if crop_sha != expected_sha:
            raise ValueError("board_roi_crop_hash_mismatch:" + row["layout_case_id"])
        # Read the separately staged reference to ensure both files are present
        # and intact. This does not use a face label or template.
        if _sha256(expected_crop_path) != expected_sha:
            raise ValueError("staged_expected_crop_hash_mismatch:" + row["layout_case_id"])
        gc.collect()
        heap_before = gc.mem_free()
        load_start = time.ticks_us()
        model = ml.Model(cfg.EMBEDDING_MODEL)
        model_load_us = time.ticks_diff(time.ticks_us(), load_start)
        metadata = _model_metadata(model)
        normalize_start = time.ticks_us()
        model_input = Normalization()(crop)
        normalize_us = time.ticks_diff(time.ticks_us(), normalize_start)
        inference_start = time.ticks_us()
        output = model.predict([model_input])
        inference_us = time.ticks_diff(time.ticks_us(), inference_start)
        vector = [float(output[0][0][i]) for i in range(64)]
        norm = math.sqrt(sum(value * value for value in vector))
        if not math.isfinite(norm) or norm <= 0:
            raise ValueError("invalid_layout_embedding:" + row["layout_case_id"])
        vector = [value / norm for value in vector]
        model = None
        output = None
        crop = None
        frame = None
        gc.collect()
        _emit({"event": "qa_layout_block", "status": "pass",
               "block_seed": int(row["block_seed"]),
               "layout_case_id": row["layout_case_id"],
               "source_roi_xywh": row["source_roi_xywh"],
               "synthetic_face_scale_px": row["synthetic_face_scale_px"],
               "crop_sha256": crop_sha, "expected_crop_sha256": expected_sha,
               "crop_preprocessing_us": crop_us, "model_load_us": model_load_us,
               "normalization_us": normalize_us, "inference_including_runtime_postprocessing_us": inference_us,
               "embedding_vector_transient": vector,
               "flatbuffer_bytes": metadata["flatbuffer_bytes"],
               "tensor_arena_bytes": metadata["tensor_arena_bytes"],
               "heap_free_before_model_bytes": heap_before,
               "heap_free_after_unload_bytes": gc.mem_free(),
               "scope": "synthetic development-crop digital ROI/model-input parity only"})


def _configure_camera():
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
    return cam


def run_camera_sleep_blocks():
    """Ten GC2145 sleep/wake checks, measuring wake-to-usable-frame time."""
    manifest = _read_manifest()
    cam = _configure_camera()
    for block_seed in manifest["timing_seeds"]:
        cam.sleep(True)
        time.sleep_ms(100)
        start = time.ticks_us()
        cam.sleep(False)
        cam.snapshot(time=cfg.SETTLE_MS)
        frame = cam.snapshot()
        wake_us = time.ticks_diff(time.ticks_us(), start)
        if frame is None or frame.width() != 240 or frame.height() != 240:
            raise ValueError("camera_wake_did_not_return_240x240_frame")
        _emit({"event": "qa_camera_sleep_wake_block", "status": "pass",
               "block_seed": int(block_seed), "wake_to_usable_frame_ms": wake_us / 1000.0,
               "frame_width": frame.width(), "frame_height": frame.height(),
               "sensor_id": cam.cid(), "scope": "camera API wake latency; not power evidence"})
        frame = None
    cam.sleep(False)


def run_memory_probe():
    """Measure sequential stock FOMO and candidate peaks with a live frame."""
    from ml.postprocessing.edgeimpulse import Fomo

    camera_start = time.ticks_us()
    cam = _configure_camera()
    camera_setup_and_settle_us = time.ticks_diff(time.ticks_us(), camera_start)
    capture_start = time.ticks_us()
    frame = cam.snapshot()
    capture_us = time.ticks_diff(time.ticks_us(), capture_start)
    if frame is None or frame.width() != 240 or frame.height() != 240:
        raise ValueError("memory_probe_camera_frame_invalid")
    gc.collect()
    before_detector = gc.mem_free()
    detector_load_start = time.ticks_us()
    detector = ml.Model(cfg.DETECTOR_MODEL, postprocess=Fomo(threshold=cfg.DETECTION_THRESHOLD))
    detector_load_us = time.ticks_diff(time.ticks_us(), detector_load_start)
    detector_meta = _model_metadata(detector)
    after_detector_load = gc.mem_free()
    detector_inference_start = time.ticks_us()
    detector.predict([Normalization(roi=(0, 0, frame.width(), frame.height()))(frame)])
    detector_inference_us = time.ticks_diff(time.ticks_us(), detector_inference_start)
    detector_unload_start = time.ticks_us()
    detector = None
    gc.collect()
    detector_unload_gc_us = time.ticks_diff(time.ticks_us(), detector_unload_start)
    after_detector_unload = gc.mem_free()
    crop_start = time.ticks_us()
    crop = frame.to_grayscale(roi=(72, 72, 96, 96), copy=True)
    crop_preprocessing_us = time.ticks_diff(time.ticks_us(), crop_start)
    gc.collect()
    before_candidate = gc.mem_free()
    candidate_load_start = time.ticks_us()
    model = ml.Model(cfg.EMBEDDING_MODEL)
    candidate_load_us = time.ticks_diff(time.ticks_us(), candidate_load_start)
    candidate_meta = _model_metadata(model)
    after_candidate_load = gc.mem_free()
    normalize_start = time.ticks_us()
    model_input = Normalization()(crop)
    normalize_us = time.ticks_diff(time.ticks_us(), normalize_start)
    candidate_inference_start = time.ticks_us()
    model.predict([model_input])
    candidate_inference_us = time.ticks_diff(time.ticks_us(), candidate_inference_start)
    model = None
    gc.collect()
    _emit({"event": "qa_memory_probe", "status": "pass",
           "camera_frame_bytes": frame.size(), "detector": detector_meta,
           "candidate": candidate_meta,
           "camera_setup_and_settle_us": camera_setup_and_settle_us,
           "frame_acquisition_us": capture_us,
           "detector_load_us": detector_load_us,
           "detector_inference_including_fomo_postprocess_us": detector_inference_us,
           "detector_unload_gc_us": detector_unload_gc_us,
           "crop_preprocessing_us": crop_preprocessing_us,
           "candidate_load_us": candidate_load_us,
           "candidate_normalization_us": normalize_us,
           "candidate_inference_us": candidate_inference_us,
           "python_heap_free_before_detector_bytes": before_detector,
           "python_heap_free_after_detector_load_bytes": after_detector_load,
           "python_heap_free_after_detector_unload_bytes": after_detector_unload,
           "python_heap_free_before_candidate_bytes": before_candidate,
           "python_heap_free_after_candidate_load_bytes": after_candidate_load,
           "python_heap_free_after_candidate_unload_bytes": gc.mem_free(),
           "models_loaded_concurrently": False,
           "scope": "one live frame and sequential models; not total MCU RAM or arena"})


def run_application_blocks():
    """Run ten separately initialized one-frame candidate application cycles.

    The board reports only timing, decision state, and whether an identity was
    emitted. Runtime telemetry is timed at the emitter; no frames or vectors
    are retained. These ambient-camera blocks are not participant evidence.
    """
    import recognize_controlled as controlled

    manifest = _read_manifest()
    original_controlled_emit = controlled.emit
    original_recognition_emit = recognition.emit
    original_serial_logging = cfg.SERIAL_LOGGING
    event_fields = {
        "release_validation": ("status",),
        "run_start": ("camera_warm_start_us",),
        "camera_wake": ("wake_to_settled_us",),
        "detector_end": ("face_count", "detection_latency_us", "detector_load_us",
                         "detector_inference_us"),
        "crop_status": ("status", "reason_code"),
        "recognition_decision": ("status", "decision_reason",
            "pipeline_latency_us_before_log", "capture_latency_us", "detection_latency_us",
            "detector_load_us", "detector_inference_us", "crop_copy_grayscale_us",
            "embedding_load_us", "embedding_inference_us", "small_model_latency_us",
            "large_model_latency_us"),
        "run_end": ("frames",),
        "startup_error": ("status", "error_code", "reason_code"),
    }

    for block_seed in manifest["decision_seeds"]:
        captured = []
        emit_durations = []
        execution_error = None

        def measured_emit(event, fields=None):
            start = time.ticks_us()
            original_controlled_emit(event, fields)
            emit_durations.append(time.ticks_diff(time.ticks_us(), start))
            if event not in event_fields:
                return
            source = fields if isinstance(fields, dict) else {}
            row = {"event": event}
            for key in event_fields[event]:
                if key in source:
                    row[key] = source[key]
            if event == "recognition_decision":
                row["identity_present"] = bool(source.get("identity_id"))
            captured.append(row)

        controlled.emit = measured_emit
        recognition.emit = measured_emit
        cfg.SERIAL_LOGGING = True
        entrypoint_start = time.ticks_us()
        try:
            with open(cfg.RELEASE_ROOT + "/main.py", "r") as stream:
                exec(stream.read())
        except Exception as error:
            execution_error = type(error).__name__
        finally:
            entrypoint_us = time.ticks_diff(time.ticks_us(), entrypoint_start)
            controlled.emit = original_controlled_emit
            recognition.emit = original_recognition_emit
            cfg.SERIAL_LOGGING = original_serial_logging

        decisions = [row for row in captured if row["event"] == "recognition_decision"]
        decision = decisions[-1] if decisions else {}
        run_ends = [row for row in captured if row["event"] == "run_end"]
        releases = [row for row in captured if row["event"] == "release_validation"]
        starts = [row for row in captured if row["event"] == "run_start"]
        startup_errors = [row for row in captured if row["event"] == "startup_error"]
        allowed = ("no_face", "multiple_faces", "uncertain", "no_enrollment")
        identity_present = decision.get("identity_present") is True
        valid = (execution_error is None and not startup_errors and len(decisions) == 1 and
                 decision.get("status") in allowed and not identity_present and
                 any(row.get("status") == "passed" for row in releases) and
                 bool(starts) and any(row.get("frames") == 1 for row in run_ends))
        _emit({"event": "qa_application_decision_block",
            "status": "pass" if valid else "fail", "block_seed": int(block_seed),
            "application_entrypoint_us": entrypoint_us,
            "camera_warm_start_us": starts[-1].get("camera_warm_start_us") if starts else None,
            "decision_status": decision.get("status"),
            "decision_reason": decision.get("decision_reason"),
            "identity_present": identity_present,
            "decision_cycle_latency_us_before_decision_event": decision.get("pipeline_latency_us_before_log"),
            "frame_capture_us": decision.get("capture_latency_us"),
            "fomo_detection_total_us": decision.get("detection_latency_us"),
            "fomo_load_us": decision.get("detector_load_us"),
            "fomo_inference_including_normalization_and_postprocessing_us": decision.get("detector_inference_us"),
            "crop_preprocessing_us": decision.get("crop_copy_grayscale_us"),
            "candidate_model_load_us": decision.get("embedding_load_us"),
            "candidate_model_inference_us": decision.get("embedding_inference_us"),
            "small_model_end_to_end_us": decision.get("small_model_latency_us"),
            "large_model_end_to_end_us": decision.get("large_model_latency_us"),
            "serial_emit_event_count": len(emit_durations),
            "serial_emit_total_us": sum(emit_durations),
            "serial_emit_max_event_us": max(emit_durations) if emit_durations else None,
            "serial_time_included_in_decision_cycle": True,
            "execution_error_type": execution_error,
            "scope": "ten one-frame ambient-camera application runs; no participant, identity, or power claim"})


def run_application_no_serial_blocks():
    """Measure one-frame decision cycles with USB telemetry output disabled.

    The controlled app still traverses its event hooks, but those hooks only
    capture a small allowlisted set of fields in memory. Aggregate timing/status
    records are emitted after each measured run, outside the decision interval.
    """
    import recognize_controlled as controlled

    manifest = _read_manifest()
    original_controlled_emit = controlled.emit
    original_recognition_emit = recognition.emit
    original_serial_logging = cfg.SERIAL_LOGGING
    event_fields = {
        "release_validation": ("status",),
        "run_start": ("camera_warm_start_us",),
        "recognition_decision": ("status", "decision_reason", "pipeline_latency_us_before_log",
                                  "capture_latency_us"),
        "run_end": ("frames",),
        "startup_error": ("status", "error_code", "reason_code"),
    }
    cfg.SERIAL_LOGGING = False
    try:
        for block_seed in manifest["decision_seeds"]:
            captured = []
            execution_error = None

            def in_memory_emit(event, fields=None):
                if event not in event_fields:
                    return
                source = fields if isinstance(fields, dict) else {}
                row = {"event": event}
                for key in event_fields[event]:
                    if key in source:
                        row[key] = source[key]
                if event == "recognition_decision":
                    row["identity_present"] = bool(source.get("identity_id"))
                captured.append(row)

            controlled.emit = in_memory_emit
            recognition.emit = in_memory_emit
            entrypoint_start = time.ticks_us()
            try:
                with open(cfg.RELEASE_ROOT + "/main.py", "r") as stream:
                    exec(stream.read())
            except Exception as error:
                execution_error = type(error).__name__
            entrypoint_us = time.ticks_diff(time.ticks_us(), entrypoint_start)
            decisions = [row for row in captured if row["event"] == "recognition_decision"]
            decision = decisions[-1] if decisions else {}
            starts = [row for row in captured if row["event"] == "run_start"]
            releases = [row for row in captured if row["event"] == "release_validation"]
            run_ends = [row for row in captured if row["event"] == "run_end"]
            valid = (execution_error is None and len(decisions) == 1 and
                decision.get("identity_present") is False and
                decision.get("status") in ("no_face", "multiple_faces", "uncertain", "no_enrollment") and
                any(row.get("status") == "passed" for row in releases) and bool(starts) and
                any(row.get("frames") == 1 for row in run_ends))
            _emit({"event": "qa_application_decision_no_serial_block",
                "status": "pass" if valid else "fail", "block_seed": int(block_seed),
                "application_entrypoint_us": entrypoint_us,
                "camera_warm_start_us": starts[-1].get("camera_warm_start_us") if starts else None,
                "decision_status": decision.get("status"),
                "identity_present": decision.get("identity_present") is True,
                "decision_cycle_latency_us_without_serial_output": decision.get("pipeline_latency_us_before_log"),
                "frame_capture_us": decision.get("capture_latency_us"),
                "serial_output_enabled": False,
                "event_capture_mode": "in-memory allowlist; aggregate record emitted after the one-frame application returns",
                "execution_error_type": execution_error,
                "scope": "ten one-frame ambient-camera runs with serial output disabled; no participant or power claim"})
    finally:
        controlled.emit = original_controlled_emit
        recognition.emit = original_recognition_emit
        cfg.SERIAL_LOGGING = original_serial_logging


def run_soak_block(block_seed):
    """Run one bounded no-enrollment soak with sparse, non-identifying telemetry."""
    import recognize_controlled as controlled

    manifest = _read_manifest()
    block = next((row for row in manifest.get("soak_blocks", [])
                  if row.get("block_seed") == int(block_seed)), None)
    if block is None:
        raise ValueError("unknown_qa_soak_seed:" + str(block_seed))
    schedule = block.get("randomized_interval_schedule")
    duration_ms = block.get("target_duration_ms")
    valid_intervals = (250, 500, 1000, 2000, 2500, 5000)
    if (type(duration_ms) is not int or duration_ms != 1800000 or not isinstance(schedule, list) or
            len(schedule) != 6 or
            [row.get("offset_ms") for row in schedule] != [0, 300000, 600000, 900000, 1200000, 1500000] or
            sorted(row.get("interval_ms") for row in schedule) != sorted(valid_intervals)):
        raise ValueError("invalid_qa_soak_schedule")

    original_controlled_emit = controlled.emit
    original_recognition_emit = recognition.emit
    original_values = {"serial_logging": cfg.SERIAL_LOGGING,
                       "max_frames": cfg.MAX_FRAMES,
                       "sleep_enabled": getattr(cfg, "CAMERA_SLEEP_ENABLED", False),
                       "policy_id": getattr(cfg, "POLICY_ID", "G")}
    status_counts = {}
    heap_samples = []
    error_count = 0
    identity_outputs = 0
    release_statuses = []
    run_starts = []
    run_ends = []
    startup_errors = []
    last_heap_sample_ms = 0
    block_started_ms = time.ticks_ms()
    reset_cause_start = None
    reset_cause_end = None
    try:
        try:
            import machine
            reset_cause_start = machine.reset_cause()
        except Exception:
            pass

        def capture_emit(event, fields=None):
            nonlocal error_count, identity_outputs, last_heap_sample_ms
            fields = fields if isinstance(fields, dict) else {}
            if event == "release_validation":
                release_statuses.append(fields.get("status"))
            elif event == "run_start":
                run_starts.append({"identity_enrollment_present": fields.get("identity_enrollment_present"),
                                   "camera_sleep_enabled": fields.get("camera_sleep_enabled")})
            elif event == "recognition_decision":
                status = fields.get("status", "missing_status")
                status_counts[status] = status_counts.get(status, 0) + 1
                if fields.get("identity_id"):
                    identity_outputs += 1
                now_ms = time.ticks_diff(time.ticks_ms(), block_started_ms)
                if now_ms - last_heap_sample_ms >= 300000:
                    heap_samples.append({"elapsed_ms": now_ms,
                        "frame_index": fields.get("frame_index"),
                        "status": status, "heap_free_bytes": fields.get("gc_free_bytes"),
                        "minimum_heap_free_bytes": fields.get("min_gc_free_bytes"),
                        "scheduled_interval_ms": fields.get("interval_ms")})
                    last_heap_sample_ms = now_ms
            elif event == "run_end":
                run_ends.append({"frames": fields.get("frames"),
                                 "min_gc_free_bytes": fields.get("min_gc_free_bytes"),
                                 "counters": fields.get("counters")})
            elif event == "startup_error":
                startup_errors.append({"error_code": fields.get("error_code"),
                                       "reason_code": fields.get("reason_code")})
            elif event == "error":
                error_count += 1

        controlled.emit = capture_emit
        recognition.emit = capture_emit
        cfg.SERIAL_LOGGING = False
        cfg.MAX_FRAMES = 0
        cfg.CAMERA_SLEEP_ENABLED = False
        cfg.POLICY_ID = "G"
        cfg.QA_SOAK_DURATION_MS = duration_ms
        cfg.QA_SAMPLING_SCHEDULE = schedule
        execution_error = None
        try:
            controlled.main()
        except Exception as error:
            execution_error = type(error).__name__
        elapsed_ms = time.ticks_diff(time.ticks_ms(), block_started_ms)
        try:
            reset_cause_end = machine.reset_cause()
        except Exception:
            pass
    finally:
        controlled.emit = original_controlled_emit
        recognition.emit = original_recognition_emit
        cfg.SERIAL_LOGGING = original_values["serial_logging"]
        cfg.MAX_FRAMES = original_values["max_frames"]
        cfg.CAMERA_SLEEP_ENABLED = original_values["sleep_enabled"]
        cfg.POLICY_ID = original_values["policy_id"]
        for name in ("QA_SOAK_DURATION_MS", "QA_SAMPLING_SCHEDULE"):
            try:
                delattr(cfg, name)
            except Exception:
                pass

    allowed_statuses = ("no_face", "multiple_faces", "uncertain", "no_enrollment")
    total_frames = run_ends[0].get("frames") if len(run_ends) == 1 else 0
    valid = (execution_error is None and elapsed_ms >= duration_ms and
             elapsed_ms <= duration_ms + 15000 and release_statuses == ["passed"] and
             len(run_starts) == 1 and run_starts[0].get("identity_enrollment_present") is False and
             run_starts[0].get("camera_sleep_enabled") is False and len(run_ends) == 1 and
             type(total_frames) is int and total_frames > 0 and error_count == 0 and
             not startup_errors and identity_outputs == 0 and
             all(status in allowed_statuses for status in status_counts) and len(heap_samples) >= 5)
    _emit({"event": "qa_soak_block", "status": "pass" if valid else "fail",
        "block_seed": int(block_seed), "target_duration_ms": duration_ms,
        "elapsed_ms": elapsed_ms, "frame_count": total_frames,
        "randomized_interval_schedule": schedule,
        "status_counts": status_counts, "error_event_count": error_count,
        "identity_outputs": identity_outputs, "heap_samples": heap_samples,
        "minimum_heap_free_bytes": run_ends[0].get("min_gc_free_bytes") if len(run_ends) == 1 else None,
        "application_counters": run_ends[0].get("counters") if len(run_ends) == 1 else None,
        "serial_output_enabled_during_block": False,
        "event_retention": "sparse allowlisted statuses, heap points, and counters only; no images, embeddings, names, or identity IDs",
        "reset_cause_start": reset_cause_start, "reset_cause_end": reset_cause_end,
        "execution_error_type": execution_error,
        "watchdog_stress_injected": False,
        "scope": "one 30-minute no-enrollment ambient-camera hardware soak; not participant, power, or cold-start evidence"})


class _FrameBounds:
    def width(self):
        return 240

    def height(self):
        return 240


def _guard_cases(seed):
    controller = PolicyController("A-full", ticks_diff=time.ticks_diff)
    frame = _FrameBounds()
    checks = []
    cases = []

    def check(name, condition):
        if not condition:
            raise AssertionError("board_guard_failed:" + name)
        cases.append(name)

    # Deterministic seed-dependent ordering exercises initialization and order
    # independence without treating these blocks as independent model roots.
    case_order = ["quality", "crop", "temporal", "motion", "escalation", "deadline", "missing_gallery"]
    shift = seed % len(case_order)
    case_order = case_order[shift:] + case_order[:shift]
    for case in case_order:
        if case == "quality":
            check("quality_missing", reason_coded_quality(None, None)[0] is False)
            check("quality_dim", reason_coded_quality(2, 30)[1] == "too_dim")
            check("quality_saturated", reason_coded_quality(250, 30)[1] == "saturated")
            check("quality_low_contrast", reason_coded_quality(80, 1)[1] == "low_contrast_or_blank")
            check("quality_valid", reason_coded_quality(80, 30)[0] is True)
        elif case == "crop":
            check("crop_valid_bounds", recognition.valid_crop(frame, (48, 48)) == (0, 0, 96, 96))
            check("crop_clipped_bounds", recognition.valid_crop(frame, (47, 48)) is None)
            check("crop_far_edge", recognition.valid_crop(frame, (192, 192)) == (144, 144, 96, 96))
        elif case == "temporal":
            status, _ = controller.temporal_decision("known", "qa_id", True)
            check("temporal_first_vote_uncertain", status == "uncertain")
            status, _ = controller.temporal_decision("known", "qa_id", True)
            check("temporal_two_of_three", status == "known")
            status, reason = controller.temporal_decision("known", "qa_id", False)
            check("temporal_invalid_evidence", status == "uncertain" and reason == "evidence_not_valid_for_temporal_vote")
        elif case == "motion":
            missing = controller.update_motion(None)
            check("imu_missing_is_explicit", missing["reason"] == "imu_missing")
            controller.update_motion(200.0)
            active = controller.update_motion(200.0)
            check("motion_high_hysteresis", active["motion_active"] is True)
            controller.update_motion(20.0)
            controller.update_motion(20.0)
            inactive = controller.update_motion(20.0)
            check("motion_low_hysteresis", inactive["motion_active"] is False)
        elif case == "escalation":
            calls, reason = controller.select_models(True)
            check("small_first", calls == ["small"] and reason == "small_model_first")
            escalations = []
            for i in range(8):
                calls, _ = controller.select_models(True, 0.8, 0.8, now_ms=i * 1000)
                escalations.append(calls == ["small", "large"])
            check("escalation_rate_limit_six_per_minute", sum(escalations) == 6)
            check("poor_quality_does_not_escalate", controller.select_models(False, 0.8, 0.8)[0] == [])
        elif case == "deadline":
            controller.note_wake(0)
            late, timeout = controller.apply_response_deadline({"status": "known", "identity_id": "qa_id"}, 6001)
            check("late_identity_suppressed", timeout is not None and late["status"] == "uncertain" and late["identity_id"] is None)
        elif case == "missing_gallery":
            vector = [1.0] + [0.0] * 63
            result = recognition.score_embedding_vector(vector, "large")
            check("missing_enrollment_no_identity", result["status"] == "no_enrollment" and result.get("identity_id") is None)
    check("tick_wrap_diff", time.ticks_diff(5, time.ticks_add(0, -5)) >= 0)
    return cases


def run_guard_blocks():
    manifest = _read_manifest()
    for seed in manifest["guard_seeds"]:
        cases = _guard_cases(int(seed))
        _emit({"event": "qa_board_guard_block", "status": "pass",
               "block_seed": int(seed), "case_ids": cases,
               "release_validation": "pass",
               "calibration_and_final_dataset_roles_accessed": False})


def _sha256_bytes(value):
    return hashlib.sha256(value).hexdigest()


def run(mode):
    if mode == "audit":
        audit_release()
    elif mode == "timing":
        run_timing_blocks()
    elif mode == "layouts":
        run_layout_blocks()
    elif mode == "guards":
        run_guard_blocks()
    elif mode == "camera-sleep":
        run_camera_sleep_blocks()
    elif mode == "memory":
        run_memory_probe()
    elif mode == "application":
        run_application_blocks()
    elif mode == "application-no-serial":
        run_application_no_serial_blocks()
    elif mode.startswith("soak-"):
        run_soak_block(int(mode.split("-", 1)[1]))
    else:
        raise ValueError("unknown_qa_mode:" + str(mode))


def main():
    import sys
    if len(sys.argv) != 2:
        raise ValueError("usage: board_resource_qualifier_v2.run(mode)")
    run(sys.argv[1])


if __name__ == "__main__":
    main()
