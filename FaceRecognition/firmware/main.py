"""Standalone Nicla Vision runtime: scene check -> detect -> crop -> identity."""

import binascii
import gc
import hashlib
import json
import math
import os
import time

import csi
import image
import imu
import ml
import pyb
from machine import LED, UART, WDT
from ml.postprocessing.edgeimpulse import Fomo

from identity_contracts import (class_index_from_labels, classifier_decision,
                                crop_for_boundary_policy, face_box_square_crop, finite_number,
                                fomo_center_camera_px, fomo_face_output_index, gallery_decision,
                                gate_unqualified_identity,
                                merge_near_duplicate_fomo_peaks,
                                normalize_embedding, normalized_face_box, openmv_dtype_name,
                                openmv_normalization_model_input,
                                pillow_bilinear_model_input,
                                prediction_values, quality_decision, single_tensor_metadata,
                                strict_score_pass)
from runtime_support import (TemporalConfirmer, calibrated_risk_escalation,
                             camera_wake_retry_ms, classifier_route_for_embedding, cost_aware_escalation,
                             config_hash_matches, enough_heap, EscalationBudget,
                             hash_matches, motion_detected, response_deadline_result, SimpleSceneTrigger,
                             sample_gray_scene_signature, SCENE_SIGNATURE_LENGTH,
                             validate_gallery_metadata)
from telemetry import LineInput, Telemetry


RELEASE_ROOT = "/flash"
CONFIG_PATH = RELEASE_ROOT.rstrip("/") + "/runtime_config.json"
ARTIFACT_ROOT = RELEASE_ROOT.rstrip("/") + "/models/deployment-v1"
MANIFEST_PATH = ARTIFACT_ROOT + "/integration_manifest.json"
FIRMWARE_PARTS = ("boot.py", "wearable_app.py", "identity_contracts.py",
                  "runtime_support.py", "telemetry.py", "mpy/build_manifest.json",
                  "mpy/wearable_app.mpy", "mpy/identity_contracts.mpy",
                  "mpy/runtime_support.mpy", "mpy/telemetry.mpy")
BATTERY_ARM_PATH = "/flash/sightings.armed"
BATTERY_PHOTO_ROOT = "/flash/sightings"
BATTERY_CONFIRM_MS = 1800
BATTERY_GAP_MS = 2400
BATTERY_COMMANDS = ("battery_status", "battery_arm", "battery_disarm",
                    "battery_list", "battery_read")
KNOWN = ["K01", "K02", "K03", "K04", "K05"]

# Reserve fields that are populated after model prediction while the identity
# arena is resident. This avoids a large MicroPython dictionary rehash at the
# lowest-heap point of an otherwise successful camera cycle.
_IDENTITY_RESULT_FIELDS = (
    "model_slot", "model_id", "model_sha256", "input_dtype", "output_dtype",
    "output_scale", "output_zero_point", "model_warm", "arena_bytes",
    "model_acquire_us", "model_bytes", "input_shape", "input_scale",
    "input_zero_point", "input_sha256", "input_byte_count", "input_packet",
    "input_capture_error", "preprocess_us", "input_transport_us", "inference_us",
    "model_predict_inclusive_us", "model_compute_and_dequantization_residual_us",
    "output_length", "output_value_representation", "output_processing_us",
    "identity_threshold", "identity_acceptance_cutoff", "small_backend", "large_backend", "class_order",
    "gallery_loaded_with_small_model_resident", "gallery_retained_after_decision",
    "identity_pipeline_us", "decision_latency_us", "python_heap_free_after_inference_bytes",
    "status", "identity_id", "candidate_identity_id", "candidate_score",
    "decision_reason", "backend", "other_probability", "known_other_margin",
    "nearest_competitor_margin", "class_probabilities", "temporal_confirmation",
    "boot_id", "imu", "python_heap_free_bytes", "python_heap_allocated_bytes",
    "detector_arena_bytes", "identity_arena_bytes", "framebuffer_count", "payloads",
    "python_heap_free_after_payload_bytes", "payload_transport",
    "pre_sent_tensor_packets", "pre_sent_tensor_bytes",
    "pre_sent_tensor_transport_us", "pre_tensor_transport_us",
    "pre_sent_tensor_packet_bytes", "frame_transport_us",
    "python_heap_free_after_payload_release_bytes", "frame_encode_us",
    "source_release_gc_us")


def _reserve_identity_result_fields(fields):
    for key in _IDENTITY_RESULT_FIELDS:
        if key not in fields:
            fields[key] = None

# boot.py reserves the C detector arena before importing this module. The
# object is transferred into ModelPool during runtime construction and these
# module references are then cleared so ModelPool.clear() can actually free it.
BOOT_PRELOADED_DETECTOR = None
BOOT_PRELOADED_FOMO = None
BOOT_PRELOADED_SMALL = None
BOOT_PRELOADED_HAAR_WIDE = None
BOOT_DETECTOR_PRELOAD_STATUS = {"status": "not_attempted"}

LED_RED = LED("LED_RED")
LED_GREEN = LED("LED_GREEN")
LED_BLUE = LED("LED_BLUE")


def configure_release_root(root):
    """Bind this imported module to the immutable release selected at boot."""
    global RELEASE_ROOT, CONFIG_PATH, ARTIFACT_ROOT, MANIFEST_PATH
    root = str(root).rstrip("/")
    if not root.startswith("/flash/releases/") or ".." in root.split("/"):
        raise ValueError("invalid_release_root")
    RELEASE_ROOT = root
    CONFIG_PATH = root + "/runtime_config.json"
    ARTIFACT_ROOT = root + "/models/deployment-v1"
    MANIFEST_PATH = ARTIFACT_ROOT + "/integration_manifest.json"


def _sha256_bytes(data):
    return binascii.hexlify(hashlib.sha256(data).digest()).decode("ascii")


def _sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        while True:
            block = stream.read(1024)
            if not block:
                break
            digest.update(block)
    return binascii.hexlify(digest.digest()).decode("ascii")


def _read_json(path):
    with open(path, "r") as stream:
        return json.load(stream)


def _firmware_hash(root=None, manifest=None):
    if root is None:
        root = RELEASE_ROOT
    rows = []
    runtime = (manifest or {}).get("precompiled_runtime", {})
    source_free = runtime.get("source_free") is True
    names = FIRMWARE_PARTS[:1] + FIRMWARE_PARTS[5:] if source_free else FIRMWARE_PARTS
    if "offline_sightings.mpy" in runtime.get("modules", {}):
        if not source_free:
            names += ("offline_sightings.py",)
        names += ("mpy/offline_sightings.mpy",)
    for name in names:
        path = root.rstrip("/") + "/" + name
        rows.append(name + "\t" + _sha256_file(path) + "\n")
    return _sha256_bytes("".join(rows).encode("utf-8"))


def _shape_tuple(value):
    while isinstance(value, (list, tuple)) and len(value) == 1 and isinstance(value[0], (list, tuple)):
        value = value[0]
    return tuple(int(item) for item in value)


def _dtype_name(value):
    return openmv_dtype_name(value)


def _flatten(values, output=None):
    if output is None:
        output = []
    if isinstance(values, (list, tuple)):
        for item in values:
            _flatten(item, output)
        return output
    try:
        length = len(values)
    except (TypeError, AttributeError):
        length = None
    if length is not None and not isinstance(values, (str, bytes)):
        for index in range(length):
            _flatten(values[index], output)
        return output
    output.append(values)
    return output


def _finite_stats(image, roi=None):
    try:
        stats = image.get_statistics(roi=roi) if roi is not None else image.get_statistics()
        mean = float(stats.mean)
        stdev = float(stats.stdev)
        if not math.isfinite(mean) or not math.isfinite(stdev):
            return None, None
        return mean, stdev
    except Exception:
        return None, None


def _labels(model):
    try:
        labels = model.labels
        return [str(value).strip() for value in labels] if labels else []
    except Exception:
        return []


def _reset_cause():
    try:
        import machine
        return int(machine.reset_cause())
    except Exception:
        return "unknown"


class ModelPool:
    """Hash-checking, bounded LRU model cache; one arena by default."""

    def __init__(self, config, telemetry):
        self.config = config
        self.telemetry = telemetry
        self.cache = []
        # Reserve keys up front so recording a loaded ROM model's observed
        # hash does not allocate another dictionary slot under arena pressure.
        self.model_hashes = {"detector": None, "small": None, "large": None}
        self.detector_postprocessor = None
        self.limit = int(config["models"].get("max_resident_models", 1))
        self.limit = max(1, min(2, self.limit))

    def _contract(self, role):
        if role == "detector":
            detector = self.config["detection"]
            if detector.get("backend") == "stock_rgb_fomo":
                return detector["stock_rgb_alternative"]
            return detector
        if role == "small":
            return self.config["identity"]["small"]
        return self.config["identity"]["large"]

    def _path(self, role):
        if role == "detector":
            return self._contract(role)["model_path"]
        return self._contract(role)["model_path"]

    def actual_model_sha256(self, role):
        return self.model_hashes.get(role) or self._contract(role).get("model_sha256")

    def _load_model(self, role, path):
        if role == "detector" and self._contract(role).get("backend") != "gray_box_regressor":
            threshold = float(self._contract(role).get("threshold", self.config["detection"]["threshold"]))
            if self.detector_postprocessor is None:
                self.detector_postprocessor = Fomo(threshold=threshold)
            return ml.Model(path, postprocess=self.detector_postprocessor)
        if role == "small" and self.config["detection"]["backend"] == "gray_haar_box":
            # OpenMV 5 tries to put the 101,576-byte model file in the Python
            # heap before sizing its 43,008-byte tensor arena there. Keep the
            # free heap below the file size during load so OpenMV puts the
            # weights in its separate allocator and leaves room for tensors.
            guard = None
            spare = gc.mem_free() - 92 * 1024
            if spare > 0:
                try:
                    guard = bytearray(spare)
                except MemoryError:
                    # A fragmented heap may already force the separate allocator.
                    pass
            model = ml.Model(path)
            if guard is not None:
                guard = None
                gc.collect()
            return model
        return ml.Model(path)

    def _check_metadata(self, role, model):
        contract = self._contract(role)
        input_shape = _shape_tuple(model.input_shape)
        expected_shape = tuple(contract["input_shape"])
        if input_shape != expected_shape:
            raise ValueError("%s_input_shape_mismatch:%s" % (role, repr(input_shape)))
        if _dtype_name(model.input_dtype) != contract["input_dtype"]:
            raise ValueError("%s_input_dtype_mismatch" % role)
        input_scale = float(single_tensor_metadata(model.input_scale, "input_scale"))
        input_zero = int(single_tensor_metadata(model.input_zero_point, "input_zero_point"))
        if abs(input_scale - float(contract["input_scale"])) > 0.000001 or input_zero != int(contract["input_zero_point"]):
            raise ValueError("%s_input_quantization_mismatch" % role)
        output_shape = _shape_tuple(model.output_shape)
        output_dtype = _dtype_name(model.output_dtype)
        stock = role == "detector" and self.config["detection"].get("backend") == "stock_rgb_fomo"
        if stock:
            if not output_shape or output_dtype not in ("int8", "uint8", "float32"):
                raise ValueError("stock_fomo_output_metadata_invalid")
        else:
            if output_shape != tuple(contract["output_shape"]):
                raise ValueError("%s_output_shape_mismatch:%s" % (role, repr(output_shape)))
            if output_dtype != contract["output_dtype"]:
                raise ValueError("%s_output_dtype_mismatch" % role)
        if contract.get("output_dtype") == "int8":
            output_scale = float(single_tensor_metadata(model.output_scale, "output_scale"))
            if abs(output_scale - float(contract["output_scale"])) > 0.000001:
                raise ValueError("%s_output_scale_mismatch" % role)
            output_zero = int(single_tensor_metadata(model.output_zero_point, "output_zero_point"))
            if output_zero != int(contract["output_zero_point"]):
                raise ValueError("%s_output_zero_point_mismatch" % role)
        labels = _labels(model)
        if role == "detector":
            if contract.get("backend") == "gray_box_regressor":
                if output_shape != (1, 5):
                    raise ValueError("box_regressor_requires_single_five_value_tensor")
            else:
                if contract.get("class_order") and labels not in (contract["class_order"], contract["class_order"][1:]):
                    raise ValueError("detector_label_order_mismatch")
                class_index_from_labels(labels, output_shape[-1], "face")
        elif role == "large" and labels != list(contract["class_order"]):
            raise ValueError("classifier_label_order_mismatch")
        return {"input_shape": list(input_shape), "input_dtype": _dtype_name(model.input_dtype),
                "input_scale": input_scale, "input_zero_point": input_zero,
                "output_shape": list(output_shape), "output_dtype": _dtype_name(model.output_dtype),
                "output_scale": float(single_tensor_metadata(model.output_scale, "output_scale")),
                "output_zero_point": int(single_tensor_metadata(model.output_zero_point, "output_zero_point")),
                "arena_bytes": int(model.ram), "model_bytes": int(model.len),
                "labels": labels}

    def acquire(self, role):
        start = time.ticks_us()
        for index, pair in enumerate(self.cache):
            if pair[0] == role:
                self.cache.append(self.cache.pop(index))
                model_id = self._contract(role).get("model_id", "stock-rgb-fomo") if role != "detector" else "C-detector"
                self.telemetry.emit("model_cache_hit", {"model_id": model_id,
                                                         "warm": True})
                return self.cache[-1][1], True
        before = gc.mem_free()
        gc.collect()
        gc_us = time.ticks_diff(time.ticks_us(), start)
        self.telemetry.emit("gc_collect", {"reason": "before_model_load", "duration_us": gc_us,
                                           "heap_free_before_bytes": before,
                                           "heap_free_after_bytes": gc.mem_free()})
        while len(self.cache) >= self.limit:
            old_role, old_model = self.cache.pop(0)
            self.model_hashes[old_role] = None
            old_id = self._contract(old_role).get("model_id", "stock-rgb-fomo") if old_role != "detector" else "C-detector"
            old_model = None
            swap_start = time.ticks_us()
            before_gc = gc.mem_free()
            gc.collect()
            self.telemetry.emit("model_swap", {"evicted_model_id": old_id,
                "duration_us": time.ticks_diff(time.ticks_us(), swap_start),
                "gc_heap_free_before_bytes": before_gc, "gc_heap_free_after_bytes": gc.mem_free()})
        minimum = int(self.config["models"].get("minimum_python_heap_free_bytes", 49152))
        if not enough_heap(gc.mem_free(), minimum):
            raise MemoryError("low_python_heap_before_model_load")
        contract = self._contract(role)
        path = self._path(role)
        cold_start = time.ticks_us()
        hash_start = time.ticks_us()
        actual_hash = _sha256_file(path)
        hash_us = time.ticks_diff(time.ticks_us(), hash_start)
        expected_hash = contract.get("model_sha256")
        if expected_hash and not hash_matches(actual_hash, expected_hash):
            raise ValueError("model_hash_mismatch:%s" % role)
        # Reading the model for SHA-256 validation creates temporary chunks.
        # On the Nicla's MicroPython heap they may remain unreclaimed until an
        # explicit collection, leaving too little contiguous room for ml.Model
        # even though a later standalone preload succeeds. Collect after the
        # hash pass and record the recovery before allocating the model arena.
        heap_free_after_hash_bytes = gc.mem_free()
        hash_gc_start = time.ticks_us()
        gc.collect()
        hash_gc_us = time.ticks_diff(time.ticks_us(), hash_gc_start)
        heap_free_after_hash_gc_bytes = gc.mem_free()
        self.telemetry.emit("gc_collect", {
            "reason": "after_model_hash_before_load",
            "duration_us": hash_gc_us,
            "heap_free_after_hash_bytes": heap_free_after_hash_bytes,
            "heap_free_after_hash_gc_bytes": heap_free_after_hash_gc_bytes,
            "recovered_bytes": max(0, heap_free_after_hash_gc_bytes - heap_free_after_hash_bytes)})
        load_start = time.ticks_us()
        heap_free_before_load_bytes = gc.mem_free()
        try:
            model = self._load_model(role, path)
        except Exception as error:
            heap_free_at_error_bytes = gc.mem_free()
            gc_start = time.ticks_us()
            gc.collect()
            heap_free_after_error_gc_bytes = gc.mem_free()
            try:
                model_file_bytes = int(os.stat(path)[6])
            except Exception:
                model_file_bytes = None
            model_id = contract.get("model_id", "stock-rgb-fomo") if role != "detector" else "C-trained-gray-fomo"
            self.telemetry.emit("model_load_failed", {
                "role": role, "model_id": model_id,
                "model_sha256": expected_hash or actual_hash,
                "model_bytes": model_file_bytes,
                "model_path": path,
                "python_heap_free_before_load_bytes": heap_free_before_load_bytes,
                "python_heap_free_after_hash_bytes": heap_free_after_hash_bytes,
                "python_heap_free_after_hash_gc_bytes": heap_free_after_hash_gc_bytes,
                "hash_gc_us": hash_gc_us,
                "python_heap_free_at_error_bytes": heap_free_at_error_bytes,
                "python_heap_free_after_error_gc_bytes": heap_free_after_error_gc_bytes,
                "failure_cleanup_gc_us": time.ticks_diff(time.ticks_us(), gc_start),
                "resident_roles": [item[0] for item in self.cache],
                "resident_model_limit": self.limit,
                "error_type": type(error).__name__, "error": str(error)[:120]})
            raise
        load_us = time.ticks_diff(time.ticks_us(), load_start)
        metadata = self._check_metadata(role, model)
        cold_load_validate_us = time.ticks_diff(time.ticks_us(), cold_start)
        if gc.mem_free() < 0:
            model = None
            gc.collect()
            raise MemoryError("negative_heap_after_model_load")
        self.cache.append((role, model))
        self.model_hashes[role] = actual_hash
        model_id = contract.get("model_id", "stock-rgb-fomo") if role != "detector" else "C-detector"
        self.telemetry.emit("model_load", {"model_id": model_id, "model_sha256": expected_hash or actual_hash,
            "load_us": load_us, "hash_check_us": hash_us,
            "hash_gc_us": hash_gc_us,
            "heap_free_after_hash_bytes": heap_free_after_hash_bytes,
            "heap_free_after_hash_gc_bytes": heap_free_after_hash_gc_bytes,
            "cold_load_validate_us": cold_load_validate_us, "warm": False, "gc_free_before_bytes": before,
            "gc_free_after_bytes": gc.mem_free(), "metadata": metadata,
            "cache_count": len(self.cache), "cache_limit": self.limit})
        return model, False

    def clear(self, reason, keep_role=None):
        count = len(self.cache)
        released_detector = keep_role != "detector" and any(
            pair[0] == "detector" for pair in self.cache)
        if keep_role is None:
            self.cache = []
            for role in self.model_hashes:
                self.model_hashes[role] = None
        else:
            kept = []
            released = 0
            for pair in self.cache:
                if pair[0] == keep_role:
                    kept.append(pair)
                else:
                    released += 1
                    self.model_hashes[pair[0]] = None
            self.cache = kept
            count = released
            pair = None
        if released_detector:
            self.detector_postprocessor = None
        start = time.ticks_us()
        before = gc.mem_free()
        gc.collect()
        self.telemetry.emit("model_cache_cleared", {"reason": reason, "released_models": count,
            "kept_roles": [item[0] for item in self.cache],
            "gc_us": time.ticks_diff(time.ticks_us(), start),
            "heap_free_before_bytes": before, "heap_free_after_bytes": gc.mem_free()})


class NiclaRuntime:
    def __init__(self, config, manifest, camera, usb, uart, telemetry):
        self.config = config
        self.manifest = manifest
        self.camera = camera
        self.usb = usb
        self.uart = uart
        self.telemetry = telemetry
        self.control = LineInput(usb, uart,
            config["telemetry"]["control_line_max_bytes"], 64)
        self.pool = ModelPool(config, telemetry)
        self.haar_cascade = None
        self.haar_wide_input = None
        self.haar_crop_input = None
        global BOOT_PRELOADED_DETECTOR, BOOT_PRELOADED_FOMO, BOOT_PRELOADED_SMALL
        if BOOT_PRELOADED_DETECTOR is not None:
            self.pool.detector_postprocessor = BOOT_PRELOADED_FOMO
            self.pool.cache.append(("detector", BOOT_PRELOADED_DETECTOR))
            self.pool.model_hashes["detector"] = BOOT_DETECTOR_PRELOAD_STATUS.get("model_sha256")
            BOOT_PRELOADED_DETECTOR = None
            BOOT_PRELOADED_FOMO = None
        if BOOT_PRELOADED_SMALL is not None:
            self.pool._check_metadata("small", BOOT_PRELOADED_SMALL)
            self.pool.cache.append(("small", BOOT_PRELOADED_SMALL))
            self.pool.model_hashes["small"] = BOOT_DETECTOR_PRELOAD_STATUS["early_small_model_sha256"]
            BOOT_PRELOADED_SMALL = None
        self.policy_id = config.get("default_policy", "G")
        self.policy = config["policies"][self.policy_id]
        self.measurement_stage_id = "full_pipeline"
        self.measurement_model_slot = "small"
        self.temporal = TemporalConfirmer(self.policy.get("temporal_confirmation") == "2_of_3")
        self._reset_scene_trigger()
        self.escalation_budget = EscalationBudget(
            self.policy.get("max_escalations_per_minute", 0), 60000, time.ticks_diff)
        self.current_cycle_start_ticks = None
        # A reset/reconnect reports the actual stopped state. Acquisition
        # starts only after the operator's explicit `start` command.
        self.running = False
        self.one_shot_pending = False
        self.one_shot_command_id = None
        self.one_shot_identity_probe_on_no_face = False
        self.one_shot_identity_probe_slot = "small"
        self.current_observation_command_id = None
        self.current_command_id = None
        self.current_command_name = None
        self.recent_command_ids = []
        self.model_override = None
        self.current_frame = None
        self.current_frame_jpeg_bytes = None
        self.current_frame_jpeg_meta = None
        self.detector_tensor_bytes = None
        self.identity_tensor_bytes = None
        self.detector_tensor_exported = False
        self.center_crop_diagnostic_tensor_exported = False
        self.latest_imu = {"acceleration_mg": None, "angular_rate_mdps": None,
                           "motion_detected": None, "sampled_ticks_ms": None}
        self.last_requested_state = None
        self.sleeping = False
        self.wake_failures = 0
        self.wake_retry_after = None
        self.next_capture = time.ticks_ms()
        self.last_face_ticks = time.ticks_ms()
        self.last_motion = False
        self.last_status = "BOOT"
        self.frame_index = 0
        self.trial_frames_remaining = None
        self.trial_deadline = None
        self.active_trial_id = None
        self.last_camera_rate = None
        self.last_resolution = None
        self.gallery = None
        self.gallery_error = None
        self.gallery_error_type = None
        self.gallery_loaded = False
        self.startup_detector_preload = dict(BOOT_DETECTOR_PRELOAD_STATUS)
        self.last_seed_id = None
        self.last_trial_id = None
        self.battery_store = None
        self.battery_list_snapshot = None
        self.battery_armed = False
        self.battery_last_face = None
        self.battery_encounter_saved = False
        self.battery_error = None
        try:
            self.battery_armed = (self._battery_available() and
                bool(os.stat(BATTERY_ARM_PATH)[0] & 0x4000))
        except OSError:
            pass
        # R27 showed that unloading and later reloading D can fail even when
        # gc.mem_free() reports ample aggregate heap. For the resident-model
        # route, validate and load the small gallery before D is allocated so
        # the checked model arena can remain resident across frames.
        if self.config["models"].get("retain_small_model_after_prediction", False):
            self._load_gallery()

    def _load_gallery(self):
        started = time.ticks_us()
        heap_before = gc.mem_free()
        try:
            path = self.config["identity"]["small"]["gallery_path"]
            actual = _sha256_file(path)
            expected_gallery = self.manifest["gallery"]["sha256"]
            if not hash_matches(actual, expected_gallery):
                raise ValueError("gallery_file_hash_mismatch")
            gallery = _read_json(path)
            ok, reason = validate_gallery_metadata(gallery,
                self.config["identity"]["small"]["model_id"],
                self.config["identity"]["small"]["model_sha256"],
                self.config["identity"]["small"]["prototype_sha256"], KNOWN)
            if not ok:
                raise ValueError(reason)
            if gallery["calibration"]["threshold"] != self.config["identity"]["small"]["threshold"]:
                raise ValueError("gallery_calibration_mismatch")
            self.gallery = gallery
            self.gallery_error = None
            self.gallery_error_type = None
            self.gallery_loaded = True
            self.telemetry.emit("gallery_ready", {"gallery_sha256": actual,
                "prototype_sha256": gallery["prototype_sha256"], "roster_size": len(gallery["entries"]),
                "heap_free_before_bytes": heap_before, "heap_free_after_bytes": gc.mem_free(),
                "load_us": time.ticks_diff(time.ticks_us(), started)})
        except Exception as error:
            self.gallery = None
            self.gallery_error_type = type(error).__name__
            self.gallery_error = str(error)[:120]
            self.gallery_loaded = True
            self.telemetry.emit("gallery_invalid", {"reason_code": self.gallery_error,
                "heap_free_before_bytes": heap_before, "heap_free_after_bytes": gc.mem_free(),
                "load_us": time.ticks_diff(time.ticks_us(), started)})

    def _preload_small_for_capture(self, purpose):
        if self.pool.limit < 2:
            return "not_enabled_single_resident_model"
        if any(pair[0] == "small" for pair in self.pool.cache):
            return "warm"
        try:
            _model, warm = self.pool.acquire("small")
            _model = None
            return "warm" if warm else "loaded"
        except Exception as error:
            self.telemetry.emit("identity_preload_failed", {
                "purpose": purpose, "model_id": self.config["identity"]["small"].get("model_id"),
                "error_type": type(error).__name__, "reason_code": str(error)[:100],
                "heap_free_bytes": gc.mem_free(), "resident_roles": [pair[0] for pair in self.pool.cache]})
            return "failed:" + type(error).__name__

    def _reset_scene_trigger(self):
        trigger = self.policy.get("scene_trigger")
        self.scene_trigger = SimpleSceneTrigger(trigger, time.ticks_diff) if trigger else None
        self.scene_signature = (bytearray(SCENE_SIGNATURE_LENGTH)
            if self.scene_trigger is not None and self.scene_trigger.spatial_enabled else None)

    def set_policy(self, policy_id):
        if policy_id not in self.config["policies"]:
            raise ValueError("unsupported_policy")
        before = self.policy_id
        self.policy_id = policy_id
        self.policy = self.config["policies"][policy_id]
        self.temporal = TemporalConfirmer(self.policy.get("temporal_confirmation") == "2_of_3")
        self._reset_scene_trigger()
        self.escalation_budget = EscalationBudget(
            self.policy.get("max_escalations_per_minute", 0), 60000, time.ticks_diff)
        resolution_before = self.last_resolution
        rate_before = self.last_camera_rate
        # _configure_camera_actions already compares each applied camera value.
        # Clearing the cache here made an idempotent set_policy("S") reapply
        # framesize/window/framebuffers and allocate another camera buffer.
        self._configure_camera_actions()
        camera_reconfigured = (resolution_before != self.last_resolution or
                               rate_before != self.last_camera_rate)
        self.telemetry.emit("policy_changed", {"previous_policy_id": before,
            "policy_id": self.policy_id, "interval_ms": self._interval_ms(self.last_motion),
            "identity_backend": self.policy["identity_backend"],
            "camera_reconfigured": camera_reconfigured})

    def _interval_ms(self, moving):
        interval = self.policy.get("interval_ms")
        if interval == "adaptive":
            return int(self.policy["moving_interval_ms"] if moving else self.policy["still_interval_ms"])
        return int(interval)

    def _configure_camera_actions(self):
        resolution = self.policy.get("resolution", self.config["camera"]["initial_framesize"])
        if resolution not in self.config["camera"]["resolution_options"]:
            raise ValueError("unsupported_camera_resolution")
        if resolution != self.last_resolution:
            self.camera.framesize(getattr(csi, resolution))
            window = self.config["camera"]["windows"][resolution]
            self.camera.window((int(window[2]), int(window[3])))
            self.camera.framebuffers(int(self.config["camera"]["framebuffer_count"]))
            self.last_resolution = resolution
            if self.scene_signature is not None:
                self.scene_trigger.reset_reference()
            self.telemetry.emit("camera_resolution", {"resolution": resolution,
                "window_px": [int(window[2]), int(window[3])],
                "sensor_window_x": int(window[0]), "sensor_window_y": int(window[1])})
        interval = self._interval_ms(self.last_motion)
        rate = max(1, min(int(self.config["camera"]["max_frame_rate_hz"]), int(1000 / max(1, interval))))
        if rate != self.last_camera_rate:
            self.camera.framerate(rate)
            self.last_camera_rate = rate
            self.telemetry.emit("camera_frame_rate", {"frame_rate_hz": rate,
                "interval_ms": interval, "action": "sensor_frame_dropping"})

    def _motion(self):
        accel = None
        angular = None
        try:
            accel = imu.acceleration_mg()
        except Exception:
            pass
        try:
            angular = imu.angular_rate_mdps()
        except Exception:
            pass
        motion, reason = motion_detected(accel, angular,
            self.config["motion"]["acceleration_delta_mg"],
            self.config["motion"]["angular_rate_mdps"])
        return motion, reason, accel, angular

    def _wake_camera(self, reason):
        if not self.sleeping:
            return
        start = time.ticks_us()
        try:
            self.camera.sleep(False)
            self.sleeping = False
            settle = int(self.config["camera"]["exposure_settle_ms"])
            end = time.ticks_add(time.ticks_ms(), settle)
            while time.ticks_diff(end, time.ticks_ms()) > 0:
                self._feed_watchdog()
                time.sleep_ms(50)
            self.telemetry.emit("camera_wake", {"reason_code": reason,
                "wake_to_settled_us": time.ticks_diff(time.ticks_us(), start),
                "first_frame_quality_attempts": int(self.config["camera"]["wake_frame_attempts"])})
            self.wake_pending = True
            if self.scene_signature is not None:
                self.scene_trigger.reset_reference()
            self.next_capture = time.ticks_ms()
            self.wake_failures = 0
            self.wake_retry_after = None
            return True
        except Exception as error:
            self.sleeping = True
            self.temporal.reset()
            self.wake_failures += 1
            retry_ms = camera_wake_retry_ms(self.wake_failures)
            now = time.ticks_ms()
            self.wake_retry_after = time.ticks_add(now, retry_ms)
            self.next_capture = self.wake_retry_after
            self.telemetry.emit("camera_wake_failed", {"reason_code": str(error)[:100],
                "failure_count": self.wake_failures, "retry_backoff_ms": retry_ms,
                "camera_state": "sleeping_or_unknown"})
            self._led("FAULT")
            return False

    def _sleep_camera(self, reason, force=False):
        if self.sleeping or (not force and reason != "measurement_stage_idle" and
                             not self.policy.get("camera_sleep", False)):
            return
        try:
            self.camera.sleep(True)
            self.sleeping = True
            self.temporal.reset()
            self.telemetry.emit("camera_sleep", {"reason_code": reason,
                "frame_dropping_rate_hz": self.last_camera_rate,
                "note": "CSI sensor sleep; distinct from framerate frame dropping"})
        except Exception as error:
            self.telemetry.emit("camera_sleep_failed", {"reason_code": str(error)[:100]})

    def _stream_model_input(self, kind, input_buffer, model_input, shape, dtype, contract,
                            model_sha256=None):
        """Hash and send the actual predict buffer before the model can reuse it."""
        if not getattr(self, "export_exact_inputs", True):
            return {"status": "not_exported_live"}
        actual_shape = []
        try:
            actual_shape = [int(value) for value in shape]
        except Exception:
            actual_shape = list(contract.get("input_shape") or [])
        actual_dtype = _dtype_name(dtype)
        metadata = {"boot_id": self.telemetry.run_id,
            "frame_id": self.frame_index,
            "shape": actual_shape,
            "dtype": actual_dtype,
            "scale": contract.get("input_scale"),
            "zero_point": contract.get("input_zero_point"),
            "model_id": contract.get("model_id"),
            "model_sha256": model_sha256 or contract.get("model_sha256"),
            "value_encoding": "raw_tensor_bytes_as_given_to_model_predict"}
        started = time.ticks_us()
        try:
            result = self.telemetry.send_packet(kind, metadata, input_buffer)
        except Exception as error:
            result = {"sent": False,
                "reason": type(error).__name__ + ":" + str(error)[:80]}
        elapsed_us = time.ticks_diff(time.ticks_us(), started)
        expected_sha = model_input.input_sha256
        packet_sha = result.get("sha256")
        sent = bool(result.get("sent"))
        hash_match = bool(sent and expected_sha and packet_sha == expected_sha)
        if sent and not hash_match:
            sent = False
            result["reason"] = "packet_hash_differs_from_predict_callback_hash"
        return {"kind": kind, "status": "sent" if sent else "failed",
            "byte_count": len(input_buffer), "sha256": expected_sha,
            "packet_sha256": packet_sha, "hash_match": hash_match,
            "packet_bytes": result.get("packet_bytes"),
            "transport_us": elapsed_us, "reason": result.get("reason"),
            "metadata": metadata}

    def _detect_haar_faces(self, frame, fields):
        contract = self.config["detection"]
        started = time.ticks_us()
        threshold, pyramid_scale = contract["threshold"], contract["pyramid_scale"]
        preprocessing_us = predict_us = 0
        boxes = []
        source_x = source_y = 0
        width, height = frame.width(), frame.height()
        source_width, source_height = width, height
        wide_ready = self.haar_wide_input is not None and width == height == 240
        distance_side = contract.get("distance_roi_side_px")
        close_ready = wide_ready and contract.get("close_face_content_px")
        scan_order = ("small",)
        if wide_ready:
            if fields.get("compact_detector_first") and distance_side and close_ready:
                scan_order = ("close", "distance", "small", "wide")
            else:
                scan_order = ("wide", "small")
                if distance_side:
                    scan_order = (("wide", "distance", "small") if contract.get("distance_roi_before_small")
                                  else ("wide", "small", "distance"))
                if close_ready:
                    scan_order += ("close",)
        for search in scan_order:
            if boxes:
                break
            preprocess_start = time.ticks_us()
            source_x = source_y = 0
            source_width, source_height = width, height
            if search == "wide":
                size = 240
                small = self.haar_wide_input
                small.draw_image(frame, 0, 0, x_scale=240.01 / 240, y_scale=240.01 / 240, hint=image.BILINEAR)
                local_contrast = contract.get("wide_local_contrast_clip_limit")
                if local_contrast is not None:
                    try:
                        small.histeq(adaptive=True, clip_limit=local_contrast)
                    except MemoryError:
                        small.draw_image(frame, 0, 0, x_scale=240.01 / 240,
                                         y_scale=240.01 / 240, hint=image.BILINEAR)
                        fields["wide_local_contrast_fallback"] = "scratch_allocation_failed"
                        local_contrast = None
                if local_contrast is None:
                    wide_gain = contract["wide_contrast_gain"]
                    wide_center = contract.get("wide_contrast_center", 127)
                    small.gamma_corr(gamma=1.0, contrast=float(wide_gain),
                                     brightness=(127 - wide_center * wide_gain + .25) / 255.0)
                input_encoding = ("native_local_contrast_haar_detector_grayscale_pixels" if local_contrast is not None
                                  else "native_contrast_stretched_haar_detector_grayscale_pixels")
            elif search == "distance":
                # Use scratch storage so the identity crop still reads the original frame.
                size = 240
                source_x = (width - distance_side) // 2
                source_y = (height - distance_side) // 2 - 20
                source_width = source_height = distance_side
                roi = (source_x, source_y, distance_side, distance_side)
                small = self.haar_wide_input
                zoom = (size + .01) / distance_side
                small.draw_image(frame, 0, 0, roi=roi, x_scale=zoom, y_scale=zoom,
                                 hint=image.BILINEAR)
                adaptive_zoom = contract.get("adaptive_distance_zoom", False)
                if adaptive_zoom:
                    stats = small.get_statistics()
                    gain = min(4.0, max(1.0, 58 / max(1, stats.stdev)))
                    small.gamma_corr(gamma=1.0, contrast=gain,
                                     brightness=(127 - stats.mean * gain + .25) / 255.0)
                input_encoding = ("native_adaptive_roi_haar_detector_grayscale_pixels"
                                  if adaptive_zoom else "native_upper_center_roi_haar_detector_grayscale_pixels")
            elif search == "close":
                size = 96
                small = self.haar_crop_input
                content = contract["close_face_content_px"]
                inset = (size - content) // 2
                high = inset + content
                mean = frame.get_statistics().mean
                small.draw_rectangle((0, 0, size, size), color=mean, fill=True)
                small.draw_image(frame, inset, inset, x_scale=(content + .01) / width,
                                 y_scale=(content + .01) / height, hint=image.BILINEAR)
                try:
                    small.histeq(adaptive=True, clip_limit=2)
                except MemoryError:
                    fields["close_face_status"] = "scratch_allocation_failed"
                    if search != scan_order[-1]:
                        preprocessing_us += time.ticks_diff(time.ticks_us(), preprocess_start)
                        continue
                    # Restore the prior narrow input for diagnostics when the
                    # close scan is the last fallback.
                    small.draw_image(frame, 0, 0, x_scale=96.01 / width,
                                     y_scale=96.01 / height, hint=image.BILINEAR)
                    small.histeq()
                    input_encoding = "native_equalized_haar_detector_grayscale_pixels"
                    preprocessing_us += time.ticks_diff(time.ticks_us(), preprocess_start)
                    break
                input_encoding = "native_zoomed_out_haar_detector_grayscale_pixels"
            else:
                size = 96
                source_x = source_y = 0
                source_width, source_height = width, height
                small = self.haar_crop_input
                small.draw_image(frame, 0, 0, x_scale=(size + .01) / width,
                                 y_scale=(size + .01) / height, hint=image.BILINEAR)
                small.histeq()
                input_encoding = "native_equalized_haar_detector_grayscale_pixels"
            preprocessing_us += time.ticks_diff(time.ticks_us(), preprocess_start)
            predict_start = time.ticks_us()
            boxes = small.find_features(self.haar_cascade, threshold=threshold, scale=pyramid_scale)
            predict_us += time.ticks_diff(time.ticks_us(), predict_start)
            if search == "close":
                valid = []
                for a, b, w, h in boxes:
                    left, top = max(inset, a), max(inset, b)
                    right, bottom = min(high, a+w), min(high, b+h)
                    if (left < right and top < bottom and inset <= a+w/2 < high and
                            inset <= b+h/2 < high and (right-left)*(bottom-top)*5 >= w*h*4):
                        valid.append((left, top, right-left, bottom-top))
                fields["close_face_status"] = "box" if valid else "no_box"
                fields["close_face_padding_rejections"] = len(boxes)-len(valid)
                boxes = valid
        fields["preprocess_us"] = preprocessing_us
        fields["model_predict_inclusive_us"] = predict_us
        if not fields.get("export_exact_inputs", True):
            fields["detector_input"] = {"shape": [1, size, size, 1], "dtype": "uint8",
                "value_encoding": input_encoding, "payload_status": "not_exported_live",
                "model_id": contract["model_id"], "model_sha256": contract["cascade_sha256"]}
        elif self.telemetry.usb_connected():
            pixels = small.bytearray()
            digest = _sha256_bytes(pixels)
            packet_meta = {"boot_id": self.telemetry.run_id, "frame_id": self.frame_index,
                "shape": [1, size, size, 1], "dtype": "uint8", "scale": 1.0, "zero_point": 0,
                "model_id": contract["model_id"], "model_sha256": contract["cascade_sha256"],
                "value_encoding": input_encoding}
            packet_start = time.ticks_us()
            sent = self.telemetry.send_packet("detector_input", packet_meta, pixels)
            match = bool(sent.get("sent") and sent.get("sha256") == digest)
            fields["detector_input_packet"] = {"kind": "detector_input",
                "status": "sent" if match else "failed", "sha256": digest,
                "packet_sha256": sent.get("sha256"), "hash_match": match,
                "byte_count": len(pixels), "metadata": packet_meta,
                "reason": None if match else sent.get("reason") or "input_packet_hash_mismatch",
                "transport_us": time.ticks_diff(time.ticks_us(), packet_start),
                "packet_bytes": sent.get("packet_bytes")}
            fields["detector_input"] = dict(packet_meta, sha256=digest, byte_count=len(pixels),
                packet_sha256=sent.get("sha256"), packet_hash_match=match)
            pixels = None
        small = None
        gc.collect()
        scale_x, scale_y = source_width / size, source_height / size
        detections, regions = [], []
        for x, y, w, h in boxes[:8]:
            if input_encoding == "native_zoomed_out_haar_detector_grayscale_pixels":
                rect = [(x-inset)*width/content, (y-inset)*height/content,
                        w*width/content, h*height/content]
            else:
                rect = [source_x + x * scale_x, source_y + y * scale_y,
                        w * scale_x, h * scale_y]
            center = (rect[0] + rect[2] / 2.0, rect[1] + rect[3] / 2.0)
            regions.append({"rect_xywh_px": rect, "center_xy_px": center, "score": None})
            if len(detections) < 2:
                detections.append({"rect": rect, "center": center, "score": None})
        count = len(boxes)
        fields.update({"detect_latency_us": time.ticks_diff(time.ticks_us(), started),
            "detection_coordinate_space": "camera_xywh_px", "detector_regions": regions,
            "detector_raw_accepted_cluster_count": count, "detector_regions_truncated": count > len(regions),
            "detector_region_semantics": "native Haar frontal face boxes; no confidence score or landmark alignment",
            "detection_threshold": contract["threshold"], "face_count": count,
            "detector_status": "NO_FACE" if count == 0 else "ONE_FACE" if count == 1 else "MULTIPLE_OR_AMBIGUOUS",
            "detector_model": {"model_id": contract["model_id"], "model_sha256": contract["cascade_sha256"],
                "warm": True, "model_bytes": contract["cascade_bytes"], "arena_bytes": None,
                "model_predict_started": True}})
        return detections, count

    def _detect_faces(self, frame, fields):
        if self.haar_cascade is not None:
            return self._detect_haar_faces(frame, fields)
        detect_started = time.ticks_us()
        contract = self.pool._contract("detector")
        acquire_start = time.ticks_us()
        model, warm = self.pool.acquire("detector")
        fields["detector_model_acquire_us"] = time.ticks_diff(time.ticks_us(), acquire_start)
        labels = _labels(model)
        detector_id = contract.get("model_id", "C-trained-gray-fomo")
        detector_sha = self.pool.actual_model_sha256("detector")
        fields["detector_model"] = {"model_id": detector_id,
            "model_sha256": detector_sha,
            "warm": bool(warm), "arena_bytes": int(model.ram),
            "model_bytes": int(model.len),
            "input_shape": list(contract["input_shape"]),
            "input_dtype": contract["input_dtype"],
            "input_scale": contract["input_scale"],
            "input_zero_point": contract["input_zero_point"],
            "model_predict_started": True}
        fomo_postprocess = self.pool.detector_postprocessor
        fomo_timing = {"duration_us": None}
        def timed_fomo_postprocess(model_for_postprocess, inputs_for_postprocess,
                                   outputs_for_postprocess):
            post_started = time.ticks_us()
            try:
                return fomo_postprocess(model_for_postprocess,
                    inputs_for_postprocess, outputs_for_postprocess)
            finally:
                fomo_timing["duration_us"] = time.ticks_diff(
                    time.ticks_us(), post_started)

        predict_started = time.ticks_us()
        def capture_detector_input(input_buffer, model_input, shape, dtype):
            summary = self._stream_model_input("detector_input", input_buffer,
                model_input, shape, dtype, contract, detector_sha)
            fields["detector_input_packet"] = summary
            fields["preprocess_us"] = model_input.preprocess_us
            fields["detector_input"] = {"shape": list(contract["input_shape"]),
                "dtype": contract["input_dtype"], "scale": contract["input_scale"],
                "zero_point": contract["input_zero_point"],
                "model_id": detector_id, "model_sha256": detector_sha,
                "sha256": model_input.input_sha256,
                "byte_count": model_input.input_byte_count,
                "packet_status": summary.get("status"),
                "packet_sha256": summary.get("packet_sha256"),
                "packet_hash_match": summary.get("hash_match"),
                "packet_transport_us": summary.get("transport_us")}
            return summary
        if contract.get("backend") == "stock_rgb_fomo":
            # Stock ROM FOMO consumes RGB. Use the pinned OpenMV firmware's
            # native RGB565 Normalization path rather than resizing it through
            # the grayscale-only reference implementation below.
            inputs = openmv_normalization_model_input(frame,
                capture_callback=capture_detector_input,
                clock_us=time.ticks_us, clock_diff=time.ticks_diff)
        else:
            # Stream the exact model-owned buffer while the callback still has
            # it. This avoids a post-predict alias and a second tensor-sized
            # allocation while the detector arena is resident.
            inputs = pillow_bilinear_model_input(frame,
                capture_callback=capture_detector_input,
                clock_us=time.ticks_us, clock_diff=time.ticks_diff)
        box_regressor = contract.get("backend") == "gray_box_regressor"
        outputs = (model.predict([inputs]) if box_regressor else
                   model.predict([inputs], callback=timed_fomo_postprocess))
        predict_total_us = time.ticks_diff(time.ticks_us(), predict_started)
        fomo_postprocess_us = fomo_timing["duration_us"]
        self.detector_tensor_bytes = None
        fields["detector_model"]["model_predict_inclusive_us"] = predict_total_us
        fields["predict_callback_transport_us"] = (fields.get("detector_input_packet") or {}).get("transport_us")
        fields["model_predict_inclusive_us"] = predict_total_us
        fields["fomo_postprocess_us"] = fomo_postprocess_us
        fields["preprocess_us"] = inputs.preprocess_us
        fields["model_compute_residual_us"] = max(0, predict_total_us -
            int(inputs.preprocess_us or 0) -
            int((fields.get("detector_input_packet") or {}).get("transport_us") or 0) -
            int(fomo_postprocess_us or 0))
        decode_started = time.ticks_us()
        if box_regressor:
            values = prediction_values(_flatten(outputs), 5)
            score = values[0]
            detections, detector_regions = [], []
            if strict_score_pass(score, contract["threshold"]):
                score, rect = normalized_face_box(values, frame.width(), frame.height())
                center = (rect[0] + rect[2] / 2.0, rect[1] + rect[3] / 2.0)
                detector_regions.append({"center_xy_px": center, "rect_xywh_px": rect, "score": score})
                detections.append({"center": center, "rect": rect, "score": score})
            outputs = inputs = None
            gc.collect()
            count = len(detections)
            fields.update({"detect_latency_us": time.ticks_diff(time.ticks_us(), detect_started),
                "detector_python_decode_us": time.ticks_diff(time.ticks_us(), decode_started),
                "detector_warm": warm, "detection_coordinate_space": "camera_xywh_px",
                "detection_threshold": contract["threshold"], "threshold_comparison": "strict_gt_float32",
                "detector_raw_accepted_cluster_count": count, "detector_regions": detector_regions,
                "detector_regions_truncated": False, "face_count": count,
                "detector_region_semantics": "predicted face box, single-box model; no landmarks or multiple-face guarantee",
                "detection_score": score, "detector_status": "ONE_FACE" if count else "NO_FACE"})
            return detections, count
        tensor_channels = _shape_tuple(model.output_shape)[-1]
        face_channel = class_index_from_labels(labels, tensor_channels, "face")
        postprocessed_channels = len(outputs) if outputs is not None else 0
        face_index = face_channel if postprocessed_channels == 0 else fomo_face_output_index(
            labels, tensor_channels, postprocessed_channels)
        detections = []
        detector_regions = []
        total = 0
        if postprocessed_channels:
            if face_index >= postprocessed_channels:
                raise ValueError("detector_face_output_index_out_of_range")
            for rect, score in outputs[face_index]:
                score = float(score)
                if strict_score_pass(score, contract["threshold"]):
                    total += 1
                    center = fomo_center_camera_px(rect, frame.width(), frame.height())
                    if len(detector_regions) < 8:
                        detector_regions.append({"center_xy_px": center,
                            "rect_xywh_px": [float(value) for value in rect],
                            "score": score})
                    if len(detections) < 2:
                        detections.append({"center": center,
                                           "rect": [float(value) for value in rect],
                                           "score": score})
        stored_accepted_count = len(detections)
        duplicate_peaks_merged = False
        merged_center_span_px = None
        face_count = total
        merge_enabled = bool(contract.get("merge_near_duplicate_peaks", False))
        if total == 2 and stored_accepted_count == 2:
            detections, duplicate_peaks_merged, merged_center_span_px = \
                merge_near_duplicate_fomo_peaks(detections, 48.0, enabled=merge_enabled)
            if duplicate_peaks_merged:
                face_count = 1
        outputs = None
        inputs = None
        gc.collect()
        decode_us = time.ticks_diff(time.ticks_us(), decode_started)
        fields.update({"detect_latency_us": time.ticks_diff(time.ticks_us(), detect_started),
            "detector_python_decode_us": decode_us,
            "detector_warm": warm,
            "detector_face_channel": face_channel,
            "detector_postprocess_face_index": face_index,
            "detection_coordinate_space": "camera_xywh_px",
            "detector_labels": labels,
            "detection_threshold": contract["threshold"], "threshold_comparison": "strict_gt_float32",
            "detector_raw_accepted_cluster_count": total,
            "detector_regions": detector_regions,
            "detector_regions_truncated": total > len(detector_regions),
            "detector_region_semantics": "FOMO postprocessed heatmap regions; not landmark-aligned face boxes",
            "detector_duplicate_peak_merge_enabled": merge_enabled,
            "detector_duplicate_peaks_merged": duplicate_peaks_merged,
            "detector_merge_policy": "near_duplicate_centers_max_span_48px_opt_in_v2",
            "detector_merged_pair_center_span_px": merged_center_span_px,
            "face_count": face_count, "face_records_truncated": total > stored_accepted_count,
            "detector_status": ("NO_FACE" if face_count == 0 else
                "ONE_FACE_DUPLICATE_PEAKS_MERGED" if duplicate_peaks_merged else
                "ONE_FACE" if face_count == 1 else "MULTIPLE_OR_AMBIGUOUS")})
        return detections, face_count

    def _predict_identity(self, role, crop=None, roi=None, prepared_input=None,
                          event_fields=None):
        contract = self.config["identity"][role]
        acquire_start = time.ticks_us()
        model, warm = self.pool.acquire(role)
        acquire_us = time.ticks_diff(time.ticks_us(), acquire_start)
        output_scale = float(single_tensor_metadata(model.output_scale, "output_scale"))
        output_zero = int(single_tensor_metadata(model.output_zero_point, "output_zero_point"))
        metadata = {"model_id": contract["model_id"], "model_sha256": contract["model_sha256"],
                    "input_dtype": _dtype_name(model.input_dtype), "output_dtype": _dtype_name(model.output_dtype),
                    "output_scale": output_scale, "output_zero_point": output_zero,
                    "model_warm": warm, "arena_bytes": int(model.ram),
                    "model_acquire_us": acquire_us,
                    "model_bytes": int(model.len),
                    "input_shape": list(contract["input_shape"]),
                    "input_scale": float(contract["input_scale"]),
                    "input_zero_point": int(contract["input_zero_point"])}
        start = time.ticks_us()
        model_input = prepared_input
        if model_input is None:
            def capture_identity_input(input_buffer, callback_input, shape, dtype):
                summary = self._stream_model_input("identity_input", input_buffer,
                    callback_input, shape, dtype, contract)
                if event_fields is not None:
                    event_fields.update({"input_sha256": callback_input.input_sha256,
                        "input_byte_count": callback_input.input_byte_count,
                        "input_shape": list(contract["input_shape"]),
                        "input_dtype": contract["input_dtype"],
                        "input_scale": contract["input_scale"],
                        "input_zero_point": contract["input_zero_point"],
                        "input_packet": summary,
                        "preprocess_us": callback_input.preprocess_us})
                return summary
            model_input = pillow_bilinear_model_input(
                crop, roi=roi, capture_callback=capture_identity_input,
                clock_us=time.ticks_us, clock_diff=time.ticks_diff)
        raw = model.predict([model_input])
        latency = time.ticks_diff(time.ticks_us(), start)
        self.identity_tensor_bytes = None
        metadata["input_sha256"] = model_input.input_sha256
        metadata["input_byte_count"] = int(getattr(model_input, "input_byte_count", 0))
        metadata["input_packet"] = getattr(model_input, "capture_result", None)
        metadata["input_capture_error"] = getattr(model_input, "capture_error", None)
        metadata["preprocess_us"] = getattr(model_input, "preprocess_us", None)
        metadata["input_transport_us"] = (metadata["input_packet"] or {}).get("transport_us")
        # OpenMV's default predict() path dequantizes integer output tensors.
        # Identity models have no postprocessor, so applying output_scale here
        # would quantize/dequantize a second time and corrupt the D/E scores.
        output_processing_start = time.ticks_us()
        raw_values = _flatten(raw)
        expected_width = 1
        for dimension in contract["output_shape"]:
            expected_width *= int(dimension)
        values = prediction_values(raw_values, expected_width)
        metadata["output_processing_us"] = time.ticks_diff(
            time.ticks_us(), output_processing_start)
        raw = None
        model_input = None
        gc.collect()
        metadata["inference_us"] = latency
        metadata["model_predict_inclusive_us"] = latency
        metadata["model_compute_and_dequantization_residual_us"] = max(0,
            latency - int(metadata.get("preprocess_us") or 0) -
            int(metadata.get("input_transport_us") or 0))
        metadata["output_length"] = len(values)
        metadata["output_value_representation"] = "openmv_predict_dequantized_float32"
        return values, metadata

    def _emit_identity_model_output(self, role, values, metadata):
        self.telemetry.emit("identity_model_output", {
            "boot_id": self.telemetry.run_id,
            "frame_id": self.frame_index, "model_slot": role,
            "model_id": metadata.get("model_id"),
            "model_sha256": metadata.get("model_sha256"),
            "input_sha256": metadata.get("input_sha256"),
            "input_byte_count": metadata.get("input_byte_count"),
            "input_shape": metadata.get("input_shape"),
            "input_dtype": metadata.get("input_dtype"),
            "output_dtype": metadata.get("output_dtype"),
            "output_scale": metadata.get("output_scale"),
            "output_zero_point": metadata.get("output_zero_point"),
            "output_representation": metadata.get("output_value_representation"),
            "raw_output": values})

    def _identity_decision(self, crop, face_fields):
        _reserve_identity_result_fields(face_fields)
        route = self.policy["identity_backend"]
        identity = self.config["identity"]
        band = float(identity["uncertainty_band"])
        if self.model_override == "small":
            route = "small"
        if self.model_override == "large":
            face_fields.update({"model_slot": "large",
                "model_id": identity["large"]["model_id"],
                "model_sha256": identity["large"]["model_sha256"],
                "score_semantics": "max_known_softmax_probability_from_logits"})
            try:
                logits, metadata = self._predict_identity("large", crop,
                    event_fields=face_fields)
                self._emit_identity_model_output("large", logits, metadata)
                decision = classifier_decision(logits, identity["large"]["class_order"],
                    float(identity["large"]["threshold"]), band,
                    other_label=self.config["roster"]["other_label"])
                face_fields.update(metadata)
                face_fields["identity_threshold"] = float(identity["large"]["threshold"])
                face_fields["large_backend"] = "six_logit_classifier"
                face_fields["class_order"] = identity["large"]["class_order"]
                face_fields["score_semantics"] = "max_known_softmax_probability_from_logits"
                return decision, "large", decision
            except Exception as error:
                face_fields["model_load_error"] = str(error)[:120]
                return {"status": "UNCERTAIN", "identity_id": None,
                        "reason": "large_model_error:" + str(error)[:100]}, "large", None
        if route in ("small", "small_then_large_near_threshold",
                     "small_then_large_risk_aware", "small_then_large_cost_aware"):
            face_fields.update({"model_slot": "small",
                "model_id": identity["small"]["model_id"],
                "model_sha256": identity["small"]["model_sha256"],
                "score_semantics": "cosine_similarity_of_l2_normalized_embeddings"})
            try:
                vector, metadata = self._predict_identity("small", crop,
                    event_fields=face_fields)
                # R28's resident route loads the hash-checked gallery before
                # model allocation, then keeps D resident. Consecutive device
                # runs showed that clearing D after frame N could make the
                # next load fail even with substantial aggregate free heap.
                # If the gallery is unavailable or the option is off, retain
                # the established release-before-decode fallback.
                retain_small = (self.config["models"].get(
                    "retain_small_model_after_prediction", False) and
                    self.gallery is not None)
                if not retain_small:
                    self.pool.clear("small_prediction_complete_before_gallery_decode")
                if not self.gallery_loaded:
                    self._load_gallery()
                if (self.gallery is None and self.gallery_error_type == "MemoryError" and
                        any(pair[0] == "small" for pair in self.pool.cache)):
                    self.pool.clear("small_model_release_after_gallery_memory_error")
                    self.gallery_loaded = False
                    self._load_gallery()
                self._emit_identity_model_output("small", vector, metadata)
                if self.gallery is None:
                    face_fields.update(metadata)
                    return {"status": "UNCERTAIN", "identity_id": None,
                            "reason": "gallery_invalid:" + str(self.gallery_error)}, "small", None
                # Flush short-lived packet/telemetry allocations before the
                # fixed-gallery score pass.  The gallery path uses scalar
                # template normalization to avoid large temporary lists.
                heap_before_gc = gc.mem_free()
                gc.collect()
                face_fields["gallery_score_heap_free_before_gc_bytes"] = heap_before_gc
                face_fields["gallery_score_heap_free_after_gc_bytes"] = gc.mem_free()
                decision = gallery_decision(vector, self.gallery["entries"],
                    float(identity["small"]["threshold"]), band,
                    expected_model_id=identity["small"]["model_id"],
                    acceptance_uses_calibrated_cutoff=identity["small"].get(
                        "acceptance_uses_calibrated_cutoff", False))
                face_fields.update(metadata)
                face_fields["identity_threshold"] = float(identity["small"]["threshold"])
                face_fields["identity_acceptance_cutoff"] = (face_fields["identity_threshold"] if
                    identity["small"].get("acceptance_uses_calibrated_cutoff", False) else
                    face_fields["identity_threshold"] + band)
                face_fields["small_backend"] = "64d_embedding_gallery"
                face_fields["score_semantics"] = "cosine_similarity_of_l2_normalized_embeddings"
                face_fields["gallery_loaded_with_small_model_resident"] = any(
                    pair[0] == "small" for pair in self.pool.cache)
                face_fields["gallery_retained_after_decision"] = self.gallery is not None
            except Exception as error:
                self.gallery = None
                self.gallery_loaded = False
                gc.collect()
                face_fields["model_load_error"] = str(error)[:120]
                return {"status": "UNCERTAIN", "identity_id": None,
                        "reason": "small_model_error:" + str(error)[:100]}, "small", None
            if route == "small" or self.model_override == "small":
                return decision, "small", decision
            calibration = self.config.get("controller_calibration", {})
            score = decision.get("candidate_score")
            route_details = {"policy_id": self.policy_id, "candidate_score": score}
            if route == "small_then_large_risk_aware":
                risk_threshold = calibration.get("risk_threshold")
                escalate, reason, risk = calibrated_risk_escalation(
                    calibration.get("risk_bins"), score, risk_threshold)
                route_details.update({"estimated_small_error_probability": risk,
                                      "risk_threshold": risk_threshold})
            elif route == "small_then_large_cost_aware":
                row = None
                numeric_score = finite_number(score)
                for candidate in calibration.get("risk_bins", []):
                    if not isinstance(candidate, dict) or numeric_score is None:
                        continue
                    lower = finite_number(candidate.get("score_min"))
                    upper = finite_number(candidate.get("score_max"))
                    if lower is None or upper is None or lower > upper:
                        continue
                    if lower <= numeric_score and (numeric_score < upper or
                                                   (upper == 1.0 and numeric_score <= upper)):
                        if row is not None:
                            row = None
                            break
                        row = candidate
                if row is None:
                    expected_gain = energy = latency = None
                else:
                    small_risk = finite_number(row.get("small_error_probability"))
                    large_risk = finite_number(row.get("large_error_probability"))
                    expected_gain = (small_risk - large_risk
                                     if small_risk is not None and large_risk is not None else None)
                    energy = finite_number(row.get("incremental_energy_mj"))
                    latency = finite_number(row.get("incremental_latency_ms"))
                elapsed = (time.ticks_diff(time.ticks_ms(), self.current_cycle_start_ticks)
                           if self.current_cycle_start_ticks is not None else
                           int(self.config.get("control", {}).get("max_decision_latency_ms", 5000)))
                maximum = int(self.config.get("control", {}).get("max_decision_latency_ms", 5000))
                escalate, reason = cost_aware_escalation(
                    expected_gain, energy, latency,
                    calibration.get("energy_value_mj_per_correct"),
                    max(0, maximum - elapsed))
                route_details.update({"expected_accuracy_gain": expected_gain,
                    "incremental_energy_mj": energy, "incremental_latency_ms": latency,
                    "remaining_response_budget_ms": max(0, maximum - elapsed)})
            else:
                escalate, reason = classifier_route_for_embedding(decision,
                    identity["small"]["threshold"], band)
            face_fields["escalation_reason"] = reason
            face_fields.update(route_details)
            self.telemetry.emit("escalation_route", dict(route_details, escalated=escalate,
                reason_code=reason, from_model=identity["small"]["model_id"],
                to_model=identity["large"]["model_id"]))
            if escalate:
                allowed, budget_reason = self.escalation_budget.allow(time.ticks_ms())
                face_fields["escalation_budget_reason"] = budget_reason
                face_fields["escalations_in_window"] = len(self.escalation_budget.events)
                if not allowed:
                    return {"status": "UNCERTAIN", "identity_id": None,
                            "candidate_identity_id": decision.get("candidate_identity_id"),
                            "candidate_score": decision.get("candidate_score"),
                            "reason": "ESCALATION_RATE_LIMITED"}, "small", decision
            if not escalate:
                return decision, "small", decision
            if not enough_heap(gc.mem_free(), self.config["models"]["minimum_python_heap_free_bytes"]):
                face_fields["escalation_skipped_reason"] = "low_python_heap"
                return {"status": "UNCERTAIN", "identity_id": None,
                        "candidate_identity_id": decision.get("candidate_identity_id"),
                        "candidate_score": decision.get("candidate_score"),
                        "reason": "escalation_skipped_low_memory"}, "small", decision
            self.telemetry.emit("escalation_decision", {"escalated": True,
                "reason_code": reason, "from_model": identity["small"]["model_id"],
                "to_model": identity["large"]["model_id"]})
        face_fields.update({"model_slot": "large",
            "model_id": identity["large"]["model_id"],
            "model_sha256": identity["large"]["model_sha256"],
            "score_semantics": "max_known_softmax_probability_from_logits"})
        try:
            logits, metadata = self._predict_identity("large", crop,
                event_fields=face_fields)
            self._emit_identity_model_output("large", logits, metadata)
            face_fields["raw_model_output"] = logits
            decision = classifier_decision(logits, identity["large"]["class_order"],
                float(identity["large"]["threshold"]), band,
                other_label=self.config["roster"]["other_label"])
            face_fields.update(metadata)
            face_fields["identity_threshold"] = float(identity["large"]["threshold"])
            face_fields["large_backend"] = "six_logit_classifier"
            face_fields["class_order"] = identity["large"]["class_order"]
            face_fields["score_semantics"] = "max_known_softmax_probability_from_logits"
            return decision, "large", decision
        except Exception as error:
            face_fields["model_load_error"] = str(error)[:120]
            return {"status": "UNCERTAIN", "identity_id": None,
                    "reason": "large_model_error:" + str(error)[:100]}, "large", None

    def _led(self, status):
        LED_RED.off()
        LED_GREEN.off()
        LED_BLUE.off()
        if status == "KNOWN_IDENTITY":
            LED_GREEN.on()
        elif status in ("OTHER", "MULTIPLE_FACES", "FAULT"):
            LED_RED.on()
        else:
            LED_BLUE.on()

    def _observation_summary(self, fields, compact=False):
        """Build a bounded same-frame event that fits with both models resident."""
        frame_id = fields.get("frame_index")
        boot_id = self.telemetry.run_id
        detector_status = fields.get("detector_status")
        if not detector_status:
            detector_status = "error" if fields.get("failure_reason") and fields.get("detector_backend") else "not_run"
        detector_status = {"NO_FACE": "no_face", "ONE_FACE": "face",
            "ONE_FACE_DUPLICATE_PEAKS_MERGED": "face",
            "MULTIPLE_OR_AMBIGUOUS": "multiple", "MULTIPLE_FACES": "multiple"}.get(
                detector_status, str(detector_status).lower())
        regions = []
        regions_truncated = bool(fields.get("detector_regions_truncated"))
        detector_regions = fields.get("detector_regions") or []
        detector_backend = fields.get("detector_backend")
        region_semantics = ("native Haar frontal face box" if detector_backend == "gray_haar_box" else
            "predicted face box" if detector_backend == "gray_box_regressor" else
            "FOMO heatmap region, not a measured face box")
        for index, region in enumerate(detector_regions):
            if index >= 8:
                regions_truncated = True
                break
            rect = region.get("rect_xywh_px") if isinstance(region, dict) else None
            if isinstance(rect, (list, tuple)) and len(rect) == 4:
                regions.append(rect if compact else {"rect_px": rect, "coordinate_space": "sensor_px",
                    "score": region.get("score"),
                    "semantics": region_semantics})

        identity_input = None
        if fields.get("input_sha256"):
            packet = fields.get("input_packet") or {}
            identity_input = {"boot_id": boot_id, "frame_id": frame_id,
                "shape": fields.get("input_shape"), "dtype": fields.get("input_dtype"),
                "sha256": fields.get("input_sha256"), "byte_count": fields.get("input_byte_count"),
                "payload_status": packet.get("status", "not_exported")}
            if not compact:
                identity_input.update({"scale": fields.get("input_scale"), "zero_point": fields.get("input_zero_point"),
                    "model_id": fields.get("model_id"), "model_sha256": fields.get("model_sha256"),
                    "source": "hash_measured_on_exact_predict_buffer_before_inference",
                    "input_ref": ("/api/v2/device/input?boot_id=%s&frame_id=%s&kind=identity" %
                        (boot_id, frame_id) if packet.get("status") == "sent" else None)})

        raw_detector_input = fields.get("detector_input")
        detector_input = None
        if isinstance(raw_detector_input, dict) and not compact:
            packet = fields.get("detector_input_packet") or {}
            detector_input = {"boot_id": boot_id, "frame_id": frame_id,
                "shape": raw_detector_input.get("shape"), "dtype": raw_detector_input.get("dtype"),
                "scale": raw_detector_input.get("scale"),
                "zero_point": raw_detector_input.get("zero_point"),
                "sha256": raw_detector_input.get("sha256"),
                "packet_sha256": raw_detector_input.get("packet_sha256"),
                "packet_hash_match": raw_detector_input.get("packet_hash_match"),
                "byte_count": raw_detector_input.get("byte_count"),
                "model_id": raw_detector_input.get("model_id"),
                "model_sha256": raw_detector_input.get("model_sha256"),
                "source": ("exact_" + raw_detector_input["value_encoding"] if
                    fields.get("detector_backend") == "gray_haar_box" else
                    "hash_measured_on_exact_predict_buffer_before_inference"),
                "payload_status": packet.get("status", raw_detector_input.get("payload_status", "not_exported")),
                "input_ref": ("/api/v2/device/input?boot_id=%s&frame_id=%s&kind=detector" %
                    (boot_id, frame_id) if packet.get("status") == "sent" else None)}

        crop = None
        if fields.get("crop_xywh") is not None:
            crop = {"frame_id": frame_id, "rect_px": fields.get("crop_xywh"),
                "coordinate_space": "sensor_px", "status": fields.get("crop_status"),
                "rejection_reason": fields.get("crop_rejection_reason")}
            if not compact:
                crop.update({"requested_rect_px": fields.get("crop_requested_rect_xywh"),
                    "boundary_policy": fields.get("crop_boundary_policy"),
                    "semantics": fields.get("crop_semantics"),
                    "quality_mean_u8": fields.get("crop_quality_mean_u8"),
                    "quality_stdev_u8": fields.get("crop_quality_stdev_u8")})

        frame_payload = None
        for payload in fields.get("payloads") or []:
            if payload.get("kind") == "frame_jpeg":
                frame_payload = payload
                break
        jpeg_sha = frame_payload.get("sha256") if frame_payload else None
        shape = fields.get("frame_shape") or [None, None]
        frame = {"boot_id": boot_id, "frame_id": frame_id,
            "width": shape[0], "height": shape[1], "sha256": jpeg_sha,
            "encoding": (fields.get("frame_encoding") or {}).get("encoding", "same_capture_framebuffer"),
            "pixel_format": fields.get("camera_pixel_format"),
            "orientation": fields.get("camera_orientation"),
            "frame_ref": "/api/v2/device/frame?boot_id=%s&frame_id=%s" % (boot_id, frame_id)}
        if not compact:
            frame["jpeg_sha256"] = jpeg_sha
        if jpeg_sha is None and fields.get("frame_encoding_error"):
            frame["encoding_error"] = fields["frame_encoding_error"]

        candidate = fields.get("candidate_identity_id") or fields.get("identity_id")
        candidate_score = fields.get("candidate_score")
        decision = {"frame_id": frame_id, "status": fields.get("status", "UNCERTAIN"),
            "candidate": candidate, "score": candidate_score,
            "score_semantics": fields.get("score_semantics", "unspecified score"),
            "threshold": fields.get("identity_threshold"),
            "acceptance_cutoff": fields.get("identity_acceptance_cutoff"),
            "known_acceptance_enabled": bool(self.config["identity"].get("allow_known_identity_acceptance", True)),
            "margin": fields.get("nearest_competitor_margin", fields.get("known_other_margin")),
            "reason": fields.get("decision_reason"), "experimental": True}
        if not compact:
            scores = fields.get("class_probabilities")
            if not isinstance(scores, dict):
                scores = {candidate: candidate_score} if candidate is not None and candidate_score is not None else {}
            decision.update({"raw_candidate": candidate, "scores": scores})

        timing_sources = (("capture", "capture_us"), ("preprocess", "preprocess_us"),
            ("detector_model_load", "detector_model_acquire_us"), ("detection", "detect_latency_us"),
            ("identity", "identity_pipeline_us"), ("identity_predict", "inference_us"),
            ("model_load", "model_acquire_us"), ("frame_encode", "frame_encode_us"),
            ("decision", "decision_latency_us"), ("transport", "frame_transport_us"))
        timings_ms = {}
        for name, source in timing_sources:
            value = fields.get(source)
            if value is not None:
                timings_ms[name] = value / 1000.0
        total_ms = fields.get("controller_response_latency_ms")
        if total_ms is not None:
            timings_ms["total"] = total_ms

        identity_model = fields.get("model_id")
        diagnostic = fields.get("identity_diagnostic_result")
        reason = str((diagnostic or {}).get("reason") or fields.get("decision_reason") or "")
        if (diagnostic and diagnostic.get("status") == "ERROR") or fields.get("failure_reason") or "model_error" in reason or reason.startswith("gallery_invalid"):
            identity_status = "error"
        elif fields.get("inference_us") is not None or (diagnostic and diagnostic.get("status") == "INFERENCE_ONLY"):
            identity_status = "ran"
        elif identity_input:
            identity_status = "error"
        else:
            identity_status = "skipped"
        identity = {"status": identity_status, "model_id": identity_model,
            "model_sha256": fields.get("model_sha256"), "model_warm": fields.get("model_warm"),
            "error": reason if identity_status == "error" else None}
        if not compact:
            identity.update({"model_bytes": fields.get("model_bytes"), "arena_bytes": fields.get("arena_bytes"),
                "input_sha256": fields.get("input_sha256"), "output_dtype": fields.get("output_dtype"),
                "output_scale": fields.get("output_scale"), "output_zero_point": fields.get("output_zero_point"),
                "score_semantics": fields.get("score_semantics"),
                "output_representation": "openmv_predict_dequantized_float32" if identity_input else None,
                "raw_output_ref": ({"boot_id": boot_id, "frame_id": frame_id,
                    "event": "identity_model_output", "model_id": identity_model,
                    "model_sha256": fields.get("model_sha256")} if identity_input else None)})
        if diagnostic:
            identity["diagnostic_result"] = {"status": diagnostic.get("status"),
                "reason": diagnostic.get("reason")}

        detection = {"frame_id": frame_id, "boot_id": boot_id,
            "coordinate_space": "sensor_px", "status": detector_status,
            "backend": fields.get("detector_backend", "unknown"), "regions_xywh": regions,
            "regions_truncated": regions_truncated, "threshold": fields.get("detection_threshold"),
            "region_semantics": fields.get("detector_region_semantics")}
        applied_state = {"running": self.running, "processing": self.running,
            "sleeping": self.sleeping, "camera_sleeping": self.sleeping,
            "policy_id": self.policy_id, "policy": self.policy_id,
            "model_override": self.model_override,
            "model": self.model_override or self.policy.get("identity_backend"),
            "measurement_stage_id": self.measurement_stage_id}
        if compact:
            # Keep camera, crop, model and decision linked to this JPEG, but
            # avoid building duplicate stage metadata and formatted traces.
            return {"type": "observation_complete", "source": "on_nicla",
                "frame_id": frame_id, "frame": frame,
                "camera": {"frame_shape": shape, "camera_orientation": frame["orientation"]},
                "detection": detection, "crop": crop, "identity": identity,
                "identity_input": identity_input, "decision": decision,
                "timings_ms": timings_ms, "applied_state": applied_state,
                "config_sha256": self.telemetry.config_sha256,
                "telemetry": {"heap_bytes": fields.get("python_heap_free_bytes"),
                    "arena_bytes": fields.get("identity_arena_bytes") or fields.get("detector_arena_bytes"),
                    "framebuffer_count": fields.get("framebuffer_count")},
                "record_truncated": True, "record_truncated_reason": "compact_live_telemetry",
                "failure_reason": fields.get("failure_reason")}

        detector_model = fields.get("detector_model") or {}
        detector = {"backend": fields.get("detector_backend"),
            "model_id": detector_model.get("model_id"), "model_sha256": detector_model.get("model_sha256"),
            "warm": detector_model.get("warm"), "arena_bytes": detector_model.get("arena_bytes"),
            "threshold": fields.get("detection_threshold"),
            "region_semantics": fields.get("detector_region_semantics")}
        camera = {"frame_shape": shape, "camera_resolution": fields.get("camera_resolution"),
            "camera_window_origin_px": fields.get("camera_window_origin_px"),
            "camera_sensor_window_px": fields.get("camera_sensor_window_px"),
            "camera_pixel_format": fields.get("camera_pixel_format"),
            "camera_orientation": fields.get("camera_orientation"),
            "frame_rate_hz": fields.get("frame_rate_hz"),
            "scene_mean_u8": fields.get("scene_mean_u8"), "scene_stdev_u8": fields.get("scene_stdev_u8"),
            "scene_quality_passed": fields.get("scene_quality_passed"),
            "scene_quality_reason": fields.get("scene_quality_reason")}
        telemetry = {"heap_bytes": fields.get("python_heap_free_bytes"),
            "heap_allocated_bytes": fields.get("python_heap_allocated_bytes"),
            "arena_bytes": fields.get("identity_arena_bytes") or fields.get("detector_arena_bytes"),
            "detector_arena_bytes": fields.get("detector_arena_bytes"),
            "identity_arena_bytes": fields.get("identity_arena_bytes"),
            "heap_free_before_bytes": fields.get("python_heap_free_before_bytes"),
            "heap_free_before_crop_release_bytes": fields.get("python_heap_free_before_crop_release_bytes"),
            "heap_free_after_crop_release_bytes": fields.get("python_heap_free_after_crop_release_bytes"),
            "heap_free_after_payload_bytes": fields.get("python_heap_free_after_payload_bytes"),
            "heap_free_after_payload_release_bytes": fields.get("python_heap_free_after_payload_release_bytes"),
            "framebuffer_count": fields.get("framebuffer_count"), "framebuffer_bytes": None,
            "errors": [fields.get("failure_reason")] if fields.get("failure_reason") else [],
            "dropped_frames": 0, "power_w": None, "capture_hz": None, "inference_hz": None}

        diagnostic_detector_bypass = (fields.get("measurement_stage_id") == "identity_center_crop_diagnostic" or
            fields.get("detector_status") == "NOT_RUN_DIAGNOSTIC_STAGE")
        detector_trace = ("Detector: not run (identity center-crop diagnostic stage); intentionally bypassed."
            if diagnostic_detector_bypass else "Detector: %s; %s accepted region(s)." %
            (fields.get("detector_backend", "not run"), fields.get("face_count", 0)))
        trace = ["Captured a new Nicla frame (%s:%s)." % (boot_id, frame_id), detector_trace,
            "Crop: %s%s." % (fields.get("crop_status", "not produced"),
                " — " + str(fields.get("crop_rejection_reason")) if fields.get("crop_rejection_reason") else ""),
            "Identity: %s; %s." % (identity_status, identity_model or "no model selected"),
            "Decision: %s — %s." % (decision.get("status"), decision.get("reason") or "no reason reported")]

        return {"schema_version": 2, "schema_id": "nicla-observation-v2",
            "type": "observation_complete", "source": "on_nicla",
            "boot_id": boot_id, "frame_id": frame_id,
            "command_id": self.current_observation_command_id,
            "frame": frame, "camera": camera,
            "detection": detection,
            "detector": detector, "detector_input": detector_input, "crop": crop,
            "identity": identity, "identity_input": identity_input,
            "decision": decision, "timings_ms": timings_ms,
            "telemetry": telemetry, "imu": fields.get("imu"),
            "requested_state": fields.get("requested_state"),
            "applied_state": applied_state,
            "model_sha256": fields.get("model_sha256"),
            "config_sha256": self.telemetry.config_sha256,
            "firmware_sha256": self.telemetry.firmware_sha256,
            "trace": trace, "failure_reason": fields.get("failure_reason")}

    def _cache_current_frame_jpeg(self, fields):
        if self.current_frame_jpeg_bytes is not None:
            return True
        frame = self.current_frame
        if frame is None:
            return False
        started = time.ticks_us()
        try:
            jpeg_image = frame.to_jpeg(quality=40)
            jpeg_bytes = bytes(jpeg_image.bytearray())
            jpeg_image = None
            frame_meta = {"boot_id": self.telemetry.run_id,
                "frame_id": fields.get("frame_index"),
                "width": int(frame.width()), "height": int(frame.height()),
                "pixel_format": fields.get("camera_pixel_format"),
                "orientation": {"rotation_deg": 0,
                    "hmirror": bool(self.config["camera"]["hmirror"]),
                    "vflip": bool(self.config["camera"]["vflip"])},
                "encoding": "jpeg_quality_40_from_same_capture_framebuffer"}
            self.current_frame_jpeg_bytes = jpeg_bytes
            self.current_frame_jpeg_meta = frame_meta
            fields.pop("frame_encoding_error", None)
            fields["frame_encoding"] = frame_meta
            fields["frame_encode_us"] = time.ticks_diff(time.ticks_us(), started)
            return True
        except Exception as error:
            fields["frame_encoding_error"] = type(error).__name__ + ":" + str(error)[:100]
            return False

    def _battery_available(self):
        # Exposed FAT must not have competing writers. Boot configures
        # serial-only USB before enumeration, even while on a power bank.
        if not self.config.get("control", {}).get("offline_sightings_enabled", False):
            return False
        try:
            return pyb.usb_mode() in ("VCP", "CDC")
        except (AttributeError, OSError, ValueError):
            return False

    def _battery_storage(self):
        if self.battery_store is None:
            from offline_sightings import OfflineSightings
            self.battery_store = OfflineSightings(BATTERY_PHOTO_ROOT)
        return self.battery_store

    def _battery_set_armed(self, armed):
        if not self._battery_available():
            raise ValueError("serial_only_usb_required")
        if armed and self.sleeping and not self._wake_camera("battery_arm"):
            raise ValueError("camera_wake_failed")
        if armed != self.battery_armed:
            if armed:
                os.mkdir(BATTERY_ARM_PATH)
            else:
                os.rmdir(BATTERY_ARM_PATH)
            if hasattr(os, "sync"):
                os.sync()
            self.battery_armed = armed
            self.battery_last_face = None
            self.battery_encounter_saved = False
            self.battery_error = None
        if armed:
            self.set_policy(self.config["default_policy"])
            self.measurement_stage_id = "full_pipeline"
            self.measurement_model_slot = "small"
            self.model_override = None
            self.active_trial_id = None
            self.trial_frames_remaining = None
            self.trial_deadline = None
            self.running = True
            self.next_capture = time.ticks_ms()

    def _record_battery_sighting(self, decision, fields):
        if not self.battery_armed or self.measurement_stage_id != "full_pipeline":
            return
        if self.current_observation_command_id or not self._battery_available():
            return
        now = time.ticks_ms()
        last = self.battery_last_face
        gap = time.ticks_diff(now, last) if last is not None else None
        if gap is None or gap >= BATTERY_GAP_MS:
            self.battery_encounter_saved = False
        count = fields.get("face_count")
        if type(count) is not int or count <= 0 or fields.get("failure_reason"):
            return
        self.battery_last_face = now
        if self.battery_encounter_saved or gap is None or gap > BATTERY_CONFIRM_MS:
            return
        # Reuse the exact observation JPEG, including if cached to release
        # the framebuffer before inference. One successful write per encounter.
        if not self._cache_current_frame_jpeg(fields):
            return
        try:
            self._battery_storage().save(self.current_frame_jpeg_bytes,
                boot_id=self.telemetry.run_id, frame_id=fields["frame_index"],
                ticks_ms=now, ticks_modulus_ms=self.telemetry.ticks_modulus_ms,
                face_count=count, decision={"status": decision.get("status"),
                    "candidate": decision.get("identity_id"),
                    "known_acceptance_enabled": self.config["identity"].get(
                        "allow_known_identity_acceptance", False)})
            self.battery_encounter_saved = True
            self.battery_error = None
        except (MemoryError, OSError, ValueError) as error:
            self.battery_error = type(error).__name__ + ":" + str(error)[:90]

    def _handle_battery_command(self, source, name, command):
        if source != "usb":
            self._control_ack(source, name, "rejected", {"reason_code": "usb_required"})
            return
        try:
            available = self._battery_available()
            if not available:
                if name != "battery_status":
                    raise ValueError("serial_only_usb_required")
                configured = self.config.get("control", {}).get("offline_sightings_enabled", False)
                self._control_ack(source, name, fields={"available": False, "armed": False,
                    "running": self.running,
                    "reason": "serial_only_usb_required" if configured else "battery_firmware_disabled"})
                return
            store = self._battery_storage()
            if name in ("battery_arm", "battery_disarm"):
                self._battery_set_armed(name == "battery_arm")
            if name in ("battery_status", "battery_arm", "battery_disarm"):
                fields = store.status()
                fields.update({"available": True, "armed": self.battery_armed,
                    "running": self.running, "reason": self.battery_error})
            elif name == "battery_list":
                index = command.get("index")
                if type(index) is not int or not 0 <= index < 24:
                    raise ValueError("invalid_sighting_index")
                if index == 0:
                    self.battery_list_snapshot = store.list_records()
                snapshot = self.battery_list_snapshot
                if snapshot is None:
                    raise ValueError("sighting_list_must_start_at_zero")
                record = snapshot[index] if index < len(snapshot) else None
                fields = {"index": index, "record_count": len(snapshot), "record": record}
                if index + 1 >= len(snapshot):
                    self.battery_list_snapshot = None
            else:
                record_id, offset, size = (command.get(key) for key in ("record_id", "offset", "size"))
                if (type(record_id) is not int or type(offset) is not int or
                        type(size) is not int or offset < 0 or not 1 <= size <= 512):
                    raise ValueError("invalid_sighting_chunk")
                data = store.read(record_id, offset, size)
                fields = {"record_id": record_id, "offset": offset,
                    "chunk_hex": binascii.hexlify(data).decode("ascii")}
            self._control_ack(source, name, fields=fields)
        except (ImportError, MemoryError, OSError, ValueError) as error:
            self._control_ack(source, name, "error", {"reason_code": str(error)[:120]})

    def _collect_after_source_release(self, fields, reason):
        before = gc.mem_free()
        started = time.ticks_us()
        gc.collect()
        after = gc.mem_free()
        fields.update({"source_release_reason": reason,
            "python_heap_free_before_source_release_gc_bytes": before,
            "python_heap_free_after_source_release_gc_bytes": after,
            "source_release_gc_recovered_bytes": max(0, after - before),
            "source_release_gc_us": time.ticks_diff(time.ticks_us(), started)})
        self.telemetry.emit("identity_memory_preflight", {
            "frame_id": fields.get("frame_index"), "reason": reason,
            "heap_free_before_gc_bytes": before, "heap_free_after_gc_bytes": after,
            "gc_us": fields["source_release_gc_us"],
            "resident_model_limit": self.pool.limit,
            "resident_roles": [item[0] for item in self.pool.cache]})

    def _copy_identity_crop(self, frame, box):
        roi = tuple(box)
        side = int(roi[2])
        if side != roi[3]:
            raise ValueError("identity_crop_requires_square")
        crop = self.haar_crop_input
        if crop is not None and side == 96:
            crop.draw_image(frame, 0, 0, roi=roi)
            return crop
        if side == 96:
            if self.config["detection"]["backend"] == "stock_rgb_fomo":
                return frame.to_grayscale(roi=roi, copy=True)
            return frame.copy(roi=roi)
        if crop is None:
            crop = image.Image(96, 96, image.GRAYSCALE)
        # OpenMV truncates the scaled extent to an integer. Use a small bias
        # so its float32 multiplication never leaves the last row/column blank.
        scale = 96.01 / side
        crop.draw_image(frame, 0, 0, roi=roi, x_scale=scale, y_scale=scale,
                        hint=image.BILINEAR)
        return crop

    def _run_no_face_identity_probe(self, frame, width, height, fields):
        """Run one identity model on this capture's center crop, never to accept.

        This diagnostic is only requested for an explicit one-shot. The
        detector remains authoritative: the final event is still NO_FACE, and
        the center crop is never treated as a face box or an identity match.
        """
        frame_released = False
        crop = None
        try:
            side = int(self.config["identity"]["crop"]["side_window_px"])
            crop_box, crop_rejection = crop_for_boundary_policy(
                (width / 2.0, height / 2.0), side, width, height,
                "reject_without_clamp_or_padding")
            fields.update({"crop_xywh": crop_box,
                "crop_requested_rect_xywh": crop_box,
                "crop_status": "valid_diagnostic_only" if crop_box else "invalid",
                "crop_rejection_reason": crop_rejection,
                "crop_boundary_policy": "fixed_sensor_center_no_padding",
                "crop_semantics": "fixed center diagnostic after detector NO_FACE; not a detected face box; never eligible for identity acceptance",
                "identity_acceptance_enabled": False,
                "identity_probe_requested": True})
            if crop_box is None:
                raise ValueError("center_crop_out_of_bounds")

            backend = self.policy.get("identity_backend", "small")
            slot = self.one_shot_identity_probe_slot
            if self.model_override in ("small", "large"):
                slot = self.model_override
            elif backend == "large":
                slot = "large"
            fields.update({"model_slot": slot,
                "model_id": self.config["identity"][slot].get("model_id"),
                "model_sha256": self.config["identity"][slot].get("model_sha256"),
                "identity_input_source": "detached_deep_copy_of_fixed_center_roi",
                "score_semantics": "raw model output only; no identity score or confidence computed"})
            copy_start = time.ticks_us()
            crop = self._copy_identity_crop(frame, crop_box)
            fields["identity_crop_copy_us"] = time.ticks_diff(time.ticks_us(), copy_start)
            crop_mean, crop_stdev = _finite_stats(crop)
            fields.update({"crop_quality_mean_u8": crop_mean,
                "crop_quality_stdev_u8": crop_stdev,
                "crop_quality_gate_applied": False,
                "crop_quality_reason": "measured_for_diagnostic_only"})

            detector_resident = any(role == "detector" for role, _model in self.pool.cache)
            if (self.config["models"].get("release_detector_before_identity", False) and
                    detector_resident):
                keep_role = slot if any(pair[0] == slot for pair in self.pool.cache) else None
                self.pool.clear("one_shot_no_face_identity_probe_release_detector",
                    keep_role=keep_role)
                fields["python_heap_free_after_detector_release_bytes"] = gc.mem_free()
            if self.config["models"].get("release_source_frame_before_identity", False):
                if self._cache_current_frame_jpeg(fields):
                    self.current_frame = None
                    frame_released = True
                    self._collect_after_source_release(fields,
                        "same_frame_jpeg_cached_then_full_camera_frame_released_after_diagnostic_roi_copy")
            fields["python_heap_free_before_identity_model_load_bytes"] = gc.mem_free()
            values, metadata = self._predict_identity(slot, crop, event_fields=fields)
            self._emit_identity_model_output(slot, values, metadata)
            # Keep the small identity model warm when the runtime policy asks
            # for retention. The diagnostic still cannot produce an identity
            # decision, and retaining it avoids a fragile cold reload on the
            # next explicitly requested probe.
            keep_role = (slot if slot == "small" and self.gallery is not None
                and self.config["models"].get("retain_small_model_after_prediction", False)
                else None)
            self.pool.clear("one_shot_no_face_probe_complete", keep_role=keep_role)
            fields.update(metadata)
            fields["raw_model_output"] = values
            fields["identity_acceptance_enabled"] = False
            fields["identity_diagnostic_result"] = {
                "status": "INFERENCE_ONLY",
                "reason": "detector_returned_no_face; fixed_center_crop; identity acceptance disabled",
                "identity_acceptance_enabled": False,
                "identity_model_id": metadata.get("model_id"),
                "identity_model_sha256": metadata.get("model_sha256"),
                "identity_input_sha256": metadata.get("input_sha256")}
            self.telemetry.emit("measurement_stage_result", {
                "stage_id": "one_shot_no_face_center_crop_diagnostic",
                "frame_id": self.frame_index, "model_slot": slot,
                "model_id": metadata.get("model_id"),
                "model_sha256": metadata.get("model_sha256"),
                "input_sha256": metadata.get("input_sha256"),
                "input_byte_count": metadata.get("input_byte_count"),
                "inference_us": metadata.get("inference_us"),
                "output_length": metadata.get("output_length"),
                "identity_decision_performed": False,
                "identity_acceptance_enabled": False,
                "python_heap_free_bytes": gc.mem_free()})
        except Exception as error:
            fields["identity_diagnostic_result"] = {
                "status": "ERROR", "reason": "no_face_identity_diagnostic:" + str(error)[:100],
                "identity_acceptance_enabled": False}
            fields["identity_diagnostic_skipped_reason"] = type(error).__name__ + ":" + str(error)[:100]
            if not fields.get("failure_reason"):
                fields["failure_reason"] = "no_face_identity_diagnostic:" + str(error)[:100]
        finally:
            crop = None
        return frame_released

    def _frame_payloads(self, fields):
        payloads = []
        if self.telemetry.usb_connected():
            try:
                if not self._cache_current_frame_jpeg(fields):
                    raise ValueError("captured_frame_jpeg_unavailable")
                jpeg_bytes = self.current_frame_jpeg_bytes
                frame_meta = dict(self.current_frame_jpeg_meta)
                packet = {"kind": "frame_jpeg", "metadata": frame_meta,
                          "data": jpeg_bytes, "sha256": _sha256_bytes(jpeg_bytes)}
                payloads.append(packet)
                fields["frame_encoding"] = frame_meta
            except Exception as error:
                if not fields.get("frame_encoding_error"):
                    fields["frame_encoding_error"] = type(error).__name__ + ":" + str(error)[:100]
        else:
            fields["frame_encoding_error"] = "usb_disconnected"

        # Tensor payloads were streamed from inside each Model.predict input
        # callback, before OpenMV could reuse that buffer. Only the smaller
        # same-frame JPEG remains to be sent after inference.
        summaries = []
        for key in ("detector_input_packet", "input_packet"):
            summary = fields.get(key)
            if isinstance(summary, dict) and summary.get("status") != "not_exported_live":
                summaries.append({"kind": summary.get("kind"),
                    "byte_count": summary.get("byte_count"),
                    "sha256": summary.get("sha256"),
                    "packet_sha256": summary.get("packet_sha256"),
                    "status": summary.get("status"),
                    "hash_match": summary.get("hash_match"),
                    "reason": summary.get("reason"),
                    "transport_us": summary.get("transport_us"),
                    "packet_bytes": summary.get("packet_bytes"),
                    "metadata": summary.get("metadata")})
        if payloads:
            summaries.extend({"kind": item["kind"],
                "byte_count": len(item["data"]), "sha256": item["sha256"],
                "status": "queued_for_post_decision_send",
                "metadata": item["metadata"]} for item in payloads)
        if not self.telemetry.usb_connected():
            fields["payload_transport"] = "usb_disconnected"
        elif not summaries:
            fields["payload_transport"] = "no_payloads"
        elif all(row.get("status") in ("sent", "queued_for_post_decision_send")
                 for row in summaries):
            fields["payload_transport"] = "NVPK1_usb_cdc"
        elif any(row.get("status") == "sent" for row in summaries):
            fields["payload_transport"] = "NVPK1_partial"
        else:
            fields["payload_transport"] = "NVPK1_failed"
        fields["payloads"] = summaries
        fields["pre_sent_tensor_packets"] = sum(1 for row in summaries
            if row.get("kind") in ("detector_input", "identity_input") and row.get("status") == "sent")
        fields["pre_sent_tensor_bytes"] = sum(int(row.get("byte_count") or 0) for row in summaries
            if row.get("kind") in ("detector_input", "identity_input") and row.get("status") == "sent")
        fields["pre_sent_tensor_transport_us"] = sum(int(row.get("transport_us") or 0) for row in summaries
            if row.get("kind") in ("detector_input", "identity_input") and row.get("status") == "sent")
        fields["pre_tensor_transport_us"] = sum(int(row.get("transport_us") or 0) for row in summaries
            if row.get("kind") in ("detector_input", "identity_input"))
        fields["pre_sent_tensor_packet_bytes"] = sum(int(row.get("packet_bytes") or 0) for row in summaries
            if row.get("kind") in ("detector_input", "identity_input") and row.get("status") == "sent")
        return payloads

    def _finish_decision(self, decision, backend, fields):
        cycle_start = self.current_cycle_start_ticks
        if cycle_start is not None:
            response_ms = time.ticks_diff(time.ticks_ms(), cycle_start)
            fields["controller_response_latency_ms"] = response_ms
            maximum_ms = int(self.config.get("control", {}).get("max_decision_latency_ms", 5000))
            original_status = decision.get("status")
            decision, late = response_deadline_result(decision, response_ms, maximum_ms)
            if late:
                fields["late_decision_status_suppressed"] = original_status
                self.temporal.reset()
            self.current_cycle_start_ticks = None
        decision = gate_unqualified_identity(decision,
            self.config["identity"].get("allow_known_identity_acceptance", True))
        if self.policy.get("temporal_confirmation") == "2_of_3":
            decision = self.temporal.apply(decision)
        status = decision.get("status", "UNCERTAIN")
        self.last_status = status
        reason = str(decision.get("reason", ""))
        if (reason == "DETECTOR_ERROR" or reason.startswith(("small_model_error", "large_model_error", "gallery_invalid"))
                or fields.get("failure_reason")):
            self._led("FAULT")
        else:
            self._led(status)
        fields.update({"status": status, "identity_id": decision.get("identity_id"),
            "candidate_identity_id": decision.get("candidate_identity_id"),
            "candidate_score": decision.get("candidate_score"),
            "decision_reason": decision.get("reason"), "backend": backend,
            "other_probability": decision.get("other_probability"),
            "known_other_margin": decision.get("known_other_margin"),
            "nearest_competitor_margin": decision.get("nearest_competitor_margin"),
            "class_probabilities": decision.get("class_probabilities"),
            "temporal_confirmation": decision.get("temporal_confirmation")})
        fields["frame_index"] = fields.get("frame_index", self.frame_index)
        fields["boot_id"] = self.telemetry.run_id
        fields["requested_state"] = getattr(self, "last_requested_state", None)
        fields["camera_orientation"] = {"rotation_deg": 0,
            "hmirror": bool(self.config["camera"]["hmirror"]),
            "vflip": bool(self.config["camera"]["vflip"])}
        fields["imu"] = dict(self.latest_imu,
            timestamp_limitation="sampled_before_capture_not_hardware_synced")
        fields["python_heap_free_bytes"] = gc.mem_free()
        try:
            fields["python_heap_allocated_bytes"] = gc.mem_alloc()
        except Exception:
            fields["python_heap_allocated_bytes"] = None
        fields["detector_arena_bytes"] = (fields.get("detector_model") or {}).get("arena_bytes")
        fields["identity_arena_bytes"] = fields.get("arena_bytes")
        fields["framebuffer_count"] = int(self.config["camera"]["framebuffer_count"])
        if self.current_observation_command_id:
            fields["command_id"] = self.current_observation_command_id
        self._record_battery_sighting(decision, fields)
        payloads = self._frame_payloads(fields)
        fields["python_heap_free_after_payload_bytes"] = gc.mem_free()
        if self.scene_trigger is not None:
            fields["scene_trigger_detector_calls"] = self.scene_trigger.detector_calls
            fields["scene_trigger_skipped_frames"] = self.scene_trigger.skipped_frames
        # The diagnostic raw vector already has its own frame-keyed event. Do
        # not duplicate 64 floats in both the detailed decision and compact
        # observation while the identity arena is resident on the small Nicla
        # heap. The exact input packet and input hash remain attached to this
        # same frame.
        if (fields.get("measurement_stage_id") == "identity_center_crop_diagnostic" or
                fields.get("identity_acceptance_enabled") is False):
            fields.pop("raw_model_output", None)
            fields.pop("measurement_stage_model", None)
        compact_live = fields.get("export_exact_inputs") is False and self.active_trial_id is None
        try:
            if compact_live:
                self.telemetry.emit("decision", {"frame_id": fields["frame_index"],
                    "policy_id": self.policy_id, "status": status,
                    "identity_id": fields.get("identity_id"), "decision_reason": reason,
                    "backend": backend})
            else:
                self.telemetry.emit("decision", fields)
        except MemoryError:
            # The verbose backward-compatible event is optional. Continue to
            # send the frame packet and authoritative same-frame observation.
            gc.collect()
        self.telemetry.flush()
        packet_start = time.ticks_us()
        packet_total_bytes = int(fields.get("pre_sent_tensor_packet_bytes") or 0)
        packet_successes = int(fields.get("pre_sent_tensor_packets") or 0)
        packet_count = 0
        # Include failed pre-inference tensor transfers in the attempted count.
        for key in ("detector_input_packet", "input_packet"):
            row = fields.get(key)
            if isinstance(row, dict) and row.get("status") != "not_attempted":
                packet_count += 1
        for item in payloads:
            packet_count += 1
            sent = self.telemetry.send_packet(item["kind"], item["metadata"], item["data"])
            item["transport_status"] = "sent" if sent.get("sent") else "failed"
            item["transport_result"] = sent
            if not sent.get("sent"):
                self.telemetry.emit("frame_transport_error", {
                    "frame_id": fields.get("frame_index"),
                    "kind": item["kind"], "reason_code": sent.get("reason"),
                    "bytes_written": sent.get("payload_bytes_written", 0),
                    "expected_bytes": len(item["data"]),
                    "sha256": item["sha256"]})
                self.telemetry.flush()
                break
            else:
                packet_total_bytes += int(sent.get("packet_bytes", 0))
                packet_successes += 1
        for summary in fields.get("payloads") or []:
            if summary.get("kind") == "frame_jpeg":
                item = next((candidate for candidate in payloads
                    if candidate.get("kind") == "frame_jpeg"), None)
                summary["status"] = (item or {}).get("transport_status", "failed")
                summary["transport_result"] = (item or {}).get("transport_result")
        payload_summaries = fields.get("payloads") or []
        if payload_summaries and all(row.get("status") == "sent" for row in payload_summaries):
            fields["payload_transport"] = "NVPK1_usb_cdc"
        elif any(row.get("status") == "sent" for row in payload_summaries):
            fields["payload_transport"] = "NVPK1_partial"
        elif payload_summaries:
            fields["payload_transport"] = "NVPK1_failed"
        if payloads or packet_count:
            fields["frame_transport_us"] = (int(fields.get("pre_tensor_transport_us") or 0) +
                time.ticks_diff(time.ticks_us(), packet_start))
            self.telemetry.emit("frame_transport_complete", {
                "frame_id": fields.get("frame_index"),
                "packet_count": packet_count, "packet_successes": packet_successes,
                "packet_bytes": packet_total_bytes,
                "transport_us": fields["frame_transport_us"],
                "python_heap_free_after_payload_bytes": gc.mem_free()})
        # Packet bytes have reached the host. Keep their metadata in `fields`,
        # but release the temporary frame/tensor byte buffers before building
        # the browser observation; the telemetry encoder otherwise fails on
        # this heap-constrained MCU while serializing the same-frame result.
        payloads = None
        item = None
        self.current_frame = None
        self.current_frame_jpeg_bytes = None
        self.current_frame_jpeg_meta = None
        self.detector_tensor_bytes = None
        self.identity_tensor_bytes = None
        self.detector_tensor_exported = False
        gc.collect()
        fields["python_heap_free_after_payload_release_bytes"] = gc.mem_free()
        self.telemetry.emit("observation_complete", self._observation_summary(fields, compact_live))
        self.telemetry.flush()
        command_id = self.current_observation_command_id
        if command_id:
            self._control_ack("usb", "one_shot", "completed", {
                "completed_frame_id": fields.get("frame_index"),
                "result_status": status, "result_reason": decision.get("reason")},
                command_id=command_id)
            self.telemetry.flush()
        self.current_frame = None
        self.current_frame_jpeg_bytes = None
        self.current_frame_jpeg_meta = None
        self.detector_tensor_bytes = None
        self.identity_tensor_bytes = None
        self.detector_tensor_exported = False
        self.current_observation_command_id = None
        return decision

    def _run_frame(self):
        cycle_start = time.ticks_us()
        self.current_cycle_start_ticks = time.ticks_ms()
        self.frame_index += 1
        frame_number = self.frame_index
        self.current_frame = None
        self.current_frame_jpeg_bytes = None
        self.current_frame_jpeg_meta = None
        self.detector_tensor_bytes = None
        self.identity_tensor_bytes = None
        self.current_observation_command_id = self.one_shot_command_id
        self.one_shot_command_id = None
        self.one_shot_pending = False
        identity_probe_on_no_face = (self.one_shot_identity_probe_on_no_face and
            self.current_observation_command_id is not None)
        self.one_shot_identity_probe_on_no_face = False
        self.export_exact_inputs = (not self.policy.get("compact_live_telemetry") or
            self.current_observation_command_id is not None or self.measurement_stage_id != "full_pipeline")
        window = self.config["camera"]["windows"].get(self.last_resolution, [0, 0, 240, 240])
        fields = {"frame_index": frame_number, "policy_id": self.policy_id,
                  "export_exact_inputs": self.export_exact_inputs,
                  "compact_detector_first": bool(self.policy.get("compact_detector_first")),
                  "camera_resolution": self.last_resolution, "frame_rate_hz": self.last_camera_rate,
                  "python_heap_free_before_bytes": gc.mem_free(),
                  "measurement_stage_id": self.measurement_stage_id,
                  "detector_backend": self.config["detection"]["backend"],
                  "camera_window_origin_px": [int(window[0]), int(window[1])],
                  "camera_sensor_window_px": [int(window[2]), int(window[3])],
                  "camera_pixel_format": ("RGB565" if
                      self.config["detection"]["backend"] == "stock_rgb_fomo" else "GRAYSCALE"),
                  "camera_orientation": {"rotation_deg": 0,
                      "hmirror": bool(self.config["camera"]["hmirror"]),
                      "vflip": bool(self.config["camera"]["vflip"])},
                  # Reserve the failure-path keys before model allocation so
                  # recording a bounded diagnostic cannot trigger a large
                  # MicroPython dict resize while the TFLM arena is live.
                  "model_load_error": None,
                  "identity_diagnostic_skipped_reason": None,
                  "identity_diagnostic_result": None,
                  "raw_model_output": None,
                  "gallery_score_heap_free_before_gc_bytes": None,
                  "gallery_score_heap_free_after_gc_bytes": None,
                  "requested_state": self.last_requested_state,
                  "command_id": self.current_observation_command_id,
                  "identity_probe_requested": identity_probe_on_no_face}
        self.telemetry.emit("capture_start", fields)
        capture_start = time.ticks_us()
        try:
            frame = self.camera.snapshot()
        except Exception as error:
            frame = None
            fields["capture_error"] = str(error)[:100]
        capture_us = time.ticks_diff(time.ticks_us(), capture_start)
        fields["capture_us"] = capture_us
        if frame is None:
            fields["failure_reason"] = "snapshot_returned_none"
            self.telemetry.emit("capture_end", fields)
            self.temporal.reset()
            return self._finish_decision({"status": "UNCERTAIN", "identity_id": None,
                "reason": "INVALID_FRAME"}, "none", fields)
        self.current_frame = frame
        try:
            width, height = int(frame.width()), int(frame.height())
        except Exception:
            width, height = 0, 0
        fields["frame_shape"] = [width, height]
        if width != 240 or height != 240:
            fields["failure_reason"] = "camera_window_shape_mismatch"
            self.temporal.reset()
            return self._finish_decision({"status": "UNCERTAIN", "identity_id": None,
                "reason": "INVALID_FRAME_GEOMETRY"}, "none", fields)
        mean, stdev = _finite_stats(frame)
        fields.update({"scene_mean_u8": mean, "scene_stdev_u8": stdev,
                       "scene_brightness_is_lux": False})
        fields["first_frame_after_wake"] = getattr(self, "wake_pending", False)
        if getattr(self, "wake_pending", False):
            self.wake_pending = False
            passed, reason = quality_decision(mean, stdev,
                int(self.config["identity"]["quality"]["mean_min_u8"]),
                int(self.config["identity"]["quality"]["mean_max_u8"]),
                int(self.config["identity"]["quality"]["stdev_min_u8"]))
            if not passed and self.policy.get("quality_gate", True):
                fields["failure_reason"] = "wake_first_frame_" + reason
                self.telemetry.emit("capture_end", fields)
                self.telemetry.emit("frame_quality_reject", fields)
                self.temporal.reset()
                return self._finish_decision({"status": "UNCERTAIN", "identity_id": None,
                    "reason": "WAKE_FRAME_QUALITY_REJECT"}, "none", fields)
            fields["wake_first_frame_quality_passed"] = passed
            fields["wake_first_frame_quality_reason"] = reason
            fields["wake_first_frame_quality_gate_applied"] = bool(self.policy.get("quality_gate", True))

        quality_pass, quality_reason = quality_decision(mean, stdev,
            int(self.config["identity"]["quality"]["mean_min_u8"]),
            int(self.config["identity"]["quality"]["mean_max_u8"]),
            int(self.config["identity"]["quality"]["stdev_min_u8"]))
        fields.update({"scene_quality_passed": quality_pass,
                       "scene_quality_reason": quality_reason,
                       "quality_gate_applied": bool(self.policy.get("quality_gate", True))})
        if self.policy.get("quality_gate", True) and not quality_pass:
            self.temporal.reset()
            fields["failure_reason"] = "scene_quality_" + quality_reason
            self.telemetry.emit("frame_quality_reject", fields)
            return self._finish_decision({"status": "UNCERTAIN", "identity_id": None,
                "reason": "INSUFFICIENT_SCENE_QUALITY:" + quality_reason}, "quality_gate", fields)

        if self.scene_trigger is not None:
            signature = None
            if self.scene_signature is not None:
                signature_start = time.ticks_us()
                signature = sample_gray_scene_signature(frame.bytearray(), width, height, self.scene_signature)
                fields["scene_signature_us"] = time.ticks_diff(time.ticks_us(), signature_start)
            trigger = self.scene_trigger.decide(mean, stdev, time.ticks_ms(), signature)
            fields.update({"scene_trigger_reason": trigger["reason"],
                           "scene_trigger_delta_mean_u8": trigger["delta_mean_u8"],
                           "scene_trigger_delta_stdev_u8": trigger["delta_stdev_u8"],
                           "scene_trigger_changed_cells": trigger.get("spatial_changed_cells"),
                           "scene_trigger_periodic_fallback": trigger["periodic_fallback"]})
            self.telemetry.emit("scene_trigger_check", {
                "frame_index": frame_number,
                "run_detector": trigger["run_detector"],
                "reason_code": trigger["reason"],
                "delta_mean_u8": trigger["delta_mean_u8"],
                "delta_stdev_u8": trigger["delta_stdev_u8"],
                "spatial_changed_cells": trigger.get("spatial_changed_cells"),
                "scene_signature_us": fields.get("scene_signature_us"),
                "periodic_fallback": trigger["periodic_fallback"]})
            if not trigger["run_detector"]:
                self.temporal.reset()
                return self._finish_decision({"status": "UNCERTAIN", "identity_id": None,
                    "reason": "SCENE_TRIGGER_SKIPPED_QUIET_CHECK"}, "scene_trigger", fields)
        self.telemetry.emit("capture_end", fields)
        if self.measurement_stage_id == "camera_capture_only":
            self.telemetry.emit("measurement_stage_result", {"stage_id": self.measurement_stage_id,
                "frame_index": frame_number, "capture_us": capture_us,
                "frame_shape": [width, height], "scene_mean_u8": mean,
                "scene_stdev_u8": stdev, "identity_inference_performed": False})
            return self._finish_decision({"status": "UNCERTAIN", "identity_id": None,
                "reason": "DIAGNOSTIC_STAGE_ONLY"}, "measurement_stage_camera_capture_only", fields)
        if self.measurement_stage_id == "identity_center_crop_diagnostic":
            fields["detector_status"] = "NOT_RUN_DIAGNOSTIC_STAGE"
            fields["detector_region_semantics"] = "detector intentionally bypassed for fixed-center identity diagnostic"
            side = int(self.config["identity"]["crop"]["side_window_px"])
            crop_box, crop_rejection = crop_for_boundary_policy(
                (width / 2.0, height / 2.0), side, width, height,
                "reject_without_clamp_or_padding")
            fields.update({"crop_xywh": crop_box,
                "crop_requested_rect_xywh": crop_box,
                "crop_status": "valid_diagnostic_only" if crop_box else "invalid",
                "crop_rejection_reason": crop_rejection,
                "crop_boundary_policy": "fixed_sensor_center_no_padding",
                "crop_semantics": "fixed center crop independent of detector; not a detected face box; recognition acceptance disabled"})
            try:
                if crop_box is None:
                    raise ValueError("center_crop_out_of_bounds")
                slot = self.measurement_model_slot
                contract = self.config["identity"][slot]
                fields.update({"model_slot": slot, "model_id": contract.get("model_id"),
                    "model_sha256": contract.get("model_sha256"),
                    "identity_input_source": "detached_deep_copy_of_fixed_center_roi",
                    "identity_acceptance_enabled": False})
                copy_start = time.ticks_us()
                crop = self._copy_identity_crop(frame, crop_box)
                fields["identity_crop_copy_us"] = time.ticks_diff(time.ticks_us(), copy_start)
                crop_mean, crop_stdev = _finite_stats(crop)
                fields.update({"crop_quality_mean_u8": crop_mean,
                    "crop_quality_stdev_u8": crop_stdev,
                    "crop_quality_gate_applied": False,
                    "crop_quality_reason": "measured_for_diagnostic_only"})
                detector_resident = any(role == "detector" for role, _model in self.pool.cache)
                if (self.config["models"].get("release_detector_before_identity", False) and
                        detector_resident):
                    keep_role = slot if any(pair[0] == slot for pair in self.pool.cache) else None
                    self.pool.clear("center_crop_diagnostic_release_detector",
                        keep_role=keep_role)
                    fields["python_heap_free_after_detector_release_bytes"] = gc.mem_free()
                if self.config["models"].get("release_source_frame_before_identity", False):
                    if self._cache_current_frame_jpeg(fields):
                        self.current_frame = None
                        frame = None
                        self._collect_after_source_release(fields,
                            "same_frame_jpeg_cached_then_full_camera_frame_released_after_deep_roi_copy")
                heap_before_model_load = gc.mem_free()
                fields["python_heap_free_before_identity_model_load_bytes"] = heap_before_model_load
                values, metadata = self._predict_identity(slot, crop,
                    event_fields=fields)
                if self.identity_tensor_bytes is not None:
                    self.center_crop_diagnostic_tensor_exported = True
                self._emit_identity_model_output(slot, values, metadata)
                self.pool.clear("center_crop_diagnostic_prediction_complete")
                fields.update(metadata)
                fields["raw_model_output"] = values
                fields["score_semantics"] = "raw model output only; no identity score or confidence computed"
                fields["measurement_stage_model"] = metadata
                fields["identity_diagnostic_result"] = {
                    "status": "INFERENCE_ONLY",
                    "reason": "center_crop_bypasses_detector_and_never_accepts_identity",
                    "identity_model_id": metadata.get("model_id"),
                    "identity_model_sha256": metadata.get("model_sha256"),
                    "identity_input_sha256": metadata.get("input_sha256")}
                self.telemetry.emit("measurement_stage_result", {
                    "stage_id": self.measurement_stage_id,
                    "frame_id": frame_number, "model_slot": slot,
                    "model_id": metadata.get("model_id"),
                    "model_sha256": metadata.get("model_sha256"),
                    "model_warm": metadata.get("model_warm"),
                    "input_sha256": metadata.get("input_sha256"),
                    "input_byte_count": metadata.get("input_byte_count"),
                    "inference_us": metadata.get("inference_us"),
                    "output_length": metadata.get("output_length"),
                    "identity_decision_performed": False,
                    "identity_acceptance_enabled": False,
                    "python_heap_free_bytes": gc.mem_free()})
                crop = None
                return self._finish_decision({"status": "UNCERTAIN", "identity_id": None,
                    "reason": "DIAGNOSTIC_CENTER_CROP_ONLY"},
                    "measurement_stage_center_crop_identity", fields)
            except Exception as error:
                crop = None
                fields["failure_reason"] = "identity_center_crop_diagnostic:" + str(error)[:100]
                fields["identity_diagnostic_skipped_reason"] = type(error).__name__ + ":" + str(error)[:100]
                fields["identity_diagnostic_result"] = {
                    "status": "ERROR", "reason": fields["failure_reason"],
                    "acceptance_enabled": False}
                return self._finish_decision({"status": "UNCERTAIN", "identity_id": None,
                    "reason": "DIAGNOSTIC_CENTER_CROP_ERROR"},
                    "measurement_stage_center_crop_identity", fields)
        self.telemetry.emit("detect_start", {"frame_index": frame_number,
            "detector_id": self.pool._contract("detector").get("model_id", "C-trained-gray-fomo")})
        detect_start = time.ticks_us()
        try:
            detections, face_count = self._detect_faces(frame, fields)
        except Exception as error:
            fields["failure_reason"] = str(error)[:140]
            fields["detect_latency_us"] = time.ticks_diff(time.ticks_us(), detect_start)
            self.temporal.reset()
            self.telemetry.emit("detect_failed", fields)
            self._led("FAULT")
            if identity_probe_on_no_face:
                fields["identity_probe_skipped_reason"] = "detector_error"
            return self._finish_decision({"status": "UNCERTAIN", "identity_id": None,
                "reason": "DETECTOR_ERROR"}, "detector", fields)
        self.telemetry.emit("detect_end", fields)
        if self.measurement_stage_id == "detector_only":
            self.telemetry.emit("measurement_stage_result", {"stage_id": self.measurement_stage_id,
                "frame_index": frame_number, "face_count": face_count,
                "detector_latency_us": fields.get("detect_latency_us"),
                "identity_inference_performed": False})
            return self._finish_decision({"status": "UNCERTAIN", "identity_id": None,
                "reason": "DIAGNOSTIC_STAGE_ONLY"}, "measurement_stage_detector_only", fields)
        if face_count == 0:
            fields["detector_status"] = "NO_FACE"
            self.temporal.reset()
            if identity_probe_on_no_face:
                frame_released = self._run_no_face_identity_probe(frame, width, height, fields)
                if frame_released:
                    frame = None
            elif fields.get("identity_probe_requested"):
                fields["identity_probe_skipped_reason"] = "no_probe_requested"
            result = self._finish_decision({"status": "NO_FACE", "identity_id": None,
                "reason": "no_face_detected"}, "detector_plus_identity_diagnostic" if identity_probe_on_no_face else "detector_only", fields)
            return result
        if identity_probe_on_no_face:
            fields["identity_probe_skipped_reason"] = "detector_returned_face; normal_identity_route_used"
        if self.config["models"].get("release_detector_before_identity", False):
            release_before = gc.mem_free()
            keep_role = (self.model_override if self.model_override in ("small", "large")
                else "small")
            if not any(pair[0] == keep_role for pair in self.pool.cache):
                keep_role = None
            self.pool.clear("sequential_release_detector_before_identity",
                keep_role=keep_role)
            fields["python_heap_free_after_detector_release_bytes"] = gc.mem_free()
            fields["python_heap_free_before_detector_release_bytes"] = release_before
        if face_count > 1:
            if self.haar_cascade is not None:
                self.temporal.reset()
                return self._finish_decision({"status": "MULTIPLE_FACES", "identity_id": None,
                    "reason": "multiple_native_face_boxes"}, "detector_only", fields)
            fields["detector_status"] = "MULTIPLE_FACES"
            fields["face_records"] = detections
            # FOMO heatmap clusters are regions, not face boxes. If several
            # clusters occur, retain the ambiguous final decision but still run
            # the configured identity model on the highest-scoring region with
            # valid crop geometry. This is diagnostic only and can never be
            # accepted as a known identity while the detector is ambiguous.
            crop_policy = self.config["identity"]["crop"].get(
                "boundary_policy", "reject_without_clamp_or_padding")
            eligible = []
            for region_index, region in enumerate(detections):
                box, rejection = crop_for_boundary_policy(region["center"],
                    int(self.config["identity"]["crop"]["side_window_px"]),
                    width, height, crop_policy)
                requested_box = box
                configured_rejection = rejection
                diagnostic_policy = crop_policy
                if rejection is not None and crop_policy == "reject_without_clamp_or_padding":
                    # Keep the configured production crop unchanged. For this
                    # ambiguous-only diagnostic, slide a fixed-size crop inward
                    # so the identity stage can run on exact camera pixels;
                    # the final decision remains MULTIPLE_FACES either way.
                    box, rejection = crop_for_boundary_policy(region["center"],
                        int(self.config["identity"]["crop"]["side_window_px"]),
                        width, height, "slide_inward_fixed_size_no_padding")
                    diagnostic_policy = "slide_inward_fixed_size_no_padding_diagnostic_only"
                if rejection is None:
                    eligible.append((float(region["score"]), region_index,
                        region, box, diagnostic_policy, requested_box, configured_rejection))
            if eligible:
                eligible.sort(key=lambda row: (-row[0], row[1]))
                (_score, region_index, region, crop_box, actual_crop_policy,
                    requested_crop_box, configured_rejection) = eligible[0]
                fields.update({"diagnostic_selected_region_index": region_index,
                    "diagnostic_selected_region_score": region["score"],
                    "crop_xywh": crop_box, "crop_status": "valid_diagnostic_only",
                    "crop_requested_rect_xywh": requested_crop_box,
                    "crop_rejection_reason": configured_rejection,
                    "crop_boundary_policy": actual_crop_policy,
                    "crop_configured_boundary_policy": crop_policy,
                    "crop_semantics": "fixed square centered on the highest-scoring FOMO centroid with valid geometry; FOMO region is not a face box; inward slide is diagnostic only; multiple-region result cannot be accepted"})
                try:
                    crop = self._copy_identity_crop(frame, crop_box)
                    if self.config["models"].get("release_source_frame_before_identity", False):
                        if self._cache_current_frame_jpeg(fields):
                            self.current_frame = None
                            frame = None
                            self._collect_after_source_release(fields,
                                "same_frame_jpeg_cached_then_camera_frame_released")
                    crop_mean, crop_stdev = _finite_stats(crop)
                    quality, quality_reason = quality_decision(crop_mean, crop_stdev,
                        int(self.config["identity"]["quality"]["mean_min_u8"]),
                        int(self.config["identity"]["quality"]["mean_max_u8"]),
                        int(self.config["identity"]["quality"]["stdev_min_u8"]))
                    fields.update({"crop_quality_mean_u8": crop_mean,
                        "crop_quality_stdev_u8": crop_stdev,
                        "quality_reason": quality_reason})
                    if not quality and self.policy.get("quality_gate", True):
                        fields["identity_diagnostic_skipped_reason"] = "crop_quality_gate:" + quality_reason
                        candidate = None
                        score = None
                        reason = "multiple_fomo_regions_and_diagnostic_crop_quality_rejected"
                    else:
                        probe, probe_backend, _raw = self._identity_decision(crop, fields)
                        candidate = probe.get("candidate_identity_id") or probe.get("identity_id")
                        score = probe.get("candidate_score")
                        fields["identity_diagnostic_result"] = {"status": probe.get("status"),
                            "candidate_identity_id": candidate, "candidate_score": score,
                            "reason": probe.get("reason"), "backend": probe_backend,
                            "identity_model_id": fields.get("model_id"),
                            "identity_model_sha256": fields.get("model_sha256"),
                            "identity_input_sha256": fields.get("input_sha256"),
                            "identity_score_semantics": fields.get("score_semantics")}
                        reason = "multiple_fomo_regions_identity_result_diagnostic_only"
                except Exception as error:
                    candidate = None
                    score = None
                    fields["identity_diagnostic_skipped_reason"] = type(error).__name__ + ":" + str(error)[:100]
                    reason = "multiple_fomo_regions_identity_diagnostic_error"
            else:
                candidate = None
                score = None
                fields["identity_diagnostic_skipped_reason"] = "no_fomo_region_has_an_in_bounds_crop"
                reason = "multiple_fomo_regions_no_valid_diagnostic_crop"
            identity_stage_attempted = fields.get("model_slot") in ("small", "large")
            self.temporal.reset()
            return self._finish_decision({"status": "MULTIPLE_FACES", "identity_id": None,
                "candidate_identity_id": candidate, "candidate_score": score,
                "reason": reason}, "multiple_regions_identity_diagnostic" if identity_stage_attempted else "detector_only", fields)
        face = detections[0]
        fields["detection_score"] = face["score"]
        fields["detection_rect"] = face["rect"]
        crop_policy = self.config["identity"]["crop"].get(
            "boundary_policy", "reject_without_clamp_or_padding")
        box_detector = self.config["detection"]["backend"] in ("gray_box_regressor", "gray_haar_box")
        if box_detector:
            crop_policy = "padded_face_box_square_slide_inward"
            crop_box, reason = face_box_square_crop(face["rect"], width, height,
                self.config["identity"]["crop"].get("face_box_padding_ratio", 1.5))
        else:
            crop_box, reason = crop_for_boundary_policy(
                face["center"], int(self.config["identity"]["crop"]["side_window_px"]),
                width, height, crop_policy)
        fields["crop_xywh"] = crop_box
        fields["crop_requested_rect_xywh"] = crop_box
        fields["crop_boundary_policy"] = crop_policy
        fields["crop_semantics"] = ("padded square containing the predicted face box; no landmark alignment"
            if box_detector else "fixed square centered on FOMO region centroid; no landmarks or eye alignment")
        if reason:
            if box_detector:
                fields["crop_status"] = "invalid"
                fields["crop_rejection_reason"] = reason
                self.temporal.reset()
                return self._finish_decision({"status": "UNCERTAIN", "identity_id": None,
                    "reason": "INVALID_CROP"}, "detector_only", fields)
            # A clipped FOMO centroid is not a measured face box. Preserve the
            # strict configured crop as rejected, but permit a separately
            # labeled, fixed-size inward-slid crop to exercise the identity
            # model. Because this changes the crop relative to the calibrated
            # geometry, its candidate is diagnostic and cannot be accepted.
            diagnostic_box, diagnostic_rejection = crop_for_boundary_policy(
                face["center"], int(self.config["identity"]["crop"]["side_window_px"]),
                width, height, "slide_inward_fixed_size_no_padding")
            if diagnostic_rejection is not None:
                fields["crop_status"] = "invalid"
                fields["crop_rejection_reason"] = reason
                self.temporal.reset()
                return self._finish_decision({"status": "UNCERTAIN", "identity_id": None,
                    "reason": "INVALID_CROP"}, "detector_only", fields)
            fields.update({"crop_xywh": diagnostic_box,
                "crop_requested_rect_xywh": crop_box,
                "crop_status": "valid_diagnostic_only",
                "crop_boundary_policy": "slide_inward_fixed_size_no_padding_diagnostic_only",
                "crop_configured_boundary_policy": crop_policy,
                "crop_rejection_reason": reason,
                "crop_semantics": "fixed square centered on a FOMO region centroid then slid inward to fit; FOMO region is not a face box; diagnostic only; acceptance disabled"})
            try:
                diagnostic_crop = self._copy_identity_crop(frame, diagnostic_box)
                if self.config["models"].get("release_source_frame_before_identity", False):
                    if self._cache_current_frame_jpeg(fields):
                        self.current_frame = None
                        frame = None
                        self._collect_after_source_release(fields,
                            "same_frame_jpeg_cached_then_camera_frame_released")
                crop_mean, crop_stdev = _finite_stats(diagnostic_crop)
                quality, quality_reason = quality_decision(crop_mean, crop_stdev,
                    int(self.config["identity"]["quality"]["mean_min_u8"]),
                    int(self.config["identity"]["quality"]["mean_max_u8"]),
                    int(self.config["identity"]["quality"]["stdev_min_u8"]))
                fields.update({"crop_quality_mean_u8": crop_mean,
                    "crop_quality_stdev_u8": crop_stdev,
                    "quality_reason": quality_reason})
                if not quality and self.policy.get("quality_gate", True):
                    fields["identity_diagnostic_skipped_reason"] = "crop_quality_gate:" + quality_reason
                    diagnostic_decision = {"status": "UNCERTAIN", "identity_id": None,
                        "reason": "single_fomo_region_inward_crop_quality_rejected"}
                    diagnostic_backend = "detector_only"
                else:
                    diagnostic_decision, diagnostic_backend, _raw = self._identity_decision(
                        diagnostic_crop, fields)
                    fields["identity_diagnostic_result"] = {
                        "status": diagnostic_decision.get("status"),
                        "candidate_identity_id": diagnostic_decision.get("candidate_identity_id") or diagnostic_decision.get("identity_id"),
                        "candidate_score": diagnostic_decision.get("candidate_score"),
                        "reason": diagnostic_decision.get("reason"),
                        "backend": diagnostic_backend,
                        "identity_model_id": fields.get("model_id"),
                        "identity_model_sha256": fields.get("model_sha256"),
                        "identity_input_sha256": fields.get("input_sha256"),
                        "identity_score_semantics": fields.get("score_semantics")}
                candidate = diagnostic_decision.get("candidate_identity_id") or diagnostic_decision.get("identity_id")
                score = diagnostic_decision.get("candidate_score")
                self.temporal.reset()
                return self._finish_decision({"status": "UNCERTAIN", "identity_id": None,
                    "candidate_identity_id": candidate, "candidate_score": score,
                    "reason": "single_fomo_region_inward_crop_diagnostic_only"},
                    "single_region_identity_diagnostic" if candidate else diagnostic_backend, fields)
            except Exception as error:
                fields["crop_status"] = "error"
                fields["identity_diagnostic_skipped_reason"] = type(error).__name__ + ":" + str(error)[:100]
                self.temporal.reset()
                return self._finish_decision({"status": "UNCERTAIN", "identity_id": None,
                    "reason": "single_fomo_region_diagnostic_error"}, "detector_only", fields)
        try:
            crop = self._copy_identity_crop(frame, crop_box)
            if self.config["models"].get("release_source_frame_before_identity", False):
                if self._cache_current_frame_jpeg(fields):
                    self.current_frame = None
                    frame = None
                    self._collect_after_source_release(fields,
                        "same_frame_jpeg_cached_then_camera_frame_released")
        except Exception as error:
            fields["crop_status"] = "invalid"
            fields["crop_rejection_reason"] = "crop_copy_failed:" + str(error)[:80]
            self.temporal.reset()
            return self._finish_decision({"status": "UNCERTAIN", "identity_id": None,
                "reason": "INVALID_CROP"}, "detector_only", fields)
        crop_mean, crop_stdev = _finite_stats(crop)
        quality, quality_reason = quality_decision(crop_mean, crop_stdev,
            int(self.config["identity"]["quality"]["mean_min_u8"]),
            int(self.config["identity"]["quality"]["mean_max_u8"]),
            int(self.config["identity"]["quality"]["stdev_min_u8"]))
        fields.update({"crop_status": "valid", "crop_quality_mean_u8": crop_mean,
            "crop_quality_stdev_u8": crop_stdev, "quality_reason": quality_reason,
            "crop_geometry_contract": ("floor_ceil_face_box_then_padded_square_native_bilinear_96"
                if box_detector else self.config["identity"]["crop"]["rounding"])})
        if not quality and self.policy.get("quality_gate", True):
            self.temporal.reset()
            return self._finish_decision({"status": "UNCERTAIN", "identity_id": None,
                "reason": "INSUFFICIENT_EVIDENCE:" + quality_reason}, "detector_only", fields)
        stage_id = self.measurement_stage_id
        if stage_id in ("identity_small_only", "identity_large_only", "cold_model_load",
                        "warm_inference", "model_swap_and_gc"):
            if stage_id == "model_swap_and_gc":
                predictions = []
                for slot in ("small", "large", "small"):
                    _values, metadata = self._predict_identity(slot, crop,
                        event_fields=fields)
                    predictions.append({"model_slot": slot, "model_id": metadata.get("model_id"),
                        "model_warm": metadata.get("model_warm"),
                        "inference_us": metadata.get("inference_us"),
                        "output_length": metadata.get("output_length")})
                fields["measurement_stage_predictions"] = predictions
                self.telemetry.emit("measurement_stage_result", {"stage_id": stage_id,
                    "frame_index": frame_number, "predictions": predictions,
                    "model_cache_after": [item[0] for item in self.pool.cache],
                    "python_heap_free_bytes": gc.mem_free()})
                return self._finish_decision({"status": "UNCERTAIN", "identity_id": None,
                    "reason": "DIAGNOSTIC_STAGE_ONLY"}, "measurement_stage_model_swap_and_gc", fields)
            slot = self.measurement_model_slot
            if stage_id == "identity_small_only":
                slot = "small"
            elif stage_id == "identity_large_only":
                slot = "large"
            elif stage_id == "cold_model_load":
                self.pool.clear("measurement_stage_cold_model_load")
            _values, metadata = self._predict_identity(slot, crop,
                event_fields=fields)
            fields["measurement_stage_model"] = metadata
            self.telemetry.emit("measurement_stage_result", {"stage_id": stage_id,
                "frame_index": frame_number, "model_slot": slot,
                "model_id": metadata.get("model_id"), "model_sha256": metadata.get("model_sha256"),
                "model_warm": metadata.get("model_warm"), "inference_us": metadata.get("inference_us"),
                "output_length": metadata.get("output_length"),
                "python_heap_free_bytes": gc.mem_free(), "identity_decision_performed": False})
            return self._finish_decision({"status": "UNCERTAIN", "identity_id": None,
                "reason": "DIAGNOSTIC_STAGE_ONLY"}, "measurement_stage_identity", fields)
        if self.policy.get("motion_gate", False) and self.last_motion:
            fields["motion_gate"] = "blocked_identity_inference"
            self.temporal.reset()
            return self._finish_decision({"status": "UNCERTAIN", "identity_id": None,
                "reason": "MOTION_GATE_ACTIVE"}, "detector_only", fields)
        identity_start = time.ticks_us()
        self.telemetry.emit("identity_start", {"frame_index": frame_number,
            "policy_id": self.policy_id, "route": self.policy["identity_backend"]})
        decision, backend, _raw_decision = self._identity_decision(crop, fields)
        fields["identity_pipeline_us"] = time.ticks_diff(time.ticks_us(), identity_start)
        fields["decision_latency_us"] = time.ticks_diff(time.ticks_us(), cycle_start)
        fields["python_heap_free_after_inference_bytes"] = gc.mem_free()
        fields["python_heap_free_before_crop_release_bytes"] = gc.mem_free()
        # Drop frame-local references before packaging. Haar retains its
        # startup buffer for reuse; the other backends can free their crop here.
        crop = None
        _raw_decision = None
        gc.collect()
        fields["python_heap_free_after_crop_release_bytes"] = gc.mem_free()
        self.telemetry.emit("identity_end", {"frame_index": frame_number,
            "backend": backend, "status": decision.get("status"),
            "identity_pipeline_us": fields["identity_pipeline_us"]})
        if decision.get("status") in ("KNOWN_IDENTITY", "OTHER"):
            self.last_face_ticks = time.ticks_ms()
        return self._finish_decision(decision, backend, fields)

    def _control_ack(self, source, command, status="ok", fields=None, command_id=None):
        if command_id is None:
            command_id = self.current_command_id
        if command_id and status == "ok":
            status = "completed"
        payload = {"source": source, "command": command, "control_status": status}
        if command_id:
            payload["command_id"] = command_id
        if fields:
            payload.update(fields)
        self.telemetry.emit("control_ack", payload)

    def _handle_command(self, source, command):
        name = command.get("cmd")
        if not isinstance(name, str):
            self._control_ack(source, "invalid", "error", {"reason_code": "missing_cmd"})
            return
        command_id = command.get("command_id")
        if command_id is not None:
            allowed_id = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-"
            if (not isinstance(command_id, str) or not command_id or len(command_id) > 64 or
                    any(character not in allowed_id for character in command_id)):
                self.current_command_id = None
                self._control_ack(source, name, "rejected", {"reason_code": "invalid_command_id"})
                return
            self.current_command_id = command_id
            if command_id in self.recent_command_ids:
                self._control_ack(source, name, "rejected", {
                    "reason_code": "duplicate_command_id", "duplicate": True})
                return
            self.recent_command_ids.append(command_id)
            self.recent_command_ids = self.recent_command_ids[-32:]
        self.current_command_name = name
        expected_boot = command.get("expected_boot_id")
        expected_frame = command.get("expected_frame_id")
        if expected_boot is not None and expected_boot != self.telemetry.run_id:
            self._control_ack(source, name, "rejected", {"reason_code": "stale_boot_id",
                "expected_boot_id": expected_boot, "actual_boot_id": self.telemetry.run_id})
            return
        if expected_frame is not None:
            if type(expected_frame) is not int or expected_frame < 0:
                self._control_ack(source, name, "rejected", {"reason_code": "invalid_expected_frame_id"})
                return
            if expected_frame > self.frame_index:
                self._control_ack(source, name, "rejected", {"reason_code": "future_expected_frame_id",
                    "expected_frame_id": expected_frame, "actual_frame_id": self.frame_index})
                return
            max_lag = command.get("max_frame_lag", 2)
            if type(max_lag) is not int or max_lag < 0 or max_lag > 5:
                self._control_ack(source, name, "rejected", {"reason_code": "invalid_frame_lag_limit"})
                return
            if name != "one_shot" and self.frame_index - expected_frame > max_lag:
                self._control_ack(source, name, "rejected", {"reason_code": "late_command",
                    "expected_frame_id": expected_frame, "actual_frame_id": self.frame_index})
                return
        if name in BATTERY_COMMANDS:
            self._handle_battery_command(source, name, command)
        elif name in ("identity", "discover"):
            self._control_ack(source, name, fields={"protocol": self.config["control"]["protocol"],
                "device": "Arduino Nicla Vision / OpenMV CSI", "policy_id": self.policy_id,
                "boot_id": self.telemetry.run_id, "frame_id": self.frame_index,
                "running": self.running, "sleeping": self.sleeping,
                "model_override": self.model_override,
                "config_sha256": self.telemetry.config_sha256,
                "firmware_sha256": self.telemetry.firmware_sha256,
                "uart_enabled": bool(self.config["telemetry"].get("uart_enabled", True)),
                "uart_bus": self.config["telemetry"]["uart_bus"],
                "uart_pins": self.config["telemetry"]["uart_pin_map_documented"]})
        elif name == "config":
            self._control_ack(source, name, fields={"schema_version": self.config["schema_version"],
                "policy_ids": list(self.config["policies"].keys()),
                "measurement_stage_ids": list(self.config.get("control", {}).get("measurement_stage_ids", [])),
                "default_policy": self.config["default_policy"],
                "detector_backend": self.config["detection"]["backend"],
                "small_model_id": self.config["identity"]["small"]["model_id"],
                "large_model_id": self.config["identity"]["large"]["model_id"],
                "models": {slot: {"model_id": self.config["identity"][slot]["model_id"],
                    "model_sha256": self.config["identity"][slot]["model_sha256"],
                    "backend": self.config["identity"][slot]["backend"],
                    "status": "experimental_unqualified"} for slot in ("small", "large")},
                "supported_commands": ["discover", "config", "status", "one_shot",
                    "start", "stop", "set_policy", "set_model", "sleep", "wake",
                    "preload_model", "trial"] + (list(BATTERY_COMMANDS)
                        if self.config.get("control", {}).get("offline_sightings_enabled") else []),
                "max_trial_frames": int(self.config["control"]["max_trial_frames"]),
                "max_trial_duration_ms": int(self.config["control"]["max_trial_duration_ms"]),
                "camera_sleep_supported": True,
                "class_order": self.config["identity"]["large"]["class_order"],
                "gallery_status": ("ready" if self.gallery is not None else
                    ("invalid" if self.gallery_loaded else "lazy_not_loaded")),
                "gallery_error": self.gallery_error})
        elif name == "ping":
            self._control_ack(source, name, fields={"pong": True})
        elif name == "seed":
            seed = command.get("seed_id")
            if not isinstance(seed, str) or not seed or len(seed) > 48 or any(ch not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-" for ch in seed):
                self._control_ack(source, name, "error", {"reason_code": "invalid_seed_id"})
            elif seed == self.last_seed_id:
                self._control_ack(source, name, fields={"seed_id": seed,
                    "duplicate": True})
            else:
                self.last_seed_id = seed
                self.telemetry.emit("sync_marker", {"marker": "seed", "seed_id": seed,
                    "host_marker": command.get("host_marker")})
                self._control_ack(source, name, fields={"seed_id": seed})
        elif name == "one_shot":
            if self.one_shot_pending:
                self._control_ack(source, name, "rejected", {"reason_code": "one_shot_busy"})
                return
            identity_probe = command.get("identity_probe_on_no_face", False)
            probe_slot = command.get("probe_model_slot", "small")
            if type(identity_probe) is not bool:
                self._control_ack(source, name, "rejected", {
                    "reason_code": "invalid_identity_probe_flag"})
                return
            if identity_probe and probe_slot not in ("small", "large"):
                self._control_ack(source, name, "rejected", {
                    "reason_code": "unsupported_probe_model_slot"})
                return
            if self.sleeping and not self._wake_camera("one_shot_command"):
                self._control_ack(source, name, "error", {"reason_code": "camera_wake_failed"})
                return
            selected_slot = (self.model_override if self.model_override in ("small", "large")
                else probe_slot if identity_probe else "small")
            preload_status = (self._preload_small_for_capture("one_shot")
                if selected_slot == "small" else "not_requested_for_selected_model")
            self.one_shot_pending = True
            self.one_shot_command_id = command_id
            self.one_shot_identity_probe_on_no_face = identity_probe
            self.one_shot_identity_probe_slot = probe_slot if identity_probe else "small"
            self.measurement_stage_id = "full_pipeline"
            self.next_capture = time.ticks_ms()
            self.last_requested_state = {"command": name, "command_id": command_id,
                "policy_id": self.policy_id, "model_override": self.model_override,
                "identity_probe_on_no_face": identity_probe,
                "probe_model_slot": self.one_shot_identity_probe_slot if identity_probe else None}
            self._control_ack(source, name, "accepted", {
                "pending_frame_id": self.frame_index + 1,
                "running_before_request": self.running,
                "identity_probe_on_no_face": identity_probe,
                "probe_model_slot": self.one_shot_identity_probe_slot if identity_probe else None,
                "identity_preload_status": preload_status,
                "sleeping": self.sleeping})
        elif name == "set_policy":
            requested = command.get("policy_id")
            if not isinstance(requested, str) or requested not in self.config["policies"]:
                self._control_ack(source, name, "rejected", {"reason_code": "unsupported_policy"})
                return
            try:
                self.set_policy(requested)
                self.last_requested_state = {"command": name, "command_id": command_id,
                    "policy_id": requested, "model_override": self.model_override}
                self._control_ack(source, name, fields={"policy_id": self.policy_id,
                    "running": self.running, "sleeping": self.sleeping,
                    "applied_state": {"policy_id": self.policy_id,
                        "running": self.running, "sleeping": self.sleeping}})
            except Exception as error:
                self._control_ack(source, name, "error", {"reason_code": str(error)[:120]})
        elif name == "set_model":
            requested = command.get("model_slot")
            if requested not in ("policy", "small", "large"):
                self._control_ack(source, name, "rejected", {"reason_code": "unsupported_model_slot"})
                return
            self.model_override = None if requested == "policy" else requested
            model_id = (self.config["identity"][requested]["model_id"]
                        if requested in ("small", "large") else None)
            self.last_requested_state = {"command": name, "command_id": command_id,
                "policy_id": self.policy_id, "model_slot": requested}
            self._control_ack(source, name, fields={"model_slot": requested,
                "model_override": self.model_override, "model_id": model_id,
                "model_status": "experimental_unqualified" if model_id else "policy_route",
                "applied_state": {"model_override": self.model_override,
                    "policy_id": self.policy_id}})
        elif name == "sleep":
            self.last_requested_state = {"command": name, "command_id": command_id,
                "sleeping": True, "policy_id": self.policy_id}
            if not self.sleeping:
                self._sleep_camera("control_command", force=True)
            if self.sleeping:
                self._control_ack(source, name, fields={"sleeping": True,
                    "applied_state": {"sleeping": True, "running": self.running}})
            else:
                self._control_ack(source, name, "error", {"reason_code": "camera_sleep_failed"})
        elif name == "wake":
            self.last_requested_state = {"command": name, "command_id": command_id,
                "sleeping": False, "policy_id": self.policy_id}
            woke = True if not self.sleeping else self._wake_camera("control_command")
            if woke:
                self._control_ack(source, name, fields={"sleeping": self.sleeping,
                    "settle_ms": int(self.config["camera"]["exposure_settle_ms"]),
                    "applied_state": {"sleeping": self.sleeping, "running": self.running}})
            else:
                self._control_ack(source, name, "error", {"reason_code": "camera_wake_failed"})
        elif name == "start":
            requested = command.get("policy_id", self.policy_id)
            stage_id = command.get("measurement_stage_id", "full_pipeline")
            model_slot = command.get("model_slot", "small")
            allowed_stages = self.config.get("control", {}).get("measurement_stage_ids", [])
            if stage_id not in allowed_stages:
                self._control_ack(source, name, "error", {"reason_code": "unsupported_measurement_stage"})
                return
            if model_slot not in ("small", "large"):
                self._control_ack(source, name, "error", {"reason_code": "unsupported_model_slot"})
                return
            try:
                self.set_policy(requested)
                selected_slot = (self.model_override if self.model_override in ("small", "large")
                    else model_slot)
                preload_status = (self._preload_small_for_capture("start_loop")
                    if stage_id == "full_pipeline" and selected_slot == "small"
                    else "not_requested_for_selected_model_or_stage")
                self.measurement_stage_id = stage_id
                self.measurement_model_slot = model_slot
                self.running = True
                self.last_requested_state = {"command": name, "command_id": command_id,
                    "policy_id": requested, "model_slot": self.model_override or model_slot,
                    "measurement_stage_id": stage_id, "running": True}
                self.trial_frames_remaining = None
                self.trial_deadline = None
                self.active_trial_id = None
                if stage_id == "idle" and not self.sleeping:
                    self._sleep_camera("measurement_stage_idle")
                elif self.sleeping:
                    if not self._wake_camera("control_start"):
                        self._control_ack(source, name, "error", {"reason_code": "camera_wake_failed"})
                        return
                self.next_capture = time.ticks_ms()
                self._control_ack(source, name, fields={"running": True, "policy_id": self.policy_id,
                    "measurement_stage_id": self.measurement_stage_id,
                    "model_slot": self.measurement_model_slot,
                    "trial_active": False,
                    "model_override": self.model_override,
                    "identity_preload_status": preload_status,
                    "applied_state": {"running": True, "sleeping": self.sleeping,
                        "policy_id": self.policy_id, "model_override": self.model_override}})
            except Exception as error:
                self._control_ack(source, name, "error", {"reason_code": str(error)[:120]})
        elif name == "preload_model":
            if self.running:
                self._control_ack(source, name, "error", {"reason_code": "stop_before_preload"})
                return
            model_slot = command.get("model_slot")
            if model_slot not in ("small", "large"):
                self._control_ack(source, name, "error", {"reason_code": "unsupported_model_slot"})
                return
            try:
                _model, warm = self.pool.acquire(model_slot)
                self._control_ack(source, name, fields={"model_slot": model_slot,
                    "model_id": self.config["identity"][model_slot]["model_id"],
                    "warm": warm, "cache_count": len(self.pool.cache),
                    "python_heap_free_bytes": gc.mem_free()})
            except Exception as error:
                self._control_ack(source, name, "error", {"reason_code": str(error)[:120]})
        elif name == "stop":
            if self.one_shot_pending and self.one_shot_command_id:
                self._control_ack(source, "one_shot", "rejected", {
                    "reason_code": "cancelled_by_stop"},
                    command_id=self.one_shot_command_id)
                self.one_shot_command_id = None
                self.one_shot_pending = False
                self.one_shot_identity_probe_on_no_face = False
            if self.active_trial_id is not None:
                self.telemetry.emit("trial_finished", {"reason_code": "stopped",
                    "trial_id": self.active_trial_id,
                    "stage_id": self.measurement_stage_id,
                    "frame_index": self.frame_index, "running": False})
            self.running = False
            self.trial_frames_remaining = None
            self.trial_deadline = None
            self.active_trial_id = None
            resident_small = self.startup_detector_preload.get("early_small_model_sha256") is not None
            self.pool.clear("control_stop", keep_role="small" if resident_small else None)
            if not resident_small:
                self.gallery = None
                self.gallery_error = None
                self.gallery_error_type = None
                self.gallery_loaded = False
            gc.collect()
            self.last_requested_state = {"command": name, "command_id": command_id,
                "running": False, "policy_id": self.policy_id}
            self._control_ack(source, name, fields={"running": False,
                "trial_active": False,
                "applied_state": {"running": False, "sleeping": self.sleeping,
                    "policy_id": self.policy_id}})
        elif name == "status":
            self._control_ack(source, name, fields={"running": self.running,
                "sleeping": self.sleeping, "policy_id": self.policy_id,
                "measurement_stage_id": self.measurement_stage_id,
                "model_slot": self.measurement_model_slot,
                "trial_active": self.active_trial_id is not None or
                    self.trial_frames_remaining is not None or self.trial_deadline is not None,
                "frame_index": self.frame_index, "last_status": self.last_status,
                "model_override": self.model_override,
                "model_id": (self.config["identity"][self.model_override]["model_id"]
                    if self.model_override in ("small", "large") else None),
                "reset_cause": self.telemetry.reset_cause,
                "config_sha256": self.telemetry.config_sha256,
                "firmware_sha256": self.telemetry.firmware_sha256,
                "model_cache": [item[0] for item in self.pool.cache],
                "startup_detector_preload": self.startup_detector_preload,
                "gallery_status": ("ready" if self.gallery is not None else
                    ("invalid" if self.gallery_loaded else "lazy_not_loaded")),
                "uart_queue_bytes": self.telemetry.uart_queue.queued_bytes,
                "usb_queue_bytes": self.telemetry.usb_queue.queued_bytes})
        elif name == "trial":
            frames = command.get("frames")
            duration_ms = command.get("duration_ms")
            policy_id = command.get("policy_id", self.policy_id)
            trial_id = command.get("trial_id")
            stage_id = command.get("measurement_stage_id", self.measurement_stage_id)
            model_slot = command.get("model_slot", self.measurement_model_slot)
            if type(frames) is not int or frames < 1 or frames > int(self.config["control"]["max_trial_frames"]):
                self._control_ack(source, name, "error", {"reason_code": "trial_frames_out_of_range"})
                return
            if type(duration_ms) is not int or duration_ms < 1 or duration_ms > int(self.config["control"]["max_trial_duration_ms"]):
                self._control_ack(source, name, "error", {"reason_code": "trial_duration_out_of_range"})
                return
            if not isinstance(trial_id, str) or not trial_id or len(trial_id) > 48 or any(ch not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-" for ch in trial_id):
                self._control_ack(source, name, "error", {"reason_code": "invalid_trial_id"})
                return
            allowed_stages = self.config.get("control", {}).get("measurement_stage_ids", [])
            if stage_id not in allowed_stages:
                self._control_ack(source, name, "error", {"reason_code": "unsupported_measurement_stage"})
                return
            if model_slot not in ("small", "large"):
                self._control_ack(source, name, "error", {"reason_code": "unsupported_model_slot"})
                return
            if trial_id and trial_id == self.last_trial_id:
                self._control_ack(source, name, "rejected", {"reason_code": "duplicate_trial_id",
                    "trial_id": trial_id})
                return
            try:
                self.set_policy(policy_id)
                selected_slot = self.model_override if self.model_override in ("small", "large") else model_slot
                preload_status = (self._preload_small_for_capture("trial")
                    if stage_id == "full_pipeline" and selected_slot == "small"
                    else "not_requested_for_selected_model_or_stage")
                self.measurement_stage_id = stage_id
                self.measurement_model_slot = model_slot
                if stage_id == "idle":
                    if not self.sleeping:
                        self._sleep_camera("measurement_stage_idle")
                elif self.sleeping and not self._wake_camera("measurement_trial"):
                    self._control_ack(source, name, "error", {"reason_code": "camera_wake_failed"})
                    return
                self.running = True
                self.trial_frames_remaining = frames
                self.trial_deadline = time.ticks_add(time.ticks_ms(), duration_ms)
                self.last_trial_id = trial_id
                self.active_trial_id = trial_id
                self.next_capture = time.ticks_ms()
                self._control_ack(source, name, fields={"accepted_frames": frames,
                    "deadline_ms": duration_ms, "policy_id": policy_id,
                    "measurement_stage_id": stage_id, "model_slot": model_slot,
                    "trial_id": trial_id, "trial_active": True, "running": True,
                    "identity_preload_status": preload_status,
                    "applied_state": {"running": True, "sleeping": self.sleeping,
                        "policy_id": self.policy_id, "measurement_stage_id": stage_id,
                        "model_slot": model_slot}})
            except Exception as error:
                self._control_ack(source, name, "error", {"reason_code": str(error)[:120]})
        else:
            self._control_ack(source, name, "unsupported", {"reason_code": "unsupported_command"})

    def _poll_control(self):
        for source, command in self.control.poll():
            self.current_command_id = None
            self.current_command_name = None
            self.telemetry.emit("control_rx", {"source": source,
                "command": str(command.get("cmd", ""))[:40]})
            self._handle_command(source, command)
            self.current_command_id = None
            self.current_command_name = None
        if self.control.errors["usb"] or self.control.errors["uart"]:
            self.telemetry.emit("control_input_error", {"usb_errors": self.control.errors["usb"],
                "uart_errors": self.control.errors["uart"]})
            self.control.errors["usb"] = 0
            self.control.errors["uart"] = 0
        now = time.ticks_ms()
        last = getattr(self, "last_control_health_ticks", None)
        if last is None or time.ticks_diff(now, last) >= 5000:
            usb_connected = False
            try:
                usb_connected = self.usb is not None and self.usb.isconnected()
            except Exception:
                pass
            self.telemetry.emit("control_input_status", {
                "usb_connected": bool(usb_connected),
                "usb_bytes_read_total": self.control.bytes_read_total["usb"],
                "uart_bytes_read_total": self.control.bytes_read_total["uart"],
                "usb_parse_errors": self.control.errors["usb"],
                "uart_parse_errors": self.control.errors["uart"],
                "usb_queue_dropped_records": self.telemetry.usb_queue.dropped_records,
                "usb_queue_dropped_oversize": self.telemetry.usb_queue.dropped_oversize,
                "uart_queue_dropped_records": self.telemetry.uart_queue.dropped_records})
            self.last_control_health_ticks = now

    def _feed_watchdog(self):
        if getattr(self, "watchdog", None) is not None:
            try:
                self.watchdog.feed()
            except Exception:
                pass

    def run(self):
        self.watchdog = None
        try:
            self.watchdog = WDT(timeout=int(self.config["watchdog_timeout_ms"]))
        except Exception as error:
            self.telemetry.emit("watchdog_unavailable", {"reason_code": str(error)[:100]})
        self._configure_camera_actions()
        settle = int(self.config["camera"]["exposure_settle_ms"])
        self.telemetry.emit("camera_start", {"camera_sleep_supported": True,
            "frame_drop_and_sleep_are_distinct": True, "settle_ms": settle})
        self.camera.snapshot(time=settle)
        self.wake_pending = True
        if self.battery_armed:
            self.running = True
            self.next_capture = time.ticks_ms()
        self.telemetry.emit("run_start", {"policy_id": self.policy_id,
            "camera_resolution": self.last_resolution, "window_shape": [240, 240],
            "pixel_format": self.config["detection"]["backend"],
            "running_without_host": True, "model_cache_limit": self.pool.limit})
        while True:
            self._feed_watchdog()
            self._poll_control()
            self.telemetry.flush()
            if self.trial_deadline is not None and time.ticks_diff(time.ticks_ms(), self.trial_deadline) >= 0:
                self.running = False
                self.trial_frames_remaining = None
                self.trial_deadline = None
                self.telemetry.emit("trial_finished", {"reason_code": "duration_limit",
                    "trial_id": self.active_trial_id, "stage_id": self.measurement_stage_id,
                    "frame_index": self.frame_index, "running": False})
                self.active_trial_id = None
            if not self.running and not self.one_shot_pending:
                pyb.delay(20)
                continue
            motion, motion_reason, acceleration, angular = self._motion()
            self.latest_imu = {"acceleration_mg": acceleration,
                "angular_rate_mdps": angular, "motion_detected": bool(motion),
                "motion_reason": motion_reason, "sampled_ticks_ms": int(time.ticks_ms())}
            if motion != self.last_motion:
                self.last_motion = motion
                self._configure_camera_actions()
                self.telemetry.emit("motion_state", {"moving": motion,
                    "reason_code": motion_reason, "acceleration_mg": acceleration,
                    "angular_rate_mdps": angular,
                    "motion_scope": "wearer_or_device_only"})
            interval = self._interval_ms(motion)
            if self.measurement_stage_id == "idle":
                if time.ticks_diff(time.ticks_ms(), self.next_capture) >= 0:
                    self.telemetry.emit("measurement_stage_idle", {
                        "stage_id": self.measurement_stage_id,
                        "policy_id": self.policy_id,
                        "python_heap_free_bytes": gc.mem_free(),
                        "motion_detected": bool(motion),
                        "camera_sleeping": bool(self.sleeping),
                        "camera_snapshot_performed": False,
                        "identity_inference_performed": False})
                    self.next_capture = time.ticks_add(time.ticks_ms(), interval)
                    self.telemetry.flush()
                else:
                    pyb.delay(10)
                continue
            if self.sleeping:
                periodic = int(self.config["camera"]["stationary_periodic_check_ms"])
                retry_ready = (self.wake_retry_after is None or
                               time.ticks_diff(time.ticks_ms(), self.wake_retry_after) >= 0)
                if retry_ready and (motion or time.ticks_diff(time.ticks_ms(), self.next_capture) >= 0):
                    if not self._wake_camera("motion" if motion else "periodic_stationary_scene_check"):
                        self.telemetry.flush()
                        pyb.delay(20)
                        continue
                else:
                    pyb.delay(20)
                    continue
            if time.ticks_diff(time.ticks_ms(), self.next_capture) < 0:
                pyb.delay(10)
                continue
            self._configure_camera_actions()
            pace_start = time.ticks_ms() if self.policy.get("pace_from_capture_start") else None
            try:
                self._run_frame()
            except Exception as error:
                self.last_status = "FAULT"
                self._led("FAULT")
                self.temporal.reset()
                failed_command_id = self.current_observation_command_id
                gc.collect()
                self.telemetry.emit("runtime_error", {"frame_index": self.frame_index,
                    "boot_id": self.telemetry.run_id,
                    "command_id": failed_command_id,
                    "reason_code": type(error).__name__, "message": str(error)[:160]})
                if failed_command_id:
                    self._control_ack("usb", "one_shot", "error", {
                        "reason_code": "frame_runtime_error",
                        "frame_id": self.frame_index,
                        "message": str(error)[:100]}, command_id=failed_command_id)
                    self.current_observation_command_id = None
                    self.current_cycle_start_ticks = None
                    self.one_shot_pending = False
                    self.one_shot_command_id = None
            self.telemetry.flush()
            # Rebase on each actual frame start so an overrun cannot queue catch-up captures.
            pace_base = time.ticks_ms() if pace_start is None else pace_start
            self.next_capture = time.ticks_add(pace_base, interval)
            gc_heap_before = gc.mem_free()
            gc_start = time.ticks_us()
            gc.collect()
            gc_us = time.ticks_diff(time.ticks_us(), gc_start)
            gc_heap_after = gc.mem_free()
            self.telemetry.emit("gc_collect", {"reason": "after_frame",
                "duration_us": gc_us, "heap_free_before_bytes": gc_heap_before,
                "heap_free_after_bytes": gc_heap_after})
            self.telemetry.flush()
            if self.trial_frames_remaining is not None:
                self.trial_frames_remaining -= 1
                if self.trial_frames_remaining <= 0:
                    self.running = False
                    self.trial_frames_remaining = None
                    self.trial_deadline = None
                    self.telemetry.emit("trial_finished", {"reason_code": "frame_limit",
                        "trial_id": self.active_trial_id, "stage_id": self.measurement_stage_id,
                        "frame_index": self.frame_index, "running": False})
                    self.active_trial_id = None
            if self.policy.get("camera_sleep", False):
                idle_ms = time.ticks_diff(time.ticks_ms(), self.last_face_ticks)
                if idle_ms >= int(self.config["camera"]["camera_sleep_idle_ms"]):
                    self._sleep_camera("stationary_idle_scene")
                    self.next_capture = time.ticks_add(time.ticks_ms(),
                        int(self.config["camera"]["stationary_periodic_check_ms"]))


def _prepare():
    global BOOT_PRELOADED_HAAR_WIDE
    config = _read_json(CONFIG_PATH)
    manifest = _read_json(MANIFEST_PATH)
    if config.get("schema_version") != "wearable-runtime-config-1":
        raise ValueError("unsupported_runtime_config_schema")
    if config.get("control", {}).get("protocol") != "nicla-face-control-2":
        raise ValueError("unsupported_control_protocol")
    config_sha = _sha256_file(CONFIG_PATH)
    if not config_hash_matches(config_sha, manifest.get("runtime_config", {}).get("sha256")):
        raise ValueError("runtime_config_hash_mismatch")
    if config.get("identity", {}).get("large", {}).get("class_order") != KNOWN + ["OTHER"]:
        raise ValueError("classifier_label_order_mismatch")
    labels = manifest.get("labels", {})
    if labels.get("class_order") != KNOWN + ["OTHER"]:
        raise ValueError("classifier_manifest_label_order_mismatch")
    for label_role, path_key, hash_key in (("detector", "detector_path", "detector_sha256"),
                                            ("classifier", "classifier_path", "classifier_sha256")):
        label_path = labels.get(path_key)
        if not label_path or not hash_matches(_sha256_file(label_path), labels.get(hash_key)):
            raise ValueError(label_role + "_labels_hash_mismatch")
    calibration = manifest.get("D_gallery_calibration", {})
    calibration_path = calibration.get("path")
    if not calibration_path or not hash_matches(_sha256_file(calibration_path), calibration.get("sha256")):
        raise ValueError("D_gallery_calibration_hash_mismatch")
    if calibration.get("sha256") != config["identity"]["small"].get("threshold_source_sha256"):
        raise ValueError("D_gallery_threshold_source_mismatch")
    calibration_doc = _read_json(calibration_path)
    if (calibration_doc.get("threshold") != config["identity"]["small"]["threshold"] or
            calibration_doc.get("prototype_sha256") != config["identity"]["small"]["prototype_sha256"]):
        raise ValueError("D_gallery_calibration_contract_mismatch")
    calibration_doc = None
    if _read_json(config["identity"]["small"]["gallery_path"]).get("calibration", {}).get("threshold") != config["identity"]["small"]["threshold"]:
        raise ValueError("gallery_calibration_mismatch")
    if config.get("identity", {}).get("crop", {}).get("rounding") != "floor(center-side/2+0.5)":
        raise ValueError("crop_contract_mismatch")
    firmware_sha = _firmware_hash(manifest=manifest)
    schema_sha = _sha256_bytes(b"nicla-face-telemetry-1|nicla-observation-v2|nicla-packet-v1|sequence|run_id|ticks_ms|model_config_sha256|firmware_sha256|schema_sha256")
    try:
        if config["telemetry"].get("uart_enabled", True):
            uart = UART(int(config["telemetry"]["uart_bus"]),
                        baudrate=int(config["telemetry"]["uart_baud"]), timeout=0, timeout_char=0)
        else:
            uart = None
    except Exception:
        uart = None
    try:
        usb = pyb.USB_VCP()
    except Exception:
        usb = None
    reset_cause = _reset_cause()
    try:
        boot_nonce = int(pyb.rng()) & 0xffffffff
    except Exception:
        # Preserve startup if a different supported build omits the RNG.
        boot_nonce = int(time.ticks_us()) & 0xffffffff
    run_id = "boot-%08x-%08x-%08x" % (int(time.ticks_ms()),
        int(time.ticks_us()) & 0xffffffff, boot_nonce)
    ticks_modulus = time.ticks_add(0, -1) + 1
    telemetry = Telemetry(usb, uart, config, time.ticks_ms, run_id, reset_cause,
                          firmware_sha, config_sha, schema_sha, ticks_modulus)
    telemetry.emit("boot", {"protocol": config["control"]["protocol"],
        "reset_cause": reset_cause, "uart_available": uart is not None,
        "usb_connected": bool(usb is not None and usb.isconnected()),
        "model_cache_strategy": config["models"]["cache_strategy"],
        "model_cache_limit": config["models"]["max_resident_models"]})
    camera = csi.CSI()
    camera.reset()
    if config["detection"]["backend"] in ("trained_gray_fomo", "gray_box_regressor", "gray_haar_box"):
        camera.pixformat(csi.GRAYSCALE)
    elif config["detection"]["backend"] == "stock_rgb_fomo":
        camera.pixformat(csi.RGB565)
    else:
        raise ValueError("unsupported_detection_backend")
    camera.auto_rotation(False)
    camera.hmirror(bool(config["camera"]["hmirror"]))
    camera.vflip(bool(config["camera"]["vflip"]))
    runtime = NiclaRuntime(config, manifest, camera, usb, uart, telemetry)
    if config["detection"]["backend"] == "gray_haar_box":
        contract = config["detection"]
        path = contract["cascade_path"]
        if (path != "/rom/haarcascade_frontalface.cascade" or
                int(os.stat(path)[6]) != int(contract["cascade_bytes"]) or
                not hash_matches(_sha256_file(path), contract["cascade_sha256"])):
            raise ValueError("native_haar_rom_cascade_mismatch")
        runtime.haar_cascade = image.HaarCascade(path)
        # Reserve contiguous image storage before camera capture or identity
        # model loading. The detector fallback and crop run sequentially.
        gc.collect()
        if contract.get("wide_contrast_gain"):
            runtime.haar_wide_input = BOOT_PRELOADED_HAAR_WIDE
            BOOT_PRELOADED_HAAR_WIDE = None
            if runtime.haar_wide_input is None:
                runtime.haar_wide_input = image.Image(240, 240, image.GRAYSCALE)
            # This view borrows the start of the aligned wide buffer. The
            # wide detector completes and sends its pixels before the narrow
            # view is written. The original camera framebuffer is separate.
            runtime.haar_crop_input = image.Image(96, 96, image.GRAYSCALE,
                buffer=runtime.haar_wide_input.bytearray())
        else:
            runtime.haar_crop_input = image.Image(96, 96, image.GRAYSCALE)
        runtime._preload_small_for_capture("haar_startup")
        runtime.startup_detector_preload.update({"status": "loaded", "model_id": contract["model_id"],
            "model_sha256": contract["cascade_sha256"], "model_bytes": contract["cascade_bytes"],
            "arena_bytes": None, "backend": "gray_haar_box"})
        runtime.telemetry.emit("detector_preloaded_before_camera_stream", runtime.startup_detector_preload)
        return runtime
    # A compatible image can carry the already-reserved detector arena from
    # boot.py, which loaded it before importing this runtime. Validate that
    # object now, then keep the ordinary post-import preload as a fallback.
    preload_start = time.ticks_us()
    try:
        detector, warm = runtime.pool.acquire("detector")
        metadata = runtime.pool._check_metadata("detector", detector)
        preloaded_early = bool(warm and BOOT_DETECTOR_PRELOAD_STATUS.get("status") == "loaded")
        detector_contract = runtime.pool._contract("detector")
        runtime.startup_detector_preload.update({"status": "loaded",
            "model_id": detector_contract.get("model_id", "C-trained-gray-fomo"),
            "model_sha256": runtime.pool.actual_model_sha256("detector"),
            "model_bytes": int(detector.len), "arena_bytes": int(detector.ram),
            "warm": bool(warm), "heap_free_bytes": gc.mem_free(),
            "preloaded_before_runtime_import": preloaded_early,
            "metadata": metadata,
            "load_plus_validation_us": time.ticks_diff(time.ticks_us(), preload_start)})
        runtime.telemetry.emit("detector_preloaded_before_camera_stream", dict(
            runtime.startup_detector_preload,
            reason=("verified_arena_preloaded_before_runtime_import" if preloaded_early else
                "reserve_the_detector_arena_before_camera_framebuffer_capture")))
    except Exception as error:
        runtime.startup_detector_preload = {"status": "failed",
            "reason_code": type(error).__name__ + ":" + str(error)[:120],
            "heap_free_bytes": gc.mem_free(),
            "elapsed_us": time.ticks_diff(time.ticks_us(), preload_start)}
        runtime.telemetry.emit("detector_preload_before_camera_stream_failed",
            runtime.startup_detector_preload)
        runtime.pool.clear("detector_preload_before_camera_stream_failed")
    return runtime


def main():
    LED_BLUE.on()
    time.sleep_ms(80)
    LED_BLUE.off()
    try:
        runtime = _prepare()
        runtime.run()
    except KeyboardInterrupt:
        LED_RED.off()
        LED_GREEN.off()
        LED_BLUE.off()
        raise
    except Exception as error:
        LED_RED.on()
        try:
            usb = pyb.USB_VCP()
            uart = UART(4, baudrate=115200, timeout=0, timeout_char=0)
        except Exception:
            usb, uart = None, None
        fallback = {"schema_version": 1, "schema_id": "nicla-face-telemetry-1",
            "event": "startup_error", "run_id": "boot-%08x" % int(time.ticks_ms()),
            "sequence": 0, "ticks_ms": int(time.ticks_ms()),
            "reason_code": type(error).__name__, "message": str(error)[:160],
            "status": "recoverable_fault"}
        try:
            line = (json.dumps(fallback, separators=(",", ":")) + "\n").encode("utf-8")
            if uart is not None:
                uart.write(line)
            if usb is not None and usb.isconnected():
                usb.write(line)
        except Exception:
            pass
        raise


if __name__ == "__main__":
    main()
