"""Verified cold-boot loader for versioned Nicla runtime releases.

The detector arena is reserved before importing the application. On this
OpenMV build, importing the source runtime first consumes enough Python heap
to make the FOMO allocator fail even though the model fits when loaded first.
Full releases carry source for recovery. Compact releases require verified,
version-matched bytecode and have no source fallback.
"""

import binascii
import gc
import hashlib
import json
import os
import time

import pyb
from machine import LED, UART

LED_RED = LED("LED_RED")
LED_BLUE = LED("LED_BLUE")
_MPY_MODULES = ("wearable_app", "identity_contracts", "runtime_support", "telemetry")


def _read(path):
    with open(path, "r") as stream:
        return stream.read(128).strip()


def _json(path):
    with open(path, "r") as stream:
        return json.load(stream)


def _sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        while True:
            block = stream.read(1024)
            if not block:
                break
            digest.update(block)
    return binascii.hexlify(digest.digest()).decode("ascii")


def _release_path():
    value = _read("/flash/face_runtime.current")
    if not value.startswith("/flash/releases/") or ".." in value.split("/"):
        raise ValueError("invalid_release_pointer")
    return value.rstrip("/")


def _insert_path(path, position=0):
    import sys
    if path in sys.path:
        sys.path.remove(path)
    sys.path.insert(position, path)


def _verified_mpy_path(release, binding=None):
    """Verify the installed bytecode and source provenance before importing."""
    root = release + "/mpy"
    path = root + "/build_manifest.json"
    binding = binding or {}
    source_free = binding.get("source_free") is True
    if source_free and (binding.get("manifest_path") != path or
                        _sha256(path) != binding.get("manifest_sha256")):
        raise ValueError("compact_runtime_manifest_hash_mismatch")
    manifest = _json(path)
    if manifest.get("schema") != "nicla-precompiled-runtime-1":
        raise ValueError("unsupported_precompiled_runtime_manifest")
    import sys
    device_mpy = int(getattr(sys.implementation, "_mpy", -1))
    if int(manifest.get("mpy_value", -2)) != device_mpy:
        raise ValueError("precompiled_runtime_abi_mismatch")
    modules = manifest.get("modules", {})
    sources = manifest.get("sources", {})
    module_names = _MPY_MODULES
    if "offline_sightings.mpy" in modules:
        module_names += ("offline_sightings",)
    if source_free:
        if (set(modules) != set(module + ".mpy" for module in module_names) or
                set(sources) != set(module + ".py" for module in module_names) or
                set(binding.get("modules", {})) != set(modules) or
                binding.get("mpy_value") != device_mpy or
                binding.get("format_low_byte") != (device_mpy & 255)):
            raise ValueError("compact_runtime_module_set_mismatch")
    for module in module_names:
        mpy_name = module + ".mpy"
        row = modules.get(mpy_name)
        source_name = module + ".py"
        source_row = sources.get(source_name)
        if not isinstance(row, dict) or not isinstance(source_row, dict):
            raise ValueError("precompiled_runtime_module_missing:" + module)
        mpy_path = root + "/" + mpy_name
        source_path = release + "/" + source_name
        if os.stat(mpy_path)[6] != int(row.get("bytes", -1)):
            raise ValueError("precompiled_runtime_size_mismatch:" + module)
        if _sha256(mpy_path) != row.get("sha256"):
            raise ValueError("precompiled_runtime_hash_mismatch:" + module)
        if source_free:
            linked = binding["modules"][mpy_name]
            if (not isinstance(linked, dict) or linked.get("path") != mpy_path or
                    linked.get("sha256") != row.get("sha256") or
                    linked.get("bytes") != row.get("bytes") or
                    linked.get("source") != source_name or row.get("source") != source_name):
                raise ValueError("compact_runtime_module_binding_mismatch:" + module)
        elif _sha256(source_path) != source_row.get("sha256"):
            raise ValueError("precompiled_runtime_source_mismatch:" + module)
    if int(manifest.get("format_low_byte", -1)) != (device_mpy & 255):
        raise ValueError("precompiled_runtime_format_mismatch")
    return root


def _preload_haar_identity(contract):
    """Reserve the verified small model before app and camera allocations."""
    path = contract["model_path"]
    digest = _sha256(path)
    if digest != contract.get("model_sha256"):
        raise ValueError("small_model_hash_mismatch")
    import ml
    gc.collect()
    # OpenMV tries GC memory for the weights first. Leave less space than
    # the file size so weights use its separate allocator.
    guards = []
    try:
        # A single large reservation can fail despite enough aggregate free
        # space after the release hash checks. Reserve in bounded chunks.
        while gc.mem_free() > 92 * 1024:
            guards.append(bytearray(min(16384, gc.mem_free() - 92 * 1024)))
        model = ml.Model(path)
    finally:
        guards = None
        gc.collect()
    return model, digest


def _preload_models(release):
    """Validate release bindings and reserve the first model before app import."""
    started = time.ticks_us()
    config_path = release + "/runtime_config.json"
    manifest_path = release + "/models/deployment-v1/integration_manifest.json"
    config = _json(config_path)
    integration = _json(manifest_path)
    expected_config_sha = integration.get("runtime_config", {}).get("sha256")
    if not expected_config_sha or _sha256(config_path) != expected_config_sha:
        raise ValueError("runtime_config_hash_mismatch")
    detection = config.get("detection", {})
    if detection.get("backend") == "gray_haar_box":
        status = {"status": "deferred_native_cascade"}
        contract = config["identity"]["small"]
        use_wide = bool(detection.get("wide_contrast_gain"))
        config = integration = detection = None
        gc.collect()
        wide = None
        try:
            # Take the identity arena while the heap is still contiguous.
            # Both allocations remain resident, so wide scratch comes next.
            small, digest = _preload_haar_identity(contract)
            status["early_small_model_sha256"] = digest
            status["early_small_arena_bytes"] = int(small.ram)
            if use_wide:
                import image
                wide = image.Image(240, 240, image.GRAYSCALE)
        except (OSError, ValueError, MemoryError) as error:
            small = None
            status["early_small_model_error"] = type(error).__name__ + ":" + str(error)[:100]
        return None, None, status, small, wide
    stock = detection.get("backend") == "stock_rgb_fomo"
    detector_contract = detection.get("stock_rgb_alternative", {}) if stock else detection
    model_path = detector_contract.get("model_path")
    expected_model_sha = detector_contract.get("model_sha256")
    if not model_path:
        raise ValueError("detector_model_hash_mismatch")
    # Stock ROMFS models are supplied by the pinned OpenMV firmware rather
    # than copied into this release. Record their observed hash at startup;
    # if a manifest pins one, still enforce it. Flash models always require a
    # hash-closed contract.
    model_sha = _sha256(model_path)
    if (expected_model_sha and model_sha != expected_model_sha) or (not stock and not expected_model_sha):
        raise ValueError("detector_model_hash_mismatch")
    label = integration.get("labels", {})
    if label.get("detector_sha256") != _sha256(label.get("detector_path", "")):
        raise ValueError("detector_label_hash_mismatch")
    gc.collect()
    heap_before = gc.mem_free()
    import ml
    postprocessor = None
    if detector_contract.get("backend") != "gray_box_regressor":
        from ml.postprocessing.edgeimpulse import Fomo
        postprocessor = Fomo(threshold=float(detector_contract.get("threshold", 0.7)))
    load_started = time.ticks_us()
    detector = (ml.Model(model_path, postprocess=postprocessor) if postprocessor is not None
                else ml.Model(model_path))
    load_us = time.ticks_diff(time.ticks_us(), load_started)
    gc.collect()
    return detector, postprocessor, {
        "status": "loaded", "model_id": detector_contract.get("model_id", "C-trained-gray-fomo"),
        "model_sha256": model_sha, "model_bytes": int(detector.len),
        "arena_bytes": int(detector.ram), "heap_free_before_bytes": heap_before,
        "heap_free_after_bytes": gc.mem_free(), "load_us": load_us,
        "validation_and_load_us": time.ticks_diff(time.ticks_us(), started),
        "preloaded_before_runtime_import": True,
    }, None, None


def main():
    LED_BLUE.on()
    time.sleep_ms(60)
    LED_BLUE.off()
    try:
        release = _release_path()
        early_detector = None
        early_postprocessor = None
        early_small = None
        early_wide = None
        preload_status = {"status": "failed", "reason_code": "not_attempted"}
        mpy_root = None
        source_free = False
        try:
            # Model bindings are verified by the preloader. Do this before
            # decoding the module manifest and its nested digest tables.
            early_detector, early_postprocessor, preload_status, early_small, early_wide = _preload_models(release)
            integration = _json(release + "/models/deployment-v1/integration_manifest.json")
            binding = integration.get("precompiled_runtime") or {}
            source_free = binding.get("source_free") is True
            mpy_root = _verified_mpy_path(release, binding)
            integration = binding = None
            if not source_free:
                _insert_path(release)
            _insert_path(mpy_root)
        except Exception as error:
            if source_free:
                raise
            # A missing or damaged compact receipt may prevent layout
            # detection. Require the selected release's source to exist
            # before permitting the legacy source recovery path.
            os.stat(release + "/wearable_app.py")
            # Never import a mismatched binary. Source is retained as a
            # diagnostic/recovery path, and app telemetry reports this fault.
            preload_status = {"status": "failed",
                "reason_code": type(error).__name__ + ":" + str(error)[:120],
                "heap_free_bytes": gc.mem_free(),
                "mpy_manifest_valid": mpy_root is not None}
            early_detector = early_postprocessor = early_small = early_wide = None
            gc.collect()
            if mpy_root is not None:
                import sys
                try:
                    sys.path.remove(mpy_root)
                except ValueError:
                    pass
            _insert_path(release)

        import wearable_app
        wearable_app.configure_release_root(release)
        wearable_app.BOOT_PRELOADED_DETECTOR = early_detector
        wearable_app.BOOT_PRELOADED_FOMO = early_postprocessor
        wearable_app.BOOT_PRELOADED_SMALL = early_small
        wearable_app.BOOT_PRELOADED_HAAR_WIDE = early_wide
        wearable_app.BOOT_DETECTOR_PRELOAD_STATUS = preload_status
        # Drop boot-local references before the runtime allocates its camera
        # framebuffer and telemetry state. ModelPool now owns the sole ref.
        early_detector = None
        early_postprocessor = None
        early_small = None
        early_wide = None
        gc.collect()
        wearable_app.main()
    except KeyboardInterrupt:
        LED_RED.off()
        raise
    except Exception as error:
        LED_RED.on()
        row = {"schema_version": 1, "schema_id": "nicla-face-telemetry-1",
               "event": "bootloader_error", "run_id": "boot-%08x" % int(time.ticks_ms()),
               "sequence": 0, "ticks_ms": int(time.ticks_ms()),
               "reason_code": type(error).__name__, "message": str(error)[:120],
               "status": "recoverable_fault"}
        try:
            line = (json.dumps(row, separators=(",", ":")) + "\n").encode("utf-8")
            UART(4, baudrate=115200, timeout=0, timeout_char=0).write(line)
        except Exception:
            pass
        try:
            usb = pyb.USB_VCP()
            if usb.isconnected():
                usb.write((json.dumps(row, separators=(",", ":")) + "\n").encode("utf-8"))
        except Exception:
            pass
        raise


if __name__ == "__main__":
    main()
