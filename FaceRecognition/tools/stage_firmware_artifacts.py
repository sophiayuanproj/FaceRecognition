#!/usr/bin/env python3
"""Stage the frozen C/D/E exports and write their versioned Nicla contracts."""

import hashlib
import json
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PROJECT = ROOT.parent
CODEX = PROJECT.parent.parent
STAGE = ROOT / "models" / "deployment-v1"
KNOWN = ["K01", "K02", "K03", "K04", "K05"]
CLASS_ORDER = KNOWN + ["OTHER"]


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True, indent=2) + "\n", encoding="utf-8")


def copy_checked(source, target, expected):
    if not source.is_file():
        raise FileNotFoundError("required frozen export missing: " + str(source))
    actual = sha256(source)
    if actual != expected:
        raise ValueError("frozen export checksum mismatch for %s: %s != %s" % (source, actual, expected))
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists() and sha256(target) != expected:
        raise FileExistsError("refusing to replace different staged artifact: " + str(target))
    if not target.exists():
        shutil.copy2(source, target)
    return actual


def main():
    study = PROJECT / "work" / "project-finish-v4"
    gallery_path = STAGE / "fixed_roster_gallery.json"
    manifest_path = STAGE / "integration_manifest.json"
    if gallery_path.is_file() and manifest_path.is_file():
        existing_gallery = json.loads(gallery_path.read_text())
        if existing_gallery.get("status") == "ready":
            existing_manifest = json.loads(manifest_path.read_text())
            checks = [sha256(gallery_path) == existing_manifest.get("gallery", {}).get("sha256"),
                      sha256(ROOT / "firmware/runtime_config.json") == existing_manifest.get("runtime_config", {}).get("sha256")]
            checks.extend((STAGE / name).is_file() and sha256(STAGE / name) == row.get("sha256")
                          for name, row in existing_manifest.get("files", {}).items())
            for role, filename in (("detector", "detector_gray_fomo_int8.txt"),
                                   ("classifier", "identity_classifier_a025_int8.txt")):
                checks.append((STAGE / filename).is_file() and
                    sha256(STAGE / filename) == existing_manifest.get("labels", {}).get(role + "_sha256"))
            if not checks or not all(checks):
                raise ValueError("ready integration stage is inconsistent; refusing to replace the gallery or config")
            print(json.dumps({"status": "already_ready", "gallery_sha256": sha256(gallery_path),
                "runtime_config_sha256": sha256(ROOT / "firmware/runtime_config.json"),
                "files_verified": len(existing_manifest.get("files", {}))}, indent=2))
            return
    c_root = CODEX / "2026-10-28" / "job-c-train-detection-and-make" / "outputs" / "C"
    c_model = c_root / "models/trained/cheap_sepconv_16/seed-20261102/detector_full_int8.tflite"
    c_labels = c_root / "models/trained/cheap_sepconv_16/seed-20261102/labels.txt"
    d_root = study / "runs/D/survface/sv_mnv1_a010_transfer_6/seed-20261110"
    d_model = d_root / "embedding_int8.tflite"
    d_export = json.loads((d_root / "ROOT_RESULT.json").read_text())
    d_output = d_export["int8_embedding_export"]["output"]
    d_selection = study / "runs/D/gallery/a01/seed-20261110/int8_uniform.CONFIG.json"
    d_calibration = json.loads(d_selection.read_text())
    d_expected = "0bf2bc5fb05b0c02d5d09217fe01834182578ebd92f9c001cd088c8ce5a9b0cd"
    e_root = study / "runs/E/roots/capacity_a025/seed-20261101/exports"
    e_model = e_root / "full_int8_ptq_train_only_rep.tflite"
    e_export_path = e_root / "full_int8_ptq_train_only_rep.json"
    e_export = json.loads(e_export_path.read_text())
    if e_export.get("class_order") != CLASS_ORDER:
        raise ValueError("E class order differs from frozen five-known roster")

    targets = {
        "detector_gray_fomo_int8.tflite": (c_model, "4a484c9b56f806b4d13b603e66c3f7df60012ac74149e7a28655ddd497e5195c"),
        "identity_embedding_a010_int8.tflite": (d_model, d_expected),
        "identity_classifier_a025_int8.tflite": (e_model, e_export["sha256"]),
    }
    files = {}
    for target_name, (source, expected) in targets.items():
        target = STAGE / target_name
        actual = copy_checked(source, target, expected)
        files[target_name] = {"source": str(source), "path": str(target), "sha256": actual,
                              "bytes": target.stat().st_size}

    detector_label_bytes = c_labels.read_bytes() if c_labels.is_file() else b"face\n"
    if not detector_label_bytes.strip():
        raise ValueError("C detector label map is empty")
    detector_label = STAGE / "detector_gray_fomo_int8.txt"
    if detector_label.exists() and detector_label.read_bytes() != detector_label_bytes:
        raise FileExistsError("refusing to replace different staged detector labels")
    detector_label.write_bytes(detector_label_bytes)
    class_label = STAGE / "identity_classifier_a025_int8.txt"
    class_label.write_text("\n".join(CLASS_ORDER) + "\n", encoding="ascii")

    selected = d_calibration["selected"]
    selection = selected["selection"]
    d_config_path = d_root / "config.json"
    d_threshold = float(selected["calibration"]["threshold"])
    e_calibration = e_export["calibration"]
    if e_calibration.get("score_name") != "max_known_softmax_probability":
        raise ValueError("E output score contract changed")
    e_threshold = float(e_calibration["threshold"])

    config = {
        "schema_version": "wearable-runtime-config-1",
        "protocol_id": "nicla-five-known-study-v4",
        "firmware_contract_version": 1,
        "default_policy": "S",
        "roster": {"ids": KNOWN, "other_label": "OTHER", "max_faces": 1},
        "camera": {
            "initial_framesize": "QVGA", "framebuffer_count": 1,
            "windows": {"QVGA": [40, 0, 240, 240], "VGA": [200, 120, 240, 240]},
            "resolution_options": ["QVGA", "VGA"], "hmirror": False, "vflip": False,
            "frame_rate_hz": 4, "max_frame_rate_hz": 30, "exposure_settle_ms": 1500,
            "wake_frame_attempts": 1, "camera_sleep_idle_ms": 2000,
            "stationary_periodic_check_ms": 5000,
            "note": "sensor frame dropping and CSI sleep are separate controls; gray values are not lux"
        },
        "telemetry": {
            "uart_enabled": False, "uart_bus": 4, "uart_baud": 115200,
            "uart_pin_map_documented": {"tx": "SDA (J2-1)", "rx": "SCL (J2-2)"},
            "usb_enabled": True, "max_record_bytes": 1024, "max_queue_bytes_per_sink": 8192,
            "flush_bytes_per_loop": 512, "uart_flush_bytes_per_loop": 512,
            "usb_flush_bytes_per_loop": 2048, "control_line_max_bytes": 192,
            "ticks_unit": "ticks_ms", "ticks_modulus_source": "ticks_add(0,-1)+1"
        },
        "detection": {
            "backend": "trained_gray_fomo", "threshold": 0.7,
            "model_path": "/flash/models/deployment-v1/detector_gray_fomo_int8.tflite",
            "model_sha256": files["detector_gray_fomo_int8.tflite"]["sha256"],
            "labels_path": "/flash/models/deployment-v1/detector_gray_fomo_int8.txt",
            "input_shape": [1, 96, 96, 1], "input_dtype": "int8",
            "input_scale": 0.003921568859368563, "input_zero_point": -128,
            "output_shape": [1, 12, 12, 2], "output_dtype": "int8",
            "output_scale": 0.00390625, "output_zero_point": -128,
            "class_order": ["background", "face"], "threshold_comparison": "strict_gt",
            "resize_contract": "grayscale 240x240 window to 96x96 bilinear; no padding",
            "coordinate_contract": "openmv-v5.0.1-fomo-240-window-v1",
            "stock_rgb_alternative": {"backend": "stock_rgb_fomo", "model_id": "stock-rom-fomo",
                "model_path": "/rom/fomo_face_detection.tflite",
                "input_shape": [1, 96, 96, 3], "input_dtype": "int8",
                "input_scale": 0.003921568859368563, "input_zero_point": -128,
                "output_dtype": None, "pixel_format": "RGB565", "threshold": 0.7,
                "output_contract": "validate from the ROM model metadata and labels at load; never use the trained-gray contract",
                "status": "optional separate RGB565 path; not selected for this release"}
        },
        "identity": {
            "uncertainty_band": 0.025,
            "quality": {"mean_min_u8": 12, "mean_max_u8": 243, "stdev_min_u8": 5,
                        "brightness_is_lux": False, "motion_is_wearer_device_only": True},
            "crop": {"side_window_px": 96, "rounding": "floor(center-side/2+0.5)",
                     "boundary_policy": "reject_without_clamp_or_padding",
                     "resize": "grayscale 96x96 identity input"},
            "small": {"backend": "embedding_gallery", "model_id": "d-a010-int8-seed-20261110",
                "model_path": "/flash/models/deployment-v1/identity_embedding_a010_int8.tflite",
                "model_sha256": files["identity_embedding_a010_int8.tflite"]["sha256"],
                "input_shape": [1, 96, 96, 1], "input_dtype": "int8",
                "input_scale": 0.003921568859368563, "input_zero_point": -128,
                "output_shape": [1, 64], "output_dtype": "int8",
                "output_scale": float(d_output["scale"]), "output_zero_point": int(d_output["zero_point"]),
                "threshold": d_threshold, "threshold_source_sha256": sha256(d_selection),
                "model_config_sha256": sha256(d_config_path),
                "gallery_path": "/flash/models/deployment-v1/fixed_roster_gallery.json",
                "prototype_sha256": selected["prototype_sha256"],
                "calibration_role": "D threshold_calibration+unknown_calibration"},
            "large": {"backend": "classifier_logits", "model_id": "e-capacity-a025-int8-seed-20261101",
                "model_path": "/flash/models/deployment-v1/identity_classifier_a025_int8.tflite",
                "model_sha256": files["identity_classifier_a025_int8.tflite"]["sha256"],
                "input_shape": [1, 96, 96, 1], "input_dtype": "int8",
                "input_scale": 0.003921568859368563, "input_zero_point": -128,
                "output_shape": [1, 6], "output_dtype": "int8", "output_activation": "logits",
                "output_scale": 0.026111828163266182, "output_zero_point": -10,
                "threshold": e_threshold, "threshold_source_sha256": sha256(e_export_path),
                "class_order": CLASS_ORDER, "calibration_score": "max_known_softmax_probability",
                "model_config_sha256": sha256(e_root.parent / "ROOT_RESULT.json")}
        },
        "controller_calibration": {"status": "pending_development_only_risk_and_measured_cost_fit",
            "source_role": "development_only", "risk_threshold": 0.1,
            "energy_value_mj_per_correct": None, "risk_bins": []},
        "models": {"cache_strategy": "bounded_lru", "max_resident_models": 1,
                   "allowed_max_resident_models": 2, "minimum_python_heap_free_bytes": 49152,
                   "measure_gc_collect": True},
        "policies": {
            "H": {"interval_ms": 250, "resolution": "QVGA", "identity_backend": "large",
                  "quality_gate": True, "motion_gate": False, "motion_adaptation": False, "temporal_confirmation": False, "camera_sleep": False},
            "L": {"interval_ms": 2000, "resolution": "QVGA", "identity_backend": "large",
                  "quality_gate": True, "motion_gate": False, "motion_adaptation": False, "temporal_confirmation": False, "camera_sleep": False},
            "S": {"interval_ms": 250, "resolution": "QVGA", "identity_backend": "small",
                  "quality_gate": True, "motion_gate": False, "motion_adaptation": False, "temporal_confirmation": False, "camera_sleep": False},
            "G": {"interval_ms": 250, "resolution": "QVGA", "identity_backend": "large",
                  "quality_gate": True, "motion_gate": False, "motion_adaptation": False, "temporal_confirmation": False, "camera_sleep": False},
            "A-sense": {"interval_ms": "adaptive", "moving_interval_ms": 250, "still_interval_ms": 1500,
                  "resolution": "QVGA", "identity_backend": "large", "quality_gate": True,
                  "motion_gate": True, "motion_adaptation": True, "temporal_confirmation": False, "camera_sleep": False},
            "A-model": {"interval_ms": 1000, "resolution": "QVGA", "identity_backend": "small_then_large_near_threshold",
                  "max_escalations_per_minute": 6,
                  "quality_gate": True, "motion_gate": False, "motion_adaptation": False, "temporal_confirmation": False, "camera_sleep": False},
            "A-risk": {"interval_ms": 1000, "resolution": "QVGA", "identity_backend": "small_then_large_risk_aware",
                  "max_escalations_per_minute": 6,
                  "quality_gate": True, "motion_gate": False, "motion_adaptation": False, "temporal_confirmation": False, "camera_sleep": False},
            "A-cost": {"interval_ms": 1000, "resolution": "QVGA", "identity_backend": "small_then_large_cost_aware",
                  "max_escalations_per_minute": 6,
                  "quality_gate": True, "motion_gate": False, "motion_adaptation": False, "temporal_confirmation": False, "camera_sleep": False},
            "A-full": {"interval_ms": "adaptive", "moving_interval_ms": 250, "still_interval_ms": 1500,
                  "resolution": "VGA", "identity_backend": "small_then_large_near_threshold", "quality_gate": True,
                  "max_escalations_per_minute": 6,
                  "motion_gate": True, "motion_adaptation": True, "temporal_confirmation": "2_of_3", "camera_sleep": True},
            "T": {"interval_ms": 250, "resolution": "QVGA", "identity_backend": "small",
                  "quality_gate": True, "motion_gate": False, "motion_adaptation": False,
                  "temporal_confirmation": False, "camera_sleep": False,
                  "scene_trigger": {"mean_delta_u8": 8, "stdev_delta_u8": 6,
                      "hysteresis_mean_u8": 4, "hysteresis_stdev_u8": 3,
                      "min_detector_interval_ms": 500, "detector_fallback_ms": 2000}},
            "U": {"interval_ms": 2000, "resolution": "QVGA", "identity_backend": "small",
                  "quality_gate": False, "motion_gate": False, "motion_adaptation": False, "temporal_confirmation": False,
                  "camera_sleep": False,
                  "stage_behavior": "fixed 2000ms capture; run detector; NO_FACE stops; reject multiple/clipped crops; do not gate on scene/crop quality; one valid crop may invoke D embedding gallery; never invoke E classifier; no temporal votes"}
        },
        "motion": {"acceleration_delta_mg": 120, "angular_rate_mdps": 18000,
                   "adaptation_only": True, "image_brightness_is_not_motion": True},
        "control": {"protocol": "nicla-face-control-1", "max_trial_frames": 50,
                    "max_trial_duration_ms": 30000, "max_decision_latency_ms": 5000,
                    "measurement_stage_ids": ["full_pipeline", "idle", "camera_capture_only",
                        "detector_only", "identity_small_only", "identity_large_only",
                        "cold_model_load", "warm_inference", "model_swap_and_gc"],
                    "max_policy_interval_ms": 10000},
        "watchdog_timeout_ms": 30000
    }
    config_path = ROOT / "firmware" / "runtime_config.json"
    write_json(config_path, config)

    # A safe, explicit placeholder blocks deployment until build_fixed_gallery.py
    # replaces it with actual vectors and verifies the frozen prototype checksum.
    write_json(gallery_path, {"schema_version": "wearable-roster-1", "status": "awaiting_selected_train_vectors",
        "roster_id": "K01-K05-public-study-roster", "identity_order": KNOWN,
        "model_id": "d-a010-int8-seed-20261110", "model_sha256": files["identity_embedding_a010_int8.tflite"]["sha256"],
        "prototype_sha256": selected["prototype_sha256"], "selection_sha256": sha256(d_selection),
        "entries": [{"identity_id": identity, "vectors": {}} for identity in KNOWN]})

    manifest = {"schema": "wearable-integration-artifacts-1", "created_utc": "2026-10-30",
        "files": files,
        "labels": {"detector_path": str(detector_label), "detector_sha256": sha256(detector_label),
                   "classifier_path": str(class_label), "classifier_sha256": sha256(class_label),
                   "class_order": CLASS_ORDER},
        "runtime_config": {"path": str(config_path), "sha256": sha256(config_path)},
        "gallery": {"path": str(gallery_path), "sha256": sha256(gallery_path),
                    "status": "placeholder_blocks_deployment_until_actual_builder_runs"},
        "source_receipts": {"C": str(c_root / "reports/RESULT.json"),
            "D": str(study / "handoffs/D/RESULT.json"), "E": str(study / "handoffs/E/RESULT.json")}}
    write_json(manifest_path, manifest)
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
