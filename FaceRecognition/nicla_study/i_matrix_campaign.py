"""Prepare and execute the hash-bound E-root Nicla pairwise matrix.

The source-side command creates a finite, portable overlay bundle for every
full-int8 E root and a paired capture ledger. Physical capture remains gated on
Windows G, exact-firmware H, per-variant board qualification, and live device
identity/configuration readback. Other E export formats remain in the source
matrix and are recorded as target-incompatible unless a future firmware
preprocessing contract explicitly supports them.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
from functools import lru_cache
import hashlib
import json
import os
from pathlib import Path
import random
import re
import shutil
import subprocess
import sys
import time

from .fs import FileLock, atomic_write_json, sha256_file

WORKSPACE = Path(__file__).resolve().parents[2]
APP = WORKSPACE / "FaceRecognition"
STUDY = WORKSPACE / "work/project-finish-v4"
MATRIX = STUDY / "config/I_PHYSICAL_MATRIX.json"
CAPTURE = APP / "nicla_study/instruments/windows_fnirsi/capture_windows.py"
DEPLOY = APP / "tools/deploy_nicla_variant.py"
FIRMWARE_NAMES = ("wearable_app.py", "identity_contracts.py", "runtime_support.py", "telemetry.py")
FIRMWARE_PATHS = {
    "wearable_app.py": APP / "firmware/main.py",
    "identity_contracts.py": APP / "firmware/identity_contracts.py",
    "runtime_support.py": APP / "firmware/runtime_support.py",
    "telemetry.py": APP / "firmware/telemetry.py",
}
DEPLOYMENT_MODEL_FILES = (
    "detector_gray_fomo_int8.tflite",
    "detector_gray_fomo_int8.txt",
    "identity_embedding_a010_int8.tflite",
    "identity_classifier_a025_int8.tflite",
    "identity_classifier_a025_int8.txt",
    "fixed_roster_gallery.json",
    "embedding_gallery_calibration.json",
    "integration_manifest.json",
)
FULL_INT8 = "full_int8_ptq_train_only_rep"
KNOWN_LABELS = ["K01", "K02", "K03", "K04", "K05", "OTHER"]
INPUT_SCALE = 0.003921568859368563


def sha256(path: Path) -> str:
    return sha256_file(path)


def _read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _json_bytes(value: dict) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n").encode("utf-8")


def _write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = _json_bytes(value)
    if path.exists():
        if path.read_bytes() == payload:
            return
        raise FileExistsError("refusing to replace a different prepared bundle file: " + str(path))
    path.write_bytes(payload)


def firmware_sha256() -> str:
    digest = hashlib.sha256()
    for name in FIRMWARE_NAMES:
        digest.update((name + "\t" + sha256(FIRMWARE_PATHS[name]) + "\n").encode("utf-8"))
    return digest.hexdigest()


@lru_cache(maxsize=1)
def source_matrix_sha256() -> str:
    return sha256(MATRIX)


def _dtype(value) -> str:
    text = str(value).lower()
    if "uint8" in text or text == "b'b'":
        return "uint8"
    if "int8" in text or text == "b":
        return "int8"
    if "float32" in text or text in ("f", "float"):
        return "float32"
    return text


def _safe_source(path_text: str) -> Path:
    raw = Path(path_text)
    source = raw if raw.is_absolute() else WORKSPACE / raw
    source = source.resolve()
    try:
        source.relative_to(WORKSPACE.resolve())
    except ValueError as error:
        raise ValueError("variant payload source escapes the portable workspace: " + str(source)) from error
    if not source.is_file():
        raise FileNotFoundError("variant payload source is missing: " + str(source))
    return source


def _bundle_id(row: dict) -> str:
    raw = "%s|%s|%s|%s|%s|%s|%s" % (row["arm_id"], row["training_root_seed"],
        row["export_format"], row["artifact_sha256"], firmware_sha256(),
        sha256(APP / "firmware/runtime_config.json"), source_matrix_sha256())
    return "I-%s-%s-%s" % (row["arm_id"], row["training_root_seed"],
                            hashlib.sha256(raw.encode()).hexdigest()[:10])


def _export_receipt(row: dict) -> tuple[Path, dict]:
    artifact = _safe_source(row["artifact_path_relative"])
    receipt = artifact.with_suffix(".json")
    if not receipt.is_file():
        raise FileNotFoundError("E export receipt is missing: " + str(receipt))
    return artifact, _read(receipt)


def _int8_contract(row: dict, export: dict) -> tuple[dict, dict]:
    flatbuffer = export.get("flatbuffer", {})
    input_meta = flatbuffer.get("input", {})
    output_meta = flatbuffer.get("output", {})
    in_dtype = _dtype(input_meta.get("dtype"))
    out_dtype = _dtype(output_meta.get("dtype"))
    input_quant = input_meta.get("quantization", [None, None])
    output_quant = output_meta.get("quantization", [None, None])
    if (row.get("export_format") != FULL_INT8 or not flatbuffer.get("all_int8_boundary") or
            input_meta.get("shape") != [1, 96, 96, 1] or in_dtype != "int8" or
            len(input_quant) != 2 or abs(float(input_quant[0]) - INPUT_SCALE) > 1e-6 or
            int(input_quant[1]) != -128):
        raise ValueError("target_runtime_requires_full_int8_96x96_single_channel_input")
    if (output_meta.get("shape") != [1, 6] or out_dtype != "int8" or
            len(output_quant) != 2 or float(output_quant[0]) <= 0 or
            int(output_quant[1]) < -128 or int(output_quant[1]) > 127):
        raise ValueError("target_runtime_requires_six_class_int8_logits")
    if export.get("class_order") != KNOWN_LABELS:
        raise ValueError("E export class order differs from frozen five-known/OTHER contract")
    calibration = export.get("calibration", {})
    threshold = calibration.get("threshold")
    if (calibration.get("score_name") != "max_known_softmax_probability" or
            calibration.get("status") != "calibrated" or not isinstance(threshold, (int, float)) or
            not 0.0 <= float(threshold) <= 1.0):
        raise ValueError("export lacks a valid model-specific development calibration")
    return input_meta, output_meta


def _release_id(bundle_id: str, fw_sha: str, model_sha: str, source_config_sha: str) -> str:
    seed = "I-variant-release-v1|%s|%s|%s|%s" % (bundle_id, fw_sha, model_sha, source_config_sha)
    return "face-i-" + hashlib.sha256(seed.encode("ascii")).hexdigest()[:14]


def _device_paths(value, release_root: str):
    prefix = "/flash/models/deployment-v1"
    target = release_root + "/models/deployment-v1"
    if isinstance(value, dict):
        return {key: _device_paths(item, release_root) for key, item in value.items()}
    if isinstance(value, list):
        return [_device_paths(item, release_root) for item in value]
    if isinstance(value, str) and value.startswith(prefix):
        return value.replace(prefix, target, 1)
    return value


def prepare_variant_bundle(row: dict, output_root: Path) -> dict:
    """Create a lightweight, immutable per-model deployment overlay."""
    if row.get("export_format") != FULL_INT8:
        return {"status": "unsupported_target_format", "artifact_sha256": row.get("artifact_sha256"),
                "reason": "current firmware preprocessing contract only supports full-int8 model inputs"}
    if row.get("artifact_status") != "complete_verified" or row.get("host_inference_status") != "pass":
        return {"status": "unsupported_or_missing_E_artifact", "artifact_sha256": row.get("artifact_sha256"),
                "artifact_status": row.get("artifact_status"),
                "host_inference_status": row.get("host_inference_status")}
    artifact_path, export = _export_receipt(row)
    actual_model_sha = sha256(artifact_path)
    if actual_model_sha != row.get("artifact_sha256") or export.get("sha256") != actual_model_sha:
        raise ValueError("E artifact SHA-256 does not match both the matrix and export receipt")
    if export.get("status") != "complete_verified" or export.get("host_inference_status") != "pass":
        return {"status": "unsupported_or_failed_E_export", "artifact_sha256": actual_model_sha,
                "export_status": export.get("status"), "host_inference_status": export.get("host_inference_status")}
    try:
        input_meta, output_meta = _int8_contract(row, export)
    except ValueError as error:
        return {"status": "unsupported_target_contract", "artifact_sha256": actual_model_sha,
                "reason": str(error), "format": row.get("export_format"),
                "training_root_seed": row.get("training_root_seed"), "arm_id": row.get("arm_id")}

    base_config_path = APP / "firmware/runtime_config.json"
    base_manifest_path = APP / "models/deployment-v1/integration_manifest.json"
    base_config = _read(base_config_path)
    base_manifest = _read(base_manifest_path)
    source_config_sha = sha256(base_config_path)
    bundle_id = _bundle_id(row)
    fw_sha = firmware_sha256()
    model_id = "E-%s-%s-%s" % (row["arm_id"], row["training_root_seed"], actual_model_sha[:12])
    release_id = _release_id(bundle_id, fw_sha, actual_model_sha, source_config_sha)
    release_root = "/flash/releases/" + release_id
    model_target = "identity_classifier_a025_int8.tflite"
    labels_target = "identity_classifier_a025_int8.txt"
    large = base_config["identity"]["large"]
    large.update({
        "model_id": model_id,
        "model_path": "/flash/models/deployment-v1/" + model_target,
        "model_sha256": actual_model_sha,
        "model_config_sha256": row.get("root_config_sha256"),
        "threshold": float(export["calibration"]["threshold"]),
        "threshold_source_sha256": sha256(artifact_path.with_suffix(".json")),
        "input_shape": list(input_meta["shape"]),
        "input_dtype": "int8",
        "input_scale": float(input_meta["quantization"][0]),
        "input_zero_point": int(input_meta["quantization"][1]),
        "output_shape": list(output_meta["shape"]),
        "output_dtype": "int8",
        "output_scale": float(output_meta["quantization"][0]),
        "output_zero_point": int(output_meta["quantization"][1]),
        "output_activation": "logits",
        "class_order": list(export["class_order"]),
        "calibration_score": "max_known_softmax_probability",
        "source_arm_id": row["arm_id"],
        "training_root_seed": int(row["training_root_seed"]),
        "export_format": row["export_format"],
    })
    config = _device_paths(base_config, release_root)
    config["identity"]["large"].update({
        "model_path": release_root + "/models/deployment-v1/" + model_target,
        "labels_path": release_root + "/models/deployment-v1/" + labels_target,
    })
    runtime_config_path = output_root / bundle_id / "runtime_config.json"
    config_bytes = _json_bytes(config)
    if runtime_config_path.exists() and runtime_config_path.read_bytes() != config_bytes:
        raise FileExistsError("refusing to replace a different per-variant runtime config")
    runtime_config_path.parent.mkdir(parents=True, exist_ok=True)
    if not runtime_config_path.exists():
        runtime_config_path.write_bytes(config_bytes)
    runtime_config_sha = hashlib.sha256(config_bytes).hexdigest()

    manifest = json.loads(json.dumps(base_manifest))
    prefix = release_root + "/models/deployment-v1/"
    for filename, record in manifest.get("files", {}).items():
        record["path"] = prefix + filename
        old_source = record.pop("source", None)
        if old_source:
            candidate = Path(old_source)
            try:
                record["source_relative"] = candidate.resolve().relative_to(WORKSPACE.resolve()).as_posix()
            except (OSError, ValueError):
                record["source_provenance"] = "historical source outside portable workspace"
    classifier = manifest.setdefault("files", {}).setdefault(model_target, {})
    classifier.update({"bytes": artifact_path.stat().st_size, "path": prefix + model_target,
                       "sha256": actual_model_sha,
                       "source_relative": artifact_path.relative_to(WORKSPACE).as_posix()})
    manifest["labels"]["detector_path"] = prefix + "detector_gray_fomo_int8.txt"
    manifest["labels"]["classifier_path"] = prefix + labels_target
    manifest["labels"]["classifier_sha256"] = sha256(APP / "models/deployment-v1/" / labels_target)
    manifest["runtime_config"] = {"path": release_root + "/runtime_config.json", "sha256": runtime_config_sha}
    manifest["gallery"]["path"] = prefix + "fixed_roster_gallery.json"
    manifest["D_gallery_calibration"]["path"] = prefix + "embedding_gallery_calibration.json"
    manifest["source_receipts"] = {
        "C": "work/project-finish-v4/handoffs/C/RESULT.json",
        "D": "work/project-finish-v4/handoffs/D/RESULT.json",
        "E": "work/project-finish-v4/handoffs/E/RESULT.json",
    }
    manifest_path = runtime_config_path.parent / "integration_manifest.json"
    manifest_bytes = _json_bytes(manifest)
    if manifest_path.exists() and manifest_path.read_bytes() != manifest_bytes:
        raise FileExistsError("refusing to replace a different per-variant integration manifest")
    if not manifest_path.exists():
        manifest_path.write_bytes(manifest_bytes)
    class_labels_path = runtime_config_path.parent / labels_target
    labels_bytes = ("\n".join(KNOWN_LABELS) + "\n").encode("ascii")
    if class_labels_path.exists() and class_labels_path.read_bytes() != labels_bytes:
        raise FileExistsError("refusing to replace a different classifier label map")
    if not class_labels_path.exists():
        class_labels_path.write_bytes(labels_bytes)

    payload_sources = {
        "wearable_app.py": FIRMWARE_PATHS["wearable_app.py"],
        "boot.py": APP / "firmware/boot.py",
        "identity_contracts.py": FIRMWARE_PATHS["identity_contracts.py"],
        "runtime_support.py": FIRMWARE_PATHS["runtime_support.py"],
        "telemetry.py": FIRMWARE_PATHS["telemetry.py"],
        "runtime_config.json": runtime_config_path,
        "models/deployment-v1/integration_manifest.json": manifest_path,
        "models/deployment-v1/identity_classifier_a025_int8.tflite": artifact_path,
        "models/deployment-v1/identity_classifier_a025_int8.txt": class_labels_path,
    }
    for name in DEPLOYMENT_MODEL_FILES:
        if name in {model_target, labels_target, "integration_manifest.json"}:
            continue
        payload_sources["models/deployment-v1/" + name] = APP / "models/deployment-v1" / name
    payload_files = {}
    for target, source in sorted(payload_sources.items()):
        source = source.resolve()
        try:
            source_relative = source.relative_to(WORKSPACE.resolve()).as_posix()
        except ValueError as error:
            raise ValueError("deployment input is outside the portable workspace: " + str(source)) from error
        if not source.is_file():
            raise FileNotFoundError("deployment payload file missing: " + str(source))
        payload_files[target] = {"source_relative": source_relative,
                                 "sha256": sha256(source), "bytes": source.stat().st_size}
    bundle_manifest = {
        "schema": "I-nicla-variant-overlay-1",
        "source_matrix_sha256": source_matrix_sha256(),
        "bundle_id": bundle_id,
        "arm_id": row["arm_id"],
        "training_root_seed": int(row["training_root_seed"]),
        "export_format": row["export_format"],
        "artifact_sha256": actual_model_sha,
        "artifact_source_relative": artifact_path.relative_to(WORKSPACE).as_posix(),
        "artifact_bytes": artifact_path.stat().st_size,
        "export_receipt_sha256": sha256(artifact_path.with_suffix(".json")),
        "source_root_result_sha256": row.get("root_config_sha256"),
        "runtime_config_source_sha256": source_config_sha,
        "device_runtime_config_sha256": runtime_config_sha,
        "device_integration_manifest_sha256": hashlib.sha256(manifest_bytes).hexdigest(),
        "firmware_sha256": fw_sha,
        "model_id": model_id,
        "release_id": release_id,
        "release_root": release_root,
        "target_model_path": prefix + model_target,
        "target_labels_path": prefix + labels_target,
        "payload_files": payload_files,
        "status": "source_bundle_ready_hardware_qualification_pending",
        "physical_capture_eligible": False,
        "notes": [
            "The bundle overlays the fixed detector and embedding assets with this exact E classifier root.",
            "The variant remains ineligible until a per-bundle Nicla model/resource probe and exact-firmware H receipt pass.",
            "Only full-int8 input-compatible artifacts are target-deployable under the current crop preprocessing contract.",
        ],
    }
    bundle_manifest_path = runtime_config_path.parent / "variant_manifest.json"
    _write_json(bundle_manifest_path, bundle_manifest)
    return {"status": bundle_manifest["status"], "bundle_id": bundle_id,
            "manifest_path": bundle_manifest_path.relative_to(WORKSPACE).as_posix(),
            "artifact_sha256": actual_model_sha, "firmware_sha256": fw_sha,
            "device_runtime_config_sha256": runtime_config_sha,
            "device_integration_manifest_sha256": bundle_manifest["device_integration_manifest_sha256"],
            "artifact_bytes": artifact_path.stat().st_size}


def _variant_key(row: dict) -> tuple[str, int, str]:
    return str(row.get("arm_id")), int(row.get("training_root_seed")), str(row.get("export_format"))


def _source_variant_status(row: dict) -> dict:
    """Classify a source row without creating a bundle or touching hardware."""
    artifact_sha = row.get("artifact_sha256")
    if row.get("export_format") != FULL_INT8:
        return {"status": "unsupported_target_format", "artifact_sha256": artifact_sha,
                "reason": "current firmware preprocessing contract only supports full-int8 model inputs"}
    if row.get("artifact_status") != "complete_verified" or row.get("host_inference_status") != "pass":
        return {"status": "unsupported_or_missing_E_artifact", "artifact_sha256": artifact_sha,
                "artifact_status": row.get("artifact_status"),
                "host_inference_status": row.get("host_inference_status")}
    try:
        artifact, export = _export_receipt(row)
        actual_sha = sha256(artifact)
        if actual_sha != artifact_sha or export.get("sha256") != actual_sha:
            return {"status": "bundle_validation_failed", "artifact_sha256": artifact_sha,
                    "reason": "E artifact SHA-256 differs from matrix/export receipt"}
        if export.get("status") != "complete_verified" or export.get("host_inference_status") != "pass":
            return {"status": "unsupported_or_failed_E_export", "artifact_sha256": artifact_sha,
                    "export_status": export.get("status"),
                    "host_inference_status": export.get("host_inference_status")}
        _int8_contract(row, export)
    except ValueError as error:
        return {"status": "unsupported_target_contract", "artifact_sha256": artifact_sha,
                "reason": str(error), "format": row.get("export_format"),
                "training_root_seed": row.get("training_root_seed"), "arm_id": row.get("arm_id")}
    except (OSError, KeyError, TypeError) as error:
        return {"status": "source_variant_validation_failed", "artifact_sha256": artifact_sha,
                "reason": "%s: %s" % (type(error).__name__, error)}
    return {"status": "source_contract_compatible_bundle_not_prepared",
            "artifact_sha256": artifact_sha,
            "reason": "run --prepare-bundles to create the immutable deployment overlay"}


def _catalog(matrix: dict, bundle_status: dict[str, dict]) -> dict:
    variants = {_variant_key(row): row for row in matrix.get("variants", [])
                if row.get("training_root_seed") is not None}
    candidate_arms = sorted({row["arm_id"] for row in matrix.get("variants", [])
                             if row.get("export_format") == FULL_INT8})
    roots = sorted({int(row["training_root_seed"]) for row in matrix.get("variants", [])
                    if row.get("export_format") == FULL_INT8 and row.get("training_root_seed") is not None})
    available_arms = []
    unsupported_arms = []
    for arm in candidate_arms:
        outcomes = []
        for seed in roots:
            variant = variants.get((arm, seed, FULL_INT8), {})
            bundle = bundle_status.get(str(variant.get("artifact_sha256", "")).lower(), {})
            outcomes.append({"training_root_seed": seed,
                "artifact_sha256": variant.get("artifact_sha256"),
                "bundle_status": bundle.get("status", "not_prepared"),
                "reason": bundle.get("reason") or bundle.get("error")})
        if len(outcomes) == 10 and all(
                row["bundle_status"] == "source_bundle_ready_hardware_qualification_pending"
                for row in outcomes):
            available_arms.append(arm)
        else:
            unsupported_arms.append({"arm_id": arm,
                "status": "not_target_deployable_for_all_ten_roots",
                "root_outcomes": outcomes})
    comparisons = []
    for left_index, left in enumerate(available_arms):
        for right in available_arms[left_index + 1:]:
            paired = []
            reasons = []
            for seed in roots:
                a = variants.get((left, seed, FULL_INT8), {})
                b = variants.get((right, seed, FULL_INT8), {})
                a_bundle = bundle_status.get(a.get("artifact_sha256"), {})
                b_bundle = bundle_status.get(b.get("artifact_sha256"), {})
                if a.get("artifact_status") != "complete_verified" or b.get("artifact_status") != "complete_verified":
                    reasons.append("missing_or_failed_E_artifact_root:%s" % seed)
                if a.get("host_inference_status") != "pass" or b.get("host_inference_status") != "pass":
                    reasons.append("host_inference_failed_or_missing:%s" % seed)
                if a_bundle.get("status") != "source_bundle_ready_hardware_qualification_pending":
                    reasons.append("arm_a_bundle_unavailable:%s" % seed)
                if b_bundle.get("status") != "source_bundle_ready_hardware_qualification_pending":
                    reasons.append("arm_b_bundle_unavailable:%s" % seed)
                paired.append({
                    "training_root_seed": seed,
                    "arm_a_artifact_sha256": a.get("artifact_sha256"),
                    "arm_a_artifact_status": a.get("artifact_status"),
                    "arm_a_hardware_probe_status": a.get("hardware_probe_status"),
                    "arm_a_bundle_manifest": a_bundle.get("manifest_path"),
                    "arm_a_device_config_sha256": a_bundle.get("device_runtime_config_sha256"),
                    "arm_a_model_id": a_bundle.get("model_id"),
                    "arm_b_artifact_sha256": b.get("artifact_sha256"),
                    "arm_b_artifact_status": b.get("artifact_status"),
                    "arm_b_hardware_probe_status": b.get("hardware_probe_status"),
                    "arm_b_bundle_manifest": b_bundle.get("manifest_path"),
                    "arm_b_device_config_sha256": b_bundle.get("device_runtime_config_sha256"),
                    "arm_b_model_id": b_bundle.get("model_id"),
                    "minimum_independently_started_paired_blocks": 1,
                    "block_id": "I-%s-%s-%s" % (hashlib.sha256((left + "|" + right).encode()).hexdigest()[:10], seed, FULL_INT8),
                    "randomized_order_seed": int.from_bytes(hashlib.sha256(
                        (left + "-vs-" + right + "|" + str(seed) + "|block-1").encode()).digest()[:4], "big"),
                    "shared_stimulus_seed": int.from_bytes(hashlib.sha256(
                        (left + "-vs-" + right + "|" + str(seed) + "|stimulus").encode()).digest()[:4], "big"),
                })
            comparison_id = "%s-vs-%s-%s" % (left, right, FULL_INT8)
            comparisons.append({
                "comparison_id": comparison_id,
                "arm_a": left,
                "arm_b": right,
                "export_format": FULL_INT8,
                "matched_training_root_seeds": roots,
                "paired_artifacts": paired,
                "minimum_independently_started_paired_blocks_per_root": 1,
                "minimum_paired_root_count": 10,
                "expected_capture_count": 2 * len(paired),
                "source_eligibility": "source_bundle_ready" if not reasons else "source_inputs_incomplete",
                "hardware_eligibility": "pending_exact_H_G_and_variant_receipts",
                "blockers": sorted(set(reasons)),
            })
    return {"schema": "I-deployable-full-int8-pairwise-campaign-1",
            "source_matrix_schema": matrix.get("schema"),
            "source_matrix_sha256": sha256(MATRIX),
            "firmware_sha256": firmware_sha256(),
            "export_formats_preserved_in_source_matrix": sorted({str(v.get("export_format")) for v in matrix.get("variants", [])}),
            "non_deployable_formats_note": "Other export formats remain listed in I_PHYSICAL_MATRIX.json; current firmware input preprocessing expects int8 [1,96,96,1] and each such export must remain explicitly unsupported unless that contract is changed and requalified.",
            "training_root_count": len(roots),
            "target_deployable_arm_count": len(available_arms),
            "target_comparison_count": len(comparisons),
            "target_capture_count_if_all_qualified": sum(row["expected_capture_count"] for row in comparisons),
            "measurement_seconds_per_capture": 50,
            "estimated_hours_if_all_qualified": round(sum(row["expected_capture_count"] for row in comparisons) * 50 / 3600, 2),
            "estimated_raw_disk_upper_bound_mb": round(sum(row["expected_capture_count"] for row in comparisons) * 0.5, 1),
            "full_catalog_pairwise_comparison_count_all_formats": len(matrix.get("all_pairwise_arm_format_comparisons", [])),
            "full_catalog_capture_count_if_every_format_supported_and_qualified": matrix.get("prospective_capture_count_if_every_pair_qualified"),
            "full_catalog_estimated_hours_if_every_format_supported_and_qualified": matrix.get("prospective_hours_if_every_pair_qualified"),
            "physical_measurements_performed": False,
            "target_deployable_arms": available_arms,
            "unsupported_target_arms": unsupported_arms,
            "variant_bundle_outcomes": [
                {"arm_id": row.get("arm_id"), "training_root_seed": row.get("training_root_seed"),
                 "export_format": row.get("export_format"), "artifact_sha256": row.get("artifact_sha256"),
                 **bundle_status.get(str(row.get("artifact_sha256", "")).lower(),
                     {"status": "not_prepared" if row.get("export_format") == FULL_INT8 else "format_not_in_target_bundle_set"})}
                for row in matrix.get("variants", [])],
            "comparisons": comparisons,
            "source_format_compatibility_summary": {"full_int8": len(comparisons),
                "all_other_formats": "preserved in the 544 comparison catalog; not scheduled for board capture under current input contract"}}


def _qualification_map(path: Path | None) -> dict[tuple[str, str], dict]:
    if path is None or not path.is_file():
        return {}
    obj = _read(path)
    rows = obj.get("variant_qualifications", obj.get("qualifications", []))
    return {(str(row.get("artifact_sha256", "")).lower(),
             str(row.get("device_config_sha256", "")).lower()): row for row in rows
            if isinstance(row, dict) and row.get("artifact_sha256") and row.get("device_config_sha256")}


def _receipt(path: Path | None, name: str) -> dict | None:
    if path is None or not path.is_file():
        return None
    return _read(path)


def _present_hash(value, expected: str) -> bool:
    if isinstance(value, dict):
        return any(_present_hash(item, expected) for item in value.values())
    if isinstance(value, list):
        return any(_present_hash(item, expected) for item in value)
    return isinstance(value, str) and value.lower() == expected.lower()


def eligibility(catalog: dict, g_receipt: dict | None, h_receipt: dict | None,
                qualification_path: Path | None) -> tuple[list[dict], dict]:
    fw = firmware_sha256()
    global_reasons = []
    if not g_receipt or g_receipt.get("status") != "complete_verified":
        global_reasons.append("G receipt absent or not complete_verified")
    if not h_receipt or h_receipt.get("status") != "complete_verified":
        global_reasons.append("H receipt absent or not complete_verified")
    elif not _present_hash(h_receipt, fw):
        global_reasons.append("H receipt does not bind the current four-file firmware SHA-256")
    qualifications = _qualification_map(qualification_path)
    candidates = []
    for comparison in catalog["comparisons"]:
        reasons = list(global_reasons)
        for pair in comparison["paired_artifacts"]:
            for side in ("arm_a", "arm_b"):
                artifact_sha = pair.get(side + "_artifact_sha256")
                bundle_config = pair.get(side + "_device_config_sha256")
                row = qualifications.get((str(artifact_sha or "").lower(),
                                          str(bundle_config or "").lower()), {})
                if row.get("status") != "complete_verified":
                    reasons.append("variant_model_resource_probe_missing_or_failed:%s:%s" % (side, pair["training_root_seed"]))
                if row.get("firmware_sha256", "").lower() != fw:
                    reasons.append("variant_probe_firmware_hash_mismatch:%s:%s" % (side, pair["training_root_seed"]))
                if row.get("device_config_sha256", "").lower() != str(bundle_config or "").lower():
                    reasons.append("variant_probe_config_hash_mismatch:%s:%s" % (side, pair["training_root_seed"]))
                if row.get("model_id") != pair.get(side + "_model_id"):
                    reasons.append("variant_probe_model_id_mismatch:%s:%s" % (side, pair["training_root_seed"]))
                if row.get("model_sha256", "").lower() != str(artifact_sha or "").lower():
                    reasons.append("variant_probe_artifact_hash_mismatch:%s:%s" % (side, pair["training_root_seed"]))
        comparison["hardware_eligibility"] = "eligible" if not reasons else "blocked"
        comparison["hardware_blockers"] = sorted(set(reasons))
        if not reasons:
            candidates.append(comparison)
    return candidates, {"global_blockers": global_reasons,
                        "qualified_variant_count": len(qualifications),
                        "eligible_comparison_count": len(candidates),
                        "firmware_sha256": fw}


def _stable_u32(*parts: object) -> int:
    raw = "|".join(str(part) for part in parts).encode("utf-8")
    return int.from_bytes(hashlib.sha256(raw).digest()[:4], "big")


def _run_id(comparison_id: str, seed: int, side: str) -> str:
    return "I-MX-%s-R%s-%s" % (hashlib.sha256(comparison_id.encode()).hexdigest()[:8], seed, side)


def _capture_result(out_dir: Path, command_code: int, pair: dict, side: str,
                    run_id: str, plan_sha: str, deployment_receipt: dict,
                    g_receipt: dict, h_receipt: dict) -> dict:
    metadata_path = out_dir / "run_metadata.json"
    metadata = _read(metadata_path) if metadata_path.is_file() else {}
    serial_path = out_dir / "nicla_serial_events.jsonl"
    power_path = out_dir / "raw_packets.jsonl"
    serial_raw_path = out_dir / "nicla_serial_raw.bin"
    stimulus_path = out_dir / "stimulus-events.jsonl"
    records, malformed = [], 0
    if serial_path.is_file():
        for line in serial_path.read_text(encoding="utf-8").splitlines():
            try:
                records.append(json.loads(line))
            except ValueError:
                malformed += 1
    accepted = [row for row in records if row.get("validation", {}).get("status") == "accepted"]
    rejected = sum(row.get("validation", {}).get("status") == "rejected" for row in records)
    event_records = [row.get("record", {}) for row in accepted]
    model_seen = any(row.get("model_sha256", "").lower() == pair.get(side + "_artifact_sha256", "").lower()
                     and row.get("model_id") == pair.get(side + "_model_id")
                     for row in event_records)
    startup = metadata.get("nicla_startup_receipt", {})
    stimulus = metadata.get("stimulus_player", {})
    checks = {
        "capture_process_exit_zero": command_code == 0,
        "capture_metadata_complete": metadata.get("status") == "complete",
        "live_firmware_hash_matches": startup.get("observed_firmware_sha256") == deployment_receipt.get("firmware_sha256"),
        "live_device_config_hash_matches": startup.get("observed_config_sha256") == deployment_receipt.get("device_config_sha256"),
        "configured_variant_identity_matches": metadata.get("model_id") == pair.get(side + "_model_id"),
        "variant_load_or_prediction_seen": model_seen,
        "nicla_raw_serial_retained": serial_raw_path.is_file(),
        "nicla_serial_events_valid": serial_path.is_file() and rejected == 0 and malformed == 0 and bool(accepted),
        "meter_raw_reports_retained": power_path.is_file() and int(metadata.get("measurement_report_count", 0)) > 0,
        "meter_invalid_packet_count_zero": int(metadata.get("invalid_packet_count", 0)) == 0,
        "development_stimulus_player_completed": stimulus.get("status") == "complete",
        "stimulus_log_hash_bound": stimulus.get("event_log_sha256") == (sha256(stimulus_path) if stimulus_path.is_file() else None),
        "G_receipt_complete_verified": g_receipt.get("status") == "complete_verified",
        "H_receipt_complete_verified": h_receipt.get("status") == "complete_verified",
    }
    return {
        "schema": "I-model-pair-capture-result-1",
        "status": "complete_verified" if all(checks.values()) else "failed_qc_or_incomplete",
        "evidence_tier": "physical_raw_capture_pending_formal_analysis",
        "physical_measurement_performed": int(metadata.get("measurement_report_count", 0)) > 0,
        "comparison_id": pair.get("comparison_id"),
        "training_root_seed": pair["training_root_seed"],
        "side": side,
        "run_id": run_id,
        "block_id": pair["block_id"],
        "shared_stimulus_seed": pair["shared_stimulus_seed"],
        "firmware_sha256": deployment_receipt.get("firmware_sha256"),
        "device_config_sha256": deployment_receipt.get("device_config_sha256"),
        "artifact_sha256": pair.get(side + "_artifact_sha256"),
        "model_id": pair.get(side + "_model_id"),
        "plan_sha256": plan_sha,
        "accepted_serial_records": len(accepted),
        "rejected_serial_records": rejected,
        "malformed_serial_log_lines": malformed,
        "raw_meter_report_count": metadata.get("measurement_report_count"),
        "raw_power_sha256": sha256(power_path) if power_path.is_file() else None,
        "raw_serial_sha256": sha256(serial_raw_path) if serial_raw_path.is_file() else None,
        "serial_event_log_sha256": sha256(serial_path) if serial_path.is_file() else None,
        "stimulus_event_log_sha256": sha256(stimulus_path) if stimulus_path.is_file() else None,
        "checks": checks,
        "claim_scope": "Raw matched full-pipeline capture only. Energy, timing and recognition claims require G-supported QC and block-level Mac analysis.",
    }


def _deployment_command(args, bundle_manifest: Path, capture_id: str) -> list[str]:
    output_receipt = args.campaign_root / "deployments" / (capture_id + ".json")
    log_path = output_receipt.with_suffix(".log")
    output_receipt.parent.mkdir(parents=True, exist_ok=True)
    command = [sys.executable, str(DEPLOY), "--mpremote", str(args.mpremote),
               "--port", args.deploy_port, "--bundle-manifest", str(bundle_manifest),
               "--output-receipt", str(output_receipt)]
    return command


def _capture_command(args, pair: dict, side: str, run_id: str, out_dir: Path,
                     input_manifest_sha: str, session_id: str) -> list[str]:
    command = [sys.executable, str(CAPTURE),
        "--project-root", str(WORKSPACE), "--duration", str(args.duration),
        "--warmup-seconds", str(args.warmup_seconds), "--run-id", run_id,
        "--experiment-id", "I-E-MATRIX", "--scenario-id", pair["comparison_id"],
        "--interface-number", str(args.meter_interface),
        "--board-type", "nicla_vision", "--board-label", args.board_label,
        "--device-reset-id", args.device_reset_id,
        "--firmware-source", str(args.firmware_source), "--firmware-binary", str(args.firmware_binary),
        "--board-configuration", str(WORKSPACE / pair[side + "_bundle_manifest"].replace("variant_manifest.json", "runtime_config.json")),
        "--firmware-version", args.firmware_version,
        "--serial-port", args.serial_port, "--serial-kind", "nicla",
        "--policy-id", "H", "--model-id", pair[side + "_model_id"],
        "--nicla-stage-id", "full_pipeline", "--nicla-model-slot", "large",
        "--physical-block-seed", str(pair["randomized_order_seed"]),
        "--training-root-seed", str(pair["training_root_seed"]),
        "--session-id", session_id, "--condition", "paired_development_mixed",
        "--input-manifest-sha256", input_manifest_sha,
        "--expected-device-firmware-sha256", args.firmware_sha256,
        "--expected-device-config-sha256", pair[side + "_device_config_sha256"],
        "--meter-label", args.meter_label, "--meter-firmware", args.meter_firmware,
        "--topology", args.topology, "--load-description", args.load_description,
        "--stimulus-block", "development", "--stimulus-condition", "mixed",
        "--stimulus-count", str(args.stimulus_count),
        "--stimulus-seed", str(pair["shared_stimulus_seed"]),
        "--stimulus-session-id", session_id,
        "--stimulus-event-log", str(out_dir / "stimulus-events.jsonl"),
        "--stimulus-lighting-state", args.lighting_state,
        "--stimulus-motion-state", args.motion_state,
        "--stimulus-range-cm", str(args.range_cm),
        "--stimulus-hold-seconds", str(args.stimulus_hold_seconds),
        "--stimulus-blank-seconds", str(args.stimulus_blank_seconds),
        "--output-dir", str(out_dir)]
    if args.meter_path_fragment:
        command.extend(["--path-fragment", args.meter_path_fragment])
    if args.nicla_dtr:
        command.append("--nicla-dtr")
    if args.nicla_rts:
        command.append("--nicla-rts")
    return command


def _event(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="\n") as stream:
        stream.write(json.dumps(row, sort_keys=True, ensure_ascii=False) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def _immutable_output(root: Path, run_id: str, resume: bool,
                      plan_sha: str) -> tuple[Path, str | None]:
    target = root / "captures" / run_id
    if not target.exists():
        return target, None
    receipt = target / "capture_result.json"
    if resume and receipt.is_file():
        try:
            prior = _read(receipt)
        except (OSError, ValueError):
            prior = {}
        if prior.get("status") == "complete_verified" and prior.get("plan_sha256") == plan_sha:
            return target, "skip_complete"
    index = 2
    while True:
        rerun = root / "captures" / (run_id + "-R%02d" % index)
        if not rerun.exists():
            return rerun, run_id
        index += 1


def _gate(args, catalog: dict) -> tuple[list[dict], dict, dict, dict]:
    g = _receipt(args.g_qualification_receipt, "G")
    h = _receipt(args.h_qualification_receipt, "H")
    eligible, report = eligibility(catalog, g, h, args.artifact_qualification_manifest)
    if report["global_blockers"]:
        raise ValueError("physical execution blocked: " + "; ".join(report["global_blockers"]))
    if not eligible:
        raise ValueError("physical execution blocked: no comparison has ten exact per-variant H/resource receipts")
    return eligible, report, g, h


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--matrix", type=Path, default=MATRIX)
    p.add_argument("--prepare-bundles", action="store_true", help="create immutable per-root config/manifest overlays; no hardware")
    p.add_argument("--dry-run", action="store_true", help="show complete matrix and hardware blockers; default")
    p.add_argument("--execute", action="store_true", help="run eligible paired captures on qualified Windows hardware")
    p.add_argument("--resume", action="store_true", help="skip only result receipts with complete_verified and matching plan hash")
    p.add_argument("--output-plan", type=Path, default=STUDY / "config/I_DEPLOYABLE_PHYSICAL_MATRIX.json")
    p.add_argument("--bundle-root", type=Path, default=STUDY / "staging/I/variant_bundles")
    p.add_argument("--artifact-qualification-manifest", type=Path,
                   default=STUDY / "handoffs/I/ARTIFACT_QUALIFICATION_MANIFEST.json")
    p.add_argument("--g-qualification-receipt", type=Path)
    p.add_argument("--h-qualification-receipt", type=Path)
    p.add_argument("--campaign-root", type=Path, default=STUDY / "runs/I/pairwise")
    p.add_argument("--mpremote", type=Path)
    p.add_argument("--deploy-port")
    p.add_argument("--serial-port")
    p.add_argument("--meter-interface", type=int)
    p.add_argument("--meter-path-fragment")
    p.add_argument("--board-label")
    p.add_argument("--device-reset-id")
    p.add_argument("--firmware-source", type=Path)
    p.add_argument("--firmware-binary", type=Path)
    p.add_argument("--firmware-version")
    p.add_argument("--firmware-sha256", default=None)
    p.add_argument("--meter-label", default="FNIRSI FNB-C2; verify against live descriptor")
    p.add_argument("--meter-firmware")
    p.add_argument("--topology")
    p.add_argument("--load-description")
    p.add_argument("--duration", type=float, default=40.0)
    p.add_argument("--warmup-seconds", type=float, default=5.0)
    p.add_argument("--stimulus-count", type=int, default=12)
    p.add_argument("--stimulus-hold-seconds", type=float, default=1.5)
    p.add_argument("--stimulus-blank-seconds", type=float, default=1.0)
    p.add_argument("--lighting-state", default="normal")
    p.add_argument("--motion-state", default="stationary")
    p.add_argument("--range-cm", type=float, default=50.0)
    p.add_argument("--nicla-dtr", action="store_true")
    p.add_argument("--nicla-rts", action="store_true")
    return p


def main(argv=None) -> int:
    args = parser().parse_args(argv)
    if args.execute and args.prepare_bundles:
        parser().error("prepare bundles and execute physical capture in separate invocations")
    if args.execute and os.name != "nt":
        parser().error("physical I matrix execution is a Windows action; this command will not emulate hardware on Mac")
    if not args.matrix.is_file():
        parser().error("I physical matrix is missing; run nicla_study.prepare_jik")
    matrix = _read(args.matrix)
    if matrix.get("schema") != "I-full-model-compression-lifecycle-matrix-1":
        parser().error("unsupported I matrix schema")
    output_root = args.bundle_root.resolve()
    bundle_status: dict[str, dict] = {}
    if args.prepare_bundles:
        variants = [row for row in matrix.get("variants", []) if row.get("export_format") == FULL_INT8]
        for index, row in enumerate(variants, start=1):
            try:
                result = prepare_variant_bundle(row, output_root)
            except (OSError, ValueError, KeyError, TypeError) as error:
                result = {"status": "bundle_validation_failed", "artifact_sha256": row.get("artifact_sha256"),
                          "error": "%s: %s" % (type(error).__name__, error)}
            bundle_status[str(row.get("artifact_sha256", "")).lower()] = result
            if index % 25 == 0:
                print("prepared %d/%d full-int8 variant overlays" % (index, len(variants)), file=sys.stderr)
    else:
        for row in matrix.get("variants", []):
            if row.get("export_format") == FULL_INT8 and row.get("artifact_sha256"):
                manifest_path = output_root / _bundle_id(row) / "variant_manifest.json"
                if manifest_path.is_file():
                    try:
                        manifest = _read(manifest_path)
                        bundle_status[str(row["artifact_sha256"]).lower()] = {
                            "status": manifest.get("status"),
                            "manifest_path": manifest_path.relative_to(WORKSPACE).as_posix(),
                            "device_runtime_config_sha256": manifest.get("device_runtime_config_sha256"),
                            "device_integration_manifest_sha256": manifest.get("device_integration_manifest_sha256"),
                            "model_id": manifest.get("model_id"),
                        }
                    except (OSError, ValueError, KeyError):
                        pass
                if str(row.get("artifact_sha256", "")).lower() not in bundle_status:
                    bundle_status[str(row.get("artifact_sha256", "")).lower()] = _source_variant_status(row)
    catalog = _catalog(matrix, bundle_status)
    g_receipt = _receipt(args.g_qualification_receipt, "G")
    h_receipt = _receipt(args.h_qualification_receipt, "H")
    qualified, gate_summary = eligibility(catalog, g_receipt, h_receipt,
                                          args.artifact_qualification_manifest)
    catalog["hardware_gate_summary"] = gate_summary
    catalog["eligible_comparison_ids"] = [row["comparison_id"] for row in qualified]
    catalog["source_matrix_sha256"] = sha256(args.matrix)
    catalog["bundle_status_counts"] = {}
    for row in bundle_status.values():
        status = row.get("status", "not_prepared")
        catalog["bundle_status_counts"][status] = catalog["bundle_status_counts"].get(status, 0) + 1
    catalog["physical_measurements_performed"] = False
    catalog["next_gate"] = "W1/G Windows qualification and a per-variant H model/resource receipt for every arm/root included in a physical comparison"
    atomic_write_json(args.output_plan, catalog)

    if not args.execute:
        print(json.dumps({"status": "source_plan_prepared" if args.prepare_bundles else "dry_run_only",
            "plan": str(args.output_plan), "plan_sha256": sha256(args.output_plan),
            "all_format_comparison_catalog": catalog["full_catalog_pairwise_comparison_count_all_formats"],
            "deployable_format": FULL_INT8,
            "deployable_pairwise_comparisons": catalog["target_comparison_count"],
            "captures_if_all_deployable_pairs_qualified": catalog["target_capture_count_if_all_qualified"],
            "eligible_comparisons_now": gate_summary["eligible_comparison_count"],
            "global_blockers": gate_summary["global_blockers"],
            "bundle_status_counts": catalog["bundle_status_counts"],
            "physical_measurement_performed": False}, indent=2))
        return 0

    if not args.firmware_sha256:
        args.firmware_sha256 = firmware_sha256()
    if args.firmware_sha256.lower() != firmware_sha256().lower():
        parser().error("--firmware-sha256 differs from the current four-file source")
    required = {"--mpremote": args.mpremote, "--deploy-port": args.deploy_port,
        "--serial-port": args.serial_port, "--meter-interface": args.meter_interface,
        "--board-label": args.board_label, "--device-reset-id": args.device_reset_id,
        "--firmware-source": args.firmware_source, "--firmware-binary": args.firmware_binary,
        "--firmware-version": args.firmware_version, "--meter-firmware": args.meter_firmware,
        "--topology": args.topology, "--load-description": args.load_description}
    missing = [name for name, value in required.items() if value is None]
    if missing:
        parser().error("physical execution requires " + ", ".join(missing))
    if not (0 < args.duration <= 600 and 0 <= args.warmup_seconds <= 60 and
            1 <= args.stimulus_count <= 100 and args.range_cm > 0 and
            args.stimulus_hold_seconds > 0 and args.stimulus_blank_seconds >= 0):
        parser().error("duration, warmup, stimulus count, range or hold settings are invalid")
    if args.stimulus_count * (args.stimulus_hold_seconds + args.stimulus_blank_seconds) > args.duration:
        parser().error("measurement window does not cover the balanced stimulus schedule")
    input_manifest = STUDY / "data/primary_split_manifest.jsonl"
    if not input_manifest.is_file():
        parser().error("development input manifest is missing")
    g_receipt, h_receipt = _receipt(args.g_qualification_receipt, "G"), _receipt(args.h_qualification_receipt, "H")
    qualified, gate_summary = _gate(args, catalog)
    plan_sha = sha256(args.output_plan)
    campaign_root = args.campaign_root.resolve()
    args.campaign_root = campaign_root
    campaign_root.mkdir(parents=True, exist_ok=True)
    campaign_events = campaign_root / "CAMPAIGN_EVENTS.jsonl"
    lock_path = campaign_root / ".executor.lock"
    with FileLock(lock_path, timeout_s=0):
        campaign_state_path = campaign_root / "CAMPAIGN_STATE.json"
        if campaign_state_path.exists():
            state = _read(campaign_state_path)
            if state.get("plan_sha256") != plan_sha:
                parser().error("existing I matrix state is bound to another plan hash; preserve it and use a new campaign root")
        else:
            state = {"schema": "I-matrix-campaign-state-1", "plan_sha256": plan_sha,
                "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                "status": "in_progress", "captures": {}, "physical_measurements_performed": False}
            atomic_write_json(campaign_state_path, state)
        input_sha = sha256(input_manifest)
        total = 0
        for comparison in qualified:
            for pair in comparison["paired_artifacts"]:
                order = ["arm_a", "arm_b"]
                random.Random(pair["randomized_order_seed"]).shuffle(order)
                pair["randomized_execution_order"] = order
                session = "I-SESSION-" + hashlib.sha256(pair["block_id"].encode()).hexdigest()[:12]
                for side in order:
                    run_id = _run_id(comparison["comparison_id"], pair["training_root_seed"],
                                     "A" if side == "arm_a" else "B")
                    out_dir, rerun_of = _immutable_output(campaign_root, run_id, args.resume, plan_sha)
                    if rerun_of == "skip_complete":
                        continue
                    if out_dir.name != run_id:
                        run_id = out_dir.name
                    out_dir.parent.mkdir(parents=True, exist_ok=True)
                    bundle_manifest = WORKSPACE / pair[side + "_bundle_manifest"]
                    deploy_cmd = _deployment_command(args, bundle_manifest, run_id)
                    deployment_receipt_path = campaign_root / "deployments" / (run_id + ".json")
                    deployment_receipt_path.parent.mkdir(parents=True, exist_ok=True)
                    try:
                        deploy_proc = subprocess.run(deploy_cmd, cwd=WORKSPACE, text=True,
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=300, check=False)
                        deploy_code, deploy_stdout = deploy_proc.returncode, deploy_proc.stdout or ""
                    except subprocess.TimeoutExpired as error:
                        deploy_code, deploy_stdout = 124, str(error)
                    try:
                        deployment_receipt = _read(deployment_receipt_path) if deployment_receipt_path.is_file() else {}
                    except (OSError, ValueError):
                        deployment_receipt = {}
                    deployment_receipt["process_return_code"] = deploy_code
                    deployment_receipt["command"] = deploy_cmd
                    deployment_receipt["stdout_tail"] = deploy_stdout[-2000:]
                    expected_bundle = _read(bundle_manifest)
                    deployment_receipt_checks = {
                        "firmware_sha256_matches_bundle": str(deployment_receipt.get("firmware_sha256", "")).lower() == str(expected_bundle.get("firmware_sha256", "")).lower(),
                        "device_config_sha256_matches_bundle": str(deployment_receipt.get("device_config_sha256", "")).lower() == str(pair[side + "_device_config_sha256"]).lower(),
                        "artifact_sha256_matches_variant": str(deployment_receipt.get("artifact_sha256", "")).lower() == str(pair[side + "_artifact_sha256"]).lower(),
                        "model_id_matches_variant": deployment_receipt.get("model_id") == pair[side + "_model_id"],
                        "bundle_id_matches_variant": deployment_receipt.get("bundle_id") == expected_bundle.get("bundle_id"),
                    }
                    deployment_receipt["identity_checks"] = deployment_receipt_checks
                    deployment_identity_ok = all(deployment_receipt_checks.values())
                    deploy_out = deployment_receipt_path
                    if deploy_code != 0 or deployment_receipt.get("status") != "installed_verified" or not deployment_identity_ok:
                        result = {"schema": "I-model-pair-capture-result-1", "status": "deployment_failed",
                            "run_id": run_id, "comparison_id": comparison["comparison_id"],
                            "training_root_seed": pair["training_root_seed"], "side": side,
                            "deployment_receipt_path": str(deploy_out), "deployment": deployment_receipt,
                            "physical_measurement_performed": False}
                        result_path = out_dir / "capture_result.json"
                        result_path.parent.mkdir(parents=True, exist_ok=True)
                        atomic_write_json(result_path, result)
                    else:
                        capture_cmd = _capture_command(args, pair, side, run_id, out_dir, input_sha, session)
                        try:
                            proc = subprocess.run(capture_cmd, cwd=WORKSPACE, text=True,
                                stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                timeout=int(args.duration + args.warmup_seconds + 180), check=False)
                            code, capture_stdout = proc.returncode, proc.stdout or ""
                        except subprocess.TimeoutExpired as error:
                            code, capture_stdout = 124, str(error)
                        out_dir.mkdir(parents=True, exist_ok=True)
                        (out_dir / "campaign_command.log").write_text(capture_stdout, encoding="utf-8")
                        result = _capture_result(out_dir, code, pair, side, run_id,
                            plan_sha, deployment_receipt, g_receipt, h_receipt)
                        result.update({"deployment_receipt_path": str(deploy_out), "rerun_of": rerun_of,
                                       "command": capture_cmd})
                        atomic_write_json(out_dir / "capture_result.json", result)
                    state["captures"][run_id] = {"status": result["status"],
                        "comparison_id": comparison["comparison_id"], "training_root_seed": pair["training_root_seed"],
                        "side": side, "result_path": str((campaign_root / "captures" / run_id / "capture_result.json").resolve()),
                        "physical_measurement_performed": bool(result.get("physical_measurement_performed", False)),
                        "artifact_sha256": pair.get(side + "_artifact_sha256")}
                    state["updated_utc"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
                    atomic_write_json(campaign_state_path, state)
                    _event(campaign_events, {"event": "variant_capture_finished", "run_id": run_id,
                        "comparison_id": comparison["comparison_id"], "training_root_seed": pair["training_root_seed"],
                        "side": side, "status": result["status"], "utc": state["updated_utc"]})
                    total += 1
                    print(json.dumps({"run_id": run_id, "status": result["status"],
                        "comparison_id": comparison["comparison_id"], "root_seed": pair["training_root_seed"], "side": side}))
        state["status"] = "capture_complete_needs_analysis" if state["captures"] and all(
            row.get("status") == "complete_verified" for row in state["captures"].values()) else "partial_or_failed"
        state["physical_measurements_performed"] = any(
            row.get("physical_measurement_performed", False) for row in state["captures"].values())
        state["updated_utc"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
        atomic_write_json(campaign_state_path, state)
    print(json.dumps({"status": state["status"], "new_capture_count": total,
        "campaign_state": str(campaign_state_path), "plan_sha256": plan_sha,
        "physical_measurements_performed": state["physical_measurements_performed"]}, indent=2))
    return 0 if state["status"] == "capture_complete_needs_analysis" else 5


if __name__ == "__main__":
    raise SystemExit(main())
