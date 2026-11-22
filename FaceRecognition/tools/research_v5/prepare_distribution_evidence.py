#!/usr/bin/env python3
"""Create pseudonymous, no-image evidence derivatives for the Windows pack."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys

APP = Path(__file__).resolve().parents[2]
WORKSPACE = APP.parent
if str(APP) not in sys.path:
    sys.path.insert(0, str(APP))
from nicla_study.v5_evidence import derive_metrics_from_predictions, strict_load


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def anon(value: str) -> str:
    return "person-" + hashlib.sha256(("nicla-v5-transfer-v1:" + value).encode("utf-8")).hexdigest()[:20]


def main() -> int:
    study = WORKSPACE / "work/project-research-v5"
    source = study / "runs/camera_adapt_task48"
    dist = study / "distribution_evidence"
    dist.mkdir(parents=True, exist_ok=True)
    source_roster_path = study / "data/LFW_ROLE_FREEZE.json"
    adaptation_path = study / "data/OPTICAL_ADAPTATION_FREEZE.json"
    source_roster = strict_load(source_roster_path)
    adaptation = strict_load(adaptation_path)
    roster = {
        "schema": "v5-shareable-pseudonymous-roster-1",
        "source_manifest_sha256": sha(source_roster_path),
        "selection_rule": source_roster["selection_rule"],
        "future_final_used": False,
        "known_labels": [
            {"label": row["label"], "identity_key": anon(row["source_identity"]), "unique_source_groups": row["unique_groups"]}
            for row in source_roster["roster"]
        ],
        "unknown_identity_counts_by_role": {
            "unknown_calibration": source_roster["unknown_role_identities"]["unknown_calibration"],
            "unknown_development": source_roster["unknown_role_identities"]["unknown_development"],
            "unknown_future_final": source_roster["unknown_role_identities"]["unknown_future_final"],
        },
        "identity_name_mapping_included": False,
    }
    roster_path = dist / "ROSTER_PSEUDONYMOUS.json"
    roster_path.write_text(json.dumps(roster, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")

    role_binding = {
        "schema": "v5-shareable-data-role-binding-1",
        "source_dataset_roles_sha256": sha(study / "data/manifests/lfw_roles.jsonl"),
        "source_freeze_sha256": sha(source_roster_path),
        "camera_parent_role_freeze_sha256": sha(adaptation_path),
        "camera_parent_manifest_sha256": adaptation["manifest_sha256"],
        "source_pre_role_manifest_sha256": source_roster["manifest_sha256"],
        "role_counts": source_roster["role_counts"],
        "unknown_identity_counts_by_role": roster["unknown_identity_counts_by_role"],
        "future_final_used": False,
        "future_final_rows_or_image_paths_included": False,
        "identity_name_mapping_included": False,
        "note": "Exact source manifests remain Mac-side; this portable record binds their frozen hashes and publishes pseudonymous role counts only.",
    }
    role_path = dist / "ROLE_BINDING.json"
    role_path.write_text(json.dumps(role_binding, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")

    manifest_rows = []
    count = 0
    for crop in (48, 64, 80, 96):
        for seed in range(53001, 53011):
            input_path = source / f"crop_{crop}" / str(seed) / "PREDICTIONS.jsonl"
            if not input_path.is_file():
                raise FileNotFoundError(input_path)
            output_path = dist / "camera_adapt_task48" / f"crop_{crop}" / str(seed) / "PREDICTIONS.jsonl"
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_rows = []
            with input_path.open(encoding="utf-8") as stream:
                for line in stream:
                    if not line.strip():
                        continue
                    row = json.loads(line, parse_constant=lambda x: (_ for _ in ()).throw(ValueError("Nonfinite JSON " + x)))
                    if "source_identity" in row:
                        row["source_identity"] = anon(str(row["source_identity"]))
                    output_rows.append(row)
            output_path.write_text("".join(json.dumps(r, sort_keys=True, allow_nan=False) + "\n" for r in output_rows), encoding="utf-8")
            source_metrics = derive_metrics_from_predictions(WORKSPACE, input_path.relative_to(WORKSPACE).as_posix())
            output_metrics = derive_metrics_from_predictions(WORKSPACE, output_path.relative_to(WORKSPACE).as_posix())
            for field in ("known_n", "known_correct_acceptance", "known_coverage", "unknown_groups", "unknown_false_groups", "unknown_group_upper95", "per_person"):
                if source_metrics[field] != output_metrics[field]:
                    raise ValueError(f"Pseudonymization changed {field} in crop {crop}, root {seed}")
            count += len(output_rows)
            manifest_rows.append({
                "crop_px": crop,
                "training_root": seed,
                "rows": len(output_rows),
                "source_sha256_mac_local": sha(input_path),
                "shareable_sha256": sha(output_path),
                "derived_metrics_unchanged": True,
            })
    pred_manifest = {
        "schema": "v5-shareable-prediction-derivative-1",
        "source_root": "work/project-research-v5/runs/camera_adapt_task48",
        "source_identities": "replaced with stable SHA-256 pseudonyms; no lookup included",
        "rows": count,
        "files": manifest_rows,
        "all_row_derived_metrics_equal_source": len(manifest_rows) == 40,
    }
    (dist / "PREDICTIONS_MANIFEST.json").write_text(json.dumps(pred_manifest, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")

    raw_refs = []
    for name in ("LATEST_SESSION.txt", "LATEST_ADAPTATION.txt", "LATEST_NEGATIVE_CALIBRATION.txt"):
        ptr = study / "runs/optical" / name
        session = ptr.read_text(encoding="utf-8").strip()
        receipt = study / "runs/optical" / session / "RECEIPTS.json"
        if not receipt.is_file():
            # Development sessions store the receipt inside analysis when present.
            receipt = study / "runs/optical" / session / "analysis/RESULT.json"
        if receipt.is_file():
            raw_refs.append({"session_alias_file": name, "session_id": session, "local_receipt_relative_path": receipt.relative_to(WORKSPACE).as_posix(), "local_receipt_sha256": sha(receipt)})
    optical = {
        "schema": "v5-shareable-optical-capture-summary-1",
        "actual_capture_count": 255,
        "physical_camera_setups": 1,
        "development": {"frames": 100, "software_reset_blocks": 10, "known_presentations": 50, "unknown_identity_presentations": 30, "checker_controls": 10, "blank_controls": 10},
        "role_preserving_adaptation": {"frames": 135, "software_reset_blocks": 20, "known_train": 60, "other_train": 20, "known_threshold_calibration": 25, "unknown_calibration": 30},
        "negative_calibration": {"frames": 20, "software_reset_blocks": 10, "blank": 10, "checker": 10},
        "private_receipt_references": raw_refs,
        "source_identity_names_or_image_paths_included": False,
        "raw_camera_frames_included": False,
        "final_role_opened": False,
        "limitations": ["All 255 frames came from one taped display setup.", "Repeated source identities and captures are not independent people/environments.", "Raw RGB565 frames and room/screen views remain Mac-private."],
    }
    (dist / "OPTICAL_CAPTURE_SUMMARY.json").write_text(json.dumps(optical, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")

    runtime = {
        "schema": "v5-shareable-runtime-contract-1",
        "board": "Arduino Nicla Vision / STM32H747",
        "openmv": "5.0.1",
        "micropython": "1.28.0-64",
        "camera_sensor": "GC2145",
        "encoder_model_relative_path": "work/project-research-v5/runs/operator_fused_encoder48/int8.tflite",
        "encoder_model_sha256": sha(source.parent / "operator_fused_encoder48/int8.tflite"),
        "parity_backend": "TensorFlow 2.10.1 BUILTIN_REF on Mac; exact 128 signed output values on 10 synthetic board fixtures",
        "device_lut_sha256": "1029f69ec63904cdc69ccb9a4f318a9558ec1fb085a2f7e23463d7628a642ec6",
        "input_scope": "48x48 RGB ROI only; Python LUT was exact but measured about 19.8ms on board",
        "recognition_head": "host-only six-class float32 linear softmax, 774 trainable parameters; not a Nicla deployable classifier",
        "source_firmware_release_manifest": "FaceRecognition/config/firmware-release.json",
        "standalone_boot_or_rollback_qualified": False,
    }
    (dist / "runtime_contract.json").write_text(json.dumps(runtime, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({"status": "pass", "prediction_files": len(manifest_rows), "prediction_rows": count,
                      "all_metrics_unchanged": pred_manifest["all_row_derived_metrics_equal_source"],
                      "pseudonymous_roster": len(roster["known_labels"]), "optical_frames": optical["actual_capture_count"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
