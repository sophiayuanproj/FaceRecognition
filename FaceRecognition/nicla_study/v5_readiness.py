"""Derive the bounded v5 Windows handoff gate from bound evidence rows."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

from .v5_evidence import (
    _upper_binomial_95,
    binding,
    derive_metrics_from_predictions,
    strict_load,
    verify_bindings,
)


SEEDS = tuple(range(53001, 53011))
CROPS = (48, 64, 80, 96)


def _sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _rel(root: Path, path: str) -> Path:
    p = (root / path).resolve()
    p.relative_to(root.resolve())
    if not p.is_file():
        raise FileNotFoundError(path)
    return p


def build(root: Path, output: Path | None = None) -> dict:
    root = root.resolve()
    research = root / "work/project-research-v5"
    dist = research / "distribution_evidence"
    gates = strict_load(research / "config/USEFULNESS_GATES.json")
    protocol = strict_load(research / "runs/camera_adapt_task48/PROTOCOL.json")
    summary = strict_load(research / "runs/camera_adapt_task48/SUMMARY.json")
    parity = strict_load(research / "reports/M5/rgb48_encoder_parity/RESULT.json")
    distribution_manifest = strict_load(dist / "PREDICTIONS_MANIFEST.json")
    distribution_rows = {
        (int(row["crop_px"]), int(row["training_root"])): row
        for row in distribution_manifest.get("files", [])
    }
    if len(distribution_rows) != 40 or distribution_manifest.get("all_row_derived_metrics_equal_source") is not True:
        raise ValueError("Pseudonymous prediction manifest is incomplete or lacks Mac-side row-metric comparison")

    predictions: dict[tuple[int, int], str] = {}
    candidate_metrics: dict[str, list[dict]] = {str(crop): [] for crop in CROPS}
    calibration_scores: dict[str, list[float]] = {str(crop): [] for crop in CROPS}
    original_comparison_count = 0
    for crop in CROPS:
        for seed in SEEDS:
            original_rel = (
                f"work/project-research-v5/runs/camera_adapt_task48/crop_{crop}/{seed}/PREDICTIONS.jsonl"
            )
            share_rel = (
                f"work/project-research-v5/distribution_evidence/camera_adapt_task48/"
                f"crop_{crop}/{seed}/PREDICTIONS.jsonl"
            )
            share_path = _rel(root, share_rel)
            distributed = derive_metrics_from_predictions(root, share_rel)
            manifest_row = distribution_rows.get((crop, seed))
            if not manifest_row or manifest_row.get("derived_metrics_unchanged") is not True:
                raise ValueError(f"Missing source-comparison receipt for crop {crop}, seed {seed}")
            if _sha(share_path) != manifest_row.get("shareable_sha256"):
                raise ValueError(f"Pseudonymous prediction hash differs from distribution manifest: crop {crop}, seed {seed}")
            original_path = root / original_rel
            if original_path.is_file():
                original = derive_metrics_from_predictions(root, original_rel)
                comparable = (
                    "known_n", "known_raw_top1", "known_correct_acceptance", "known_coverage",
                    "known_wrong_accepted", "unknown_n", "unknown_false_accepts", "unknown_groups",
                    "unknown_false_groups", "unknown_group_upper95", "known_calibration_n",
                    "known_calibration_correct_acceptance", "unknown_calibration_n",
                    "unknown_calibration_false_groups", "per_person", "response_observations_n",
                    "response_p95_ms",
                )
                for field in comparable:
                    if original[field] != distributed[field]:
                        raise ValueError(f"Distribution row metrics changed for crop {crop}, seed {seed}: {field}")
            original_comparison_count += 1
            complete_rel = (
                f"work/project-research-v5/runs/camera_adapt_task48/crop_{crop}/{seed}/COMPLETE.json"
            )
            complete = strict_load(_rel(root, complete_rel))
            if complete.get("actual_gradient_training") is not True or complete.get("future_final_used") is not False:
                raise ValueError(f"Training root {crop}/{seed} is incomplete or has future-final contamination")
            if distributed["known_n"] != 50 or distributed["unknown_groups"] != 30:
                raise ValueError(f"Unexpected development composition in {crop}/{seed}")
            if distributed["known_calibration_n"] != 25 or distributed["unknown_calibration_n"] != 30:
                raise ValueError(f"Unexpected calibration composition in {crop}/{seed}")
            predictions[(crop, seed)] = share_rel
            candidate_metrics[str(crop)].append(distributed)
            calibration_scores[str(crop)].append(float(distributed["known_calibration_correct_acceptance"]))

    # Predeclared selection is calibration known-correct acceptance, then smaller crop on ties.
    selected_crop = max(CROPS, key=lambda c: (sum(calibration_scores[str(c)]) / len(SEEDS), -c))
    selected = candidate_metrics[str(selected_crop)]
    mean_correct = sum(m["known_correct_acceptance"] for m in selected) / len(selected)
    mean_coverage = sum(m["known_coverage"] for m in selected) / len(selected)
    max_unknown_upper = max(m["unknown_group_upper95"] for m in selected)
    false_group_counts = [m["unknown_false_groups"] for m in selected]
    response_values = [m["response_p95_ms"] for m in selected if m["response_p95_ms"] is not None]
    response_p95 = max(response_values) if len(response_values) == len(selected) else None

    encoder_rel = "work/project-research-v5/runs/operator_fused_encoder48/int8.tflite"
    encoder = _rel(root, encoder_rel)
    if _sha(encoder) != parity["model_sha256"]:
        raise ValueError("Exact-parity model hash does not match the bound model artifact")
    parity_dir = research / "reports/M5/rgb48_encoder_parity"
    if parity.get("status") != "verified_exact" or parity.get("input_exact_count") != 10 or parity.get("output_exact_count") != 10:
        raise ValueError("48px parity receipt does not report the required exact fixtures")
    for key, filename in (("device_stage1_sha256", "DEVICE_STAGE1.py"), ("device_stage2_sha256", "DEVICE_STAGE2.py")):
        if _sha(parity_dir / filename) != parity[key]:
            raise ValueError(f"48px parity stage hash mismatch: {filename}")
    if parity.get("output_lsb_differences", 0) != 0 and any(
        row.get("output_lsb_differences") != 0 for row in parity.get("results", [])
    ):
        raise ValueError("48px parity has output differences")

    wheel_dir = research / "handoffs/windows_wheels"
    wheels = []
    for wheel in sorted(wheel_dir.glob("*.whl")):
        wheels.append({"path": wheel.relative_to(root).as_posix(), "bytes": wheel.stat().st_size, "sha256": _sha(wheel)})
    if len(wheels) != 6:
        raise ValueError("Expected six pinned Windows CPython 3.12 wheels")

    bind_rows = [
        binding(root, "FaceRecognition/nicla_study/v5_camera_task48.py", "source"),
        binding(root, "FaceRecognition/nicla_study/v5_evidence.py", "source"),
        binding(root, "FaceRecognition/nicla_study/v5_readiness.py", "readiness_verifier_source"),
        binding(root, "FaceRecognition/nicla_study/v5_rgb48_parity.py", "source"),
        binding(root, "FaceRecognition/nicla_study/v5_native_rgb.py", "source"),
        binding(root, "FaceRecognition/tools/research_v5/windows_research.py", "windows_command_runtime"),
        binding(root, "FaceRecognition/tools/research_v5/package_v5_return.py", "windows_return_runtime"),
        binding(root, "FaceRecognition/tools/research_v5/build_windows_transfer.py", "transfer_builder_source"),
        binding(root, "FaceRecognition/tools/portable_transfer.py", "transfer_verifier_source"),
        binding(root, "FaceRecognition/tools/windows_v5/Setup-Research.ps1", "windows_setup_source"),
        binding(root, "FaceRecognition/tools/windows_v5/Build-NanoDiagnostic.ps1", "windows_firmware_build_source"),
        binding(root, "work/project-research-v5/02_WINDOWS_RESEARCH_AND_RETURN.md", "windows_runbook"),
        binding(root, encoder_rel, "model"),
        binding(root, "work/project-research-v5/runs/camera_adapt_task48/PROTOCOL.json", "configuration"),
        binding(root, "work/project-research-v5/config/WINDOWS_STUDY_PLAN.json", "configuration"),
        binding(root, "work/project-research-v5/config/USEFULNESS_GATES.json", "configuration"),
        binding(root, "FaceRecognition/requirements-instrument-windows.lock", "windows_runtime_lock"),
        binding(root, "work/project-research-v5/instrument/nano_meter_diagnostic/nano_meter_diagnostic.ino", "commissioning_firmware_source"),
        binding(root, "work/project-research-v5/instrument/PROVENANCE.json", "commissioning_firmware_provenance"),
        binding(root, "work/project-research-v5/instrument/COMMISSIONING_CONFIG_TEMPLATE.json", "commissioning_setup_template"),
        binding(root, "work/project-research-v5/distribution_evidence/ROLE_BINDING.json", "data_roles"),
        binding(root, "work/project-research-v5/distribution_evidence/ROSTER_PSEUDONYMOUS.json", "roster"),
        binding(root, "work/project-research-v5/distribution_evidence/runtime_contract.json", "runtime"),
        binding(root, "FaceRecognition/config/firmware-release.json", "runtime"),
        binding(root, "work/project-research-v5/reports/M5/rgb48_encoder_parity/RESULT.json", "parity_receipt"),
        binding(root, "work/project-research-v5/reports/M5/operator_fused_encoder48_PREFLIGHT.json", "camera_memory_receipt"),
        binding(root, "work/project-research-v5/runs/camera_adapt_task48/SUMMARY.json", "camera_task_summary"),
        binding(root, "work/project-research-v5/distribution_evidence/OPTICAL_CAPTURE_SUMMARY.json", "optical_capture_summary"),
        binding(root, "work/project-research-v5/distribution_evidence/PREDICTIONS_MANIFEST.json", "row_derivative_manifest"),
        binding(root, "work/project-research-v5/distribution_evidence/DETECTION_OPTICAL_SUMMARY.json", "optical_detector_summary"),
    ]
    verify_bindings(root, bind_rows)
    for rel in predictions.values():
        _rel(root, rel)
    roles = strict_load(dist / "ROLE_BINDING.json")
    roster = strict_load(dist / "ROSTER_PSEUDONYMOUS.json")
    runtime = strict_load(dist / "runtime_contract.json")
    if roles.get("future_final_used") is not False or roster.get("future_final_used") is not False:
        raise ValueError("Role/roster file indicates future-final use")

    passed = {
        "ten_complete_gradient_roots_each_crop": original_comparison_count == 40,
        "same_metrics_after_shareable_identity_pseudonymization": original_comparison_count == 40,
        "exact_48px_encoder_input_and_output_parity": parity.get("status") == "verified_exact",
        "ten_48px_input_and_output_fixtures_exact": parity.get("input_exact_count") == 10 and parity.get("output_exact_count") == 10,
        "wheelhouse_closed_and_hashed": len(wheels) == 6,
        "future_final_role_sealed": roles.get("future_final_used") is False and roster.get("future_final_used") is False,
        "known_correct_acceptance_target": mean_correct >= float(gates["known_macro_top1_min"]),
        "known_coverage_target": mean_coverage >= float(gates["known_macro_coverage_min"]),
        "unknown_group_upper95_target": max_unknown_upper <= float(gates["unknown_upper_95_max"]),
        "response_p95_target": response_p95 is not None and response_p95 <= float(gates["response_p95_ms_max"]),
    }
    recognition_qualified = all(passed[k] for k in list(passed)[6:])
    parity_dir_rel = "work/project-research-v5/reports/M5/rgb48_encoder_parity"
    # Directory-level private raw captures stay out of distribution; bind only
    # the sanitized aggregate record that ships with the transfer.
    capture_receipt_rel = "work/project-research-v5/distribution_evidence/OPTICAL_CAPTURE_SUMMARY.json"
    capture_receipt = binding(root, capture_receipt_rel, "shareable_optical_capture_summary")

    record = {
        "schema": "nicla-v5-ready-for-windows-1",
        "generated_by": "FaceRecognition/nicla_study/v5_readiness.py",
        "status": "ready_for_windows_instrument_commissioning_with_recognition_gated",
        "ready_for_windows": True,
        "ready_scope": [
            "Windows CPython 3.12 x64 instrument environment setup and non-invasive device discovery",
            "FNIRSI/Nano commissioning after fresh physical identity and wiring checks",
            "offline parity-fixture demo, raw packet audit, and return-archive preparation",
        ],
        "recognition_qualified": recognition_qualified,
        "candidate_application_deployable": False,
        "ready_for_final_build": False,
        "final_role_opened": False,
        "selection_rule": "maximum mean row-derived known-correct calibration acceptance; smaller crop tie-break",
        "selected_diagnostic_crop_fomo_input_px": selected_crop,
        "selected_candidate_is_not_a_deployable_application": True,
        "row_derived_selected_candidate_metrics": {
            "training_roots": list(SEEDS),
            "known_development_rows_per_root": 50,
            "unknown_independent_groups_per_root": 30,
            "mean_known_correct_acceptance": mean_correct,
            "mean_known_coverage": mean_coverage,
            "unknown_false_groups_per_root": false_group_counts,
            "worst_root_unknown_group_upper95": max_unknown_upper,
            "zero_event_upper95_if_no_false_accepts_of_30": _upper_binomial_95(0, 30),
            "response_p95_ms": response_p95,
            "response_observations_present": bool(response_values),
            "future_final_used": False,
        },
        "encoder_parity_only": {
            "scope": parity.get("scope"),
            "model_sha256": parity.get("model_sha256"),
            "synthetic_fixture_count": parity.get("fixture_count"),
            "exact_input_count": parity.get("input_exact_count"),
            "exact_output_count": parity.get("output_exact_count"),
            "board_software_reset_blocks": 1,
            "median_native_transform_us": parity.get("median_native_transform_us"),
            "median_python_lut_apply_us": parity.get("median_python_lut_apply_us"),
            "median_predict_total_us": parity.get("median_predict_total_us"),
            "sha256": _sha(parity_dir / "RESULT.json"),
        },
        "gate_checks": passed,
        "failed_recognition_targets": [name for name, ok in passed.items() if not ok],
        "bindings": bind_rows,
        "row_prediction_bindings": [
            {"path": rel, "bytes": _rel(root, rel).stat().st_size, "sha256": _sha(_rel(root, rel)), "crop_px": crop, "seed": seed}
            for (crop, seed), rel in sorted(predictions.items())
        ],
        "wheelhouse": wheels,
        "local_mac_private_capture_receipt": capture_receipt,
        "excluded_from_transfer": [
            "raw camera frames, room/screen views, contact sheets, and original ROMFS backup",
            "all face images and source dataset archives with no verified redistribution grant",
            "derived per-image embeddings/feature matrices and local Python environments",
            "future-final data rows and any identity-to-name lookup beyond pseudonymous labels",
        ],
        "external_dependencies": [
            "actual Windows host with 64-bit CPython 3.12 and Python Launcher; CPython 3.14 is incompatible with the supplied cp312 wheels",
            "freshly identified FNIRSI meter, Nano 33 BLE commissioning board, and documented measured supply topology",
            "ten independently started randomized paired physical commissioning blocks and raw return logs",
            "a later useful camera-to-identity application before any deploy, Nicla cost comparison, or final build",
        ],
        "gate_note": "The 30 unknown groups are repeated across training roots and are not 300 independent identities. Even zero false accepts among 30 groups gives a one-sided exact 95% upper bound of about 9.5%, so the 1% team target requires a larger independent unknown roster.",
        "runtime_contract": runtime,
        "readiness_derived_from": [
            "row-level PREDICTIONS.jsonl for all 40 crop/root runs",
            "exact 48px model input/output parity result plus source/stage hashes",
            "six pinned wheel files and their SHA-256 digests",
        ],
    }
    if output:
        output = output.resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        temporary = output.with_suffix(output.suffix + ".tmp")
        temporary.write_text(json.dumps(record, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
        temporary.replace(output)
    return record


def main(argv=None) -> int:
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    record = build(args.workspace, args.output)
    print(json.dumps({
        "status": record["status"],
        "recognition_qualified": record["recognition_qualified"],
        "selected_diagnostic_crop": record["selected_diagnostic_crop_fomo_input_px"],
        "mean_correct_acceptance": record["row_derived_selected_candidate_metrics"]["mean_known_correct_acceptance"],
        "mean_coverage": record["row_derived_selected_candidate_metrics"]["mean_known_coverage"],
        "worst_unknown_upper95": record["row_derived_selected_candidate_metrics"]["worst_root_unknown_group_upper95"],
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
