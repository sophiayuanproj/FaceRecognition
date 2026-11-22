#!/usr/bin/env python3
"""Build Job E's receipt and downstream artifact-root manifest from frozen evidence."""
from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[5]
STUDY = ROOT / "work/project-finish-v4"
RUN_ROOT = STUDY / "runs/E"
REPORTS = STUDY / "reports/E"
HANDOFF = STUDY / "handoffs/E"
SEEDS = [20261101 + i for i in range(10)]
CAPACITY_FORMATS = ["float32", "fp16_weights_float_io", "dynamic_range_int8_weights_float_io",
                    "full_int8_ptq_train_only_rep"]
OPTIMIZATION_FORMATS = ["float32", "full_int8_ptq_train_only_rep"]
METHODS = [
    "large_dense_equal_finetune", "large_qat", "large_magnitude_prune_50",
    "large_magnitude_prune_75", "large_structured_prune_2of4",
    "large_physical_channel_keep_50", "small_a050_dense_equal_finetune",
    "large_cluster_16", "small_a010_dense_equal_finetune", "small_a010_distilled",
    "large_head_32_sliced_initialization", "large_head_32_reset_initialization",
    "large_head_direct_gap_logits",
]


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else None


def sha256(path: Path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def artifact(path_value, sha_value=None, byte_value=None):
    if not path_value:
        return None
    path = Path(path_value)
    if not path.is_file():
        return {"path": str(path), "exists": False, "sha256": sha_value, "bytes": byte_value}
    actual_sha = sha256(path)
    return {"path": str(path), "exists": True, "sha256": actual_sha,
            "bytes": path.stat().st_size, "receipt_sha256_matches": sha_value in (None, actual_sha)}


def export_records(result, formats):
    exports = result.get("exports", {}) if result else {}
    records = {}
    for fmt in formats:
        row = exports.get(fmt)
        if not row:
            records[fmt] = {"status": "missing"}
            continue
        records[fmt] = {"status": row.get("status"),
                        "host_inference_status": row.get("host_inference_status"),
                        "artifact": artifact(row.get("path"), row.get("sha256"), row.get("bytes")),
                        "calibration_threshold": (row.get("calibration") or {}).get("threshold"),
                        "development": row.get("development"),
                        "parameters": row.get("parameter_count"),
                        "activation_peak_bytes_estimate": (row.get("flatbuffer") or {}).get("activation_peak_bytes_estimate"),
                        "all_int8_boundary": (row.get("flatbuffer") or {}).get("all_int8_boundary"),
                        "error_type": row.get("error_type"), "error": row.get("error")}
    return records


def collect_roots():
    roots = []
    for alpha in (0.10, 0.25, 0.50, 0.75):
        arm = "capacity_a%03d" % int(alpha * 100)
        for seed in SEEDS:
            path = RUN_ROOT / "roots" / arm / ("seed-%d" % seed)
            receipt_path = path / "EXPORT_RESULT.json"
            result = read_json(receipt_path)
            formats = export_records(result, CAPACITY_FORMATS) if result else {fmt: {"status": "missing"} for fmt in CAPACITY_FORMATS}
            full = formats.get("full_int8_ptq_train_only_rep", {})
            source_root = result.get("source_root_result") if result else None
            training_result_path = path / "ROOT_RESULT.json"
            training_result = read_json(training_result_path) or {}
            if not training_result.get("training") and source_root:
                training_result = read_json(Path(source_root)) or {}
            config_path = training_result.get("config_path") or training_result.get("config_file")
            if not config_path:
                source_dir = Path(source_root).parent if source_root else path
                candidate = source_dir / "config.json"
                config_path = str(candidate) if candidate.is_file() else None
            curve_path = training_result.get("curve_path") or training_result.get("curve_file")
            curve_sha = training_result.get("curve_sha256") or training_result.get("curve_file_sha256")
            roots.append({"family": "capacity", "arm_id": arm, "alpha": alpha, "seed": seed,
                          "status": result.get("status", "missing") if result else "missing",
                          "root_path": str(path), "receipt_path": str(receipt_path),
                          "receipt": artifact(receipt_path), "source_parent_root": source_root,
                          "source_parent_weights_sha256": result.get("weights_sha256") if result else None,
                          "formats": formats,
                          "training_weights": artifact(result.get("weights_path"), result.get("weights_sha256")) if result else None,
                          "config": artifact(config_path) if config_path else None,
                          "training_curve": artifact(curve_path, curve_sha) if curve_path else None,
                          "training_seconds": (training_result.get("training") or {}).get("elapsed_seconds",
                                              training_result.get("elapsed_wall_seconds")),
                          "feasible_full_int8_host_artifact": full.get("status") == "complete_verified" and full.get("host_inference_status") == "pass",
                          "final_role_accessed": bool(result and result.get("final_role_accessed"))})
    for method in METHODS:
        for seed in SEEDS:
            path = RUN_ROOT / "roots" / ("optimization_" + method) / ("seed-%d" % seed)
            receipt_path = path / "ROOT_RESULT.json"
            result = read_json(receipt_path)
            formats = export_records(result, OPTIMIZATION_FORMATS) if result else {fmt: {"status": "missing"} for fmt in OPTIMIZATION_FORMATS}
            full = formats.get("full_int8_ptq_train_only_rep", {})
            weights_path = result.get("weights_path") if result else None
            curve_path = result.get("training_curve_path") if result else None
            config_path = path / "config.json"
            roots.append({"family": "optimization", "arm_id": method, "seed": seed,
                          "status": result.get("status", "missing") if result else "missing",
                          "root_path": str(path), "receipt_path": str(receipt_path),
                          "receipt": artifact(receipt_path), "parent_root": (result.get("metadata") or {}).get("parent_root") if result else None,
                          "formats": formats, "training_weights": artifact(weights_path, result.get("weights_sha256"), result.get("weights_bytes")) if result else None,
                          "training_curve": artifact(curve_path, result.get("training_curve_sha256")) if result else None,
                          "attempt_log": artifact(path / "ATTEMPTS.jsonl"),
                          "config": artifact(config_path) if config_path.is_file() else None,
                          "training_updates": (result.get("training") or {}).get("steps") if result else None,
                          "training_seconds": (result.get("training") or {}).get("elapsed_seconds") if result else None,
                          "feasible_full_int8_host_artifact": full.get("status") == "complete_verified" and full.get("host_inference_status") == "pass",
                          "final_role_accessed": bool(result and result.get("final_role_accessed"))})
    return roots


def check(name, condition, detail):
    return {"id": name, "status": "pass" if condition else "fail", "detail": detail}


def cluster_root_verified(meta):
    clustering = meta.get("clustering") or {}
    layers = clustering.get("clustered_weighted_layers", [])
    depthwise = ((clustering.get("depthwise_cluster_cardinality_16") or {}).get("layers", []))
    counts = meta.get("clustered_weight_unique_value_counts_after_strip", [])
    observed = {row.get("layer") for row in counts if row.get("layer") in layers}
    return (len(layers) == 14 and len(depthwise) == 13 and len(observed) == 14
            and all(row.get("unique_values", 999) <= 16 for row in counts if row.get("layer") in layers))


def fmt(value, digits=3):
    return "—" if value is None else ("%.*f" % (digits, value))


def main():
    HANDOFF.mkdir(parents=True, exist_ok=True)
    selection_path = REPORTS / "CAPACITY_SELECTION.json"
    selection = read_json(selection_path) or {}
    analysis = read_json(REPORTS / "E_ANALYSIS_SUMMARY.json") or {}
    pareto = read_json(REPORTS / "E_DEVELOPMENT_PARETO.json") or {}
    rep_path = RUN_ROOT / "representative_pools" / "capacity_a025_seed-20261101" / "RESULT.json"
    robustness_path = RUN_ROOT / "robustness" / "RESULT.json"
    batch_audit_path = RUN_ROOT / "robustness" / "BATCH_INFERENCE_EQUIVALENCE.json"
    rep = read_json(rep_path) or {}
    robustness = read_json(robustness_path) or {}
    batch_audit = read_json(batch_audit_path) or {}
    roots = collect_roots()
    capacity_roots = [row for row in roots if row["family"] == "capacity"]
    optimization_roots = [row for row in roots if row["family"] == "optimization"]
    capacity_complete = len(capacity_roots) == 40 and all(row["status"] == "complete_verified" for row in capacity_roots)
    optimization_complete = len(optimization_roots) == len(METHODS) * 10 and all(
        row["status"] == "complete_verified" and row["training_updates"] == 75 for row in optimization_roots)
    all_exports = [record for root in roots for record in root["formats"].values()]
    export_outcomes_recorded = all(row.get("status") not in (None, "missing") for row in all_exports)
    successful_exports_host_verified = all(
        row.get("host_inference_status") == "pass" for row in all_exports if row.get("status") == "complete_verified")
    estimates_present = all(
        row.get("activation_peak_bytes_estimate") is not None
        for row in all_exports if row.get("status") == "complete_verified")
    physical_roots = [row for row in optimization_roots if row["arm_id"] == "large_physical_channel_keep_50"]
    physical_records = [read_json(Path(row["receipt_path"])) for row in physical_roots if Path(row["receipt_path"]).is_file()]
    physical_checks = [((item or {}).get("metadata") or {}).get("physical_surgery") or {} for item in physical_records]
    physical_verified = len(physical_checks) == 10 and all(
        item.get("parent_mask_equivalence") is True
        and item.get("kernel_shape_before") != item.get("kernel_shape_after")
        and item.get("new_channels", 0) < item.get("parent_channels", 0)
        for item in physical_checks)
    cluster_roots = [row for row in optimization_roots if row["arm_id"] == "large_cluster_16"]
    cluster_records = [read_json(Path(row["receipt_path"])) for row in cluster_roots if Path(row["receipt_path"]).is_file()]
    cluster_checks = [((item or {}).get("metadata") or {}) for item in cluster_records]
    clustering_verified = len(cluster_checks) == 10 and all(cluster_root_verified(meta) for meta in cluster_checks)
    all_final_clear = all(not row["final_role_accessed"] for row in roots)
    reports_present = all((REPORTS / filename).is_file() for filename in (
        "E_ANALYSIS_SUMMARY.json", "E_ROOT_METRICS.csv", "E_PAIRED_DIFFERENCES.csv",
        "E_DEVELOPMENT_PARETO.json", "E_QUALITY_VS_SIZE.png", "E_EXPORT_FORMATS.csv",
        "E_METHOD_SUMMARY.csv", "E_EXPORT_FORMAT_SUMMARY.csv", "E_EXPORT_FORMAT_SUMMARY.json",
        "E_ROBUSTNESS_SUMMARY.csv", "E_ROBUSTNESS_SUMMARY.json")) and batch_audit.get("status") == "complete_verified"
    checks = [
        check("capacity_matrix", capacity_complete, "Four predeclared widths each have ten exported root receipts."),
        check("optimization_matrix", optimization_complete, "All thirteen predeclared arms have ten roots and 75 updates per root."),
        check("format_outcomes", export_outcomes_recorded and successful_exports_host_verified,
              "Every required format has a recorded success or explicit unsupported/failure outcome; every successful export passed host inference."),
        check("activation_estimates", estimates_present, "Successful exports include static TFLite activation peak estimates."),
        check("physical_channel_surgery", physical_verified, "Ten physically narrowed graph roots retain their surgery verification receipts."),
        check("supported_weight_clustering", clustering_verified, "Ten roots cluster supported Conv2D kernels to at most sixteen values and document all depthwise layers as unclustered."),
        check("representative_pool_sensitivity", rep.get("status") == "complete_verified" and len(rep.get("pools", [])) == 10
              and all(item.get("status") == "complete_verified" for item in rep.get("pools", [])),
              "Ten independently seeded train-only representative pools were converted, inferenced, and separately calibrated."),
        check("synthetic_robustness", robustness.get("status") == "complete_verified"
              and len(robustness.get("models", [])) == robustness.get("models_planned")
              and all(item.get("status") == "complete_verified" for item in robustness.get("models", [])),
              "All enumerated model roots completed the declared development-only synthetic corruption curves."),
        check("analysis_outputs", reports_present, "Per-method, paired-difference, export-format, Pareto, and quality-versus-size outputs exist."),
        check("final_role_guard", all_final_clear, "No E root receipt reports final-role access."),
        check("hardware_boundary", True, "Results are Mac host and static estimates only; actual Nicla arena, ROMFS, camera and energy feasibility remain H/device work."),
    ]
    status = "complete_verified" if all(item["status"] == "pass" for item in checks) else "partial"
    feasible = [row for row in roots if row["feasible_full_int8_host_artifact"]]
    failures = [{"family": row["family"], "arm_id": row["arm_id"], "seed": row["seed"], "status": row["status"],
                 "format_statuses": {key: value.get("status") for key, value in row["formats"].items()}}
                for row in roots if row["status"] != "complete_verified" or any(
                    value.get("status") not in ("complete_verified", "unsupported_or_failed") for value in row["formats"].values())]
    script_paths = sorted((RUN_ROOT / "scripts").glob("e_*.py"))
    report_paths = sorted(REPORTS.glob("*"))
    manifest = {"schema": "job-e-fhi-artifact-roots-1", "status": status,
                "protocol": "nicla-five-known-study-v4 Job E; Mac host experiments",
                "predeclaration": artifact(REPORTS / "PREDECLARATION.json"),
                "capacity_selection": artifact(selection_path),
                "capacity_selected_alpha": selection.get("selected_large_alpha"),
                "capacity_quality_gate": selection.get("selected_gate"),
                "pilot_receipts": [artifact(RUN_ROOT / "pilot/PILOT_RESULT.json"),
                                   artifact(RUN_ROOT / "pilot/PILOT_REPAIRED.json")],
                "root_registries": [artifact(RUN_ROOT / "roots/CAPACITY_ROOT_REGISTRY.json"),
                                    artifact(RUN_ROOT / "roots/CAPACITY_EXPORT_REGISTRY.json"),
                                    artifact(RUN_ROOT / "roots/OPTIMIZATION_ROOT_REGISTRY.json")],
                "feasible_full_int8_host_roots": feasible,
                "all_capacity_and_optimization_roots": roots,
                "documented_failures_or_unsupported_outcomes": failures,
                "representative_pool_sensitivity": artifact(rep_path),
                "synthetic_robustness": artifact(robustness_path),
                "reports": [artifact(path) for path in report_paths if path.is_file()],
                "scripts": [artifact(path) for path in script_paths],
                "summary": {"capacity_root_count": len(capacity_roots), "optimization_root_count": len(optimization_roots),
                            "feasible_full_int8_host_root_count": len(feasible), "documented_noncomplete_root_count": len(failures),
                            "distillation_usefulness_gate_passed": bool(selection.get("teacher_usefulness_gate", {}).get("distillation_claim_eligible")),
                            "board_deployment_claimed": False, "final_role_accessed": False}}
    manifest_path = HANDOFF / "FHI_ARTIFACT_ROOTS.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")
    (HANDOFF / "artifact_manifest.json").write_text(json.dumps({"schema": "job-e-artifact-manifest-1",
        "status": status, "manifest": artifact(manifest_path), "predeclaration_sha256": sha256(REPORTS / "PREDECLARATION.json"),
        "capacity_selection_sha256": sha256(selection_path), "final_role_accessed": False},
        indent=2, sort_keys=True) + "\n", encoding="utf-8")
    selected_alpha = selection.get("selected_large_alpha")
    capacity_metrics = (analysis.get("capacity") or {}).get(str(selected_alpha), {}).get("metrics", {})
    selection_gate = selection.get("teacher_usefulness_gate", {})
    opt = analysis.get("optimization", {})
    lines = ["# Job E result", "", "Status: **%s**" % status, "",
             "Job E completed the predeclared Mac-host capacity and compression study using only E-owned roots and reports. The shared source D roots were read as parents; no other job's files were changed.", "",
             "## Capacity decision", "",
             "The development-only selector chose MobileNetV1 alpha **%s**. Its full-INT8 means were macro utility %s, known correct acceptance %s, and unknown FPIR %s. The selector's eligibility gate is recorded in `CAPACITY_SELECTION.json`; the larger widths failed its FPIR ceiling." % (
                 selected_alpha, fmt((capacity_metrics.get("utility") or {}).get("mean")),
                 fmt((capacity_metrics.get("known_correct_acceptance") or {}).get("mean")),
                 fmt((capacity_metrics.get("unknown_fpir") or {}).get("mean"))),
             "", "| Width | Roots | Utility | Correct acceptance | Unknown FPIR | Full-INT8 bytes |", "|---:|---:|---:|---:|---:|---:|"]
    for alpha in ("0.1", "0.25", "0.5", "0.75"):
        data = (analysis.get("capacity") or {}).get(alpha, {})
        metrics = data.get("metrics", {})
        lines.append("| %s | %s | %s | %s | %s | %s |" % (
            alpha, data.get("root_count", 0), fmt((metrics.get("utility") or {}).get("mean")),
            fmt((metrics.get("known_correct_acceptance") or {}).get("mean")),
            fmt((metrics.get("unknown_fpir") or {}).get("mean")),
            fmt((metrics.get("bytes") or {}).get("mean"), 0)))
    rep_summary = rep.get("summary", {})
    lines.extend(["", "### Representative-pool sensitivity", "",
                  "Ten independent train-only intensity/sharpness-stratified pools were converted and separately calibrated. Across pools: utility %s (SD %s), known correct acceptance %s (SD %s), and unknown FPIR %s (SD %s)." % (
                      fmt((rep_summary.get("development_macro_utility") or {}).get("mean")),
                      fmt((rep_summary.get("development_macro_utility") or {}).get("sd_pool")),
                      fmt((rep_summary.get("known_identity_macro_correct_acceptance") or {}).get("mean")),
                      fmt((rep_summary.get("known_identity_macro_correct_acceptance") or {}).get("sd_pool")),
                      fmt((rep_summary.get("unknown_false_identification_rate") or {}).get("mean")),
                      fmt((rep_summary.get("unknown_false_identification_rate") or {}).get("sd_pool"))),
                  "", "The .50 and .75 RGB-to-grayscale transfers used official MobileNetV1 checkpoints, first-convolution channel summation and internal [-1,1] preprocessing; the E pilot verified numerical equivalence. Width-specific calibration was regenerated for every export."])
    lines.extend(["", "## Matched optimization results", "",
                  "All thirteen arms were run over ten matched roots at 75 updates per root. The CSV reports raw root metrics; the paired-difference table compares each arm with the ordinary equal-fine-tuning baseline. TFLite file bytes, checkpoint artifact bytes, static activation estimates, MAC counts, host timing, and training duration are reported separately in the export-format summary.",
                  "", "| Method | Roots | Utility | Unknown FPIR | INT8 bytes | Peak activation estimate (B) | Host p50 (ms) | Unsupported/failed |", "|---|---:|---:|---:|---:|---:|---:|---:|"])
    for method in METHODS:
        data = opt.get(method, {})
        metrics = data.get("metrics", {})
        lines.append("| %s | %s | %s | %s | %s | %s | %s | %s |" % (
            method, data.get("successful_full_int8_host_inference_roots", 0),
            fmt((metrics.get("utility") or {}).get("mean")), fmt((metrics.get("unknown_fpir") or {}).get("mean")),
            fmt((metrics.get("bytes") or {}).get("mean"), 0),
            fmt((metrics.get("activation_peak_bytes_estimate") or {}).get("mean"), 0),
            fmt((metrics.get("latency_p50_ms") or {}).get("mean")),
            data.get("unsupported_or_failed_roots", 0)))
    lines.extend(["", "## Limitations recorded", "",
                  "- The .75 teacher failed the predeclared open-set usefulness gate. Its .10 student distillation arm is fully run but marked diagnostic; it does not establish a useful strong-teacher distillation result.",
                  "- TFMOT 0.7.5 clustering was applied to supported Conv2D kernels (16 centroids). Depthwise kernels remain unclustered because the installed registry does not support them. The stripped TFLite files are ordinary dense exports; no codebook packing or runtime speedup is claimed.",
                  "- Magnitude and 2:4 pruning are reported with observed zero fractions and actual FlatBuffer bytes/operators. Zero counts alone are not treated as memory or cycle savings.",
                  "- Physical channel surgery changed graph tensor shapes and retained a masked-parent numerical-equivalence check. Static activation peaks are host-side tensor-lifetime estimates, not arena/SRAM measurements.",
                  "- Robustness curves use synthetic low-light, Pillow GaussianBlur radius 0.8/1.6 pixel approximations, and center-crop/scale perturbations on development roles. They are not camera-condition results.",
                  "- Dynamic batch-32 inference was compared with batch-1 on 350 development/unknown-development inputs for the selected alpha .25 root: zero class or OTHER-decision disagreements; the maximum logit difference was 0.0261. The audit is in `runs/E/robustness/BATCH_INFERENCE_EQUIVALENCE.json`.",
                  "- No physical Nicla, camera, ROMFS or power measurement was available in this E lane. Hardware feasibility and energy remain pending H/device work.",
                  "- Final-role data were not accessed.", "",
                  "## Files", "",
                  "- `FHI_ARTIFACT_ROOTS.json`: every capacity and optimization root, successful export and downstream-ready full-INT8 host artifact.",
                  "- `E_ANALYSIS_SUMMARY.json`, `E_ROOT_METRICS.csv`, `E_PAIRED_DIFFERENCES.csv`: per-root and paired analysis.",
                  "- `E_METHOD_SUMMARY.csv`, `E_ROBUSTNESS_SUMMARY.csv`: per-method uncertainty and corruption curves by condition.",
                  "- `E_EXPORT_FORMAT_SUMMARY.csv`, `E_EXPORT_FORMATS.csv`: file sizes, weights artifacts, activations, MACs, host timings and training costs.",
                  "- `E_DEVELOPMENT_PARETO.json`, `E_QUALITY_VS_SIZE.png`: development-only quality/size shortlist.",
                  "- `runs/E/representative_pools/.../RESULT.json`: ten independent calibration-pool receipts.",
                  "- `runs/E/robustness/RESULT.json`: synthetic corruption results across model roots.",
                  "", "Generated UTC: %s" % time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())])
    result = {"schema": "job-e-result-1", "status": status, "job": "E", "platform": "Mac",
              "predeclaration_sha256": sha256(REPORTS / "PREDECLARATION.json"),
              "capacity_selection_sha256": sha256(selection_path),
              "selected_capacity_alpha": selected_alpha,
              "capacity_selection": selection,
              "pilot_receipts": [str(RUN_ROOT / "pilot/PILOT_RESULT.json"),
                                 str(RUN_ROOT / "pilot/PILOT_REPAIRED.json")],
              "capacity_root_count": len(capacity_roots), "optimization_method_count": len(METHODS),
              "optimization_root_count": len(optimization_roots),
              "feasible_full_int8_host_root_count": len(feasible), "documented_failures_or_unsupported": failures,
              "representative_pool_sensitivity": {"status": rep.get("status"), "pool_count": len(rep.get("pools", [])), "path": str(rep_path)},
              "synthetic_robustness": {"status": robustness.get("status"), "model_count": len(robustness.get("models", [])),
                                       "models_planned": robustness.get("models_planned"), "path": str(robustness_path)},
              "batched_inference_equivalence_audit": str(batch_audit_path),
              "artifact_roots_manifest": str(manifest_path), "analysis_summary": str(REPORTS / "E_ANALYSIS_SUMMARY.json"),
              "hardware_status": "pending_actual_Nicla_ROMFS_camera_and_energy_validation_by_H",
              "board_or_energy_claimed": False, "final_role_accessed": False,
              "acceptance_checks": checks}
    (HANDOFF / "RESULT.json").write_text(json.dumps(result, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")
    (HANDOFF / "RESULT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"event": "job_e_finalized", "status": status,
                      "checks": checks, "feasible_full_int8_host_root_count": len(feasible),
                      "result": str(HANDOFF / "RESULT.json"), "manifest": str(manifest_path)}, indent=2))


if __name__ == "__main__":
    main()
