#!/usr/bin/env python3
"""Summarize matched-root capacity/compression results and create Pareto outputs."""
from __future__ import annotations

import csv
import json
import math
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[5]
STUDY = ROOT / "work/project-finish-v4"
RUN_ROOT = STUDY / "runs/E"
REPORTS = STUDY / "reports/E"

import numpy as np

try:
    import matplotlib.pyplot as plt
except Exception:
    plt = None

SEEDS = [20261101 + i for i in range(10)]
METHODS = [
    "large_dense_equal_finetune", "large_qat", "large_magnitude_prune_50",
    "large_magnitude_prune_75", "large_structured_prune_2of4",
    "large_physical_channel_keep_50", "small_a050_dense_equal_finetune",
    "large_cluster_16", "small_a010_dense_equal_finetune", "small_a010_distilled",
    "large_head_32_sliced_initialization", "large_head_32_reset_initialization",
    "large_head_direct_gap_logits",
]


def recorded_path(value):
    """Resolve original result references inside this clone."""
    normalized = str(value).replace("\\", "/")
    marker = "/work/project-finish-v4/"
    if marker in normalized:
        normalized = "work/project-finish-v4/" + normalized.split(marker, 1)[1]
    path = Path(normalized)
    return path if path.is_absolute() else ROOT / path


def mean_ci(values):
    values = np.asarray(values, dtype=np.float64)
    if not len(values):
        return {"n": 0, "mean": None, "sd_root": None, "ci95_t_low": None, "ci95_t_high": None}
    mean = float(np.mean(values))
    sd = float(np.std(values, ddof=1)) if len(values) > 1 else 0.0
    half = 2.262 * sd / math.sqrt(len(values)) if len(values) > 1 else None
    return {"n": int(len(values)), "mean": mean, "sd_root": sd,
            "ci95_t_low": mean - half if half is not None else None,
            "ci95_t_high": mean + half if half is not None else None}


def read_capacity(alpha):
    arm = "capacity_a%03d" % int(alpha * 100)
    roots = []
    for seed in SEEDS:
        p = RUN_ROOT / "roots" / arm / ("seed-%d" % seed) / "EXPORT_RESULT.json"
        if not p.is_file():
            continue
        result = json.loads(p.read_text(encoding="utf-8"))
        export = result.get("exports", {}).get("full_int8_ptq_train_only_rep", {})
        if result.get("status") != "complete_verified" or export.get("status") != "complete_verified":
            continue
        dev = export["development"]
        roots.append({"seed": seed, "utility": dev["development_macro_utility"],
                      "known_correct_acceptance": dev["known_identity_macro_correct_acceptance"],
                      "unknown_fpir": dev["unknown_false_identification_rate"],
                      "known_identification": dev["known_identity_macro_identification"],
                      "bytes": export["bytes"], "parameters": export["parameter_count"],
                      "macs": export["mac_accounting"]["tflite_operator_macs"],
                      "activation_peak_bytes_estimate": export["flatbuffer"]["activation_peak_bytes_estimate"],
                      "latency_p50_ms": export["host_latency_ms"]["p50"],
                      "threshold": export["calibration"]["threshold"]})
    return roots


def read_methods():
    result = {}
    for method in METHODS:
        roots = []
        for seed in SEEDS:
            p = RUN_ROOT / "roots" / ("optimization_" + method) / ("seed-%d" % seed) / "ROOT_RESULT.json"
            if not p.is_file():
                continue
            row = json.loads(p.read_text(encoding="utf-8"))
            if row.get("status") != "complete_verified":
                roots.append({"seed": seed, "status": row.get("status"), "error": row.get("error")})
                continue
            export = row.get("exports", {}).get("full_int8_ptq_train_only_rep", {})
            dev = export.get("development") or {}
            roots.append({"seed": seed, "status": row.get("status"),
                          "utility": dev.get("development_macro_utility"),
                          "known_correct_acceptance": dev.get("known_identity_macro_correct_acceptance"),
                          "unknown_fpir": dev.get("unknown_false_identification_rate"),
                          "known_identification": dev.get("known_identity_macro_identification"),
                          "bytes": export.get("bytes"), "parameters": export.get("parameter_count"),
                          "macs": (export.get("mac_accounting") or {}).get("tflite_operator_macs"),
                          "activation_peak_bytes_estimate": (export.get("flatbuffer") or {}).get("activation_peak_bytes_estimate"),
                          "latency_p50_ms": (export.get("host_latency_ms") or {}).get("p50"),
                          "threshold": (export.get("calibration") or {}).get("threshold"),
                          "training_seconds": (row.get("training") or {}).get("elapsed_seconds"),
                          "training_updates": (row.get("training") or {}).get("steps"),
                          "metadata": row.get("metadata", {})})
        result[method] = roots
    return result


def read_export_formats():
    rows = []
    for alpha in (0.10, 0.25, 0.50, 0.75):
        arm = "capacity_a%03d" % int(alpha * 100)
        for seed in SEEDS:
            path = RUN_ROOT / "roots" / arm / ("seed-%d" % seed) / "EXPORT_RESULT.json"
            if not path.is_file():
                continue
            result = json.loads(path.read_text(encoding="utf-8"))
            weights_path = result.get("weights_path")
            weights_bytes = Path(weights_path).stat().st_size if weights_path and Path(weights_path).is_file() else None
            training_result_path = path.parent / "ROOT_RESULT.json"
            training_result = json.loads(training_result_path.read_text(encoding="utf-8")) if training_result_path.is_file() else {}
            if not training_result.get("training") and result.get("source_root_result"):
                source_result_path = recorded_path(result["source_root_result"])
                if source_result_path.is_file():
                    training_result = json.loads(source_result_path.read_text(encoding="utf-8"))
            training = training_result.get("training") or {}
            training_seconds = training.get("elapsed_seconds", training_result.get("elapsed_wall_seconds"))
            training_updates = training.get("steps", training.get("steps_run"))
            for format_id, export in result.get("exports", {}).items():
                flatbuffer = export.get("flatbuffer") or {}
                latency = export.get("host_latency_ms") or {}
                rows.append({"family": "capacity", "arm_id": arm, "seed": seed,
                             "format": format_id, "status": export.get("status"),
                             "host_inference_status": export.get("host_inference_status"),
                             "file_bytes": export.get("bytes"), "weights_artifact_bytes": weights_bytes,
                             "parameters": export.get("parameter_count"),
                             "tflite_macs": (export.get("tflite_macs") or {}).get("macs_per_inference"),
                             "activation_peak_bytes_estimate": flatbuffer.get("activation_peak_bytes_estimate"),
                             "all_int8_boundary": flatbuffer.get("all_int8_boundary"),
                             "internal_float_tensor_count": flatbuffer.get("internal_float_tensor_count"),
                             "host_latency_p50_ms": latency.get("p50"), "host_latency_p95_ms": latency.get("p95"),
                             "training_seconds": training_seconds, "training_updates": training_updates})
    for method in METHODS:
        for seed in SEEDS:
            path = RUN_ROOT / "roots" / ("optimization_" + method) / ("seed-%d" % seed) / "ROOT_RESULT.json"
            if not path.is_file():
                continue
            result = json.loads(path.read_text(encoding="utf-8"))
            weights_bytes = result.get("weights_bytes")
            if weights_bytes is None and result.get("weights_path") and recorded_path(result["weights_path"]).is_file():
                weights_bytes = recorded_path(result["weights_path"]).stat().st_size
            for format_id, export in result.get("exports", {}).items():
                flatbuffer = export.get("flatbuffer") or {}
                latency = export.get("host_latency_ms") or {}
                rows.append({"family": "optimization", "arm_id": method, "seed": seed,
                             "format": format_id, "status": export.get("status"),
                             "host_inference_status": export.get("host_inference_status"),
                             "file_bytes": export.get("bytes"), "weights_artifact_bytes": weights_bytes,
                             "parameters": export.get("parameter_count"),
                             "tflite_macs": (export.get("tflite_macs") or {}).get("macs_per_inference"),
                             "activation_peak_bytes_estimate": flatbuffer.get("activation_peak_bytes_estimate"),
                             "all_int8_boundary": flatbuffer.get("all_int8_boundary"),
                             "internal_float_tensor_count": flatbuffer.get("internal_float_tensor_count"),
                             "host_latency_p50_ms": latency.get("p50"), "host_latency_p95_ms": latency.get("p95"),
                             "training_seconds": (result.get("training") or {}).get("elapsed_seconds"),
                             "training_updates": (result.get("training") or {}).get("steps")})
    return rows


def paired_differences(methods, baseline="large_dense_equal_finetune"):
    base = {row["seed"]: row for row in methods.get(baseline, [])
            if row.get("status") == "complete_verified" and row.get("utility") is not None}
    table = []
    metrics = ("utility", "known_correct_acceptance", "unknown_fpir", "known_identification", "bytes", "macs")
    for method, roots in methods.items():
        if method == baseline:
            continue
        values_by_metric = {metric: [] for metric in metrics}
        seeds = []
        for row in roots:
            other = base.get(row.get("seed"))
            if not other or row.get("status") != "complete_verified":
                continue
            seeds.append(row["seed"])
            for metric in metrics:
                if row.get(metric) is not None and other.get(metric) is not None:
                    values_by_metric[metric].append(float(row[metric]) - float(other[metric]))
        if not seeds:
            table.append({"method": method, "baseline": baseline, "paired_roots": 0, "differences": {}})
            continue
        table.append({"method": method, "baseline": baseline, "paired_roots": len(seeds),
                      "root_seeds": seeds,
                      "differences": {metric: mean_ci(values) for metric, values in values_by_metric.items()}})
    return table


def main():
    REPORTS.mkdir(parents=True, exist_ok=True)
    capacity = {str(alpha): read_capacity(alpha) for alpha in (0.10, 0.25, 0.50, 0.75)}
    methods = read_methods()
    export_rows = read_export_formats()
    metric_keys = ("utility", "known_correct_acceptance", "unknown_fpir", "known_identification",
                   "bytes", "parameters", "macs", "activation_peak_bytes_estimate", "latency_p50_ms", "training_seconds")
    summaries = {"schema": "job-e-analysis-summary-1", "capacity": {}, "optimization": {},
                 "failure_counts": {}, "final_role_accessed": False}
    long_rows = []
    for alpha, roots in capacity.items():
        summary = {"root_count": len(roots), "successful_full_int8_host_inference_roots": len(roots),
                   "metrics": {key: mean_ci([row[key] for row in roots if row.get(key) is not None]) for key in metric_keys}}
        summaries["capacity"][alpha] = summary
        for row in roots:
            long_rows.append({"family": "capacity", "arm_id": "capacity_a%03d" % int(float(alpha) * 100), **row})
    for method, roots in methods.items():
        successful = [row for row in roots if row.get("status") == "complete_verified" and row.get("utility") is not None]
        failed = [row for row in roots if row.get("status") not in ("complete_verified", "already_recorded")]
        summary = {"root_count": len(roots), "successful_full_int8_host_inference_roots": len(successful),
                   "unsupported_or_failed_roots": len(failed),
                   "unsupported_or_failed_statuses": [row.get("status") for row in failed],
                   "metrics": {key: mean_ci([row[key] for row in successful if row.get(key) is not None]) for key in metric_keys}}
        summaries["optimization"][method] = summary
        summaries["failure_counts"][method] = {"unsupported_or_failed": len(failed),
                                               "completed": len(successful)}
        for row in roots:
            long_rows.append({"family": "optimization", "arm_id": method, **row})
    paired = paired_differences(methods)
    (REPORTS / "E_ANALYSIS_SUMMARY.json").write_text(json.dumps(summaries, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (REPORTS / "E_PAIRED_DIFFERENCES.json").write_text(json.dumps(paired, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    with (REPORTS / "E_ROOT_METRICS.csv").open("w", encoding="utf-8", newline="") as stream:
        fields = ["family", "arm_id", "seed", "status", "utility", "known_correct_acceptance", "unknown_fpir",
                  "known_identification", "bytes", "parameters", "macs", "activation_peak_bytes_estimate",
                  "latency_p50_ms", "training_seconds", "training_updates", "threshold"]
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(long_rows)
    with (REPORTS / "E_PAIRED_DIFFERENCES.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["method", "baseline", "metric", "paired_roots", "mean_difference", "sd_root", "ci95_t_low", "ci95_t_high"])
        writer.writeheader()
        for row in paired:
            for metric, stats in row.get("differences", {}).items():
                writer.writerow({"method": row["method"], "baseline": row["baseline"], "metric": metric,
                                 "paired_roots": stats["n"], "mean_difference": stats["mean"],
                                 "sd_root": stats["sd_root"], "ci95_t_low": stats["ci95_t_low"],
                                 "ci95_t_high": stats["ci95_t_high"]})
    with (REPORTS / "E_METHOD_SUMMARY.csv").open("w", encoding="utf-8", newline="") as stream:
        fields = ["family", "arm_id", "root_count", "successful_full_int8_roots", "unsupported_or_failed_roots"]
        fields += ["%s_%s" % (metric, stat) for metric in ("utility", "known_correct_acceptance", "unknown_fpir",
                  "known_identification", "bytes", "parameters", "macs", "activation_peak_bytes_estimate",
                  "latency_p50_ms", "training_seconds") for stat in ("mean", "sd_root", "ci95_t_low", "ci95_t_high")]
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for alpha, summary in summaries["capacity"].items():
            row = {"family": "capacity", "arm_id": "capacity_a%03d" % int(float(alpha) * 100),
                   "root_count": summary["root_count"], "successful_full_int8_roots": summary["successful_full_int8_host_inference_roots"],
                   "unsupported_or_failed_roots": 0}
            for metric in metric_keys:
                row.update({"%s_%s" % (metric, stat): summary["metrics"][metric].get(stat)
                            for stat in ("mean", "sd_root", "ci95_t_low", "ci95_t_high")})
            writer.writerow(row)
        for method, summary in summaries["optimization"].items():
            row = {"family": "optimization", "arm_id": method, "root_count": summary["root_count"],
                   "successful_full_int8_roots": summary["successful_full_int8_host_inference_roots"],
                   "unsupported_or_failed_roots": summary["unsupported_or_failed_roots"]}
            for metric in metric_keys:
                row.update({"%s_%s" % (metric, stat): summary["metrics"][metric].get(stat)
                            for stat in ("mean", "sd_root", "ci95_t_low", "ci95_t_high")})
            writer.writerow(row)
    pareto = []
    points = []
    for alpha, roots in capacity.items():
        summary = summaries["capacity"][alpha]
        if summary["root_count"]:
            points.append({"family": "capacity", "arm_id": "capacity_a%03d" % int(float(alpha) * 100),
                           "alpha": float(alpha), "utility": summary["metrics"]["utility"]["mean"],
                           "unknown_fpir": summary["metrics"]["unknown_fpir"]["mean"],
                           "bytes": summary["metrics"]["bytes"]["mean"]})
    for method, summary in summaries["optimization"].items():
        if summary["metrics"]["utility"]["mean"] is not None:
            points.append({"family": "optimization", "arm_id": method,
                           "utility": summary["metrics"]["utility"]["mean"],
                           "unknown_fpir": summary["metrics"]["unknown_fpir"]["mean"],
                           "bytes": summary["metrics"]["bytes"]["mean"]})
    eligible = [row for row in points if row["unknown_fpir"] is not None and row["unknown_fpir"] <= 0.05]
    for point in eligible:
        dominated = any(other["utility"] >= point["utility"] and other["bytes"] <= point["bytes"]
                        and (other["utility"] > point["utility"] or other["bytes"] < point["bytes"])
                        for other in eligible if other is not point)
        if not dominated:
            pareto.append(point)
    (REPORTS / "E_DEVELOPMENT_PARETO.json").write_text(json.dumps({"points": points, "eligible_fp_ir_at_most_005": eligible,
        "pareto_shortlist": pareto, "role": "development-only; no final role", "status": "complete_verified"}, indent=2) + "\n", encoding="utf-8")
    with (REPORTS / "E_DEVELOPMENT_PARETO.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["family", "arm_id", "alpha", "utility", "unknown_fpir", "bytes", "pareto"])
        writer.writeheader()
        for point in points:
            writer.writerow({**point, "pareto": point in pareto})
    export_fields = ["family", "arm_id", "seed", "format", "status", "host_inference_status",
                     "file_bytes", "weights_artifact_bytes", "parameters", "tflite_macs",
                     "activation_peak_bytes_estimate", "all_int8_boundary", "internal_float_tensor_count",
                     "host_latency_p50_ms", "host_latency_p95_ms", "training_seconds", "training_updates"]
    with (REPORTS / "E_EXPORT_FORMATS.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=export_fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(export_rows)
    format_summary = []
    for family, arm_id, format_id in sorted({(row["family"], row["arm_id"], row["format"]) for row in export_rows}):
        group = [row for row in export_rows if (row["family"], row["arm_id"], row["format"]) == (family, arm_id, format_id)]
        good = [row for row in group if row.get("status") == "complete_verified" and row.get("host_inference_status") == "pass"]
        format_summary.append({"family": family, "arm_id": arm_id, "format": format_id,
                               "roots": len(group), "host_inference_pass": len(good),
                               "file_bytes_mean": mean_ci([row["file_bytes"] for row in good if row.get("file_bytes") is not None]),
                               "weights_artifact_bytes_mean": mean_ci([row["weights_artifact_bytes"] for row in good if row.get("weights_artifact_bytes") is not None]),
                               "activation_peak_bytes_estimate_mean": mean_ci([row["activation_peak_bytes_estimate"] for row in good if row.get("activation_peak_bytes_estimate") is not None]),
                               "tflite_macs_mean": mean_ci([row["tflite_macs"] for row in good if row.get("tflite_macs") is not None]),
                               "host_latency_p50_ms_mean": mean_ci([row["host_latency_p50_ms"] for row in good if row.get("host_latency_p50_ms") is not None]),
                               "training_seconds_mean": mean_ci([row["training_seconds"] for row in good if row.get("training_seconds") is not None])})
    (REPORTS / "E_EXPORT_FORMAT_SUMMARY.json").write_text(json.dumps({"schema": "job-e-export-format-summary-1",
        "status": "complete_verified", "rows": format_summary,
        "size_interpretation": "TFLite FlatBuffer file bytes are the exported distribution size; weights_artifact_bytes records the training checkpoint artifact separately.",
        "activation_interpretation": "Static TFLite tensor-lifetime peak estimate; not measured device arena/SRAM use.",
        "latency_interpretation": "Host-only TFLite timing; not board latency or energy.",
        "final_role_accessed": False}, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    with (REPORTS / "E_EXPORT_FORMAT_SUMMARY.csv").open("w", encoding="utf-8", newline="") as stream:
        fields = ["family", "arm_id", "format", "roots", "host_inference_pass", "file_bytes_mean", "weights_artifact_bytes_mean",
                  "activation_peak_bytes_estimate_mean", "tflite_macs_mean", "host_latency_p50_ms_mean", "training_seconds_mean"]
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in format_summary:
            writer.writerow({key: json.dumps(row.get(key), sort_keys=True) if isinstance(row.get(key), dict) else row.get(key)
                             for key in fields})
    robustness_path = RUN_ROOT / "robustness/RESULT.json"
    if robustness_path.is_file():
        robustness = json.loads(robustness_path.read_text(encoding="utf-8"))
        robust_groups = {}
        for model in robustness.get("models", []):
            group_key = (model.get("family"), model.get("arm_id"))
            robust_groups.setdefault(group_key, []).append(model)
        robustness_summary = []
        for (family, arm_id), group in sorted(robust_groups.items()):
            for condition_id in [row.get("id") for row in robustness.get("conditions", [])]:
                values = []
                for model in group:
                    condition_row = next((row for row in model.get("conditions", [])
                                          if row.get("condition", {}).get("id") == condition_id), None)
                    if condition_row:
                        values.append(condition_row.get("metrics", {}))
                robustness_summary.append({"family": family, "arm_id": arm_id, "condition": condition_id,
                    "roots": len(values),
                    "utility": mean_ci([row.get("development_macro_utility") for row in values if row.get("development_macro_utility") is not None]),
                    "known_correct_acceptance": mean_ci([row.get("known_identity_macro_correct_acceptance") for row in values if row.get("known_identity_macro_correct_acceptance") is not None]),
                    "unknown_fpir": mean_ci([row.get("unknown_false_identification_rate") for row in values if row.get("unknown_false_identification_rate") is not None])})
        with (REPORTS / "E_ROBUSTNESS_SUMMARY.csv").open("w", encoding="utf-8", newline="") as stream:
            fields = ["family", "arm_id", "condition", "roots", "utility_mean", "utility_sd_root", "utility_ci95_low", "utility_ci95_high",
                      "known_correct_acceptance_mean", "known_correct_acceptance_sd_root", "known_correct_acceptance_ci95_low", "known_correct_acceptance_ci95_high",
                      "unknown_fpir_mean", "unknown_fpir_sd_root", "unknown_fpir_ci95_low", "unknown_fpir_ci95_high"]
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            for row in robustness_summary:
                output = {"family": row["family"], "arm_id": row["arm_id"], "condition": row["condition"], "roots": row["roots"]}
                for metric in ("utility", "known_correct_acceptance", "unknown_fpir"):
                    output.update({"%s_%s" % (metric, target): row[metric].get(source)
                                   for target, source in (("mean", "mean"), ("sd_root", "sd_root"),
                                                          ("ci95_low", "ci95_t_low"), ("ci95_high", "ci95_t_high"))})
                writer.writerow(output)
        (REPORTS / "E_ROBUSTNESS_SUMMARY.json").write_text(json.dumps({"schema": "job-e-robustness-summary-1",
            "status": robustness.get("status"), "source": str(robustness_path), "groups": robustness_summary,
            "synthetic_only": True, "final_role_accessed": False}, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    pareto_ids = {point["arm_id"] for point in pareto}
    if plt is not None and points:
        fig, ax = plt.subplots(figsize=(9, 6), dpi=160)
        for point in (row for row in points if row["family"] == "optimization"):
            ax.scatter(point["bytes"] / 1024.0, point["utility"], marker="s",
                       alpha=0.72, s=44, color="#d05a37",
                       edgecolors="#172033" if point["arm_id"] in pareto_ids else "none",
                       linewidths=1.5 if point["arm_id"] in pareto_ids else 0)
        for point in (row for row in points if row["family"] == "capacity"):
            ax.scatter(point["bytes"] / 1024.0, point["utility"], marker="o",
                       alpha=0.95, s=64, color="#2374ab", edgecolors="white", linewidths=1.2,
                       zorder=3)
        for point in points:
            if point["family"] == "capacity" or point["arm_id"] in pareto_ids:
                label = "a=%.2f" % point["alpha"] if point["family"] == "capacity" else "Pareto: direct GAP"
                ax.annotate(label, (point["bytes"] / 1024.0, point["utility"]),
                            fontsize=7, xytext=(6, 5), textcoords="offset points")
        ax.set_xscale("log")
        ax.set_xlabel("Mean full-int8 model bytes (KiB, log scale)")
        ax.set_ylabel("Mean development macro utility")
        ax.set_title("Job E capacity and optimization: development utility vs full-int8 size")
        ax.grid(True, alpha=0.25)
        fig.tight_layout()
        fig.savefig(REPORTS / "E_QUALITY_VS_SIZE.png")
        plt.close(fig)
    elif points:
        # Keep the required static chart available in the lightweight model
        # environment, which does not bundle Matplotlib.
        from PIL import Image, ImageDraw, ImageFont
        width, height = 1440, 900
        left, top, right, bottom = 150, 115, 1365, 760
        image = Image.new("RGB", (width, height), "white")
        draw = ImageDraw.Draw(image)
        font_paths = ["/System/Library/Fonts/Supplemental/Arial.ttf",
                      "/Library/Fonts/Arial.ttf",
                      "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"]
        font = ImageFont.load_default()
        small_font = ImageFont.load_default()
        for font_path in font_paths:
            try:
                font = ImageFont.truetype(font_path, 18)
                small_font = ImageFont.truetype(font_path, 14)
                break
            except OSError:
                continue
        min_x = min(point["bytes"] / 1024.0 for point in points)
        max_x = max(point["bytes"] / 1024.0 for point in points)
        min_y = min(point["utility"] for point in points)
        max_y = max(point["utility"] for point in points)
        x_lo, x_hi = math.log10(min_x) - 0.15, math.log10(max_x) + 0.15
        y_pad = max(0.025, (max_y - min_y) * 0.15)
        y_lo, y_hi = min_y - y_pad, max_y + y_pad
        x_map = lambda value: left + (math.log10(value) - x_lo) / (x_hi - x_lo) * (right - left)
        y_map = lambda value: bottom - (value - y_lo) / (y_hi - y_lo) * (bottom - top)
        draw.text((left, 35), "Job E capacity and optimization: development utility vs full-int8 size",
                  fill="#172033", font=font)
        for tick in range(int(math.floor(y_lo * 10)), int(math.ceil(y_hi * 10)) + 1):
            value = tick / 10.0
            y = y_map(value)
            if top <= y <= bottom:
                draw.line((left, y, right, y), fill="#e6eaf0", width=1)
                draw.text((92, y - 7), "%.2f" % value, fill="#384152", font=small_font)
        for value in (10, 100, 1000, 10000):
            if min_x <= value <= max_x:
                x = x_map(value)
                draw.line((x, top, x, bottom), fill="#edf0f4", width=1)
                draw.text((x - 14, bottom + 15), str(value), fill="#384152", font=small_font)
        draw.line((left, top, left, bottom), fill="#172033", width=2)
        draw.line((left, bottom, right, bottom), fill="#172033", width=2)
        draw.text((left + 430, bottom + 48), "Mean full-int8 model bytes (KiB, log scale)",
                  fill="#172033", font=font)
        draw.text((25, top + 245), "Development macro utility", fill="#172033", font=font)
        for point in (row for row in points if row["family"] == "optimization"):
            x = x_map(point["bytes"] / 1024.0)
            y = y_map(point["utility"])
            outline = "#172033" if point["arm_id"] in pareto_ids else None
            draw.polygon(((x, y - 8), (x + 8, y), (x, y + 8), (x - 8, y)),
                         fill="#d05a37", outline=outline)
        for point in (row for row in points if row["family"] == "capacity"):
            x = x_map(point["bytes"] / 1024.0)
            y = y_map(point["utility"])
            outline = "#172033" if point["arm_id"] in pareto_ids else "white"
            draw.ellipse((x - 8, y - 8, x + 8, y + 8), fill="#2374ab", outline=outline, width=2)
        for point in points:
            if point["family"] == "capacity" or point["arm_id"] in pareto_ids:
                x = x_map(point["bytes"] / 1024.0)
                y = y_map(point["utility"])
                label = "alpha %.2f" % point["alpha"] if point["family"] == "capacity" else "Pareto: direct GAP"
                draw.text((x + 11, y - 20), label, fill="#303846", font=small_font)
        draw.ellipse((left + 5, height - 35, left + 17, height - 23), fill="#2374ab")
        draw.text((left + 24, height - 36), "Capacity", fill="#303846", font=small_font)
        draw.polygon(((left + 130, height - 36), (left + 137, height - 29),
                      (left + 130, height - 22), (left + 123, height - 29)), fill="#d05a37")
        draw.text((left + 145, height - 36), "Optimization", fill="#303846", font=small_font)
        draw.text((left + 305, height - 36), "Dark outline = development Pareto shortlist; method IDs are in Job E method summary.",
                  fill="#596274", font=small_font)
        image.save(REPORTS / "E_QUALITY_VS_SIZE.png")
    print(json.dumps({"event": "analysis_complete", "capacity_roots": {k: len(v) for k, v in capacity.items()},
                      "optimization_roots": {k: len(v) for k, v in methods.items()},
                      "pareto_count": len(pareto), "reports": str(REPORTS)}), flush=True)


if __name__ == "__main__":
    main()
