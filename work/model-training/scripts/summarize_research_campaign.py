#!/usr/bin/env python3
"""Aggregate a completed multi-seed training/export campaign into CSV, JSON, and Markdown."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
import re
import statistics
from pathlib import Path

from common import image_path, load_manifest, remove_exact_duplicate_conflicts, select_training_rows, sha256


ROOT = Path(__file__).resolve().parents[3]
TRAINING_ROOT = ROOT / "work/model-training"
METRIC_SET = "survface_identity_disjoint_validation"
_CLEAN_TRAINING_ROWS = None
_DUPLICATE_AUDIT = None


def read_json(path):
    return json.loads(path.read_text())


def reconstructed_training_rows():
    global _CLEAN_TRAINING_ROWS, _DUPLICATE_AUDIT
    if _CLEAN_TRAINING_ROWS is None:
        all_rows, _, raw_surv_rows, _ = load_manifest(include_olivetti=False)
        unlabeled_survface_rows = [row for row in all_rows
                                   if row["dataset"] == "survface_96" and not row["identity_id"]]
        _CLEAN_TRAINING_ROWS, _DUPLICATE_AUDIT = remove_exact_duplicate_conflicts(
            raw_surv_rows, unlabeled_survface_rows,
        )
    return _CLEAN_TRAINING_ROWS, _DUPLICATE_AUDIT


def validate_calibration(run_dir, evaluation, summary, config, split, manifest):
    calibration = evaluation.get("calibration", {})
    records = manifest.get("representative_records", [])
    if (calibration.get("representative_source") !=
            "exact selected training records after duplicate exclusion and per-identity cap"):
        raise SystemExit("Evaluation lacks corrected exact-selected training-only calibration marker: " + str(run_dir))
    if calibration.get("representative_validation_identity_count") != 0:
        raise SystemExit("Validation identities appeared in calibration metadata: " + str(run_dir))
    digest = hashlib.sha256(json.dumps(
        records, sort_keys=True, separators=(",", ":"),
    ).encode("utf-8")).hexdigest()
    if (len(records) != calibration.get("representative_count")
            or digest != calibration.get("representative_manifest_sha256")
            or digest != manifest.get("representative_manifest_sha256")):
        raise SystemExit("Calibration manifest count/hash differs from evaluation metadata: " + str(run_dir))

    clean_rows, duplicate_audit = reconstructed_training_rows()
    if sorted(duplicate_audit["excluded_record_ids"]) != sorted(split["excluded_exact_duplicate_record_ids"]):
        raise SystemExit("Reconstructed duplicate exclusions differ from split metadata: " + str(run_dir))
    training_ids = list(split["train_identity_ids"])
    identity_limit = int(config.get("train_identity_limit") or 0)
    if identity_limit:
        training_ids = sorted(random.Random(int(summary["training_seed"]) + 31).sample(
            training_ids, min(identity_limit, len(training_ids)),
        ))
    selected = select_training_rows(
        clean_rows, training_ids, int(config["max_images_per_train_identity"]), int(split["seed"]),
    )
    if len(selected) != int(summary["training_image_count"]):
        raise SystemExit("Reconstructed training-record count differs from run summary: " + str(run_dir))
    train_digest = hashlib.sha256("".join(
        row["record_id"] + "\t" + row["sha256"] + "\n" for row in selected
    ).encode("utf-8")).hexdigest()
    if train_digest != manifest.get("training_selection_sha256"):
        raise SystemExit("Training-selection digest differs from calibration manifest: " + str(run_dir))

    expected = list(selected)
    random.Random(int(split["seed"])).shuffle(expected)
    expected = expected[:min(512, len(expected))]
    actual_records = []
    for row in expected:
        image_hash = sha256(image_path(row))
        if image_hash != row["sha256"]:
            raise SystemExit("Prepared image SHA-256 changed: " + row["record_id"])
        actual_records.append({
            "record_id": row["record_id"],
            "identity_id": row["identity_id"],
            "image_sha256": image_hash,
        })
    if actual_records != records:
        raise SystemExit("Calibration records are not the exact deterministic training-only sample: " + str(run_dir))
    if (calibration.get("training_selection_sha256") != train_digest
            or calibration.get("reconstructed_training_record_count") != len(selected)):
        raise SystemExit("Evaluation training-selection provenance differs from reconstructed selection: " + str(run_dir))


def metrics_for(evaluation, model_name):
    return evaluation.get(model_name, {}).get(METRIC_SET, {})


def logged_epoch_duration(run_dir, summary):
    """Sum rounded Keras epoch durations without counting overnight wall gaps."""
    campaign_root = run_dir.parents[2]
    variant = summary["model_variant"]
    split_seed = int(summary["split_seed"])
    training_seed = int(summary["training_seed"])
    log_paths = [campaign_root / "logs" / (
        "%s-split-%d-seed-%d.log" % (variant, split_seed, training_seed)
    )]
    log_paths.extend(sorted((campaign_root / "logs").glob(
        "%s-split-%d-seed-%d-resume*.log" % (variant, split_seed, training_seed),
    )))
    durations = []
    pattern = re.compile(r"^\d+/\d+\s+-\s+(?:(\d+)m\s+)?([\d.]+)s\s+-")
    for log_path in log_paths:
        if not log_path.is_file():
            continue
        for line in log_path.read_text(errors="replace").splitlines():
            match = pattern.match(line.strip())
            if match:
                durations.append((int(match.group(1) or 0) * 60) + float(match.group(2)))
    return len(durations), sum(durations), (statistics.median(durations) if durations else None)


def summarize_values(values):
    values = [float(x) for x in values if x is not None and math.isfinite(float(x))]
    if not values:
        return {"n": 0, "mean": None, "sample_sd": None, "min": None, "max": None}
    return {
        "n": len(values),
        "mean": statistics.mean(values),
        "sample_sd": statistics.stdev(values) if len(values) > 1 else 0.0,
        "min": min(values),
        "max": max(values),
    }


def discover_runs(campaign_dir):
    rows = []
    for summary_path in sorted(campaign_dir.rglob("run_summary.json")):
        run_dir = summary_path.parent
        config_path = run_dir / "train_config.json"
        history_path = run_dir / "validation_history.jsonl"
        eval_path = run_dir / "float_int8_evaluation.json"
        missing = [str(path) for path in (config_path, history_path, eval_path) if not path.is_file()]
        if missing:
            raise SystemExit("Completed run is missing required evaluation inputs: " + ", ".join(missing))
        summary = read_json(summary_path)
        config = read_json(config_path)
        history = [json.loads(line) for line in history_path.read_text().splitlines() if line.strip()]
        evaluation = read_json(eval_path)
        manifest_path = run_dir / "calibration_manifest.json"
        if not manifest_path.is_file():
            raise SystemExit("Missing exact training-only calibration manifest: " + str(manifest_path))
        manifest = read_json(manifest_path)
        split = read_json(run_dir / "split.json")
        validate_calibration(run_dir, evaluation, summary, config, split, manifest)
        records = manifest["representative_records"]
        digest = manifest["representative_manifest_sha256"]
        for artifact_key in ("float_tflite", "full_integer_int8_tflite"):
            artifact = evaluation.get("model_files", {}).get(artifact_key, {})
            artifact_path = ROOT / artifact.get("path", "")
            if (not artifact_path.is_file()
                    or artifact_path.stat().st_size != artifact.get("file_bytes")
                    or sha256(artifact_path) != artifact.get("sha256")):
                raise SystemExit("Exported model file is missing or hash-mismatched: " + str(artifact_path))
        int8_artifact = evaluation["model_files"]["full_integer_int8_tflite"]
        if not int8_artifact.get("input_output_full_int8"):
            raise SystemExit("Candidate does not have signed full-INT8 input and output: " + str(run_dir))
        best = max(history, key=lambda item: item["validation_verification"]["roc_auc"])
        epoch_log_count, epoch_log_seconds, median_epoch_seconds = logged_epoch_duration(run_dir, summary)
        int8 = metrics_for(evaluation, "tflite_full_int8")
        float_tflite = metrics_for(evaluation, "tflite_float")
        int8_file = evaluation.get("model_files", {}).get("full_integer_int8_tflite", {})
        float_file = evaluation.get("model_files", {}).get("float_tflite", {})
        timing = evaluation.get("host_inference_timing", {})
        float_timing = timing.get("tflite_float", {})
        int8_timing = timing.get("tflite_full_int8", {})
        rows.append({
            "model_variant": summary["model_variant"],
            "campaign_id": run_dir.relative_to(TRAINING_ROOT / "runs").parts[0],
            "split_seed": int(summary["split_seed"]),
            "training_seed": int(summary["training_seed"]),
            "augmentation_seed": int(summary["augmentation_seed"]),
            "run_dir": str(run_dir.resolve()),
            "epochs_completed": int(summary["epochs_completed"]),
            "epochs_requested": int(config.get("epochs_requested", 0)),
            "resumed_from_checkpoint": float(summary.get("resume_previous_runtime_seconds", 0.0)) > 0,
            "best_epoch": int(best["epoch"]),
            "runtime_seconds": float(summary["runtime_seconds"]),
            "runtime_measurement": "reported wall clock; includes image loading and may include idle/sleep gaps",
            "logged_epoch_duration_count": epoch_log_count,
            "epoch_log_elapsed_seconds_sum_with_sleep_gaps": epoch_log_seconds,
            "median_logged_epoch_elapsed_seconds": median_epoch_seconds,
            "logged_epoch_duration_measurement": "Keras wall-clock epoch durations rounded to whole seconds; may include host-sleep gaps and are not CPU time",
            "train_image_count": int(summary["training_image_count"]),
            "train_identity_count": int(summary["training_identity_count"]),
            "validation_identity_count": int(summary["validation_identity_count"]),
            "validation_pair_count": int(int8.get("pair_count", 0)),
            "float_tflite_auc": float_tflite.get("roc_auc"),
            "float_tflite_eer": float_tflite.get("eer"),
            "float_tflite_tar_at_val_far": float_tflite.get("tar_at_decision_threshold"),
            "float_tflite_far_at_val_threshold": float_tflite.get("far_at_decision_threshold"),
            "int8_auc": int8.get("roc_auc"),
            "int8_eer": int8.get("eer"),
            "int8_validation_decision_threshold": int8.get("decision_threshold"),
            "int8_tar_at_val_far": int8.get("tar_at_decision_threshold"),
            "int8_far_at_val_threshold": int8.get("far_at_decision_threshold"),
            "int8_auc_pair_bootstrap_ci_low": (int8.get("bootstrap_95pct_ci", {}).get("roc_auc") or [None, None])[0],
            "int8_auc_pair_bootstrap_ci_high": (int8.get("bootstrap_95pct_ci", {}).get("roc_auc") or [None, None])[1],
            "int8_auc_identity_cluster_ci_low": (int8.get("identity_cluster_bootstrap", {}).get("bootstrap_95pct_ci", {}).get("roc_auc") or [None, None])[0],
            "int8_auc_identity_cluster_ci_high": (int8.get("identity_cluster_bootstrap", {}).get("bootstrap_95pct_ci", {}).get("roc_auc") or [None, None])[1],
            "int8_bytes": int(int8_file.get("file_bytes", 0)),
            "float_tflite_bytes": int(float_file.get("file_bytes", 0)),
            "int8_to_float_auc_delta": (int8.get("roc_auc") - float_tflite.get("roc_auc")
                                         if int8.get("roc_auc") is not None and float_tflite.get("roc_auc") is not None else None),
            "float_host_mean_ms_per_image": float_timing.get("mean_ms_per_image"),
            "float_host_p95_ms_per_image": float_timing.get("p95_ms_per_image"),
            "int8_host_mean_ms_per_image": int8_timing.get("mean_ms_per_image"),
            "int8_host_p95_ms_per_image": int8_timing.get("p95_ms_per_image"),
            "host_timing_cpu_threads": int8_timing.get("host_cpu_threads"),
            "int8_input_dtype": int8_file.get("input", {}).get("dtype"),
            "int8_output_dtype": int8_file.get("output", {}).get("dtype"),
            "int8_operators": ",".join(int8_file.get("operators", [])),
            "train_intra_threads": config.get("intra_threads"),
            "train_inter_threads": config.get("inter_threads"),
            "validation_selection_bias_note": "best checkpoint selected by validation ROC AUC",
            "calibration_manifest_sha256": digest,
            "calibration_representative_count": len(records),
        })
    return rows


def format_mean_sd_range(values, digits=4):
    stats = summarize_values(values)
    if stats["n"] == 0:
        return "n/a"
    return ("%.{}f +/- %.{}f (%.{}f to %.{}f)".format(digits, digits, digits, digits)
            % (stats["mean"], stats["sample_sd"], stats["min"], stats["max"]))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign-dir", type=Path, required=True)
    parser.add_argument("--also-campaign-dir", type=Path, action="append", default=[])
    parser.add_argument("--output-prefix", type=Path, default=None)
    args = parser.parse_args()
    campaign_dir = args.campaign_dir if args.campaign_dir.is_absolute() else ROOT / args.campaign_dir
    prefix = args.output_prefix
    if prefix is None:
        prefix = TRAINING_ROOT / "reports/research_seed_architecture_campaign_20261021"
    elif not prefix.is_absolute():
        prefix = ROOT / prefix
    campaign_dirs = [campaign_dir]
    for other in args.also_campaign_dir:
        campaign_dirs.append(other if other.is_absolute() else ROOT / other)
    rows = []
    for directory in campaign_dirs:
        rows.extend(discover_runs(directory))
    if not rows:
        raise SystemExit("No fully trained and evaluated campaign runs found under " + str(campaign_dir))
    prefix.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0])
    csv_path = prefix.with_suffix(".csv")
    with csv_path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    groups = {}
    for row in rows:
        # Pool predeclared seed extensions that share the same model/split protocol.
        key = "%s/split-%d" % (row["model_variant"], row["split_seed"])
        groups.setdefault(key, []).append(row)
    group_summary = {}
    for key, group_rows in sorted(groups.items()):
        seeds = [row["training_seed"] for row in group_rows]
        if len(seeds) != len(set(seeds)):
            raise SystemExit("Duplicate training seed within aggregate group %s" % key)
        group_summary[key] = {
            "seed_count": len(group_rows),
            "training_seeds": sorted(seeds),
            "campaign_ids": sorted({row["campaign_id"] for row in group_rows}),
            "int8_auc": summarize_values(row["int8_auc"] for row in group_rows),
            "int8_eer": summarize_values(row["int8_eer"] for row in group_rows),
            "int8_validation_decision_threshold": summarize_values(
                row["int8_validation_decision_threshold"] for row in group_rows),
            "int8_tar_at_validation_1pct_far_threshold": summarize_values(
                row["int8_tar_at_val_far"] for row in group_rows),
            "int8_far_at_validation_threshold": summarize_values(
                row["int8_far_at_val_threshold"] for row in group_rows),
            "float_tflite_auc": summarize_values(row["float_tflite_auc"] for row in group_rows),
            "int8_to_float_auc_delta": summarize_values(row["int8_to_float_auc_delta"] for row in group_rows),
            "int8_model_bytes": summarize_values(row["int8_bytes"] for row in group_rows),
            "float_host_mean_ms_per_image": summarize_values(row["float_host_mean_ms_per_image"] for row in group_rows),
            "float_host_p95_ms_per_image": summarize_values(row["float_host_p95_ms_per_image"] for row in group_rows),
            "int8_host_mean_ms_per_image": summarize_values(row["int8_host_mean_ms_per_image"] for row in group_rows),
            "int8_host_p95_ms_per_image": summarize_values(row["int8_host_p95_ms_per_image"] for row in group_rows),
            "completed_epochs": summarize_values(row["epochs_completed"] for row in group_rows),
            "median_logged_epoch_elapsed_seconds": summarize_values(
                row["median_logged_epoch_elapsed_seconds"] for row in group_rows),
        }
    paired_differences = {}
    by_key = {(row["model_variant"], row["split_seed"], row["training_seed"]): row for row in rows}
    for split_seed in sorted({row["split_seed"] for row in rows}):
        differences = []
        for seed in sorted({row["training_seed"] for row in rows}):
            base = by_key.get(("base", split_seed, seed))
            tiny = by_key.get(("tiny", split_seed, seed))
            if base and tiny and base["int8_auc"] is not None and tiny["int8_auc"] is not None:
                differences.append(tiny["int8_auc"] - base["int8_auc"])
        if differences:
            paired_differences["tiny_minus_base_split_%d" % split_seed] = summarize_values(differences)
    for variant in sorted({row["model_variant"] for row in rows}):
        variant_splits = sorted({row["split_seed"] for row in rows if row["model_variant"] == variant})
        if len(variant_splits) > 1:
            reference_split = variant_splits[0]
            for target_split in variant_splits[1:]:
                differences = []
                for seed in sorted({row["training_seed"] for row in rows}):
                    reference = by_key.get((variant, reference_split, seed))
                    target = by_key.get((variant, target_split, seed))
                    if reference and target and reference["int8_auc"] is not None and target["int8_auc"] is not None:
                        differences.append(target["int8_auc"] - reference["int8_auc"])
                if differences:
                    paired_differences["%s_split_%d_minus_%d" % (
                        variant, target_split, reference_split,
                    )] = summarize_values(differences)

    machine = {
        "campaign_dirs": [str(directory.resolve()) for directory in campaign_dirs],
        "completed_evaluated_run_count": len(rows),
        "cumulative_reported_wall_clock_hours": sum(float(row["runtime_seconds"]) for row in rows) / 3600.0,
        "groups": group_summary,
        "paired_int8_auc_differences": paired_differences,
        "per_run": rows,
        "interpretation": [
            "Validation metrics are checkpoint-selected and should not be described as unbiased test estimates.",
            "Current INT8 export metrics use representative calibration images from training identities only; validation identities are excluded from the calibration pool.",
            "The supplied final pair annotations are excluded from this campaign.",
            "Pair-bootstrap intervals are included per run but may be optimistic due to repeated identities/images across pairs.",
            "Seed standard deviation is the primary descriptive measure of training-run variability within each group.",
            "Recorded runtimes are wall-clock durations from the host and include image loading and any overnight idle/sleep gaps; they are not CPU-time measurements or suitable for throughput comparisons.",
        ],
    }
    json_path = prefix.with_suffix(".json")
    json_path.write_text(json.dumps(machine, indent=2, sort_keys=True) + "\n")

    lines = [
        "# Multi-seed architecture campaign summary",
        "",
        "Completed evaluated runs: %d" % len(rows),
        "Cumulative recorded wall-clock duration: %.1f hours (includes image loading and overnight idle/sleep gaps; not CPU time or a throughput measurement)." % (sum(row["runtime_seconds"] for row in rows) / 3600.0),
        "",
        "Metrics are validation-only. Each checkpoint was chosen by validation ROC AUC, so these are model-selection results, not unbiased test estimates. Current INT8 exports use representative calibration images from training identities only; validation identities are excluded from the calibration pool. The provided final pair annotations were not loaded.",
        "",
        "## Group summaries",
        "",
        "| Model / split | Seeds (n) | INT8 AUC mean +/- SD (range) | INT8 EER mean +/- SD (range) | INT8 TAR at validation 1% FAR threshold | INT8 bytes mean +/- SD |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for key, group_rows in sorted(groups.items()):
        stats = group_summary[key]
        lines.append("| %s | %s (%d) | %s | %s | %s | %s |" % (
            key, ", ".join(map(str, stats["training_seeds"])), stats["seed_count"],
            format_mean_sd_range(row["int8_auc"] for row in group_rows),
            format_mean_sd_range(row["int8_eer"] for row in group_rows),
            format_mean_sd_range(row["int8_tar_at_val_far"] for row in group_rows),
            format_mean_sd_range((row["int8_bytes"] for row in group_rows), digits=0),
        ))
    if paired_differences:
        lines += ["", "## Paired INT8 AUC differences", "",
                  "| Comparison | Matched runs | Difference mean +/- SD (range) |",
                  "|---|---:|---:|"]
        for comparison, stats in sorted(paired_differences.items()):
            lines.append("| %s | %d | %.4f +/- %.4f (%.4f to %.4f) |" % (
                comparison, stats["n"], stats["mean"], stats["sample_sd"], stats["min"], stats["max"],
            ))
    lines += [
        "",
        "## Per-run INT8 results",
        "",
        "| Model | Split | Train seed | Best epoch / completed | AUC | EER | TAR at 1% FAR threshold | FAR | Bytes |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in sorted(rows, key=lambda x: (x["model_variant"], x["split_seed"], x["training_seed"])):
        lines.append("| %s | %d | %d | %d / %d | %.4f | %.4f | %.4f | %.4f | %d |" % (
            row["model_variant"], row["split_seed"], row["training_seed"], row["best_epoch"],
            row["epochs_completed"], row["int8_auc"], row["int8_eer"],
            row["int8_tar_at_val_far"], row["int8_far_at_val_threshold"],
            row["int8_bytes"],
        ))
    lines += [
        "",
        "## Interpretation limits",
        "",
        "- The grouped seed counts (10 or 15, depending on model/split) describe run-to-run variability on those fixed splits; they do not create independent test sets.",
        "- The second identity split provides a limited split-sensitivity check, not broad domain generalization.",
        "- Recorded runtime is wall-clock time and includes image loading and any overnight idle/sleep gaps; do not compare it as active training compute.",
        "- Pair-level bootstrap resamples pairs independently even though identities/images recur; intervals can be too narrow.",
        "- The historical seed pilot uses a different TensorFlow thread configuration and is kept separate from this standardized campaign.",
        "- See each run's `float_int8_evaluation.json` for bootstrap intervals, thresholds, timing, quantization metadata, operators, and model hashes.",
        "",
    ]
    markdown_path = prefix.with_name(prefix.name + "_summary.md")
    markdown_path.write_text("\n".join(lines))
    print(json.dumps({"runs": len(rows), "csv": str(csv_path), "json": str(json_path),
                      "markdown": str(markdown_path), "groups": group_summary}, sort_keys=True))


if __name__ == "__main__":
    main()
