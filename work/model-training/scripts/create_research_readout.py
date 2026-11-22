#!/usr/bin/env python3
"""Create a detailed, local research readout from the aggregate campaign JSON."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]


def fmt_stats(stats, digits=4):
    if not stats or not stats.get("n"):
        return "n/a"
    return ("{mean:.{d}f} ± {sd:.{d}f} ({lo:.{d}f}–{hi:.{d}f})".format(
        mean=stats["mean"], sd=stats["sample_sd"], lo=stats["min"], hi=stats["max"], d=digits,
    ))


def finite(values):
    return [float(v) for v in values if v is not None and math.isfinite(float(v))]


def mean_value(values, digits=4):
    values = finite(values)
    return ("%.{}f".format(digits) % (sum(values) / len(values))) if values else "n/a"


def interval_width(rows, low_key, high_key):
    widths = finite([
        (row.get(high_key) - row.get(low_key))
        if row.get(low_key) is not None and row.get(high_key) is not None else None
        for row in rows
    ])
    return mean_value(widths)


def read_quant_metadata(rows):
    by_variant = {}
    for row in rows:
        evaluation_path = Path(row["run_dir"]) / "float_int8_evaluation.json"
        evaluation = json.loads(evaluation_path.read_text())
        item = evaluation["model_files"]["full_integer_int8_tflite"]
        by_variant.setdefault(row["model_variant"], []).append(item)
    result = {}
    for variant, items in by_variant.items():
        inputs = [item["input"] for item in items]
        outputs = [item["output"] for item in items]
        result[variant] = {
            "input_shape": inputs[0]["shape"],
            "input_dtype": inputs[0]["dtype"],
            "input_scale_min": min(x["scale"] for x in inputs),
            "input_scale_max": max(x["scale"] for x in inputs),
            "input_zero_points": sorted({x["zero_point"] for x in inputs}),
            "output_shape": outputs[0]["shape"],
            "output_dtype": outputs[0]["dtype"],
            "output_scale_min": min(x["scale"] for x in outputs),
            "output_scale_max": max(x["scale"] for x in outputs),
            "output_zero_points": sorted({x["zero_point"] for x in outputs}),
            "operator_sets": sorted({", ".join(item["operators"]) for item in items}),
            "unsupported_op_sets": sorted({", ".join(item.get("unsupported_openmv_candidate_ops", [])) for item in items}),
            "float_bytes_min": min(
                json.loads((Path(row["run_dir"]) / "float_int8_evaluation.json").read_text())
                ["model_files"]["float_tflite"]["file_bytes"]
                for row in rows if row["model_variant"] == variant
            ),
            "float_bytes_max": max(
                json.loads((Path(row["run_dir"]) / "float_int8_evaluation.json").read_text())
                ["model_files"]["float_tflite"]["file_bytes"]
                for row in rows if row["model_variant"] == variant
            ),
        }
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary-json", type=Path, required=True)
    parser.add_argument("--pilot-json", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    summary_path = args.summary_json if args.summary_json.is_absolute() else ROOT / args.summary_json
    pilot_path = args.pilot_json if args.pilot_json.is_absolute() else ROOT / args.pilot_json
    output = args.output if args.output.is_absolute() else ROOT / args.output
    summary = json.loads(summary_path.read_text())
    pilot = json.loads(pilot_path.read_text())
    rows = summary["per_run"]
    groups = summary["groups"]
    paired = summary.get("paired_int8_auc_differences", {})
    required_groups = {
        "base/split-20261021", "tiny/split-20261021",
        "base/split-20261022", "tiny/split-20261022",
    }
    missing_groups = sorted(required_groups - set(groups))
    if missing_groups:
        raise SystemExit("Final readout requires fully evaluated groups; missing: " + ", ".join(missing_groups))
    expected_seed_counts = {
        "base/split-20261021": 10,
        "tiny/split-20261021": 10,
        "base/split-20261022": 15,
        "tiny/split-20261022": 10,
    }
    expected_seed_sets = {
        "base/split-20261021": set(range(20261022, 20261027)) | set(range(20261032, 20261037)),
        "tiny/split-20261021": set(range(20261022, 20261027)) | set(range(20261032, 20261037)),
        "base/split-20261022": set(range(20261022, 20261037)),
        "tiny/split-20261022": set(range(20261022, 20261027)) | set(range(20261032, 20261037)),
    }
    if len(rows) != sum(expected_seed_counts.values()):
        raise SystemExit("Expected exactly 45 corrected standard runs; found %d" % len(rows))
    for key, count in expected_seed_counts.items():
        if groups[key]["seed_count"] != count:
            raise SystemExit("Expected %d runs in %s; found %d" % (count, key, groups[key]["seed_count"]))
        if set(groups[key]["training_seeds"]) != expected_seed_sets[key]:
            raise SystemExit("Unexpected training-seed set in %s" % key)
    pilot_rows = pilot.get("per_run", [])
    if (len(pilot_rows) != 5 or
            {row["training_seed"] for row in pilot_rows} != set(range(20261022, 20261027))):
        raise SystemExit("Expected exactly five separately summarized pilot runs; found %d" % len(pilot_rows))
    quant = read_quant_metadata(rows)
    resumed_runs = [row for row in rows if row["resumed_from_checkpoint"]]
    target40_runs = sum(1 for row in rows if row["epochs_requested"] == 40)
    target50_runs = sum(1 for row in rows if row["epochs_requested"] == 50)

    lines = [
        "# Expanded Nicla Vision face-verification research readout",
        "",
        "**Updated:** 2026-10-22  ",
        "**Result:** **Partially functional on Nicla**; identity verification for a wearer remains unvalidated.  ",
        "**Scope:** local SurvFace training and identity-disjoint validation; float and full-INT8 TFLite comparison; independent initialization and identity-split sensitivity; prior actual-board deployment evidence.",
        "",
        "## Executive summary",
        "",
        "The expanded campaign contains **%d validation-scored checkpoints**: 45 standardized runs across two channel-width variants and two identity-disjoint splits, plus a separate five-seed pilot. The earlier seed-20261021 base/tiny starter comparison remains reported separately. The extra five-seed-per-cell extension was specified before those new runs began but followed interim results from the earlier cohorts, so it is a replication/sensitivity extension rather than a fully pre-registered confirmatory study." % len(rows),
        "",
        "The chosen epoch and operating threshold for every candidate were selected from that candidate's own validation data. Before this campaign's consolidated scoring, an audit found the older export helper sampled INT8 representative images from both train and validation identities and ignored exact duplicate exclusions and the training image cap. The helper was corrected, and all 45 standardized checkpoints and the separate five-run pilot were re-exported and rescored using the exact records selected for training, after duplicate filtering and the per-identity cap. Each run retains a hash-checked calibration manifest. The supplied final SurvFace pair annotations and Olivetti were not loaded. This is a documented protocol correction after initial campaign training results were known; corrected scores below supersede earlier validation-only export scores for these runs. The historical seed-20261021 final-pair results and installed board model are unchanged and not directly comparable to the corrected calibration protocol.",
        "",
        "## Study design",
        "",
        "| Factor | Design |",
        "|---|---|",
        "| Recognition task | 64-D face embedding trained with supervised contrastive loss; cosine verification |",
        "| Data | SurvFace cropped 96×96 grayscale training identities; Olivetti excluded |",
        "| Split | Identity-disjoint 80/20 split stratified by image-count decile; split seeds 20261021 and 20261022 |",
        "| Identity coverage | 4,259 training identities / 1,060 validation identities per split; 44,897 or 44,892 selected training images depending on split |",
        "| Batch and sampling | 8 identities × 4 images; up to 24 training images per identity |",
        "| Augmentation | Training only: horizontal flip, translation ≤1.5%, contrast ±12% |",
        "| Stopping/checkpoint | Maximum 30 epochs, patience 5; validation ROC AUC chooses checkpoint; predeclared same-seed continuations up to 40/50 where a cap was reached before patience |",
        "| Architecture contrast | Base versus narrower tiny channel widths; same input, loss, 64-D output, sampler, augmentation, seeds and optimizer |",
        "| Export and scoring | Float TFLite and signed full-integer INT8; representative calibration uses the exact selected training records after duplicate exclusion and the per-identity cap; validation-only; 1,000 pair and 1,000 identity-cluster bootstrap replicates per run |",
        "| Host | Intel Core i9 MacBook Pro, macOS Darwin 25.6.0, 16 logical CPUs / 64 GiB, Python 3.9.6, TensorFlow 2.10.1 CPU |",
        "",
        "Both splits use the same 4,259/1,060 identity counts; the exact per-split training image count is recorded in each run's configuration and CSV. Checkpoints and operating thresholds use validation-only metrics. INT8 representative sets are drawn from the exact per-run training records, after the same duplicate exclusion and per-identity cap used by training. No final pair annotation or image-level identity inference was used for this campaign.",
        "",
        "## Validation-only performance by architecture and split",
        "",
        "Each cell reports across-seed mean ± sample SD (observed range). TAR is measured at a decision threshold selected on that run's validation negatives for a nominal 1% FAR target; the measured FAR is shown beside it because the finite negative set makes the realized FAR discrete.",
        "",
        "| Model / split | Seeds (n) | INT8 ROC AUC | INT8 EER | TAR at validation 1% FAR threshold | Realized FAR | Float-to-INT8 ΔAUC | INT8 bytes |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for key in ("base/split-20261021", "tiny/split-20261021", "base/split-20261022", "tiny/split-20261022"):
        stats = groups[key]
        delta = stats["int8_to_float_auc_delta"]
        lines.append("| %s | %d | %s | %s | %s | %s | %s | %s |" % (
            key, stats["seed_count"], fmt_stats(stats["int8_auc"]), fmt_stats(stats["int8_eer"]),
            fmt_stats(stats["int8_tar_at_validation_1pct_far_threshold"]),
            fmt_stats(stats["int8_far_at_validation_threshold"]), fmt_stats(delta),
            fmt_stats(stats["int8_model_bytes"], digits=0),
        ))
    lines += [
        "",
        "The combined group sizes are base/split A=10, tiny/split A=10, base/split B=15, and tiny/split B=10. Split-B base has five additional distinct initialization seeds (20261027–20261031); all other architecture/split cells have matched seeds 20261022–20261026 and 20261032–20261036.",
        "",
        "Patience-driven continuation counts: %d of the 45 standardized runs resumed from a saved checkpoint; %d runs were assigned a 40-epoch total target and %d a 50-epoch total target. These targets extend the same seeds and are not extra independent runs." % (
            len(resumed_runs), target40_runs, target50_runs,
        ),
        "",
        "## Convergence and training duration",
        "",
        "Keras records per-epoch wall-clock durations rounded to seconds. We report the median epoch duration per run and then summarize those medians across seeds, which reduces the effect of isolated long interruptions. Some runs include laptop idle/sleep gaps; therefore these values are not active CPU time, and exact cumulative compute time cannot be reconstructed from the historical logs.",
        "",
        "| Model / split | Completed epochs mean ± SD (range) | Median logged epoch seconds mean ± SD (range) |",
        "|---|---:|---:|",
    ]
    for key in ("base/split-20261021", "tiny/split-20261021", "base/split-20261022", "tiny/split-20261022"):
        stats = groups[key]
        lines.append("| %s | %s | %s |" % (
            key, fmt_stats(stats["completed_epochs"], digits=1),
            fmt_stats(stats["median_logged_epoch_elapsed_seconds"], digits=1),
        ))
    lines += [
        "",
        "## Matched comparisons",
        "",
        "Paired differences subtract the first named metric from the second according to the row label below; they use identical training seeds/augmentation seeds and compare the same identity split where indicated. Mean ± sample SD and range describe seed-to-seed paired changes, not a population-level confidence interval.",
        "",
        "| Comparison (difference) | Matched seeds | INT8 AUC difference mean ± SD (range) |",
        "|---|---:|---:|",
    ]
    pair_labels = [
        ("tiny − base, split A", "tiny_minus_base_split_20261021"),
        ("tiny − base, split B", "tiny_minus_base_split_20261022"),
        ("base, split B − split A", "base_split_20261022_minus_20261021"),
        ("tiny, split B − split A", "tiny_split_20261022_minus_20261021"),
    ]
    for label, key in pair_labels:
        item = paired.get(key)
        if item:
            lines.append("| %s | %d | %+.4f ± %.4f (%+.4f to %+.4f) |" % (
                label, item["n"], item["mean"], item["sample_sd"], item["min"], item["max"],
            ))
    lines += [
        "",
        "## Export, quantization, and embedded suitability",
        "",
        "| Variant | Float TFLite bytes | Full INT8 bytes | INT8 interface | Operators in all evaluated candidates | Unsupported OpenMV candidate ops |",
        "|---|---:|---:|---|---|---|",
    ]
    for variant in ("base", "tiny"):
        item = quant[variant]
        byte_stats = groups[variant + "/split-20261021"]["int8_model_bytes"]
        lines.append("| %s | %d–%d | %s | INT8 %s %s → %s %s | %s | %s |" % (
            variant, item["float_bytes_min"], item["float_bytes_max"], round(byte_stats["mean"]),
            item["input_shape"], item["input_dtype"].split(".")[-1].rstrip("'>"),
            item["output_shape"], item["output_dtype"].split(".")[-1].rstrip("'>"),
            "<br/>".join(item["operator_sets"]),
            "; ".join(item["unsupported_op_sets"]) or "none found",
        ))
    lines += [
        "",
        "Signed INT8 input shape is `[1,96,96,1]`, scale `1/255` (approximately 0.00392157), zero point −128; output shape is `[1,64]`. Output scales and zero points are candidate-specific and remain in each run's untouched TFLite metadata. Conversion inspection found the same supported candidate operator set across variants: `CONV_2D`, `DEPTHWISE_CONV_2D`, `FULLY_CONNECTED`, and `MEAN`. The tiny model is about 41.8% smaller in INT8 bytes than base (15,632 B vs 26,872 B); whether this storage reduction is worth any observed verification loss is a validation tradeoff, not a board latency result. No structured-pruning experiment was performed; the practical width-reduced tiny architecture was trained/exported instead.",
        "",
        "Host inference timing used ten warmup inputs and 200 timed validation images per exported candidate, with three TensorFlow Lite host threads; evaluation processes ran serially to avoid cross-process contention during timing. JPEG decode is outside the timer; tensor quantization/dequantization, model invocation, and embedding normalization are inside it. Values below are mean ± SD across candidate runs for per-image mean and per-image p95, not Nicla timings.",
        "",
        "| Model / split | Float TFLite mean ms | Float TFLite p95 ms | Full INT8 mean ms | Full INT8 p95 ms |",
        "|---|---:|---:|---:|---:|",
    ]
    for key in ("base/split-20261021", "tiny/split-20261021", "base/split-20261022", "tiny/split-20261022"):
        stats = groups[key]
        lines.append("| %s | %s | %s | %s | %s |" % (
            key,
            fmt_stats(stats["float_host_mean_ms_per_image"], digits=3),
            fmt_stats(stats["float_host_p95_ms_per_image"], digits=3),
            fmt_stats(stats["int8_host_mean_ms_per_image"], digits=3),
            fmt_stats(stats["int8_host_p95_ms_per_image"], digits=3),
        ))
    lines += [
        "",
        "Host float-versus-INT8 AUC deltas are summarized per cell above. Each exported candidate's input/output quantization metadata, graph inspection, hash, host timing, threshold, pair metrics, and bootstrap intervals are retained in its `float_int8_evaluation.json`; per-run aggregate metrics are in the companion CSV.",
        "",
        "## Statistical interpretation",
        "",
        "- The standard deviation and range across independent initialization seeds describe training-run variability on these fixed splits. They are not independent test-set variability.",
        "- Every checkpoint was selected by validation ROC AUC, and every reported threshold was set from that same validation cohort. This produces optimistic selection estimates; AUC is less threshold-dependent but still subject to checkpoint selection.",
        "- Pair-bootstrap intervals resample 2,120 validation pairs and can be too narrow because pair records reuse the same identities and images. Identity-cluster bootstrap intervals resample identity units and are more appropriate for dependence, but neither corrects for model selection or the limited number of identity splits.",
        "- Two identity partitions provide a limited split-sensitivity check, not broad domain generalization. The fixed dataset contains no capture-session metadata, so disjoint identities do not prove session/source-disjoint images.",
        "- The separate five-seed pilot used 12 intra-op / 2 inter-op threads during training, unlike the 4/1-thread standardized cohort; all corrected export/evaluation runs use three host TFLite threads. Pilot INT8 AUC was %s, EER %s, and TAR at the validation 1%% FAR threshold %s. Keep it descriptive and separate from pooled standardized statistics." % (
            fmt_stats(next(iter(pilot["groups"].values()))["int8_auc"]),
            fmt_stats(next(iter(pilot["groups"].values()))["int8_eer"]),
            fmt_stats(next(iter(pilot["groups"].values()))["int8_tar_at_validation_1pct_far_threshold"]),
        ),
        "- Reported training runtime fields are wall-clock durations and may include image-loading plus host idle/sleep gaps. They are retained for provenance but are not interpreted as active CPU time or throughput.",
        "",
        "## Actual Nicla evidence (the expanded candidates were not flashed)",
        "",
        "The board remains Arduino Nicla Vision (`2341:045F`, NICLAV H7 1024), OpenMV 5.0.1 / MicroPython 1.28.0, bootloader v24. The installed model is the original base seed-20261021 INT8 model, 26,872 B, loaded from writable `/flash`; the expanded campaign candidates have not been installed or camera-tested. The stock 4 MiB ROMFS and stock FOMO asset remain intact.",
        "",
        "A prior headless live synthetic-target check ran 20 frames: FOMO detections 18/20; embedding invoked 13 times; five detected crops were edge-clipped and marked uncertain; two frames had no face; zero pipeline errors/resets. All 13 embedding results were `no_enrollment`, because no template is installed. Capture was 34–82 ms, FOMO 65.5–66.3 ms, embedding 12.749–12.763 ms; minimum Python heap was 342,896 B. FOMO and embedding arenas (237,936 B and 51,040 B) were loaded sequentially. Six held-out neutral scenes produced no FOMO regions at threshold 0.4. A separate corrected synthetic scene passed the stock detector and embedder; host/board embedding cosine was 0.999382. The camera viewer stayed closed and no live frames were saved.",
        "",
        "This proves that the original exported embedder and recognition pipeline execute on the target, not that the model recognizes the user or partner. No consented wearer enrollment template exists. The stock FOMO model misses neutral-scene inputs and uses heatmap centers rather than calibrated face boxes; the 20-frame synthetic-target result is not a camera-domain accuracy estimate.",
        "",
        "## Limitations and responsible interpretation",
        "",
        "- The training archive contains cropped faces, not full camera scenes or detector boxes; no detector was trained. The FOMO detector remains the stock model.",
        "- SurvFace final pair labels were used once in the historical baseline evaluation and were not reopened for these additional runs. The historical values remain in the project report and are not updated by this validation-only campaign.",
        "- SurvFace transformed-archive provenance and reuse rights remain unresolved; source materials specify academic-research-only use and upstream image-owner copyright. Olivetti is excluded and also has incomplete archive provenance. Keep all data, weights, checkpoints, logs, and embeddings local and do not redistribute them.",
        "- The FNIRSI HID tester reports 0 V during board activity and its physical model/path are unresolved. Nicla power, energy per inference, and any battery-life effect are unmeasured. No power claim is made.",
        "- Real-person or wearable claims require consented enrollment, independent camera-matched evaluation identities, representative lighting/distance/pose conditions, and a validated detector/crop pipeline. The existing synthetic-target test is intentionally non-identifying.",
        "",
        "## Reference sources",
        "",
        "- Cheng, Zhu, and Gong, [Surveillance Face Recognition Challenge (QMUL-SurvFace)](https://arxiv.org/abs/1804.09691), 2018. The paper describes native low-resolution surveillance faces and emphasizes the difficulty of domain-matched evaluation; it does not validate this project's crop pipeline or wearer use.",
        "- [Arduino Nicla Vision official documentation](https://docs.arduino.cc/hardware/nicla-vision) and [OpenMV Nicla Vision MicroPython quick reference](https://docs.openmv.io/dev/openmvcam/quickref/arduino-nicla-vision.html) for board and firmware context. This report's memory, timings, detections, and execution claims come from the retained local board logs, not the vendor specifications.",
        "",
        "## Reproduction and retained records",
        "",
        "Run from `/Users/omristeinberg-tatman/Documents/Codex/2026-10-04/ok-x20` with the pinned environment. The added seed plan is `work/model-training/experiments/validation_seed_extension_20261022.json`; the consolidated validation-only scoring plan is `work/model-training/experiments/consolidated_validation_evaluation_20261022.json`. Main, matched-split, and tiny-split plans are beside them. Training histories, run configs, checkpoints, model artifacts, logs, and evaluation JSON files remain under `work/model-training/runs/`.",
        "",
        "```sh",
        "work/model-training/.venv/bin/python work/model-training/scripts/run_training_matrix.py \\",
        "  --plan work/model-training/experiments/validation_seed_extension_20261022.json --max-parallel 3",
        "work/model-training/.venv/bin/python work/model-training/scripts/run_eval_matrix.py \\",
        "  --plan work/model-training/experiments/consolidated_validation_evaluation_20261022.json --max-parallel 1",
        "work/model-training/.venv/bin/python work/model-training/scripts/run_eval_matrix.py \\",
        "  --plan work/model-training/experiments/pilot_validation_evaluation_20261022.json --max-parallel 1",
        "work/model-training/.venv/bin/python work/model-training/scripts/summarize_research_campaign.py \\",
        "  --campaign-dir work/model-training/runs/research_seed_architecture_campaign_20261021 \\",
        "  --also-campaign-dir work/model-training/runs/matched_split_sensitivity_20261021 \\",
        "  --also-campaign-dir work/model-training/runs/tiny_splitB_replication_20261021 \\",
        "  --also-campaign-dir work/model-training/runs/validation_seed_extension_20261022 \\",
        "  --output-prefix work/model-training/reports/research_seed_architecture_campaign_20261021",
        "work/model-training/.venv/bin/python work/model-training/scripts/summarize_research_campaign.py \\",
        "  --campaign-dir work/model-training/runs/seed_sweep_fixed_split_20261021 \\",
        "  --output-prefix work/model-training/reports/pilot_seed_sweep_20261021",
        "```",
        "",
        "Recreate the actual-board headless pipeline using `FaceRecognition/bin/nicla recognize-embedding \"$PORT\"`; no viewer is required. The currently installed model can be rolled back using the exact predeployment `/flash` backups and checksums in `work/model-training/deployment-backup/baseline.json`. The production config, stock ROMFS, original `/flash/main.py`, and README were preserved; no commit or push was made.",
        "",
    ]
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines))
    print(json.dumps({"output": str(output), "evaluated_runs": len(rows), "groups": sorted(groups)}, sort_keys=True))


if __name__ == "__main__":
    main()
