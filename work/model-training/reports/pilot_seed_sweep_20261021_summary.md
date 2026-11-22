# Multi-seed architecture campaign summary

Completed evaluated runs: 5
Cumulative recorded wall-clock duration: 1.0 hours (includes image loading and overnight idle/sleep gaps; not CPU time or a throughput measurement).

Metrics are validation-only. Each checkpoint was chosen by validation ROC AUC, so these are model-selection results, not unbiased test estimates. Current INT8 exports use representative calibration images from training identities only; validation identities are excluded from the calibration pool. The provided final pair annotations were not loaded.

## Group summaries

| Model / split | Seeds (n) | INT8 AUC mean +/- SD (range) | INT8 EER mean +/- SD (range) | INT8 TAR at validation 1% FAR threshold | INT8 bytes mean +/- SD |
|---|---:|---:|---:|---:|---:|
| base/split-20261021 | 20261022, 20261023, 20261024, 20261025, 20261026 (5) | 0.8795 +/- 0.0109 (0.8621 to 0.8896) | 0.2011 +/- 0.0113 (0.1915 to 0.2189) | 0.3157 +/- 0.0467 (0.2528 to 0.3783) | 26872 +/- 0 (26872 to 26872) |

## Per-run INT8 results

| Model | Split | Train seed | Best epoch / completed | AUC | EER | TAR at 1% FAR threshold | FAR | Bytes |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| base | 20261021 | 20261022 | 14 / 19 | 0.8821 | 0.1962 | 0.3783 | 0.0094 | 26872 |
| base | 20261021 | 20261023 | 9 / 14 | 0.8621 | 0.2189 | 0.3170 | 0.0094 | 26872 |
| base | 20261021 | 20261024 | 23 / 28 | 0.8770 | 0.2057 | 0.2943 | 0.0094 | 26872 |
| base | 20261021 | 20261025 | 24 / 29 | 0.8869 | 0.1915 | 0.3358 | 0.0094 | 26872 |
| base | 20261021 | 20261026 | 28 / 30 | 0.8896 | 0.1934 | 0.2528 | 0.0094 | 26872 |

## Interpretation limits

- The grouped seed counts (10 or 15, depending on model/split) describe run-to-run variability on those fixed splits; they do not create independent test sets.
- The second identity split provides a limited split-sensitivity check, not broad domain generalization.
- Recorded runtime is wall-clock time and includes image loading and any overnight idle/sleep gaps; do not compare it as active training compute.
- Pair-level bootstrap resamples pairs independently even though identities/images recur; intervals can be too narrow.
- The historical seed pilot uses a different TensorFlow thread configuration and is kept separate from this standardized campaign.
- See each run's `float_int8_evaluation.json` for bootstrap intervals, thresholds, timing, quantization metadata, operators, and model hashes.
