# Job E result

Status: **complete_verified**

Job E completed the predeclared Mac-host capacity and compression study using only E-owned roots and reports. The shared source D roots were read as parents; no other job's files were changed.

## Capacity decision

The development-only selector chose MobileNetV1 alpha **0.25**. Its full-INT8 means were macro utility 0.503, known correct acceptance 0.040, and unknown FPIR 0.034. The selector's eligibility gate is recorded in `CAPACITY_SELECTION.json`; the larger widths failed its FPIR ceiling.

| Width | Roots | Utility | Correct acceptance | Unknown FPIR | Full-INT8 bytes |
|---:|---:|---:|---:|---:|---:|
| 0.1 | 10 | 0.501 | 0.016 | 0.013 | 99064 |
| 0.25 | 10 | 0.503 | 0.040 | 0.034 | 327072 |
| 0.5 | 10 | 0.549 | 0.154 | 0.055 | 1024360 |
| 0.75 | 10 | 0.575 | 0.220 | 0.070 | 2109704 |

### Representative-pool sensitivity

Ten independent train-only intensity/sharpness-stratified pools were converted and separately calibrated. Across pools: utility 0.522 (SD 0.006), known correct acceptance 0.054 (SD 0.016), and unknown FPIR 0.010 (SD 0.007).

The .50 and .75 RGB-to-grayscale transfers used official MobileNetV1 checkpoints, first-convolution channel summation and internal [-1,1] preprocessing; the E pilot verified numerical equivalence. Width-specific calibration was regenerated for every export.

## Matched optimization results

All thirteen arms were run over ten matched roots at 75 updates per root. The CSV reports raw root metrics; the paired-difference table compares each arm with the ordinary equal-fine-tuning baseline. TFLite file bytes, checkpoint artifact bytes, static activation estimates, MAC counts, host timing, and training duration are reported separately in the export-format summary.

| Method | Roots | Utility | Unknown FPIR | INT8 bytes | Peak activation estimate (B) | Host p50 (ms) | Unsupported/failed |
|---|---:|---:|---:|---:|---:|---:|---:|
| large_dense_equal_finetune | 10 | 0.502 | 0.032 | 327072 | 75280 | 9.184 | 0 |
| large_qat | 10 | 0.499 | 0.002 | 327072 | 75280 | 9.017 | 0 |
| large_magnitude_prune_50 | 10 | 0.502 | 0.039 | 327072 | 75280 | 8.973 | 0 |
| large_magnitude_prune_75 | 10 | 0.499 | 0.034 | 327072 | 75280 | 9.058 | 0 |
| large_structured_prune_2of4 | 10 | 0.510 | 0.037 | 327072 | 75280 | 9.132 | 0 |
| large_physical_channel_keep_50 | 10 | 0.504 | 0.052 | 280248 | 75280 | 8.686 | 0 |
| small_a050_dense_equal_finetune | 10 | 0.564 | 0.054 | 1024360 | 150560 | 30.065 | 0 |
| large_cluster_16 | 10 | 0.508 | 0.035 | 327712 | 75280 | 9.129 | 0 |
| small_a010_dense_equal_finetune | 10 | 0.497 | 0.016 | 99064 | 37296 | 2.453 | 0 |
| small_a010_distilled | 10 | 0.492 | 0.019 | 99064 | 37296 | 2.465 | 0 |
| large_head_32_sliced_initialization | 10 | 0.504 | 0.041 | 316296 | 75280 | 9.134 | 0 |
| large_head_32_reset_initialization | 10 | 0.495 | 0.040 | 316296 | 75280 | 9.119 | 0 |
| large_head_direct_gap_logits | 10 | 0.510 | 0.036 | 309472 | 75280 | 9.129 | 0 |

## Limitations recorded

- The .75 teacher failed the predeclared open-set usefulness gate. Its .10 student distillation arm is fully run but marked diagnostic; it does not establish a useful strong-teacher distillation result.
- TFMOT 0.7.5 clustering was applied to supported Conv2D kernels (16 centroids). Depthwise kernels remain unclustered because the installed registry does not support them. The stripped TFLite files are ordinary dense exports; no codebook packing or runtime speedup is claimed.
- Magnitude and 2:4 pruning are reported with observed zero fractions and actual FlatBuffer bytes/operators. Zero counts alone are not treated as memory or cycle savings.
- Physical channel surgery changed graph tensor shapes and retained a masked-parent numerical-equivalence check. Static activation peaks are host-side tensor-lifetime estimates, not arena/SRAM measurements.
- Robustness curves use synthetic low-light, Pillow GaussianBlur radius 0.8/1.6 pixel approximations, and center-crop/scale perturbations on development roles. They are not camera-condition results.
- Dynamic batch-32 inference was compared with batch-1 on 350 development/unknown-development inputs for the selected alpha .25 root: zero class or OTHER-decision disagreements; the maximum logit difference was 0.0261. The audit is in `runs/E/robustness/BATCH_INFERENCE_EQUIVALENCE.json`.
- No physical Nicla, camera, ROMFS or power measurement was available in this E lane. Hardware feasibility and energy remain pending H/device work.
- Final-role data were not accessed.

## Files

- `FHI_ARTIFACT_ROOTS.json`: every capacity and optimization root, successful export and downstream-ready full-INT8 host artifact.
- `E_ANALYSIS_SUMMARY.json`, `E_ROOT_METRICS.csv`, `E_PAIRED_DIFFERENCES.csv`: per-root and paired analysis.
- `E_METHOD_SUMMARY.csv`, `E_ROBUSTNESS_SUMMARY.csv`: per-method uncertainty and corruption curves by condition.
- `E_EXPORT_FORMAT_SUMMARY.csv`, `E_EXPORT_FORMATS.csv`: file sizes, weights artifacts, activations, MACs, host timings and training costs.
- `E_DEVELOPMENT_PARETO.json`, `E_QUALITY_VS_SIZE.png`: development-only quality/size shortlist.
- `runs/E/representative_pools/.../RESULT.json`: ten independent calibration-pool receipts.
- `runs/E/robustness/RESULT.json`: synthetic corruption results across model roots.

Generated UTC: 2026-10-30T14:12:21Z
