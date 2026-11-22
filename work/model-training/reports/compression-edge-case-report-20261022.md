# Nicla Vision Compression and Edge-Case Validation

**Date:** 2026-10-22  
**Verdict:** **Partially functional**  
**Scope:** Model-format ranking, pruning/clustering/QAT, input robustness, board compatibility, firmware guard edges, storage cleanup, and final deployment verification.

## Executive result

The local face-verification prototype is built and executable on the Nicla Vision. The selected base full-INT8 model and standalone FOMO-to-embedding application run on the board. This follow-up exercised all 14 compression candidates on the actual Nicla, tested 80 synthetic numeric inputs, evaluated 45 controlled perturbation/model combinations, reran 14 host edge-case tests, and passed eight on-device MicroPython guard categories.

The evidence supports a balanced compact baseline, not production-ready face recognition:

- **Best balanced deployable choice remains base full INT8:** 26,872 bytes, 51,040-byte tensor arena, 12.692 ms median embedding inference on the Nicla, validation AUC 0.8723.
- **Smallest and fastest passing candidate is tiny full INT8:** 15,632 bytes, 31,392-byte arena, 7.567 ms median; validation AUC falls to 0.8277 and EER rises to 0.2604.
- **Highest observed validation AUC is 50% magnitude-pruned dense full INT8:** 0.8790, but it has the same file size, arena, and board latency as the unpruned baseline. This is a one-seed, five-epoch compression fine-tune, not independent evidence to replace the more extensively trained base.
- **Sparse encodings, FP16-weight exports, and dynamic-range INT8-weight exports failed to load on this firmware**, all reporting “Failed to allocate tensors.” Sparse encodings also increased file size and add DENSIFY to the graph.
- The latest post-guard live sample had 6 no-face frames and 6 uncertain, edge-clipped crops; the embedder was not invoked. It is not a camera-domain recognition result.
- No consented real-person enrollment template is present. USB power path validation remains unresolved, so no power or battery-life claim is made.

No camera images were saved, the Codex viewer stayed closed, verification-pair labels and Olivetti were not used in this study, and the stock 4 MiB ROMFS was not modified.

This supplement extends the separate research campaign of 45 standardized validation-only training runs plus a five-run pilot. That campaign also smoke-tested 120 unique trained float/INT8 exports on the Nicla, with three repeated inferences per export and 120/120 passing. The earlier board matrix was a runtime-only probe on one centered crop per candidate; it did not measure identity accuracy. Its 41 base-INT8, 26 tiny-INT8, 31 base-float, and 22 tiny-float variants are documented in the seed matrix report and raw board logs.

## Test environment and evaluation protocol

- **Board:** Arduino Nicla Vision, NICLAV H7 1024, board ID 32393931353251180037002E.
- **Firmware/runtime:** OpenMV 5.0.1, MicroPython 1.28.0.
- **Connected port:** /dev/cu.usbmodem3270395F35322.
- **Host:** Intel MacBook Pro, macOS Darwin 25.6.0, Python 3.9.6 in the isolated training environment, TensorFlow 2.10.1 CPU, 4 TensorFlow threads. TFMOT 0.7.5 was installed and pinned for the advanced strategy sweep.
- **Development validation:** 2,120 identity-disjoint pairs: 1,060 positive and 1,060 negative, 3,196 unique face records across 1,060 held-out identities. Pair reuse means pair-bootstrap intervals may be optimistic.
- **Selection boundary:** Compression metrics use the existing development validation cohort. The final SurvFace verification-pair labels remained closed and Olivetti was not loaded. Thresholds were recalibrated per candidate to target 1% validation FAR; this is model selection evidence, not an independent final estimate.
- **Data audit:** From 220,888 labeled training records, the upstream duplicate audit excluded 268 conflicting-label records in 134 exact-hash groups, collapsed 6 same-identity duplicates, and excluded 3 records overlapping unlabeled holdout images. It retained 220,611 records across 5,319 identities. The compression fine-tunes used the selected 44,897-record, 4,259-identity training subset; the 1,060 validation identities did not overlap.
- **Quantization calibration:** Each pool had 512 actual representative images, drawn only from exact selected training records after duplicate exclusion and the per-identity cap. Image hashes were rechecked with zero mismatches and zero validation-identity overlap. Final pair labels and Olivetti were not accessed. The two privacy-safe pool digests are recorded below; the record/identity manifest itself remains local and is not distributed.

The post-training quantization pool manifest digest is `146ea45dc92b06a1796166a180ffecb67dda443dbff16f361391d11257fbd242`; the pruning/clustering/QAT calibration pool digest is `8b0d8c7b9fee99803e599d79a7905e99e27ff47557fe725074c0f5c66deb21aa`. Both were drawn from training selection digest `c70961a7c81d39044f468f8d2a64a04bc66c838049d19261669adc2675017e88`.

## Standard export comparison

Every model emits a 64-dimensional embedding. The eight standard exports compare base and narrower-channel tiny architectures across four TFLite representations. AUC confidence intervals are from the development-validation bootstrap. FAR/TAR values use each model's own validation threshold selected for the 1% FAR target.

| Candidate | TFLite bytes | Validation AUC (95% CI) | EER | TAR / FAR at target threshold | Nicla |
|---|---:|---:|---:|---:|---|
| Base float32 | 51,912 | 0.871881 [0.856971, 0.884534] | 0.209434 | 0.308491 / 0.009434 | Pass |
| Base float16 weights, float I/O | 32,176 | 0.871869 [0.856967, 0.884526] | 0.210377 | 0.307547 / 0.009434 | Fail |
| Base dynamic INT8 weights, float I/O | 27,280 | 0.871874 [0.856984, 0.884587] | 0.211321 | 0.309434 / 0.009434 | Fail |
| **Base full INT8 I/O** | **26,872** | **0.872270 [0.857400, 0.884721]** | **0.210377** | **0.307547 / 0.009434** | **Pass** |
| Tiny float32 | 23,384 | 0.828674 [0.811544, 0.845478] | 0.257547 | 0.183962 / 0.009434 | Pass |
| Tiny float16 weights, float I/O | 17,912 | 0.828628 [0.811548, 0.845424] | 0.257547 | 0.185849 / 0.009434 | Fail |
| Tiny dynamic INT8 weights, float I/O | 17,320 | 0.828722 [0.811619, 0.845533] | 0.257547 | 0.186792 / 0.009434 | Fail |
| Tiny full INT8 I/O | 15,632 | 0.827696 [0.810605, 0.844445] | 0.260377 | 0.203774 / 0.009434 | Pass |

The base full-INT8 export is 48.2% smaller than base float32. On this board it also reduced the embedding arena by 72.7% and median embedding latency by 78.7%. The tiny full-INT8 export is 41.8% smaller than base full INT8, with a 38.5% smaller arena and 40.4% lower median embedding latency, at a 0.0446 lower validation AUC and 0.0500 higher EER.

The smaller tiny architecture is the only tested strategy that reduced actual dense model structure and board resource use. It is a viable memory/latency point only if its lower validation quality is acceptable for a future, independently evaluated application.

## Pruning, structured sparsity, clustering, and QAT

TFMOT 0.7.5 ran the following four methods using the same identity-disjoint split and training-only record selection. Each method fine-tuned for five epochs, 200 balanced steps per epoch, with training seed 20261022. This is a one-seed compression pilot, not a multi-seed accuracy study. Runtime model selection did not touch final pair annotations.

| Strategy/export | Bytes | Validation AUC (95% CI) | EER | Board result | Board arena / median |
|---|---:|---:|---:|---|---|
| Base full-INT8 reference | 26,872 | 0.872270 [0.857400, 0.884721] | 0.210377 | Pass | 51,040 B / 12.692 ms |
| 50% magnitude pruning, dense INT8 | 26,872 | **0.879018 [0.865065, 0.891978]** | **0.205660** | Pass | 51,040 B / 12.690 ms |
| 50% magnitude pruning, sparse INT8 encoding | 33,080 | 0.879018 [0.865065, 0.891978] | 0.205660 | Fail | Allocation failure |
| 2:4 structured pruning, dense INT8 | 26,872 | 0.873823 [0.858876, 0.887765] | 0.207547 | Pass | 51,040 B / 12.690 ms |
| 2:4 structured pruning, sparse INT8 encoding | 32,128 | 0.873823 [0.858876, 0.887765] | 0.207547 | Fail | Allocation failure |
| 16-cluster weight clustering, dense INT8 | 26,872 | 0.866015 [0.850654, 0.880352] | 0.219811 | Pass | 51,040 B / 12.689 ms |
| QAT full INT8 | 27,888 | 0.866493 [0.850451, 0.880249] | 0.212264 | Pass | 51,040 B / 12.695 ms |

Pruning produced about 45.13% zeros among the measured rank-2-or-higher weights for magnitude pruning and 44.48% for 2:4 pruning. Ordinary dense INT8 exports still occupy exactly 26,872 bytes and show no arena or inference-time reduction. Sparse encoding increased the FlatBuffer by 23.1% for magnitude pruning and 19.6% for 2:4 pruning; the sparse graphs include DENSIFY and both failed to allocate tensors on the Nicla. Zero-valued weights alone did not yield a smaller deployed dense model.

Clustering did not emit a packed cluster-code model in this deployment path: its dense full-INT8 export is the same 26,872 bytes and same measured board resource profile. QAT exported 1,016 additional bytes versus the base reference, with slightly lower validation AUC; it did not improve measured board inference.

**Ranking interpretation.** The magnitude-pruned candidate has the highest observed AUC on this cohort, but its confidence interval overlaps the baseline interval and its fine-tuning is only five epochs on one seed. It has no deployed-size or latency advantage. Base full INT8 remains the balanced selected model because it has the broader training/seed history and the same practical board cost. Tiny full INT8 is the clear size/latency alternative. These results do not establish statistically significant ranking differences.

## Numeric and image-perturbation stress

### Numeric boundary inputs

Ten controlled 96×96 grayscale inputs were applied to each of eight standard formats: all black, near-black value 1, midgray 127 and 128, all-white, full-range horizontal ramp, 4-pixel checkerboard, seeded uniform noise, and centered square on black and white backgrounds. **All 80/80 model-input runs produced finite, nonzero, normalized embeddings.** These cases probe numeric stability only; they do not evaluate identity recognition.

### Controlled image perturbations

Nine transforms were run against five candidate variants (base float32, base float16-weight, base dynamic-INT8-weight, base full INT8, and tiny full INT8), each on the same 2,120 validation pairs: contrast 0.5 and 1.5; downsample-to-32 and downsample-to-48 then resize; gamma 0.6 and 1.4; Gaussian blur radius 1; center-band occlusion; and a 6-pixel right shift. All 45 candidate-condition runs completed. The table reports the base full-INT8 results; the complete 45-condition metrics remain in the raw result JSON.

| Transform | AUC | EER | Mean embedding cosine vs clean |
|---|---:|---:|---:|
| Contrast 0.5 | 0.789730 | 0.298113 | 0.830777 |
| Contrast 1.5 | 0.846457 | 0.234906 | 0.901538 |
| Downsample to 32, then resize | 0.867968 | 0.216038 | 0.992306 |
| Downsample to 48, then resize | 0.870324 | 0.215094 | 0.996070 |
| Gamma 0.6 | 0.813917 | 0.263208 | 0.808220 |
| Gamma 1.4 | 0.821458 | 0.258491 | 0.875872 |
| Gaussian blur radius 1 | 0.871299 | 0.214151 | 0.998803 |
| Center-band occlusion | 0.851016 | 0.233019 | 0.944633 |
| Shift right 6 pixels | 0.833798 | 0.250000 | 0.800834 |

The metric helper recalibrated the 1%-FAR decision threshold within each transformed condition. Therefore these numbers describe ranking and separability under each controlled perturbation; they are not fixed-threshold robustness or measured camera-domain performance. Lower contrast, exposure changes, occlusion, and translation stress the representation more than modest blur or downsampling.

## Actual Nicla format matrix

Fourteen files totaling 386,200 bytes were staged one at a time under temporary /flash/qa_compression_20261022/; each candidate received five same-input inference repeats. The passing outputs had zero maximum absolute difference across repeats. Board candidate probes are runtime-only checks, not board-side accuracy tests.

This 14-format study is separate from the earlier 120-export seed/architecture smoke matrix above. Prior stock-board checks also covered the ROM FOMO face model, the person detector, force-quant and audio/speech model inputs, and three Haar cascades. Those were model-load or synthetic-input runtime probes; BlazeFace and YOLO-LC failed tensor allocation, and no identity accuracy was inferred from these checks.

| Candidate | Bytes | Model load | Board | Tensor arena | Median embedding | Minimum GC free during inference / at failure |
|---|---:|---:|---|---:|---:|---:|
| Base float32 | 51,912 | 6.038 ms | Pass | 186,752 B | 59.462 ms | 92,560 B |
| Base float16 weights | 32,176 | -- | Fail: allocation | -- | -- | 38,144 B at failure |
| Base dynamic INT8 weights | 27,280 | -- | Fail: allocation | -- | -- | 43,040 B at failure |
| Base full INT8 I/O | 26,872 | 5.608 ms | Pass | 51,040 B | 12.692 ms | 254,960 B |
| Tiny float32 | 23,384 | 5.052 ms | Pass | 113,024 B | 27.558 ms | 194,816 B |
| Tiny float16 weights | 17,912 | -- | Fail: allocation | -- | -- | 52,400 B at failure |
| Tiny dynamic INT8 weights | 17,320 | -- | Fail: allocation | -- | -- | 52,976 B at failure |
| Tiny full INT8 I/O | 15,632 | 5.044 ms | Pass | 31,392 B | 7.567 ms | 285,824 B |
| Magnitude-pruned dense INT8 | 26,872 | 5.629 ms | Pass | 51,040 B | 12.690 ms | 254,928 B |
| Magnitude-pruned sparse INT8 | 33,080 | -- | Fail: allocation | -- | -- | 37,232 B at failure |
| 2:4-pruned dense INT8 | 26,872 | 5.632 ms | Pass | 51,040 B | 12.690 ms | 254,928 B |
| 2:4-pruned sparse INT8 | 32,128 | -- | Fail: allocation | -- | -- | 38,192 B at failure |
| Cluster-16 dense INT8 | 26,872 | 5.644 ms | Pass | 51,040 B | 12.689 ms | 254,928 B |
| QAT full INT8 | 27,888 | 5.675 ms | Pass | 51,040 B | 12.695 ms | 253,952 B |

The six failures each returned Failed to allocate tensors. The evidence does not isolate whether each failure is due to operator support, transient memory, or another runtime allocation constraint. FP16 graphs contain DEQUANTIZE, sparse exports contain DENSIFY, and all successful INT8 paths used the pinned firmware runtime. No candidate load failure was silently treated as unsupported solely from its host operator list.

For the signed full-INT8 candidates, input shape is [1,96,96,1]; output shape is [1,64]. Inputs use signed INT8 with scale near 1/255 and zero point -128; output quantization scale/zero point are model-specific and are preserved in the full JSON records. Float32 candidates use float I/O. Dynamic-range INT8-weight and FP16-weight exports use float I/O and failed on the board.

### Resource and storage cleanup

- The existing stock ROMFS remained read-only at 4,194,304 bytes and its 14 original entries remained present.
- The compression matrix reported /flash total 15,695,872 bytes and free 15,638,528 bytes before staging. After deleting the QA directory, it again reported 15,638,528 bytes free; qa_compression_20261022 was absent.
- A separate firmware-guard probe also removed its temporary module and JSON files. It confirmed the standalone startup could be requested after QA.
- The five-repeat matrix includes embedding model arena and Python GC free memory; these are not whole-system peak RAM or end-to-end pipeline latency.
- No ROMFS upload, bootloader change, model erase, or power-meter operation was part of this matrix.

## Firmware edge cases and deployed guard update

### Host test suite

The rerun is saved in firmware_edge_case_tests_20261022.txt: **14 tests passed**.

1. Crop exactly at image bounds and one pixel outside each edge.
2. FOMO output labels/index layouts and ambiguous layouts.
3. Empty detector output and bounded handling of multiple detections.
4. Frame pipeline outcomes for no face, multiple faces, clipped crop, low quality, and detector error.
5. Missing template file.
6. Valid template normalization and pseudonymous identifier checks.
7. Malformed, duplicate, oversized, path-traversal, wrong-shape, zero, NaN, infinity, string and boolean vector rejection.
8. Calibration thresholds at -1, 0, 1 and rejection of out-of-range, bool, wrong-type, null and non-finite values.
9. Embedder vector size, zero/NaN and model contract guards.
10. Enrollment mean behavior for empty inputs, wrong shape, non-finite values and cancellation to zero.
11. Enrollment request bounds, types, replace flag and safe IDs.
12. Known, unknown, uncertain and no-threshold decision behavior.
13. Atomic template write/replace and rollback after simulated failed commit.
14. Brightness/contrast low-quality boundaries short-circuit before embedding.

The suite uses mocked camera/model APIs; it validates helper behavior, not the camera optics or identity accuracy.

### On-device guard probe

The actual OpenMV/MicroPython probe passed all eight categories on the Nicla:

- finite numeric parsing;
- bounded ASCII pseudonymous IDs;
- exact crop bounds and out-of-frame rejection;
- FOMO output-index layouts;
- template schema, duplicate IDs, dimensions, vector norm, and traversal protection;
- calibration range and type guards;
- enrollment request types and mean-vector edge cases;
- temporary on-device template write/readback round trip.

It recorded camera_accessed=false, saved no image, cleaned temporary JSON files and /flash/qa_enroll_identity.py, rechecked storage/ROMFS, and requested a standalone restart. The final current board report is FaceRecognition/reports/nicla_edge_guard_probe_20261022T130455.jsonl.

### Deployed firmware guard

The updated recognize_embedding.py was backed up before replacement. The board readback SHA-256 exactly matched the local source:

- Prior deployed file: 13,894 bytes, SHA-256 9a1ddadd393d6d32f82e9aea114e6ece71aabdfa0e7ce852d5385ee1ae6ce87f.
- New guarded file: 16,394 bytes, SHA-256 1cc5b8240c3a773cfe4a42d085383b120089feb9d86e55ba9321818744fa0f30.
- Previous file backup: work/model-training/deployment-backup/edgeguards/20261022T1246Z/recognize_embedding.py.pre_edge_guard.
- Update/readback/rollback manifest: work/model-training/deployment-backup/edgeguards/20261022T1246Z/deployment-update-manifest.json.

After update, software reset and standalone restart succeeded. Twelve subsequent status-only frames had 6 no-face and 6 uncertain results due to crop_clipped_by_frame_edge; no pipeline errors occurred, and the embedding model was not invoked in those frames. The user may have held the synthetic target, but this sample cannot establish exact target alignment. No image or identity output was recorded. The viewer remained closed.

The preceding broader live integration run, recorded in the original deployment report before the guard-only source update, used the user-held synthetic screen target for 20 frames: FOMO detected a region in 18/20, the embedder ran in 13, five detections were rejected as edge-clipped, and two frames returned no-face. It produced no pipeline errors or resets, and all 13 embedding results correctly remained no-enrollment because no template existed. Capture was 34.070-81.741 ms, detector inference 65.498-66.335 ms, and the minimum reported Python heap was 342,896 bytes. That demonstrated the detector-to-embedder integration on a synthetic target, not identity recognition. The new post-guard sample now shows why alignment needs another centered-target run.

## Recommendation and unresolved work

**Selected deployment:** Keep the base full-INT8 model for now. It is the baseline with broader prior seed evidence, fits the board, and runs at 12.692 ms median embedding latency. The five-epoch magnitude-pruned candidate merits a longer multi-seed follow-up only if its slight validation improvement persists; it currently offers no resource reduction.

**Alternate constrained candidate:** Tiny full INT8 is the only measured option that materially reduces model size, arena, and inference latency, but it has lower host verification metrics. A product decision would need an explicit acceptable accuracy/latency trade-off and camera-matched testing.

**Not validated yet:**

- Current camera framing and the stock FOMO centroid-to-fixed-crop geometry. The latest 12 frames show that every accepted detector region was too close to a frame edge for a safe crop.
- A live end-to-end embedding on a centered, non-clipped synthetic target after the guard update.
- Camera-domain verification accuracy under real operating light, distance, orientation, blur, and motion.
- Consented wearer enrollment. No identity template was installed, so no user/partner recognition claim is supported.
- Full power-off/unplugged-host cold start after this latest guard update.
- Valid power/current/energy measurement. The FNIRSI path previously read 0 V during board activity; there is no verified electrical load measurement.
- Dataset provenance and redistribution rights. Keep source data, checkpoints, models, embeddings, and enrollment data local and do not redistribute them.

The most useful next step is a bounded live run with the synthetic face fully centered and in-frame, requiring the embedder path to execute without clipping, followed by consented camera-matched validation before any person-specific claim. The project verdict stays **partially functional**.

## Reproduction and rollback

Run from the ok-x20 project root:

    work/model-training/.venv/bin/python work/model-training/scripts/compression_sweep.py --output-dir work/model-training/compression/strategy_sweep_20261022

    work/model-training/.venv/bin/python work/model-training/scripts/tfmot_strategy_sweep.py --epochs 5 --output-dir work/model-training/compression/advanced_strategy_sweep_corrected_20261022

    work/model-training/.venv/bin/python work/model-training/scripts/tfmot_strategy_sweep.py --epochs 5 --only-strategy qat_full_int8 --output-dir work/model-training/compression/qat_strategy_sweep_20261022

    work/model-training/.venv/bin/python work/model-training/scripts/audit_compression_calibration.py
    work/model-training/.venv/bin/python work/model-training/scripts/firmware_edge_case_tests.py
    FaceRecognition/bin/nicla compression-matrix /dev/cu.usbmodem3270395F35322
    FaceRecognition/bin/nicla edge-probe /dev/cu.usbmodem3270395F35322

The board matrix and guard probe temporarily stage only QA model/code/JSON files under /flash/qa_*, record cleanup, and restart the standalone app. Do not run both serial probes at once.

To restore the previous guard module, first stop the running application and copy the local backup over /flash/recognize_embedding.py using the existing mpremote fs cp workflow. Then reset/restart the board. The exact prior file is preserved in the backup path above; verify against its SHA-256 before relying on rollback. Restore /flash/main.py.stock with FaceRecognition/bin/nicla restore-startup PORT only if the standalone startup itself must be rolled back. ROMFS was never replaced.

## Machine-readable evidence

The full aggregate and per-candidate evidence is preserved in these local files:

- work/model-training/compression/strategy_sweep_20261022/compression_robustness_results.json - eight standard model formats, AUC/EER/TAR/FAR and bootstrap CIs, operators, quantization, host timing, all input-stress and perturbation results.
- work/model-training/compression/advanced_strategy_sweep_corrected_20261022/advanced_compression_results.json - pruning, structured sparsity, clustering, dense/sparse encodings, training history and validation bootstrap.
- work/model-training/compression/qat_strategy_sweep_20261022/advanced_compression_results.json - QAT history, export, validation and board outcome.
- work/model-training/compression/strategy_sweep_20261022/nicla_compression_matrix_20261022.jsonl - actual 14-candidate board load, five-repeat timing, arena, heap and cleanup records.
- work/model-training/compression/strategy_sweep_20261022/nicla_edge_guard_probe_pass_20261022.jsonl - on-device guard result.
- work/model-training/compression/strategy_sweep_20261022/nicla_post_guard_startup_frames_20261022.jsonl - 12 live status-only records; no images.
- work/model-training/compression/strategy_sweep_20261022/final_compression_summary_20261022.json - compact combined summary.
- work/model-training/compression/strategy_sweep_20261022/representative_pool_audit.json - private local representative-sample audit; do not distribute the identity/record manifest.
- work/model-training/reports/firmware_edge_case_tests_20261022.txt - durable host unit-test output.
- work/model-training/deployment-backup/edgeguards/20261022T1246Z/deployment-update-manifest.json - prior/new deployed source hashes and readback verification.

The source distribution is private/local research material. SurvFace source materials are described as academic-research-only, and transformed archive provenance/reuse permission remains uncertain. Olivetti archive-level provenance/license is also incomplete. Keep raw images, manifests containing identity links, model weights, checkpoints, embeddings, and enrollment data local.

## Fresh Nicla board retest — 2026-10-22 20:19–20:33 UTC

A second headless retest was run on the connected NICLAV H7 with the camera viewer closed. All numeric serial output is saved in `outputs/Nicla_Board_Retest_2026-10-22_201950Z/`.

- Rechecked 120 unique trained TFLite exports, with three same-input board inferences per export: 120 passed, 0 failed. Variant median inference: base INT8 12,687 us; tiny INT8 7,555 us; base float 59,392 us; tiny float 27,484 us.
- Rechecked all 14 compression candidates: 8 loaded and inferred, 6 failed tensor allocation. All five-repeat timings, model bytes, arenas, minimum heap values, quantization metadata, cleanup, and errors are in the raw JSONL.
- The board probe exited 0; its host wrapper exited 1 by design because it requires all 14 candidates to pass. It recorded every row, removed its temporary directory, and restored free flash to baseline.
- Rechecked stock/custom model compatibility: 9/11 TFLite models inferred; stock BlazeFace and YOLO-LC failed tensor allocation. All three Haar cascades loaded and ran.
- All 8 on-device guard categories passed; all 14 host firmware edge-case tests passed.
- Direct deployed INT8 smoke: 100/100 inferences completed; mean inference 12,922 us, mean capture 54,130 us, minimum heap 278,752 B.
- Live scene repeat: FOMO detected no face in 0/10 frames; the person classifier labeled 10/10 frames as person. Full FOMO-to-embedding pipeline returned no_face in 12/12 frames, made 0 embedder calls, and had 0 pipeline errors. No identity or camera-domain accuracy claim is supported by this sample.
- Final inventory: 15,634,432 B free /flash both before and after tests; 4 MiB ROMFS and all 14 stock entries remain; no QA files remain. The standalone reset command exited 0.
- No camera images or embeddings were saved.

The timestamped retest bundle contains the full board inventories, every model row and per-repeat timing, live status records, camera/IMU measurements, edge-test output, and a SHA-256 manifest. Its README and machine-readable summary are the convenient entry points.
