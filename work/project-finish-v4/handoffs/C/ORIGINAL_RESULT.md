# Job C — detector training and cropping result

**Status:** Host training, TFLite export audit, full-scene development evaluation, one frozen final evaluation, and raw-fixture audit are complete. Physical Nicla tests remain pending because no Nicla is attached to this Mac.

## Decision

Keep the stock RGB FOMO detector at threshold **0.40** and the app’s current fixed **96 × 96 px** crop as the functional reference. This is continuity with the existing app, not a claim that its false-trigger rate or crop coverage is acceptable: the stock reference triggered on 15/100 development negatives and 8/50 final negatives, and its final face-center recall was 6.6%.

The compact separable-convolution detector is the best learned research challenger: root `20261102`, full-INT8 TFLite, threshold **0.70**, candidate 96 px crop. Its development F1 is higher than stock, but it triggered on 72/100 development negatives and 37/50 final negatives. Only 7/267 final scenes reached `ONE_FACE_CROP_READY`; it is **not deployable**. No learned threshold in the tested 0.30–0.70 grid met the ≤5% false-trigger target. The MobileNetV2 alpha 0.1 arm produced no detections in any of its ten roots.

The 96 px native crop was the best balanced tested side among 80, 96, 112 and 128 px for face-box coverage. Development evidence supports a *pixel-scale range* of 12–31 px on the 96 px detector input (30–77.5 px in the 240 px camera window). That is not a calibrated distance range. Faces below 12 detector pixels are often missed; faces at 32+ pixels often exceed the fixed crop. See the [development freeze](../../outputs/C/reports/development_freeze.json) and [development-only range addendum](../../outputs/C/reports/development_freeze_addendum.json).

## Data and splits

B’s public-data and role contract was verified. Its staged WIDER FACE images and annotations were indexed into event-disjoint roles: 10,792 train scenes, 303 development scenes and 217 final scenes. There were no decode errors, no cross-role event groups and no exact cross-role duplicates. Training used 10,792 WIDER train scenes plus 100 PASS train negatives; development used 303 WIDER scenes plus 100 PASS negatives. Final images were not loaded or scored until the development shortlist was frozen.

WIDER face boxes were mapped into the centered QVGA camera window. The resulting in-window face-center counts and maximum face-side bins were:

| Role | Scenes | In-window face centers | `<4` | `4–7` | `8–11` | `12–19` | `20–31` | `32+` |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Train | 10,792 | 97,236 | 64,126 | 19,062 | 5,856 | 4,196 | 1,957 | 2,039 |
| Development | 303 | 1,881 | 694 | 648 | 233 | 181 | 69 | 56 |
| Final | 217 | 1,172 | 770 | 188 | 94 | 58 | 32 | 30 |

The three PASS roles contain 100 train, 100 development and 50 final no-face scenes, with disjoint uploader groups and attribution retained. All 200 train/development images were checked in ten rendered contact sheets. One stylized cartoon face-like scene was retained as a conservative hard negative. PASS dataset metadata is CC BY 4.0 and selected Flickr items are CC BY 2.0 Attribution License. WIDER’s official project page does not state terms clearly enough for redistribution, so source images remain local and are excluded from the model package. Exact SHA-256 overlap against B’s identity source, primary, presentation and pretraining manifests was zero. The overlap audit is exact-byte only; it did not run pHash or scene-to-crop perceptual comparison.

Primary sources: [WIDER FACE project](https://mmlab.ie.cuhk.edu.hk/projects/WIDERFace/index.html), [WIDER FACE CVPR 2016 paper](https://openaccess.thecvf.com/content_cvpr_2016/html/Yang_WIDER_FACE_A_CVPR_2016_paper.html), [PASS v3 record](https://zenodo.org/records/6615455), [Oxford PASS page](https://www.robots.ox.ac.uk/~vgg/data/pass/).

## Training and model audit

Both arms trained on full scenes with 96 × 96 grayscale input and a 12 × 12 × 2 background/face centroid grid. The class order is channel 0 background, channel 1 face. INT8 input uses scale `1/255`, zero point `-128` (`q = gray8 - 128`); output uses scale `1/256`, zero point `-128`. Representative images came only from training roles.

| Learned arm | Roots | Parameters | INT8 size | INT8 operators | Training outcome |
|---|---:|---:|---:|---|---|
| MobileNetV2 α=0.1 | 10 | 5,170 | 27,624 B | ADD, CONV_2D, DEPTHWISE_CONV_2D, MUL, PAD, SOFTMAX | No detections; all roots stopped after 5 epochs with best epoch 1. Keras had no α=0.1 ImageNet checkpoint, so weights started randomly. |
| SepConv-16 compact | 10 | 1,127 | 10,160 B | CONV_2D, DEPTHWISE_CONV_2D, SOFTMAX | All ten trained 14 epochs; best epochs were 12 (1), 13 (4), 14 (5). |
| Stock RGB FOMO reference | Fixed artifact | — | 57,040 B | ADD, CONV_2D, DEPTHWISE_CONV_2D, PAD, SOFTMAX | Fixed reference, not a newly trained grayscale model. |

The compact INT8 artifact is 82% smaller than the stock file and uses three operator types, but those size/operator counts are only a resource rationale. Host TFLite medians were 1.07 ms/scene for compact INT8, 4.89 ms/scene for MobileNetV2 INT8 and 9.78 ms/scene for stock; these are **not Nicla timings or arena measurements**. Historical Nicla evidence lists stock FOMO at 237,936 arena bytes and 65.5 ms, with 0/10 current-scene detections. A person detector is not interchangeable with face localization, and the small identity embedder is not a detector.

The Edge Impulse open FOMO route was inspected; its public example targets PyTorch/ExecuTorch `.pte`, not the requested OpenMV TFLite contract. The reproducible local Keras/TFLite route is documented in [`training_method.json`](../../outputs/C/models/trained/training_method.json). H5 files are training checkpoints only and are excluded from the export bundle. Every root’s float and INT8 artifact, label file, shape, signedness, input/output and internal tensor quantization, operators, hashes, history and seed are in [`root_export_audit.json`](../../outputs/C/reports/root_export_audit.json) and [`ten_root_summary.json`](../../outputs/C/models/trained/ten_root_summary.json).

## Development and final evaluation

Thresholds were swept from 0.30 to 0.70, and crop sides from 80 to 128 px. The learned root threshold selector falls back to best F1 only when no threshold satisfies the 5% no-face trigger limit; no learned root satisfied it. The compact arm’s ten-root medians at that fallback were F1 0.331, precision 0.257, recall 0.473 and false-trigger scene rate 77% for INT8. Float median F1 was 0.334 with a 75% false-trigger rate. The MobileNetV2 arm had zero median F1 and zero recall for float and INT8.

The selected challenger is seed `20261102` because it had the highest individual learned INT8 development F1 (0.345) among the roots. At 0.70 it reached 47.1% face-center recall but false-triggered on 72/100 PASS negatives; 323/403 development scenes had multiple detections and were rejected as uncertain. Its candidate 96 px crop covered 61.4% of matched annotated face boxes and 28.9% of all labeled face centers, but only 12/403 scenes reached a single valid crop.

| Frozen final result | Stock RGB FOMO, 0.40, legacy 96 px | Compact INT8 seed 20261102, 0.70, candidate 96 px |
|---|---:|---:|
| Face-center precision | 45.8% | 17.4% |
| Face-center recall | 6.6% | 30.1% |
| Face-center F1 | 0.115 | 0.221 |
| PASS false-trigger scenes | 8/50 (16%) | 37/50 (74%) |
| `ONE_FACE_CROP_READY` scenes | 40/267 (15.0%) | 7/267 (2.6%) |
| Conditional annotated face-box coverage at 96 px | 50.6% | 51.3% |
| End-to-end annotated face-center coverage at 96 px | 3.3% | 15.4% |

The compact model’s final F1 is higher because recall is higher, but its precision, no-face false triggers and one-face crop-ready coverage are worse. Final data was used once for reporting only; no final score changed the frozen model, threshold or crop. The [frozen final report](../../outputs/C/reports/frozen_final_evaluation.json) preserves counts, status denominators, float/INT8 differences, face-size strata and crop metrics.

The D handoff is absent. The available generic 64-D embedding requires enrollment templates and is not a classifier; its training also overlaps four of B’s five identities, so it was excluded from crop selection. Conditional identity accuracy/coverage is **not measured**. Gray mean/stdev gates and annotated-box coverage are input-readiness proxies, not identity results.

## Exact transform and crop contract

The current stock app captures RGB565 QVGA, applies a centered 240 × 240 camera window at sensor origin `(40, 0)`, and uses OpenMV 5.0.1 `Normalization` for the stock RGB FOMO input. The challenger captures the same square window in GRAYSCALE. Rotation and auto-rotation are off; horizontal mirror and vertical flip are false. The camera window maps to 96 × 96 with bilinear resizing; because both regions are square, no letterbox padding is used. Detector coordinates map back to window coordinates by ×2.5 and to QVGA sensor coordinates by adding `(40,0)`.

FOMO output is a stride-8 centroid grid. OpenMV’s FOMO rectangles are artificial NMS suppression regions, not face-size boxes. The current stock crop uses integer center `(x + w//2, y + h//2)` and origin `center − 48`. The challenger candidate rounds the postprocessed center to the nearest integer, then uses a 96 px square. Out-of-window crops are rejected without clamping or padding. On the RGB565 stock path, the native crop is copied to grayscale with `Image.to_grayscale(roi=..., copy=True)`; the gray crop is then converted to the 96 × 96 gray8 identity input using OpenMV `Normalization`. Both native and resized input quality gates require mean 12–243 and standard deviation ≥5. Unsupported transforms, multiple faces, clipped crops or failed gates map to `UNCERTAIN`; no detections map to `NO_FACE`.

The 96 px crop was better balanced than 80, 112 and 128 px on the development split. With the stock legacy crop, 12–19 px faces had 46.4% detection recall and 75.0% conditional full-box coverage; 20–31 px faces had 69.6% recall and 64.6% coverage. At 32+ px, detected faces had only 2.9% conditional full-box coverage. Small faces below 12 px were frequently missed. These ranges are in resized-image pixels, not physical distance.

## Raw fixtures and visual review

The frozen challenger has 13 fixtures: ten seeded transform/fault blocks plus three real development PASS no-face examples. Each fixture records source SHA-256, transform parameters, exact camera-window bytes, exact detector gray8/int8 bytes, exact raw TFLite output bytes, expected OpenMV FOMO detections/crop geometry and the raw gray identity input when a valid crop exists. The audit re-ran genuine TFLite inference against all raw bytes and checked that PGM previews decode back to exactly the stored bytes; no JPEG is used for inference.

I inspected the three rendered contact sheets at original resolution. The baseline contains a valid crop with the annotated face inside it. The rotated, low-light and blurred fixtures returned `NO_FACE`; center jitter returned `UNCERTAIN_CROP_OUTSIDE`; mirror/flip/edge/small/multiple cases returned multiple detections and were held as uncertain. Of the three PASS fixtures, two triggered and one returned `NO_FACE`. See [`fixture_visual_audit.json`](../../outputs/C/reports/fixture_visual_audit.json), [`raw_fixture_audit.json`](../../outputs/C/reports/raw_fixture_audit.json) and [`fixture_manifest.json`](../../outputs/C/fixtures/selected_challenger/fixture_manifest.json).

## F/H integration contract and hardware status

C is face existence/localization only. It does not classify identity or track people. For F, emit an `image-crop-1` record with 96 × 96, row-major `gray8` bytes, `GRAY` channel order, transform ID, crop box in camera-window coordinates, source/detector metadata and `bytes_sha256`. F may invoke D only after exactly one face is detected, the native crop stays inside the frame, and both crop quality gates pass. Map zero detections to `NO_FACE`; map multiple detections, edge clipping, low-quality pixels, unsupported labels/tensors or transforms to `UNCERTAIN`. A valid crop is a `CROP_READY` event. D’s current int8 input path, if retained, quantizes `gray8` as scale `1/255`, zero point `-128`; D’s output states remain `OTHER`, `KNOWN_IDENTITY` or `UNCERTAIN` under A’s labels contract.

The H scripts are ready: [`nicla_fixture_parity.py`](../../outputs/C/board/nicla_fixture_parity.py) checks exact raw input/output bytes, FOMO rectangles and identity crops; [`nicla_scene_crop_smoke.py`](../../outputs/C/board/nicla_scene_crop_smoke.py) runs 20 bounded frames per stock/challenger stage and records model load, arena, capture, inference and crop-gate fields without calling D or tracking. The current port inventory only shows Bluetooth ports. Both physical cells remain **pending**; host timing and historical Nicla measurements are not substitutes.

## Evidence files

- [Ten-root TFLite export bundle](../../outputs/C/models/trained/fomo_tflite_20_roots_and_audit.zip)
- [Selected compact full-INT8 challenger](../../outputs/C/models/trained/cheap_sepconv_16/seed-20261102/detector_full_int8.tflite)
- [Development comparison](../../outputs/C/reports/development_comparison.json), [freeze receipt](../../outputs/C/reports/development_freeze.json), [final evaluation](../../outputs/C/reports/frozen_final_evaluation.json)
- [Detection trade-off figure](../../outputs/C/reports/figures/development_detection_tradeoff.png), [crop coverage figure](../../outputs/C/reports/figures/development_crop_coverage.png), [resource/quality figure](../../outputs/C/reports/figures/development_resource_quality.png)
- Required machine handoff: [`RESULT.json`](../../outputs/C/reports/RESULT.json)

OpenMV preprocessing references: [Normalization](https://docs.openmv.io/v5.0.1/library/omv.ml.preprocessing.html) and [ML model interface](https://docs.openmv.io/v5.0.1/library/omv.ml.html).
