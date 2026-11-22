# Final training and Nicla deployment report

**Date:** 2026-10-22  
**Result:** **Partially functional**  
**Host:** Intel Core i9 MacBook Pro, macOS Darwin 25.6.0, 16 logical CPUs, 64 GiB RAM; Python 3.9.6 training environment, TensorFlow 2.10.1 CPU  
**Board:** Arduino Nicla Vision (`2341:045F`, NICLAV H7 1024), OpenMV 5.0.1, MicroPython 1.28.0, system bootloader v24

The selected compact embedding model trains, exports as full signed INT8, loads
from the Nicla's writable flash, and produces embeddings that closely match the
host TFLite model. The multi-seed research extension now covers 45 standardized
validation-only runs across two model sizes and two identity splits, plus a
separate five-seed pilot; full results and per-run records are in the campaign
appendix and linked readout. The board still runs the original base model; no
candidate from the expanded campaign has been installed or camera-tested. There
is no user or partner enrollment template, so this work does not validate
real-person recognition. Power was not measured reliably.

## Data handling and split

Training and evaluation ran locally. No dataset, image, model, embedding, or
enrollment data was uploaded. Inputs are already-cropped 96×96 grayscale faces;
they do not support training a face detector or measuring full-scene false
detections.

The prepared SurvFace training manifest contains 220,888 records across 5,319
identity labels. Before splitting, the audit excluded 268 records in 134 exact
hash groups that had conflicting identity labels, excluded 3 records whose
hashes also appeared in the unlabeled SurvFace verification-image set, and
collapsed 6 same-identity duplicate copies. It retained 220,611 unique records
and all 5,319 identities. The identity-disjoint, image-count-decile split used
4,259 training identities and 1,060 validation identities. It had zero exact
hash or flagged perceptual-hash candidates crossing the split. The training
loader capped each training identity at 24 images and selected 44,897 images;
balanced batches used 8 identities × 4 images.

The validation set contains 1,060 positive and 1,060 negative pairs. The final
SurvFace verification annotations were held out until after candidate
selection; they contain 5,320 positive and 5,320 negative pairs, with 9,913
unique pair-referenced images. Pair labels were not used to train, choose a
model, select the threshold, or choose board-evaluation samples. Images are
reused across supplied pairs, so pair-level bootstrap confidence intervals may
be optimistic. Confidence intervals below use 100 pair bootstrap replicates.

SurvFace source materials are described as academic-research-only, while the
exact provenance and reuse rights of the transformed local archive are
unresolved. Olivetti archive provenance and license details are also
incomplete. Keep the archives, checkpoints, exported models, and derived
embeddings local; do not redistribute them. See
[`work/dataset-prep/README.md`](../../dataset-prep/README.md) and
[`work/dataset-prep/reports/dataset-audit.md`](../../dataset-prep/reports/dataset-audit.md).
The prepared SurvFace and Olivetti archive SHA-256 values are
`f61b38dd7364e2dd912ec573c5e9180a5de3d68b95b910b4b5dddda42b265fe6` and
`552c65c1b925374ecae40b0808b1c91c280f0d0baa8c74080bc26e4dafdad256`,
respectively.

## Training and model choice

**Historical INT8 calibration caveat (added 2026-10-22).** A methods audit found
that the original converter sampled representative INT8 calibration images from
both training and validation identities, and did not honor exact-duplicate
exclusions or the per-identity training cap. Its validation INT8 metrics below
are retained as historical results, not as strictly held-out validation
estimates. The final-pair annotations were not used for calibration and were
evaluated once as recorded below; they were not reopened for the expanded
campaign. The appended multi-seed campaign and pilot were re-exported using the
exact records selected by each training run, after duplicate filtering and the
per-identity cap. Per-run calibration manifests preserve the representative
record IDs, image SHA-256 values, and full training-selection digest. Corrected
results supersede the old validation-only comparisons for current conclusions.

Both candidates used the same clean split, supervised contrastive loss,
64-dimensional feature output, seed `20261021`, horizontal flips, translations
up to 1.5%, contrast variation up to 12%, and validation-AUC early stopping.
The base model trained for 12 epochs in 446 seconds. The smaller variant
stopped after 10 epochs in 276 seconds.

The base feature extractor uses a 16-channel stride-2 convolution, then
separable-convolution blocks with 24, 32, 48, and 64 channels (strides 2, 2, 2,
and 1), batch normalization and ReLU6, global average pooling, and a biasless
64-value dense head. Supervised contrastive loss uses temperature 0.1. The tiny
variant uses 8 stem channels and blocks of 12, 16, 24, and 32 channels. The base
checkpoint's best validation AUC was at epoch 12; the tiny checkpoint's best
was at epoch 7 and its run stopped at epoch 10 after three non-improving checks.

| Candidate | Best INT8 validation AUC | EER | TAR at validation 1% FAR | INT8 size | Decision |
|---|---:|---:|---:|---:|---|
| Base | 0.8721 | 0.2113 | 0.3009 | 26,872 B | Selected |
| Tiny | 0.8280 | 0.2585 | 0.2009 | 15,632 B | Not selected |

The tiny variant is 41.8% smaller, but its validation AUC was about 0.044 lower.
The base model was selected using validation metrics only. Its float TFLite file
is 51,912 bytes; the full-INT8 file is about 48% smaller. On the four-thread
Intel host, 200-image mean/p95 inference times were 0.639/1.582 ms for float
TFLite and 1.770/2.222 ms for full INT8. These timings include array conversion,
quantization/dequantization, model invocation, and L2 normalization; they exclude
model load and image-file decoding. They do not predict Nicla timing; INT8 was
not faster than float TFLite on this host.

## Selected full-INT8 model

The deployable file is
[`FaceRecognition/models/identity_embedding_int8.tflite`](../../../FaceRecognition/models/identity_embedding_int8.tflite).
It is 26,872 bytes with SHA-256
`1f94fe7b3c48ee6c4b326272bf285b1381dd4ba6c68e0f4528b096fc1dcf0969`.

| Tensor | Shape | Type | Scale | Zero point |
|---|---|---|---:|---:|
| Input | `[1,96,96,1]` | INT8 | 0.0039215689 | −128 |
| Output | `[1,64]` | INT8 | 0.0121202506 | −36 |

The graph uses only `CONV_2D`, `DEPTHWISE_CONV_2D`, `FULLY_CONNECTED`, and
`MEAN`. Input and output are both signed INT8; quantization metadata was
generated by TensorFlow Lite and was not edited. The selected run, checkpoints,
split, validation pairs, training history, exact environment lock, float and
tiny comparison models, and logs are under `work/model-training/`.
Selected conversion metadata and final evaluation metrics are recorded in
`work/model-training/runs/survface_supcon_v2_holdout-clean/float_int8_evaluation.json`.

## Host evaluation

The cosine threshold `0.8776755` was calibrated from the selected INT8 model's
identity-disjoint validation negatives to permit at most 1% false accepts.
Validation achieved 0.943% FAR and 30.09% TAR at that threshold. The threshold
was then fixed for the final checks.

| Evaluation set | Model | Pairs (positive / negative) | ROC AUC | EER | TAR at fixed threshold | FAR at fixed threshold |
|---|---|---:|---:|---:|---:|---:|
| SurvFace identity-disjoint validation | Float TFLite | 2,120 (1,060 / 1,060) | 0.8719 | 0.2094 | 0.3085 | 0.0094 |
| SurvFace identity-disjoint validation | Full INT8 | 2,120 (1,060 / 1,060) | 0.8721 | 0.2113 | 0.3009 | 0.0094 |
| SurvFace final pair annotations | Float TFLite | 10,640 (5,320 / 5,320) | 0.8765 | 0.2075 | 0.3175 | 0.0145 |
| SurvFace final pair annotations | Full INT8 | 10,640 (5,320 / 5,320) | 0.8769 | 0.2062 | 0.3122 | 0.0139 |
| Olivetti external sanity set | Float TFLite | 400 (200 / 200) | 0.8739 | 0.2250 | 0.9600 | 0.5400 |
| Olivetti external sanity set | Full INT8 | 400 (200 / 200) | 0.8722 | 0.2300 | 0.9650 | 0.5250 |

Keras float and float TFLite were effectively identical on these scores. On the
final SurvFace pairs, INT8 differed from float TFLite by +0.0004 AUC, −0.0013
EER, −0.0053 TAR, and −0.0006 FAR. The host metric is stable across float and
INT8 exports at this evaluation scale; it does not resolve the camera-domain
or enrollment gaps.

The final SurvFace pair-bootstrap 95% intervals for INT8 were AUC
`[0.8703, 0.8834]`, EER `[0.1985, 0.2135]`, TAR `[0.3011, 0.3244]`, and FAR
`[0.0105, 0.0177]`. The Olivetti AUC looks similar, but the SurvFace-calibrated
threshold accepts 52.5% of its negative pairs. This demonstrates poor threshold
transfer across image domains; it is not evidence of useful wearable-camera
accuracy. No claim is made about a wearer, partner, or other live person.

## Nicla deployment and board checks

The model, validation calibration, and
[`embedding_config.py`](../../../FaceRecognition/firmware/embedding_config.py)
were copied to `/flash`. `ml.Model` loaded the selected model from
`/flash/identity_embedding_int8.tflite` on the actual Nicla:

- File size: 26,872 bytes; tensor arena: 51,040 bytes; model load: 5,503 µs.
- Input and output dtypes were signed byte (`b`), with shapes and quantization
  matching the host export. Output scale and zero point were `0.012120252` and
  `−36`.
- Six local SurvFace validation crops, not used in validation-pair generation,
  produced mean board-vs-host embedding cosine similarity `0.999660` (minimum
  `0.999439`; maximum absolute component error `0.01277`). These six samples
  establish numeric parity only.
- During the camera-loop run, the stock FOMO arena was 237,936 bytes and the
  minimum GC-free heap after loading it was 110,944 bytes (about 354,160 bytes
  before loading); after unloading it, the heap returned to about 353,680 bytes.
  The embedder was loaded separately: its model arena was 51,040 bytes and
  post-load GC-free heap was about 279,856 bytes. The six-crop board test, while
  retaining vectors and two RAM templates, reached 267,472 bytes minimum free.
  The two model arenas were not simultaneously loaded.
- The board ran eight temporary-template comparisons in RAM at the calibrated
  threshold; five decisions were correct. Three of four positive comparisons
  were rejected. This tiny, selected validation subset is not an accuracy
  estimate. No template file was written.
- A 100-frame actual-camera crop smoke completed. Mean capture time was 80.31
  ms, mean embedding inference 12.916 ms, and minimum observed MicroPython heap
  was 278,832 bytes. The tensor arena is separate from the reported Python
  heap; these are not a whole-board peak-RAM measurement.
- The smoke emitted about 10.25 forced-crop iterations per second between its
  first and last frame records, including the serial logging cadence. This is
  not full face-recognition throughput because it skips FOMO and template
  matching.
- The bounded stock-FOMO camera loop completed ten frames with ten `no_face`
  results. Mean camera capture was 57.47 ms, detector load 9.17 ms per frame,
  detector inference 65.53 ms, and pipeline duration before serial logging
  141.32 ms. It saved no images. As no
  face was detected, this live loop did not exercise embedding or identity
  comparison; the separate forced-crop and held-out-crop checks did. The loop
  took 10.002 seconds (about 1.00 frame/s), capped by its configured 1,000 ms
  sample period; 141.32 ms is the mean no-face processing time before logging,
  not the throughput of an embedded identity decision.
- The initial live camera run exposed a crop bug: `frame.copy(roi=...)` returned
  a zero-filled image on Nicla Vision/OpenMV 5.0.1 RGB565 frames. Firmware now
  uses `frame.to_grayscale(roi=roi, copy=True)`, which was verified on-board to
  preserve the selected pixels. A one-scene synthetic screen-target pipeline
  check detected 1/1 at the production FOMO threshold, embedded it in 12.75 ms,
  and matched the host embedding at cosine 0.999382 (maximum component error
  0.01256). This is an integration check, not an accuracy result.
- The board-input diagnostic also had two input/contract bugs that hid its
  behavior: grayscale PPM payloads were written with a `.pgm` suffix and read
  as the wrong format, and copying an already-RGB565 scene into the camera
  framebuffer darkened it. Scenes are now true RGB PPM files and the board
  passes RGB565 inputs directly to FOMO. The corrected six-scene held-out run
  returned no FOMO regions at threshold 0.4, so those scenes did not reach the
  embedder. The synthetic target test still passed at 0.4.
- OpenMV returned the embedder dtype as a tuple (`('b',)`), while JSON logs
  serialize it as an array. The production guard now checks the sequence's
  first value, and the stock detector's empty-output and nested-shape cases
  remain guarded.
- A target-facing 20-frame live test was run headlessly while the user held the
  synthetic target in view. Stock FOMO detected a region in 18/20 frames
  (median score 0.9961; range 0.4258–0.9961). The corrected grayscale crop and
  INT8 embedder ran in 13 frames; five detected regions were too close to an
  image edge for the configured 96×96 crop, so the pipeline returned
  `uncertain/crop_clipped_by_frame_edge`. Two frames returned `no_face`; none
  returned a pipeline error. All 13 crops passed the current brightness and
  contrast checks (mean 72–153, standard deviation 58–90), and embedding
  inference took 12.749–12.763 ms. Each resulting status was `no_enrollment`,
  as expected because no template is installed; this validates detection and
  embedding execution, not identity recognition.
- In the live target run, detector inference was 65.498–66.335 ms, capture was
  34.070–81.741 ms, and the minimum observed Python heap was 342,896 bytes.
  The run completed all 20 frames without a pipeline error or reset. The
  detector and embedder arenas were 237,936 and 51,040 bytes and were loaded
  sequentially. The temporary 20-frame limit was removed afterward; the
  production config was restored byte-for-byte (SHA-256
  `a26ff021e9281ccd5be91d2dc94e832bdb3a5186b8b2e687919a62378399dc1b`). No
  viewer was opened and no camera frames were saved. The retained JSONL log
  contains numeric diagnostics only.

Remote SHA-256 checksums after deployment matched the local files:

| `/flash` path | Bytes | SHA-256 |
|---|---:|---|
| `identity_embedding_int8.tflite` | 26,872 | `1f94fe7b3c48ee6c4b326272bf285b1381dd4ba6c68e0f4528b096fc1dcf0969` |
| `embedding_calibration.json` | 356 | `b37799c85be8e1cf2bfde9304256f1eb550655699078c9b57a8985296692b36e` |
| `embedding_config.py` | 1,008 | `a26ff021e9281ccd5be91d2dc94e832bdb3a5186b8b2e687919a62378399dc1b` |

After temporary board validation, the six PGM files and manifest were deleted
from `/flash/board_eval/`. The six FOMO-input PPM files and manifest were also
deleted from `/flash/fomo_eval/`; the local FOMO PPM scenes and manifest were
removed after the diagnostic. The local embedding-eval PGM files, raw board
vectors, and host reference vectors were deleted after comparison. The retained board
summary contains timing, scores, and pseudonymous record IDs only. There is no
`/flash/embedding_templates.json`; user enrollment is intentionally absent.
The retained evidence is `work/model-training/logs/embedding-smoke.jsonl`,
`work/model-training/logs/recognize-live-smoke.jsonl`,
`work/model-training/logs/live-synthetic-target-final.jsonl`,
[`work/model-training/reports/board-host-comparison.json`](board-host-comparison.json),
[`work/model-training/reports/fomo-board-pipeline-comparison.json`](fomo-board-pipeline-comparison.json),
[`work/model-training/reports/synthetic-target-fomo-board-validation.json`](synthetic-target-fomo-board-validation.json),
[`work/model-training/reports/synthetic-crop-board-validation.json`](synthetic-crop-board-validation.json),
[`postdeployment-flash-inventory.txt`](postdeployment-flash-inventory.txt),
[`postdeployment-flash-sha256.txt`](postdeployment-flash-sha256.txt), and
[`postdeployment-board-info.jsonl`](postdeployment-board-info.jsonl). The raw
board vector log and host reference vectors were removed after the parity
calculation.

After deployment, `/flash` reported 15,659,008 bytes available from 15,695,872
total. The original `/flash/main.py` and `README.txt` hashes still match their
predeployment backups. The board still reports the stock 4 MiB ROMFS and all
14 original entries. A read-only SHA-256 of `/rom/fomo_face_detection.tflite`
still matches the saved stock detector
(`fc3aec94b4b02dc6819db3e2d63bfd8055c9757fb350008538d4c1604aea4cb4`). No firmware,
bootloader, or ROMFS update was made during this training/deployment work.

## Expanded multi-seed research campaign (2026-10-22)

The validation-only research extension is complete: **45 standardized runs** (two channel-width variants, two identity-disjoint splits) plus a separately reported five-run pilot. The 20-run matched-seed extension completed 20/20 with no training failures. Three capped runs continued from 30 to 40 total epochs and one from 40 to 50; all four continuations completed successfully under the existing patience rule. No run remained eligible for further continuation.

The user-facing [expanded research readout](../../../../../2026-10-21/files-pasted-by-the-user-agent-3/outputs/Nicla_Expanded_Training_Research_Readout.md), [standardized 45-run CSV](../../../../../2026-10-21/files-pasted-by-the-user-agent-3/outputs/Nicla_Validation_Seed_Campaign.csv), [separate pilot CSV](../../../../../2026-10-21/files-pasted-by-the-user-agent-3/outputs/Nicla_Initial_Seed_Pilot.csv), and [one-page PDF handoff](../../../../../2026-10-21/files-pasted-by-the-user-agent-3/outputs/Nicla_Vision_Face_Recognition_Handoff.pdf) are in the task outputs folder.

Before re-export, a methods audit found that the older INT8 representative pool included validation identities and ignored the exact duplicate exclusions and per-identity image cap. Every scored candidate was re-exported using 512 images sampled from that run's exact selected training records. The summarizer independently reconstructed the full training selection, checked duplicate exclusions and count, verified every calibration record and image SHA-256, and checked the float and INT8 artifact hashes. The supplied final-pair annotation file and Olivetti were not loaded. Per-run manifests and evaluation JSON remain under `work/model-training/runs/`.

| Architecture / split | Runs | INT8 AUC mean +/- SD (range) | EER mean +/- SD (range) | TAR at validation 1% FAR threshold | Realized FAR | INT8 bytes | INT8 host mean / p95 ms per image |
|---|---:|---:|---:|---:|---:|---:|---:|
| base/split-20261021 | 10 | 0.8846 +/- 0.0101 (0.8657-0.9000) | 0.1991 +/- 0.0124 (0.1821-0.2217) | 0.3552 +/- 0.0453 (0.2840-0.4358) | 0.0094 +/- 0.0000 (0.0094-0.0094) | 26872 | 2.003 +/- 0.060 (1.941-2.097) / 2.301 +/- 0.224 (2.113-2.856) |
| tiny/split-20261021 | 10 | 0.8658 +/- 0.0106 (0.8492-0.8792) | 0.2162 +/- 0.0115 (0.1991-0.2396) | 0.3281 +/- 0.0270 (0.2981-0.3764) | 0.0094 +/- 0.0000 (0.0094-0.0094) | 15632 | 1.316 +/- 0.004 (1.313-1.323) / 1.456 +/- 0.018 (1.433-1.485) |
| base/split-20261022 | 15 | 0.8884 +/- 0.0077 (0.8747-0.8996) | 0.1986 +/- 0.0103 (0.1811-0.2179) | 0.3413 +/- 0.0197 (0.3094-0.3717) | 0.0094 +/- 0.0000 (0.0094-0.0094) | 26872 | 1.971 +/- 0.034 (1.944-2.067) / 2.207 +/- 0.096 (2.116-2.452) |
| tiny/split-20261022 | 10 | 0.8687 +/- 0.0069 (0.8568-0.8825) | 0.2174 +/- 0.0080 (0.2038-0.2311) | 0.2873 +/- 0.0420 (0.2208-0.3509) | 0.0094 +/- 0.0000 (0.0094-0.0094) | 15632 | 1.386 +/- 0.038 (1.323-1.427) / 1.728 +/- 0.155 (1.506-1.917) |

Base/split A and tiny/split A each contain ten runs; base/split B contains fifteen, and tiny/split B contains ten. Seed-level AUC standard deviations describe variability across these training runs on the fixed identity splits; they are not test-population confidence intervals. The threshold selects each candidate's operating point on its own validation negatives, yielding a discrete 10/1,060 FAR (0.9434%). Validation AUC and TAR remain subject to checkpoint and threshold selection on the same validation cohort.

| Matched comparison | Matched runs | INT8 AUC difference mean +/- SD (range) |
|---|---:|---:|
| Tiny minus base, split A | 10 | -0.0188 +/- 0.0137 (-0.0406 to +0.0013) |
| Tiny minus base, split B | 10 | -0.0176 +/- 0.0085 (-0.0281 to -0.0012) |
| Base split B minus split A | 10 | +0.0017 +/- 0.0130 (-0.0151 to +0.0188) |
| Tiny split B minus split A | 10 | +0.0029 +/- 0.0119 (-0.0127 to +0.0238) |

The tiny model is 15,632 B versus 26,872 B for base (41.8% smaller). Across matched seeds, tiny-minus-base AUC averaged -0.0188 on split A and -0.0176 on split B. Host timing was measured serially with three TFLite threads to avoid cross-evaluation contention; it is not Nicla timing. Full float/INT8 comparisons, uncertainty intervals, per-run seeds and hashes are in the companion research readout and CSV.

The separate five-run pilot was also re-exported and audited under the corrected calibration protocol. Its INT8 AUC was 0.8795 +/- 0.0109 (0.8621-0.8896), EER 0.2011 +/- 0.0113 (0.1915-0.2189), and TAR at the validation 1% FAR threshold 0.3157 +/- 0.0467 (0.2528-0.3783) (n=5). It used different training thread settings and is not pooled with the standardized campaign.

These are validation-selection results, not independent test estimates. The 45 new candidates remain host-only; the Nicla continues to run the original base seed-20261021 model, and no expanded candidate has been flashed or live-camera-tested. The board evidence and final-pair results above remain historical for that installed model and are not substituted with these new scores.


## Power and safety status

The meter's USB HID descriptor identified as FNIRSI FNB-58 (`2E3C:5558`); an
FNB-C2 box/manual was also seen, so the physical tester model and electrical
measurement path were not confirmed. The concurrent camera-trial log recorded
230 reports and 920 samples with no CRC errors/timeouts, but reported 0 V and
mean 2.162 mA while the board continued to operate. See
[`fnirsi-camera-trial-current-2026-10-21-1823-pdt.csv`](../../connected-hardware-validation/fnirsi-camera-trial-current-2026-10-21-1823-pdt.csv),
its [summary](../../connected-hardware-validation/fnirsi-camera-trial-current-2026-10-21-1823-pdt-summary.txt),
and the [concurrent Nicla camera log](../../connected-hardware-validation/nicla-camera-meter-current-2026-10-21-1823-pdt.jsonl).
The measurement is unusable as a board power result; average/peak power, energy,
and battery-life effects remain unmeasured. No power-saving claim is made.

No bystander or user image was saved. The scripts retain no camera frame
history. The custom pipeline uses stock FOMO heatmap centers and a fixed
96-pixel crop; the centers are not calibrated face boxes. Its low-quality and
uncertainty settings are provisional. A consenting, camera-matched enrollment
and evaluation set is needed before real-person known/unknown decisions can be
assessed.

## Reproduction and rollback

Run locally from the `ok-x20` directory with the pinned environment:

```sh
work/model-training/.venv/bin/python work/model-training/scripts/train_embedder.py \
  --seed 20261021 --objective supcon --epochs 12 --model-variant base \
  --output work/model-training/runs/survface_supcon_v2_holdout-clean
work/model-training/.venv/bin/python work/model-training/scripts/convert_eval.py \
  --run-dir work/model-training/runs/survface_supcon_v2_holdout-clean
```

The final conversion command is for reproducing the selected run's complete
evaluation. For model selection, use `--validation-only` for each candidate and
do not read final pair annotations until selection is complete. See
[`work/model-training/README.md`](../README.md) for both candidate commands and
board-evaluation steps.

After confirming the backup and the current port, the deployed model and
calibration can be copied and the configuration uploaded with:

```sh
PORT=/dev/cu.usbmodemREPLACE
FaceRecognition/.venv-nicla/bin/mpremote connect "$PORT" fs cp \
  FaceRecognition/models/identity_embedding_int8.tflite \
  :/flash/identity_embedding_int8.tflite
FaceRecognition/.venv-nicla/bin/mpremote connect "$PORT" fs cp \
  work/model-training/artifacts/embedding_calibration.json \
  :/flash/embedding_calibration.json
FaceRecognition/bin/nicla upload-embedding-config "$PORT"
FaceRecognition/bin/nicla embedding-smoke "$PORT"
FaceRecognition/bin/nicla recognize-embedding "$PORT"
```

The 100-frame smoke is bounded. `recognize-embedding` runs until stopped when
the production `MAX_FRAMES = 0` config is installed; the recorded ten-frame
camera run used a temporary config with `MAX_FRAMES = 10` and then restored the
production config with `upload-embedding-config`. For the pseudonymous
board-vs-host crop run, follow the temporary PGM copy, compare, and cleanup
commands in the training README.

To remove the deployed prototype while the board is stopped, use:

```sh
FaceRecognition/.venv-nicla/bin/mpremote connect "$PORT" fs rm \
  :/flash/identity_embedding_int8.tflite \
  :/flash/embedding_calibration.json \
  :/flash/embedding_config.py
```

The predeployment backups are under
`work/model-training/deployment-backup/`. The stock ROMFS restore image is
1,006,372 bytes with SHA-256
`f5dd7a04155fa819aab39d9b8d20528f8dd5669c2f9b41bc9dcc751277857e9b`. The
original `/flash/main.py` is 259 bytes with SHA-256
`4f565bcdc513ba2c76cd52b8951b6b8475210bf76e1746c5daa406f726453ff5`; the
original `/flash/README.txt` is 514 bytes with SHA-256
`431464c1e671e182b3c74b760f642adcbdb734e782ed808677e415467e143a64`.
Backup details are in
[`work/model-training/deployment-backup/baseline.json`](../deployment-backup/baseline.json).

No commit or push was made. The `ok-x20` workspace root is not itself a Git repository; the applicable source repository is `FaceRecognition/`. Its final `git status --short` listed the project contents as untracked (`??`); no tracked or staged changes were reported.

## Standalone Nicla startup implementation (2026-10-22)

The previous deployment only ran the recognizer when a Mac command streamed it
to the board. To provide a genuinely board-local boot path, the firmware now
has `FaceRecognition/firmware/main.py`, which calls the recognizer as the
OpenMV application entry point. OpenMV 5.0.1 runs `/flash/main.py` during cold
boot; the local install command packages the recognizer, embedding config,
26,872-byte INT8 model, and validation calibration into `/flash`, leaving the
stock `/rom/fomo_face_detection.tflite` and 4 MiB ROMFS unchanged. The pinned
OpenMV [boot documentation](https://docs.openmv.io/v5.0.1/reference/reset_boot.html)
describes this `main.py` startup behavior.

The installer `FaceRecognition/bin/nicla deploy-standalone PORT` preflights
OpenMV CSI/ML and the stock detector, takes a timestamped local backup of
existing startup and target files, preserves the original startup as
`/flash/main.py.stock`, copies deployment assets, verifies remote SHA-256
hashes, installs `main.py` last, and issues a hard reset. The rollback command
`FaceRecognition/bin/nicla restore-startup PORT` restores the on-device copy
and reboots. Model/config files remain on `/flash` after rollback; no custom
ROMFS, bootloader, or application firmware is written.

An explicit local enrollment path is also present:
`FaceRecognition/bin/nicla enroll PORT PSEUDONYM`. It captures multiple
quality-approved embeddings from one consenting person, retains only a
normalized mean vector in `/flash/embedding_templates.json`, and never stores
raw frames. Matching remains a research prototype, not a security decision.
When no serial terminal is attached, the application avoids continuous USB
logging and uses the built-in RGB LED for a coarse known/unknown/attention
signal.

**Hardware state at this update:** the last known board ran OpenMV 5.0.1 and
had successfully executed the detector-plus-embedder pipeline in an earlier
headless session. At the time this standalone setup was added, the local serial
inventory contained only Bluetooth pseudo-ports and no Nicla USB port. Thus the
new entry point and installer are implemented but **not yet flashed or
cold-boot tested**; the prior verified runtime still has the original
`/flash/main.py`. Reconnect the Nicla and run
`FaceRecognition/bin/nicla deploy-standalone <port>` to complete that hardware step. The deployment remains
partially functional until a cold boot and laptop-disconnected operation are
verified. The earlier power, consented enrollment, camera-domain accuracy,
dataset provenance, and expanded-candidate board-validation limitations are
unchanged.

## Compression and edge-guard supplement - 2026-10-22

A separate expanded study compared all standard export formats plus 50%
magnitude pruning, 2:4 pruning, 16-cluster weights, and QAT. Each candidate
was checked on the Nicla; the matrix passed 8/14 and documented six allocation
failures. Host robustness covered 80 synthetic numeric inputs and 45
perturbation-condition evaluations. Fourteen host edge-case tests and eight
on-device MicroPython guard categories passed. The recognizer guard update was
backed up, deployed, SHA-256 read back, and followed by a standalone restart.

The full compression ranking, AUC/EER and bootstrap intervals, operator and
quantization details, board memory/latency, privacy-safe camera observations,
cleanup proof, limits, and reproduction commands are documented in
[compression-edge-case-report-20261022.md](compression-edge-case-report-20261022.md).
The matrix and raw logs remain under work/model-training/compression/strategy_sweep_20261022/.
The current scene still does not establish camera-domain identity accuracy,
real-person enrollment, or power behavior; the verdict remains partially
functional.
