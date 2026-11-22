# Local face embedding training and Nicla deployment

This is a private, local-only research/course prototype for the SurvFace 96×96
grayscale crops already prepared under `work/dataset-prep/`. No images,
embeddings, checkpoints, or enrollment data are sent to cloud services. Do not
redistribute the dataset or any trained model; the exact transformed archive
provenance and reuse permissions remain unresolved.

## What this pipeline does

- Trains an identity-supervised compact CNN with a 64-value feature output.
- Splits SurvFace training identities into train and validation groups by
  identity, stratified by image-count decile. Validation pairs are generated
  only from held-out identities. Olivetti is never used for training.
- Removes exact image hashes duplicated across identity labels, same-identity
  duplicate copies, and training images that exactly overlap the unlabeled
  SurvFace verification-image set. Perceptual-hash matches are recorded as
  candidates and are not removed automatically.
- Uses balanced identity sampling (eight identities × four images per batch),
  training-only mild augmentation, a fixed seed, checkpoints, validation-pair
  early stopping, and a full-integer INT8 TFLite export.
- Calibrates the verification threshold from the identity-disjoint validation
  pairs at a 1% false-accept target. The provided SurvFace verification-pair
  annotations are only used in the final evaluation stage; their 10,051 images
  remain unlabeled at image level.
- Evaluates a Keras float baseline, a float TFLite export, and the actual
  signed INT8 export. It records model I/O quantization, operators, file sizes,
  host inference timing, ROC-AUC, EER, TAR/FAR, pair counts, and bootstrap
  intervals. Pair reuse in the supplied test annotations means pair-level
  bootstrap intervals can be optimistic.
- Deploys the selected custom model to writable `/flash` and keeps the stock
  FOMO detector in the unchanged `/rom` partition. Board loading, inference,
  host parity, and the bounded camera path are recorded in the final report.
- Crops detector ROIs with the OpenMV ROI-aware grayscale conversion. This was
  verified on the Nicla after `Image.copy(roi=...)` returned zero-filled crops
  on RGB565 camera frames. Face-scene diagnostics use RGB PPM files so OpenMV
  loads the RGB565 input correctly.

The learning input is a cropped face, not a full camera scene. It cannot train
or evaluate a detector. The Nicla continues to use the stock OpenMV FOMO
face-presence model. Its heatmap centroids are not calibrated face boxes, so a
fixed crop around a detected center remains an approximation. No consenting
user/partner enrollment images were found in the workspace; the board therefore
has no real-person template and cannot make a valid user identity claim.

## Environment

Training uses a separate macOS Intel Python 3.9 environment. The Nicla host
environment at `FaceRecognition/.venv-nicla` is left unchanged.

```sh
python3.9 -m venv work/model-training/.venv
work/model-training/.venv/bin/python -m pip install --upgrade pip
work/model-training/.venv/bin/python -m pip install -r work/model-training/requirements-lock.txt
```

The verified host was an Intel MacBook Pro with Darwin 25.6.0, 16 logical CPU
threads, and 64 GiB RAM. Training used TensorFlow 2.10.1 CPU only. The separate
environment and all reports stay under `work/model-training/`.

## Recreate training and evaluation

Run from the `ok-x20` project root. The full clean base run and a smaller
variant use the same identity-disjoint split and early-stopping rule:

```sh
work/model-training/.venv/bin/python work/model-training/scripts/train_embedder.py \
  --seed 20261021 --objective supcon --epochs 12 \
  --model-variant base \
  --output work/model-training/runs/survface_supcon_v2_holdout-clean

work/model-training/.venv/bin/python work/model-training/scripts/convert_eval.py \
  --run-dir work/model-training/runs/survface_supcon_v2_holdout-clean \
  --validation-only \
  --float-output work/model-training/artifacts/base_float.tflite \
  --int8-output work/model-training/artifacts/base_int8.tflite

work/model-training/.venv/bin/python work/model-training/scripts/train_embedder.py \
  --seed 20261021 --objective supcon --epochs 12 --model-variant tiny \
  --output work/model-training/runs/survface_supcon_tiny_v1

work/model-training/.venv/bin/python work/model-training/scripts/convert_eval.py \
  --run-dir work/model-training/runs/survface_supcon_tiny_v1 \
  --validation-only \
  --float-output work/model-training/artifacts/tiny_float.tflite \
  --int8-output work/model-training/artifacts/tiny_int8.tflite

```

Each training command writes the split, held-out validation pairs, checkpoints,
training configuration, run summary, and validation history. Use
`convert_eval.py --validation-only` to compare candidates; it does not load the
final pair annotations. The final SurvFace pair annotations were already used
once for the historical seed-20261021 baseline; do not rerun that final check,
or use those labels to select or tune the expanded campaign. Its historical
metrics remain in the report and original evaluation JSON. `--smoke` runs one
short non-result pipeline check under a `smoke/` subdirectory. `--resume` loads
the last saved Keras checkpoint and prior validation history for the same split
and output path; `--epochs` is the total target epoch count when resuming, and
the early-stopping patience continues from the last recorded best epoch.

The 96×96 face crops are decoded from the existing dataset manifest. Training
uses at most 24 uniformly selected images per training identity to limit memory
and balance the long tail; identity classes are sampled uniformly, then four
images are selected within each class per batch. All surviving identities are
included in the full run.

## Expanded seed and architecture campaign

The campaign contains 45 standardized validation-only runs plus a separate
five-seed pilot. The original 15-run matrix compares five seeds on split A,
repeats base on split B with distinct seeds, and compares base/tiny on split A
with matched seeds. A five-run matched base split-B cohort and five-run tiny
split-B replication complete five-seed matched comparisons on both splits. A
sequential exploratory extension, specified before launching those runs, adds
five new matched seeds (20261032–20261036) to each of the four architecture/split
cells. Intended final group sizes are base/split
A=10, tiny/split A=10, base/split B=15 (ten matched seeds plus five
distinct-seed replications), and tiny/split B=10. The five-seed pilot used
different TensorFlow thread settings and remains separate.

Before consolidated scoring, a methods audit found that the older INT8 export
helper included validation identities in its representative calibration pool
and did not apply exact-duplicate exclusions or the per-identity training cap.
The corrected helper reconstructs the exact records selected for each training
run and draws representative images only from those records. Each export keeps
a hash-checked record/image manifest plus a digest of its full training-record
selection. All 45 standard runs and all five pilot runs are re-exported and
rescored validation-only under this corrected protocol. This correction was made after initial
campaign training results were known and is disclosed in the research readout;
the historical seed-20261021 final-pair evaluation and installed board model
remain archival and are not directly comparable to corrected INT8 scores.

All standardized variants use the same supervised-contrastive objective and
64-D embedding; only channel widths differ. Training is capped at 30 epochs
with validation-AUC checkpoint selection and patience 5. The initial matrix
used up to four simultaneous training processes, each limited to four
intra-op and one inter-op CPU thread. The 20-run seed extension was launched
with up to three jobs while the remaining original cap continuation ran
concurrently. This uses the Intel host's 16 logical CPUs on a 64-GiB machine.
All input data stays local.
Any run that reaches epoch 30 before the five-epoch patience condition will be
resumed to a total target of 40 epochs; this is reported as a continuation of
that seed, not as another independent run. If a continued run reaches epoch 40
before patience stops it, the same run is continued to a total target of 50 so
the patience rule can stop it. This second continuation was added on
2026-10-22 and is reported separately.

```sh
work/model-training/.venv/bin/python work/model-training/scripts/run_training_matrix.py \
  --plan work/model-training/experiments/research_seed_architecture_campaign_20261021.json

# Supplemental matched-seed split-sensitivity runs.
work/model-training/.venv/bin/python work/model-training/scripts/run_training_matrix.py \
  --plan work/model-training/experiments/matched_split_sensitivity_20261021.json

# Supplemental tiny-model replication on split B.
work/model-training/.venv/bin/python work/model-training/scripts/run_training_matrix.py \
  --plan work/model-training/experiments/tiny_splitB_replication_20261021.json \
  --max-parallel 2

# Five additional matched train/augmentation seeds on both splits/variants.
work/model-training/.venv/bin/python work/model-training/scripts/run_training_matrix.py \
  --plan work/model-training/experiments/validation_seed_extension_20261022.json \
  --max-parallel 3

# Apply the predeclared cap continuations; safe to rerun after checking that
# previous continuations have exited. Only capped runs still inside patience
# are resumed, in their original run directories.
work/model-training/.venv/bin/python work/model-training/scripts/resume_capped_runs.py \
  --campaign-dir work/model-training/runs/research_seed_architecture_campaign_20261021 \
  --campaign-dir work/model-training/runs/matched_split_sensitivity_20261021 \
  --campaign-dir work/model-training/runs/tiny_splitB_replication_20261021 \
  --campaign-dir work/model-training/runs/validation_seed_extension_20261022 \
  --cap-epochs 30 --target-epochs 40 --max-parallel 3

# If any run from that pass reached epoch 40 before patience stopping, rerun
# the same continuation script with --cap-epochs 40 --target-epochs 50.

# Score every final campaign checkpoint only after all cap continuations.
work/model-training/.venv/bin/python work/model-training/scripts/run_eval_matrix.py \
  --plan work/model-training/experiments/consolidated_validation_evaluation_20261022.json \
  --max-parallel 1

work/model-training/.venv/bin/python work/model-training/scripts/summarize_research_campaign.py \
  --campaign-dir work/model-training/runs/research_seed_architecture_campaign_20261021 \
  --also-campaign-dir work/model-training/runs/matched_split_sensitivity_20261021 \
  --also-campaign-dir work/model-training/runs/tiny_splitB_replication_20261021 \
  --also-campaign-dir work/model-training/runs/validation_seed_extension_20261022 \
  --output-prefix work/model-training/reports/research_seed_architecture_campaign_20261021

# Re-score and summarize the separate five-seed pilot under the corrected
# exact-selected-training-record INT8 calibration protocol.
work/model-training/.venv/bin/python work/model-training/scripts/run_eval_matrix.py \
  --plan work/model-training/experiments/pilot_validation_evaluation_20261022.json \
  --max-parallel 1
work/model-training/.venv/bin/python work/model-training/scripts/summarize_research_campaign.py \
  --campaign-dir work/model-training/runs/seed_sweep_fixed_split_20261021 \
  --output-prefix work/model-training/reports/pilot_seed_sweep_20261021

work/model-training/.venv/bin/python work/model-training/scripts/create_research_readout.py \
  --summary-json work/model-training/reports/research_seed_architecture_campaign_20261021.json \
  --pilot-json work/model-training/reports/pilot_seed_sweep_20261021.json \
  --output /Users/omristeinberg-tatman/Documents/Codex/2026-10-21/files-pasted-by-the-user-agent-3/outputs/Nicla_Expanded_Training_Research_Readout.md
```

Every candidate is exported to its own run directory as float TFLite and full
signed-INT8 TFLite, with representative calibration images from the exact
duplicate-filtered, per-identity-capped training set and 1,000 pair-bootstrap and 1,000 identity-cluster bootstrap
replicates. Evaluation processes run serially so host inference timing is not
affected by competing evaluation jobs. Evaluations use only that run's identity-disjoint validation
pairs; the final SurvFace pair annotations and Olivetti are not loaded. The selected epoch and threshold use validation AUC/negatives, so
validation metrics are selection estimates, not unbiased test estimates. Pair
bootstrap intervals can be optimistic because identities and images recur
across pairs. The older five-seed pilot is stored separately under
`runs/seed_sweep_fixed_split_20261021/`; it used 12 intra-op / 2 inter-op
threads and is not pooled with the standardized campaign. The aggregate
summarizer combines runs by architecture and split across campaign roots and
rejects duplicate seeds within each group. Per-run records retain exact cohort
and checkpoint provenance.

## Board deployment and exercise

Check the current port first:

```sh
FaceRecognition/bin/nicla ports
```

After verifying the report’s predeployment backup, copy the INT8 model and
configuration to writable `/flash`. These commands do not rebuild or replace
ROMFS:

```sh
PORT=/dev/cu.usbmodemACTUALPORT
FaceRecognition/.venv-nicla/bin/mpremote connect "$PORT" fs cp \
  FaceRecognition/models/identity_embedding_int8.tflite \
  :/flash/identity_embedding_int8.tflite
FaceRecognition/.venv-nicla/bin/mpremote connect "$PORT" fs cp \
  FaceRecognition/firmware/embedding_config.py :/flash/embedding_config.py
FaceRecognition/.venv-nicla/bin/mpremote connect "$PORT" fs cp \
  work/model-training/artifacts/embedding_calibration.json \
  :/flash/embedding_calibration.json
```

Run the 100-frame forced-crop model smoke (diagnostic only; it does not claim a
face or identity result):

```sh
FaceRecognition/bin/nicla embedding-smoke "$PORT"
```

Run the stock detector → crop → embedding → template comparison camera loop:

```sh
FaceRecognition/bin/nicla recognize-embedding "$PORT"
```

Both commands stream JSON over `mpremote` without opening the OpenMV viewer,
so the camera can stay pointed at a displayed test target. The recognition
loop uses the production sampling interval and runs until interrupted unless
a temporary bounded config is uploaded.

Enrollment is explicit and local. If a future consenting enrollment cohort is
collected, the board accepts `/flash/embedding_templates.json` with version 1
and one normalized 64-value vector per pseudonymous `identity_id`. No template
is installed for this prototype. Without one, the firmware reports
`no_enrollment`; it does not label a person known or unknown.

## Board-vs-host sample comparison

`prepare_board_eval.py` chooses a small number of SurvFace validation
images that were not used in the generated validation pairs, writes temporary
96×96 grayscale PGM files under `work/model-training/board-eval-local/`, and
computes host INT8 reference embeddings. The final report records the exact
sample count and parity measurements. These face crops are temporary and local.
Run the following from the repository root after the trained model is selected:

```sh
work/model-training/.venv/bin/python work/model-training/scripts/prepare_board_eval.py \
  --run-dir work/model-training/runs/survface_supcon_v2_holdout-clean
FaceRecognition/.venv-nicla/bin/mpremote connect "$PORT" fs mkdir :/flash/board_eval
for sample in work/model-training/board-eval-local/*.pgm; do
  name=$(basename "$sample")
  FaceRecognition/.venv-nicla/bin/mpremote connect "$PORT" fs cp \
    "$sample" ":/flash/board_eval/$name"
done
FaceRecognition/.venv-nicla/bin/mpremote connect "$PORT" fs cp \
  work/model-training/board-eval-local/board-eval-manifest.json \
  :/flash/board_eval/board-eval-manifest.json
FaceRecognition/.venv-nicla/bin/mpremote connect "$PORT" run \
  FaceRecognition/firmware/board_eval_embeddings.py \
  > work/model-training/board-eval-local/raw-board-eval.jsonl
work/model-training/.venv/bin/python work/model-training/scripts/compare_board_eval.py \
  --board-log work/model-training/board-eval-local/raw-board-eval.jsonl

# Remove the temporary crops and manifest from the board after comparison.
for sample in work/model-training/board-eval-local/*.pgm; do
  name=$(basename "$sample")
  FaceRecognition/.venv-nicla/bin/mpremote connect "$PORT" fs rm \
    ":/flash/board_eval/$name"
done
FaceRecognition/.venv-nicla/bin/mpremote connect "$PORT" fs rm \
  :/flash/board_eval/board-eval-manifest.json
FaceRecognition/.venv-nicla/bin/mpremote connect "$PORT" fs rmdir :/flash/board_eval
```

The board test creates templates only in RAM and logs pseudonymous IDs, scores,
and timings. It stores no camera frames. Keep the raw vector log and host
reference embeddings only long enough to run the comparison, then delete them
along with the local PGM crops and manifest.

### FOMO-to-embedding board-input diagnostic

When no consenting live face is available, this optional diagnostic puts six
held-out, pair-unused SurvFace face crops on neutral 160×160 grayscale scenes
and runs stock FOMO, the production 96-pixel crop rule, and the INT8 embedder
sequentially on the Nicla. The FOMO threshold defaults to the production value
of 0.4. This is an offline board-input check, not a camera-domain evaluation.
If FOMO returns no region, the embedder is intentionally not invoked. Any raw
vectors appear only in the temporary serial log and must be removed after the
host comparison.

```sh
work/model-training/.venv/bin/python work/model-training/scripts/prepare_fomo_board_eval.py \
  --run-dir work/model-training/runs/survface_supcon_v2_holdout-clean \
  --scene-side 160
FaceRecognition/.venv-nicla/bin/mpremote connect "$PORT" fs mkdir :/flash/fomo_eval
for sample in work/model-training/fomo-board-eval-local/*.ppm; do
  name=$(basename "$sample")
  FaceRecognition/.venv-nicla/bin/mpremote connect "$PORT" fs cp \
    "$sample" ":/flash/fomo_eval/$name"
done
FaceRecognition/.venv-nicla/bin/mpremote connect "$PORT" fs cp \
  work/model-training/fomo-board-eval-local/fomo-board-eval-manifest.json \
  :/flash/fomo_eval/fomo-board-eval-manifest.json
FaceRecognition/.venv-nicla/bin/mpremote connect "$PORT" run \
  FaceRecognition/firmware/board_eval_fomo_pipeline.py \
  > work/model-training/logs/fomo-board-pipeline-raw.jsonl
work/model-training/.venv/bin/python \
  work/model-training/scripts/compare_fomo_board_eval.py \
  --board-log work/model-training/logs/fomo-board-pipeline-raw.jsonl
```

Afterward remove the temporary `/flash/fomo_eval/` files and directory, and
delete local PPM scenes, the manifest, and raw vector log. The retained report
should contain only aggregate detection, parity, timing, and pseudonymous
pair-decision results.

## Rollback

The custom model, config, and threshold live only in `/flash`. ROMFS, firmware,
the bootloader, and the original startup script are not replaced. Remove the
three custom files after stopping the running script:

```sh
FaceRecognition/.venv-nicla/bin/mpremote connect "$PORT" fs rm \
  :/flash/identity_embedding_int8.tflite \
  :/flash/embedding_config.py \
  :/flash/embedding_calibration.json
```

If any original `/flash` file is changed accidentally, restore the exact
predeployment copy from `work/model-training/deployment-backup/flash/` and
compare the SHA-256 values in `deployment-backup/baseline.json`. The official
stock ROMFS recovery image and the stock FOMO model plus labels are saved under
`deployment-backup/`; the existing project copy of the official v5.0.1 release
is also preserved. This task does not require flashing ROMFS.

## Important limits

- The supplied transformed archive has unresolved provenance and reuse rights.
  The SurvFace source page describes academic-research-only use and upstream
  image-owner copyright; the transformed archive’s exact source is unconfirmed.
- Dataset images are cropped faces and have no full-scene boxes or negatives.
  No detector was trained.
- SurvFace provides no capture-session metadata; identity-disjoint splitting
  does not guarantee session-disjoint or duplicate-source-disjoint capture.
- Olivetti remains an external sanity set, with the same missing session
  metadata caveat.
- The unlabeled SurvFace verification images are only represented by the
  supplied pair labels during final evaluation; no image identity is inferred.
- A headless 20-frame live check of the displayed synthetic target detected
  regions in 18 frames and ran the embedder in 13. Five detections were too
  close to a frame edge for a safe fixed crop, and two frames had no detection.
  The stock FOMO detector also embedded one fictional synthetic target in an
  offline board-input check, but returned no regions for six identity-disjoint
  held-out faces arranged on neutral scenes. These checks do not establish
  robust camera-domain detection or identity accuracy.
- No real-person enrollment images exist in the project, so no wearer or partner
  recognition claim is supported.
- Power remains unmeasured: recent meter capture read 0 V during board activity,
  and physical tester identity/port labels and the Nicla electrical path remain
  unresolved. No energy or power-saving claim is made.

## Compression and edge-case validation - 2026-10-22

The follow-on compression study tested float32, float16 weights, dynamic-range
INT8 weights, full INT8, a narrower-channel network, 50% magnitude pruning,
2:4 structured pruning, 16-cluster weight clustering, and quantization-aware
training (QAT). TFMOT 0.7.5 is pinned in the isolated training environment.
All candidates were scored only on the existing identity-disjoint
2,120-pair validation cohort; final pair annotations and Olivetti were not
loaded. The full-INT8 base remains the balanced selection; the tiny full-INT8
variant is the smaller/faster option with lower validation verification quality.

Reproduce from the ok-x20 root:
- Standard export sweep: work/model-training/.venv/bin/python work/model-training/scripts/compression_sweep.py --output-dir work/model-training/compression/strategy_sweep_20261022
- Pruning, 2:4, clustering: work/model-training/.venv/bin/python work/model-training/scripts/tfmot_strategy_sweep.py --epochs 5 --output-dir work/model-training/compression/advanced_strategy_sweep_corrected_20261022
- QAT: work/model-training/.venv/bin/python work/model-training/scripts/tfmot_strategy_sweep.py --epochs 5 --only-strategy qat_full_int8 --output-dir work/model-training/compression/qat_strategy_sweep_20261022
- Calibration pool audit: work/model-training/.venv/bin/python work/model-training/scripts/audit_compression_calibration.py
- Host edge cases: work/model-training/.venv/bin/python work/model-training/scripts/firmware_edge_case_tests.py

On the connected Nicla, FaceRecognition/bin/nicla compression-matrix PORT
tests all 14 candidate formats and cleans its temporary QA directory;
FaceRecognition/bin/nicla edge-probe PORT exercises the MicroPython guards,
template round-trip, cleanup, and standalone restart. These commands alter only
temporary QA files under /flash, not the stock ROMFS. The dated detailed
report, raw JSON/JSONL, manifest, and test log are in reports/ and
compression/; no trained weights or faces belong in the user-facing outputs.
See reports/compression-edge-case-report-20261022.md.
