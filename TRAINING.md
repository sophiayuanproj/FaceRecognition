# Train and evaluate from this repository

The current Nicla application, the research trainers, and the result-analysis code are included in this clone. Run commands from the repository root, which contains `FaceRecognition/`, `workflow/`, `detector/`, and `work/`.

The device uses the adaptive face-detection fallback described in [START_HERE.md](START_HERE.md). Its compact identity model executes with known-person acceptance disabled. Retraining produces a new experimental candidate that needs its own calibration and device checks before deployment. The historical host crop diagnostic and the current device use different identity pipelines.

## Set up the training environment

The recorded training stack uses TensorFlow 2.10 with Python 3.9 or 3.10. It has its own environment under `work/model-training/.venv`. The camera application has a separate environment.

On an Intel or Apple Silicon Mac with native Python 3.10 installed, run

```sh
python3.10 workflow/setup_training.py
python3 workflow/verify.py
```

The setup selects `tensorflow==2.10.1` on Intel and `tensorflow-macos==2.10.0` on Apple Silicon. It installs the shared versions in `workflow/requirements-training.txt` and checks their imports. Set `NICLA_TRAIN_PYTHON` or pass `--python /path/to/python3.10` if needed. On 64-bit Windows, `py -3.10 workflow/setup_training.py` creates `work/model-training/.venv/Scripts/python.exe`. The commands below use the macOS/Linux interpreter path; substitute that Windows interpreter when training there.

## Restore the crop data

Follow [DATA.md](DATA.md) to obtain the exact original archive and verify its hash. The five-person benchmark needs the SurvFace-derived archive. Once it is available locally, run

```sh
python3 workflow/prepare_data.py /path/to/survface_96.zip
python3 workflow/verify.py --data
```

This restores the role-selected images used by the recorded five-person protocol. Images stay in the ignored local dataset directory. Calibration, stopping, and development roles retain their existing separation.

## Train the two compact baselines

The wrapper uses ten registered seeds, 20261101 through 20261110, for each width. It uses the recorded development-selected configuration, training-only representative images for full INT8 export, and separate threshold calibration. Each output directory must be new.

```sh
python3 workflow/train.py --width 0.1 --out work/model-training/new-small --plan
work/model-training/.venv/bin/python workflow/train.py \
  --width 0.1 --out work/model-training/new-small
work/model-training/.venv/bin/python workflow/train.py \
  --width 0.25 --out work/model-training/new-large
```

Run the training commands sequentially on a laptop. `--plan` prints the chosen seeds and configuration without importing TensorFlow or opening images. Add `--quick` for a one-epoch diagnostic across the same ten seeds. A quick diagnostic does not reproduce the recorded study.

Each campaign writes `CAMPAIGN.json`, and each seed writes its fitted weights, float and INT8 TFLite files, configuration, calibrations, and `ROOT_RESULT.json`. The current wrapper names the width families `sv_mnv1_a010_transfer_6` and `sv_mnv1_a025_transfer_6`.

Re-evaluate a newly trained seed against the pinned development roles with

```sh
work/model-training/.venv/bin/python workflow/evaluate.py \
  work/model-training/new-small/survface/sv_mnv1_a010_transfer_6/seed-20261101/ROOT_RESULT.json \
  --output work/model-training/new-small/seed-20261101-replay.json
```

The evaluator checks the manifest and model hashes, invokes both exported TFLites, and uses their recorded calibration thresholds. Report correct known acceptance, wrong-known acceptance, unknown false acceptance, and coverage together. Classification top-1 alone does not establish useful open-set recognition.

## Detector and compression research

Sofia's detector files remain under `detector/scripts/`. The original FOMO implementation, data preparation, export, and evaluation dependencies are alongside them. Start with [detector/README.md](detector/README.md) and the `--help` output of `train_fomo.py`, `train_fomo_v2.py`, and `evaluate_development.py`. WIDER FACE and PASS supply full scenes. Face-crop data alone cannot train a detector. Keep a separate physical camera session for validation after reviewing any auto-generated boxes.

The full capacity, export, pruning, quantization-aware training, physical channel reduction, clustering, and distillation methods are in `work/project-finish-v4/runs/E/scripts/`. Reports and method declarations are under `work/project-finish-v4/reports/E/`, with a concise result in `work/project-finish-v4/handoffs/E/RESULT.md`. These are the preserved study recipes. They expect the original seed directories and source checkpoints, which the optional artifact download restores.

```sh
python3 workflow/restore_artifacts.py
work/model-training/.venv/bin/python work/project-finish-v4/runs/E/scripts/e_analyze.py
```

The first command downloads the pinned GitHub release, verifies every file, and restores the recorded fitted artifacts. It is unnecessary for a fresh compact-baseline training run. Add `--archive /path/to/P04_Training_Artifacts_2026-11-22.zip` to reuse a download, or `--check` to verify it without restoring. Existing differing local artifacts cause a refusal, preserving your new experiments.

Original result receipts contain historical machine paths. Preserve their hashes. The current compact train/evaluate workflow resolves its files from this clone. Broad historical resumption may require rebasing a recorded path when the relevant script reads it directly. Record a new output directory and any changes to the original recipe. The early custom embedding study remains under [work/model-training](work/model-training/README.md), with its seed plans and corrected train-only INT8 representative-pool implementation.

## Deploying a changed model

Use [DEPLOYMENT.md](DEPLOYMENT.md) and the current `FaceRecognition/tools/deploy_nicla_runtime.py`. A classifier and an embedding model have different output and calibration contracts. A newly trained six-class classifier cannot replace the 64-value embedding slot. Keep the candidate in a separate release and retain its model/configuration/calibration hashes, actual input tensor, memory receipt, and matched device frames. Measure quality and latency again before interpreting a power comparison of that candidate.

Ten training seeds measure optimization variability. The recorded power experiment used ten separate physical starts. Keep those experimental units explicit in the report.
