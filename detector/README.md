# Detector training and evaluation

Run these commands from the cloned repository root. [TRAINING.md](../TRAINING.md) describes the pinned TensorFlow 2.10 environment and [DATA.md](../DATA.md) records the dataset provenance. The current R19 application uses native Haar face boxes. This directory preserves the learned FOMO baseline and Sofia's follow-up trainer as research paths.

## Files and dependencies

| Files | Purpose |
| --- | --- |
| `scripts/common.py` | WIDER parsing, camera geometry, FOMO decoding and quantization |
| `scripts/prepare_wider.py` | Event-grouped WIDER role manifests |
| `scripts/prepare_pass_negatives.py` | Uploader-grouped PASS negatives with attribution |
| `scripts/prepare_training_cache.py` | Training/development arrays and aligned scene metadata |
| `scripts/train_fomo.py` | Original two-arm, ten-seed trainer and float/INT8 exports |
| `scripts/evaluate_development.py` | Original detector/crop comparison against stock RGB FOMO |
| `scripts/train_fomo_v2.py`, `scripts/fomo_v2_data.py` | Sofia's follow-up model, augmentations and weighted batches |
| `scripts/label_own_frames.py` | Sofia's local camera-frame annotation helper |
| `scripts/audit_*.py`, `scripts/make_*.py` | Export, overlap, parity and report tools |
| `scripts/prepare_frozen_final_cache.py`, `scripts/evaluate_frozen_final.py` | Separate frozen-evaluation path |
| `models/stock_reference/` | The stock RGB FOMO reference and its tensor/hash audit |
| `board/` | OpenMV board parity and crop checks |

The three Sofia scripts are preserved. They import the original trainer, decoder and evaluator directly from this same `scripts/` directory. NumPy, Pillow, TensorFlow and OpenCV are in the training requirements. PASS downloading also uses `certifi`, installed as a TensorFlow dependency.

Set up training as described in [TRAINING.md](../TRAINING.md), then confirm the actual entry points

```sh
work/model-training/.venv/bin/python detector/scripts/train_fomo.py --help
work/model-training/.venv/bin/python detector/scripts/train_fomo_v2.py --help
work/model-training/.venv/bin/python detector/scripts/evaluate_development.py --help
```

On Windows, use `work/model-training/.venv/Scripts/python.exe` for those commands.

## Prepare the scene data

Obtain WIDER FACE and PASS metadata from the sources in [DATA.md](../DATA.md). The WIDER directory passed below must contain `WIDER_train/images/`, `WIDER_val/images/`, and `wider_face_split/`. PASS preparation expects the original v3 CSV metadata. Source images and regenerated caches stay in the ignored `detector/local/` directory.

```sh
work/model-training/.venv/bin/python detector/scripts/prepare_wider.py \
  --wider-root /path/to/wider \
  --outdir detector/local/manifests/wider

work/model-training/.venv/bin/python detector/scripts/prepare_pass_negatives.py \
  --metadata /path/to/pass_v3.csv \
  --image-dir detector/local/pass-images \
  --outdir detector/local/manifests/pass

work/model-training/.venv/bin/python detector/scripts/prepare_training_cache.py \
  --wider-root /path/to/wider \
  --wider-manifest-dir detector/local/manifests/wider \
  --pass-manifest detector/local/manifests/pass/pass_negative_manifest.jsonl \
  --outdir detector/local/cache
```

The cache builder creates `train_cache.npz`, `development_cache.npz`, `train_scene_metadata.jsonl`, and `development_scene_metadata.jsonl`. It maps each full scene through the centered camera view to the 96×96 grayscale input. Rebuild these files together so the labels, face geometry and arrays match. The distributed manifests contain the role definition and source summary; run the preparers to create local image paths for this computer.

## Original baseline and comparison

Initialize the local research lease before starting the original trainer. This records ownership of a training run and opens no device connection.

```sh
PYTHONPATH=FaceRecognition work/model-training/.venv/bin/python \
  -m nicla_study.cli init --workspace .

work/model-training/.venv/bin/python detector/scripts/train_fomo.py \
  --cache-dir detector/local/cache \
  --outdir detector/local/baseline-models

work/model-training/.venv/bin/python detector/scripts/evaluate_development.py \
  --cache-dir detector/local/cache \
  --model-dir detector/local/baseline-models \
  --stock-model detector/models/stock_reference/fomo_face_detection-stock.tflite \
  --out detector/local/baseline-evaluation.json
```

The baseline trains both recorded architectures on all ten registered seeds. The original development evaluator expects that two-arm export layout. It evaluates actual TFLite outputs and reports false triggers and crop quality as well as detection quality. [The original result](reports/RESULT.md) records the limited accuracy of that learned detector study.

## Sofia's follow-up trainer

Use the same cache for a ten-seed follow-up run

```sh
work/model-training/.venv/bin/python detector/scripts/train_fomo_v2.py \
  --cache-dir detector/local/cache \
  --outdir detector/local/v2-models \
  --arm rf_sepconv_24 \
  --seeds 20261101 20261102 20261103 20261104 20261105 \
          20261106 20261107 20261108 20261109 20261110
```

Explicitly specifying those seeds expands the preserved script's three-seed default to the project’s ten-seed comparison. Its new `rf_sepconv_24` architecture requires its own model comparison; the original two-arm evaluator is for the baseline layout above.

For a local camera adaptation, prepare independently captured positive and negative folders, then run

```sh
work/model-training/.venv/bin/python detector/scripts/label_own_frames.py \
  --pos /path/to/frames_with_face \
  --neg /path/to/frames_no_face \
  --out detector/local/own_frames.npz
```

Review the generated contact sheet and rejected-frame list. Add `--own-npz detector/local/own_frames.npz` to the v2 training command to enable its built-in held-out local-frame report. Keep a further physical session for independent validation. The original helper’s filename-order split is a development convenience and does not establish independence between adjacent camera frames.

FOMO outputs face-center cells on a 12×12 grid. Its confidence and centered crop need their own calibration and device validation. A detector export alone does not qualify identity accuracy or establish a working Nicla crop. Use the current [deployment instructions](../DEPLOYMENT.md) when evaluating a candidate release.
