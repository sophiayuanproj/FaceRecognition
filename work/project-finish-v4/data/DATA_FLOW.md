# Public-image data flow and experiment map

## Data flow

```mermaid
flowchart TD
  A[Existing SurvFace archive and extraction] --> B[Prior source manifest and October archive receipt]
  B --> C[Read v3 train, development and calibration role manifests only]
  C --> D[Infer sealed v3-final identity complement; skip those image files]
  B --> E[Rehash and decode permitted labeled images plus Olivetti]
  E --> F[source_manifest.jsonl + quality_inventory.jsonl]
  E --> U[Optional v3-train-only fresh pretraining manifest]
  F --> G[Exact SHA and scoped within-identity pHash groups]
  G --> H[Metadata-only five-ID ranking from v3 development identities]
  H --> I[Freeze K01-K05 mapping; use quality only for training/gallery selection]
  G --> J[Assign identity-disjoint OTHER roles from v3 development/calibration pools]
  I --> K[Primary group-safe split and 10 predeclared sensitivity blocks]
  J --> K
  K --> L[Role-gated loader: gray8, float32 [0,1], or quantized input]
  L --> M[Balanced 6-class training batches]
  U --> V[Identity-balanced pretraining batches]
  V --> W[Optional fresh embedding pretraining]
  K --> N[Development and final evaluation manifests; no train augmentation]
  K --> O[Separate presentation manifest and randomized stimulus player]
  P[Olivetti archive] --> Q[Independent 40-person, 6/2/2 sanity split]
  R[WIDER FACE train/validation scenes and boxes] -. external C dependency .-> S[Face localization training]
  T[Target-camera empty scenes] -. pending capture dependency .-> S
```

## Which dataset supports which experiment

| Dataset | Experiment | Records and split | What it can support | Limits |
|---|---|---|---|---|
| SurvFace-derived 96×96 gray crops | Primary five-known classification and embedding/gallery comparison | K01–K05 each have 60 training, 10 stopping, 10 threshold-calibration, 10 development and 20 final unique image groups. Gallery uses the same 60 train records. OTHER is 50 identities × 6 images per role; identities are disjoint across train, calibration, development, final and presentation. | Reproducible public-crop identity benchmark with matched information for classifier and gallery arms. Training examples are selected by the declared quality score within filename camera groups. | All five known identities were in v3 development and their images could have influenced historical model selection. All v4 public results are exploratory. Filename `camN` is only a source grouping tag, not a capture session. This 96×96 derivative does not establish native resolution or target-camera performance. |
| SurvFace v3 train-role identities | Optional fresh embedding pretraining | One quality-valid, exact/pHash group representative per identity/image group; all source IDs are disjoint from v3 development/calibration and the v4 K01–K05 roster. See `pretraining_manifest.jsonl` and `pretraining_class_map.json`. | D can start a new encoder training run with many pseudonymous identities and image diversity. The manifest is group-deduplicated and class-balanced batches are available. | Train new weights for a fresh comparison. Reusing a historical v3 checkpoint makes the arm diagnostic because those checkpoints already trained on this identity pool. |
| Olivetti / Database of Faces transformed derivative | Small-data sanity experiment | Separate by each of 40 identities into target 6 train, 2 development and 2 final images, keeping exact and same-identity pHash groups together. Actual counts may move where a duplicate group contains multiple files. | Exercises loaders and identity code on a distinct legacy dataset. | Only ten source images per identity; no capture/session metadata; transformed derivative provenance and reuse rights are unclear. Never pool its IDs or results with SurvFace. |
| WIDER FACE v1.0 | Face localization / crop training for C | Raw official-page-linked train/validation and bounding-box archives are staged and their archive SHA-256/CRC checks pass; per-image indexing stopped before a verified manifest. See `../reports/scene_data_dependency.json`. | Full scenes with varied scale, pose, occlusion, and backgrounds, with train/validation face boxes. | Per-image decode and annotation-count receipt is incomplete. The source filters images without faces, so session-separated target-camera empty scenes remain required. The official test split has no released box ground truth. |
| Future target-camera empty scenes | Detector false-positive and no-face behavior | Capture full-frame scenes in two separately recorded sessions with lighting and view-location metadata. Keep one session for training and one for development. | Real no-face examples from the camera domain. | Pending access to the instrument camera. No images are fabricated or treated as collected. |

## Frozen image budget and evaluation policy

The known-ID total is 130 unique near-duplicate groups per identity: 60 train, 10 stopping, 10 threshold calibration, 10 development, 20 final, and 20 separate presentation groups. Each selected identity must have at least 130 verified unique groups, at least 60 quality-valid groups available for training, and at least two filename camera groups with ten quality-valid groups each. This screen chooses the roster from metadata only; it does not remove low-quality images from nontraining roles.

## D's paired quality-selection sensitivity (planned, not run)

Use the ten registered `training_roots` as ten paired runs. In the `quality_aware` arm, retain the frozen primary rule: choose 60 unique, quality-valid groups per known identity, distributed across the four filename camera strata (15 per stratum when available; K05 uses the reproducible available-group allocation). In the `uniform_group` arm, sample 60 verified unique groups per known identity without a quality cutoff, uniformly within the same filename-camera strata and matching the quality-aware arm's per-stratum counts. Use each root's registered seed for this selection. Keep OTHER training groups, model initialization, augmentation, training schedule, stopping, calibration, development, final, and presentation data identical within each pair. Do not use development or final scores to change the roster or selection rule. Report all ten paired differences and selected-set quality summaries; do not treat roots as ten independent identities.

This ablation is a D training task and has not been run by B. It distinguishes the fixed primary quality-curated manifest from a same-budget, group-safe sample without quality ranking. The training quality score is not a deployed rejection threshold.

Unknown identity selection requires at least six quality-valid groups per identity so `unknown_train` can receive six quality-selected groups. The other unknown roles draw six reproducibly selected groups from all verified images, including groups below the quality threshold. The same 250 eligible public identities are assigned into five disjoint pools of 50 using each registered root; each root independently shuffles which eligible identities fill the role pools. No result or score enters any assignment.

Reports should give macro-averaged class metrics with fixed class order K01, K02, K03, K04, K05, OTHER. Do not weight a pooled image count as if 300 OTHER images were 30 times more important than ten images from one known class. Threshold, development and final uncertainty must preserve the identity and image-group structure.

The loader reads the source file in place, checks its SHA-256, validates its gray8 96×96 format, and creates transforms in memory. It never creates a second image copy. `final` and `unknown_final` loading requires a frozen receipt with model, config, calibration and protocol hashes; training-only augmentation cannot run on any evaluation row.

## Runtime quality rejection and coverage

Keep the training curation score separate from the deployed quality gate. If a deployed crop-quality gate is used, choose its threshold using only `threshold_calibration`, freeze it with the model/config/calibration receipt, and report the no-gate baseline and gated result together. On the controlled presentation task, define face-present coverage as the number of face-present trials that produce a detected crop passing the gate and an identity/OTHER output divided by all face-present trials. Count both detector misses and quality rejections in the denominator and report them separately; a rejection is an abstention, not a removed sample. Also report conditional recognition performance among accepted outputs, false accepts on empty trials, and unknown-to-known false accepts. Break out coverage and misses by displayed face size, position, and recorded lighting, motion, and range. Tune on calibration/development only and freeze before final; never improve final coverage by dropping rejected trials.

## Reproduction commands

From the `ok-x20` workspace root:

```sh
PYTHONPATH=FaceRecognition FaceRecognition/.venv-nicla/bin/python -m nicla_study.curate_public_data --workspace "$PWD"
PYTHONPATH=FaceRecognition FaceRecognition/.venv-nicla/bin/python -m nicla_study.prepare_pretraining_manifest --workspace "$PWD"
PYTHONPATH=FaceRecognition FaceRecognition/.venv-nicla/bin/python -m nicla_study.validate_public_data --workspace "$PWD"
```

Training code should explicitly select the dataset and training roles, then use `balanced_class_batches`:

```python
rows = select_rows(
    data_root / "primary_split_manifest.jsonl",
    "train",
    dataset="survface_96",
    roles=("train", "unknown_train"),
)
for images, labels, metadata in balanced_class_batches(
    rows, workspace, batch_size=24, seed=20261101
):
    ...
```

For a fresh optional identity encoder, select `pretraining_manifest.jsonl` with `action="train"` and use `balanced_identity_batches`; it samples one image from each pseudonymous identity in each batch. Do not initialize from a historical v3 checkpoint for a fresh independent comparison.

The selected images stay private/local. Shareable reports contain only counts, pseudonyms and source citations; the K01–K05 source-ID map and image manifests remain in the local project data directory.
