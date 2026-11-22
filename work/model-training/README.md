# Earlier custom embedding experiments

The current clone entry point is [TRAINING.md](../../TRAINING.md). This directory preserves the custom supervised-contrastive embedding trainers, conversion and compression scripts, seed plans, requirements, and concise historical reports from the October 2026 study.

[README.historical.md](README.historical.md) describes the original sequence, source data roles, metrics and later correction to INT8 representative-data selection. Paths in those saved reports describe the original run. Use new output directories for further experiments, and use the current [data restoration instructions](../../DATA.md).

`scripts/train_embedder.py`, `scripts/convert_eval.py`, `scripts/run_training_matrix.py`, and `scripts/run_eval_matrix.py` are the corresponding methods. The broader dataset inventory can be regenerated with `work/dataset-prep/scripts/audit_datasets.py --extract`. The five-person MobileNet study and matched compression experiments are under `work/project-finish-v4/`. The current deployed device assets are under `FaceRecognition/models/deployment-v1/`.
