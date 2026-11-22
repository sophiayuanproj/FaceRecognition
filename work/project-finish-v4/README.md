# Preserved five-person and compression study

The current reproducible training route is [TRAINING.md](../../TRAINING.md). This directory retains the original frozen configuration, seed registry, role manifests, code and result summaries. Source images and newly generated output are local inputs.

| Directory | Contents |
| --- | --- |
| `config/`, `D_MATRIX_v1.json` | Experiment definitions, registered seeds, contracts and recorded environment |
| `data/` | Frozen non-final role manifests and derived training selections |
| `models/pretrained/` | Two hash-pinned grayscale Edge Impulse MobileNetV1 starting checkpoints |
| `runs/D/development_search/` | Recorded configuration selection used by the current wrapper |
| `runs/E/scripts/` | Capacity, export, QAT, pruning, physical channel reduction, clustering, distillation and analysis methods |
| `reports/`, `handoffs/` | Historical outcomes, summary tables and claim limitations |

The optional GitHub release restores saved fitted weights, converted models and their matching per-root receipts at their original relative paths. It excludes source photos and private camera files. Restore it with `python3 workflow/restore_artifacts.py` from the repository root when replaying the original experiments. The archival `CURRENT_STATUS.md` refers to its dated research run; use the root `START_HERE.md` for the current application state.
