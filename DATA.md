# Dataset provenance and restoration

This repository contains preparation scripts, frozen split manifests, seed plans, model code, and result summaries. Original images and private camera captures remain local. The optional saved-model artifact download comes from this repository's GitHub release through `workflow/restore_artifacts.py`.

| Data | Recorded use | Original source |
| --- | --- | --- |
| SurvFace-derived 96×96 grayscale crops | Main K01–K05 plus OTHER identity study | [SurvFace](https://qmul-survface.github.io/) and the exact transformed ZIP originally supplied by the team |
| Olivetti 96×96 archive | Separate sanity study | [Scikit-learn Olivetti documentation](https://scikit-learn.org/stable/datasets/real_world.html#olivetti-faces-dataset) and the exact transformed team ZIP |
| WIDER FACE and annotations | Full-scene detector training and development | [Official WIDER FACE project](https://shuoyang1213.me/WIDERFACE/) |
| PASS v3 | Scenes without people for detector negatives | [Official PASS project](https://www.robots.ox.ac.uk/~vgg/data/pass/) |
| LFW | Historical screen-photo and host-alignment diagnostics | [LFW project](https://vis-www.cs.umass.edu/lfw/) |

The transformed SurvFace and Olivetti archives are existing study inputs. An upstream original download can differ in filenames, pixels, and hashes, so obtaining it does not automatically reproduce the transformed archive. Keep the original provider's usage conditions and attribution with the locally obtained data. WIDER FACE is released for non-commercial research. The provenance and redistribution permission for the team-transformed crop archives remain incompletely documented, so the repository supplies their hashes and restoration code rather than republishing the images. These source links identify inputs; the software, checkpoints, and tools have their own GitHub distribution.

For a course review that requires retraining on the exact crops, contact the project authors for access to the original transformed archives under the applicable dataset terms. Use the hashes below to check the copy before restoring it. The saved-model restoration and raw power replay can run without those archives.

## Exact crop inputs

```text
survface_96.zip       f61b38dd7364e2dd912ec573c5e9180a5de3d68b95b910b4b5dddda42b265fe6
olivetti_96.zip       552c65c1b925374ecae40b0808b1c91c280f0d0baa8c74080bc26e4dafdad256
WIDER_train.zip      e23b76129c825cafae8be944f65310b2e1ba1c76885afe732f179c41e5ed6d59
WIDER_val.zip        f9efbd09f28c5d2d884be8c0eaef3967158c866a593fc36ab0413e4b2a58a17a
wider_face_split.zip c7561e4f5e7a118c249e0a5c5c902b0de90bbf120d7da9fa28d99041f68a8a5c
```

Run from the cloned repository root with the matching local ZIP

```sh
python3 workflow/prepare_data.py /path/to/survface_96.zip
python3 workflow/prepare_data.py /path/to/olivetti_96.zip --dataset olivetti_96
python3 workflow/verify.py --data
```

The restorer verifies the archive SHA-256 before reading images, checks the pixel-file hash from each selected manifest row, and writes only paths beneath the repository. It restores the frozen non-final roles, including the derived training-selection manifests. The primary benchmark has 1,350 rows. Known people have disjoint image groups across roles. OTHER people are disjoint across training, calibration, and development. The five known identities had prior exploratory exposure in the earlier study, which limits claims about a pristine final evaluation.

## Earlier embedding study

The original `work/model-training/scripts/` campaign uses the broader dataset inventory. To regenerate it, place the original crop ZIPs in `work/dataset-prep/raw/` with the exact filenames above, then run

```sh
work/model-training/.venv/bin/python work/dataset-prep/scripts/audit_datasets.py --extract
work/model-training/.venv/bin/python work/dataset-prep/scripts/inspect_survface_pairs.py
```

The first command validates and extracts the locally supplied archives and recreates `work/dataset-prep/reports/manifest.jsonl`. The second recreates the original verification-pair inventory. Keep any previously inspected final pair outcomes out of new model selection. The historical embedding README records the existing exposure and the later representative-pool correction.

## Detector scenes

WIDER and PASS have their own source and attribution receipts under `detector/manifests/` and `work/project-finish-v4/reports/scene_data_dependency.json`. Use the preparation scripts' `--help` to provide your actual downloaded locations. Split WIDER at the event level and PASS by uploader. Preserve source annotations when producing the detector's 96×96 input and do not turn unannotated faces into negative scenes.

Nicla captures used during development and camera-to-screen trials belong to their recorded physical session. Keep their boot/frame IDs, source role, lighting conditions, and hashes when reviewing them. Repeated pictures of one displayed portrait provide a functional camera check. Fresh consenting people and independently captured sessions are needed for broader live-camera claims.
