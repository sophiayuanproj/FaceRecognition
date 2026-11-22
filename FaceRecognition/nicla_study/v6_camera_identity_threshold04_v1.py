"""Compare the repository's pre-existing 0.4 FOMO setting on the same scenes.

This is a separate volatile board experiment. It reuses the seven-scene
render/dwell/preview protocol but writes only to a new threshold-v1 namespace.
The identity head remains the unqualified one-root diagnostic.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
STUDY = ROOT / "work/project-research-v5"
EXPERIMENT = STUDY / "runs/camera_identity_threshold04_v1"
PRIVATE = STUDY / "private/camera_identity_threshold04_v1"
PROGRAM = EXPERIMENT / "DEVICE_PROGRAM.py"
BASE_PROGRAM = STUDY / "runs/camera_identity_prototype_v1/DEVICE_PROGRAM.py"
RUNNER = Path(__file__).with_name("v6_run_verified_live_scene_survey_v2.py")


def sha(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def prepare_program() -> dict:
    if EXPERIMENT.exists() or PRIVATE.exists():
        raise FileExistsError("preserving any prior camera_identity_threshold04_v1 output")
    source = BASE_PROGRAM.read_text(encoding="utf-8")
    replacements = {
        "Fomo(threshold=.5)": "Fomo(threshold=.4)",
        "float(item[1])>.5": "float(item[1])>.4",
        '"fomo_coordinate_space":"camera_xywh_px",':
            '"fomo_coordinate_space":"camera_xywh_px","fomo_postprocess_threshold_configured":0.4,"fomo_score_filter":">0.4",',
    }
    counts = {old: source.count(old) for old in replacements}
    if counts != {old: 1 for old in replacements}:
        raise ValueError("volatile probe source changed; exact replacement contract failed: " + repr(counts))
    for old, new in replacements.items():
        source = source.replace(old, new)
    EXPERIMENT.mkdir(parents=True, exist_ok=False)
    PRIVATE.mkdir(parents=True, exist_ok=False)
    payload = source.encode("utf-8")
    PROGRAM.write_bytes(payload)
    manifest = {
        "schema": "nicla-camera-fomo-threshold-comparison-v1",
        "experiment": "camera_identity_threshold04_v1",
        "base_program": str(BASE_PROGRAM.relative_to(ROOT)),
        "base_program_sha256": sha(BASE_PROGRAM.read_bytes()),
        "program": str(PROGRAM.relative_to(ROOT)),
        "program_sha256": sha(payload),
        "fomo_postprocess_threshold_configured": 0.4,
        "fomo_cluster_score_filter": ">0.4",
        "changed_source_occurrences": counts,
        "identity_encoder_sha256": "35f587d51b26bdcc833b4b65791ec91401a98664dbffaa4c33f7088f27bcb1d2",
        "identity_head_sha256": "987b7776721a38b5b773d4d97af4b221ce9c795b8c5936fa825d4ec3fe7a9747",
        "identity_head_training_roots": [56002],
        "scene_set": ["K01", "K02", "K03", "K04", "K05", "OTHER", "BLANK"],
        "scene_dwell_s": 2.5,
        "future_final_used": False,
        "board_files_written": False,
        "startup_path_changed": False,
        "power_measured": False,
        "purpose": "diagnostic sensitivity check of the repository's existing 0.4 detector setting against the prior matched 0.5 one-shot set; not a selected/qualified operating point",
    }
    (EXPERIMENT / "PROGRAM_MANIFEST.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return manifest


def run() -> None:
    manifest = prepare_program()
    spec = importlib.util.spec_from_file_location("v6_verified_scene_runner_threshold04", RUNNER)
    if spec is None or spec.loader is None:
        raise RuntimeError("could not load verified scene runner")
    runner = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(runner)
    runner.PROGRAM = PROGRAM
    runner.PROGRAM_SHA = manifest["program_sha256"]
    runner.PUBLIC_RESULT = EXPERIMENT / "RESULT.json"
    runner.PRIVATE_RESULT = PRIVATE / "PRIVATE_RESULT.json"
    runner.PREVIEW_RUNS = PRIVATE / "device_runs"
    runner.PREVIEW_FRAMES = PRIVATE / "device_previews"
    runner.main()


if __name__ == "__main__":
    run()
