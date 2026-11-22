"""Score a frozen C shortlist on final scenes once; never tunes thresholds or crop side."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from common import dump_json, sha256_file
from evaluate_development import evaluate_probabilities, read_jsonl, run_model, compare_float_int8


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache-dir", type=Path, required=True)
    ap.add_argument("--freeze-receipt", type=Path, required=True)
    ap.add_argument("--stock-model", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    freeze = json.loads(args.freeze_receipt.read_text(encoding="utf-8"))
    if not freeze.get("final_access_authorized_after_development_freeze"):
        raise ValueError("A frozen development selection is required before final evaluation")
    selected = freeze["selected_detector"]
    crop_policy = freeze["selected_crop_policy"]
    crop_side = int(freeze["selected_crop_side_window_px"])
    if crop_policy not in ("candidate", "legacy_current"):
        raise ValueError(f"Unsupported frozen crop policy: {crop_policy}")
    model_dir = Path(selected["root_artifact_dir"])
    float_path, int8_path = model_dir / "detector_float.tflite", model_dir / "detector_full_int8.tflite"
    if sha256_file(int8_path) != selected["full_int8_sha256"] or sha256_file(float_path) != selected["float_sha256"]:
        raise ValueError("Frozen model artifact hash mismatch")
    with np.load(args.cache_dir / "final_cache.npz") as cache:
        gray, rgb, windows = cache["gray96"], cache["rgb96"], cache["gray_window"]
    metadata = read_jsonl(args.cache_dir / "final_scene_metadata.jsonl")
    threshold = float(selected["threshold"])
    stock_probabilities, stock_audit = run_model(args.stock_model, gray, rgb, True)
    float_probabilities, float_audit = run_model(float_path, gray, rgb, False)
    int8_probabilities, int8_audit = run_model(int8_path, gray, rgb, False)
    stock_metrics = evaluate_probabilities(stock_probabilities, metadata, 0.40, windows,
                                           crop_selection="legacy_current")
    float_metrics = evaluate_probabilities(float_probabilities, metadata, threshold, windows,
                                           crop_selection=crop_policy, pipeline_crop_side=crop_side)
    int8_metrics = evaluate_probabilities(int8_probabilities, metadata, threshold, windows,
                                          crop_selection=crop_policy, pipeline_crop_side=crop_side)
    result = {"schema": "job-c-frozen-final-evaluation-1",
        "development_freeze_sha256": sha256_file(args.freeze_receipt),
        "final_data_used_for_threshold_or_crop_selection": False,
        "scene_count": len(metadata), "roles": sorted({row.get("role") for row in metadata}),
        "thresholds": {"stock_fixed_reference": 0.40, "selected_detector_frozen_development": threshold},
        "selected_crop_policy": crop_policy, "selected_crop_side_window_px": crop_side,
        "selected_detector": selected,
        "stock_fixed_reference": {"audit": stock_audit,
            "metrics": {key: value for key, value in stock_metrics.items() if key != "scene_records"}},
        "selected_float": {"audit": float_audit,
            "metrics": {key: value for key, value in float_metrics.items() if key != "scene_records"}},
        "selected_full_int8": {"audit": int8_audit,
            "metrics": {key: value for key, value in int8_metrics.items() if key != "scene_records"}},
        "float_vs_full_int8_at_frozen_threshold": compare_float_int8(
            float_probabilities, int8_probabilities, metadata, threshold),
        "final_integration_note": "Results are one held-out report only; no candidate, threshold, or crop method is selected from these metrics."}
    dump_json(args.out, result)
    print(json.dumps({"event":"frozen_final_evaluation_complete", "scene_count":len(metadata),
        "stock_f1":stock_metrics["f1"], "selected_full_int8_f1":int8_metrics["f1"],
        "no_face_false_trigger_rate":int8_metrics["false_trigger_scene_rate"], "output":str(args.out)}, indent=2))


if __name__ == "__main__":
    main()
