"""Replay a frozen TFLite classifier on calibration-separated development crops."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "FaceRecognition"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("result", type=Path, help="ROOT_RESULT.json from a training seed")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    from nicla_study import d_campaign as d
    import numpy as np
    result = json.loads(args.result.read_text())
    paths = d.study_paths(ROOT)
    if d.sha256_file(paths["primary"]) != result["train_manifest_source_sha256"]:
        raise ValueError("Development manifest differs from training receipt")
    known = d.dataset_role_rows(ROOT, paths["primary"], "survface_96", ("development",), "select")
    unknown = d.dataset_role_rows(ROOT, paths["primary"], "survface_96", ("unknown_development",), "select")
    gray = d.cache_gray8(ROOT, known + unknown, "select")
    metrics = {}
    for backend in ("float", "int8"):
        model = args.result.parent / f"classifier_{backend}.tflite"
        if d.sha256_file(model) != result[f"{backend}_classifier_export"]["sha256"]:
            raise ValueError("Model differs from training receipt")
        logits = d.tflite_outputs(model, gray)
        if not np.isfinite(logits).all():
            raise ValueError("Nonfinite model output")
        metrics[backend] = d.known_macro_and_open_metrics(
            logits, known, unknown, result[f"{backend}_calibration"]["threshold"], 6)
    d.write_json(args.output, {"scope": "public crop development, host only",
                               "seed": result["root_seed"], "metrics": metrics})
    print(args.output)


if __name__ == "__main__":
    main()
