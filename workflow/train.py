"""Train and evaluate ten matched six-class MobileNetV1 candidates."""
import argparse
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "FaceRecognition"))
os.environ.setdefault("TF_NUM_INTRAOP_THREADS", "2")
os.environ.setdefault("TF_NUM_INTEROP_THREADS", "1")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--width", type=float, choices=(0.1, 0.25), default=0.25)
    parser.add_argument("--quick", action="store_true", help="One-epoch diagnostic for each of the ten seeds")
    parser.add_argument("--plan", action="store_true", help="Print the frozen configuration without loading images or TensorFlow")
    args = parser.parse_args()
    study = ROOT / "work/project-finish-v4"
    search = json.loads((study / "runs/D/development_search/DEVELOPMENT_SEARCH.json").read_text())
    plan = json.loads((study / "D_MATRIX_v1.json").read_text())
    seeds = plan["training_roots"]
    if len(seeds) != 10 or len(set(seeds)) != 10:
        raise ValueError("Expected ten distinct registered training seeds")
    if args.quick:
        for choice in search["selected"].values():
            choice["selected_config"].update(head_epochs_max=1, fine_tune_epochs_max=0)
    width = f"{round(args.width * 100):03d}"
    arm = {"id": f"sv_mnv1_a{width}_transfer_6", "width": args.width}
    if args.plan:
        print(json.dumps({"arm": arm, "training_seeds": seeds,
                          "output": str(args.out.resolve()), "diagnostic_only": args.quick,
                          "configuration": search["selected"][f"a{str(args.width).replace('.', '')}_transfer"]["selected_config"],
                          "data_manifest": str((study / "data/primary_split_manifest.jsonl").relative_to(ROOT))}, indent=2))
        return
    from nicla_study import d_campaign as d
    paths = d.study_paths(ROOT)
    paths["runs"] = args.out.resolve()
    paths["runs"].mkdir(parents=True, exist_ok=False)
    data = d.load_role_arrays(ROOT, paths, "survface_96")
    roots = []
    for seed in seeds:
        result = d.train_one_classifier(ROOT, paths, arm, seed, search, data)
        result_path = paths["runs"] / "survface" / arm["id"] / f"seed-{seed}/ROOT_RESULT.json"
        roots.append({
            "seed": seed, "root": str(result_path.relative_to(paths["runs"])),
            "float_development": result["float_development"],
            "int8_development": result["int8_development"],
            "int8_bytes": result["int8_classifier_export"]["bytes"],
        })
        print(f"Completed seed {seed}", flush=True)
    from nicla_study.fs import atomic_write_json
    atomic_write_json(paths["runs"] / "CAMPAIGN.json", {
        "status": "complete", "diagnostic_only": args.quick,
        "scope": "public crop development, host only", "width": args.width,
        "primary_manifest_sha256": d.sha256_file(paths["primary"]), "roots": roots,
    })


if __name__ == "__main__":
    main()
