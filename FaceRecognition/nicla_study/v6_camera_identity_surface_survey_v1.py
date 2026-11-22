"""Summarize a bounded volatile Nicla identity survey on visible development scenes."""

from __future__ import annotations

import hashlib
import json
import statistics
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
STUDY = ROOT / "work/project-research-v5"
PRIVATE = STUDY / "private/camera_identity_survey_v1"
OUT = STUDY / "runs/camera_identity_survey_v1"
PROGRAM_SHA = "aa25bec39c71fd4ba147486f08f27c04bafed397fa12a0f90005ede6f5c44263"
ENCODER_SHA = "35f587d51b26bdcc833b4b65791ec91401a98664dbffaa4c33f7088f27bcb1d2"
HEAD_SHA = "987b7776721a38b5b773d4d97af4b221ce9c795b8c5936fa825d4ec3fe7a9747"

CASES = {
    "K01": ("K01", "development", "device_K01"),
    "K02": ("K02", "development", "device_K02_retry2"),
    "K03": ("K03", "development", "device_K03"),
    "K04": ("K04", "development", "device_K04"),
    "K05": ("K05", "development", "device_K05"),
    "OTHER": ("OTHER", "unknown_development", "device_OTHER"),
    "BLANK": ("BLANK", "negative_control", "device_BLANK"),
}


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    result_path = OUT / "RESULT.json"
    private_path = PRIVATE / "DEVICE_RESULTS_PRIVATE.json"
    if result_path.exists() or private_path.exists():
        raise FileExistsError("preserving existing survey aggregates")
    display = json.loads((STUDY / "runs/optical/DISPLAY_SOURCES.json").read_text())
    sources = {}
    for row in display["items"]:
        label = row.get("label")
        if label in {"K01", "K02", "K03", "K04", "K05"} and row.get("role") == "development":
            sources.setdefault(label, row)
        elif label == "OTHER" and row.get("role") == "unknown_development":
            sources.setdefault("OTHER", row)

    private_rows = []
    outcomes = {}
    elapsed = []
    for case, (label, role, run_dir) in CASES.items():
        path = PRIVATE / run_dir / "RESULT.json"
        rows = json.loads(path.read_text())
        if not rows or rows[-1].get("status") != "executed":
            raise ValueError("No successful device record for " + case)
        record = rows[-1]
        if record.get("program_sha256") != PROGRAM_SHA:
            raise ValueError("Device program hash mismatch for " + case)
        device = record["records"][0]
        if device.get("encoder_sha256") != ENCODER_SHA or device.get("head_sha256") != HEAD_SHA:
            raise ValueError("Model binding mismatch for " + case)
        if device.get("board_files_written") is not False or device.get("startup_path_changed") is not False:
            raise ValueError("Unexpected persistent board mutation in " + case)
        elapsed_ms = device["total_elapsed_us"] / 1000.0
        elapsed.append(elapsed_ms)
        source = sources.get(case)
        prediction = device.get("predicted_class")
        expected = case if case in {"K01", "K02", "K03", "K04", "K05"} else None
        correct_top1 = prediction == expected if expected else None
        correct_accept = device.get("decision") == expected if expected else None
        wrong_accept = bool(expected and device.get("decision") in {"K01", "K02", "K03", "K04", "K05"} and device.get("decision") != expected)
        row = {
            "case": case,
            "expected_role": role,
            "expected_label": expected,
            "source_record_id": source.get("record_id") if source else None,
            "source_identity": source.get("source_identity") if source else None,
            "source_group": source.get("source_group") if source else None,
            "display_token": source.get("token") if source else None,
            "device_output_path": str(path.relative_to(ROOT)),
            "device_stdout_sha256": record.get("stdout_sha256"),
            "decision": device.get("decision"),
            "predicted_class": prediction,
            "correct_raw_top1": correct_top1,
            "correct_accepted": correct_accept,
            "wrong_known_accept": wrong_accept,
            "fomo_accepted_clusters": device.get("fomo_accepted_cluster_count"),
            "fomo_raw_clusters": device.get("fomo_raw_cluster_count"),
            "fomo_load_ms": (device.get("fomo_load_us") or 0) / 1000.0,
            "fomo_predict_ms": (device.get("fomo_predict_us") or 0) / 1000.0,
            "fomo_scores": device.get("fomo_accepted_scores"),
            "max_known_probability_binary32": device.get("max_known_probability_binary32"),
            "class_probabilities": device.get("class_probabilities"),
            "elapsed_ms": elapsed_ms,
            "encoder_predict_ms": (device.get("encoder_predict_us") or 0) / 1000.0,
            "tensor_sha256": device.get("tensor_sha256"),
            "encoder_output_sha256": device.get("encoder_output_sha256"),
        }
        private_rows.append(row)
        if expected:
            outcomes[case] = {
                "n": 1,
                "detector_single_cluster": int(device.get("fomo_accepted_cluster_count") == 1),
                "no_face": int(device.get("decision") == "NO_FACE"),
                "raw_top1_correct": int(bool(correct_top1)),
                "correct_accepted": int(bool(correct_accept)),
                "wrong_known_accepts": int(wrong_accept),
                "uncertain": int(device.get("decision") == "UNCERTAIN"),
            }

    known = [row for row in private_rows if row["expected_label"]]
    unknown = next(row for row in private_rows if row["case"] == "OTHER")
    blank = next(row for row in private_rows if row["case"] == "BLANK")
    result = {
        "schema": "nicla-live-camera-identity-survey-v1",
        "status": "volatile_end_to_end_diagnostic_not_qualified",
        "future_final_used": False,
        "display_size_vmin": 100,
        "physical_setup_count": 1,
        "known_development_rows": len(known),
        "unknown_development_groups": 1,
        "blank_controls": 1,
        "training_roots_for_on_device_head": [56002],
        "learned_comparison": False,
        "board_program_sha256": PROGRAM_SHA,
        "encoder_sha256": ENCODER_SHA,
        "head_sha256": HEAD_SHA,
        "software_reset_blocks": len(private_rows),
        "cold_power_cycles": 0,
        "board_files_written": False,
        "startup_path_changed": False,
        "power_measured": False,
        "known_metrics_all_rows": {
            "raw_top1_correct": sum(bool(row["correct_raw_top1"]) for row in known),
            "correct_accepts": sum(bool(row["correct_accepted"]) for row in known),
            "wrong_known_accepts": sum(row["wrong_known_accept"] for row in known),
            "accepted_coverage": sum(row["decision"] in {"K01", "K02", "K03", "K04", "K05"} for row in known) / len(known),
            "single_cluster_detection": sum(row["fomo_accepted_clusters"] == 1 for row in known),
            "rows": len(known),
        },
        "known_per_label_counts": outcomes,
        "unknown_result": {
            "groups": 1,
            "decision": unknown["decision"],
            "fomo_accepted_clusters": unknown["fomo_accepted_clusters"],
            "false_known_accepts": int(unknown["decision"] in {"K01", "K02", "K03", "K04", "K05"}),
        },
        "blank_result": {"controls": 1, "decision": blank["decision"], "fomo_accepted_clusters": blank["fomo_accepted_clusters"]},
        "latency_ms": {"n": len(elapsed), "min": min(elapsed), "median": statistics.median(elapsed), "max": max(elapsed), "p95_claimed": False},
        "arena_bytes": 243856,
        "model_stage_ms_median": {
            "fomo_load": statistics.median(row["fomo_load_ms"] for row in private_rows),
            "fomo_predict": statistics.median(row["fomo_predict_ms"] for row in private_rows),
            "encoder_predict_single_cluster": statistics.median(row["encoder_predict_ms"] for row in private_rows if row["encoder_predict_ms"] > 0),
        },
        "per_condition_probabilities_and_predictions": "private/camera_identity_survey_v1/DEVICE_RESULTS_PRIVATE.json",
        "raw_transcripts": "private/camera_identity_survey_v1/device_*/",
        "limitations": [
            "one development frame per known identity, one unknown group and one blank control; all from one fixed screen setup",
            "not independent physical blocks, startup ACKs, cold starts, a final evaluation, or a population risk estimate",
            "K02 face was visible in the separately captured live-preview frame but FOMO returned zero clusters",
            "the K05 frame was accepted as K03, a wrong known identity; do not deploy this head",
            "the actual REPL frames were not retained as RGB565 files for each row; tensor hashes and private full transcripts are retained",
            "seven one-shot latencies are not a p95 estimate",
        ],
    }
    OUT.mkdir(parents=True, exist_ok=False)
    result_path.write_text(json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n")
    private_path.write_text(json.dumps({"schema": "nicla-live-camera-identity-survey-private-v1", "rows": private_rows}, indent=2, sort_keys=True, allow_nan=False) + "\n")
    print(json.dumps({"known_rows": len(known), "known_raw_top1_correct": result["known_metrics_all_rows"]["raw_top1_correct"], "known_correct_accepts": result["known_metrics_all_rows"]["correct_accepts"], "wrong_known_accepts": result["known_metrics_all_rows"]["wrong_known_accepts"], "unknown_decision": unknown["decision"], "blank_decision": blank["decision"], "elapsed_ms": result["latency_ms"]}, sort_keys=True))


if __name__ == "__main__":
    main()
