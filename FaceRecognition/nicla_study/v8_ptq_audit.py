"""Independent artifact audit for the v8 alpha0.25 host INT8 PTQ replay.

This reads only the already-produced checkpoint, model, preflight, results,
and private predictions. It does not load images, access a camera, or train.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import tensorflow as tf

ROOT = Path(__file__).resolve().parents[2]
RESEARCH = ROOT / "work/project-research-v5"
RUN_ID = "v8_lfw_source_mnv1_a025_ei_scout57101_launch01"
RUN = RESEARCH / "runs/accuracy_search_v1" / RUN_ID
REPORT_PATH = RUN / "INT8_QUANTIZATION_EVAL.json"
RECEIPT_PATH = RESEARCH / "reports/mac_verification_v1/V8_ALPHA025_INT8_PRIMARY_REVIEW.json"
CLASSES = ["K01", "K02", "K03", "K04", "K05", "OTHER"]
KNOWN = set(CLASSES[:5])


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def add_check(checks: list[dict], name: str, condition: bool, detail: str) -> None:
    checks.append({"name": name, "passed": bool(condition), "detail": detail})


def summarize_predictions(rows: list[dict], scope: str, role: str) -> dict:
    subset = [row for row in rows if row["scope"] == scope and row["role"] == role]
    if not subset:
        raise ValueError(f"missing predictions for {scope}/{role}")
    known_rows = [row for row in subset if row["true_label"] in KNOWN]
    unknown_rows = [row for row in subset if row["true_label"] == "OTHER"]
    known_valid = [row for row in known_rows if row["valid_input"]]
    unknown_groups: dict[str, list[dict]] = defaultdict(list)
    for row in unknown_rows:
        unknown_groups[row["source_identity"]].append(row)
    raw_correct = sum(row["predicted_class"] == row["true_label"] for row in known_valid)
    known_accepted = [row for row in known_rows if row["accepted"]]
    unknown_false_groups = sum(any(row["accepted"] for row in group) for group in unknown_groups.values())
    return {
        "known_rows": len(known_rows),
        "known_valid_rows": len(known_valid),
        "known_raw_top1": raw_correct,
        "known_correct_accepted": sum(row["predicted_class"] == row["true_label"] for row in known_accepted),
        "known_wrong_accepted": sum(row["predicted_class"] != row["true_label"] for row in known_accepted),
        "known_accepted": len(known_accepted),
        "unknown_rows": len(unknown_rows),
        "unknown_valid_rows": sum(row["valid_input"] for row in unknown_rows),
        "unknown_identity_groups": len(unknown_groups),
        "unknown_false_accepted_identity_groups": unknown_false_groups,
    }


def summarize_calibration(rows: list[dict], scope: str) -> dict:
    known = [row for row in rows if row["scope"] == scope and row["role"] == "threshold_calibration"]
    unknown = [row for row in rows if row["scope"] == scope and row["role"] == "unknown_calibration"]
    if not known or not unknown:
        raise ValueError(f"missing calibration predictions for {scope}")
    groups: dict[str, float] = {}
    for row in unknown:
        key = str(row["source_identity"])
        groups[key] = max(groups.get(key, -np.inf), float(row["score_max_known_float32"]))
    maximum = np.float32(max(groups.values()))
    threshold = np.nextafter(maximum, np.float32(np.inf), dtype=np.float32)
    known_correct_accepted = sum(row["accepted"] and row["predicted_class"] == row["true_label"] for row in known)
    false_groups = sum(any(row["accepted"] for row in unknown if str(row["source_identity"]) == key)
                       for key in groups)
    all_thresholds = {np.float32(row["threshold_float32"]).tobytes() for row in known + unknown}
    return {
        "known_rows": len(known), "unknown_rows": len(unknown),
        "unknown_identity_groups": len(groups),
        "max_unknown_identity_score": float(maximum),
        "threshold_nextafter_float32": float(threshold),
        "known_correct_accepted": int(known_correct_accepted),
        "unknown_false_accepted_groups": int(false_groups),
        "prediction_threshold_values": len(all_thresholds),
        "prediction_threshold": float(np.frombuffer(next(iter(all_thresholds)), dtype=np.float32)[0]),
    }


def report_metric_block(summary: dict, report: dict, scope: str, key: str) -> bool:
    block = report[scope][key]
    return (
        summary["known_rows"] == block["known_rows"]
        and summary["known_valid_rows"] == block["known_detection_valid_rows"]
        and summary["known_raw_top1"] == round(block["known_raw_top1_end_to_end"] * block["known_rows"])
        and summary["known_correct_accepted"] == block["known_correct_accepted"]
        and summary["known_wrong_accepted"] == block["known_wrong_accepted"]
        and summary["unknown_rows"] == block["unknown_rows"]
        and summary["unknown_valid_rows"] == block["unknown_detection_valid_rows"]
        and summary["unknown_identity_groups"] == block["unknown_identity_groups"]
        and summary["unknown_false_accepted_identity_groups"] == block["unknown_false_accepted_identity_groups"]
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--write-receipt", action="store_true")
    args = ap.parse_args()

    report = read_json(REPORT_PATH)
    result_path = RUN / "RESULT.json"
    config_path = RESEARCH / "runs/accuracy_search_v1/configs" / f"{RUN_ID}.json"
    preflight_path = RESEARCH / "runs/accuracy_search_v1/preflight" / f"{RUN_ID}.json"
    float_predictions_path = RUN / "private/PREDICTIONS.jsonl"
    predictions_path = RUN / "private/INT8_PREDICTIONS.jsonl"
    model_path = RUN / "model_int8_trainrep_v1.tflite"
    result, config, preflight = map(read_json, (result_path, config_path, preflight_path))
    float_predictions = [json.loads(line) for line in float_predictions_path.read_text().splitlines() if line]
    predictions = [json.loads(line) for line in predictions_path.read_text().splitlines() if line]

    checks: list[dict] = []
    add_check(checks, "run_binding", report["run_id"] == RUN_ID and report["status"] == "complete_host_reference_only",
              f"run={report['run_id']} status={report['status']}")
    add_check(checks, "config_binding", sha(config_path) == result["config_sha256"] == report["config_sha256"],
              f"config_sha256={sha(config_path)}")
    add_check(checks, "preflight_binding", sha(preflight_path) == report["preflight_sha256"],
              f"preflight_sha256={sha(preflight_path)}")
    add_check(checks, "parent_result_binding", sha(result_path) == report["parent_result"]["sha256"],
              f"result_sha256={sha(result_path)}")
    model_sha = sha(model_path)
    add_check(checks, "model_binding", model_sha == report["export"]["sha256"] and model_path.stat().st_size == report["export"]["bytes"],
              f"model_sha256={model_sha} bytes={model_path.stat().st_size}")
    add_check(checks, "selected_weights_binding", {row["model_weight_sha256"] for row in float_predictions} == {report["selected_model_weight_sha256"]},
              f"selected_model_weight_sha256={report['selected_model_weight_sha256']}")
    allowed_representative_roles = {"train", "other_train"}
    representative = report["representative_source"]
    add_check(checks, "representative_role_boundary", set(representative["roles"]) == allowed_representative_roles
              and set(representative["role_counts"]) == allowed_representative_roles
              and representative["calibration_development_future_final_used_for_representative_set"] is False,
              f"roles={representative['roles']} rows={representative['rows']}")
    camera_receipt = report["old_optical_camera"]["camera_receipt"]
    role_names = set(preflight.get("source_rows", {}).get("selection", []))
    prediction_roles = {row["role"] for row in predictions}
    add_check(checks, "future_final_boundary", config["future_final_access"] is False
              and preflight.get("source_rows", {}).get("future_final_paths_resolved") is False
              and "future_final" not in role_names and "future_final" not in prediction_roles
              and camera_receipt.get("future_final_frames_opened") is False,
              "config disabled; source preflight did not resolve final paths; camera receipt and saved predictions contain no final evaluation")
    add_check(checks, "private_prediction_binding", len(predictions) == report["private_predictions"]["rows"]
              and sha(predictions_path) == report["private_predictions"]["sha256"],
              f"prediction_rows={len(predictions)} sha256={sha(predictions_path)}")

    role_pairs = [("LFW", "threshold_calibration"), ("LFW", "unknown_calibration"),
                  ("old_optical_camera", "threshold_calibration"), ("old_optical_camera", "unknown_calibration")]
    thresholds = {}
    for scope, _ in role_pairs:
        if scope in thresholds:
            continue
        summary = summarize_calibration(predictions, scope)
        report_scope = "source_lfw" if scope == "LFW" else "old_optical_camera"
        saved = report[report_scope]["calibration_int8"]
        threshold_matches = np.float32(summary["threshold_nextafter_float32"]).tobytes() == np.float32(saved["threshold_float32"]).tobytes()
        max_key = "calibration_max_unknown_identity_score" if scope == "LFW" else "calibration_max_unknown_score"
        add_check(checks, f"{scope}_calibration_threshold", threshold_matches
                  and np.float32(summary["max_unknown_identity_score"]).tobytes() == np.float32(saved[max_key]).tobytes()
                  and summary["prediction_threshold_values"] == 1
                  and np.float32(summary["prediction_threshold"]).tobytes() == np.float32(saved["threshold_float32"]).tobytes(),
                  json.dumps(summary, sort_keys=True))
        thresholds[scope] = summary

    dev_summaries = {}
    for scope, report_scope in [("LFW", "source_lfw"), ("old_optical_camera", "old_optical_camera")]:
        summary = summarize_predictions(predictions, scope, "development" if scope == "LFW" else "development")
        # Camera development known and unknown use the same role; compute the unknowns separately below.
        if scope == "old_optical_camera":
            known_rows = [row for row in predictions if row["scope"] == scope and row["role"] == "development" and row["true_label"] in KNOWN]
            unknown_rows = [row for row in predictions if row["scope"] == scope and row["role"] == "unknown_development"]
            unknown_groups = defaultdict(list)
            for row in unknown_rows:
                unknown_groups[row["source_identity"]].append(row)
            summary.update({
                "known_rows": len(known_rows), "known_valid_rows": sum(row["valid_input"] for row in known_rows),
                "known_raw_top1": sum(row["valid_input"] and row["predicted_class"] == row["true_label"] for row in known_rows),
                "known_correct_accepted": sum(row["accepted"] and row["predicted_class"] == row["true_label"] for row in known_rows),
                "known_wrong_accepted": sum(row["accepted"] and row["predicted_class"] != row["true_label"] for row in known_rows),
                "unknown_rows": len(unknown_rows), "unknown_valid_rows": sum(row["valid_input"] for row in unknown_rows),
                "unknown_identity_groups": len(unknown_groups),
                "unknown_false_accepted_identity_groups": sum(any(row["accepted"] for row in group) for group in unknown_groups.values()),
            })
        else:
            summary["known_raw_top1"] = sum(row["valid_input"] and row["predicted_class"] == row["true_label"]
                                                  for row in predictions if row["scope"] == scope and row["role"] == "development" and row["true_label"] in KNOWN)
            summary["known_rows"] = sum(row["scope"] == scope and row["role"] == "development" and row["true_label"] in KNOWN for row in predictions)
            summary["known_valid_rows"] = sum(row["scope"] == scope and row["role"] == "development" and row["true_label"] in KNOWN and row["valid_input"] for row in predictions)
            summary["unknown_rows"] = sum(row["scope"] == scope and row["role"] == "unknown_development" for row in predictions)
            summary["unknown_valid_rows"] = summary["unknown_rows"]
            groups = {row["source_identity"] for row in predictions if row["scope"] == scope and row["role"] == "unknown_development"}
            summary["unknown_identity_groups"] = len(groups)
            summary["unknown_false_accepted_identity_groups"] = sum(any(row["accepted"] for row in predictions if row["scope"] == scope and row["role"] == "unknown_development" and row["source_identity"] == group) for group in groups)
        dev_summaries[scope] = summary
        add_check(checks, f"{scope}_development_tallies", report_metric_block(summary, report, report_scope, "development_int8"),
                  json.dumps(summary, sort_keys=True))

    add_check(checks, "prediction_acceptance_rule", all(
        row["accepted"] == (row["valid_input"] and row["predicted_class"] in KNOWN
                            and np.float32(row["score_max_known_float32"]) >= np.float32(row["threshold_float32"]))
        for row in predictions), "all 876 decisions recomputed from saved score, threshold, class and input validity")

    interpreter = tf.lite.Interpreter(model_path=str(model_path), num_threads=2,
        experimental_op_resolver_type=tf.lite.experimental.OpResolverType.BUILTIN_REF)
    interpreter.allocate_tensors()
    inp = interpreter.get_input_details()[0]
    out = interpreter.get_output_details()[0]
    tensors = interpreter.get_tensor_details()
    ops = [str(row["op_name"]) for row in interpreter._get_ops_details()]
    internal_float = sum(row["dtype"] in (np.float16, np.float32, np.float64) for row in tensors)
    full_int8_contract = (inp["dtype"] == np.int8 and out["dtype"] == np.int8
                          and inp["shape"].tolist() == [1, 96, 96, 1]
                          and out["shape"].tolist() == [1, 6] and internal_float == 0)
    audit = report["export"]["audit"]
    tensor_dtype_counts = dict(Counter(str(row["dtype"]) for row in tensors))
    add_check(checks, "tflite_builtin_ref_tensor_contract", full_int8_contract
              and audit["builtin_ref_available"] is True
              and audit["internal_float_tensor_count"] == internal_float
              and audit["operator_count"] == len(ops)
              and audit["ops"] == ops
              and audit["tensor_dtype_counts"] == tensor_dtype_counts
              and audit["input"]["quantization"] == list(inp["quantization"])
              and audit["output"]["quantization"] == list(out["quantization"]),
              f"input={inp['dtype']}{inp['shape'].tolist()} output={out['dtype']}{out['shape'].tolist()} ops={len(ops)} internal_float={internal_float}")

    float_source = report["source_lfw"]["development_float"]
    int8_source = report["source_lfw"]["development_int8"]
    float_camera = report["old_optical_camera"]["development_float"]
    int8_camera = report["old_optical_camera"]["development_int8"]
    outcome = {
        "source_lfw": {
            "float_raw_top1": f"{round(float_source['known_raw_top1_end_to_end'] * float_source['known_rows'])}/{float_source['known_rows']}",
            "int8_raw_top1": f"{dev_summaries['LFW']['known_raw_top1']}/{dev_summaries['LFW']['known_rows']}",
            "float_correct_accepted": f"{float_source['known_correct_accepted']}/{float_source['known_rows']}",
            "int8_correct_accepted": f"{int8_source['known_correct_accepted']}/{int8_source['known_rows']}",
            "float_unknown_false_groups": f"{float_source['unknown_false_accepted_identity_groups']}/{float_source['unknown_identity_groups']}",
            "int8_unknown_false_groups": f"{int8_source['unknown_false_accepted_identity_groups']}/{int8_source['unknown_identity_groups']}",
        },
        "old_optical_camera": {
            "float_known_raw_top1": f"{round(float_camera['known_raw_top1_end_to_end'] * float_camera['known_rows'])}/{float_camera['known_rows']}",
            "int8_known_raw_top1": f"{dev_summaries['old_optical_camera']['known_raw_top1']}/{dev_summaries['old_optical_camera']['known_rows']}",
            "float_correct_accepted": f"{float_camera['known_correct_accepted']}/{float_camera['known_rows']}",
            "int8_correct_accepted": f"{int8_camera['known_correct_accepted']}/{int8_camera['known_rows']}",
            "float_unknown_false_groups": f"{float_camera['unknown_false_accepted_identity_groups']}/{float_camera['unknown_identity_groups']}",
            "int8_unknown_false_groups": f"{int8_camera['unknown_false_accepted_identity_groups']}/{int8_camera['unknown_identity_groups']}",
        },
        "model": {"bytes": model_path.stat().st_size, "sha256": model_sha, "operators": len(ops), "internal_float_tensors": internal_float},
        "host_only": True,
    }
    add_check(checks, "ptq_not_qualified", report["export"]["device_qualified"] is False
              and "no Nicla operator/output/memory/latency test" in report["limits"],
              "host BUILTIN_REF only; no device claim")

    receipt = {
        "schema": "v8-alpha025-int8-primary-review-v1",
        "status": "passed" if all(check["passed"] for check in checks) else "failed",
        "checks_passed": sum(check["passed"] for check in checks),
        "checks_total": len(checks),
        "checks": checks,
        "outcome": outcome,
        "artifacts": {
            "config_sha256": sha(config_path), "preflight_sha256": sha(preflight_path),
            "parent_result_sha256": sha(result_path), "quantization_report_sha256": sha(REPORT_PATH),
            "model_sha256": model_sha, "private_predictions_sha256": sha(predictions_path),
            "float_predictions_sha256": sha(float_predictions_path),
        },
        "limitations": report["limits"],
    }
    if args.write_receipt:
        RECEIPT_PATH.parent.mkdir(parents=True, exist_ok=True)
        RECEIPT_PATH.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(receipt, indent=2, sort_keys=True))
    if receipt["status"] != "passed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
