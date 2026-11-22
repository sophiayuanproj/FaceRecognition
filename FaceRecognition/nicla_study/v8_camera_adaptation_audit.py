"""Primary independent-style audit of a completed archived-camera adaptation run."""
from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
RESEARCH = ROOT / "work/project-research-v5"
RUN_ROOT = RESEARCH / "runs/accuracy_search_v1"
RUN_ID = "v8_camera_adaptation_from_v7_a010_s57101_launch01"
RUN = RUN_ROOT / RUN_ID
CONFIG = RUN_ROOT / "configs" / f"{RUN_ID}.json"
PREFLIGHT = RUN_ROOT / "preflight" / f"{RUN_ID}.json"
RESULT = RUN / "RESULT.json"
RUN_STATE = RUN / "RUN_STATE.json"
CURVES = RUN / "CURVES.jsonl"
INITIAL = RUN / "INITIAL_CAMERA_TRAINING_DIAGNOSTICS.json"
PREDICTIONS = RUN / "private/PREDICTIONS.jsonl"
EVAL_INPUTS = RUN / "private/CAMERA_EVALUATION_INPUTS.jsonl"
TRAIN_INDEX = RUN / "private/CAMERA_TRAIN_POOL.jsonl"
PARENT = RUN_ROOT / "v7_lfw_source_mnv1_a010_ei_scout57101"
V8_FLOAT = RUN_ROOT / "v8_lfw_source_mnv1_a025_ei_scout57101_launch01"
RECEIPT = RESEARCH / "reports/mac_verification_v1/V8_CAMERA_ADAPTATION_PRIMARY_REVIEW.json"
CLASSES = ["K01", "K02", "K03", "K04", "K05", "OTHER"]
KNOWN = set(CLASSES[:5])
CAL_ROLES = {"threshold_calibration", "unknown_calibration"}
DEV_ROLES = {"development", "unknown_development"}
ALLOWED_EVAL_ROLES = CAL_ROLES | DEV_ROLES


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def json_file(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def pct_close(actual: float, expected_count: int, denominator: int) -> bool:
    return np.float32(actual).tobytes() == np.float32(expected_count / denominator).tobytes()


def main() -> int:
    config, preflight = json_file(CONFIG), json_file(PREFLIGHT)
    result, local_state = json_file(RESULT), json_file(RUN_STATE)
    initial, global_state = json_file(INITIAL), json_file(RESEARCH / "RUN_STATE.json")
    curves, predictions, inputs = jsonl(CURVES), jsonl(PREDICTIONS), jsonl(EVAL_INPUTS)
    parent_result = json_file(PARENT / "RESULT.json")
    v8_result = json_file(V8_FLOAT / "RESULT.json")
    v8_ptq = json_file(V8_FLOAT / "INT8_QUANTIZATION_EVAL.json")
    checks: list[dict] = []

    def check(name: str, condition: bool, detail: str) -> None:
        checks.append({"name": name, "passed": bool(condition), "detail": detail})

    config_sha, preflight_sha, runner_sha = sha(CONFIG), sha(PREFLIGHT), sha(ROOT / "FaceRecognition/nicla_study/v7_accuracy_runner.py")
    parent_result_sha = sha(PARENT / "RESULT.json")
    parent_weights_sha = sha(PARENT / "best_model.weights.h5")
    selected_weights = RUN / "selected_camera_model.weights.h5"
    float_export = RUN / "model_float.h5"
    eval_receipt = result["old_optical_camera"]["evaluation_inputs_receipt"]
    threshold = result["old_optical_camera"]["threshold"]
    development = result["old_optical_camera"]["development"]
    initial_parent = initial["parent_selected_model_weight_sha256"]
    curve_updates = sum(int(row["gradient_updates"]) for row in curves)
    curve_epochs = len(curves)

    check("run_status_and_id", result["run_id"] == RUN_ID and result["status"] == "complete_camera_adaptation_host_only",
          f"run_id={result['run_id']} status={result['status']}")
    check("config_preflight_runner_bindings", result["config_sha256"] == config_sha
          and result["preflight_sha256"] == preflight_sha == result["preflight_sha256"]
          and result["runner_sha256"] == runner_sha
          and config["runner_sha256"] == runner_sha
          and preflight["runner_sha256"] == runner_sha
          and preflight["config_sha256"] == config_sha,
          f"config={config_sha} preflight={preflight_sha} runner={runner_sha}")
    check("parent_result_and_weights", result["parent_source_scout"]["result_sha256"] == parent_result_sha
          == config["parent_result_sha256"]
          and result["parent_source_scout"]["selected_model_weights_sha256"] == parent_weights_sha
          == config["parent_model_weights_sha256"]
          and parent_result["run_id"] == config["parent_run_id"],
          f"parent_result={parent_result_sha} parent_weights={parent_weights_sha}")
    parent_prediction_models = {row["model_weight_sha256"] for row in jsonl(PARENT / "private/PREDICTIONS.jsonl")}
    check("parent_loaded_model_fingerprint", initial_parent == result["parent_source_scout"]["selected_model_weight_sha256_after_load"]
          == preflight["parent_source_scout"]["loaded_selected_weight_model_sha256"]
          and parent_prediction_models == {initial_parent}, f"parent_weight_model_digest={initial_parent}")
    check("fit_and_stop_roles_groups", result["grouped_split"]["fit_rows"] == 48
          and result["grouped_split"]["stopping_rows"] == 10
          and result["grouped_split"]["fit_stopping_group_overlap"] == 0
          and result["grouped_split"]["fit_other_identity_groups"] == 13
          and result["grouped_split"]["stopping_other_identity_groups"] == 3
          and preflight["fit_stopping_contract"]["other_identity_overlap"] == 0
          and result["training"]["fit_rows_only_train_other_train"] is True
          and result["training"]["calibration_development_used_for_stopping"] is False,
          json.dumps(result["grouped_split"], sort_keys=True))
    check("train_pool_and_initial_diagnostics", result["camera_training_data"]["training_inputs_sha256"]
          == preflight["camera_training_data"]["training_inputs_sha256"]
          and sha(TRAIN_INDEX) == preflight["private_input_index"]["sha256"]
          and initial["fit_rows"] == 48 and initial["stopping_rows"] == 10
          and initial["fit_stopping_group_overlap"] == 0
          and initial["calibration_development_loaded"] is False
          and initial["future_final_files_opened"] is False,
          f"train_index_sha256={sha(TRAIN_INDEX)}")
    check("full_gradient_budget_and_curve", result["training"]["completed_gradient_epochs"] == 25
          and result["training"]["completed_gradient_updates"] == 75
          and result["training"]["maximum_gradient_updates"] == 75
          and curve_epochs == 25 and curve_updates == 75
          and all(row["camera_only_gradient_source"] is True and row["gradient_updates"] == 3 for row in curves),
          f"epochs={curve_epochs} updates={curve_updates} phases={[row['phase'] for row in curves[:1]+curves[-1:]]}")
    check("local_state_freeze_before_evaluation", local_state["status"] == "complete"
          and local_state["global_epoch"] == 25 and local_state["gradient_updates_completed"] == 75
          and local_state["model_selection_frozen"] is True
          and local_state["postfreeze_evaluation_started"] is True
          and local_state["result_sha256"] == sha(RESULT),
          f"state={local_state['status']} frozen={local_state['model_selection_frozen']} eval={local_state['postfreeze_evaluation_started']}")
    roles = {row["role"] for row in inputs}
    check("future_final_and_role_boundary", config["future_final_access"] is False
          and eval_receipt["future_final_frames_opened"] is False
          and result["training"]["postfreeze_calibration_development_loaded"] is True
          and roles == ALLOWED_EVAL_ROLES
          and all(row["role"] in ALLOWED_EVAL_ROLES for row in predictions)
          and not any("final" in row["role"] for row in inputs + predictions),
          f"input_roles={sorted(roles)} prediction_roles={sorted({row['role'] for row in predictions})}")
    input_by_id = {row["capture_id"]: row for row in inputs}
    predictions_match_inputs = (len(input_by_id) == len(inputs) == 135 and len(predictions) == 135
        and set(input_by_id) == {row["record_id"] for row in predictions}
        and all(input_by_id[row["record_id"]]["role"] == row["role"]
                and input_by_id[row["record_id"]]["label"] == row["true_label"]
                and input_by_id[row["record_id"]]["valid_input"] == row["valid_input"]
                and input_by_id[row["record_id"]]["source_identity"] == row["source_identity"]
                for row in predictions))
    check("evaluation_rows_bind_to_saved_inputs", predictions_match_inputs,
          f"evaluation_input_rows={len(inputs)} prediction_rows={len(predictions)}")

    cal_known_all = [row for row in predictions if row["role"] == "threshold_calibration"]
    cal_unknown_all = [row for row in predictions if row["role"] == "unknown_calibration"]
    cal_known = [row for row in cal_known_all if row["valid_input"]]
    cal_unknown = [row for row in cal_unknown_all if row["valid_input"]]
    group_max: dict[str, float] = {}
    for row in cal_unknown:
        identity = str(row["source_identity"])
        group_max[identity] = max(group_max.get(identity, -np.inf), float(row["score_max_known_float32"]))
    max_unknown = np.float32(max(group_max.values()))
    next_threshold = np.nextafter(max_unknown, np.float32(np.inf), dtype=np.float32)
    all_cal_thresholds = {np.float32(row["threshold_float32"]).tobytes() for row in cal_known + cal_unknown}
    cal_correct = sum(row["accepted"] and row["predicted_class"] == row["true_label"] for row in cal_known)
    cal_false_groups = sum(any(row["accepted"] for row in cal_unknown if row["source_identity"] == identity)
                           for identity in group_max)
    check("calibration_only_threshold_nextafter", len(cal_known_all) == 25 and len(cal_unknown_all) == 30
          and len(cal_known) == 22 and len(cal_unknown) == 24
          and len(group_max) == 24 and np.float32(threshold["threshold_float32"]).tobytes() == next_threshold.tobytes()
          and np.float32(threshold["calibration_max_unknown_score"]).tobytes() == max_unknown.tobytes()
          and threshold["threshold_candidates_from_calibration_only"] is True
          and all_cal_thresholds == {next_threshold.tobytes()}
          and cal_correct == 0 and cal_false_groups == 0,
          f"valid known calibration={len(cal_known)} unknown rows={len(cal_unknown)} groups={len(group_max)} max={float(max_unknown)} threshold={float(next_threshold)}")

    decisions_valid = all(row["accepted"] == (
        row["valid_input"] and row["predicted_class"] in KNOWN
        and np.float32(row["score_max_known_float32"]) >= np.float32(row["threshold_float32"]))
        for row in predictions)
    check("all_saved_decisions_recomputed", decisions_valid, "all 135 accept/reject flags recomputed from class, valid crop, score, and calibration threshold")

    dev_known = [row for row in predictions if row["role"] == "development" and row["true_label"] in KNOWN]
    dev_unknown = [row for row in predictions if row["role"] == "unknown_development"]
    valid_known = [row for row in dev_known if row["valid_input"]]
    valid_unknown = [row for row in dev_unknown if row["valid_input"]]
    raw_top1 = sum(row["predicted_class"] == row["true_label"] for row in valid_known)
    known_accepted = [row for row in dev_known if row["accepted"]]
    correct_accepted = sum(row["predicted_class"] == row["true_label"] for row in known_accepted)
    wrong_accepted = len(known_accepted) - correct_accepted
    unknown_groups: dict[str, list[dict]] = defaultdict(list)
    for row in dev_unknown:
        unknown_groups[str(row["source_identity"])].append(row)
    false_groups = sum(any(row["accepted"] for row in group) for group in unknown_groups.values())
    tallies_match = (
        len(dev_known) == development["known_rows"] == 50
        and len(valid_known) == development["known_detection_valid_rows"] == 35
        and raw_top1 == round(development["known_raw_top1_end_to_end"] * development["known_rows"]) == 5
        and correct_accepted == development["known_correct_accepted"] == 1
        and wrong_accepted == development["known_wrong_accepted"] == 2
        and len(known_accepted) == round(development["known_coverage_end_to_end"] * development["known_rows"]) == 3
        and len(dev_unknown) == development["unknown_rows"] == 30
        and len(valid_unknown) == development["unknown_detection_valid_rows"] == 20
        and len(unknown_groups) == development["unknown_identity_groups"] == 30
        and false_groups == development["unknown_false_accepted_identity_groups"] == 0
    )
    check("development_denominators_and_tallies", tallies_match,
          f"known raw={raw_top1}/50 accepted_correct={correct_accepted}/50 accepted_wrong={wrong_accepted}/50; unknown={false_groups}/30 groups false, valid rows={len(valid_unknown)}/30")
    people = {}
    for label in sorted(KNOWN):
        rows = [row for row in dev_known if row["true_label"] == label]
        valid = [row for row in rows if row["valid_input"]]
        accepted = [row for row in rows if row["accepted"]]
        people[label] = {
            "rows": len(rows), "valid_detection_rows": len(valid),
            "raw_top1_correct": sum(row["predicted_class"] == label for row in valid),
            "correct_accepted": sum(row["predicted_class"] == label for row in accepted),
            "wrong_known_accepted": sum(row["predicted_class"] != label for row in accepted),
        }
    saved_people = development["per_person"]
    person_match = all(people[label][key] == saved_people[label][key]
                       for label in people for key in people[label])
    check("per_person_confusion_counts", person_match, json.dumps(people, sort_keys=True))

    selected_weight_sha = sha(selected_weights)
    float_export_sha = sha(float_export)
    weight_models = {row["model_weight_sha256"] for row in predictions}
    check("selected_checkpoint_and_export_hashes", selected_weight_sha == result["training"]["selected_weights_sha256"]
          == local_state["selected_weights_file_sha256"]
          and result["training"]["selected_model_weight_sha256"] == local_state["selected_model_weight_sha256"]
          and weight_models == {result["training"]["selected_model_weight_sha256"]}
          and float_export_sha == result["exports"]["float_model_sha256"],
          f"selected_weights_file_sha256={selected_weight_sha} model_digest={result['training']['selected_model_weight_sha256']} float_export_sha256={float_export_sha}")

    parent_camera = parent_result["old_optical_camera"]["development"]
    v8_camera = v8_result["old_optical_camera"]["development"]
    comparison = {
        "v7_alpha010_parent": {"raw_top1_valid": "0/35", "correct_accepted": "0/50", "wrong_accepted": "0/50", "unknown_false_groups": "0/30"},
        "v8_alpha025_source_float": {"raw_top1_valid": "6/35", "correct_accepted": "0/50", "wrong_accepted": "0/50", "unknown_false_groups": "0/30"},
        "v8_alpha010_camera_adapted": {"raw_top1_valid": f"{raw_top1}/35", "correct_accepted": f"{correct_accepted}/50", "wrong_accepted": f"{wrong_accepted}/50", "unknown_false_groups": f"{false_groups}/30"},
        "alpha025_ptq_float_reference": {"raw_top1_valid": f"{round(v8_ptq['old_optical_camera']['development_float']['known_conditional_raw_top1']*35)}/35", "correct_accepted": f"{v8_ptq['old_optical_camera']['development_float']['known_correct_accepted']}/50", "wrong_accepted": f"{v8_ptq['old_optical_camera']['development_float']['known_wrong_accepted']}/50", "unknown_false_groups": f"{v8_ptq['old_optical_camera']['development_float']['unknown_false_accepted_identity_groups']}/30"},
        "interpretation": "Adaptation produced one correct accept but also two wrong known accepts; raw top-1 is below the alpha0.25 float source model. This one-seed historical-camera pilot is not a useful candidate.",
    }
    baseline_integrity = (parent_camera["known_rows"] == 50 and parent_camera["known_detection_valid_rows"] == 35
        and parent_camera["known_correct_accepted"] == 0 and parent_camera["known_wrong_accepted"] == 0
        and v8_camera["known_rows"] == 50 and v8_camera["known_detection_valid_rows"] == 35
        and v8_camera["known_correct_accepted"] == 0 and v8_camera["known_wrong_accepted"] == 0)
    check("matched_historical_baseline_denominators", baseline_integrity,
          "v7 alpha0.1 and v8 alpha0.25 are compared on the same 50 known / 35 valid crop / 30 unknown group development denominators")
    check("no_device_or_multiseed_claim", any(limit.startswith("one exploratory camera adaptation seed")
          for limit in result["limits"])
          and any("no current-scene recapture, int8 conversion, or Nicla qualification" in limit
                  for limit in result["limits"])
          and result["training"]["fit_source_kind_counts"] == {"camera_replay_train": 48},
          "single historical-capture host float pilot only; no deployment, energy, latency, or ten-seed claim")
    check("global_run_lease_released", global_state["leases"]["active"] == []
          and global_state["phases"]["v8_camera_adaptation_old_optical_host"]["status"] == "in_progress",
          f"active_leases={len(global_state['leases']['active'])} phase={global_state['phases']['v8_camera_adaptation_old_optical_host']['status']}")

    acceptance = [{"name": row["name"], "status": "pass" if row["passed"] else "fail", "detail": row["detail"]}
                  for row in checks]
    receipt = {
        "schema": "v8-camera-adaptation-primary-review-v1",
        "status": "complete_verified" if all(row["passed"] for row in checks) else "failed_needs_repair",
        "checks_passed": sum(row["passed"] for row in checks), "checks_total": len(checks),
        "acceptance_checks": acceptance,
        "run_id": RUN_ID, "training_started": True,
        "scientific_outcome": "no_success_candidate",
        "comparison": comparison,
        "per_person_development": people,
        "artifacts": {
            "config_sha256": config_sha, "preflight_sha256": preflight_sha,
            "runner_sha256": runner_sha, "parent_result_sha256": parent_result_sha,
            "parent_weights_sha256": parent_weights_sha,
            "result_sha256": sha(RESULT), "run_state_sha256": sha(RUN_STATE),
            "curves_sha256": sha(CURVES), "selected_weights_file_sha256": selected_weight_sha,
            "float_export_sha256": float_export_sha,
            "predictions_sha256": sha(PREDICTIONS), "evaluation_inputs_sha256": sha(EVAL_INPUTS),
        },
        "metrics": {
            "gradient_epochs": curve_epochs, "gradient_updates": curve_updates,
            "fit_rows": 48, "stopping_rows": 10,
            "camera_known_raw_top1": f"{raw_top1}/50 end-to-end; {raw_top1}/35 conditional",
            "camera_known_correct_accepted": f"{correct_accepted}/50",
            "camera_known_wrong_accepted": f"{wrong_accepted}/50",
            "camera_unknown_false_groups": f"{false_groups}/30",
            "camera_threshold_float32": threshold["threshold_float32"],
        },
        "limitations": result["limits"],
    }
    if RECEIPT.exists():
        raise FileExistsError(f"refusing to replace immutable result audit: {RECEIPT}")
    RECEIPT.parent.mkdir(parents=True, exist_ok=True)
    RECEIPT.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(receipt, indent=2, sort_keys=True))
    return 0 if receipt["status"] == "complete_verified" else 1


if __name__ == "__main__":
    raise SystemExit(main())
