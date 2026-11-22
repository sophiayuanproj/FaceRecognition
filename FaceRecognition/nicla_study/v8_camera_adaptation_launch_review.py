"""Primary prelaunch audit for the grouped v8 archived-camera adaptation pilot."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RESEARCH = ROOT / "work/project-research-v5"
RUN_ROOT = RESEARCH / "runs/accuracy_search_v1"
RUN_ID = "v8_camera_adaptation_from_v7_a010_s57101_launch01"
CONFIG_PATH = RUN_ROOT / "configs" / f"{RUN_ID}.json"
PREFLIGHT_PATH = RUN_ROOT / "preflight" / f"{RUN_ID}.json"
RUN_DIR = RUN_ROOT / RUN_ID
PLAN_PATH = RESEARCH / "config/MAC_PHASE_PLAN_V2.json"
STATE_PATH = RESEARCH / "RUN_STATE.json"
RECEIPT_PATH = RESEARCH / "reports/mac_verification_v1/V8_CAMERA_ADAPTATION_PRELAUNCH_PRIMARY_REVIEW.json"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> int:
    config, preflight = load(CONFIG_PATH), load(PREFLIGHT_PATH)
    runner_path = ROOT / "FaceRecognition/nicla_study/v7_accuracy_runner.py"
    parent_result_path = ROOT / config["parent_result_path"]
    parent_weights_path = ROOT / config["parent_model_weights_path"]
    parent_prediction_path = parent_result_path.parent / "private/PREDICTIONS.jsonl"
    camera_index_path = ROOT / preflight["private_input_index"]["path"]
    state, plan = load(STATE_PATH), load(PLAN_PATH)
    camera = preflight["camera_training_data"]
    split = preflight["grouped_split"]
    contract = preflight["fit_stopping_contract"]
    training = preflight["training_plan"]
    group = preflight["grouped_split"]
    phase = state["phases"]["v8_camera_adaptation_old_optical_host"]
    plan_phase = next(row for row in plan["phases"] if row["id"] == "v8_camera_adaptation_old_optical_host")
    parent_predictions = [json.loads(line) for line in parent_prediction_path.read_text().splitlines() if line]
    parent_prediction_models = {row["model_weight_sha256"] for row in parent_predictions}
    run_files = sorted(path.relative_to(RUN_DIR).as_posix() for path in RUN_DIR.rglob("*") if path.is_file())

    checks = []
    def check(name: str, passed: bool, detail: str) -> None:
        checks.append({"name": name, "passed": bool(passed), "detail": detail})

    runner_sha = sha(runner_path)
    config_sha = sha(CONFIG_PATH)
    preflight_sha = sha(PREFLIGHT_PATH)
    parent_result_sha = sha(parent_result_path)
    parent_weights_sha = sha(parent_weights_path)
    check("config_run_and_training_gate", config["run_id"] == RUN_ID
          and config["training_mode"] == "grouped_camera_adaptation_from_parent"
          and config["training_launch_enabled"] is True
          and config["expected_max_gradient_updates"] == 75,
          f"run_id={config['run_id']} max_updates={config['expected_max_gradient_updates']}")
    check("exact_config_preflight_binding", preflight["run_id"] == RUN_ID
          and preflight["config_sha256"] == config_sha
          and preflight["runner_sha256"] == runner_sha
          and preflight["training_launch_enabled"] is True,
          f"config_sha256={config_sha} preflight_sha256={preflight_sha} runner_sha256={runner_sha}")
    check("parent_result_and_weights_binding", parent_result_sha == config["parent_result_sha256"]
          and parent_weights_sha == config["parent_model_weights_sha256"]
          and parent_result_sha == preflight["parent_source_scout"]["result_sha256"]
          and parent_weights_sha == preflight["parent_source_scout"]["selected_model_weights_sha256"],
          f"parent_result_sha256={parent_result_sha} parent_weights_sha256={parent_weights_sha}")
    check("parent_loaded_weight_fingerprint", preflight["parent_source_scout"]["loaded_selected_weight_model_sha256"]
          == preflight["parent_source_scout"]["parent_prediction_model_weight_sha256"]
          and parent_prediction_models == {preflight["parent_source_scout"]["loaded_selected_weight_model_sha256"]},
          f"matching_parent_prediction_models={len(parent_prediction_models)}")
    check("train_roles_only", set(camera["selected_roles"]) == {"train", "other_train"}
          and set(contract["fit_rows_only_role_allowlist"]) == {"train", "other_train"}
          and training["camera_fraction_of_adaptation_gradient_presentations"] == 1.0
          and training["LFW_training_repeated"] is False,
          f"selected_roles={camera['selected_roles']} fit_allowlist={contract['fit_rows_only_role_allowlist']}")
    check("camera_training_pool_counts", camera["eligible_rows"] == 80 and camera["valid_rows"] == 58
          and preflight["private_input_index"]["rows"] == 80
          and sha(camera_index_path) == preflight["private_input_index"]["sha256"],
          f"eligible={camera['eligible_rows']} valid={camera['valid_rows']} index_sha256={sha(camera_index_path)}")
    check("grouped_split_size", split["fit_rows"] == 48 and split["stopping_rows"] == 10
          and config["expected_camera_fit_rows"] == 48 and config["expected_camera_stopping_rows"] == 10,
          f"fit={split['fit_rows']} stop={split['stopping_rows']}")
    check("identity_group_disjointness", split["fit_stopping_group_overlap"] == 0
          and contract["fit_stopping_group_overlap"] == 0
          and contract["other_identity_overlap"] == 0,
          f"group_overlap={split['fit_stopping_group_overlap']} other_identity_overlap={contract['other_identity_overlap']}")
    check("balanced_batch_budget", preflight["contract_checks"]["camera_batch_contract"]["camera_only"] is True
          and preflight["contract_checks"]["camera_batch_contract"]["batches"] == 3
          and preflight["contract_checks"]["camera_batch_contract"]["rows_per_batch"] == 24
          and training["gradient_updates_max"] == 75,
          f"3 batches/epoch × 25 maximum epochs = {training['gradient_updates_max']} updates")
    check("stopping_is_camera_only", training["stopping_role"].startswith("identity/source-group-disjoint")
          and training["threshold_calibration_used_for_fit_stopping_or_checkpoint_selection"] is False
          and training["development_used_for_fit_stopping_or_checkpoint_selection"] is False,
          training["stopping_role"])
    check("calibration_and_development_not_opened", training["threshold_or_development_roles_loaded"] is False
          and camera["calibration_or_development_frame_paths_opened"] is False
          and preflight["contract_checks"]["calibration_development_and_future_final_not_opened"] is True,
          "camera calibration/development frame roles not loaded for fit/stop/checkpoint selection")
    check("future_final_sealed", config["future_final_access"] is False
          and camera["future_final_frames_opened"] is False
          and contract["future_final_frame_folders_opened"] is False
          and training["future_final_roles_opened"] is False,
          "config, train-only preflight, and role contract all record no future-final access")
    check("batch_norm_frozen", config["batch_norm_frozen"] is True and training["batch_norm_frozen"] is True,
          "batch normalization remains frozen")
    check("host_phase_separated_from_nicla_phase", plan_phase["resources"] == ["heavy_training"]
          and plan_phase["scope"].startswith("host-only")
          and phase["resources"] == ["heavy_training"]
          and phase["requires_nicla"] is False
          and state["phases"]["v7_camera_adaptation"]["resources"] == ["heavy_training", "nicla"],
          "archived-frame training phase requires only the shared training lease; physical phase still requires Nicla")
    check("phase_ready_and_prerequisite", phase["status"] == "pending"
          and state["phases"]["v7_accuracy_method_expansion"]["status"] == "complete_verified"
          and state["leases"]["active"] == [],
          f"phase_status={phase['status']} active_leases={len(state['leases']['active'])}")
    check("no_gradient_artifacts_exist", not any(path in run_files for path in
          ("RUN_STATE.json", "CURVES.jsonl", "RESULT.json", "private/PREDICTIONS.jsonl"))
          and run_files == ["private/CAMERA_TRAIN_POOL.jsonl"],
          f"run_files={run_files}")

    receipt = {
        "schema": "v8-camera-adaptation-prelaunch-primary-review-v1",
        "status": "prelaunch_review_passed" if all(row["passed"] for row in checks) else "prelaunch_review_failed",
        "checks_passed": sum(row["passed"] for row in checks),
        "checks_total": len(checks),
        "training_started": False,
        "run_id": RUN_ID,
        "config_sha256": config_sha,
        "preflight_sha256": preflight_sha,
        "runner_sha256": runner_sha,
        "parent_result_sha256": parent_result_sha,
        "parent_weights_sha256": parent_weights_sha,
        "camera_train_index_sha256": sha(camera_index_path),
        "checks": checks,
        "limitations": ["one exploratory seed", "historical taped-screen camera captures only",
                        "no current-scene camera data or Nicla qualification", "not a ten-seed comparison"],
    }
    if RECEIPT_PATH.exists():
        raise FileExistsError(f"refusing to replace prelaunch review: {RECEIPT_PATH}")
    RECEIPT_PATH.parent.mkdir(parents=True, exist_ok=True)
    RECEIPT_PATH.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(receipt, indent=2, sort_keys=True))
    return 0 if receipt["status"] == "prelaunch_review_passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
