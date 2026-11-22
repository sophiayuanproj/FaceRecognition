"""Run the preflight-bound v10 MobileNetV1 width comparison serially.

The default is a read-only validation/dry run. --run is an explicit coordinator
gate. Each child runner acquires/releases the shared heavy_training lease; this
orchestrator never starts more than one child at once and never writes RUN_STATE.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[2]
RESEARCH = ROOT / "work/project-research-v5"
RUN_ROOT = RESEARCH / "runs/accuracy_search_v1"
INDEX = RUN_ROOT / "v10_MNV1_WIDTH10_LAUNCH_INDEX.json"
CAMPAIGN_DIR = RUN_ROOT / "v10_MNV1_WIDTH10_CAMPAIGN"
CAMPAIGN_STATE = CAMPAIGN_DIR / "CAMPAIGN_STATE.json"
PHASE_ID = "v10_mnv1_width10_source_search"
RUNNER = ROOT / "FaceRecognition/nicla_study/v7_accuracy_runner.py"
TRAINING_PYTHON = ROOT / "work/model-training/.venv/bin/python"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def ordered_records() -> tuple[dict, list[dict]]:
    index = load(INDEX)
    preflights = {row["run_id"]: row for row in index.get("preflights", [])}
    configs = index.get("configs", [])
    if index.get("schema") != "v10-mnv1-width10-launch-config-index-v1":
        raise ValueError("unexpected v10 launch-index schema")
    if len(configs) != 18 or len(preflights) != 18 or index.get("validation", {}).get("all_role_budget_hash_and_no_training_artifact_checks_passed") is not True:
        raise ValueError("launch index must bind all 18 configs and 18 passing preflights")
    if sha(RUNNER) != index["runner_sha256"]:
        raise ValueError("runner hash changed after preflight")
    if not TRAINING_PYTHON.is_file() or Path(sys.executable).resolve() != TRAINING_PYTHON.resolve():
        raise RuntimeError("run this campaign with work/model-training/.venv/bin/python (TF 2.10.1)")
    records = []
    for cfg_row in configs:
        run_id = cfg_row["run_id"]
        pre_row = preflights.get(run_id)
        if pre_row is None:
            raise ValueError("missing exact preflight for " + run_id)
        cfg_path = ROOT / cfg_row["config_path"]
        pre_path = ROOT / pre_row["preflight_path"]
        cfg, pre = load(cfg_path), load(pre_path)
        if sha(cfg_path) != cfg_row["config_sha256"] or sha(pre_path) != pre_row["preflight_sha256"]:
            raise ValueError("config/preflight hash mismatch for " + run_id)
        if cfg.get("run_id") != run_id or cfg.get("training_launch_enabled") is not True:
            raise ValueError("launch gate or run ID mismatch for " + run_id)
        if cfg.get("future_final_access") is not False:
            raise ValueError("future-final access must remain false for " + run_id)
        if (pre.get("run_id") != run_id or pre.get("config_sha256") != cfg_row["config_sha256"]
                or pre_row.get("runner_sha256") != index["runner_sha256"]):
            raise ValueError("preflight lineage mismatch for " + run_id)
        if pre_row.get("training_started") is not False or int(pre_row.get("gradient_updates_completed", -1)) != 0:
            raise ValueError("preflight indicates training already started for " + run_id)
        controls, split = pre.get("training_controls", {}), pre.get("split", {})
        if (int(controls.get("fit_rows", -1)) != 1066 or int(controls.get("stopping_rows", -1)) != 107
                or int(split.get("fit_stop_source_group_overlap", -1)) != 0):
            raise ValueError("unexpected grouped split for " + run_id)
        if controls.get("future_final_files_opened") is not False or pre_row.get("future_final_files_opened") is not False:
            raise ValueError("future-final files must remain unopened for " + run_id)
        if controls.get("threshold_calibration_or_development_used_for_epoch_selection") is not False:
            raise ValueError("calibration/development cannot select epochs for " + run_id)
        if sha(ROOT / cfg_row["initialization_checkpoint_path"]) != cfg_row["initialization_checkpoint_sha256"]:
            raise ValueError("initialization checkpoint hash changed for " + run_id)
        parent_result = RUN_ROOT / cfg_row["parent_run_id"] / "RESULT.json"
        if (cfg.get("matched_parent_result_sha256") != cfg_row["parent_result_sha256"]
                or sha(parent_result) != cfg_row["parent_result_sha256"]):
            raise ValueError("parent result hash changed for " + run_id)
        records.append({"run_id": run_id, "seed": int(cfg_row["seed"]), "alpha": float(cfg_row["alpha"]),
                        "config_path": cfg_row["config_path"], "config_sha256": cfg_row["config_sha256"],
                        "preflight_path": pre_row["preflight_path"], "preflight_sha256": pre_row["preflight_sha256"],
                        "result_path": f"work/project-research-v5/runs/accuracy_search_v1/{run_id}/RESULT.json"})
    records.sort(key=lambda row: (row["seed"], row["alpha"]))
    pairs = {}
    for row in records:
        pairs.setdefault(row["seed"], set()).add(row["alpha"])
    if sorted(pairs) != list(range(57102, 57111)) or any(pairs[s] != {0.1, 0.25} for s in pairs):
        raise ValueError("expected nine paired seeds 57102..57110 at widths 0.10 and 0.25")
    return index, records


def check_coordinator_state(*, require_started: bool) -> dict:
    state = load(RESEARCH / "RUN_STATE.json")
    if state.get("leases", {}).get("active") != []:
        raise RuntimeError("shared resource lease is occupied")
    phase = state.get("phases", {}).get(PHASE_ID)
    expected = "in_progress" if require_started else {"pending", "ready", "in_progress"}
    if phase is None or (phase.get("status") != expected if isinstance(expected, str) else phase.get("status") not in expected):
        raise RuntimeError(f"coordinator phase must be {expected}; found {None if phase is None else phase.get('status')}")
    if state.get("phases", {}).get("v7_accuracy_method_expansion", {}).get("status") != "complete_verified":
        raise RuntimeError("required v7 method-expansion audit is not complete")
    return state


def existing_result(record: dict) -> dict | None:
    result_path = ROOT / record["result_path"]
    if not result_path.exists():
        return None
    result = load(result_path)
    if result.get("run_id") != record["run_id"] or not str(result.get("status", "")).startswith("complete"):
        raise RuntimeError("partial or mismatched output already exists; inspect it and resume explicitly: " + record["run_id"])
    if result.get("config_sha256") != record["config_sha256"]:
        raise RuntimeError("existing completed result has a different config: " + record["run_id"])
    return {"status": result["status"], "result_sha256": sha(result_path), "skipped": True}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--dry-run", action="store_true", help="validate all bindings and print the serial paired order")
    group.add_argument("--run", action="store_true", help="launch one config at a time after the coordinator phase is started")
    args = parser.parse_args()
    index, records = ordered_records()
    state = check_coordinator_state(require_started=args.run)
    print(json.dumps({"status": "preflight_validated" if args.dry_run else "campaign_starting",
                      "index_sha256": sha(INDEX), "runner_sha256": index["runner_sha256"],
                      "phase_status": state["phases"][PHASE_ID]["status"], "paired_order": records}, indent=2), flush=True)
    if args.dry_run:
        return 0

    inhibitor = shutil.which("caffeinate") if sys.platform == "darwin" else None
    campaign = load(CAMPAIGN_STATE) if CAMPAIGN_STATE.exists() else {
        "schema": "v10-mnv1-width10-campaign-state-v1", "index_sha256": sha(INDEX),
        "runner_sha256": index["runner_sha256"], "status": "running", "runs": {}, "events": []}
    if campaign.get("index_sha256") != sha(INDEX) or campaign.get("runner_sha256") != index["runner_sha256"]:
        raise RuntimeError("campaign state is bound to different input hashes")
    campaign["status"] = "running"
    campaign["events"].append({"event": "campaign_resumed_or_started", "utc": utc_now(), "pid": os.getpid()})
    atomic_json(CAMPAIGN_STATE, campaign)

    for position, record in enumerate(records, 1):
        already = existing_result(record)
        if already:
            campaign["runs"][record["run_id"]] = already
            continue
        check_coordinator_state(require_started=True)
        run_dir = ROOT / "work/project-research-v5/runs/accuracy_search_v1" / record["run_id"]
        if run_dir.exists() and not (run_dir / "RUN_STATE.json").exists():
            raise RuntimeError("run directory has partial artifacts without a resumable state: " + record["run_id"])
        resume = (run_dir / "RUN_STATE.json").exists()
        command = [sys.executable, str(RUNNER), "--config", str(ROOT / record["config_path"])]
        if resume:
            command.append("--resume")
        if inhibitor:
            command = [inhibitor, "-i"] + command
        log_path = CAMPAIGN_DIR / "logs" / (record["run_id"] + ".stdout.txt")
        log_path.parent.mkdir(parents=True, exist_ok=True)
        started = utc_now()
        campaign["runs"][record["run_id"]] = {"status": "running", "started_utc": started,
                                                "config_sha256": record["config_sha256"],
                                                "preflight_sha256": record["preflight_sha256"],
                                                "log_path": str(log_path.relative_to(ROOT)), "pid": None}
        campaign["events"].append({"event": "run_started", "run_id": record["run_id"], "utc": started,
                                   "position": position, "total": len(records), "resume": resume,
                                   "idle_sleep_inhibitor": bool(inhibitor)})
        atomic_json(CAMPAIGN_STATE, campaign)
        print(f"\n[{position}/{len(records)}] START {record['run_id']} resume={resume} caffeinate={bool(inhibitor)}", flush=True)
        try:
            with log_path.open("a", encoding="utf-8", newline="\n") as log:
                log.write(f"\n=== launch {started}; command={json.dumps(command)} ===\n")
                log.flush()
                proc = subprocess.Popen(command, cwd=ROOT, env={**os.environ, "OMP_NUM_THREADS": "2",
                                          "TF_NUM_INTRAOP_THREADS": "2", "TF_NUM_INTEROP_THREADS": "1"},
                                        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
                campaign["runs"][record["run_id"]]["pid"] = proc.pid
                atomic_json(CAMPAIGN_STATE, campaign)
                assert proc.stdout is not None
                for line in proc.stdout:
                    log.write(line)
                    log.flush()
                    sys.stdout.write(line)
                    sys.stdout.flush()
                code = proc.wait()
        except KeyboardInterrupt:
            try:
                proc.terminate()
                proc.wait(timeout=20)
            except Exception:
                proc.kill()
            campaign["runs"][record["run_id"]].update({"status": "interrupted", "finished_utc": utc_now(),
                                                        "exit_code": None})
            campaign["status"] = "interrupted"
            campaign["events"].append({"event": "run_interrupted", "run_id": record["run_id"], "utc": utc_now()})
            atomic_json(CAMPAIGN_STATE, campaign)
            raise
        finished = utc_now()
        state_after = load(RESEARCH / "RUN_STATE.json")
        if state_after.get("leases", {}).get("active") != []:
            code = code or 2
            detail = "shared training lease remained active after runner exit"
        elif code != 0:
            detail = f"runner exited {code}; inspect checkpoint/log before resuming"
        else:
            result_path = ROOT / record["result_path"]
            result = load(result_path)
            if result.get("run_id") != record["run_id"] or result.get("config_sha256") != record["config_sha256"] or not str(result.get("status", "")).startswith("complete"):
                code, detail = 2, "runner result did not pass identity/status/config binding"
            else:
                detail = "result saved; independent prediction/threshold review remains required"
        if code != 0:
            campaign["runs"][record["run_id"]].update({"status": "failed_needs_review", "finished_utc": finished,
                                                        "exit_code": code, "detail": detail})
            campaign["status"] = "stopped_on_failure"
            campaign["events"].append({"event": "run_failed", "run_id": record["run_id"], "utc": finished, "detail": detail})
            atomic_json(CAMPAIGN_STATE, campaign)
            print(json.dumps({"status": campaign["status"], "run_id": record["run_id"], "detail": detail}), flush=True)
            return code
        result_path = ROOT / record["result_path"]
        campaign["runs"][record["run_id"]].update({"status": "complete_needs_primary_review", "finished_utc": finished,
                                                    "exit_code": code, "result_sha256": sha(result_path), "detail": detail})
        campaign["events"].append({"event": "run_completed", "run_id": record["run_id"], "utc": finished,
                                   "result_sha256": sha(result_path)})
        atomic_json(CAMPAIGN_STATE, campaign)
        print(f"[{position}/{len(records)}] COMPLETE {record['run_id']} ({detail})", flush=True)

    campaign["status"] = "complete_needs_independent_and_primary_review"
    campaign["finished_utc"] = utc_now()
    campaign["events"].append({"event": "campaign_complete", "utc": campaign["finished_utc"]})
    atomic_json(CAMPAIGN_STATE, campaign)
    print(json.dumps({"status": campaign["status"], "state_path": str(CAMPAIGN_STATE.relative_to(ROOT)),
                      "runs": len(campaign["runs"])}, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
