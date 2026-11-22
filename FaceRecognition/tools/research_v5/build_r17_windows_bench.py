"""Build the offline Windows video and power bench for a verified R17 or R18 release."""

import argparse
import json
from pathlib import Path
import re
import shutil
import sys
import tempfile


ROOT = Path(__file__).resolve().parents[3]
APP = ROOT / "FaceRecognition"
STUDY = ROOT / "work/project-research-v5"
SOURCE = STUDY / "private/live-pipeline-20261110"
STAGED = Path("work/project-research-v5/bench/live-pipeline-20261110")
POWER = STAGED / "conditions/power"
INSTALL = ROOT / "work/project-finish-v4/deployments/backups/20261111T194959Z/deployment-manifest.json"
sys.path.insert(0, str(APP / "tools"))
sys.path.insert(0, str(SOURCE / "conditions/power"))
from portable_transfer import create, sha
from prepare_windows_bench import RELEASE_ROLLBACKS, release_inputs


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8", newline="\n")


def copy_file(source, target):
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, target)


def copy_code(stage):
    paths = ("control_center", "host", "nicla_study", "tools/windows_v5", "firmware", "models/deployment-v1")
    for path in paths:
        shutil.copytree(APP / path, stage / "FaceRecognition" / path,
                        ignore=shutil.ignore_patterns("__pycache__", "*.pyc", ".DS_Store"))
    files = ("requirements-instrument-windows.lock", "requirements-live-control-center-windows.lock",
             "tools/mpremote_path.py", "tools/deploy_nicla_runtime.py", "tools/portable_transfer.py",
             "tools/research_v5/optical_display.py")
    for path in files:
        copy_file(APP / path, stage / "FaceRecognition" / path)


def stage_release(stage, install, staged_release=None):
    receipt = json.loads(install.read_text(encoding="utf-8-sig"))
    payload, firmware_hash = release_inputs(receipt, staged_release)
    hashes = receipt["post_promotion_hashes"]
    for name, data in payload.items():
        target = stage / "bench/release" / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)
    model_hashes = {}
    for name, data in receipt["reused_model_files"].items():
        source = APP / "models/deployment-v1" / name
        if sha(source) != data["sha256"] or source.stat().st_size != data["bytes"]:
            raise ValueError(f"Shared release asset hash mismatch: {name}")
        model_hashes[name] = data["sha256"]
    rollback = receipt.get("rollback_release_id") or RELEASE_ROLLBACKS.get(receipt["release_id"])
    if not isinstance(rollback, str) or not re.fullmatch(r"[A-Za-z0-9_-]{1,48}", rollback):
        raise ValueError("Deployment receipt does not identify a valid rollback release")
    copy_file(install, stage / "bench/INSTALL_RECEIPT.json")
    write_json(stage / "bench/RELEASE_HASHES.json", {"release_id": receipt["release_id"],
        "firmware_sha256": firmware_hash, "config_sha256": hashes["runtime_config.json"],
        "release_files": hashes, "model_asset_root": receipt["model_asset_root"],
        "model_asset_files": model_hashes, "rollback_release_id": rollback})


def stage_startup_repair(stage, repair_path):
    repair = json.loads(repair_path.read_text(encoding="utf-8"))
    files = (
        (APP / "firmware/boot.py", "main.py", "startup_sha256"),
        (APP / "firmware/battery_usb_boot.py", "boot.py", "usb_hook_sha256"),
    )
    for source, board_name, receipt_key in files:
        if sha(source) != repair[receipt_key]:
            raise ValueError(f"Current startup file differs from the verified board readback: {board_name}")
        copy_file(source, stage / "bench/startup" / board_name)
    copy_file(repair_path, stage / "bench/STARTUP_REPAIR.json")


def stage_power(stage):
    local = SOURCE / "conditions/power"
    files = ("run_video_power.py", "analyze_video_energy.py", "test_analyze_video_energy.py",
             "calibrate_video_layout.py", "prepare_windows_bench.py", "CHROME_LAYOUT.json",
             "WINDOWS_VIDEO_POWER_BENCH.md", "VIDEO_ENERGY_ANALYSIS.md", "power-bodycam-v1.webm")
    for name in files:
        copy_file(local / name, stage / POWER / name)
    for name in ("screen-variation/run_screen_trial.py", "bodycam-distance/measure_distance_trial.py"):
        copy_file(SOURCE / name, stage / STAGED / name)
    manifest = json.loads((local / "VIDEO_MANIFEST.json").read_text(encoding="utf-8"))
    video, = manifest["items"]
    video["path"] = (POWER / "power-bodycam-v1.webm").as_posix()
    if sha(stage / video["path"]) != video["sha256"]:
        raise ValueError("Optical workload hash mismatch")
    display = json.loads((local / "DISPLAY_SOURCES_WITH_VIDEO.json").read_text(encoding="utf-8"))
    row, = [item for item in display["items"] if item["token"] == video["token"]]
    row["path"] = video["path"]
    write_json(stage / POWER / "VIDEO_MANIFEST.json", manifest)
    write_json(stage / POWER / "DISPLAY_SOURCES_WITH_VIDEO.json", {"items": [row]})
    write_json(stage / "work/project-research-v5/runs/optical/DISPLAY_SOURCES.json", {"items": []})
    copy_file(STUDY / "config/WINDOWS_STUDY_PLAN.json",
              stage / "work/project-research-v5/config/WINDOWS_STUDY_PLAN.json")
    shutil.copytree(STUDY / "instrument/nano_meter_diagnostic",
                    stage / "work/project-research-v5/instrument/nano_meter_diagnostic",
                    ignore=shutil.ignore_patterns("__pycache__", "*.pyc", ".DS_Store"))
    for folder in ("windows_wheels", "windows_calibration_wheels"):
        for path in (STUDY / "handoffs" / folder).iterdir():
            if path.suffix in {".whl", ".lock"}:
                copy_file(path, stage / "work/project-research-v5/handoffs" / folder / path.name)
    staged_wheels = stage / "work/project-research-v5/handoffs"
    pillow = staged_wheels / "windows_wheels/pillow-12.3.0-cp312-cp312-win_amd64.whl"
    pillow_hash = sha(pillow)
    if pillow_hash != "a2b55dd6b2a4c4b7d87ffa56bdb33fdc5fdb9a462173861a7bc097f17d91cb09":
        raise ValueError("Pillow wheel hash mismatch")
    lock = staged_wheels / "windows_calibration_wheels/requirements-windows-calibration.lock"
    with lock.open("a", encoding="utf-8", newline="\n") as stream:
        stream.write(f"Pillow==12.3.0 --hash=sha256:{pillow_hash}\n")
    copy_file(local / "WINDOWS_VIDEO_POWER_BENCH.md", stage / "WINDOWS_VIDEO_POWER_BENCH.md")
    copy_file(local / "SETUP_WINDOWS.cmd", stage / "SETUP_WINDOWS.cmd")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--deployment-receipt", type=Path, default=INSTALL)
    parser.add_argument("--staged-release", type=Path,
                        help="optional exact release directory from a previous verified packet")
    parser.add_argument("--startup-repair-receipt", type=Path,
                        help="optional verified board startup files installed after release activation")
    args = parser.parse_args()
    with tempfile.TemporaryDirectory(prefix="nicla-windows-bench-stage-") as temp:
        stage = Path(temp)
        copy_code(stage)
        stage_release(stage, args.deployment_receipt, args.staged_release)
        if args.startup_repair_receipt:
            stage_startup_repair(stage, args.startup_repair_receipt)
        stage_power(stage)
        result = create(stage, args.output, [path.name for path in stage.iterdir()])
    print(json.dumps({"archive": result["archive"], "verification": result["verification"]}, indent=2))


if __name__ == "__main__":
    main()
