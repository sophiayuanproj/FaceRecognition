"""Exercise the maintained A-full sleep controller with live Nicla sensing.

The controller source is embedded into a volatile raw-REPL program so the
exact repository implementation is exercised without deploying board files.
The K02 development target is used because the verified one-shot set showed a
clean FOMO miss there while the camera frame still contains a textured face.
"""

from __future__ import annotations

import base64
import hashlib
import importlib.util
import json
from pathlib import Path
import time
from urllib.request import urlopen


ROOT = Path(__file__).resolve().parents[2]
STUDY = ROOT / "work/project-research-v5"
FIRMWARE = ROOT / "FaceRecognition/firmware"
RUNS = STUDY / "runs/camera_policy_sleep_wake_device_v3"
PRIVATE = STUDY / "private/camera_policy_sleep_wake_device_v3"
PROGRAM = RUNS / "DEVICE_PROGRAM.py"
MANIFEST = RUNS / "PROGRAM_MANIFEST.json"
RUNNER_PATH = Path(__file__).with_name("v6_run_verified_live_scene_survey_v2.py")
POLICY_SOURCE = FIRMWARE / "policy_controller.py"


def sha(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def build_program() -> dict:
    if RUNS.exists() or PRIVATE.exists():
        raise FileExistsError("preserving any prior camera_policy_sleep_wake_device_v3 output")
    policy_bytes = POLICY_SOURCE.read_bytes()
    policy_b64 = base64.b64encode(policy_bytes).decode("ascii")
    program = r'''import csi,ml,gc,json,time,ubinascii
from ml.postprocessing.edgeimpulse import Fomo
exec(ubinascii.a2b_base64("@@POLICY_B64@@").decode())
camera=csi.CSI();camera.reset();camera.pixformat(csi.RGB565);camera.framesize(csi.QVGA);camera.window((240,240));camera.framebuffers(1);camera.auto_exposure(True);camera.snapshot(time=600)
frame=camera.snapshot()
if frame is None or frame.width()!=240 or frame.height()!=240:raise ValueError("initial_camera_frame_not_240x240")
controller=PolicyController("A-full",ticks_diff=time.ticks_diff,allow_camera_sleep=True)
detector=ml.Model("/rom/fomo_face_detection.tflite",postprocess=Fomo(threshold=.5))
cycles=[];sleep_row=None;wake_row=None;last_interval=1500
for attempt in range(6):
 if attempt>0:time.sleep_ms(last_interval)
 begin=time.ticks_us();frame=camera.snapshot()
 if frame is None or frame.width()!=240 or frame.height()!=240:raise ValueError("camera_frame_not_240x240")
 stats=frame.get_statistics();mean=float(stats.mean);stdev=float(stats.stdev)
 quality_ok,quality_reason=reason_coded_quality(mean,stdev)
 controller.counters["detector_calls"]+=1;pred=detector.predict([frame]);faces=pred[1] if len(pred)>1 else [];face_count=len(faces)
 interval,interval_reason=controller.next_interval(quality_ok,face_count)
 now_ms=time.ticks_ms();controller.note_capture(now_ms);controller.note_result(now_ms,"no_face" if face_count==0 else "uncertain");heartbeat=controller.heartbeat(now_ms)
 sleep_request,sleep_reason=controller.should_sleep_camera(quality_ok,face_count)
 cycles.append({"attempt":attempt+1,"elapsed_us":time.ticks_diff(time.ticks_us(),begin),"quality_mean_u8":mean,"quality_stdev_u8":stdev,"quality_ok":quality_ok,"quality_reason":quality_reason,"fomo_cluster_count":face_count,"controller_status_without_identity":("no_face" if face_count==0 else "uncertain"),"heartbeat_timeout":heartbeat,"interval_ms":interval,"interval_reason":interval_reason,"sleep_request":sleep_request,"sleep_reason":sleep_reason,"gc_free_bytes":gc.mem_free()})
 last_interval=interval;frame=None;pred=None;faces=None;gc.collect()
 if sleep_request:
  sleep_begin=time.ticks_us();camera.sleep(True);sleep_row={"sleep_api_us":time.ticks_diff(time.ticks_us(),sleep_begin),"requested_by":"A-full stationary no-face hysteresis","interval_ms_before_periodic_wake":interval}
  time.sleep_ms(interval);wake_begin=time.ticks_us();camera.sleep(False);camera.snapshot(time=600);frame=camera.snapshot();wake_us=time.ticks_diff(time.ticks_us(),wake_begin)
  if frame is None or frame.width()!=240 or frame.height()!=240:raise ValueError("periodic_wake_frame_not_240x240")
  wake_stats=frame.get_statistics();wake_mean=float(wake_stats.mean);wake_stdev=float(wake_stats.stdev);wake_quality,wake_quality_reason=reason_coded_quality(wake_mean,wake_stdev)
  controller.counters["detector_calls"]+=1;wake_pred=detector.predict([frame]);wake_faces=wake_pred[1] if len(wake_pred)>1 else [];wake_count=len(wake_faces);now_ms=time.ticks_ms();previous_frame_ms=controller.last_frame_ms;gap_ms=time.ticks_diff(now_ms,previous_frame_ms) if previous_frame_ms is not None else None;controller.note_wake(now_ms);controller.note_capture(now_ms);controller.note_result(now_ms,"no_face" if wake_count==0 else "uncertain");heartbeat=controller.heartbeat(now_ms)
  wake_row={"wake_to_usable_frame_ms":wake_us/1000.0,"frame_width":frame.width(),"frame_height":frame.height(),"quality_mean_u8":wake_mean,"quality_stdev_u8":wake_stdev,"quality_ok":wake_quality,"quality_reason":wake_quality_reason,"fomo_cluster_count":wake_count,"bounded_gap_timeout":heartbeat,"elapsed_since_previous_capture_ms":gap_ms,"gc_free_bytes":gc.mem_free()}
  sleep_row.update(wake_row)
  frame=None;wake_pred=None;wake_faces=None;gc.collect();break
row={"schema":"nicla-policy-sleep-wake-device-probe-v3","status":"completed","policy_id":"A-full","policy_controller_source_sha256":"@@POLICY_SHA@@","camera_frame_geometry":[240,240],"camera_id":camera.cid(),"fomo_threshold":0.5,"cycles":cycles,"sleep_and_wake":sleep_row,"sleep_requested":sleep_row is not None,"wake_frame_verified":bool(sleep_row and sleep_row.get("frame_width")==240 and sleep_row.get("frame_height")==240),"controller_counters":controller.counters,"controller_interval_ms":controller.interval_ms,"motion_input":"not sampled; motion adaptation not claimed","identity_model_calls":0,"identity_decisions":0,"board_files_written":False,"startup_path_changed":False,"future_final_used":False,"power_measured":False}
print(json.dumps(row))
frame=None;detector=None;camera.sleep(False);gc.collect()
'''.replace("@@POLICY_B64@@", policy_b64).replace("@@POLICY_SHA@@", sha(policy_bytes))
    RUNS.mkdir(parents=True, exist_ok=False)
    PRIVATE.mkdir(parents=True, exist_ok=False)
    payload = program.encode("utf-8")
    PROGRAM.write_bytes(payload)
    doc = {
        "schema": "nicla-policy-sleep-wake-device-probe-manifest-v3",
        "program": str(PROGRAM.relative_to(ROOT)),
        "program_sha256": sha(payload),
        "embedded_policy_source": str(POLICY_SOURCE.relative_to(ROOT)),
        "policy_source_sha256": sha(policy_bytes),
        "policy_id": "A-full",
        "target_condition": "K02 verified development screen face; FOMO no-face case",
        "target_scene_is_not_device_training_or_final_data": True,
        "board_files_written": False,
        "startup_path_changed": False,
        "power_measured": False,
        "motion_sampled": False,
    }
    MANIFEST.write_text(json.dumps(doc, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return doc


def load_runner():
    spec = importlib.util.spec_from_file_location("verified_scene_runner_for_policy_v1", RUNNER_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError("could not load the verified scene runner")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.PROGRAM = PROGRAM
    module.PROGRAM_SHA = sha(PROGRAM.read_bytes())
    module.PREVIEW_RUNS = PRIVATE / "device_runs"
    module.PREVIEW_FRAMES = PRIVATE / "device_previews"
    return module


def main() -> None:
    manifest = build_program()
    runner = load_runner()
    if not runner.port_owners():
        raise RuntimeError("expected the existing live preview to own the Nicla serial port")
    sources = json.loads(runner.SOURCES.read_text(encoding="utf-8"))["items"]
    source = next(row for row in sources if row.get("label") == "K02" and row.get("role") == "development")
    source = dict(source)
    reviewed_manifest, reviewed_frames = runner.load_reviewed_sources()
    if source["token"] != reviewed_frames["K02"]["display_token"]:
        raise ValueError("K02 development token differs from the reviewed scene")

    runner.PREVIEW_RUNS.mkdir(parents=True, exist_ok=False)
    runner.PREVIEW_FRAMES.mkdir(parents=True, exist_ok=False)

    case_receipt = None
    probe = None
    stopped_pid = None
    old_pid = runner.preview_pid()
    try:
        version, offset = runner.set_control("K02", source)
        case_receipt = runner.capture_scene("K02", source, version, offset)
        stopped_pid = runner.stop_preview()
        probe, stdout, stderr = runner.run_probe("K02", source)
        (PRIVATE / "board_lab.stdout.bin").write_bytes(stdout)
        (PRIVATE / "board_lab.stderr.bin").write_bytes(stderr)
        if stderr:
            raise RuntimeError("board stderr was non-empty: " + repr(stderr[:300]))
    finally:
        if not runner.port_owners():
            runner.start_preview(old_pid=stopped_pid or old_pid)
        source_k01 = next(row for row in sources if row.get("label") == "K01" and row.get("role") == "development")
        version, offset = runner.set_control("K01", source_k01)
        runner.wait_for_render(version, offset)
        time.sleep(runner.DWELL_S)
        restore_status = runner.get_json("http://127.0.0.1:8766/status")
        if restore_status.get("error") or restore_status.get("age_s", 99) > 0.5:
            raise RuntimeError("K01 restore preview is stale: " + repr(restore_status))
        with urlopen("http://127.0.0.1:8766/snapshot.jpg", timeout=5) as response:
            restore = response.read()
        (PRIVATE / "RESTORE_K01.jpg").write_bytes(restore)
        (PRIVATE / "RESTORE_K01.json").write_text(json.dumps({
            "control_version": version,
            "target_token": json.loads(runner.CONTROL.read_text(encoding="utf-8"))["token"],
            "fresh_camera_status": restore_status,
            "preview_sha256": sha(restore),
        }, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    if probe is None or case_receipt is None:
        raise RuntimeError("policy diagnostic did not produce a complete preview and board result")
    device = probe["device_output"]
    summary = {
        "schema": "nicla-policy-sleep-wake-device-result-v3",
        "status": "completed" if device.get("status") == "completed" else "device_probe_incomplete",
        "experiment_manifest": str(MANIFEST.relative_to(ROOT)),
        "program_sha256": manifest["program_sha256"],
        "policy_source_sha256": manifest["policy_source_sha256"],
        "policy_id": device.get("policy_id"),
        "target_scene": "K02",
        "target_preview_sha256": case_receipt["preview_frame_sha256"],
        "target_token": case_receipt["display_token"],
        "device_result": device,
        "device_result_path": probe["device_result_path"],
        "board_stdout_sha256": probe["device_stdout_sha256"],
        "board_files_written": False,
        "startup_path_changed": False,
        "power_measured": False,
        "future_final_used": False,
        "independent_physical_blocks": 0,
        "cold_power_cycles": 0,
    }
    (RUNS / "RESULT.json").write_text(json.dumps(summary, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({
        "status": summary["status"],
        "policy_id": summary["policy_id"],
        "cycles": len(device.get("cycles", [])),
        "sleep_requested": device.get("sleep_requested"),
        "wake_frame_verified": device.get("wake_frame_verified"),
        "wake_to_usable_frame_ms": (device.get("sleep_and_wake") or {}).get("wake_to_usable_frame_ms"),
        "board_files_written": False,
        "startup_path_changed": False,
        "restored_target": "K01",
    }, sort_keys=True))


if __name__ == "__main__":
    main()
