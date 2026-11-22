#!/usr/bin/env python3
"""Volatile GC2145 scene exposure sweep with unsupported readbacks recorded safely."""

import argparse
import base64
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import time

from board_lab import connect


BOARD_PROGRAM = r'''import csi, json, time, ubinascii, gc
cam=csi.CSI(); cam.reset(); cam.pixformat(csi.RGB565); cam.framesize(csi.QVGA); cam.window((240,240)); cam.framebuffers(1)
cam.snapshot(time=900)
def emit(row):
    print('V7EXP:'+json.dumps(row))
def readback(method):
    try: return method()
    except Exception as e: return {'unsupported_or_error':repr(e)}
def set_control(name, fn):
    try:
        value=fn()
        emit({'kind':'control','name':name,'status':'setter_returned','return_value':value})
        return True
    except Exception as e:
        emit({'kind':'control','name':name,'status':'error','error':repr(e)})
        return False
def send_frame(arm, exposure_req, index):
    frame=cam.snapshot()
    jpeg=frame.to_jpeg(quality=70)
    payload=jpeg.bytearray()
    emit({'kind':'frame','arm':arm,'requested_exposure_us':exposure_req,'frame_index':index,
          'width':frame.width(),'height':frame.height(),
          'reported_exposure_us':readback(cam.exposure_us),
          'reported_gain_db':readback(cam.gain_db),
          'jpeg_b64':ubinascii.b2a_base64(payload).strip().decode()})
    frame=None; jpeg=None; payload=None; gc.collect()
emit({'kind':'camera','sensor_id':readback(cam.cid),'reported_width':readback(cam.width),
      'reported_height':readback(cam.height),'auto_start_exposure_us':readback(cam.exposure_us),
      'auto_start_gain_db':readback(cam.gain_db)})
for i in range(3): send_frame('auto_baseline',None,i)
gain_fixed=set_control('auto_gain_off_gain_0db',lambda:cam.auto_gain(False,gain_db=0.0))
wb_fixed=set_control('auto_whitebal_off_rgb_zero',lambda:cam.auto_whitebal(False,rgb_gain_db=(0.0,0.0,0.0)))
for exposure in (1000,2000,4000,8000,16000):
    if set_control('auto_exposure_off_%dus'%exposure,lambda:cam.auto_exposure(False,exposure_us=exposure)):
        cam.snapshot(time=350)
        emit({'kind':'setting','requested_exposure_us':exposure,'status':'setter_returned',
              'gain_fixed_setter_returned':gain_fixed,'whitebal_fixed_setter_returned':wb_fixed,
              'reported_exposure_us':readback(cam.exposure_us),'reported_gain_db':readback(cam.gain_db)})
        for i in range(3): send_frame('manual',exposure,i)
set_control('auto_exposure_restore',lambda:cam.auto_exposure(True))
set_control('auto_gain_restore',lambda:cam.auto_gain(True))
set_control('auto_whitebal_restore',lambda:cam.auto_whitebal(True))
cam.snapshot(time=900)
emit({'kind':'restored_auto','reported_exposure_us':readback(cam.exposure_us),
      'reported_gain_db':readback(cam.gain_db),'frame_width':cam.snapshot().width(),
      'frame_height':cam.snapshot().height()})
'''


def sha256(data):
    return hashlib.sha256(data).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--target-control", type=Path, required=True)
    parser.add_argument("--timeout", type=int, default=180)
    args = parser.parse_args()
    if not 30 <= args.timeout <= 300:
        raise ValueError("timeout must be bounded to 30..300 seconds")
    if args.out.exists() and any(args.out.iterdir()):
        raise FileExistsError("refusing to overwrite evidence directory: " + str(args.out))
    target_control = json.loads(args.target_control.read_text())
    if target_control.get("kind") != "photo" or target_control.get("token") != "b56ac1de32d8311b4bc8ffa6":
        raise ValueError("Codex side-window display is not the expected K01 target")
    args.out.mkdir(parents=True, exist_ok=True)
    private = args.out.parents[1] / "private/current_scene_exposure_v2"
    private.mkdir()
    source = BOARD_PROGRAM.encode("utf-8")
    (args.out / "board_program.py").write_bytes(source)
    row = {
        "schema": "current-scene-exposure-probe-v2",
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "port": args.port,
        "reset_kind": "one_soft_reset_volatile_repl; no board flash writes",
        "source_sha256": sha256(source),
        "requested_exposures_us": [1000, 2000, 4000, 8000, 16000],
        "frames_per_arm": {"auto_baseline": 3, "manual_each_setting": 3},
        "target": {"label": "K01 side-window screen target", "control": target_control},
        "status": "running",
    }
    (args.out / "RUN.json").write_text(json.dumps(row, indent=2) + "\n")
    target = None
    started = time.monotonic()
    try:
        target = connect(args.port)
        target.enter_raw_repl(soft_reset=True)
        stdout, stderr = target.exec_raw(source, timeout=args.timeout)
        (args.out / "block_00.stdout.bin").write_bytes(stdout)
        (args.out / "block_00.stderr.bin").write_bytes(stderr)
        parsed = []
        for line in stdout.splitlines():
            if not line.startswith(b"V7EXP:"):
                continue
            try:
                event = json.loads(line[6:].decode("utf-8"))
            except (ValueError, UnicodeDecodeError):
                continue
            encoded = event.pop("jpeg_b64", None)
            if encoded is not None:
                payload = base64.b64decode(encoded)
                name = "%s_exp%s_frame%02d.jpg" % (
                    event.get("arm", "unknown"), event.get("requested_exposure_us", "auto"),
                    event.get("frame_index", 0))
                path = private / name
                path.write_bytes(payload)
                event["jpeg_path"] = str(path.relative_to(args.out.parents[1]))
                event["jpeg_bytes"] = len(payload)
                event["jpeg_sha256"] = sha256(payload)
            parsed.append(event)
        row.update({
            "status": "captured" if not stderr else "target_stderr",
            "finished_utc": datetime.now(timezone.utc).isoformat(),
            "host_elapsed_s": round(time.monotonic() - started, 3),
            "stdout_sha256": sha256(stdout), "stderr_sha256": sha256(stderr),
            "stderr": stderr.decode("utf-8", "replace")[-4000:],
            "events": parsed,
        })
        if not any(event.get("kind") == "restored_auto" for event in parsed):
            row["status"] = "incomplete_auto_restore_event"
        if len([event for event in parsed if event.get("kind") == "frame"]) != 18:
            row["frame_count_warning"] = True
    except Exception as error:
        row.update({"status": "transport_or_execution_error", "error": repr(error),
                    "host_elapsed_s": round(time.monotonic() - started, 3),
                    "finished_utc": datetime.now(timezone.utc).isoformat()})
    finally:
        if target is not None:
            try:
                target.exit_raw_repl()
            except Exception:
                pass
            target.close()
    (args.out / "RESULT.json").write_text(json.dumps(row, indent=2, sort_keys=True) + "\n")
    print(json.dumps({k: v for k, v in row.items() if k != "events"}, indent=2))
    return 0 if row.get("status") == "captured" and not row.get("frame_count_warning") else 1


if __name__ == "__main__":
    raise SystemExit(main())
