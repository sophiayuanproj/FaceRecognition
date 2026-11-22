#!/usr/bin/env python3
"""Backup, preflight, install and activate a versioned standalone Nicla release."""

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys
import tempfile
import time

ROOT = Path(__file__).resolve().parents[1]
PROJECT = ROOT.parent
from mpremote_path import resolve_mpremote

MPREMOTE = resolve_mpremote(ROOT)
KEEP_WATCHDOG_FED = False
NATIVE_HAAR_CONTRACT = {
    "backend": "gray_haar_box", "model_id": "openmv-haar-wide240-gray96",
    "cascade_path": "/rom/haarcascade_frontalface.cascade", "cascade_bytes": 52396,
    "cascade_sha256": "e3d67398d384b4bd421a19fb98cbc240eab8ff4b151bfa661260d847196e90f3",
    "threshold": 0.6, "pyramid_scale": 1.25, "wide_contrast_gain": 3, "wide_contrast_center": 100,
}
HAAR_DISTANCE_PROFILE = {"model_id": "openmv-haar-wide240-gray96-roi128",
                         "distance_roi_side_px": 128}
HAAR_LOCAL_CONTRAST_PROFILE = {"model_id": "openmv-haar-clahe240-gray96-roi128",
                               "wide_local_contrast_clip_limit": 2}
HAAR_WALKING_PROFILE = {"model_id": "openmv-haar-clahe240-gray96-adaptive-roi128",
                        "adaptive_distance_zoom": True}
HAAR_RESPONSIVE_PROFILE = {"model_id": "openmv-haar-clahe240-adaptive-roi128-gray96",
                           "distance_roi_before_small": True}
HAAR_CLOSE_PROFILE = {"close_face_content_px": 80,
                      "model_id": "openmv-haar-clahe240-adaptive-roi128-gray96-close80"}
HAAR_FAST_LIVE_POLICY = {"interval_ms": 125, "pace_from_capture_start": True,
                         "compact_detector_first": True, "compact_live_telemetry": True}
MODEL_NAMES = ("detector_gray_fomo_int8.tflite", "identity_embedding_a010_int8.tflite",
               "identity_classifier_a025_int8.tflite", "detector_gray_fomo_int8.txt",
               "identity_classifier_a025_int8.txt", "fixed_roster_gallery.json",
               "embedding_gallery_calibration.json",
               "integration_manifest.json")
PY_NAMES = ("identity_contracts.py", "runtime_support.py", "telemetry.py")
MPY_NAMES = ("wearable_app.mpy", "identity_contracts.mpy",
             "runtime_support.mpy", "telemetry.mpy", "build_manifest.json")
FIRMWARE_PARTS = ("boot.py", "wearable_app.py", "identity_contracts.py",
                  "runtime_support.py", "telemetry.py", "mpy/build_manifest.json",
                  "mpy/wearable_app.mpy", "mpy/identity_contracts.mpy",
                  "mpy/runtime_support.mpy", "mpy/telemetry.mpy")
OFFLINE_SOURCE = "offline_sightings.py"
OFFLINE_MPY = "offline_sightings.mpy"
STANDARD_FLASH_RESERVE = 32768
BATTERY_JPEG_RESERVE = 96 * 1024
BATTERY_MAX_RECORDS = 24
BATTERY_RECORD_OVERHEAD = 38 + 768
NICLA_USB_VID = 0x2341
NICLA_USB_PID = 0x045F
NICLA_USB_PRODUCT = "nicla vision"


def sha256_bytes(data):
    return hashlib.sha256(data).hexdigest()


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def invoke(port, *args, check=True, capture=True, retry_transient=True, timeout=None):
    if not MPREMOTE.is_file():
        raise RuntimeError("Missing Nicla host tooling: " + str(MPREMOTE))
    transient_errors = ("failed to access", "device not configured", "input/output error",
                        "device i/o error", "unicodeerror:",
                        "no device with serial number", "no device found for serial")
    result = None
    command = [str(MPREMOTE), "connect", port]
    if KEEP_WATCHDOG_FED:
        command.extend(("exec", "from machine import WDT; WDT(timeout=30000).feed()", "+"))
    command.extend(map(str, args))
    attempts = 5 if retry_transient else 1
    for attempt in range(attempts):
        result = subprocess.run(command,
            text=True, stdout=subprocess.PIPE if capture else None,
            stderr=subprocess.STDOUT if capture else None, timeout=timeout)
        output = result.stdout or ""
        output_folded = output.casefold()
        if result.returncode == 0 or not any(error.casefold() in output_folded
                for error in transient_errors) or attempt == attempts - 1:
            break
        time.sleep(1.0 + attempt * 0.5)
    if check and result.returncode:
        raise RuntimeError("mpremote failed (%d): %s\n%s" % (result.returncode,
            " ".join(map(str, args)), result.stdout or ""))
    return result


def remote_sha(port, path):
    result = invoke(port, "fs", "sha256sum", ":" + path)
    match = re.search(r"\b([0-9a-f]{64})\b", result.stdout or "")
    if not match:
        raise RuntimeError("No remote SHA-256 for " + path + ": " + (result.stdout or ""))
    return match.group(1)


def assert_no_mounted_openmv_disk():
    """Prevent concurrent macOS FAT and Nicla serial filesystem writers."""
    if sys.platform != "darwin":
        return
    result = subprocess.run(["/sbin/mount"], check=True, capture_output=True, text=True)
    for line in result.stdout.splitlines():
        match = re.search(r" on (/Volumes/.+) \(msdos(?:,|\))", line)
        if not match:
            continue
        volume = Path(match.group(1).replace("\\040", " "))
        if (volume / ".openmv_disk").is_file():
            raise RuntimeError("OpenMV USB disk is mounted at %s; unmount that volume "
                               "without unplugging USB before writing over serial" % volume)


def release_inputs():
    paths = [ROOT / "firmware/main.py", ROOT / "firmware/boot.py",
             ROOT / "firmware/runtime_config.json"]
    paths.extend(ROOT / "firmware" / name for name in PY_NAMES)
    paths.extend(ROOT / "firmware/mpy" / name for name in MPY_NAMES)
    build_manifest = json.loads((ROOT / "firmware/mpy/build_manifest.json").read_text())
    if OFFLINE_MPY in build_manifest.get("modules", {}):
        paths.extend((ROOT / "firmware" / OFFLINE_SOURCE, ROOT / "firmware/mpy" / OFFLINE_MPY))
    paths.extend(ROOT / "models/deployment-v1" / name for name in MODEL_NAMES)
    return paths


def default_release_id(model_asset_root=None, detector_backend=None, haar_distance_roi=False,
                       haar_local_contrast=False, haar_walking_zoom=False, haar_responsive=False,
                       haar_close_face=False, haar_fast_live=False, compact=False,
                       enable_battery_sightings=False):
    release_digest = hashlib.sha256()
    for path in sorted(release_inputs()):
        if not path.is_file():
            raise FileNotFoundError("Deployment input missing: " + str(path))
        release_digest.update(path.name.encode("utf-8") + b"\t")
        release_digest.update(sha256(path).encode("ascii") + b"\n")
    if model_asset_root:
        release_digest.update(b"model_asset_root\t" + model_asset_root.encode("utf-8") + b"\n")
    if detector_backend:
        release_digest.update(b"detector_backend\t" + detector_backend.encode("ascii") + b"\n")
        release_digest.update(json.dumps(NATIVE_HAAR_CONTRACT, sort_keys=True).encode("ascii"))
    if haar_distance_roi:
        release_digest.update(json.dumps(HAAR_DISTANCE_PROFILE, sort_keys=True).encode("ascii"))
    if haar_local_contrast:
        release_digest.update(json.dumps(HAAR_LOCAL_CONTRAST_PROFILE, sort_keys=True).encode("ascii"))
    if haar_walking_zoom:
        release_digest.update(json.dumps(HAAR_WALKING_PROFILE, sort_keys=True).encode("ascii"))
    if haar_responsive:
        release_digest.update(json.dumps(HAAR_RESPONSIVE_PROFILE, sort_keys=True).encode("ascii"))
    if haar_close_face:
        release_digest.update(json.dumps(HAAR_CLOSE_PROFILE, sort_keys=True).encode("ascii"))
    if haar_fast_live:
        release_digest.update(json.dumps(HAAR_FAST_LIVE_POLICY, sort_keys=True).encode("ascii"))
    if compact:
        release_digest.update(b"source_free_runtime\ttrue\n")
    if enable_battery_sightings:
        release_digest.update(b"offline_sightings_enabled\ttrue\n")
        release_digest.update(sha256(ROOT / "firmware/battery_usb_boot.py").encode("ascii"))
    return "face-v1-" + release_digest.hexdigest()[:12]


def normalize_release_root(value):
    value = str(value).strip().rstrip("/")
    if value.startswith("/flash/releases/"):
        root = value
    else:
        if not re.fullmatch(r"[A-Za-z0-9_-]{1,48}", value):
            raise ValueError("invalid model asset release ID")
        root = "/flash/releases/" + value
    if not re.fullmatch(r"/flash/releases/[A-Za-z0-9_-]{1,48}", root):
        raise ValueError("model asset release root must be an immutable /flash/releases/<id> path")
    return root


def validate_reuse_root(model_asset_root, active_root, previous_root=None,
                        referenced_roots=()):
    """Allow reuse from active/rollback roots or their hash-checked asset roots."""
    allowed = {str(active_root).strip()}
    if previous_root:
        allowed.add(str(previous_root).strip())
    allowed.update(str(root).strip() for root in referenced_roots if root)
    if model_asset_root not in allowed:
        raise RuntimeError("--reuse-models-from must equal an active/previous release or a model root referenced by one")


def referenced_model_asset_roots(port, release_roots, backup_dir):
    """Read active/rollback manifests and return their declared immutable model roots."""
    roots = set()
    receipts = {}
    for release_root in sorted({str(root).strip() for root in release_roots if root}):
        manifest_remote = release_root + "/models/deployment-v1/integration_manifest.json"
        local = Path(backup_dir) / (release_root.rsplit("/", 1)[-1] + ".integration_manifest.json")
        result = invoke(port, "fs", "cp", ":" + manifest_remote, str(local), check=False)
        if result.returncode != 0 or not local.is_file():
            receipts[release_root] = {"present": False}
            continue
        local_hash = sha256(local)
        if local_hash != remote_sha(port, manifest_remote):
            raise RuntimeError("release manifest readback hash mismatch: " + manifest_remote)
        try:
            manifest = json.loads(local.read_text())
            if not isinstance(manifest, dict):
                raise ValueError("invalid manifest root")
        except (ValueError, UnicodeError):
            # Historical pointers can refer to a damaged release. Preserve the
            # exact bytes in the backup, but never derive trust from that file.
            # A directly selected active/rollback root still has to pass every
            # separate immutable model-file checksum below.
            receipts[release_root] = {"present": True, "path": str(local),
                "sha256": local_hash, "status": "invalid_json_untrusted"}
            continue
        declared = manifest.get("model_asset_root")
        if declared:
            model_root = normalize_release_root(declared)
            roots.add(model_root)
        else:
            model_root = release_root
        receipts[release_root] = {"present": True, "path": str(local),
            "sha256": local_hash, "model_asset_root": model_root}
    return roots, receipts


def device_preflight(port, release_root, model_asset_root=None):
    model_root = model_asset_root or release_root
    command = (
        "import csi, ml, machine, gc, os; "
        "root=%r; "
        "assert csi.CSI is not None; "
        "det=ml.Model(root+'/models/deployment-v1/detector_gray_fomo_int8.tflite'); "
        "assert len(det.input_shape)==1 and tuple(det.input_shape[0])==(1,96,96,1); "
        "assert len(det.output_shape)==1 and tuple(det.output_shape[0])==(1,12,12,2); "
        "assert det.input_dtype[0]=='b' and abs(det.input_scale[0]-0.003921568859368563)<1e-6 and det.input_zero_point[0]==-128; "
        "assert det.output_dtype[0]=='b' and abs(det.output_scale[0]-0.00390625)<1e-7 and det.output_zero_point[0]==-128; "
        "print('C_MODEL_OK', det.input_dtype, det.input_scale, det.input_zero_point); "
        "det=None; gc.collect(); "
        "emb=ml.Model(root+'/models/deployment-v1/identity_embedding_a010_int8.tflite'); "
        "assert len(emb.input_shape)==1 and tuple(emb.input_shape[0])==(1,96,96,1); "
        "assert len(emb.output_shape)==1 and tuple(emb.output_shape[0])==(1,64); "
        "assert emb.input_dtype[0]=='b' and abs(emb.input_scale[0]-0.003921568859368563)<1e-6 and emb.input_zero_point[0]==-128; "
        "assert emb.output_dtype[0]=='b' and abs(emb.output_scale[0]-0.009992608800530434)<1e-6 and emb.output_zero_point[0]==-128; "
        "print('D_EMBEDDING_OK', emb.input_dtype, emb.output_dtype, emb.ram); "
        "emb=None; gc.collect(); "
        "clf=ml.Model(root+'/models/deployment-v1/identity_classifier_a025_int8.tflite'); "
        "assert len(clf.input_shape)==1 and tuple(clf.input_shape[0])==(1,96,96,1); "
        "assert len(clf.output_shape)==1 and tuple(clf.output_shape[0])==(1,6); "
        "assert clf.input_dtype[0]=='b' and abs(clf.input_scale[0]-0.003921568859368563)<1e-6 and clf.input_zero_point[0]==-128; "
        "assert clf.output_dtype[0]=='b' and abs(clf.output_scale[0]-0.026111828163266182)<1e-6 and clf.output_zero_point[0]==-10; "
        "assert clf.labels==['K01','K02','K03','K04','K05','OTHER']; "
        "print('E_CLASSIFIER_OK', clf.input_dtype, clf.output_dtype, clf.ram); "
        "clf=None; gc.collect(); "
        "uart=machine.UART(4, baudrate=115200, timeout=0); print('UART4_CONSTRUCTED', uart is not None); "
        "uart=None; print('NICLA_RUNTIME_PREFLIGHT_OK')" % model_root)
    result = invoke(port, "exec", command)
    output = result.stdout or ""
    if "C_MODEL_OK" not in output or "D_EMBEDDING_OK" not in output or "E_CLASSIFIER_OK" not in output or "UART4_CONSTRUCTED" not in output or "NICLA_RUNTIME_PREFLIGHT_OK" not in output:
        raise RuntimeError("Target did not complete model/UART preflight:\n" + output)
    return output.strip()


def readback_existing(port, path, destination):
    # CDC can briefly drop a filesystem transfer while the Nicla finishes
    # rebooting or re-enumerates after the broker releases its serial handle.
    # Retry this read-only backup step with a fresh mpremote session; later
    # deployment steps still require explicit remote hash verification.
    for attempt in range(5):
        result = invoke(port, "fs", "cp", ":" + path, str(destination), check=False)
        if result.returncode == 0 and destination.is_file():
            return {"present": True, "sha256": sha256(destination), "bytes": destination.stat().st_size}
        destination.unlink(missing_ok=True)
        if attempt < 4:
            time.sleep(0.5 + attempt * 0.5)
    return {"present": False}


def ensure_remote_directory(port, path):
    """Create each parent and verify it; never hide a flash mkdir failure."""
    command = (
        "import os\n"
        "target=%r\n"
        "cur=''\n"
        "for part in target.split('/'):\n"
        "    if not part: continue\n"
        "    cur += '/' + part\n"
        "    try:\n        st=os.stat(cur)\n"
        "    except OSError:\n        os.mkdir(cur); st=os.stat(cur)\n"
        "    if not (st[0] & 0x4000): raise OSError('not_a_directory:'+cur)\n"
        "print('REMOTE_DIRECTORY_VERIFIED',target)\n" % path)
    invoke(port, "exec", command)


def create_remote_directory(port, path):
    """Create a new directory and fail if it already exists or readback fails."""
    command = (
        "import os\n"
        "p=%r\n"
        "try:\n    os.stat(p)\n"
        "except OSError:\n    pass\n"
        "else:\n    raise RuntimeError('staging_path_already_exists:'+p)\n"
        "os.mkdir(p)\n"
        "st=os.stat(p)\n"
        "if not (st[0] & 0x4000): raise OSError('staging_path_not_directory:'+p)\n"
        "print('REMOTE_STAGING_DIRECTORY_VERIFIED',p)\n" % path)
    invoke(port, "exec", command)


def flash_headroom(payload, block_bytes, battery=False):
    if block_bytes <= 0:
        raise ValueError("invalid Nicla flash allocation unit")
    allocated = sum(((path.stat().st_size + block_bytes - 1) // block_bytes) * block_bytes
                    for path in payload.values())
    reserve = STANDARD_FLASH_RESERVE
    if battery:
        spool = BATTERY_JPEG_RESERVE + BATTERY_MAX_RECORDS * (BATTERY_RECORD_OVERHEAD + block_bytes - 1)
        reserve += ((spool + block_bytes - 1) // block_bytes + 1) * block_bytes
    return allocated, reserve


def nicla_usb_ports():
    try:
        from serial.tools import list_ports
    except ImportError as error:
        raise RuntimeError("Battery USB activation requires pyserial in the deployment environment") from error
    return [port for port in list_ports.comports()
            if port.vid == NICLA_USB_VID and port.pid == NICLA_USB_PID and
            NICLA_USB_PRODUCT in (port.product or "").casefold()]


def capture_battery_usb_identity(port):
    matches = [item for item in nicla_usb_ports() if item.device.casefold() == port.casefold()]
    if len(matches) != 1 or not matches[0].serial_number:
        raise RuntimeError("Cannot identify the requested Nicla by USB VID, PID, product and serial number")
    match = matches[0]
    return {"vid": match.vid, "pid": match.pid, "serial_number": match.serial_number}


def verify_battery_boot_after_usb_change(port, identity, digest, saved, old_digest, *, timeout_s=20):
    """Follow only the previously identified board, then verify its hook and mode."""
    command = ("import pyb,hashlib,binascii\n"
               "def sha(path):\n"
               "    if path is None: return '-'\n"
               "    try:\n        stream=open(path,'rb')\n"
               "    except OSError as error:\n        if error.args[0] == 2: return '-'\n        raise\n"
               "    with stream:\n        digest=hashlib.sha256()\n"
               "        while True:\n            block=stream.read(512)\n"
               "            if not block: break\n            digest.update(block)\n"
               "    return binascii.hexlify(digest.digest()).decode()\n"
               "print('NICLA_BATTERY_USB_READBACK',pyb.usb_mode(),sha('/flash/boot.py'),sha(%r))\n" % saved)
    deadline = time.monotonic() + timeout_s
    detail = "matching USB descriptor absent"
    while time.monotonic() < deadline:
        candidates = [item.device for item in nicla_usb_ports()
                      if item.vid == identity["vid"] and item.pid == identity["pid"] and
                      item.serial_number == identity["serial_number"]]
        if len(candidates) == 1:
            selected = candidates[0]
            try:
                result = invoke(selected, "exec", command, check=False, retry_transient=False,
                                timeout=min(5, max(0.1, deadline - time.monotonic())))
            except subprocess.TimeoutExpired:
                detail = "USB readback timed out on " + selected
                continue
            output = result.stdout or ""
            proof = re.search(r"^NICLA_BATTERY_USB_READBACK\s+(\S+)\s+([0-9a-f]{64}|-)"
                              r"\s+([0-9a-f]{64}|-)\s*$", output, re.MULTILINE)
            if result.returncode == 0 and proof:
                mode, actual, backup = proof.groups()
                if actual != digest or (saved and backup != old_digest):
                    raise RuntimeError("Promoted USB boot hook or retained original failed hash readback")
                if mode not in ("VCP", "CDC"):
                    raise RuntimeError("Battery USB hook did not establish CDC-only mode: " + mode)
                return {"port": selected, "initial_port": port, "usb_mode": mode,
                        "usb_serial_verified": True, "port_changed": selected != port}
            detail = "no verified readback on " + selected
        elif candidates:
            detail = "more than one port matches the same Nicla serial number"
        else:
            detail = "matching USB descriptor absent"
        time.sleep(0.2)
    raise RuntimeError("Cannot verify the same Nicla after battery USB hook activation: " + detail +
                       ". The release pointer has not been activated")


def inspect_battery_boot(port, backup_dir):
    capability = invoke(port, "exec", "import pyb; mode=getattr(pyb,'usb_mode',None); "
        "print('NICLA_BATTERY_USB_MODE', mode() if callable(mode) else 'unsupported')").stdout or ""
    mode = re.search(r"^NICLA_BATTERY_USB_MODE\s+(VCP(?:\+MSC)?|CDC(?:\+MSC)?)\s*$",
                     capability, re.MULTILINE)
    if not mode:
        raise RuntimeError("Target does not report a supported pyb.usb_mode(); preserved existing USB boot")
    usb_mode = mode.group(1)
    command = ("import os\n"
               "try:\n    os.stat('/flash/boot.py')\n"
               "except OSError as error:\n    if error.args[0] != 2: raise\n"
               "    print('NICLA_BATTERY_BOOT_ABSENT')\n"
               "else:\n    print('NICLA_BATTERY_BOOT_PRESENT')\n")
    response = invoke(port, "exec", command).stdout or ""
    if "NICLA_BATTERY_BOOT_ABSENT" in response:
        return {"present": False, "usb_mode_before": usb_mode}
    if "NICLA_BATTERY_BOOT_PRESENT" not in response:
        raise RuntimeError("Could not establish the existing USB boot hook state")
    local = backup_dir / "boot.py"
    prior = readback_existing(port, "/flash/boot.py", local)
    if not prior["present"] or prior["sha256"] != remote_sha(port, "/flash/boot.py"):
        raise RuntimeError("Could not hash-verify backup of /flash/boot.py")
    hook = ROOT / "firmware/battery_usb_boot.py"
    if prior["sha256"] != sha256(hook):
        if any(line.strip() and not line.lstrip().startswith("#")
               for line in local.read_text(encoding="utf-8-sig").splitlines()):
            raise RuntimeError("Custom /flash/boot.py preserved; battery USB hook was not installed")
    prior["already_vcp_hook"] = prior["sha256"] == sha256(hook)
    prior["usb_mode_before"] = usb_mode
    return prior


def install_battery_boot(port, prior, stamp, identity):
    hook = ROOT / "firmware/battery_usb_boot.py"
    digest = sha256(hook)
    if prior.get("already_vcp_hook"):
        verified = verify_battery_boot_after_usb_change(port, identity, digest, None, None)
        return dict(verified, sha256=digest, status="already_installed")
    stage = "/flash/boot.py.battery.next"
    invoke(port, "exec", ("import os\n"
        "try:\n    os.stat(%r)\nexcept OSError as error:\n    if error.args[0] != 2: raise\n"
        "else:\n    raise RuntimeError('battery_boot_staging_path_exists')\n" % stage))
    invoke(port, "fs", "cp", str(hook), ":" + stage)
    if remote_sha(port, stage) != digest:
        raise RuntimeError("Battery USB boot hook staging hash mismatch")
    saved = "/flash/boot.py.pre-battery." + stamp if prior["present"] else None
    if saved and remote_sha(port, "/flash/boot.py") != prior["sha256"]:
        raise RuntimeError("Existing USB boot hook changed during staging")
    command = ("import os\n"
               "p='/flash/boot.py'; old=%r; stage=%r\n"
               "if old:\n    os.rename(p,old)\n"
               "else:\n    try:\n        os.stat(p)\n"
               "    except OSError as error:\n        if error.args[0] != 2: raise\n"
               "    else:\n        raise RuntimeError('battery_boot_changed_during_staging')\n"
               "try:\n    os.rename(stage,p)\n"
               "except OSError:\n    if old: os.rename(old,p)\n    raise\n"
               "if hasattr(os,'sync'): os.sync()\n" % (saved, stage))
    # A completed rename can immediately invalidate the old port. Do not
    # replay the mutation. The same-board USB and file readback is decisive.
    try:
        result = invoke(port, "exec", command, check=False, retry_transient=False, timeout=8)
        promotion_returned = result.returncode == 0
    except subprocess.TimeoutExpired:
        promotion_returned = False
    try:
        verified = verify_battery_boot_after_usb_change(port, identity, digest, saved, prior.get("sha256"))
    except RuntimeError as error:
        if not promotion_returned:
            raise RuntimeError(str(error) + " (hook promotion command did not return success)") from error
        raise
    return dict(verified, sha256=digest, status="installed", original_board_backup=saved)


def install_startup(port, boot=None):
    boot = ROOT / "firmware/boot.py" if boot is None else boot
    digest = sha256(boot)
    invoke(port, "fs", "cp", str(boot), ":/flash/main.py.next")
    if remote_sha(port, "/flash/main.py.next") != digest:
        raise RuntimeError("boot startup readback mismatch")
    promote = ("import os\n"
        "try:\n    os.remove('/flash/main.py')\nexcept OSError:\n    pass\n"
        "os.rename('/flash/main.py.next','/flash/main.py')\n"
        "print('NICLA_VERSIONED_STARTUP_INSTALLED')\n")
    invoke(port, "exec", promote)
    if remote_sha(port, "/flash/main.py") != digest:
        raise RuntimeError("final startup readback mismatch")
    return digest


def load_resume_target(path):
    receipt = json.loads(path.read_text(encoding="utf-8"))
    release_id = receipt["release_id"]
    if (not isinstance(release_id, str) or not re.fullmatch(r"[A-Za-z0-9_-]{1,48}", release_id) or
            receipt["release_root"] != "/flash/releases/" + release_id or
            receipt.get("source_free") is not True or type(receipt.get("offline_sightings_enabled")) is not bool):
        raise ValueError("resume requires a compact deployment target manifest")
    model_root = normalize_release_root(receipt["model_asset_root"])
    if model_root != receipt["model_asset_root"]:
        raise ValueError("resume model asset root must be canonical")
    rows = receipt["payload"]
    if not isinstance(rows, dict):
        raise ValueError("invalid resume payload manifest")
    required = {"boot.py", "runtime_config.json", "models/deployment-v1/integration_manifest.json"}
    required.update("mpy/" + name for name in MPY_NAMES)
    if receipt["offline_sightings_enabled"] or "mpy/" + OFFLINE_MPY in rows:
        required.add("mpy/" + OFFLINE_MPY)
    if model_root == receipt["release_root"]:
        required.update("models/deployment-v1/" + name for name in MODEL_NAMES)
    if set(rows) != required or any(
            not isinstance(row, dict) or not isinstance(row.get("sha256"), str) or
            not re.fullmatch(r"[0-9a-f]{64}", row["sha256"]) or
            type(row.get("bytes")) is not int or row["bytes"] <= 0 for row in rows.values()):
        raise ValueError("resume target manifest does not bind the exact compact payload")
    return receipt


def verify_resume_prior(port, receipt, prior):
    old = receipt["prior_target"]
    if not old.get("face_runtime.current", {}).get("present"):
        raise RuntimeError("resume requires an unchanged active release and rollback pointer backup")
    for name in ("main.py", "face_runtime.current", "face_runtime.current.prev", "boot.py"):
        if name not in old and name not in prior:
            continue
        before, now = old.get(name, {}), prior.get(name, {})
        if (before.get("present") != now.get("present") or
                (now.get("present") and (now.get("sha256") != before.get("sha256") or
                 remote_sha(port, "/flash/" + name) != before.get("sha256")))):
            raise RuntimeError("resume target changed since staging: /flash/" + name)


def read_promoted_metadata(port, receipt, backup_dir):
    root = receipt["release_root"]
    command = ("import os,json\n"
               "root=%r\n"
               "assert os.stat(root)[0] & 0x4000\n"
               "try:\n    os.stat(root+'.next')\n"
               "except OSError as error:\n    if error.args[0] != 2: raise\n"
               "else:\n    raise RuntimeError('unexpected_release_staging_directory')\n"
               "def names(path):\n    result=[]\n    for entry in os.ilistdir(root+path):\n"
               "        child=path+'/'+entry[0]\n        if entry[1] & 0x4000: result.extend(names(child))\n"
               "        else: result.append(child[1:])\n    return result\n"
               "print('RESUME_PROMOTED_FILES',json.dumps(names('')))\n" % root)
    result = invoke(port, "exec", command).stdout or ""
    inventory = re.search(r"^RESUME_PROMOTED_FILES\s+(\[[^\r\n]*\])\s*$", result, re.MULTILINE)
    if not inventory or set(json.loads(inventory.group(1))) != set(receipt["payload"]):
        raise RuntimeError("promoted release file list differs from the resume target manifest")
    readback = {}
    for name in ("boot.py", "runtime_config.json", "mpy/build_manifest.json",
                 "models/deployment-v1/integration_manifest.json"):
        local = backup_dir / ("promoted-" + name.rsplit("/", 1)[-1])
        actual = readback_existing(port, root + "/" + name, local)
        expected = receipt["payload"][name]
        if not actual.get("present") or actual["sha256"] != expected["sha256"] or actual["bytes"] != expected["bytes"]:
            raise RuntimeError("promoted release metadata readback mismatch: " + name)
        readback[name] = local
    config = json.loads(readback["runtime_config.json"].read_text())
    integration = json.loads(readback["models/deployment-v1/integration_manifest.json"].read_text())
    build = json.loads(readback["mpy/build_manifest.json"].read_text())
    binding = integration["precompiled_runtime"]
    if (integration.get("model_asset_root") != receipt["model_asset_root"] or
            integration["runtime_config"]["sha256"] != receipt["payload"]["runtime_config.json"]["sha256"] or
            binding.get("source_free") is not True or binding["manifest_path"] != root + "/mpy/build_manifest.json" or
            binding["manifest_sha256"] != receipt["payload"]["mpy/build_manifest.json"]["sha256"] or
            binding["mpy_value"] != build["mpy_value"] or
            (config.get("control", {}).get("offline_sightings_enabled") is True) != receipt["offline_sightings_enabled"]):
        raise RuntimeError("promoted runtime or model bindings differ from the resume target manifest")
    return readback, config


def make_payload(release_root, temp_dir, model_asset_root=None, detector_backend=None,
                 haar_distance_roi=False, haar_local_contrast=False, haar_walking_zoom=False,
                 haar_responsive=False, haar_close_face=False, haar_fast_live=False,
                 compact=False, enable_battery_sightings=False):
    stage = ROOT / "models/deployment-v1"
    model_asset_root = model_asset_root or release_root
    model_prefix = model_asset_root.rstrip("/") + "/models/deployment-v1"
    source = {
        "wearable_app.py": ROOT / "firmware/main.py",
        "boot.py": ROOT / "firmware/boot.py",
        "runtime_config.json": ROOT / "firmware/runtime_config.json",
        **{name: ROOT / "firmware" / name for name in PY_NAMES},
        **{"mpy/" + name: ROOT / "firmware/mpy" / name for name in MPY_NAMES},
        **{"models/deployment-v1/" + name: stage / name for name in MODEL_NAMES},
    }
    mpy_manifest_path = source["mpy/build_manifest.json"]
    mpy_manifest = json.loads(mpy_manifest_path.read_text())
    if mpy_manifest.get("schema") != "nicla-precompiled-runtime-1":
        raise RuntimeError("unsupported precompiled runtime manifest")
    module_rows = mpy_manifest.get("modules", {})
    if OFFLINE_MPY in module_rows:
        source[OFFLINE_SOURCE] = ROOT / "firmware" / OFFLINE_SOURCE
        source["mpy/" + OFFLINE_MPY] = ROOT / "firmware/mpy" / OFFLINE_MPY
    elif enable_battery_sightings:
        raise RuntimeError("battery sightings requires build_nicla_mpy.py --include-offline-sightings")
    for path in source.values():
        if not path.is_file():
            raise FileNotFoundError("Required deployment file missing: " + str(path))
    if compact or enable_battery_sightings:
        expected_mpy = {name for name in MPY_NAMES if name.endswith(".mpy")}
        if OFFLINE_MPY in module_rows:
            expected_mpy.add(OFFLINE_MPY)
        source_rows = mpy_manifest.get("sources", {})
        if (set(module_rows) != expected_mpy or
                set(source_rows) != {name[:-4] + ".py" for name in expected_mpy}):
            raise RuntimeError("unexpected compact runtime build manifest module set")
        abi = mpy_manifest.get("mpy_value")
        if type(abi) is not int or mpy_manifest.get("format_low_byte") != abi & 255:
            raise RuntimeError("invalid compact runtime bytecode ABI")
        for name in expected_mpy:
            row = module_rows[name]
            py_name = name[:-4] + ".py"
            source_row = source_rows[py_name]
            binary = source["mpy/" + name]
            with binary.open("rb") as stream:
                prefix = stream.read(2)
            if (row["source"] != py_name or row["module"] != name[:-4] or
                    prefix != b"M" + bytes((abi & 255,)) or row["bytes"] != binary.stat().st_size or
                    row["sha256"] != sha256(binary) or source_row["bytes"] != source[py_name].stat().st_size or
                    source_row["sha256"] != sha256(source[py_name])):
                raise RuntimeError("compact runtime bytecode/source does not match build manifest: " + name)
    payload = {}
    config = json.loads(source["runtime_config.json"].read_text())
    if enable_battery_sightings:
        config["control"]["offline_sightings_enabled"] = True
    if detector_backend == "gray_haar_box":
        config["detection"] = dict(NATIVE_HAAR_CONTRACT)
        if haar_distance_roi:
            config["detection"].update(HAAR_DISTANCE_PROFILE)
        if haar_local_contrast:
            config["detection"].update(HAAR_LOCAL_CONTRAST_PROFILE)
            baseline = config["policies"]["T"]
            spatial = {"cell_delta_u8": 12, "hysteresis_cells": 1, "onset_cells": 3}
            config["policies"]["V"] = dict(baseline,
                scene_trigger=dict(baseline["scene_trigger"], spatial=spatial))
        if haar_walking_zoom:
            config["detection"].update(HAAR_WALKING_PROFILE)
        if haar_responsive:
            config["detection"].update(HAAR_RESPONSIVE_PROFILE)
            config["policies"]["W"] = dict(config["policies"]["S"], pace_from_capture_start=True)
            config["default_policy"] = "W"
        if haar_close_face:
            if not haar_responsive:
                raise ValueError("Close-face fallback requires the responsive Haar profile")
            config["detection"].update(HAAR_CLOSE_PROFILE)
        if haar_fast_live:
            if not haar_close_face:
                raise ValueError("Fast live requires the close-face profile")
            config["policies"]["F"] = dict(config["policies"]["S"], **HAAR_FAST_LIVE_POLICY)
            config["default_policy"] = "F"
        config["models"]["release_detector_before_identity"] = True
        config["identity"]["small"]["acceptance_uses_calibrated_cutoff"] = True
        config["identity"]["allow_known_identity_acceptance"] = False
    elif detector_backend is not None:
        raise ValueError("unsupported detector backend override")
    def rewrite(value):
        if isinstance(value, dict):
            return {k: rewrite(v) for k, v in value.items()}
        if isinstance(value, list):
            return [rewrite(v) for v in value]
        if isinstance(value, str) and value.startswith("/flash/models/deployment-v1/"):
            return value.replace("/flash/models/deployment-v1", model_prefix, 1)
        return value
    config = rewrite(config)
    config_temp = Path(temp_dir) / "runtime_config.json"
    config_temp.write_text(json.dumps(config, sort_keys=True, indent=2) + "\n")
    payload["runtime_config.json"] = config_temp

    # Rebind the integration manifest to release-local absolute paths and the
    # rewritten config checksum before it is copied to the board.
    manifest = json.loads(source["models/deployment-v1/integration_manifest.json"].read_text())
    for name, row in manifest.get("files", {}).items():
        row["path"] = model_prefix + "/" + name
    for row in manifest.get("labels", {}).values():
        if isinstance(row, dict) and isinstance(row.get("path"), str):
            basename = Path(row["path"]).name
            row["path"] = model_prefix + "/" + basename
    for key in ("detector_path", "classifier_path"):
        if isinstance(manifest.get("labels", {}).get(key), str):
            basename = Path(manifest["labels"][key]).name
            manifest["labels"][key] = model_prefix + "/" + basename
    manifest["runtime_config"] = {"path": release_root + "/runtime_config.json",
                                  "sha256": sha256(config_temp)}
    manifest["gallery"]["path"] = model_prefix + "/fixed_roster_gallery.json"
    if "D_gallery_calibration" in manifest:
        manifest["D_gallery_calibration"]["path"] = model_prefix + "/embedding_gallery_calibration.json"
    manifest["model_asset_root"] = model_asset_root
    manifest["model_asset_release_id"] = model_asset_root.rsplit("/", 1)[-1]
    manifest["precompiled_runtime"] = {
        "manifest_path": release_root + "/mpy/build_manifest.json",
        "manifest_sha256": sha256(mpy_manifest_path),
        "compiler": mpy_manifest.get("compiler"),
        "compiler_package": mpy_manifest.get("compiler_package"),
        "compile_opt_level": mpy_manifest.get("compile_opt_level"),
        "mpy_value": mpy_manifest.get("mpy_value"),
        "format_low_byte": mpy_manifest.get("format_low_byte"),
        "modules": {
            name: {"path": release_root + "/mpy/" + name,
                   "sha256": row["sha256"], "bytes": row["bytes"],
                   "source": row["source"]}
            for name, row in mpy_manifest.get("modules", {}).items()
        },
    }
    if compact:
        manifest["precompiled_runtime"]["source_free"] = True
    model_asset_files = {
        name: {"path": model_prefix + "/" + name, "sha256": sha256(stage / name),
               "bytes": (stage / name).stat().st_size}
        for name in MODEL_NAMES if name != "integration_manifest.json"
    }
    manifest["model_asset_files"] = model_asset_files
    manifest["external_model_files"] = model_asset_files if model_asset_root != release_root else {}
    manifest_temp = Path(temp_dir) / "integration_manifest.json"
    manifest_temp.write_text(json.dumps(manifest, sort_keys=True, indent=2) + "\n")
    payload["models/deployment-v1/integration_manifest.json"] = manifest_temp

    for name, path in source.items():
        if name in ("runtime_config.json", "models/deployment-v1/integration_manifest.json"):
            continue
        if compact and name in mpy_manifest["sources"]:
            continue
        if model_asset_root == release_root or not name.startswith("models/deployment-v1/"):
            payload[name] = path
    return payload


def main():
    global KEEP_WATCHDOG_FED
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", required=True, help="explicit Nicla USB CDC maintenance port")
    parser.add_argument("--no-reset", action="store_true", help="leave activation for a later manual reset")
    parser.add_argument("--release-id", help="optional release ID; defaults to content-derived")
    parser.add_argument("--resume-promoted", type=Path,
                        help="verify an already promoted compact release from its target-manifest.json, then activate")
    parser.add_argument("--reuse-models-from", help="reuse immutable model assets from the active or recorded previous release ID/root")
    parser.add_argument("--keep-watchdog-fed", action="store_true",
                        help="feed the Nicla's 30-second hardware watchdog between serial operations")
    parser.add_argument("--detector-backend", choices=("gray_haar_box",),
        help="select the pinned native face-box alternative instead of the legacy FOMO baseline")
    parser.add_argument("--haar-distance-roi", action="store_true",
        help="test an extra upper-center Haar search for smaller faces after a miss")
    parser.add_argument("--haar-local-contrast", action="store_true",
        help="use adaptive local equalization for the full 240 px Haar pass")
    parser.add_argument("--haar-walking-zoom", action="store_true",
        help="adjust contrast in the upper-center distance scan before running Haar")
    parser.add_argument("--haar-responsive", action="store_true",
        help="try the distance scan before the 96 px fallback and default to faster live frame pacing")
    parser.add_argument("--haar-close-face", action="store_true",
        help="after all normal scans miss, search a zoomed-out copy of the full frame")
    parser.add_argument("--haar-fast-live", action="store_true",
        help="add a fast live policy with compact-first detection and JPEG plus result telemetry")
    parser.add_argument("--compact", action="store_true",
        help="install verified bytecode without the four runtime source copies")
    parser.add_argument("--enable-battery-sightings", action="store_true",
        help="include the offline spool, reserve its flash budget and install the opt-in CDC-only USB boot hook")
    args = parser.parse_args()
    resume = load_resume_target(args.resume_promoted) if args.resume_promoted else None
    if resume:
        if (args.release_id or args.reuse_models_from or args.detector_backend or args.compact or
                args.enable_battery_sightings or args.haar_distance_roi or args.haar_local_contrast or
                args.haar_walking_zoom or args.haar_responsive or args.haar_close_face or args.haar_fast_live):
            parser.error("--resume-promoted takes the release configuration from the target manifest")
        args.release_id = resume["release_id"]
        args.compact = True
        args.enable_battery_sightings = resume["offline_sightings_enabled"]
        if resume["model_asset_root"] != resume["release_root"]:
            args.reuse_models_from = resume["model_asset_root"]
    if args.haar_distance_roi and args.detector_backend != "gray_haar_box":
        parser.error("--haar-distance-roi requires --detector-backend gray_haar_box")
    if args.haar_local_contrast and not args.haar_distance_roi:
        parser.error("--haar-local-contrast requires --haar-distance-roi")
    if args.haar_walking_zoom and not args.haar_local_contrast:
        parser.error("--haar-walking-zoom requires --haar-local-contrast")
    if args.haar_responsive and not args.haar_walking_zoom:
        parser.error("--haar-responsive requires --haar-walking-zoom")
    if args.haar_close_face and not args.haar_responsive:
        parser.error("--haar-close-face requires --haar-responsive")
    if args.haar_fast_live and not args.haar_close_face:
        parser.error("--haar-fast-live requires --haar-close-face")
    KEEP_WATCHDOG_FED = args.keep_watchdog_fed
    assert_no_mounted_openmv_disk()
    battery_identity = capture_battery_usb_identity(args.port) if args.enable_battery_sightings else None
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    backups = PROJECT / "work/project-finish-v4/deployments/backups"
    backups.mkdir(parents=True, exist_ok=True)
    backup_dir = backups / stamp
    backup_dir.mkdir()
    prior = {}
    prior["main.py"] = readback_existing(args.port, "/flash/main.py", backup_dir / "main.py")
    prior["face_runtime.current"] = readback_existing(args.port, "/flash/face_runtime.current", backup_dir / "face_runtime.current")
    prior["face_runtime.current.prev"] = readback_existing(args.port,
        "/flash/face_runtime.current.prev", backup_dir / "face_runtime.current.prev")
    if not prior["main.py"]["present"]:
        raise RuntimeError("Could not read back /flash/main.py; no target files changed")
    if (prior["main.py"].get("sha256") != remote_sha(args.port, "/flash/main.py")):
        raise RuntimeError("Startup backup failed readback checksum")
    if args.enable_battery_sightings:
        prior["boot.py"] = inspect_battery_boot(args.port, backup_dir)
    if resume:
        verify_resume_prior(args.port, resume, prior)
    rollback_release_id = None
    if prior["face_runtime.current"]["present"]:
        rollback_pointer = (backup_dir / "face_runtime.current").read_text().strip()
        match = re.fullmatch(r"/flash/releases/([A-Za-z0-9_-]{1,48})", rollback_pointer)
        rollback_release_id = match.group(1) if match else None

    with tempfile.TemporaryDirectory(prefix="nicla-release-") as temp_dir:
        model_asset_root = normalize_release_root(args.reuse_models_from) if args.reuse_models_from else None
        release_id = args.release_id or default_release_id(model_asset_root, args.detector_backend,
            args.haar_distance_roi, args.haar_local_contrast, args.haar_walking_zoom, args.haar_responsive,
            args.haar_close_face, args.haar_fast_live, args.compact, args.enable_battery_sightings)
        if not re.fullmatch(r"[A-Za-z0-9_-]{1,48}", release_id):
            raise ValueError("invalid release ID")
        release_root = "/flash/releases/" + release_id
        reused_model_files = {}
        referenced_roots = set()
        referenced_manifests = {}
        if model_asset_root:
            active_root = (backup_dir / "face_runtime.current").read_text().strip()
            previous_pointer_path = backup_dir / "face_runtime.current.prev"
            previous_root = (previous_pointer_path.read_text().strip()
                             if previous_pointer_path.is_file() else None)
            referenced_roots, referenced_manifests = referenced_model_asset_roots(
                args.port, (active_root, previous_root), backup_dir)
            validate_reuse_root(model_asset_root, active_root, previous_root, referenced_roots)
            for name in MODEL_NAMES:
                if name == "integration_manifest.json":
                    continue
                local_path = ROOT / "models/deployment-v1" / name
                expected_hash = sha256(local_path)
                remote_path = model_asset_root + "/models/deployment-v1/" + name
                actual_hash = remote_sha(args.port, remote_path)
                if actual_hash != expected_hash:
                    raise RuntimeError("reused model asset checksum mismatch: " + remote_path)
                reused_model_files[name] = {"path": remote_path, "sha256": expected_hash,
                                            "bytes": local_path.stat().st_size}
        if resume:
            if reused_model_files != resume["reused_model_files"]:
                raise RuntimeError("shared model assets differ from the original staging receipt")
            payload, config = read_promoted_metadata(args.port, resume, backup_dir)
            args.detector_backend = ("gray_haar_box" if config["detection"]["backend"] == "gray_haar_box" else None)
            installed_files = resume["payload"]
        else:
            payload = make_payload(release_root, temp_dir, model_asset_root, args.detector_backend,
                args.haar_distance_roi, args.haar_local_contrast, args.haar_walking_zoom, args.haar_responsive,
                args.haar_close_face, args.haar_fast_live, args.compact, args.enable_battery_sightings)
            installed_files = {name: {"sha256": sha256(path), "bytes": path.stat().st_size}
                               for name, path in payload.items()}
        expected = {name: row["sha256"] for name, row in installed_files.items()}
        flash_result = invoke(args.port, "exec", "import os,sys; s=os.statvfs('/flash'); "
            "print('FLASH_FREE_BYTES',s[0]*s[3],s[0],getattr(sys.implementation,'_mpy',-1))")
        free_match = re.search(r"FLASH_FREE_BYTES\s+(\d+)\s+(\d+)\s+(-?\d+)", flash_result.stdout or "")
        free_bytes = int(free_match.group(1)) if free_match else None
        block_bytes = int(free_match.group(2)) if free_match else 0
        payload_bytes = sum(row["bytes"] for row in installed_files.values())
        if free_bytes is None:
            raise RuntimeError("could not read Nicla flash free bytes and allocation unit")
        if args.compact or args.enable_battery_sightings:
            build = json.loads(payload["mpy/build_manifest.json"].read_text())
            if int(free_match.group(3)) != build["mpy_value"]:
                raise RuntimeError("target MicroPython ABI differs from the verified build manifest")
        allocated_bytes, reserve_bytes = flash_headroom({} if resume else payload, block_bytes,
                                                        args.enable_battery_sightings)
        if free_bytes < allocated_bytes + reserve_bytes:
            raise RuntimeError("insufficient flash headroom for versioned release: free=%d payload=%d "
                               "allocated=%d required_margin=%d allocation_unit=%d" %
                               (free_bytes, payload_bytes, allocated_bytes, reserve_bytes, block_bytes))
        (backup_dir / "target-manifest.json").write_text(json.dumps({
            "release_id": release_id, "release_root": release_root,
            "payload": installed_files, "prior_target": prior,
            "model_asset_root": model_asset_root or release_root,
            "referenced_model_asset_roots": sorted(referenced_roots),
            "referenced_release_manifests": referenced_manifests,
            "reused_model_files": reused_model_files,
            "flash_free_bytes_before_install": free_bytes,
            "flash_allocation_unit": block_bytes, "payload_allocated_bytes": allocated_bytes,
            "reserved_flash_bytes": reserve_bytes, "source_free": args.compact,
            "offline_sightings_enabled": args.enable_battery_sightings,
            "resumed_from_target_sha256": sha256(args.resume_promoted) if resume else None,
            "payload_bytes": payload_bytes}, indent=2) + "\n")
        assert_no_mounted_openmv_disk()
        if not resume:
            ensure_remote_directory(args.port, "/flash/releases")
            temporary_root = release_root + ".next"
            create_remote_directory(args.port, temporary_root)
            for name, path in payload.items():
                remote = temporary_root + "/" + name
                parent = remote.rsplit("/", 1)[0]
                ensure_remote_directory(args.port, parent)
                invoke(args.port, "fs", "cp", str(path), ":" + remote)
                actual = remote_sha(args.port, remote)
                if actual != expected[name]:
                    raise RuntimeError("readback hash mismatch: " + remote)
            promote_release = (
                "import os\n"
                "final=%r; temporary=%r\n"
                "try:\n    os.stat(final)\nexcept OSError:\n    os.rename(temporary, final)\nelse:\n    raise OSError('release already exists; refusing overwrite')\n"
                "print('RELEASE_PROMOTED')\n" % (release_root, temporary_root))
            invoke(args.port, "exec", promote_release)
        # A successful hash of each temporary file is not sufficient: the
        # directory promotion itself must preserve every byte before the
        # active pointer can change. This catches damaged/truncated files in
        # the final release tree while the previous release is still active.
        promoted_hashes = {}
        for name, expected_hash in expected.items():
            remote_path = release_root + "/" + name
            actual_hash = remote_sha(args.port, remote_path)
            if actual_hash != expected_hash:
                raise RuntimeError("post-promotion readback hash mismatch before activation: " + remote_path)
            promoted_hashes[name] = actual_hash
        device_preflight(args.port, release_root, model_asset_root)
        if args.detector_backend == "gray_haar_box" and remote_sha(args.port,
                NATIVE_HAAR_CONTRACT["cascade_path"]) != NATIVE_HAAR_CONTRACT["cascade_sha256"]:
            raise RuntimeError("native Haar ROM cascade checksum mismatch before activation")
        if resume:
            verify_resume_prior(args.port, resume, prior)
        assert_no_mounted_openmv_disk()

        # Preserve the original startup file once on the board for recovery.
        preserve = ("import os\n"
            "try:\n    os.stat('/flash/main.py.stock')\nexcept OSError:\n"
            "    with open('/flash/main.py','rb') as src: data=src.read()\n"
            "    with open('/flash/main.py.stock','wb') as dst: dst.write(data)\n"
            "print('STARTUP_BACKUP_PRESENT')\n")
        invoke(args.port, "exec", preserve)

        pointer_temp = Path(temp_dir) / "face_runtime.current"
        pointer_temp.write_text(release_root + "\n")
        pointer_sha = sha256(pointer_temp)
        # Keep the previous rollback pointer as a separate immutable history
        # file before the atomic switch rotates `.prev`; this preserves an
        # older corrupt/recovery pointer across consecutive versioned deploys.
        archived_pointer = None
        if prior.get("face_runtime.current.prev", {}).get("present"):
            archived_pointer = "/flash/face_runtime.history." + stamp
            archived_local = backup_dir / "face_runtime.current.prev"
            invoke(args.port, "fs", "cp", str(archived_local), ":" + archived_pointer)
            if remote_sha(args.port, archived_pointer) != prior["face_runtime.current.prev"]["sha256"]:
                raise RuntimeError("archived previous pointer readback checksum mismatch")
        invoke(args.port, "fs", "cp", str(pointer_temp), ":/flash/face_runtime.current.next")
        if remote_sha(args.port, "/flash/face_runtime.current.next") != pointer_sha:
            raise RuntimeError("release pointer readback mismatch")
        # The new loader can boot the previous full release. Install it
        # before activating source-free bytecode when a rollback root exists.
        startup_digest = None
        if args.compact and prior["face_runtime.current"]["present"]:
            startup_digest = install_startup(args.port, payload["boot.py"] if resume else None)
        battery_hook = None
        if args.enable_battery_sightings:
            battery_hook = install_battery_boot(args.port, prior["boot.py"], stamp, battery_identity)
            args.port = battery_hook["port"]
            assert_no_mounted_openmv_disk()
        pointer_promote = (
            "import os\n"
            "try:\n    os.remove('/flash/face_runtime.current.prev')\nexcept OSError:\n    pass\n"
            "try:\n    os.rename('/flash/face_runtime.current','/flash/face_runtime.current.prev')\nexcept OSError:\n    pass\n"
            "os.rename('/flash/face_runtime.current.next','/flash/face_runtime.current')\n"
            "print('RELEASE_POINTER_PROMOTED')\n")
        invoke(args.port, "exec", pointer_promote)
        if startup_digest is None:
            startup_digest = install_startup(args.port)
        (backup_dir / "deployment-manifest.json").write_text(json.dumps({
            "schema": "nicla-versioned-deployment-1", "created_utc": stamp,
            "release_id": release_id, "release_root": release_root,
            "model_asset_root": model_asset_root or release_root,
            "referenced_model_asset_roots": sorted(referenced_roots),
            "referenced_release_manifests": referenced_manifests,
            "reused_model_files": reused_model_files,
            "port_used": args.port, "preflight": "C/D/E model metadata and UART4 constructor passed",
            "installed_files": installed_files,
            "post_promotion_hashes": promoted_hashes,
            "post_promotion_hashes_all_match": promoted_hashes == expected,
            "prior_target": prior, "startup_sha256": startup_digest,
            "source_free": args.compact, "offline_sightings_enabled": args.enable_battery_sightings,
            "battery_usb_boot": battery_hook,
            "resumed_from_target_sha256": sha256(args.resume_promoted) if resume else None,
            "rollback_release_id": rollback_release_id,
            "flash_free_bytes_before_install": free_bytes, "flash_allocation_unit": block_bytes,
            "payload_allocated_bytes": allocated_bytes, "reserved_flash_bytes": reserve_bytes,
            "preserved_previous_pointer": archived_pointer,
            "rollback_pointer_backup": str(backup_dir / "face_runtime.current"),
            "physical_camera_inference": "pending real board smoke"}, indent=2) + "\n")
    if not args.no_reset:
        invoke(args.port, "reset")
    print(json.dumps({"status": "installed", "release_id": release_id,
        "release_root": release_root, "model_asset_root": model_asset_root or release_root,
        "backup_dir": str(backup_dir), "port_used": args.port,
        "reset": not args.no_reset, "physical_acceptance": "pending"}, indent=2))


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, ValueError, KeyError, RuntimeError, subprocess.CalledProcessError) as error:
        print("Nicla install stopped safely:", error, file=sys.stderr)
        raise SystemExit(1)
