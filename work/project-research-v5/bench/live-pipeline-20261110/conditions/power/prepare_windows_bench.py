"""Stage exact power-bench inputs from a verified R17, R18, or R19 install receipt."""

import argparse
import hashlib
import json
from pathlib import Path
import re
import sys
import tempfile


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[5]
R17 = "haar-close-r17-20261111"
R18 = "haar-fast-r18-20261111"
R19 = "haar-sightings-r19-20261111"
RELEASE_ROLLBACKS = {R17: "haar-responsive-r16-20261111", R18: R17, R19: R18}
FIRMWARE_PARTS = ("boot.py", "wearable_app.py", "identity_contracts.py", "runtime_support.py", "telemetry.py",
                  "mpy/build_manifest.json", "mpy/wearable_app.mpy", "mpy/identity_contracts.mpy",
                  "mpy/runtime_support.mpy", "mpy/telemetry.mpy")
COMPACT_FIRMWARE_PARTS = (FIRMWARE_PARTS[0],) + FIRMWARE_PARTS[5:]
STAGED_RECEIPT = ROOT / "bench/INSTALL_RECEIPT.json"
STAGED_RELEASE = ROOT / "bench/release"
LEGACY_STAGED_RECEIPT = ROOT / "bench/R17_INSTALL_RECEIPT.json"
LEGACY_STAGED_RELEASE = ROOT / "bench/release-r17"
LOCAL_RECEIPT = (ROOT / "work/project-finish-v4/deployments/backups/20261111T194959Z"
                 / "deployment-manifest.json")
CONFIG_NAME = "runtime_config.json"
INTEGRATION_MANIFEST = "models/deployment-v1/integration_manifest.json"


def sha256(data):
    return hashlib.sha256(data).hexdigest()


def load(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def video_inputs():
    manifest_path = HERE / "VIDEO_MANIFEST.json"
    video = load(manifest_path)["items"][0]
    display_path = HERE / "DISPLAY_SOURCES_WITH_VIDEO.json"
    matches = [row for row in load(display_path)["items"] if row["token"] == video["token"]]
    if len(matches) != 1:
        raise ValueError("Expected one display source for the scheduled video token")
    display = matches[0]
    if (display.get("media_type") != "video/webm" or display["path"] != video["path"]
            or display["sha256"] != video["sha256"]):
        raise ValueError("Video manifest and optical display source disagree")
    path = ROOT / video["path"]
    if sha256(path.read_bytes()) != video["sha256"]:
        raise ValueError(f"Video payload failed SHA-256 verification: {path}")
    return {"token": video["token"], "path": video["path"], "sha256": video["sha256"],
            "duration_ms": video["duration_ms"], "video_manifest_sha256": sha256(manifest_path.read_bytes()),
            "display_manifest_sha256": sha256(display_path.read_bytes())}


def release_inputs(receipt, staged_release=None):
    release_id = receipt["release_id"]
    compact = receipt.get("source_free") is True
    if (not isinstance(release_id, str) or not re.fullmatch(r"[A-Za-z0-9_-]{1,48}", release_id)
            or (release_id not in RELEASE_ROLLBACKS and not (compact and staged_release))
            or receipt["release_root"] != f"/flash/releases/{release_id}"
            or receipt.get("post_promotion_hashes_all_match") is not True):
        raise ValueError("Expected a successful R17/R18/R19 or staged compact deployment readback receipt")
    hashes = receipt["post_promotion_hashes"]
    if not isinstance(hashes, dict):
        raise ValueError("Deployment readback has invalid component hashes")
    firmware_parts = COMPACT_FIRMWARE_PARTS if compact else FIRMWARE_PARTS
    if "mpy/offline_sightings.mpy" in hashes:
        if not compact:
            firmware_parts += ("offline_sightings.py",)
        firmware_parts += ("mpy/offline_sightings.mpy",)
    required = {*firmware_parts, CONFIG_NAME, INTEGRATION_MANIFEST}
    allowed_models = {f"models/deployment-v1/{name}" for name in (
        "detector_gray_fomo_int8.tflite", "identity_embedding_a010_int8.tflite", "identity_classifier_a025_int8.tflite",
        "detector_gray_fomo_int8.txt", "identity_classifier_a025_int8.txt", "fixed_roster_gallery.json",
        "embedding_gallery_calibration.json")}
    if (not required.issubset(hashes) or set(hashes) - required - allowed_models
            or not all(isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) for value in hashes.values())):
        raise ValueError("Deployment readback is missing runtime component hashes or has invalid entries")
    if staged_release:
        files = {name: (staged_release / name).read_bytes() for name in hashes}
    else:
        sys.path.insert(0, str(ROOT / "FaceRecognition/tools"))
        from deploy_nicla_runtime import make_payload

        fast = {"haar_fast_live": True} if release_id in (R18, R19) else {}
        with tempfile.TemporaryDirectory(prefix="nicla-bench-config-") as tmp:
            payload = make_payload(receipt["release_root"], tmp, receipt["model_asset_root"],
                detector_backend="gray_haar_box", haar_distance_roi=True, haar_local_contrast=True,
                haar_walking_zoom=True, haar_responsive=True, haar_close_face=True, compact=compact,
                enable_battery_sightings=receipt.get("offline_sightings_enabled") is True, **fast)
            files = {name: path.read_bytes() for name, path in payload.items()}
            # Deployment receipts bind the canonical LF JSON produced on the Mac.
            files[CONFIG_NAME] = (json.dumps(load(payload[CONFIG_NAME]), sort_keys=True, indent=2) + "\n").encode("utf-8")
            manifest = load(payload[INTEGRATION_MANIFEST])
            manifest["runtime_config"]["sha256"] = sha256(files[CONFIG_NAME])
            files[INTEGRATION_MANIFEST] = (json.dumps(manifest, sort_keys=True, indent=2) + "\n").encode("utf-8")
    if files.keys() != hashes.keys():
        raise ValueError(f"Local {release_id} payload file list differs from the installed release")
    actual = {name: sha256(data) for name, data in files.items()}
    for name, digest in actual.items():
        if digest != hashes[name]:
            raise ValueError(f"Current source cannot reproduce {release_id} file {name}")
    if compact:
        integration = json.loads(files[INTEGRATION_MANIFEST])
        binding = integration.get("precompiled_runtime") or {}
        runtime_files = {name.removeprefix("mpy/") for name in firmware_parts if name.endswith(".mpy")}
        if (binding.get("source_free") is not True or
                binding.get("manifest_sha256") != actual["mpy/build_manifest.json"] or
                binding.get("manifest_path") != receipt["release_root"] + "/mpy/build_manifest.json" or
                set(binding.get("modules", {})) != runtime_files or
                any(binding["modules"][name].get("sha256") != actual["mpy/" + name] or
                    binding["modules"][name].get("path") != receipt["release_root"] + "/mpy/" + name
                    for name in runtime_files)):
            raise ValueError("Compact runtime module binding mismatch")
    composite = sha256("".join(f"{name}\t{actual[name]}\n" for name in firmware_parts).encode("utf-8"))
    return files, composite


def default_receipt_path():
    return next((path for path in (STAGED_RECEIPT, LEGACY_STAGED_RECEIPT) if path.is_file()), LOCAL_RECEIPT)


def default_staged_release(receipt_path):
    root = {STAGED_RECEIPT: STAGED_RELEASE, LEGACY_STAGED_RECEIPT: LEGACY_STAGED_RELEASE}.get(receipt_path.resolve())
    return root if root is not None and root.is_dir() else None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--deployment-receipt", type=Path)
    parser.add_argument("--staged-release", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        receipt_path = args.deployment_receipt or default_receipt_path()
        receipt = load(receipt_path)
        staged_release = args.staged_release or default_staged_release(receipt_path)
        payload, composite = release_inputs(receipt, staged_release)
        selected = (CONFIG_NAME, "wearable_app.py", "mpy/wearable_app.mpy")
        if receipt.get("source_free") is True:
            selected = (CONFIG_NAME, "mpy/build_manifest.json", "mpy/wearable_app.mpy")
        files = {name: payload[name] for name in selected}
        video = video_inputs()
        args.output.mkdir(parents=True, exist_ok=False)
        for name, data in files.items():
            target = args.output / name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(data)
        record = {"schema": "nicla-windows-video-bench-inputs-v1", "scope": "offline_input_verification",
                  "device_opened": False, "release_id": receipt["release_id"], "model_asset_root": receipt["model_asset_root"],
                  "deployment_receipt_sha256": sha256(receipt_path.read_bytes()),
                  "expected_device_firmware_sha256": composite,
                  "files": {name: sha256(data) for name, data in files.items()}, "video": video}
        (args.output / "BENCH_INPUTS.json").write_bytes((json.dumps(record, indent=2) + "\n").encode("utf-8"))
    except (KeyError, OSError, ValueError) as error:
        parser.exit(2, f"Bench inputs could not be prepared: {error}\n")
    print(f"Verified offline {receipt['release_id']} firmware, config and video inputs in {args.output}")


if __name__ == "__main__":
    main()
