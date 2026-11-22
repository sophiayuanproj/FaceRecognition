#!/usr/bin/env python3
"""Safely install one hash-bound I-matrix release on a Nicla Vision.

This Windows-capable deployment helper consumes the lightweight overlay
manifest produced by nicla_study.i_matrix_campaign. It never modifies the
canonical source model stage and keeps the same versioned-release/pointer
pattern as deploy_nicla_runtime.py.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
from functools import lru_cache
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import subprocess
import sys
import tempfile
import time
import uuid

APP = Path(__file__).resolve().parents[1]
WORKSPACE = APP.parent
STUDY = WORKSPACE / "work/project-finish-v4"
FIRMWARE_PARTS = ("wearable_app.py", "identity_contracts.py", "runtime_support.py", "telemetry.py")
FIRMWARE_SOURCE = {"wearable_app.py": APP / "firmware/main.py",
                   "identity_contracts.py": APP / "firmware/identity_contracts.py",
                   "runtime_support.py": APP / "firmware/runtime_support.py",
                   "telemetry.py": APP / "firmware/telemetry.py"}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def firmware_sha256() -> str:
    digest = hashlib.sha256()
    for name in FIRMWARE_PARTS:
        digest.update((name + "\t" + sha256(FIRMWARE_SOURCE[name]) + "\n").encode("utf-8"))
    return digest.hexdigest()


def _read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


@lru_cache(maxsize=1)
def _source_matrix_index() -> tuple[str, dict[tuple[str, int, str], dict]]:
    matrix_path = STUDY / "config/I_PHYSICAL_MATRIX.json"
    matrix = _read(matrix_path)
    if matrix.get("schema") != "I-full-model-compression-lifecycle-matrix-1":
        raise ValueError("unsupported source I matrix")
    index = {}
    for row in matrix.get("variants", []):
        if row.get("training_root_seed") is not None:
            index[(str(row.get("arm_id")), int(row["training_root_seed"]),
                   str(row.get("export_format")))] = row
    return sha256(matrix_path), index


def _invoke(mpremote: Path, port: str, *args: str, timeout: float = 30.0,
            check: bool = True) -> subprocess.CompletedProcess:
    if not mpremote.is_file():
        raise FileNotFoundError("mpremote executable does not exist: " + str(mpremote))
    transient = ("failed to access", "device not configured", "UnicodeError:")
    result = None
    for attempt in range(5):
        result = subprocess.run([str(mpremote), "connect", port, *map(str, args)],
            text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            timeout=timeout, check=False)
        output = result.stdout or ""
        if result.returncode == 0 or not any(term in output for term in transient) or attempt == 4:
            break
        time.sleep(1.0 + 0.5 * attempt)
    if check and result.returncode:
        raise RuntimeError("mpremote failed (%d): %s\n%s" % (result.returncode,
            " ".join(map(str, args)), result.stdout or ""))
    return result


def _remote_sha(mpremote: Path, port: str, remote_path: str) -> str:
    result = _invoke(mpremote, port, "fs", "sha256sum", ":" + remote_path)
    match = re.search(r"\b([0-9a-f]{64})\b", result.stdout or "")
    if not match:
        raise RuntimeError("device did not return a SHA-256 for " + remote_path + ": " + (result.stdout or ""))
    return match.group(1)


def _readback(mpremote: Path, port: str, remote_path: str, local_path: Path) -> dict:
    result = _invoke(mpremote, port, "fs", "cp", ":" + remote_path, str(local_path), check=False)
    if result.returncode == 0 and local_path.is_file():
        return {"present": True, "bytes": local_path.stat().st_size, "sha256": sha256(local_path)}
    local_path.unlink(missing_ok=True)
    return {"present": False}


def _validate_bundle(manifest_path: Path) -> tuple[dict, dict[str, Path]]:
    manifest_path = manifest_path.resolve()
    manifest = _read(manifest_path)
    if manifest.get("schema") != "I-nicla-variant-overlay-1":
        raise ValueError("unsupported I variant bundle manifest")
    if not re.fullmatch(r"face-i-[0-9a-f]{14}", str(manifest.get("release_id", ""))):
        raise ValueError("variant manifest release ID is invalid")
    if not re.fullmatch(r"[0-9a-f]{64}", str(manifest.get("artifact_sha256", ""))):
        raise ValueError("variant manifest artifact SHA-256 is invalid")
    if not re.fullmatch(r"[0-9a-f]{64}", str(manifest.get("device_runtime_config_sha256", ""))):
        raise ValueError("variant manifest runtime config SHA-256 is invalid")
    if not re.fullmatch(r"[0-9a-f]{64}", str(manifest.get("device_integration_manifest_sha256", ""))):
        raise ValueError("variant manifest integration manifest SHA-256 is invalid")
    source_matrix_sha, matrix_rows = _source_matrix_index()
    if str(manifest.get("source_matrix_sha256", "")).lower() != source_matrix_sha.lower():
        raise ValueError("source I matrix differs from the one used to prepare this bundle")
    release_id = manifest["release_id"]
    release_root = "/flash/releases/" + release_id
    if manifest.get("release_root") != release_root:
        raise ValueError("variant release root does not match its immutable release ID")
    if manifest.get("target_model_path") != release_root + "/models/deployment-v1/identity_classifier_a025_int8.tflite":
        raise ValueError("variant target model path does not match the immutable release root")
    if manifest.get("target_labels_path") != release_root + "/models/deployment-v1/identity_classifier_a025_int8.txt":
        raise ValueError("variant target labels path does not match the immutable release root")
    sources = {}
    for target, row in manifest.get("payload_files", {}).items():
        posix = PurePosixPath(target)
        if (posix.is_absolute() or "\\" in target or ":" in target or "\x00" in target or
                any(part in ("", ".", "..") for part in posix.parts)):
            raise ValueError("unsafe release payload target: " + target)
        source_rel = str(row.get("source_relative", ""))
        source = (WORKSPACE / source_rel).resolve()
        try:
            source.relative_to(WORKSPACE.resolve())
        except ValueError as error:
            raise ValueError("payload source escapes workspace: " + source_rel) from error
        if not source.is_file():
            raise FileNotFoundError("variant deployment source missing: " + str(source))
        if source.stat().st_size != int(row.get("bytes", -1)) or sha256(source) != row.get("sha256"):
            raise ValueError("variant payload source hash/size mismatch: " + source_rel)
        sources[target] = source
    required = {"boot.py", "wearable_app.py", "identity_contracts.py", "runtime_support.py",
                "telemetry.py", "runtime_config.json"}
    required.update("models/deployment-v1/" + name for name in (
        "detector_gray_fomo_int8.tflite", "detector_gray_fomo_int8.txt",
        "identity_embedding_a010_int8.tflite", "identity_classifier_a025_int8.tflite",
        "identity_classifier_a025_int8.txt", "fixed_roster_gallery.json",
        "embedding_gallery_calibration.json", "integration_manifest.json"))
    if set(sources) != required:
        raise ValueError("variant payload inventory differs; missing=%s unexpected=%s" % (
            ",".join(sorted(required - set(sources))), ",".join(sorted(set(sources) - required))))
    if sha256(sources["runtime_config.json"]) != manifest["device_runtime_config_sha256"]:
        raise ValueError("runtime config file does not match the variant's device config SHA")
    if sha256(sources["models/deployment-v1/integration_manifest.json"]) != manifest["device_integration_manifest_sha256"]:
        raise ValueError("integration manifest file does not match the variant SHA")
    if sha256(sources["models/deployment-v1/identity_classifier_a025_int8.tflite"]) != manifest["artifact_sha256"]:
        raise ValueError("classifier payload differs from the variant artifact SHA")
    artifact_source_rel = str(manifest.get("artifact_source_relative", ""))
    artifact_source = (WORKSPACE / artifact_source_rel).resolve()
    try:
        artifact_source.relative_to(WORKSPACE.resolve())
    except ValueError as error:
        raise ValueError("E artifact source escapes workspace") from error
    if (not artifact_source.is_file() or sha256(artifact_source) != manifest["artifact_sha256"] or
            artifact_source.stat().st_size != int(manifest.get("artifact_bytes", -1))):
        raise ValueError("E artifact source no longer matches the bundle's immutable model hash")
    export_receipt = artifact_source.with_suffix(".json")
    if not export_receipt.is_file() or sha256(export_receipt) != manifest.get("export_receipt_sha256"):
        raise ValueError("E export receipt is missing or differs from the prepared bundle")
    export = _read(export_receipt)
    if (export.get("status") != "complete_verified" or export.get("host_inference_status") != "pass" or
            str(export.get("sha256", "")).lower() != manifest["artifact_sha256"].lower()):
        raise ValueError("E export receipt no longer qualifies as the prepared host-verified artifact")
    matrix_row = matrix_rows.get((str(manifest.get("arm_id")),
        int(manifest.get("training_root_seed", -1)), str(manifest.get("export_format"))))
    if (not matrix_row or matrix_row.get("artifact_status") != "complete_verified" or
            matrix_row.get("host_inference_status") != "pass" or
            str(matrix_row.get("artifact_sha256", "")).lower() != manifest["artifact_sha256"].lower() or
            str(matrix_row.get("root_config_sha256", "")).lower() != str(manifest.get("source_root_result_sha256", "")).lower() or
            str(matrix_row.get("artifact_path_relative", "")) != artifact_source_rel):
        raise ValueError("variant identity/artifact/config differs from the source I matrix")
    arm_id, root_seed = str(manifest["arm_id"]), int(manifest["training_root_seed"])
    bundle_seed = "%s|%s|%s|%s|%s|%s|%s" % (arm_id, root_seed, manifest["export_format"],
        manifest["artifact_sha256"], manifest["firmware_sha256"],
        manifest.get("runtime_config_source_sha256", ""), manifest["source_matrix_sha256"])
    expected_bundle_id = "I-%s-%s-%s" % (arm_id, root_seed,
        hashlib.sha256(bundle_seed.encode()).hexdigest()[:10])
    if manifest.get("bundle_id") != expected_bundle_id:
        raise ValueError("variant bundle ID does not match its source artifact and runtime generation")
    expected_model_id = "E-%s-%s-%s" % (arm_id, root_seed, manifest["artifact_sha256"][:12])
    if manifest.get("model_id") != expected_model_id:
        raise ValueError("variant model ID does not match its E artifact")
    release_seed = "I-variant-release-v1|%s|%s|%s|%s" % (manifest["bundle_id"],
        manifest["firmware_sha256"], manifest["artifact_sha256"], manifest.get("runtime_config_source_sha256", ""))
    expected_release_id = "face-i-" + hashlib.sha256(release_seed.encode("ascii")).hexdigest()[:14]
    if manifest["release_id"] != expected_release_id:
        raise ValueError("variant release ID does not match immutable source inputs")
    runtime_config = _read(sources["runtime_config.json"])
    large = runtime_config.get("identity", {}).get("large", {})
    if (large.get("model_id") != manifest.get("model_id") or
            large.get("model_sha256", "").lower() != manifest["artifact_sha256"].lower() or
            large.get("model_path") != manifest["target_model_path"] or
            large.get("labels_path") != manifest["target_labels_path"] or
            large.get("source_arm_id") != manifest["arm_id"] or
            int(large.get("training_root_seed", -1)) != int(manifest["training_root_seed"]) or
            large.get("export_format") != manifest["export_format"]):
        raise ValueError("variant runtime config identity/path differs from its bundle manifest")
    integration = _read(sources["models/deployment-v1/integration_manifest.json"])
    if integration.get("runtime_config", {}).get("sha256", "").lower() != manifest["device_runtime_config_sha256"].lower():
        raise ValueError("integration manifest does not bind the variant runtime config")
    classifier = integration.get("files", {}).get("identity_classifier_a025_int8.tflite", {})
    if (classifier.get("path") != manifest["target_model_path"] or
            str(classifier.get("sha256", "")).lower() != manifest["artifact_sha256"].lower()):
        raise ValueError("integration manifest classifier identity differs from the E artifact")
    if firmware_sha256().lower() != str(manifest.get("firmware_sha256", "")).lower():
        raise ValueError("current firmware source differs from the variant bundle's source hash")
    return manifest, sources


def _mkdir_remote(mpremote: Path, port: str, remote_parent: str) -> None:
    script = (
        "import os\n"
        "p=%r\n"
        "cur=''\n"
        "for part in p.split('/'):\n"
        "    if not part: continue\n"
        "    cur += '/' + part\n"
        "    try: os.mkdir(cur)\n"
        "    except OSError: pass\n"
        "print('I_VARIANT_DIR_READY')\n" % remote_parent)
    _invoke(mpremote, port, "exec", script)


def _device_preflight(mpremote: Path, port: str, release_root: str,
                      manifest: dict, sources: dict[str, Path]) -> dict:
    config_path = sources["runtime_config.json"]
    config = _read(config_path)
    expected_model = config["identity"]["large"]
    script = (
        "import csi,ml,gc,json\n"
        "root=%r\n"
        "cfg=json.load(open(root+'/runtime_config.json'))\n"
        "def shape(x):\n    return tuple(x[0]) if isinstance(x,(tuple,list)) and len(x)==1 and isinstance(x[0],(tuple,list)) else tuple(x)\n"
        "def dtype(x):\n    x=x[0] if isinstance(x,(tuple,list)) else x\n    return {'b':'int8','B':'uint8','f':'float32'}.get(x,str(x))\n"
        "assert csi.CSI is not None\n"
        "spec=cfg['identity']['large']\n"
        "model_path=spec['model_path'].replace(%r,root,1)\n"
        "m=ml.Model(model_path)\n"
        "assert shape(m.input_shape)==tuple(spec['input_shape'])\n"
        "assert dtype(m.input_dtype)==spec['input_dtype']\n"
        "assert abs(float(m.input_scale[0])-float(spec['input_scale']))<0.000001\n"
        "assert int(m.input_zero_point[0])==int(spec['input_zero_point'])\n"
        "assert shape(m.output_shape)==tuple(spec['output_shape'])\n"
        "assert dtype(m.output_dtype)==spec['output_dtype']\n"
        "assert abs(float(m.output_scale[0])-float(spec['output_scale']))<0.000001\n"
        "assert int(m.output_zero_point[0])==int(spec['output_zero_point'])\n"
        "assert list(m.labels)==list(spec['class_order'])\n"
        "print('I_VARIANT_MODEL_OK',spec['model_id'],spec['model_sha256'],m.ram,gc.mem_free())\n"
        "m=None;gc.collect()\n"
        "print('I_VARIANT_PREFLIGHT_OK')\n" % (manifest["release_root"], release_root))
    result = _invoke(mpremote, port, "exec", script, timeout=90)
    output = result.stdout or ""
    if "I_VARIANT_MODEL_OK" not in output or "I_VARIANT_PREFLIGHT_OK" not in output:
        raise RuntimeError("target did not complete exact I variant model metadata/resource preflight:\n" + output)
    return {"status": "passed", "output": output.strip(), "model_id": expected_model["model_id"],
            "artifact_sha256": expected_model["model_sha256"], "device_config_sha256": sha256(config_path)}


def _install(args) -> dict:
    if os.name != "nt":
        raise RuntimeError("Nicla variant deployment is a Windows hardware action; no device commands were sent")
    manifest, sources = _validate_bundle(args.bundle_manifest)
    mpremote = args.mpremote.resolve()
    if not args.port:
        raise ValueError("an explicit freshly discovered Nicla maintenance port is required")
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    backup_dir = (args.backup_dir or STUDY / "deployments/backups" /
                  ("I-variant-%s-%s" % (manifest["bundle_id"], stamp))).resolve()
    backup_dir.mkdir(parents=True, exist_ok=False)
    prior = {
        "main.py": _readback(mpremote, args.port, "/flash/main.py", backup_dir / "main.py"),
        "face_runtime.current": _readback(mpremote, args.port, "/flash/face_runtime.current",
                                           backup_dir / "face_runtime.current"),
    }
    if not prior["main.py"]["present"]:
        raise RuntimeError("Could not read back /flash/main.py; no target files changed")
    if prior["main.py"].get("sha256") != _remote_sha(mpremote, args.port, "/flash/main.py"):
        raise RuntimeError("startup backup checksum failed; no target files changed")
    if prior["face_runtime.current"]["present"] and prior["face_runtime.current"].get("sha256") != _remote_sha(
            mpremote, args.port, "/flash/face_runtime.current"):
        raise RuntimeError("active release pointer backup checksum failed; no target files changed")

    release_id = manifest["release_id"]
    release_root = "/flash/releases/" + release_id
    _invoke(mpremote, args.port, "exec",
        "import os\ntry: os.mkdir('/flash/releases')\nexcept OSError: pass\nprint('I_RELEASES_READY')\n")
    existing = _invoke(mpremote, args.port, "exec",
        "import os\np=%r\ntry: os.stat(p)\nexcept OSError: p=''\nprint('I_RELEASE_EXISTS' if p else 'I_RELEASE_ABSENT')\n" % release_root,
        check=True)
    reused = "I_RELEASE_EXISTS" in (existing.stdout or "")
    expected_remote = {target: row["sha256"] for target, row in manifest["payload_files"].items()
                       if target != "boot.py"}
    if reused:
        for target, digest in expected_remote.items():
            if _remote_sha(mpremote, args.port, release_root + "/" + target) != digest:
                raise RuntimeError("existing immutable release has a different payload: " + target)
    else:
        stage_root = release_root + ".stage-" + uuid.uuid4().hex[:10]
        _mkdir_remote(mpremote, args.port, stage_root)
        for target, source in sources.items():
            if target == "boot.py":
                continue
            remote = stage_root + "/" + target
            _mkdir_remote(mpremote, args.port, remote.rsplit("/", 1)[0])
            _invoke(mpremote, args.port, "fs", "cp", str(source), ":" + remote, timeout=120)
            if _remote_sha(mpremote, args.port, remote) != manifest["payload_files"][target]["sha256"]:
                raise RuntimeError("variant release payload readback hash mismatch: " + target)
        _device_preflight(mpremote, args.port, stage_root, manifest, sources)
        promote = ("import os\nfinal=%r; stage=%r\n"
            "try: os.stat(final)\nexcept OSError: os.rename(stage,final)\nelse: raise OSError('release exists; refusing overwrite')\n"
            "print('I_RELEASE_PROMOTED')\n" % (release_root, stage_root))
        _invoke(mpremote, args.port, "exec", promote)

    # Recheck the immutable final release and its model resource behavior
    # before changing the active pointer.
    for target, digest in expected_remote.items():
        if _remote_sha(mpremote, args.port, release_root + "/" + target) != digest:
            raise RuntimeError("final release verification failed: " + target)
    final_preflight = _device_preflight(mpremote, args.port, release_root, manifest, sources)
    pointer_tmp = backup_dir / "face_runtime.current.next"
    pointer_tmp.write_text(release_root + "\n", encoding="ascii")
    pointer_sha = sha256(pointer_tmp)
    _invoke(mpremote, args.port, "fs", "cp", str(pointer_tmp), ":/flash/face_runtime.current.next")
    if _remote_sha(mpremote, args.port, "/flash/face_runtime.current.next") != pointer_sha:
        raise RuntimeError("I variant release pointer readback failed")
    pointer_swap = (
        "import os\n"
        "try: os.remove('/flash/face_runtime.current.prev')\nexcept OSError: pass\n"
        "try: os.rename('/flash/face_runtime.current','/flash/face_runtime.current.prev')\nexcept OSError: pass\n"
        "os.rename('/flash/face_runtime.current.next','/flash/face_runtime.current')\n"
        "print('I_VARIANT_POINTER_PROMOTED')\n")
    _invoke(mpremote, args.port, "exec", pointer_swap)

    # Install the boot selector only after all release payload and pointer reads pass.
    boot_source = sources["boot.py"]
    _invoke(mpremote, args.port, "fs", "cp", str(boot_source), ":/flash/main.py.next")
    if _remote_sha(mpremote, args.port, "/flash/main.py.next") != manifest["payload_files"]["boot.py"]["sha256"]:
        raise RuntimeError("I variant boot selector readback failed")
    _invoke(mpremote, args.port, "exec",
        "import os\ntry: os.remove('/flash/main.py')\nexcept OSError: pass\n"
        "os.rename('/flash/main.py.next','/flash/main.py')\nprint('I_VARIANT_BOOT_INSTALLED')\n")
    if _remote_sha(mpremote, args.port, "/flash/main.py") != manifest["payload_files"]["boot.py"]["sha256"]:
        raise RuntimeError("final I variant boot selector checksum failed")
    if not args.no_reset:
        _invoke(mpremote, args.port, "reset", timeout=20)
    receipt = {
        "schema": "I-nicla-variant-deployment-1",
        "status": "installed_verified",
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "bundle_id": manifest["bundle_id"], "arm_id": manifest["arm_id"],
        "training_root_seed": manifest["training_root_seed"], "model_id": manifest["model_id"],
        "artifact_sha256": manifest["artifact_sha256"], "firmware_sha256": manifest["firmware_sha256"],
        "device_config_sha256": manifest["device_runtime_config_sha256"],
        "integration_manifest_sha256": manifest["device_integration_manifest_sha256"],
        "release_id": release_id, "release_root": release_root,
        "board_port": args.port, "mpremote": str(mpremote), "backup_dir": str(backup_dir),
        "prior_target": prior, "immutable_release_reused": reused,
        "payload_files": {name: {"sha256": row["sha256"], "bytes": row["bytes"]}
                           for name, row in manifest["payload_files"].items()},
        "target_metadata_resource_preflight": final_preflight,
        "reset_requested": not args.no_reset,
        "physical_inference_and_camera_parity": "not established by deployment preflight",
    }
    return receipt


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--bundle-manifest", type=Path, required=True)
    p.add_argument("--mpremote", type=Path, required=True)
    p.add_argument("--port", required=True, help="fresh, explicit Nicla USB CDC maintenance port")
    p.add_argument("--output-receipt", type=Path, required=True)
    p.add_argument("--backup-dir", type=Path)
    p.add_argument("--no-reset", action="store_true")
    args = p.parse_args(argv)
    try:
        receipt = _install(args)
        output = args.output_receipt.resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        if output.exists():
            raise FileExistsError("deployment receipt path already exists; refusing replacement: " + str(output))
        output.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(json.dumps(receipt, indent=2, sort_keys=True))
        return 0
    except (OSError, ValueError, KeyError, RuntimeError, subprocess.CalledProcessError,
            subprocess.TimeoutExpired) as error:
        print("I variant deployment stopped safely: %s: %s" % (type(error).__name__, error), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
