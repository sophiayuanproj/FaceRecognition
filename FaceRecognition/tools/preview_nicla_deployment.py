#!/usr/bin/env python3
"""Reproduce the deterministic Nicla release hashes without connecting a board."""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import importlib.util
import argparse
import json
from pathlib import Path
import tempfile

ROOT = Path(__file__).resolve().parents[1]
WORKSPACE = ROOT.parent
STUDY = WORKSPACE / "work/project-finish-v4"
DEPLOY_TOOL = ROOT / "tools/deploy_nicla_runtime.py"
SOURCE_PARTS = ("wearable_app.py", "identity_contracts.py", "runtime_support.py", "telemetry.py")
SOURCE_PATHS = {
    "wearable_app.py": ROOT / "firmware/main.py",
    "identity_contracts.py": ROOT / "firmware/identity_contracts.py",
    "runtime_support.py": ROOT / "firmware/runtime_support.py",
    "telemetry.py": ROOT / "firmware/telemetry.py",
}


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reuse-models-from", help="preview immutable model reuse from a release ID/root")
    parser.add_argument("--release-id", help="optional explicit release ID; defaults to the content-derived ID")
    parser.add_argument("--output", type=Path, help="write preview receipt here; defaults to the historical preview path")
    args = parser.parse_args()
    spec = importlib.util.spec_from_file_location("nicla_deploy_module", DEPLOY_TOOL)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load deployment tool")
    deploy = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(deploy)

    model_asset_root = deploy.normalize_release_root(args.reuse_models_from) if args.reuse_models_from else None
    model_names = deploy.MODEL_NAMES
    release_id = args.release_id or deploy.default_release_id(model_asset_root)
    release_root = "/flash/releases/" + release_id
    with tempfile.TemporaryDirectory(prefix="nicla-deployment-preview-") as temp_dir:
        payload = deploy.make_payload(release_root, temp_dir, model_asset_root)
        payload_rows = {name: {"sha256": _sha(path), "bytes": path.stat().st_size}
                        for name, path in payload.items()}
        config_row = payload_rows["runtime_config.json"]
        manifest_row = payload_rows["models/deployment-v1/integration_manifest.json"]
        device_firmware_digest = hashlib.sha256()
        source_rows = {}
        for name in deploy.FIRMWARE_PARTS:
            payload_path = payload[name]
            source_path = (SOURCE_PATHS.get(name) or ROOT / "firmware" / name)
            source_rows[name] = {"path": source_path.relative_to(WORKSPACE).as_posix(),
                "sha256": _sha(payload_path), "bytes": payload_path.stat().st_size}
            device_firmware_digest.update((name + "\t" + source_rows[name]["sha256"] + "\n").encode("utf-8"))
        expected_firmware = device_firmware_digest.hexdigest()
        config_doc = json.loads(payload["runtime_config.json"].read_text(encoding="utf-8"))
        manifest_doc = json.loads(payload["models/deployment-v1/integration_manifest.json"].read_text(encoding="utf-8"))
        if manifest_doc.get("runtime_config", {}).get("sha256") != config_row["sha256"]:
            raise ValueError("preview runtime config hash is not bound into the deployment manifest")
        receipt = {
            "schema": "nicla-deployment-preview-1",
            "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "status": "software_preview_only_not_deployed",
            "release_id": release_id,
            "release_root": release_root,
            "model_asset_root": model_asset_root or release_root,
            "release_id_derivation": "deploy_nicla_runtime.default_release_id() over sorted source and precompiled runtime hashes plus model asset root",
            "expected_firmware_sha256": expected_firmware,
            "expected_firmware_sources": source_rows,
            "expected_firmware_hash_contract": "SHA-256 over ordered FIRMWARE_PARTS rows: file_name<TAB>deployed_file_sha256<LF>",
            "source_runtime_config_sha256": _sha(ROOT / "firmware/runtime_config.json"),
            "expected_device_config_sha256": config_row["sha256"],
            "expected_device_integration_manifest_sha256": manifest_row["sha256"],
            "expected_device_config_release_paths": True,
            "expected_runtime_policy_ids": list(config_doc.get("policies", {})),
            "expected_measurement_stage_ids": config_doc.get("control", {}).get("measurement_stage_ids", []),
            "payload_files": payload_rows,
            "reused_model_asset_files": manifest_doc.get("external_model_files", {}),
            "physical_board_checked": False,
            "note": "Deployment script installs and verifies these payload hashes before reset. The board smoke must still compare live identity/config readback against the exact hashes above.",
        }
    out = args.output or STUDY / "handoffs/J/DEPLOYMENT_PREVIEW.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    tmp.replace(out)
    print(json.dumps(receipt, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
