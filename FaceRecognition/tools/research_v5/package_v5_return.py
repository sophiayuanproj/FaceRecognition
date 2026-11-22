#!/usr/bin/env python3
"""Create an immutable Windows-to-Mac v5 return ZIP with raw meter runs."""
from __future__ import annotations

import argparse
import json
import shutil
import tempfile
from datetime import datetime, timezone
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
APP_ROOT = HERE.parents[1]
TOOLS = APP_ROOT / "tools"
if str(TOOLS) not in sys.path:
    sys.path.insert(0, str(TOOLS))
from portable_transfer import create


OMIT_PARTS = {".git", ".venv", "venv", "downloads", "__pycache__", "node_modules", ".cache", "private", "participant", "templates", "backups"}
OMIT_SUFFIXES = {".pyc", ".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff", ".whl", ".exe", ".dll"}


def copy_filtered(source: Path, root: Path, stage: Path, *, allow_binary: bool) -> int:
    copied = 0
    if not source.exists():
        return copied
    for path in sorted(source.rglob("*")):
        rel = path.relative_to(root)
        if any(part.casefold() in OMIT_PARTS or part.casefold().startswith(".venv") for part in rel.parts):
            continue
        if path.is_symlink():
            raise ValueError("refusing symlink in return input: " + str(path))
        if not path.is_file() or path.suffix.casefold() in OMIT_SUFFIXES:
            continue
        if not allow_binary and path.suffix.casefold() not in {".py", ".ps1", ".md", ".json", ".jsonl", ".txt", ".csv", ".svg", ".log", ".ino"}:
            continue
        target = stage / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target)
        copied += 1
    return copied


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--destination", type=Path, required=True)
    args = parser.parse_args(argv)
    workspace = args.workspace.resolve()
    destination = args.destination.resolve()
    if destination.exists():
        raise SystemExit("refusing to overwrite return archive: " + str(destination))
    research = workspace / "work/project-research-v5"
    with tempfile.TemporaryDirectory(prefix="nicla-v5-return-stage-") as temp:
        stage = Path(temp)
        source_count = copy_filtered(workspace / "FaceRecognition", workspace, stage, allow_binary=False)
        windows = research / "windows"
        raw_count = copy_filtered(windows, workspace, stage, allow_binary=True)
        for name in ("READY_FOR_WINDOWS.json", "STATE.json", "STATUS.md", "BRANCH_LEDGER.json", "CLAIM_EVIDENCE.json", "EXPERIMENT_REGISTRY.jsonl", "DECISIONS.md", "PLAN.md", "ARTIFACT_INVENTORY.json", "MAC_FINDINGS.md", "FINAL_BUILD_GAPS.md"):
            source = research / name
            if source.is_file():
                target = stage / source.relative_to(workspace)
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source, target)
        stage_root = stage / "RETURN_IMPORT.md"
        stage_root.write_text("""# Windows research return import\n\n1. Keep the return ZIP immutable and verify it with `python FaceRecognition/tools/research_v5/portable_transfer.py verify <archive>`.\n2. Extract it into an empty path and run `verify-extracted`.\n3. Compare every returned source patch against the newer Mac checkout before applying it. Preserve all raw accepted, rejected, failed, and interrupted meter runs under `work/project-research-v5/windows/`.\n4. Keep Windows COM identifiers, meter raw packets, and build outputs in the private return workspace. Do not merge stale state over the current Mac evidence.\n5. Run the `03_MAC_RESEARCH_SYNTHESIS.md` entry point in the full camera-first prompt pack.\n\nNo returned result by itself opens future-final data or qualifies recognition, Nicla power, or a final build.\n""", encoding="utf-8")
        metadata = {
            "schema": "nicla-v5-windows-return-1",
            "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "scope": "Windows v5 instrument commissioning, bounded raw FNIRSI/Nano logs, QC, annotations, and source patches.",
            "source_files": source_count,
            "windows_files": raw_count,
            "raw_runs_are_immutable": True,
            "no_private_mac_data": True,
            "no_future_final_data": True,
        }
        (stage / "RETURN_METADATA.json").write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        roots = ["FaceRecognition", "work/project-research-v5", "RETURN_IMPORT.md", "RETURN_METADATA.json"]
        result = create(stage, destination, roots)
    print(json.dumps({"status": "pass", "archive": str(destination), "sha256": result["archive"]["sha256"],
                      "files": result["verification"]["files_checked"], "bytes": result["verification"]["bytes_checked"]}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
