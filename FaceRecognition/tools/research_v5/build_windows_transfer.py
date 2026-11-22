#!/usr/bin/env python3
"""Assemble the privacy-filtered, hash-closed v5 Windows research transfer."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import sys
import tempfile
from pathlib import Path

APP = Path(__file__).resolve().parents[2]
WORKSPACE = APP.parent
TOOLS = APP / "tools"
if str(APP) not in sys.path:
    sys.path.insert(0, str(APP))
if str(TOOLS) not in sys.path:
    sys.path.insert(0, str(TOOLS))
from nicla_study.v5_readiness import build as build_readiness
from portable_transfer import create


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
PRIVATE_PARTS = {".git", ".venv", "venv", "__pycache__", "node_modules", ".cache", "cache", "private", "downloads", "build", "raw", "lfw_source", "chokepoint_development", "participant", "templates", "backups"}
TEXT_SUFFIXES = {".py", ".ps1", ".cmd", ".md", ".txt", ".json", ".jsonl", ".log", ".csv", ".ino", ".yml", ".yaml", ".toml", ".xml", ".html", ".css", ".js", ".svg", ".c", ".h"}


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def anon(value: str) -> str:
    return "person-" + hashlib.sha256(("nicla-v5-transfer-v1:" + value).encode("utf-8")).hexdigest()[:20]


def identity_names() -> list[str]:
    path = WORKSPACE / "work/project-research-v5/data/manifests/lfw_roles.jsonl"
    values = set()
    if path.is_file():
        with path.open(encoding="utf-8") as stream:
            for line in stream:
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                name = row.get("source_identity")
                if isinstance(name, str) and name:
                    values.add(name)
    return sorted(values, key=len, reverse=True)


NAMES: list[str] = []
IDENTITY_PATTERN: re.Pattern[str] | None = None


def scrub_text(value: str) -> str:
    if IDENTITY_PATTERN is not None:
        value = IDENTITY_PATTERN.sub(lambda match: anon(match.group(0)), value)
    value = re.sub(r"/Users/[^\s\"'<>]+", "[Mac-local-path-omitted]", value)
    value = re.sub(r"/dev/cu\.usbmodem[\w.-]*", "[local-serial-port-omitted]", value)
    value = re.sub(r"(?:[A-Za-z]:)?[/\\][^\s\"']*(?:lfw_source|chokepoint_development|optical-captures|original_romfs)[^\s\"']*", "[private-data-path-omitted]", value, flags=re.I)
    return value


def scrub_json(value):
    if isinstance(value, dict):
        out = {}
        for key, item in value.items():
            if isinstance(item, str) and key.casefold() in {"source_identity", "identity_name", "person_name", "subject_name"}:
                # Keep only exact roster-class labels (for example K01). A
                # prefixed identifier such as K01_George_W_Bush still carries
                # the source identity and must have its name component scrubbed.
                out[key] = item if re.fullmatch(r"K\d{2}", item) else scrub_text(item)
            else:
                out[key] = scrub_json(item)
        return out
    if isinstance(value, list):
        return [scrub_json(item) for item in value]
    if isinstance(value, str):
        return scrub_text(value)
    return value


def strict_json_loads(text: str):
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ValueError("duplicate JSON key: " + key)
            result[key] = value
        return result
    def nonfinite(token):
        raise ValueError("nonfinite JSON number: " + token)
    return json.loads(text, object_pairs_hook=pairs, parse_constant=nonfinite)


def copy_one(source: Path, rel: str, stage: Path, *, transform=True) -> None:
    if not source.is_file() or source.is_symlink():
        raise FileNotFoundError(source)
    target = stage / rel
    target.parent.mkdir(parents=True, exist_ok=True)
    if not transform or source.suffix.casefold() not in TEXT_SUFFIXES:
        shutil.copy2(source, target)
        return
    text = source.read_text(encoding="utf-8", errors="strict")
    if source.suffix.casefold() in {".json", ".jsonl"}:
        if source.suffix.casefold() == ".jsonl":
            rows = [strict_json_loads(line) for line in text.splitlines() if line.strip()]
            encoded = "".join(json.dumps(scrub_json(row), sort_keys=True, allow_nan=False) + "\n" for row in rows)
        else:
            encoded = json.dumps(scrub_json(strict_json_loads(text)), indent=2, sort_keys=True, allow_nan=False) + "\n"
        target.write_text(encoded, encoding="utf-8")
        return
    target.write_text(scrub_text(text), encoding="utf-8", newline="\n")


def copy_tree(source_rel: str, stage: Path, predicate, *, transform=True) -> int:
    source = WORKSPACE / source_rel
    if not source.is_dir():
        return 0
    count = 0
    for path in sorted(source.rglob("*")):
        rel = path.relative_to(WORKSPACE).as_posix()
        parts = {part.casefold() for part in Path(rel).parts}
        if parts & PRIVATE_PARTS or any(part.startswith(".venv") for part in parts):
            continue
        if path.is_symlink():
            raise ValueError("refusing symlink: " + rel)
        if not path.is_file() or path.name == ".DS_Store" or path.suffix.casefold() == ".pyc":
            continue
        if not predicate(path, rel):
            continue
        copy_one(path, rel, stage, transform=transform)
        count += 1
    return count


def copy_face_recognition(stage: Path) -> int:
    exact = {
        "FaceRecognition/README.md", "FaceRecognition/.gitignore",
        "FaceRecognition/requirements-host.txt", "FaceRecognition/requirements-host.lock",
        "FaceRecognition/requirements-live-control-center-windows.lock",
        "FaceRecognition/requirements-instrument-windows.txt", "FaceRecognition/requirements-instrument-windows.lock",
        "FaceRecognition/requirements-nicla-board-smoke.txt", "FaceRecognition/config/firmware-release.json",
        "FaceRecognition/models/README.md",
    }
    count = 0
    for rel in sorted(exact):
        path = WORKSPACE / rel
        if path.is_file():
            copy_one(path, rel, stage)
            count += 1
    roots = [
        "FaceRecognition/bin", "FaceRecognition/control_center", "FaceRecognition/docs", "FaceRecognition/firmware",
        "FaceRecognition/host", "FaceRecognition/nicla_study", "FaceRecognition/romfs",
        "FaceRecognition/tools/research_v5", "FaceRecognition/tools/windows",
        "FaceRecognition/tools/windows_v5",
    ]
    for rel in roots:
        count += copy_tree(rel, stage, lambda p, r: p.suffix.casefold() in TEXT_SUFFIXES or p.suffix.casefold() in {".tflite", ".pb", ".bin", ".u8"})
    # The new Windows research packet carries the actual experimental device
    # release assets. The older transfer intentionally omitted these because
    # it was an instrument-only archive; preserve that archive unchanged.
    count += copy_tree(
        "FaceRecognition/models/deployment-v1", stage,
        lambda p, r: p.suffix.casefold() in {".tflite", ".txt", ".json"},
    )
    for rel in ("FaceRecognition/tools/portable_transfer.py", "FaceRecognition/tools/deploy_nicla_runtime.py", "FaceRecognition/tools/restore_nicla_runtime.py"):
        path = WORKSPACE / rel
        if path.is_file():
            copy_one(path, rel, stage)
            count += 1
    return count


def copy_research(stage: Path) -> int:
    research = WORKSPACE / "work/project-research-v5"
    count = 0
    root_files = [
        "BRANCH_LEDGER.json", "CLAIM_EVIDENCE.json", "COMMANDS.jsonl", "DECISIONS.md",
        "EXPERIMENT_REGISTRY.jsonl", "HANDOFF_CHECKPOINT.json", "HYPOTHESES.jsonl",
        "LUNA_CONTINUE_CAMERA_RESEARCH.md", "PLAN.md", "STATE.json", "STATUS.md",
        "READY_FOR_WINDOWS.json", "START_HERE.md", "FOR_AI_REVIEWER.md", "MAC_FINDINGS.md",
        "ARTIFACT_INVENTORY.json", "FINAL_BUILD_GAPS.md", "DATA_DISTRIBUTION.md",
        "02_WINDOWS_RESEARCH_AND_RETURN.md", "03_MAC_RESEARCH_SYNTHESIS.md", "handoffs/TRANSFER_QA.json",
        "RESULTS_AT_A_GLANCE.md", "DEMO_RUNBOOK.md", "GUI_ACCEPTANCE.md", "DEVICE_INTEGRATION.md",
        "WINDOWS_PACKET_README.md", "PROPOSAL_ALIGNMENT.md", "OBLIGATION_MATRIX_V1.json",
        "CONTROL_CENTER_EVENT_CONTRACT_V1.json",
    ]
    for name in root_files:
        path = research / name
        if path.is_file():
            copy_one(path, path.relative_to(WORKSPACE).as_posix(), stage)
            count += 1
    count += copy_tree("work/project-research-v5/config", stage, lambda p, r: p.suffix.casefold() in TEXT_SUFFIXES)
    # This tree was already pseudonymized and its row hashes are bound by
    # PREDICTIONS_MANIFEST.json. Preserve it byte-for-byte; re-scrubbing would
    # pseudonymize source_identity a second time and invalidate those receipts.
    count += copy_tree("work/project-research-v5/distribution_evidence", stage, lambda p, r: p.suffix.casefold() in TEXT_SUFFIXES, transform=False)
    count += copy_tree("work/project-research-v5/handoffs/windows_wheels", stage, lambda p, r: p.suffix.casefold() == ".whl" or p.suffix.casefold() == ".log")
    count += copy_tree("work/project-research-v5/models/pretrained/mobilefacenet_tf", stage, lambda p, r: p.name == "LICENSE" or p.suffix.casefold() in TEXT_SUFFIXES or p.suffix.casefold() == ".pb")
    count += copy_tree("work/project-research-v5/reports", stage, lambda p, r: (p.suffix.casefold() in {".json", ".jsonl", ".md", ".log", ".py", ".txt"}) or ("rgb48_encoder_parity" in r and p.suffix.casefold() in {".bin", ".npy"}))
    count += copy_tree("work/project-research-v5/runs", stage, lambda p, r: (
        "runs/optical/" not in r
        and p.name not in {"CAPTURE_FEATURES.json", "CAPTURE_FEATURE_INDEX.jsonl"}
        and (p.suffix.casefold() in {".json", ".jsonl", ".tflite", ".pb"}
             or ("runs/camera_adapt_task48/" in r and p.name in {"weights.npy", "bias.npy"}))
        and not (r.endswith("/PREDICTIONS.jsonl") and "runs/camera_adapt_task48/" in r)
    ))
    # Current technical acceptance is compactly indexed at the top level;
    # include the reproducible accuracy matrix/leaderboard as well.
    for name in ("ACCURACY_SEARCH_MATRIX.json", "ACCURACY_LEADERBOARD.md"):
        path = research / "runs" / "accuracy_search_v1" / name
        if path.is_file():
            copy_one(path, path.relative_to(WORKSPACE).as_posix(), stage)
            count += 1
    count += copy_tree("work/project-research-v5/data/native_fixtures", stage, lambda p, r: p.suffix.casefold() in {".u8", ".json", ".jsonl", ".md"})
    count += copy_tree("work/project-research-v5/data/native_fixtures_rgb", stage, lambda p, r: p.suffix.casefold() in {".u8", ".json", ".jsonl", ".md"})
    for name in ("DOWNLOADS.json", "OPTICAL_ADAPTATION_FREEZE.json"):
        path = research / "data" / name
        if path.is_file():
            copy_one(path, path.relative_to(WORKSPACE).as_posix(), stage)
            count += 1
    count += copy_tree("work/project-research-v5/staging/romfs-fused48", stage, lambda p, r: p.name == "romfs0.img" or p.suffix.casefold() in {".json", ".md"}, transform=False)
    count += copy_tree("work/project-research-v5/source_snapshots", stage, lambda p, r: p.suffix.casefold() in TEXT_SUFFIXES or p.suffix.casefold() in {".c", ".h"})
    for rel in (
        "work/project-research-v5/instrument/nano_meter_diagnostic/nano_meter_diagnostic.ino",
        "work/project-research-v5/instrument/PROVENANCE.json",
        "work/project-research-v5/instrument/COMMISSIONING_CONFIG_TEMPLATE.json",
    ):
        path = WORKSPACE / rel
        if path.is_file():
            copy_one(path, rel, stage)
            count += 1
    # Shareable training predictions are pseudonymous derivatives. Source rows stay local.
    return count


def write_inventory(stage: Path) -> None:
    root = stage / "work/project-research-v5"
    classes = {
        "models": [], "configurations": [], "data_and_roles": [], "results": [], "windows_runtime": [],
    }
    for path in sorted(stage.rglob("*")):
        if not path.is_file() or path.name == "ARTIFACT_INVENTORY.json":
            continue
        rel = path.relative_to(stage).as_posix()
        row = {"path": rel, "bytes": path.stat().st_size, "sha256": sha(path)}
        if path.suffix.casefold() in {".tflite", ".pb"} or path.name in {"weights.npy", "bias.npy"} or rel.endswith("romfs0.img"):
            classes["models"].append(row)
        if "/config/" in rel or rel.endswith("PROTOCOL.json") or rel.endswith("USEFULNESS_GATES.json"):
            classes["configurations"].append(row)
        if "/distribution_evidence/" in rel or rel.endswith("native_fixtures/manifest.json") or rel.endswith("native_fixtures_rgb/manifest.json"):
            classes["data_and_roles"].append(row)
        if "/reports/" in rel or "/runs/" in rel:
            classes["results"].append(row)
        if rel.endswith("/handoffs/TRANSFER_QA.json"):
            classes["results"].append(row)
        if rel.startswith("work/project-research-v5/handoffs/windows_wheels/") or rel.endswith("requirements-instrument-windows.lock"):
            classes["windows_runtime"].append(row)
    inventory = {"schema": "nicla-v5-transfer-inventory-1", "qualification": "research artifacts only; face imagery and private frames excluded", **classes}
    (root / "ARTIFACT_INVENTORY.json").write_text(json.dumps(inventory, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def privacy_scan(stage: Path) -> None:
    banned_path_parts = {"private", "downloads", "lfw_source", "chokepoint_development", "participant", "backups", ".venv", ".git"}
    failures = []
    for path in stage.rglob("*"):
        if not path.is_file():
            continue
        rel = path.relative_to(stage).as_posix()
        parts = {x.casefold() for x in Path(rel).parts}
        if parts & banned_path_parts or path.suffix.casefold() in IMAGE_SUFFIXES:
            failures.append("forbidden package file: " + rel)
            continue
        if path.suffix.casefold() in TEXT_SUFFIXES:
            text = path.read_text(encoding="utf-8", errors="replace")
            if "/Users/omristeinberg" in text or "/dev/cu.usbmodem" in text:
                failures.append("host-specific path remains: " + rel)
            if IDENTITY_PATTERN is not None and IDENTITY_PATTERN.search(text):
                failures.append("source identity text remains: " + rel)
    if failures:
        raise ValueError("privacy/path scan failed: " + "; ".join(failures[:12]))


def main(argv=None) -> int:
    global NAMES, IDENTITY_PATTERN
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", type=Path, default=WORKSPACE)
    parser.add_argument("--archive", type=Path, required=True)
    args = parser.parse_args(argv)
    root = args.workspace.resolve()
    if root != WORKSPACE.resolve():
        raise SystemExit("run the builder from the canonical Mac workspace")
    destination = args.archive.resolve()
    if destination.exists():
        raise SystemExit("refusing to overwrite transfer archive: " + str(destination))
    NAMES = identity_names()
    # A single longest-first pass keeps large prediction JSONL files practical;
    # repeatedly scanning each row once per source identity grows quadratically.
    IDENTITY_PATTERN = re.compile(
        r"(?<![A-Za-z0-9])(?:" + "|".join(re.escape(name) for name in NAMES) + r")(?![A-Za-z0-9])"
    ) if NAMES else None
    with tempfile.TemporaryDirectory(prefix="nicla-v5-transfer-Ω ") as temp:
        stage = Path(temp)
        copy_face_recognition(stage)
        copy_research(stage)
        # The full original prompt pack is retained, with the updated single Windows and Mac-return launchers.
        copy_tree("work/camera-first-research-pack-2026-10-31", stage, lambda p, r: p.suffix.casefold() in TEXT_SUFFIXES)
        readiness = build_readiness(stage, stage / "work/project-research-v5/READY_FOR_WINDOWS.json")
        if not readiness["ready_for_windows"] or readiness["recognition_qualified"]:
            raise ValueError("unexpected readiness gate state")
        # Inventory hashes the final verifier-derived readiness record. It is
        # generated after READY_FOR_WINDOWS because that record is rewritten
        # from the staged, relocatable evidence tree above.
        write_inventory(stage)
        privacy_scan(stage)
        result = create(stage, destination, ["FaceRecognition", "work/camera-first-research-pack-2026-10-31", "work/project-research-v5"])
    print(json.dumps({"status": "pass", "archive": str(destination), "sha256": result["archive"]["sha256"],
                      "bytes": result["archive"]["bytes"], "files": result["verification"]["files_checked"],
                      "uncompressed_bytes": result["verification"]["bytes_checked"],
                      "readiness": readiness["status"], "recognition_qualified": readiness["recognition_qualified"]}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
