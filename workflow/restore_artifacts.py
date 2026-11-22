"""Restore optional recorded model artifacts from the hash-pinned GitHub release."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import shutil
import stat
from tempfile import TemporaryDirectory
import urllib.request
import zipfile


ROOT = Path(__file__).resolve().parents[1]
MANIFEST = "RESEARCH_ASSET_MANIFEST.json"
PREFIXES = ("work/project-finish-v4/runs/", "work/project-finish-v4/models/pretrained/",
            "work/project-finish-v4/models/keras_cache/models/")


def digest_file(path):
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def safe_path(name):
    path = PurePosixPath(name)
    if (not name or path.is_absolute() or "\\" in name or ":" in name or
            ".." in path.parts or not name.startswith(PREFIXES)):
        raise ValueError("Unsafe artifact path: " + name)
    return path


def verify_archive(archive, metadata):
    if archive.stat().st_size != metadata["bytes"] or digest_file(archive) != metadata["sha256"]:
        raise ValueError("Artifact archive size or SHA-256 does not match the repository receipt")
    with zipfile.ZipFile(archive) as source:
        names = source.namelist()
        if len(names) != len(set(names)) or MANIFEST not in names:
            raise ValueError("Archive manifest is absent or entries are duplicated")
        encoded = source.read(MANIFEST)
        if hashlib.sha256(encoded).hexdigest() != metadata["manifest_sha256"]:
            raise ValueError("Artifact manifest SHA-256 mismatch")
        manifest = json.loads(encoded)
        rows = manifest["files"]
        if len(rows) != metadata["files"] or set(names) != {row["path"] for row in rows} | {MANIFEST}:
            raise ValueError("Artifact archive does not match its complete file manifest")
        for row in rows:
            safe_path(row["path"])
            info = source.getinfo(row["path"])
            if stat.S_ISLNK(info.external_attr >> 16) or info.file_size != row["bytes"]:
                raise ValueError("Invalid artifact entry: " + row["path"])
            value = hashlib.sha256()
            with source.open(info) as stream:
                for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                    value.update(chunk)
            if value.hexdigest() != row["sha256"]:
                raise ValueError("Artifact content SHA-256 mismatch: " + row["path"])
    return rows


def restore(archive, rows):
    targets = []
    for row in rows:
        target = ROOT.joinpath(*safe_path(row["path"]).parts)
        if not target.resolve().is_relative_to(ROOT):
            raise ValueError("Artifact destination leaves the repository")
        if target.exists() and (not target.is_file() or digest_file(target) != row["sha256"]):
            raise ValueError("Existing local artifact differs. Move it aside before restoring: " + row["path"])
        targets.append((row, target))
    restored = 0
    with zipfile.ZipFile(archive) as source:
        for row, target in targets:
            if target.exists():
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            partial = target.with_name(target.name + ".partial")
            with source.open(row["path"]) as src, partial.open("wb") as dst:
                shutil.copyfileobj(src, dst, 1024 * 1024)
            partial.replace(target)
            restored += 1
    return restored


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, help="Use a previously downloaded release ZIP")
    parser.add_argument("--check", action="store_true", help="Verify the download without restoring files")
    args = parser.parse_args()
    metadata = json.loads((ROOT / "workflow/research-assets.json").read_text())
    with TemporaryDirectory(prefix="nicla-research-assets-") as temporary:
        archive = args.archive
        if archive is None:
            archive = Path(temporary) / metadata["name"]
            print("Downloading optional historical models from", metadata["url"], flush=True)
            with urllib.request.urlopen(metadata["url"], timeout=60) as src, archive.open("wb") as dst:
                shutil.copyfileobj(src, dst, 1024 * 1024)
        rows = verify_archive(archive, metadata)
        restored = 0 if args.check else restore(archive, rows)
        print(json.dumps({"status": "verified", "files_checked": len(rows),
                          "files_restored": restored, "archive_sha256": metadata["sha256"]}, indent=2))


if __name__ == "__main__":
    main()
