"""Content-addressed artifact manifests and verification."""
from __future__ import annotations
from pathlib import Path, PurePosixPath
from .fs import sha256_file


def build_artifact_manifest(workspace: Path, relative_paths: list[str]) -> dict:
    entries = []
    seen = set()
    for rel_text in sorted(relative_paths):
        rel = PurePosixPath(rel_text)
        if rel.is_absolute() or ".." in rel.parts:
            raise ValueError("artifact paths must remain relative to workspace: " + rel_text)
        if rel_text in seen:
            raise ValueError("duplicate artifact path: " + rel_text)
        seen.add(rel_text)
        path = workspace.joinpath(*rel.parts)
        if not path.is_file():
            raise FileNotFoundError(str(path))
        entries.append({"path": rel.as_posix(), "bytes": path.stat().st_size, "sha256": sha256_file(path)})
    return {"schema": "nicla-artifact-manifest-1", "self_hash_excluded": True, "files": entries}


def verify_artifact_manifest(workspace: Path, manifest: dict) -> dict:
    errors = []
    entries = manifest.get("files", [])
    if manifest.get("schema") != "nicla-artifact-manifest-1" or not isinstance(entries, list):
        return {"status": "failed", "file_count": 0, "errors": ["unsupported artifact manifest"]}
    seen = set()
    for row in entries:
        rel_text = row.get("path")
        rel = PurePosixPath(rel_text or "")
        if rel.is_absolute() or ".." in rel.parts:
            errors.append("unsafe manifest path: " + str(rel_text))
            continue
        if rel_text in seen:
            errors.append("duplicate path: " + str(rel_text))
            continue
        seen.add(rel_text)
        path = workspace.joinpath(*rel.parts)
        if not path.is_file():
            errors.append("missing: " + str(rel_text))
        elif path.stat().st_size != row.get("bytes"):
            errors.append("size mismatch: " + str(rel_text))
        elif sha256_file(path) != row.get("sha256"):
            errors.append("hash mismatch: " + str(rel_text))
    return {"status": "pass" if not errors else "failed", "file_count": len(entries), "errors": errors}
