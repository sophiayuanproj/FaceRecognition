"""Resolve portable workspace-relative paths for Mac and transferred Windows roots."""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any


class PathResolutionError(RuntimeError):
    pass


def resolve_study_root(workspace: Path | str, requested: Path | str | None = None) -> Path:
    """Select a versioned study without changing the legacy v4 default."""
    root = Path(workspace).resolve()
    selected = requested if requested is not None else os.environ.get(
        "NICLA_STUDY_ROOT", "work/project-finish-v4")
    path = Path(selected)
    path = (root / path).resolve() if not path.is_absolute() else path.resolve()
    try:
        path.relative_to(root)
    except ValueError as error:
        raise PathResolutionError("study root escapes the selected workspace") from error
    return path


def resolve_registry(registry: dict, *, workspace: Path | str | None = None,
                     require_existing: bool = False) -> dict:
    platform_name = os.name
    if workspace is None:
        if platform_name == "nt":
            raise PathResolutionError("Windows requires --workspace for the transferred project root")
        local = registry.get("workspace", {}).get("mac_local_absolute")
        if not local:
            raise PathResolutionError("no local workspace root is registered")
        workspace_path = Path(local).expanduser().resolve()
    else:
        workspace_path = Path(workspace).expanduser().resolve()
    resolved = {}
    for name, entry in registry.get("paths", {}).items():
        relative = entry.get("relative")
        if not isinstance(relative, str):
            resolved[name] = {"path": None, "status": entry.get("status", "unresolved")}
            continue
        relative_path = Path(relative)
        if relative_path.is_absolute() or ".." in relative_path.parts:
            raise PathResolutionError("registry path escapes workspace: " + name)
        # Keep the lexical path. In particular, a venv interpreter may be a
        # symlink to a system interpreter outside the workspace; resolving it
        # here would lose the portable path that should be used on Windows.
        candidate = Path(os.path.abspath(workspace_path / relative_path))
        exists = candidate.exists()
        if require_existing and not exists:
            raise PathResolutionError("required path is missing: %s (%s)" % (name, candidate))
        resolved[name] = {"path": str(candidate), "relative": relative, "exists": exists}
    if platform_name == "nt":
        relative = registry.get("windows", {}).get("instrument_interpreter_relative")
        if relative:
            relative_path = Path(relative)
            if relative_path.is_absolute() or ".." in relative_path.parts:
                raise PathResolutionError("Windows instrument interpreter escapes workspace")
            candidate = Path(os.path.abspath(workspace_path / relative_path))
            resolved["instrument_interpreter"] = {"path": str(candidate), "relative": relative,
                                                  "exists": candidate.is_file()}
    return {"schema": "resolved-path-registry-1", "platform": os.name,
            "workspace": str(workspace_path), "paths": resolved}


def registry_path(workspace: Path, study_root: Path) -> Path:
    return study_root / "config" / "paths.json"
