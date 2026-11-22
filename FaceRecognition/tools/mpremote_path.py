"""Resolve the host-tool executable from PATH or a project-local instrument env."""
from __future__ import annotations
import shutil
from pathlib import Path

def resolve_mpremote(project_root: Path) -> Path:
    found = shutil.which("mpremote")
    if found:
        return Path(found)
    for candidate in (
        project_root.parent / ".venv-instrument/Scripts/mpremote.exe",
        project_root.parent / ".venv-instrument/bin/mpremote",
        project_root / ".venv-nicla/Scripts/mpremote.exe",
        project_root / ".venv-nicla/bin/mpremote",
        project_root / ".venv-nicla-windows/Scripts/mpremote.exe",
        project_root / ".venv-instrument/Scripts/mpremote.exe",
        project_root / ".venv-instrument/bin/mpremote",
    ):
        if candidate.is_file():
            return candidate
    return project_root.parent / ".venv-instrument/Scripts/mpremote.exe"
