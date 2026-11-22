#!/usr/bin/env python3
"""Package Windows measurement evidence without mutating source runs."""
from __future__ import annotations
import argparse, json, shutil, tempfile
from datetime import datetime, timezone
from pathlib import Path
from portable_transfer import create

EXCLUDE_DIRS={".git",".venv","venv","build","downloads","__pycache__","node_modules",".cache","cache","private","participant","templates","backups","pretrained","keras_cache"}
EXCLUDE_FILES={"setup-status.json"}
def copy_tree(src:Path,root:Path,stage:Path):
    for path in sorted(src.rglob("*")):
        rel=path.relative_to(root)
        if any(part.lower() in EXCLUDE_DIRS or part.lower().startswith(".venv") for part in rel.parts):continue
        if path.is_symlink():raise ValueError("return input contains a symlink: "+str(path))
        if not path.is_file() or path.name in EXCLUDE_FILES or path.suffix==".pyc":continue
        target=stage/rel;target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(path,target)
def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("--workspace",type=Path,required=True);p.add_argument("--destination",type=Path,required=True);a=p.parse_args(argv)
    workspace=a.workspace.resolve();dest=a.destination.resolve()
    if dest.exists():raise SystemExit("refusing to overwrite return archive: "+str(dest))
    study=workspace/"work/project-finish-v4"
    roots=[workspace/"FaceRecognition",study/"config",study/"staging/I",study/"models/C",study/"models/host_reference"]
    for area in ("G","H","I","J","K"):
        for kind in ("handoffs","reports","runs"):
            q=study/kind/area
            if q.exists():roots.append(q)
    with tempfile.TemporaryDirectory(prefix="nicla-return-stage-") as td:
        stage=Path(td)
        for name in ("CURRENT_STATUS.md","RUN_STATE.json","RUN_EVENTS.jsonl","CLAIM_EVIDENCE_LEDGER.json"):
            q=study/name
            if q.is_file():shutil.copy2(q,stage/"work/project-finish-v4"/name)
        for q in roots:copy_tree(q,workspace,stage)
        meta={"schema":"nicla-windows-return-1","created_utc":datetime.now(timezone.utc).isoformat(timespec="seconds"),"source_workspace_relative":".","scope":"G/H/I/J/K receipts, all accepted/rejected/interrupted raw runs, source patches, current state and selected study artifacts.","runs_are_immutable":"The packager copies raw runs and does not rewrite them.","import_instruction":"Verify TRANSFER_MANIFEST.json and reconcile source changes before applying; do not overwrite newer Mac files."}
        (stage/"RETURN_IMPORT.md").write_text("# Windows return import\n\nVerify this ZIP with `FaceRecognition/tools/portable_transfer.py verify <archive>`. It contains all files under the Windows G/H/I/J/K run and receipt directories, including failed and interrupted runs, plus the returned source and the current state. Reconcile source patches against the newer Mac tree before applying them.\n",encoding="utf-8")
        (stage/"RETURN_METADATA.json").write_text(json.dumps(meta,indent=2)+"\n",encoding="utf-8")
        result=create(stage,dest,["FaceRecognition","work/project-finish-v4","RETURN_IMPORT.md","RETURN_METADATA.json"])
    print(json.dumps({"status":"pass","archive":str(dest),"sha256":result["archive"]["sha256"],"files":result["verification"]["files_checked"],"bytes":result["verification"]["bytes_checked"]},indent=2));return 0
if __name__=="__main__":raise SystemExit(main())
