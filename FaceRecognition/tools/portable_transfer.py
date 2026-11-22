#!/usr/bin/env python3
"""Create, verify, and safely extract immutable workspace transfer ZIPs."""
from __future__ import annotations
import argparse, hashlib, json, random, shutil, stat, tempfile, zipfile
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
MANIFEST="TRANSFER_MANIFEST.json"; SCHEMA="codex-workspace-transfer-1"
FORBIDDEN_PARTS={".git",".venv","venv","__pycache__","node_modules",".cache","cache","downloads","private","participant","templates"}
FORBIDDEN_PREFIXES=("work/project-finish-v4/deployments/backups/","work/project-finish-v4/models/pretrained/","work/project-finish-v4/models/keras_cache/")
def sha(path):
 h=hashlib.sha256()
 with Path(path).open("rb") as f:
  for b in iter(lambda:f.read(1024*1024),b""):h.update(b)
 return h.hexdigest()
def safe_name(name):
 n=name.replace("\\","/"); p=PurePosixPath(n)
 reserved={"CON","PRN","AUX","NUL"}|{"COM%d"%i for i in range(1,10)}|{"LPT%d"%i for i in range(1,10)}
 if not n or p.is_absolute() or ":" in n or any(x in {"",".",".."} for x in p.parts):raise ValueError("unsafe archive path: "+name)
 for part in p.parts:
  stem=part.split(".",1)[0].upper()
  if any(ch in part for ch in '<>"|?*') or part.endswith(("."," ")) or stem in reserved:raise ValueError("non-portable Windows path: "+name)
 return p.as_posix()
def reject_path(rel):
 parts={x.lower() for x in PurePosixPath(rel).parts}
 if parts&FORBIDDEN_PARTS or any(x.startswith(".venv") for x in parts) or rel.lower().startswith(FORBIDDEN_PREFIXES):raise ValueError("protected/cache path is not packageable: "+rel)
def walk_inputs(root,requested):
 root=Path(root).resolve(); rows=[]; seen=set()
 for item in requested:
  rel=safe_name(item); src=root.joinpath(*PurePosixPath(rel).parts)
  if src.is_symlink():raise ValueError("refusing symlink: "+rel)
  if not src.exists():raise FileNotFoundError(str(src))
  for path in ([src] if src.is_file() else sorted(src.rglob("*"))):
   if path.is_symlink():raise ValueError("refusing symlink: "+str(path))
   if not path.is_file():continue
   arc=path.relative_to(root).as_posix(); safe_name(arc); reject_path(arc)
   if arc==MANIFEST:raise ValueError(MANIFEST+" is reserved")
   if arc not in seen:rows.append((arc,path));seen.add(arc)
 if not rows:raise ValueError("no files selected")
 return sorted(rows)
def create(root,destination,requested):
 root=Path(root).resolve(); dst=Path(destination).resolve()
 if dst.exists():raise FileExistsError("refusing to overwrite archive: "+str(dst))
 rows=walk_inputs(root,requested); manifest={"schema":SCHEMA,"created_utc":datetime.now(timezone.utc).isoformat(timespec="seconds"),"files":[{"path":n,"bytes":p.stat().st_size,"sha256":sha(p)} for n,p in rows]}; encoded=(json.dumps(manifest,indent=2,sort_keys=True)+"\n").encode()
 dst.parent.mkdir(parents=True,exist_ok=True)
 try:
  with zipfile.ZipFile(dst,"x",compression=zipfile.ZIP_DEFLATED,compresslevel=6,allowZip64=True) as z:
   for n,p in rows:z.write(p,n)
   z.writestr(MANIFEST,encoded)
 except Exception:dst.unlink(missing_ok=True);raise
 result=verify(dst); side={"status":"pass",**manifest,"archive":{"path":dst.name,"bytes":dst.stat().st_size,"sha256":sha(dst)},"embedded_manifest_sha256":hashlib.sha256(encoded).hexdigest(),"verification":result}
 dst.with_suffix(dst.suffix+".manifest.json").write_text(json.dumps(side,indent=2,sort_keys=True)+"\n",encoding="utf-8");dst.with_suffix(dst.suffix+".sha256").write_text(side["archive"]["sha256"]+"  "+dst.name+"\n",encoding="utf-8");return side
def _zip_index(z):
 infos={};casefolded={}
 for info in z.infolist():
  n=safe_name(info.filename);mode=(info.external_attr>>16)&0xffff
  if stat.S_ISLNK(mode):raise ValueError("symlink entry refused: "+n)
  if n in infos:raise ValueError("duplicate archive path: "+n)
  folded=n.casefold()
  if folded in casefolded:raise ValueError("case-insensitive duplicate archive path: "+n)
  for prior in infos:
   p_fold=prior.casefold();n_fold=n.casefold()
   if p_fold.startswith(n_fold+"/") or n_fold.startswith(p_fold+"/"):raise ValueError("file/directory path collision: "+n)
  casefolded[folded]=n
  if n!=MANIFEST:reject_path(n)
  infos[n]=info
 if MANIFEST not in infos:raise ValueError("archive has no embedded manifest")
 m=json.loads(z.read(MANIFEST).decode("utf-8"))
 if m.get("schema")!=SCHEMA or not isinstance(m.get("files"),list):raise ValueError("invalid embedded manifest schema")
 expected={}
 for r in m["files"]:
  n=safe_name(r.get("path",""));reject_path(n)
  if n in expected:raise ValueError("duplicate manifest entry: "+n)
  if not isinstance(r.get("bytes"),int) or r["bytes"]<0 or len(r.get("sha256",""))!=64:raise ValueError("invalid manifest metadata: "+n)
  expected[n]=r
 actual=set(infos)-{MANIFEST}
 if actual!=set(expected):raise ValueError("archive/manifest closure mismatch")
 return m,{n:infos[n] for n in expected},expected
def verify(archive):
 with zipfile.ZipFile(archive,"r") as z:
  m,infos,expected=_zip_index(z); total=0
  for n,info in infos.items():
   h=hashlib.sha256();size=0
   with z.open(info) as f:
    for b in iter(lambda:f.read(1024*1024),b""):h.update(b);size+=len(b)
   row=expected[n]
   if size!=row["bytes"] or h.hexdigest()!=row["sha256"]:raise ValueError("size/hash mismatch: "+n)
   total+=size
 return {"status":"pass","files_checked":len(infos),"bytes_checked":total,"archive_sha256":sha(archive)}
def verify_extracted(archive,destination):
 verify(archive);root=Path(destination).resolve()
 with zipfile.ZipFile(archive,"r") as z:
  _,_,expected=_zip_index(z)
 checked=0;total=0
 for name,row in expected.items():
  target=root.joinpath(*PurePosixPath(name).parts)
  probe=root
  for part in PurePosixPath(name).parts:
   probe=probe/part
   if probe.is_symlink():raise ValueError("extracted path is a symlink: "+name)
  target.resolve().relative_to(root)
  if not target.is_file():raise FileNotFoundError("extracted file is missing: "+name)
  if target.stat().st_size!=row["bytes"] or sha(target)!=row["sha256"]:raise ValueError("extracted size/hash mismatch: "+name)
  checked+=1;total+=row["bytes"]
 return {"status":"pass","files_checked":checked,"bytes_checked":total,"workspace":str(root)}
def safe_extract(archive,destination,merge=False):
 v=verify(archive);dst=Path(destination).resolve()
 if dst.exists() and not merge and any(dst.iterdir()):raise FileExistsError("destination must be empty: "+str(dst))
 dst.mkdir(parents=True,exist_ok=True)
 created=[]
 try:
  with zipfile.ZipFile(archive,"r") as z:
   m,infos,expected=_zip_index(z)
   targets=[]
   for n in infos:
    target=dst.joinpath(*PurePosixPath(n).parts);target.resolve().relative_to(dst)
    probe=dst
    for part in PurePosixPath(n).parts:
     probe=probe/part
     if probe.is_symlink():raise ValueError("refusing extraction through symlink: "+str(probe))
    if target.exists():
     if not merge or not target.is_file() or target.stat().st_size!=expected[n]["bytes"] or sha(target)!=expected[n]["sha256"]:raise FileExistsError("refusing to overwrite differing file: "+str(target))
    targets.append((n,target))
   for n,target in targets:
    if target.exists():continue
    target.parent.mkdir(parents=True,exist_ok=True)
    with z.open(infos[n]) as src,target.open("xb") as out:shutil.copyfileobj(src,out,1024*1024)
    created.append(target)
   manifest_name=("TRANSFER_MANIFEST-"+Path(archive).stem+".json") if merge else MANIFEST
   mt=dst/manifest_name
   if mt.exists():
    if not merge or mt.read_bytes()!=z.read(MANIFEST):raise FileExistsError("refusing to overwrite transfer manifest")
   else:
    with z.open(MANIFEST) as src,mt.open("xb") as out:shutil.copyfileobj(src,out)
    created.append(mt)
 except Exception:
  for path in reversed(created):path.unlink(missing_ok=True)
  raise
 return {**v,"extracted_to":str(dst)}
def _bad_zip(path,entries):
 with zipfile.ZipFile(path,"w") as z:
  for name,data,mode in entries:
   i=zipfile.ZipInfo(name)
   if mode is not None:i.external_attr=mode<<16
   z.writestr(i,data)
def self_test(seed=49107):
 outcomes=[]
 with tempfile.TemporaryDirectory(prefix="transfer-Ω space ") as td:
  base=Path(td);src=base/"source with Ω";(src/"α folder").mkdir(parents=True);rng=random.Random(seed);payload=bytes(rng.randrange(256) for _ in range(4096));(src/"α folder/fixture.bin").write_bytes(payload)
  arc=base/"good archive.zip";create(src,arc,["α folder"]);outcomes.append(("seeded_unicode_space_archive",verify(arc)["status"]=="pass"));dst=base/"relocated Ω directory";safe_extract(arc,dst);outcomes.append(("clean_relocated_extract",(dst/"α folder/fixture.bin").read_bytes()==payload))
  bad_specs=[("../escape.txt",b"x",None),("/absolute.txt",b"x",None),("C:/drive.txt",b"x",None),("linked.txt",b"x",stat.S_IFLNK|0o777),("duplicate.txt",b"x",None)]
  for i,(n,d,mode) in enumerate(bad_specs,1):
   p=base/("bad-%02d.zip"%i);_bad_zip(p,[(n,d,mode)]+([(n,d,None)] if i==5 else []));rejected=False
   try:verify(p)
   except (ValueError,KeyError,zipfile.BadZipFile):rejected=True
   outcomes.append(("reject_unsafe_path_or_link_%02d"%i,rejected))
  # Correctly shaped manifests with missing, extra, wrong-size, wrong-hash, and unsafe entries.
  cases=[("missing",[],None),("extra",[("extra",b"x",None)],"extra"),("wrong_size",[("fixture.txt",b"xy",None)],"fixture.txt"),("wrong_hash",[("fixture.txt",b"x",None)],"fixture.txt"),("unsafe_manifest",[("../escape",b"x",None)],"../escape")]
  for i,(label,entries,name) in enumerate(cases,6):
   files=[]
   if label in {"missing","wrong_size","wrong_hash","unsafe_manifest"}:files=[{"path":"fixture.txt" if label!="unsafe_manifest" else "../escape","bytes":1,"sha256":"0"*64}]
   elif label=="extra":files=[]
   manifest={"schema":SCHEMA,"files":files};archive_entries=[(MANIFEST,json.dumps(manifest).encode(),None)]+entries
   p=base/("bad-%02d.zip"%i);_bad_zip(p,archive_entries);rejected=False
   try:verify(p)
   except (ValueError,KeyError,zipfile.BadZipFile):rejected=True
   outcomes.append(("reject_%s"%label,rejected))
 failed=[n for n,ok in outcomes if not ok]
 return {"status":"pass" if not failed and len(outcomes)>=10 else "fail","seed":seed,"fixture_blocks":len(outcomes),"cases":[{"id":n,"status":"pass" if ok else "fail"} for n,ok in outcomes],"failed":failed}
def main(argv=None):
 p=argparse.ArgumentParser(description=__doc__);s=p.add_subparsers(dest="cmd",required=True);c=s.add_parser("create");c.add_argument("--root",type=Path,required=True);c.add_argument("--archive",type=Path,required=True);c.add_argument("paths",nargs="+");v=s.add_parser("verify");v.add_argument("archive",type=Path);e=s.add_parser("extract");e.add_argument("archive",type=Path);e.add_argument("destination",type=Path);e.add_argument("--merge",action="store_true");x=s.add_parser("verify-extracted");x.add_argument("archive",type=Path);x.add_argument("workspace",type=Path);t=s.add_parser("self-test");t.add_argument("--seed",type=int,default=49107);a=p.parse_args(argv)
 try:
  r=create(a.root,a.archive,a.paths) if a.cmd=="create" else verify(a.archive) if a.cmd=="verify" else safe_extract(a.archive,a.destination,a.merge) if a.cmd=="extract" else verify_extracted(a.archive,a.workspace) if a.cmd=="verify-extracted" else self_test(a.seed);print(json.dumps(r,indent=2,sort_keys=True));return 0 if r.get("status")=="pass" else 1
 except (OSError,ValueError,KeyError,zipfile.BadZipFile) as exc:print(json.dumps({"status":"failed_closed","error":"%s: %s"%(type(exc).__name__,exc)}));return 2
if __name__=="__main__":raise SystemExit(main())
