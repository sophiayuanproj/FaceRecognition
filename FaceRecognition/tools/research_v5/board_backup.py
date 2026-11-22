"""Verified filesystem backup and narrowly scoped inactive-release cleanup."""
import argparse
import hashlib
import json
from pathlib import Path
from mpremote.transport_serial import SerialTransport


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", required=True)
    parser.add_argument("--backup", type=Path, required=True)
    parser.add_argument("--cleanup-inactive", action="store_true")
    args = parser.parse_args()
    args.backup.mkdir(parents=True, exist_ok=True)
    target = SerialTransport(args.port, timeout=10)
    try:
        target.serial.rts = False
        target.serial.dtr = True
        # OpenMV's protocol negotiation is not mpremote raw-paste compatible.
        # The bounded standard raw REPL was independently verified first.
        target.use_raw_paste = False
        target.enter_raw_repl(soft_reset=False)
        target.exec("import os,hashlib,binascii,json,gc;gc.collect()\n"
                    "def _sha(p):\n h=hashlib.sha256()\n f=open(p,'rb')\n"
                    " while True:\n  b=f.read(2048)\n  if not b:break\n  h.update(b)\n"
                    " f.close()\n return binascii.hexlify(h.digest()).decode()")
        def walk(directory):
            found = []
            for entry in target.fs_listdir(directory):
                name = directory + "/" + entry.name
                if entry.st_mode & 0x4000:
                    found.extend(walk(name))
                else:
                    try:
                        digest = target.exec("gc.collect();print(_sha(%r))" % name).decode().strip()
                        size = target.fs_stat(name).st_size
                        found.append({"path": name, "bytes": size, "sha256": digest})
                    except Exception as error:
                        found.append({"path": name, "error": str(error)})
            return found
        rows = walk("/flash") + walk("/rom")
        saved = []
        unreadable = []
        for index, row in enumerate(rows):
            if "error" in row:
                unreadable.append(row)
                continue
            relative = row["path"].lstrip("/")
            dest = args.backup / relative
            if not dest.is_file() or hashlib.sha256(dest.read_bytes()).hexdigest() != row["sha256"]:
                data = bytes(target.fs_readfile(row["path"], chunk_size=4096))
                if len(data) != row["bytes"] or hashlib.sha256(data).hexdigest() != row["sha256"]:
                    raise ValueError("Board backup size/hash mismatch")
                dest.parent.mkdir(parents=True, exist_ok=True)
                dest.write_bytes(data)
            saved.append(row)
            if (index + 1) % 25 == 0:
                print(json.dumps({"verified_files": index + 1, "total": len(rows)}), flush=True)
        receipt = {"schema": "nicla-v5-private-backup-1",
                   "status": "verified_readable_files_with_errors" if unreadable else "verified",
                   "scope": "All readable /flash and /rom files; not a raw flash partition image",
                   "private_do_not_distribute": True, "files": saved, "unreadable": unreadable,
                   "bytes": sum(row["bytes"] for row in saved)}
        (args.backup / "manifest.json").write_text(json.dumps(receipt, indent=2) + "\n")
        if args.cleanup_inactive:
            roots = set()
            for name in ["face_runtime.current", "face_runtime.current.prev"]:
                pointer = args.backup / "flash" / name
                if pointer.exists():
                    text = pointer.read_text().strip()
                    if text.startswith("/flash/releases/") and ".." not in text:
                        roots.add(text.rstrip("/"))
            if len(roots) != 2:
                raise ValueError("Two distinct active/fallback roots required before cleanup")
            entries = target.fs_listdir("/flash/releases")
            remove = ["/flash/releases/" + row.name for row in entries
                      if row.st_mode & 0x4000 and row.name.startswith("face-v1-")
                      and "/flash/releases/" + row.name not in roots]
            recorded = {row["path"]: row for row in saved}
            removed = []
            for directory in remove:
                current = walk(directory)
                if any("error" in row for row in current):
                    continue  # Never delete a release containing unbacked-up bytes.
                if any(recorded.get(row["path"]) != row for row in current):
                    raise ValueError("Inactive release changed after verified backup")
                target.exec("def _rm(p):\n for r in os.ilistdir(p):\n"
                            "  q=p+'/'+r[0]\n  if r[1]&0x4000:_rm(q)\n"
                            "  else:os.remove(q)\n os.rmdir(p)\n_rm(%r)" % directory)
                removed.append({"release": directory, "files": len(current),
                                "bytes": sum(r["bytes"] for r in current)})
            cleanup = {"status": "verified_inactive_cleanup", "preserved_roots": sorted(roots),
                       "removed": removed, "statvfs": target.exec("print(os.statvfs('/flash'))").decode().strip()}
            (args.backup / "cleanup.json").write_text(json.dumps(cleanup, indent=2) + "\n")
            print(json.dumps(cleanup), flush=True)
        print(json.dumps({"status": "backup_verified", "files": len(saved), "bytes": receipt["bytes"]}), flush=True)
        target.exit_raw_repl()
    finally:
        target.close()


if __name__ == "__main__":
    main()
