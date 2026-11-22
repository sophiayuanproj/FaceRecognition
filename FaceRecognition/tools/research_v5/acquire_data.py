"""Acquire audited public data inputs without using final-role imagery."""
import hashlib
import json
from pathlib import Path
import urllib.request

ROOT = Path(__file__).resolve().parents[3]
DATA = ROOT / "work/project-research-v5/data/raw"


def fetch(url, target, expected=None):
    if target.exists():
        digest = hashlib.sha256(target.read_bytes()).hexdigest()
        if expected and digest != expected:
            raise ValueError("Existing download checksum mismatch")
        return {"file": str(target.relative_to(ROOT)), "bytes": target.stat().st_size,
                "sha256": digest, "source": url, "reused": True}
    temp = target.with_suffix(target.suffix + ".partial")
    target.parent.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha256()
    total = 0
    with urllib.request.urlopen(url, timeout=45) as response, temp.open("wb") as stream:
        while True:
            block = response.read(1024 * 1024)
            if not block:
                break
            stream.write(block)
            digest.update(block)
            total += len(block)
            if total % (32 * 1024 * 1024) == 0:
                print(json.dumps({"downloading": target.name, "MiB": total // 2**20}), flush=True)
    if expected and digest.hexdigest() != expected:
        raise ValueError("Downloaded checksum mismatch")
    temp.replace(target)
    return {"file": str(target.relative_to(ROOT)), "bytes": total,
            "sha256": digest.hexdigest(), "source": url, "reused": False}


def main():
    receipts = []
    receipts.append(fetch("https://ndownloader.figshare.com/files/5976018", DATA / "lfw.tgz",
        "055f7d9c632d7370e6fb4afc7468d40f970c34a80d4c6f50ffec63f5a8d536c0"))
    print(json.dumps(receipts[-1]), flush=True)
    receipts.append(fetch("https://zenodo.org/api/records/815657", DATA / "chokepoint_record.json"))
    metadata = json.loads((DATA / "chokepoint_record.json").read_text())
    selected = {"README.txt", "groundtruth.tar.xz", "P1E_S1.tar.xz"}
    for row in metadata["files"]:
        if row["key"] in selected:
            url = row["links"]["self"]
            receipt = fetch(url, DATA / ("chokepoint_" + row["key"]))
            receipt["publisher_checksum"] = row.get("checksum")
            if str(row.get("checksum", "")).startswith("md5:"):
                actual = hashlib.md5((ROOT / receipt["file"]).read_bytes()).hexdigest()
                if actual != row["checksum"].split(":", 1)[1]:
                    raise ValueError("Publisher ChokePoint MD5 mismatch")
                receipt["publisher_checksum_verified"] = True
            receipts.append(receipt)
            print(json.dumps(receipt), flush=True)
    (DATA.parent / "DOWNLOADS.json").write_text(json.dumps({"status": "verified", "files": receipts},
        indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
