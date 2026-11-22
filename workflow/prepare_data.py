"""Restore approved role-selected crops from their exact source archive."""
import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import zipfile

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "work/project-finish-v4/data"
ARCHIVES = {
    "survface_96": ("f61b38dd7364e2dd912ec573c5e9180a5de3d68b95b910b4b5dddda42b265fe6",
                    "primary_split_manifest.jsonl"),
    "olivetti_96": ("552c65c1b925374ecae40b0808b1c91c280f0d0baa8c74080bc26e4dafdad256",
                    "olivetti_sanity_manifest.jsonl"),
}
ROLES = {"train", "unknown_train", "stopping", "threshold_calibration",
         "unknown_calibration", "development", "unknown_development"}


def digest(path):
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive", type=Path)
    parser.add_argument("--dataset", choices=ARCHIVES, default="survface_96")
    args = parser.parse_args()
    expected, manifest = ARCHIVES[args.dataset]
    if digest(args.archive) != expected:
        raise SystemExit("Dataset ZIP SHA-256 differs from the recorded source. See DATA.md.")
    manifests = [DATA / manifest]
    if args.dataset == "survface_96":
        manifests.extend(sorted((DATA / "derived/D_selection").glob("*.jsonl")))
    prefix = PurePosixPath("work/dataset-prep/extracted") / args.dataset
    files = {}
    for path in manifests:
        for line in path.read_text().splitlines():
            row = json.loads(line)
            if row["role"] not in ROLES:
                raise ValueError("Unexpected role in portable data manifest: " + row["role"])
            relative = PurePosixPath(row["relative_path"])
            member = relative.relative_to(prefix)
            if relative.is_absolute() or ".." in member.parts:
                raise ValueError("Unsafe image path")
            if files.setdefault(member, row["sha256"]) != row["sha256"]:
                raise ValueError("Conflicting hashes for " + str(member))
    with zipfile.ZipFile(args.archive) as source:
        for member, expected_sha in files.items():
            content = source.read(str(member))
            if hashlib.sha256(content).hexdigest() != expected_sha:
                raise ValueError("Source image hash mismatch: " + str(member))
            path = ROOT / prefix / member
            if not path.resolve().is_relative_to(ROOT):
                raise ValueError("Extraction would leave the package")
            if path.exists() and digest(path) != expected_sha:
                raise ValueError("Existing image differs: " + str(path))
            path.parent.mkdir(parents=True, exist_ok=True)
            if not path.exists():
                temp = path.with_suffix(path.suffix + ".partial")
                temp.write_bytes(content)
                temp.replace(path)
    print(json.dumps({"status": "verified", "dataset": args.dataset,
                      "source_sha256": expected, "restored_files": len(files)}, indent=2))


if __name__ == "__main__":
    main()
