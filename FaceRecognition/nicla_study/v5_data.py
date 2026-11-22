"""Versioned source indexing and prospective five-public-ID LFW roles."""
from collections import Counter, defaultdict
import hashlib
import io
import json
from pathlib import Path
import random
import tarfile
import time

import cv2
import numpy as np
from PIL import Image

from .paths import resolve_study_root

WORKSPACE = Path(__file__).resolve().parents[2]


def dump(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def main():
    study = resolve_study_root(WORKSPACE, "work/project-research-v5")
    data = study / "data"
    output = data / "lfw_source"
    output.mkdir(exist_ok=True)
    manifest = data / "manifests/lfw_source.jsonl"
    if manifest.exists():
        rows = [json.loads(line) for line in manifest.read_text().splitlines()]
    else:
        rows = []
        with tarfile.open(data / "raw/lfw.tgz", "r:gz") as archive:
            for member in archive:
                if not member.isfile() or not member.name.lower().endswith(".jpg"):
                    continue
                parts = Path(member.name).parts
                if len(parts) != 3 or parts[0] != "lfw" or ".." in parts:
                    raise ValueError("Unexpected source archive path")
                payload = archive.extractfile(member).read()
                digest = hashlib.sha256(payload).hexdigest()
                image = np.asarray(Image.open(io.BytesIO(payload)).convert("RGB"))
                gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
                low = cv2.dct(cv2.resize(gray, (32, 32)).astype(np.float32))[:8, :8].ravel()
                phash = int.from_bytes(np.packbits(low > np.median(low[1:])).tobytes(), "big")
                dest = output / parts[1] / parts[2]
                dest.parent.mkdir(exist_ok=True)
                if not dest.exists():
                    dest.write_bytes(payload)
                rows.append({"record_id": "lfw:" + parts[1] + "/" + parts[2],
                    "source_identity": parts[1], "path": dest.relative_to(WORKSPACE).as_posix(),
                    "sha256": digest, "phash64": phash, "width": int(image.shape[1]),
                    "height": int(image.shape[0]), "channels": 3, "bytes": len(payload)})
                if len(rows) % 2000 == 0:
                    print(json.dumps({"indexed_lfw": len(rows)}), flush=True)
        manifest.write_text("".join(json.dumps(row) + "\n" for row in rows))

    parent = list(range(len(rows)))
    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i
    def union(i, j):
        i, j = find(i), find(j)
        if i != j:
            parent[j] = i

    # Five disjoint bit partitions guarantee a shared partition for distance <=4.
    buckets = [defaultdict(list) for _ in range(5)]
    slices = [(0, 13), (13, 13), (26, 13), (39, 13), (52, 12)]
    comparisons = edges = 0
    for i, row in enumerate(rows):
        code = row["phash64"]
        candidates = set()
        keys = []
        for table, (shift, width) in zip(buckets, slices):
            key = (code >> shift) & ((1 << width) - 1)
            keys.append(key)
            candidates.update(table[key])
        for j in candidates:
            comparisons += 1
            if bin(code ^ rows[j]["phash64"]).count("1") <= 4:
                union(i, j)
                edges += 1
        for table, key in zip(buckets, keys):
            table[key].append(i)
    groups = defaultdict(list)
    for i, row in enumerate(rows):
        groups[find(i)].append(row)
    by_person = defaultdict(list)
    ambiguous = []
    for group in groups.values():
        identities = {r["source_identity"] for r in group}
        if len(identities) != 1:
            ambiguous.append([r["record_id"] for r in group])
            continue
        chosen = min(group, key=lambda r: r["sha256"])
        chosen = dict(chosen, source_group="lfw-group:" + min(r["sha256"] for r in group),
                      group_records=[r["record_id"] for r in group])
        by_person[chosen["source_identity"]].append(chosen)
    ranking = sorted(by_person, key=lambda identity: (-len(by_person[identity]), identity))
    known = [identity for identity in ranking if len(by_person[identity]) >= 100][:5]
    if len(known) != 5:
        raise ValueError("Five metadata-eligible identities with >=100 unique groups unavailable")
    rng = random.Random(53101)
    split_rows = []
    roster = []
    for i, identity in enumerate(known):
        values = sorted(by_person[identity], key=lambda r: r["sha256"])
        rng.shuffle(values)
        label = "K%02d" % (i + 1)
        roles = ["future_final"] * 20 + ["development"] * 20 + ["threshold_calibration"] * 10 + ["stopping"] * 10
        roles += ["train"] * (len(values) - len(roles))
        for row, role in zip(values, roles):
            split_rows.append(dict(row, dataset="lfw-v5", label=label, role=role, known=True))
        roster.append({"label": label, "source_identity": identity, "unique_groups": len(values)})
    eligible = [identity for identity in ranking if identity not in known and len(by_person[identity]) >= 10]
    rng.shuffle(eligible)
    if len(eligible) < 90:
        raise ValueError("Insufficient separate unknown people for three 30-ID x10-group roles")
    unknown_roles = {}
    for role, identities in zip(["unknown_calibration", "unknown_development", "unknown_future_final"],
                               [eligible[:30], eligible[30:60], eligible[60:90]]):
        unknown_roles[role] = identities
        for identity in identities:
            values = sorted(by_person[identity], key=lambda r: r["sha256"])
            rng.shuffle(values)
            for row in values[:10]:
                split_rows.append(dict(row, dataset="lfw-v5", label="OTHER", role=role, known=False))
    excluded = set(known) | set(eligible[:90])
    train_unknown = [identity for identity in ranking if identity not in excluded]
    rng.shuffle(train_unknown)
    for identity in train_unknown[:200]:
        values = sorted(by_person[identity], key=lambda r: r["sha256"])
        rng.shuffle(values)
        for row in values[:3]:
            split_rows.append(dict(row, dataset="lfw-v5", label="OTHER", role="other_train", known=False))
    seen = defaultdict(set)
    for row in split_rows:
        seen[row["source_group"]].add(row["role"])
    assert all(len(v) == 1 for v in seen.values())
    path = data / "manifests/lfw_roles.jsonl"
    if path.exists() and path.read_text() != "".join(json.dumps(r) + "\n" for r in split_rows):
        raise FileExistsError("Refusing to replace an existing different role freeze")
    path.write_text("".join(json.dumps(r) + "\n" for r in split_rows))
    freeze = {"schema": "v5-lfw-prospective-roles-1", "dataset": "lfw-v5", "roster": roster,
        "source_images": len(rows), "source_identities": len(by_person),
        "selection_rule": "Top five by unique conservative source-group count, tie-break source ID; no recognition scores",
        "source_pre_role_decode": "All source pixels decoded for grouping before role assignment; no model fitting/evaluation used for selection",
        "future_final_used_for_training_or_model_assessment": False,
        "role_counts": dict(Counter(r["role"] for r in split_rows)),
        "unknown_role_identities": {role: len(ids) for role, ids in unknown_roles.items()},
        "manifest_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "phash_grouping": {"distance": 4, "candidate_pairs": comparisons, "edges": edges,
                           "ambiguous_cross_identity_groups_excluded": len(ambiguous)},
        "limitations": ["Still news photographs; not target-camera or verified independent capture sessions",
                       "Pretrained face-model identity exposure cannot be excluded",
                       "Five coverage-selected identities are not a population-representative benchmark"]}
    dump(data / "LFW_ROLE_FREEZE.json", freeze)
    dump(data / "LFW_AMBIGUOUS_GROUPS.json", ambiguous)
    print(json.dumps(freeze), flush=True)


if __name__ == "__main__":
    main()
