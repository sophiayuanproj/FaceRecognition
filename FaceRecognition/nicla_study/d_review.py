"""Development-role visual error review for the frozen Job D candidates."""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from nicla_study import d_campaign as d
from nicla_study.fs import sha256_file


def _known_train(data: dict) -> tuple[list[dict], np.ndarray]:
    indices = [i for i, row in enumerate(data["train_rows"])
               if row.get("identity_partition") != "unknown" and row.get("label_id") in d.KNOWN]
    return [data["train_rows"][i] for i in indices], data["train_gray"][indices]


def _decisions_from_scores(scores: np.ndarray, rows: list[dict], threshold: float,
                           outputs: int | None = None) -> tuple[list[str], np.ndarray]:
    if outputs is not None:
        pred, confidence, other = d.prediction_components(scores, outputs)
        decisions = []
        for p, score, is_other in zip(pred, confidence, other):
            if is_other:
                decisions.append("OTHER")
            elif float(score) < threshold:
                decisions.append("REJECT")
            else:
                decisions.append(d.KNOWN[int(p)])
        return decisions, confidence
    pred = np.argmax(scores, axis=1)
    confidence = np.max(scores, axis=1)
    return [d.KNOWN[int(p)] if float(score) >= threshold else "REJECT"
            for p, score in zip(pred, confidence)], confidence


def _features(path: Path, images: np.ndarray, rows: list[dict]) -> dict[str, np.ndarray]:
    values = d._normalize_vectors(d.tflite_outputs(path, images))
    if values.shape[0] != len(rows):
        raise ValueError("visual review TFLite output rows are misaligned")
    return {str(row["record_id"]): values[i] for i, row in enumerate(rows)}


def _route_predictions(workspace: Path, paths: dict, route: dict, data: dict,
                       gallery_results: list[dict]) -> tuple[list[dict], list[str], np.ndarray]:
    known_train, train_gray = _known_train(data)
    known_rows, unknown_rows = data["dev_known"], data["dev_unknown"]
    query_rows = known_rows + unknown_rows
    query_gray = np.concatenate([data["dev_known_gray"], data["dev_unknown_gray"]], axis=0)
    route_type, family_id = route["family_type"], route["family_id"]
    if route_type == "classifier":
        seed = min(int(s["root_seed"]) for s in json.loads(paths["seed_registry"].read_text())["training_roots"])
        root = paths["runs"] / "survface" / family_id / ("seed-%d" % seed)
        result = json.loads((root / "ROOT_RESULT.json").read_text())
        logits = d.tflite_outputs(root / "classifier_float.tflite", query_gray)
        calibration = result["float_calibration"]
        threshold = float(calibration["threshold"])
        decisions, confidence = _decisions_from_scores(logits, query_rows, threshold,
                                                       int(calibration["outputs"]))
        return query_rows, decisions, confidence
    if route_type == "embedding_gallery":
        result = next(r for r in gallery_results if
                      ("a%s_%s_%s" % ("010" if r["width"] == 0.1 else "025", r["backend"], r["policy"])) == family_id)
        seed, backend, arm_id = int(result["root_seed"]), result["backend"], result["arm_id"]
        root = paths["runs"] / "survface" / arm_id / ("seed-%d" % seed)
        export = root / ("embedding_float.tflite" if backend == "float" else "embedding_int8.tflite")
        selection = dict(result["selected"])
        selection["policy"] = result["policy"]
        threshold = float(selection["calibration"]["threshold"])
    elif route_type == "sface_distilled_student":
        width_tag, backend = family_id.split("_")
        width_root = paths["runs"] / "teacher" / "students" / ("a" + width_tag)
        root = min(width_root.glob("seed-*/ROOT_RESULT.json"), key=lambda p: p.parent.name)
        result = json.loads(root.read_text())
        selected = result["selected_float" if backend == "float" else "selected_int8"]
        export = root.parent / ("embedding_float.tflite" if backend == "float" else "embedding_int8.tflite")
        selection = selected
        threshold = float(selection["calibration"]["threshold"])
        seed = int(result["root_seed"])
    elif route_type == "fixed_sface_host_teacher":
        teacher_result_path = paths["reports"] / "SFACE_TEACHER_RESULT.json"
        teacher_result = json.loads(teacher_result_path.read_text())
        selected = teacher_result["crop_mode_results"]["selected"]
        selection = selected
        threshold = float(selected["calibration"]["threshold"])
        cache_path = paths["runs"] / "teacher" / "sface_nonfinal_features.npz"
        index_path = paths["runs"] / "teacher" / "sface_nonfinal_feature_index.json"
        index = json.loads(index_path.read_text())
        if not cache_path.is_file() or index.get("status") != "complete_verified":
            raise FileNotFoundError("frozen SFace development feature cache is missing")
        raw = np.load(cache_path)["raw"]
        embed_map = {rid: raw[i] for i, rid in enumerate(index["record_ids"])}
        prototypes, _, _ = d.build_gallery_prototypes(embed_map, known_train,
            int(selected["template_count_requested"]), selected["policy"], 20261300)
        scores = np.stack([embed_map[str(row["record_id"])] for row in query_rows]) @ prototypes.T
        decisions, confidence = _decisions_from_scores(scores, query_rows, threshold)
        return query_rows, decisions, confidence
    else:
        raise ValueError("unsupported development review route: " + str(route_type))
    all_rows = known_train + query_rows
    all_images = np.concatenate([train_gray, query_gray], axis=0)
    embedding_map = _features(export, all_images, all_rows)
    prototypes, _, _ = d.build_gallery_prototypes(embedding_map, known_train,
        int(selection["template_count_requested"]), selection["policy"], seed)
    scores = np.stack([embedding_map[str(row["record_id"])] for row in query_rows]) @ prototypes.T
    decisions, confidence = _decisions_from_scores(scores, query_rows, threshold)
    return query_rows, decisions, confidence


def _sample_outcomes(rows: list[dict], decisions: list[str], scores: np.ndarray, seed: int) -> list[dict]:
    rng = np.random.default_rng(seed ^ 0xD339)
    outcome_bins = defaultdict(list)
    known_count = sum(row.get("label_id") in d.KNOWN for row in rows)
    for i, (row, decision, score) in enumerate(zip(rows, decisions, scores)):
        truth = str(row["label_id"]) if i < known_count else "UNKNOWN"
        if truth == "UNKNOWN":
            if decision in d.KNOWN:
                category = "unknown_false_identification"
            elif decision == "OTHER":
                category = "unknown_routed_to_other"
            else:
                category = "unknown_rejected"
        elif decision == truth:
            category = "known_correctly_identified"
        elif decision == "REJECT":
            category = "known_rejected"
        elif decision == "OTHER":
            category = "known_routed_to_other"
        else:
            category = "known_wrong_identity"
        outcome_bins[(truth, category)].append(i)
    chosen = []
    # One successful and one failed example per known identity where available.
    for label in d.KNOWN:
        successes = outcome_bins.get((label, "known_correctly_identified"), [])
        if successes:
            chosen.append(int(rng.choice(successes)))
        else:
            candidates = [i for i, row in enumerate(rows[:known_count]) if row.get("label_id") == label]
            if candidates:
                chosen.append(int(max(candidates, key=lambda i: float(scores[i]))))
        for category in ("known_wrong_identity", "known_rejected", "known_routed_to_other"):
            failures = outcome_bins.get((label, category), [])
            if failures:
                chosen.append(int(rng.choice(failures)))
                break
    for category in ("unknown_rejected", "unknown_routed_to_other", "unknown_false_identification"):
        candidates = outcome_bins.get(("UNKNOWN", category), [])
        if candidates:
            chosen.append(int(rng.choice(candidates)))
    out = []
    for i in dict.fromkeys(chosen):
        row = rows[i]
        truth = str(row["label_id"]) if row.get("label_id") in d.KNOWN else "UNKNOWN"
        out.append({"index": i, "record_id": row["record_id"], "true_label": truth,
                    "decision": decisions[i], "score": float(scores[i]),
                    "role": row.get("role"), "relative_path": row.get("relative_path"),
                    "camera_source_group": row.get("camera_source_group"),
                    "quality_score": row.get("quality_score"),
                    "duplicate_group_id": row.get("duplicate_group_id")})
    return out


def _contact_sheet(path: Path, examples: list[dict], gray: np.ndarray) -> None:
    cell_w, cell_h, image_size = 360, 190, 128
    columns = 3
    rows = max(1, int(np.ceil(len(examples) / columns)))
    canvas = Image.new("RGB", (columns * cell_w, rows * cell_h), "white")
    draw = ImageDraw.Draw(canvas)
    for n, item in enumerate(examples):
        x, y = (n % columns) * cell_w, (n // columns) * cell_h
        image = Image.fromarray(np.asarray(gray[item["index"]]).squeeze(), mode="L").convert("RGB")
        image = image.resize((image_size, image_size), Image.Resampling.NEAREST)
        canvas.paste(image, (x + 8, y + 8))
        text = "%s → %s\nscore %.3f | %s\n%s" % (item["true_label"], item["decision"], item["score"],
                                                      item["camera_source_group"], item["record_id"])
        draw.text((x + image_size + 18, y + 12), text, fill="black", spacing=6)
    path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(path)


def build_development_visual_review(workspace: Path, paths: dict, selection: dict,
                                   gallery_results: list[dict], data: dict) -> dict:
    routes = [selection.get("candidate_small_best_host_route"),
              selection.get("candidate_large_best_host_route"),
              selection.get("strong_reference")]
    deduped, seen = [], set()
    for route in routes:
        if not route:
            continue
        key = (route.get("family_type"), route.get("family_id"))
        if key not in seen:
            seen.add(key)
            deduped.append(route)
    out_root = paths["reports"] / "DEVELOPMENT_VISUAL_REVIEW"
    query_rows = data["dev_known"] + data["dev_unknown"]
    query_gray = np.concatenate([data["dev_known_gray"], data["dev_unknown_gray"]], axis=0)
    receipts = []
    for i, route in enumerate(deduped):
        rows, decisions, scores = _route_predictions(workspace, paths, route, data, gallery_results)
        if rows != query_rows:
            raise ValueError("development visual review attempted to use non-development rows")
        examples = _sample_outcomes(rows, decisions, scores, 20261930 + i)
        token = ("%s_%s" % (route["family_type"], route["family_id"])).replace("/", "_")
        image_path = out_root / (token + ".png")
        examples_path = out_root / (token + ".json")
        _contact_sheet(image_path, examples, query_gray)
        receipt = {"schema": "job-d-development-visual-review-1", "status": "complete_verified",
                   "route": route, "role": "development+unknown_development",
                   "selected_examples": examples, "contact_sheet_path": str(image_path),
                   "contact_sheet_sha256": sha256_file(image_path),
                   "final_role_opened": False, "created_utc": d.utc_now()}
        d.write_json(examples_path, receipt)
        receipt["examples_receipt_path"] = str(examples_path)
        receipts.append(receipt)
    summary = {"schema": "job-d-development-visual-review-summary-1", "status": "complete_verified",
               "routes_reviewed": receipts, "roles": ["development", "unknown_development"],
               "final_role_opened": False, "created_utc": d.utc_now()}
    d.write_json(out_root / "SUMMARY.json", summary)
    return summary
