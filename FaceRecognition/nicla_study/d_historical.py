"""Audited v3 custom_base/custom_tiny gallery controls for Job D.

This module evaluates only frozen v3 TFLite outputs. It never trains or adapts
the historical encoders. Development and final role access follows D's role
helpers; the historical final metrics are explicitly exploratory because the
five public identities appeared in v3 development.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import tensorflow as tf

from nicla_study import d_campaign as d
from nicla_study.fs import sha256_file
from nicla_study.data_pipeline import read_jsonl


METHODS = ("custom_base", "custom_tiny")
SEEDS = (20261022, 20261023, 20261024, 20261025, 20261026,
         20261032, 20261033, 20261034, 20261035, 20261036)
BACKENDS = ("float", "int8")
POLICIES = ("quality", "uniform")
COUNTS = (1, 3, 5, 10, 60)
MODEL_ROOT = Path("work/research-complete-v3/project/work/research-next/model-system/runs/training-v3")
DATA_ROOT = Path("work/research-complete-v3/project/work/research-next/model-system/data/v3")


def now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def json_hash(value) -> str:
    body = json.dumps(_json_safe(value), sort_keys=True, separators=(",", ":"), ensure_ascii=False) + "\n"
    return hashlib.sha256(body.encode()).hexdigest()


def _json_safe(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return value


def write_json(path: Path, value: dict) -> None:
    d.write_json(path, _json_safe(value))


def _study_paths(workspace: Path) -> dict:
    paths = d.study_paths(workspace)
    hist = paths["runs"] / "historical_gallery"
    return paths | {
        "historical_runs": hist,
        "historical_results": paths["reports"] / "HISTORICAL_GALLERY_RESULTS.json",
        "historical_audit": paths["reports"] / "D_HISTORICAL_CONTROL_AUDIT.json",
        "historical_registry": paths["reports"] / "HISTORICAL_MODEL_REGISTRY_FROZEN.json",
        "historical_final": paths["handoff"] / "HISTORICAL_FINAL_SUMMARY.json",
    }


def _jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def _source_folder(row: dict) -> str | None:
    pieces = str(row.get("relative_local_path", "")).replace("\\", "/").split("/")
    return pieces[3] if len(pieces) > 3 else None


def audit_historical(workspace: Path, paths: dict | None = None) -> dict:
    """Hash and role-audit v3 training records/checkpoints without reading D final rows."""
    paths = paths or _study_paths(workspace)
    v3_data = workspace / DATA_ROOT
    role_rows = {role: _jsonl(v3_data / (role + "_records.jsonl"))
                 for role in ("train", "development", "calibration")}
    train_keyset = {(r.get("record_id"), r.get("sha256"), r.get("relative_local_path"))
                    for r in role_rows["train"]}
    train_identity_ids = {r.get("identity_id") for r in role_rows["train"]}
    role_identity_ids = {role: {r.get("identity_id") for r in rows}
                         for role, rows in role_rows.items()}
    roster = json.loads(paths["roster"].read_text(encoding="utf-8"))
    known_source_ids = {str(r["source_identity_id"]) for r in roster["identity_map"]}
    known_roles = {role: {str(_source_folder(r)) for r in rows if _source_folder(r) is not None}
                   for role, rows in role_rows.items()}
    roster_exposure = {
        str(entry["label"]): {
            "source_identity_id": str(entry["source_identity_id"]),
            "v3_train_record_count": sum(_source_folder(r) == str(entry["source_identity_id"])
                                         for r in role_rows["train"]),
            "v3_development_record_count": sum(_source_folder(r) == str(entry["source_identity_id"])
                                               for r in role_rows["development"]),
            "v3_calibration_record_count": sum(_source_folder(r) == str(entry["source_identity_id"])
                                               for r in role_rows["calibration"]),
        } for entry in roster["identity_map"]
    }
    roots = []
    errors = []
    inventory_path = workspace / "work/research-complete-v3/MODEL_INVENTORY.csv"
    inventory_rows = {}
    if inventory_path.is_file():
        with inventory_path.open(newline="", encoding="utf-8") as stream:
            for row in csv.DictReader(stream):
                inventory_rows.setdefault(row.get("original_project_path"), []).append(row)
    else:
        errors.append("v3 MODEL_INVENTORY.csv is missing")
    for method in METHODS:
        method_root = workspace / MODEL_ROOT / method
        observed = sorted(int(p.name.removeprefix("seed-")) for p in method_root.glob("seed-*")
                          if p.is_dir())
        if observed != sorted(SEEDS):
            errors.append("%s root seeds differ from the declared ten: %r" % (method, observed))
        for seed in SEEDS:
            root = method_root / ("seed-%d" % seed)
            manifest_path = root / "run_manifest.json"
            selected_path = root / "selected_train_records.jsonl"
            architecture_path = root / "architecture.json"
            checkpoint_path = root / "best_embedding.h5"
            history_path = root / "training_history.jsonl"
            exports = {
                "float": root / "exports/embedding_float32.tflite",
                "int8": root / "exports/embedding_full_int8.tflite",
            }
            required = [manifest_path, selected_path, architecture_path, checkpoint_path, history_path, *exports.values()]
            missing = [str(p) for p in required if not p.is_file()]
            if missing:
                errors.append("%s/%s missing files: %s" % (method, seed, ", ".join(missing)))
                continue
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            selected_rows = _jsonl(selected_path)
            selected_keys = {(r.get("record_id"), r.get("sha256"), r.get("relative_local_path"))
                             for r in selected_rows}
            mismatches = [r for r in selected_rows
                          if (r.get("record_id"), r.get("sha256"), r.get("relative_local_path"))
                          not in train_keyset]
            selected_source_ids = {str(_source_folder(r)) for r in selected_rows}
            root_errors = []
            if manifest.get("status") != "trained":
                root_errors.append("run status is not trained")
            if len(selected_rows) != int(manifest.get("training_record_count", -1)):
                root_errors.append("selected training row count mismatch")
            if len(selected_keys) != len(selected_rows):
                root_errors.append("selected training records contain duplicate triples")
            if mismatches:
                root_errors.append("selected training records are not all in v3 train role")
            if hashlib.sha256(selected_path.read_bytes()).hexdigest() != manifest.get("selected_train_records_file_sha256"):
                root_errors.append("selected training manifest file hash mismatch")
            if sha256_file(architecture_path) != manifest.get("architecture_sha256"):
                root_errors.append("architecture hash mismatch")
            if sha256_file(checkpoint_path) != manifest.get("best_checkpoint", {}).get("sha256"):
                root_errors.append("best checkpoint hash mismatch")
            if len(selected_source_ids & known_source_ids):
                root_errors.append("training source folder overlaps D known roster")
            artifacts = {name: {"path": str(path), "bytes": path.stat().st_size,
                                "sha256": sha256_file(path)} for name, path in exports.items()}
            artifact_inventory_checks = {}
            for kind, path in {"best_checkpoint": checkpoint_path, **exports}.items():
                project_relative = str(path.relative_to(workspace / "work/research-complete-v3/project"))
                hits = inventory_rows.get(project_relative, [])
                inventory_sha = hits[0].get("sha256") if len(hits) == 1 else None
                check = {"inventory_path": project_relative, "inventory_rows": len(hits),
                         "inventory_sha256": inventory_sha,
                         "file_sha256": sha256_file(path),
                         "match": len(hits) == 1 and inventory_sha == sha256_file(path)}
                artifact_inventory_checks[kind] = check
                if not check["match"]:
                    root_errors.append("%s inventory checksum/path mismatch" % kind)
            root_receipt = {
                "method": method, "root_seed": seed, "status": manifest.get("status"),
                "root_path": str(root), "training_record_count": len(selected_rows),
                "training_identity_count": len({r.get("identity_id") for r in selected_rows}),
                "training_role_record_mismatches": len(mismatches),
                "training_source_identity_overlap_with_D_known": sorted(selected_source_ids & known_source_ids),
                "selected_train_records_sha256": sha256_file(selected_path),
                "architecture_sha256": sha256_file(architecture_path),
                "best_checkpoint_sha256": sha256_file(checkpoint_path),
                "run_manifest_sha256": sha256_file(manifest_path),
                "training_history_sha256": sha256_file(history_path),
                "initialization": "random initialization of the v3 custom CNN; not face-pretrained",
                "target_adaptation": "none for D; trained on v3 train identities and model-selected on v3 development",
                "best_epoch_selected_by_v3_development_auc": manifest.get("best_epoch"),
                "best_v3_development_roc_auc": manifest.get("best_development_roc_auc"),
                "exports": artifacts, "artifact_inventory_checks": artifact_inventory_checks,
                "errors": root_errors,
            }
            roots.append(root_receipt)
            errors.extend("%s/seed-%s: %s" % (method, seed, e) for e in root_errors)
    role_hashes = {role: {"path": str(v3_data / (role + "_records.jsonl")),
                          "sha256": sha256_file(v3_data / (role + "_records.jsonl")),
                          "rows": len(rows), "identity_count": len(role_identity_ids[role])}
                   for role, rows in role_rows.items()}
    cross_role_overlap = {"train_development": len(role_identity_ids["train"] & role_identity_ids["development"]),
                          "train_calibration": len(role_identity_ids["train"] & role_identity_ids["calibration"]),
                          "development_calibration": len(role_identity_ids["development"] & role_identity_ids["calibration"])}
    if cross_role_overlap != {"train_development": 0, "train_calibration": 0, "development_calibration": 0}:
        errors.append("v3 role identity partitions overlap")
    if not known_source_ids <= known_roles["development"]:
        errors.append("D roster source identities are not all present in v3 development")
    if known_source_ids & known_roles["train"] or known_source_ids & known_roles["calibration"]:
        errors.append("D roster source identities overlap v3 train or calibration")
    output = {
        "schema": "job-d-historical-control-audit-1",
        "status": "complete_verified" if not errors and len(roots) == 20 else "invalid",
        "created_utc": now(), "methods": list(METHODS), "roots_per_method": 10,
        "v3_model_inventory": {"path": str(inventory_path),
                               "sha256": sha256_file(inventory_path) if inventory_path.is_file() else None},
        "v3_protocol_sha256": sha256_file(workspace / "work/research-complete-v3/project/work/research-next/model-system/protocol.json"),
        "v3_role_manifests": role_hashes,
        "v3_role_identity_overlap": cross_role_overlap,
        "per_file_training_and_artifact_audit": roots,
        "D_known_roster_v3_exposure": roster_exposure,
        "historical_selection_exposure": {
            "checkpoint_selection_metric": "best v3 development ROC-AUC",
            "D_roster_source_images_in_v3_development": True,
            "interpretation": "All source image groups for K01-K05 were in v3 development. Historical results on D roles are exploratory and cannot establish an independent final benchmark.",
        },
        "D_final_role_opened": False,
        "historical_audit_read_D_primary_split_manifest": False,
        "errors": errors,
        "created_files": [str(paths["historical_audit"])],
    }
    write_json(paths["historical_audit"], output)
    return output


def _gallery_dir(paths: dict, method: str, seed: int) -> Path:
    return paths["historical_runs"] / method / ("seed-%d" % seed)


def _run_development(workspace: Path, paths: dict, audit: dict) -> dict:
    matrix = d._load_matrix(workspace, paths)
    base = d.load_role_arrays(workspace, paths, "survface_96")
    train_rows = [r for r in base["train_rows"] if r.get("identity_partition") != "unknown"
                  and r.get("label_id") in d.KNOWN]
    train_indices = [i for i, r in enumerate(base["train_rows"])
                     if r.get("identity_partition") != "unknown" and r.get("label_id") in d.KNOWN]
    train_gray = base["train_gray"][train_indices]
    all_rows = train_rows + base["cal_known"] + base["cal_unknown"] + base["dev_known"] + base["dev_unknown"]
    all_gray = np.concatenate([train_gray, base["cal_known_gray"], base["cal_unknown_gray"],
                               base["dev_known_gray"], base["dev_unknown_gray"]], axis=0)
    if len({str(r["record_id"]) for r in all_rows}) != len(all_rows):
        raise ValueError("non-final D roles contain duplicate record IDs")
    offsets = {"train": 0, "cal_known": len(train_rows),
               "cal_unknown": len(train_rows) + len(base["cal_known"]),
               "dev_known": len(train_rows) + len(base["cal_known"]) + len(base["cal_unknown"]),
               "dev_unknown": len(train_rows) + len(base["cal_known"]) + len(base["cal_unknown"]) + len(base["dev_known"])}
    all_results = []
    for method in METHODS:
        for seed in SEEDS:
            root = workspace / MODEL_ROOT / method / ("seed-%d" % seed)
            emb_map_by_backend = {}
            io_contracts = {}
            for backend in BACKENDS:
                export = root / "exports" / ("embedding_float32.tflite" if backend == "float"
                                              else "embedding_full_int8.tflite")
                model = tf.lite.Interpreter(model_path=str(export), num_threads=2)
                model.allocate_tensors()
                inputs, outputs = model.get_input_details()[0], model.get_output_details()[0]
                expected_dtype = np.float32 if backend == "float" else np.int8
                if list(inputs["shape"]) != [1, 96, 96, 1] or inputs["dtype"] != expected_dtype:
                    raise ValueError("%s %s input contract mismatch: %s %s" %
                                     (method, backend, inputs["shape"], inputs["dtype"]))
                if list(outputs["shape"]) != [1, 64]:
                    raise ValueError("%s %s expected 64-D output, got %s" % (method, backend, outputs["shape"]))
                expected_output_dtype = np.float32 if backend == "float" else np.int8
                if outputs["dtype"] != expected_output_dtype:
                    raise ValueError("%s %s output dtype mismatch: %s" % (method, backend, outputs["dtype"]))
                if backend == "int8" and (not inputs["quantization"][0] or not outputs["quantization"][0]):
                    raise ValueError("historical full-int8 route lacks int8 IO quantization")
                io_contracts[backend] = {
                    "input": {"shape": [int(value) for value in inputs["shape"]], "dtype": str(inputs["dtype"]),
                              "scale": float(inputs["quantization"][0]), "zero_point": int(inputs["quantization"][1])},
                    "output": {"shape": [int(value) for value in outputs["shape"]], "dtype": str(outputs["dtype"]),
                               "scale": float(outputs["quantization"][0]), "zero_point": int(outputs["quantization"][1])},
                    "input_transform": "gray8/255 then quantize using this TFLite tensor's scale/zero point",
                }
                del model
                features = d._normalize_vectors(d.tflite_outputs(export, all_gray))
                emb_map_by_backend[backend] = {str(r["record_id"]): features[i] for i, r in enumerate(all_rows)}
            for backend in BACKENDS:
                embedding_map = emb_map_by_backend[backend]
                for policy in POLICIES:
                    candidates = []
                    for count in COUNTS:
                        prototypes, _, selection = d.build_gallery_prototypes(embedding_map, train_rows,
                                                                              count, policy, seed)
                        def score(rows):
                            return np.stack([embedding_map[str(r["record_id"])] for r in rows]) @ prototypes.T
                        cal = d.fit_gallery_threshold(score(base["cal_known"]), base["cal_known"],
                                                      score(base["cal_unknown"]), base["cal_unknown"])
                        dev_metrics = d.gallery_metrics(score(base["dev_known"]), base["dev_known"],
                                                        score(base["dev_unknown"]), base["dev_unknown"],
                                                        cal["threshold"])
                        candidates.append({"template_count_requested": count,
                                           "template_count_actual": selection[d.KNOWN[0]]["template_count"],
                                           "calibration": cal, "development": dev_metrics,
                                           "selection": selection,
                                           "prototype_sha256": d.sha256_json(prototypes.tolist())})
                    chosen = max(candidates, key=lambda r: (float(r["development"]["development_macro_utility"]),
                                                            float(r["development"]["known_macro_identification"]),
                                                            -int(r["template_count_requested"])))
                    out_dir = _gallery_dir(paths, method, seed)
                    result_path = out_dir / (backend + "_" + policy + ".json")
                    result = {
                        "schema": "job-d-historical-gallery-result-1", "status": "complete_verified",
                        "method": method, "root_seed": seed, "backend": backend, "policy": policy,
                        "model_path": str(root / "exports" / ("embedding_float32.tflite" if backend == "float"
                                                                   else "embedding_full_int8.tflite")),
                        "model_sha256": sha256_file(root / "exports" / ("embedding_float32.tflite" if backend == "float"
                                                                          else "embedding_full_int8.tflite")),
                        "representation": "frozen v3 custom_base/custom_tiny 64-D embedding; no target fine-tuning",
                        "input_contract": io_contracts[backend],
                        "candidate_counts": candidates, "selected": chosen,
                        "train_gallery_rows_sha256": d.rows_fingerprint(train_rows),
                        "calibration_roles": ["threshold_calibration", "unknown_calibration"],
                        "selection_roles": ["development", "unknown_development"],
                        "v3_known_identity_development_exposure": True,
                        "final_role_opened": False, "created_utc": now(),
                    }
                    write_json(result_path, result)
                    result["result_path"] = str(result_path)
                    all_results.append(result)
    by_family = defaultdict(list)
    for result in all_results:
        by_family[(result["method"], result["backend"], result["policy"])].append(result)
    summary = {"schema": "job-d-historical-gallery-development-1", "status": "complete_verified",
               "final_role_opened": False, "audit_path": str(paths["historical_audit"]),
               "audit_sha256": sha256_file(paths["historical_audit"]),
               "methods": {"%s_%s_%s" % key: {
                   "root_count": len(rows),
                   "development_macro_utility_mean": float(np.mean([r["selected"]["development"]["development_macro_utility"] for r in rows])),
                   "development_macro_utility_sd": float(np.std([r["selected"]["development"]["development_macro_utility"] for r in rows], ddof=1)),
                   "known_identification_mean": float(np.mean([r["selected"]["development"]["known_macro_identification"] for r in rows])),
                   "unknown_fpir_mean": float(np.mean([r["selected"]["development"]["unknown_false_identification_rate"] for r in rows])),
                   "roots": [{"root_seed": r["root_seed"],
                              "selected_template_count": r["selected"]["template_count_requested"],
                              "development": r["selected"]["development"]} for r in rows],
               } for key, rows in by_family.items()},
               "result_count": len(all_results), "created_utc": now()}
    write_json(paths["historical_results"], summary)
    return summary


def freeze_historical_registry(workspace: Path, paths: dict, audit: dict, results_summary: dict) -> dict:
    if audit.get("status") != "complete_verified" or results_summary.get("status") != "complete_verified":
        raise ValueError("historical audit and development results must be complete")
    protocol_hash = json_hash({"matrix": sha256_file(paths["matrix"]),
                               "audit": sha256_file(paths["historical_audit"]),
                               "development_results": sha256_file(paths["historical_results"])})
    frozen_roots, frozen_galleries = [], []
    for method in METHODS:
        for seed in SEEDS:
            model_root = workspace / MODEL_ROOT / method / ("seed-%d" % seed)
            audit_row = next(r for r in audit["per_file_training_and_artifact_audit"]
                             if r["method"] == method and r["root_seed"] == seed)
            frozen_roots.append({"method": method, "root_seed": seed,
                                 "root_path": str(model_root), "audit": audit_row})
            for backend in BACKENDS:
                for policy in POLICIES:
                    result_path = _gallery_dir(paths, method, seed) / (backend + "_" + policy + ".json")
                    result = json.loads(result_path.read_text(encoding="utf-8"))
                    model_path = Path(result["model_path"])
                    if sha256_file(model_path) != result["model_sha256"]:
                        raise ValueError("historical TFLite export changed after development scoring")
                    config_path = result_path.with_suffix(".CONFIG.json")
                    calibration_path = result_path.with_suffix(".CALIBRATION.json")
                    write_json(config_path, {"method": method, "root_seed": seed, "backend": backend,
                                             "policy": policy, "selected": result["selected"],
                                             "model_sha256": result["model_sha256"]})
                    write_json(calibration_path, result["selected"]["calibration"])
                    receipt = {"schema": "job-d-historical-gallery-freeze-1", "status": "frozen",
                               "model_sha256": result["model_sha256"],
                               "config_sha256": sha256_file(config_path),
                               "calibration_sha256": sha256_file(calibration_path),
                               "protocol_sha256": protocol_hash, "final_role_opened": False,
                               "method": method, "root_seed": seed, "backend": backend, "policy": policy,
                               "created_utc": now()}
                    freeze_path = result_path.with_suffix(".FROZEN.json")
                    write_json(freeze_path, receipt)
                    frozen_galleries.append({"result_path": str(result_path),
                                             "result_sha256": sha256_file(result_path),
                                             "config_path": str(config_path),
                                             "calibration_path": str(calibration_path),
                                             "freeze_receipt": str(freeze_path), **receipt})
    registry = {"schema": "job-d-historical-model-registry-1", "status": "complete_verified",
                "status_detail": "all v3 models, gallery counts, policies, thresholds, and development selections frozen before D final access",
                "methods": list(METHODS), "roots_per_method": 10, "root_count": len(frozen_roots),
                "backend_policy_evaluation_count": len(frozen_galleries),
                "protocol_sha256": protocol_hash,
                "audit_path": str(paths["historical_audit"]),
                "audit_sha256": sha256_file(paths["historical_audit"]),
                "development_results_path": str(paths["historical_results"]),
                "development_results_sha256": sha256_file(paths["historical_results"]),
                "roots": frozen_roots, "galleries": frozen_galleries,
                "known_identity_exposure": "K01-K05 source identities and all their source images were in v3 development; the final comparison is exploratory.",
                "final_role_opened": False, "created_utc": now()}
    write_json(paths["historical_registry"], registry)
    return registry


def run_development_campaign(workspace: Path) -> dict:
    paths = _study_paths(workspace)
    audit = audit_historical(workspace, paths)
    if audit["status"] != "complete_verified":
        raise ValueError("historical v3 per-file audit failed")
    summary = _run_development(workspace, paths, audit)
    registry = freeze_historical_registry(workspace, paths, audit, summary)
    return {"audit": audit, "development": summary, "registry": registry}


def _verify_registry(paths: dict) -> dict:
    registry = json.loads(paths["historical_registry"].read_text(encoding="utf-8"))
    if registry.get("status") != "complete_verified" or registry.get("final_role_opened") is not False:
        raise PermissionError("historical gallery registry is not frozen before final access")
    if sha256_file(Path(registry["audit_path"])) != registry.get("audit_sha256"):
        raise ValueError("historical per-file exposure/hash audit changed after freeze")
    if sha256_file(Path(registry["development_results_path"])) != registry.get("development_results_sha256"):
        raise ValueError("historical development results changed after freeze")
    for gallery in registry["galleries"]:
        result = json.loads(Path(gallery["result_path"]).read_text(encoding="utf-8"))
        receipt = json.loads(Path(gallery["freeze_receipt"]).read_text(encoding="utf-8"))
        if receipt.get("status") != "frozen" or receipt.get("final_role_opened") not in (False, True):
            raise PermissionError("historical gallery freeze receipt is invalid")
        if sha256_file(Path(gallery["result_path"])) != gallery.get("result_sha256"):
            raise ValueError("historical development selection changed after freeze")
        if sha256_file(Path(gallery["config_path"])) != receipt.get("config_sha256"):
            raise ValueError("historical gallery config changed after freeze")
        if sha256_file(Path(gallery["calibration_path"])) != receipt.get("calibration_sha256"):
            raise ValueError("historical gallery calibration changed after freeze")
        if sha256_file(Path(result["model_path"])) != receipt["model_sha256"]:
            raise ValueError("historical model artifact hash changed after freeze")
        if receipt.get("final_role_opened") is True:
            final_path = Path(gallery["result_path"]).with_suffix(".FINAL.json")
            if not final_path.is_file():
                raise ValueError("opened historical receipt has no final result")
            final = json.loads(final_path.read_text(encoding="utf-8"))
            prediction_path = Path(final.get("predictions", {}).get("path", ""))
            if (final.get("status") != "complete_verified" or final.get("final_role_opened") is not True or
                    not prediction_path.is_file() or
                    sha256_file(prediction_path) != final.get("predictions", {}).get("sha256")):
                raise ValueError("opened historical receipt has no hash-verified final evidence")
    return registry


def _gallery_confusion(known_matrix: np.ndarray, known_rows: list[dict],
                       unknown_matrix: np.ndarray, unknown_rows: list[dict], threshold: float) -> dict:
    columns = d.KNOWN + ["REJECT"]
    rows = d.KNOWN + ["UNKNOWN"]
    counts = {label: {column: 0 for column in columns} for label in rows}
    for row, scores in zip(known_rows, known_matrix):
        pred = int(np.argmax(scores))
        decision = d.KNOWN[pred] if float(np.max(scores)) >= threshold else "REJECT"
        counts[str(row["label_id"])][decision] += 1
    for scores in unknown_matrix:
        pred = int(np.argmax(scores))
        decision = d.KNOWN[pred] if float(np.max(scores)) >= threshold else "REJECT"
        counts["UNKNOWN"][decision] += 1
    return {"row_labels": rows, "column_labels": columns, "counts": counts,
            "unknown_false_identification_count": int(sum(counts["UNKNOWN"][label] for label in d.KNOWN)),
            "unknown_rejection_count": int(counts["UNKNOWN"]["REJECT"])}


def _root_mean_interval(values: list[float], seed: int, replicates: int = 2000) -> list[float]:
    rng = np.random.default_rng(int(seed) ^ 0xD5129)
    data = np.asarray(values, dtype=np.float64)
    picks = rng.integers(0, len(data), size=(replicates, len(data)))
    means = np.mean(data[picks], axis=1)
    return [float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))]


def _independent_root_delta_interval(left: list[float], right: list[float], seed: int,
                                     replicates: int = 10000) -> list[float]:
    rng = np.random.default_rng(int(seed) ^ 0xD510B)
    a, b = np.asarray(left, dtype=np.float64), np.asarray(right, dtype=np.float64)
    a_picks = rng.integers(0, len(a), size=(replicates, len(a)))
    b_picks = rng.integers(0, len(b), size=(replicates, len(b)))
    delta = np.mean(a[a_picks], axis=1) - np.mean(b[b_picks], axis=1)
    return [float(np.quantile(delta, 0.025)), float(np.quantile(delta, 0.975))]


def _finish_from_saved_historical_finals(paths: dict, registry: dict, main_registry: dict,
                                         finals: list[dict]) -> dict:
    if len(finals) != len(registry["galleries"]) or len(finals) != 80:
        raise ValueError("saved historical final evidence is incomplete")
    for final in finals:
        prediction = final["predictions"]
        if (final.get("status") != "complete_verified" or final.get("final_role_opened") is not True or
                sha256_file(Path(prediction["path"])) != prediction.get("sha256")):
            raise ValueError("saved historical final evidence failed verification")
    by_family = defaultdict(list)
    backend_comparisons = {}
    for final in finals:
        by_family[(final["method"], final["backend"], final["policy"])].append(final)
        pair = final.get("float_int8_comparison")
        if pair is None:
            raise ValueError("historical float/INT8 comparison is missing")
        backend_comparisons["%s_seed-%d_%s" % (final["method"], final["root_seed"], final["policy"])] = pair
    aggregate = {}
    for key, rows in by_family.items():
        if len(rows) != 10:
            raise ValueError("historical family does not contain ten roots: %s" % (key,))
        values_by_metric = {
            "known_macro_identification": [r["metrics"]["known_macro_identification"] for r in rows],
            "known_macro_correct_acceptance": [r["metrics"]["known_macro_correct_acceptance"] for r in rows],
            "unknown_false_identification_rate": [r["metrics"]["unknown_false_identification_rate"] for r in rows],
            "unknown_rejection_rate": [r["metrics"]["unknown_rejection_rate"] for r in rows],
        }
        aggregate["%s_%s_%s" % key] = {
            "root_count": len(rows),
            "metrics_mean_across_roots": {name: float(np.mean(values)) for name, values in values_by_metric.items()},
            "metrics_sd_across_roots": {name: float(np.std(values, ddof=1)) for name, values in values_by_metric.items()},
            "root_mean_bootstrap_intervals_95_percent": {
                name: _root_mean_interval(values, 20261900 + index)
                for index, (name, values) in enumerate(values_by_metric.items())},
            "selected_template_counts": [r["selected_template_count"] for r in rows],
            "per_root_image_group_bootstrap_intervals_95_percent": [r["bootstrap"]["intervals_95_percent"] for r in rows],
            "per_root_metrics": [r["metrics"] for r in rows],
        }
    main_summary_path = paths["handoff"] / "FINAL_SUMMARY.json"
    if not main_summary_path.is_file():
        raise FileNotFoundError("main D final summary is required for same-role historical comparison")
    main_summary = json.loads(main_summary_path.read_text(encoding="utf-8"))
    task_gallery = main_summary.get("gallery_final_by_family", {})
    comparisons = []
    compared_metrics = ("known_macro_identification", "known_macro_correct_acceptance",
                        "unknown_false_identification_rate", "unknown_rejection_rate")
    for method in METHODS:
        for backend in BACKENDS:
            for policy in POLICIES:
                hist_key = "%s_%s_%s" % (method, backend, policy)
                hist_values = aggregate[hist_key]["metrics_mean_across_roots"]
                for width_tag in ("010", "025"):
                    task_key = "a%s_%s_%s" % (width_tag, backend, policy)
                    task_values = task_gallery.get(task_key)
                    if not task_values:
                        continue
                    folder = "a01" if width_tag == "010" else "a025"
                    task_root_paths = sorted((paths["runs"] / "gallery" / folder).glob(
                        "seed-*/" + backend + "_" + policy + ".FINAL.json"))
                    task_root_rows = [json.loads(path.read_text(encoding="utf-8")) for path in task_root_paths]
                    if len(task_root_rows) != 10:
                        raise ValueError("same-role D gallery comparison requires ten final root results for " + task_key)
                    task_map = {
                        "known_macro_identification": task_values["known_macro_identification_mean"],
                        "known_macro_correct_acceptance": task_values["known_macro_correct_acceptance_mean"],
                        "unknown_false_identification_rate": task_values["unknown_fpir_mean"],
                        "unknown_rejection_rate": task_values["unknown_rejection_mean"],
                    }
                    intervals = {}
                    hist_rows = [r for r in finals if r["method"] == method and
                                 r["backend"] == backend and r["policy"] == policy]
                    for index, metric in enumerate(compared_metrics):
                        intervals[metric] = _independent_root_delta_interval(
                            [r["metrics"][metric] for r in task_root_rows],
                            [r["metrics"][metric] for r in hist_rows],
                            20261950 + len(comparisons) * len(compared_metrics) + index)
                    comparisons.append({
                        "historical_family": hist_key, "D_finetuned_family": task_key,
                        "historical_root_count": len(hist_rows), "D_root_count": task_values["root_count"],
                        "delta_D_minus_historical": {
                            metric: float(task_map[metric] - hist_values[metric]) for metric in compared_metrics},
                        "independent_seed_resampling_interval_95_percent_for_delta": intervals,
                        "interval_interpretation": "seed-to-seed variability only; it is not a population interval because final image groups are shared",
                        "interpretation": "descriptive same-final-role comparison; root sets are independent and all five identities were exposed in v3 development"})
    sample_counts = finals[0]["sample_counts"]
    registry_sha_before_final = sha256_file(paths["historical_registry"])
    output = {"schema": "job-d-historical-final-summary-1", "status": "complete_verified",
              "historical_exploratory_only": True, "reason": registry["known_identity_exposure"],
              "audit_path": registry["audit_path"], "registry_path": str(paths["historical_registry"]),
              "registry_sha256_before_final": registry_sha_before_final,
              "result_count": len(finals), "family_metrics": aggregate,
              "final_role_opened": True,
              "final_role_counts": {"known": sample_counts["known"], "unknown": sample_counts["unknown"]},
              "float_int8_final_comparisons": backend_comparisons,
              "same_role_descriptive_comparison_to_D_finetuned_galleries": comparisons,
              "created_utc": now()}
    write_json(paths["historical_final"], output)
    registry["final_role_opened"] = True
    registry["final_role_opened_utc"] = now()
    registry["final_summary_path"] = str(paths["historical_final"])
    write_json(paths["historical_registry"], registry)
    summary = json.loads(main_summary_path.read_text(encoding="utf-8"))
    summary["historical_v3_gallery_control"] = output
    summary["historical_control_registry_sha256_after_final"] = sha256_file(paths["historical_registry"])
    summary["v3_comparison"]["historical_control_result_path"] = str(paths["historical_final"])
    summary["v3_comparison"]["historical_control_interpretation"] = (
        "Exploratory only: all K01-K05 source images were in v3 development, which selected historical checkpoints.")
    summary["v3_comparison"]["same_role_descriptive_deltas"] = comparisons
    write_json(main_summary_path, summary)
    main_registry["historical_control_final_summary"] = str(paths["historical_final"])
    main_registry["historical_control_final_summary_sha256"] = sha256_file(paths["historical_final"])
    main_registry["historical_control_registry_sha256_after_final"] = sha256_file(paths["historical_registry"])
    write_json(paths["reports"] / "MODEL_REGISTRY_FROZEN.json", main_registry)
    summary["model_registry_sha256"] = sha256_file(paths["reports"] / "MODEL_REGISTRY_FROZEN.json")
    write_json(main_summary_path, summary)
    return output


def finish_historical_final(workspace: Path) -> dict:
    paths = _study_paths(workspace)
    registry = _verify_registry(paths)
    main_registry_path = paths["reports"] / "MODEL_REGISTRY_FROZEN.json"
    if not main_registry_path.is_file():
        raise PermissionError("main D model registry must be frozen before historical final evaluation")
    main_registry = json.loads(main_registry_path.read_text(encoding="utf-8"))
    if main_registry.get("status") != "complete_verified":
        raise PermissionError("main D model registry is incomplete")
    saved_final_paths = [Path(row["result_path"]).with_suffix(".FINAL.json") for row in registry["galleries"]]
    if all(path.is_file() for path in saved_final_paths):
        saved_finals = [json.loads(path.read_text(encoding="utf-8")) for path in saved_final_paths]
        return _finish_from_saved_historical_finals(paths, registry, main_registry, saved_finals)
    receipt = json.loads(Path(registry["galleries"][0]["freeze_receipt"]).read_text(encoding="utf-8"))
    known_rows = d.dataset_role_rows(workspace, paths["primary"], "survface_96", ("final",),
                                     "evaluate_final", receipt)
    unknown_rows = d.dataset_role_rows(workspace, paths["primary"], "survface_96", ("unknown_final",),
                                       "evaluate_final", receipt)
    known_gray = d.cache_gray8(workspace, known_rows, "evaluate_final", receipt)
    unknown_gray = d.cache_gray8(workspace, unknown_rows, "evaluate_final", receipt)
    base = d.load_role_arrays(workspace, paths, "survface_96")
    train_rows = [r for r in base["train_rows"] if r.get("identity_partition") != "unknown"
                  and r.get("label_id") in d.KNOWN]
    train_indices = [i for i, r in enumerate(base["train_rows"])
                     if r.get("identity_partition") != "unknown" and r.get("label_id") in d.KNOWN]
    train_gray = base["train_gray"][train_indices]
    final_data = {"known_rows": known_rows, "unknown_rows": unknown_rows,
                  "known_gray": known_gray, "unknown_gray": unknown_gray}
    cache = {}
    finals = []
    final_arrays = {}
    for gallery in registry["galleries"]:
        result_path = Path(gallery["result_path"])
        result = json.loads(result_path.read_text(encoding="utf-8"))
        method, seed, backend, policy = (result[k] for k in ("method", "root_seed", "backend", "policy"))
        key = (method, seed, backend)
        if key not in cache:
            model_path = Path(result["model_path"])
            all_gray = np.concatenate([train_gray, known_gray, unknown_gray], axis=0)
            features = d._normalize_vectors(d.tflite_outputs(model_path, all_gray))
            rows = train_rows + known_rows + unknown_rows
            cache[key] = {str(row["record_id"]): features[i] for i, row in enumerate(rows)}
        embedding_map = cache[key]
        selection = result["selected"]["selection"]
        prototypes = []
        for label in d.KNOWN:
            values = np.stack([embedding_map[rid] for rid in selection[label]["record_ids"]])
            prototypes.append(d._normalize_vectors(np.mean(values, axis=0, keepdims=True))[0])
        prototypes = np.stack(prototypes)
        known_matrix = np.stack([embedding_map[str(r["record_id"])] for r in known_rows]) @ prototypes.T
        unknown_matrix = np.stack([embedding_map[str(r["record_id"])] for r in unknown_rows]) @ prototypes.T
        threshold = float(result["selected"]["calibration"]["threshold"])
        metrics = d.gallery_metrics(known_matrix, known_rows, unknown_matrix, unknown_rows, threshold)
        final_arrays.setdefault((method, seed, policy), {})[backend] = (
            known_matrix, unknown_matrix, threshold,
            result["selected"]["template_count_requested"])
        known_pred, known_scores = np.argmax(known_matrix, axis=1), np.max(known_matrix, axis=1)
        unknown_pred, unknown_scores = np.argmax(unknown_matrix, axis=1), np.max(unknown_matrix, axis=1)
        known_accept, unknown_accept = known_scores >= threshold, unknown_scores >= threshold
        decisions = [d.KNOWN[int(p)] if a else "REJECT" for p, a in zip(known_pred, known_accept)] + \
                    [d.KNOWN[int(p)] if a else "REJECT" for p, a in zip(unknown_pred, unknown_accept)]
        truths = [str(row["label_id"]) for row in known_rows] + ["UNKNOWN"] * len(unknown_rows)
        prediction_rows = d._prediction_rows(
            known_rows + unknown_rows, truths, decisions,
            np.concatenate([known_pred, unknown_pred]),
            np.concatenate([known_scores, unknown_scores]),
            np.concatenate([known_accept, unknown_accept]), backend + "_historical_gallery")
        prediction_path = result_path.with_suffix(".FINAL_PREDICTIONS.jsonl")
        prediction_sha = d._write_predictions(prediction_path, prediction_rows)
        all_scores = np.concatenate([known_scores, unknown_scores])
        thresholds = sorted(set(float(v) for v in np.quantile(all_scores, np.linspace(0.0, 1.0, 101))))
        thresholds.append(float(np.nextafter(max(thresholds), np.inf)))
        true_known = np.asarray([d.row_label_index(row) for row in known_rows])
        curve = [{"threshold": point,
                  "tpir_correct_identification": float(np.mean((known_scores >= point) & (known_pred == true_known))),
                  "fpir_unknown_false_identification": float(np.mean(unknown_scores >= point)),
                  "known_acceptance": float(np.mean(known_scores >= point))}
                 for point in thresholds]
        final_path = result_path.with_suffix(".FINAL.json")
        final = {"schema": "job-d-historical-gallery-final-1", "status": "complete_verified",
                 "method": method, "root_seed": seed, "backend": backend, "policy": policy,
                 "selected_template_count": result["selected"]["template_count_requested"],
                 "selected_templates": selection, "threshold": threshold,
                 "calibration": result["selected"]["calibration"], "metrics": metrics,
                 "per_person_confusion": _gallery_confusion(known_matrix, known_rows, unknown_matrix,
                                                            unknown_rows, threshold),
                 "bootstrap": d.bootstrap_gallery_metrics(known_matrix, known_rows, unknown_matrix,
                                                           unknown_rows, threshold, int(seed), 1000),
                 "reliability": d.gallery_reliability(known_matrix, known_rows, unknown_matrix, threshold),
                 "tpir_fpir_curve": curve,
                 "predictions": {"path": str(prediction_path), "sha256": prediction_sha},
                 "final_role_opened": True, "evidence_tier": "exploratory_v3_development_identity_exposure",
                 "sample_counts": {"known": len(known_rows), "unknown": len(unknown_rows),
                                   "unknown_identity_count": len({r["identity_id"] for r in unknown_rows})},
                 "frozen_receipt": gallery["freeze_receipt"], "created_utc": now()}
        write_json(final_path, final)
        freeze_receipt = json.loads(Path(gallery["freeze_receipt"]).read_text(encoding="utf-8"))
        freeze_receipt["final_role_opened"] = True
        freeze_receipt["final_opened_utc"] = now()
        write_json(Path(gallery["freeze_receipt"]), freeze_receipt)
        finals.append(final)
    backend_comparisons = {}
    for (method, seed, policy), backends in final_arrays.items():
        if set(backends) != {"float", "int8"}:
            raise ValueError("historical float/int8 final comparison is incomplete")
        fk, fu, ft, fcount = backends["float"]
        ik, iu, it, icount = backends["int8"]
        fpred = np.concatenate([np.argmax(fk, axis=1), np.argmax(fu, axis=1)])
        ipred = np.concatenate([np.argmax(ik, axis=1), np.argmax(iu, axis=1)])
        faccept = np.concatenate([np.max(fk, axis=1) >= ft, np.max(fu, axis=1) >= ft])
        iaccept = np.concatenate([np.max(ik, axis=1) >= it, np.max(iu, axis=1) >= it])
        pair = {"method": method, "root_seed": seed, "policy": policy,
                "float_template_count": fcount, "int8_template_count": icount,
                "float_threshold": ft, "int8_threshold": it,
                "int8_minus_float_threshold_drift": float(it - ft),
                "final_decision_disagreement_count": int(np.sum((fpred != ipred) | (faccept != iaccept))),
                "final_decision_disagreement_rate": float(np.mean((fpred != ipred) | (faccept != iaccept))),
                "final_score_mean_absolute_difference": float(np.mean(np.abs(
                    np.concatenate([np.max(fk, axis=1), np.max(fu, axis=1)]) -
                    np.concatenate([np.max(ik, axis=1), np.max(iu, axis=1)]))))}
        backend_comparisons["%s_seed-%d_%s" % (method, seed, policy)] = pair
        for final in finals:
            if final["method"] == method and final["root_seed"] == seed and final["policy"] == policy:
                final["float_int8_comparison"] = pair
                result_path = _gallery_dir(paths, method, seed) / (final["backend"] + "_" + policy + ".FINAL.json")
                write_json(result_path, final)
    by_family = defaultdict(list)
    for result in finals:
        by_family[(result["method"], result["backend"], result["policy"])].append(result)
    aggregate = {}
    for key, rows in by_family.items():
        values_by_metric = {
            "known_macro_identification": [r["metrics"]["known_macro_identification"] for r in rows],
            "known_macro_correct_acceptance": [r["metrics"]["known_macro_correct_acceptance"] for r in rows],
            "unknown_false_identification_rate": [r["metrics"]["unknown_false_identification_rate"] for r in rows],
            "unknown_rejection_rate": [r["metrics"]["unknown_rejection_rate"] for r in rows],
        }
        aggregate["%s_%s_%s" % key] = {
            "root_count": len(rows),
            "metrics_mean_across_roots": {name: float(np.mean(values)) for name, values in values_by_metric.items()},
            "metrics_sd_across_roots": {name: float(np.std(values, ddof=1)) for name, values in values_by_metric.items()},
            "root_mean_bootstrap_intervals_95_percent": {name: _root_mean_interval(values, 20261900 + index)
                                                           for index, (name, values) in enumerate(values_by_metric.items())},
            "selected_template_counts": [r["selected_template_count"] for r in rows],
            "per_root_image_group_bootstrap_intervals_95_percent": [r["bootstrap"]["intervals_95_percent"] for r in rows],
            "per_root_metrics": [r["metrics"] for r in rows],
        }
    main_summary_path = paths["handoff"] / "FINAL_SUMMARY.json"
    if not main_summary_path.is_file():
        raise FileNotFoundError("main D final summary is required for same-role historical comparison")
    main_summary = json.loads(main_summary_path.read_text(encoding="utf-8"))
    task_gallery = main_summary.get("gallery_final_by_family", {})
    comparisons = []
    compared_metrics = ("known_macro_identification", "known_macro_correct_acceptance",
                        "unknown_false_identification_rate", "unknown_rejection_rate")
    for method in METHODS:
        for backend in BACKENDS:
            for policy in POLICIES:
                hist_key = "%s_%s_%s" % (method, backend, policy)
                hist_values = aggregate[hist_key]["metrics_mean_across_roots"]
                for width_tag in ("010", "025"):
                    task_key = "a%s_%s_%s" % (width_tag, backend, policy)
                    task_values = task_gallery.get(task_key)
                    if not task_values:
                        continue
                    task_gallery_folder = "a01" if width_tag == "010" else "a025"
                    task_root_paths = sorted((paths["runs"] / "gallery" / task_gallery_folder).glob(
                        "seed-*/" + backend + "_" + policy + ".FINAL.json"))
                    task_root_rows = [json.loads(path.read_text(encoding="utf-8")) for path in task_root_paths]
                    if len(task_root_rows) != 10:
                        raise ValueError("same-role D gallery comparison requires ten final root results for " + task_key)
                    task_map = {
                        "known_macro_identification": task_values["known_macro_identification_mean"],
                        "known_macro_correct_acceptance": task_values["known_macro_correct_acceptance_mean"],
                        "unknown_false_identification_rate": task_values["unknown_fpir_mean"],
                        "unknown_rejection_rate": task_values["unknown_rejection_mean"],
                    }
                    per_metric_intervals = {}
                    for metric in compared_metrics:
                        left_values = [r["metrics"][metric] for r in task_root_rows]
                        right_values = [r["metrics"][metric] for r in finals
                                        if r["method"] == method and r["backend"] == backend and r["policy"] == policy]
                        per_metric_intervals[metric] = _independent_root_delta_interval(
                            left_values, right_values, 20261950 + len(comparisons))
                    comparisons.append({"historical_family": hist_key, "D_finetuned_family": task_key,
                        "historical_root_count": aggregate[hist_key]["root_count"],
                        "D_root_count": task_values["root_count"],
                        "delta_D_minus_historical": {metric: float(task_map[metric] - hist_values[metric])
                                                     for metric in compared_metrics},
                        "independent_seed_resampling_interval_95_percent_for_delta": per_metric_intervals,
                        "interval_interpretation": "seed-to-seed variability only; it is not a population interval because final image groups are shared",
                        "interpretation": "descriptive same-final-role comparison; root sets are independent and all five identities were exposed in v3 development"})
    output = {"schema": "job-d-historical-final-summary-1", "status": "complete_verified",
              "historical_exploratory_only": True, "reason": registry["known_identity_exposure"],
              "audit_path": registry["audit_path"], "registry_path": str(paths["historical_registry"]),
              "registry_sha256_before_final": sha256_file(paths["historical_registry"]),
              "result_count": len(finals), "family_metrics": aggregate,
              "final_role_opened": True, "final_role_counts": {"known": len(known_rows), "unknown": len(unknown_rows)},
              "float_int8_final_comparisons": backend_comparisons,
              "same_role_descriptive_comparison_to_D_finetuned_galleries": comparisons,
              "created_utc": now()}
    write_json(paths["historical_final"], output)
    registry["final_role_opened"] = True
    registry["final_role_opened_utc"] = now()
    registry["final_summary_path"] = str(paths["historical_final"])
    write_json(paths["historical_registry"], registry)
    # Link the comparison into the main machine summary after all controls finish.
    summary_path = paths["handoff"] / "FINAL_SUMMARY.json"
    if summary_path.is_file():
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        summary["historical_v3_gallery_control"] = output
        summary["historical_control_registry_sha256_after_final"] = sha256_file(paths["historical_registry"])
        summary["v3_comparison"]["historical_control_result_path"] = str(paths["historical_final"])
        summary["v3_comparison"]["historical_control_interpretation"] = "Exploratory only: all K01-K05 source images were in v3 development, which selected historical checkpoints."
        summary["v3_comparison"]["same_role_descriptive_deltas"] = comparisons
        write_json(summary_path, summary)
        main_registry["historical_control_final_summary"] = str(paths["historical_final"])
        main_registry["historical_control_final_summary_sha256"] = sha256_file(paths["historical_final"])
        main_registry["historical_control_registry_sha256_after_final"] = sha256_file(paths["historical_registry"])
        write_json(main_registry_path, main_registry)
        summary["model_registry_sha256"] = sha256_file(main_registry_path)
        write_json(summary_path, summary)
    return output


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", type=Path, default=Path.cwd())
    parser.add_argument("command", choices=("audit", "train", "finish"))
    args = parser.parse_args()
    workspace = args.workspace.resolve()
    paths = _study_paths(workspace)
    if args.command == "audit":
        audit = audit_historical(workspace, paths)
        print(json.dumps({"status": audit["status"], "root_count": len(audit["per_file_training_and_artifact_audit"]),
                          "path": str(paths["historical_audit"]), "errors": audit["errors"]}))
    elif args.command == "train":
        with d.resource_lease(paths["study"], "heavy_training", owner="D_historical_gallery_controls", timeout_s=0):
            result = run_development_campaign(workspace)
        print(json.dumps({"status": "complete_verified", "development_path": str(paths["historical_results"]),
                          "registry_path": str(paths["historical_registry"]),
                          "result_count": result["development"]["result_count"]}))
    else:
        result = finish_historical_final(workspace)
        print(json.dumps({"status": result["status"], "path": str(paths["historical_final"]),
                          "result_count": result["result_count"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
