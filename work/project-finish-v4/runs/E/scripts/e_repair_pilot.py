#!/usr/bin/env python3
"""Complete the repaired wider-weight, surgery, conversion and MAC pilot."""
from __future__ import annotations

import json
import os
import sys
import time
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[5]
STUDY = ROOT / "work/project-finish-v4"
PILOT_DIR = STUDY / "runs/E/pilot"
os.environ["KERAS_HOME"] = str(STUDY / "models/keras_cache")
os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("TF_NUM_INTRAOP_THREADS", "4")
os.environ.setdefault("TF_NUM_INTEROP_THREADS", "2")
sys.path.insert(0, str(ROOT / "FaceRecognition"))

import numpy as np
import tensorflow as tf

from nicla_study import d_campaign as d
from nicla_study.macs import layer_macs, tflite_operator_macs, compare_mac_accounts
from nicla_study.state import resource_lease
from e_pilot import (digest_file, imagenet_gray_mobilenet, load_model,
                     create_physical_channel_surgery, export_int8, write_json)


def corrected_keras_macs(model):
    rows, total, seen_layers, seen_models = [], 0, set(), set()

    def visit(container):
        nonlocal total
        if id(container) in seen_models:
            return
        seen_models.add(id(container))
        for layer in container.layers:
            if id(layer) in seen_layers:
                continue
            seen_layers.add(id(layer))
            if isinstance(layer, tf.keras.Model):
                visit(layer)
                continue
            kind = type(layer).__name__
            if kind not in {"Conv2D", "DepthwiseConv2D", "SeparableConv2D", "Dense"}:
                continue
            kwargs = {"input_shape": layer.input_shape, "output_shape": layer.output_shape}
            if kind in {"Conv2D", "DepthwiseConv2D", "SeparableConv2D"}:
                kwargs["kernel_size"] = layer.kernel_size
            if kind == "Conv2D":
                kwargs["filters"] = int(layer.filters)
                kwargs["groups"] = int(getattr(layer, "groups", 1))
            elif kind == "DepthwiseConv2D":
                kwargs["depth_multiplier"] = int(layer.depth_multiplier)
            elif kind == "SeparableConv2D":
                kwargs["filters"] = int(layer.filters)
                kwargs["depth_multiplier"] = int(layer.depth_multiplier)
            else:
                kwargs["units"] = int(layer.units)
            count = layer_macs(kind, **kwargs)
            rows.append({"name": layer.name, "type": kind, "macs": count})
            total += count

    visit(model)
    return {"method": "recursive_unique_model_layers_v1",
            "macs_per_inference": total, "counted_layers": rows, "complete": True}


def main():
    tf.config.threading.set_intra_op_parallelism_threads(4)
    tf.config.threading.set_inter_op_parallelism_threads(1)
    repair = {"schema": "job-e-pilot-repair-1", "status": "in_progress",
              "final_role_accessed": False, "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
              "steps": [], "failures": []}
    PILOT_DIR.mkdir(parents=True, exist_ok=True)
    with resource_lease(STUDY, "heavy_training", owner="E_repair_wider_surgery_macs_pilot", timeout_s=0):
        wider_rows = []
        for alpha in (0.50, 0.75):
            try:
                _, _, info = imagenet_gray_mobilenet(ROOT, alpha, True, 76101)
                row = {"alpha": alpha, "status": "pass", "checkpoint_path": info["checkpoint_path"],
                       "checkpoint_sha256": info["checkpoint_sha256"],
                       "checkpoint_bytes": Path(info["checkpoint_path"]).stat().st_size,
                       "checkpoint_source_url": info["checkpoint_source_url"],
                       "backbone_feature_shape": info["backbone_feature_shape"],
                       "grayscale_adaptation": info["rgb_to_gray_weight_transfer"],
                       "equivalence": info["rgb_grayscale_numeric_equivalence"]}
                wider_rows.append(row)
                repair["steps"].append({"id": "imagenet_gray_alpha_%03d" % int(alpha * 100), "status": "pass"})
                tf.keras.backend.clear_session()
            except Exception as exc:
                row = {"alpha": alpha, "status": "failed", "error_type": type(exc).__name__,
                       "error": str(exc)[:1200], "traceback_tail": traceback.format_exc()[-2500:]}
                wider_rows.append(row)
                repair["failures"].append(row)
        repair["wider_pretrained_grayscale_adaptation"] = wider_rows

        paths = d.study_paths(ROOT)
        data = d.load_role_arrays(ROOT, paths, "survface_96")
        root_rows = d.read_jsonl(STUDY / "runs/D/survface/sv_mnv1_a025_uniform_select_6/seed-20261101/training_rows.jsonl")
        train_gray = d.cache_gray8(ROOT, root_rows, "train")
        representatives = np.linspace(0, len(train_gray) - 1, min(32, len(train_gray)), dtype=np.int64).tolist()
        conversion_rows = []
        for alpha in (0.50, 0.75):
            if not any(row.get("alpha") == alpha and row["status"] == "pass" for row in wider_rows):
                continue
            try:
                model, _, info = load_model(alpha, 76101, True)
                label = "capacity_a%03d_ptq_int8" % int(alpha * 100)
                row = export_int8(model, train_gray, PILOT_DIR / (label + ".tflite"), representatives)
                row.update({"alpha": alpha, "status": "pass", "initialization": info.get("initialization")})
                conversion_rows.append(row)
                repair["steps"].append({"id": label, "status": "pass"})
                del model
                tf.keras.backend.clear_session()
            except Exception as exc:
                row = {"alpha": alpha, "status": "unsupported_or_failed", "error_type": type(exc).__name__,
                       "error": str(exc)[:1200], "traceback_tail": traceback.format_exc()[-2500:]}
                conversion_rows.append(row)
                repair["failures"].append(row)
        repair["wider_int8_conversions"] = conversion_rows

        try:
            model, _, _ = load_model(0.25, 20261101, True,
                STUDY / "runs/D/survface/sv_mnv1_a025_uniform_select_6/seed-20261101/classifier.weights.h5")
            physical, masked, surgery = create_physical_channel_surgery(model)
            x = d.gray_float(train_gray[:8])
            expected = masked(x, training=False).numpy()
            actual = physical(x, training=False).numpy()
            surgery["masked_parent_max_abs_error"] = float(np.max(np.abs(expected - actual)))
            surgery["masked_parent_graph_equivalence"] = bool(np.allclose(expected, actual, atol=1e-5, rtol=1e-5))
            surgery["logit_shape"] = [int(value) if value is not None else None for value in physical.output_shape]
            surgery["parameter_count_before"] = int(model.count_params())
            surgery["parameter_count_after"] = int(physical.count_params())
            surgery["pilot_pass"] = surgery["masked_parent_graph_equivalence"]
            repair["physical_channel_surgery"] = surgery
            repair["steps"].append({"id": "physical_channel_graph_surgery", "status": "pass" if surgery["pilot_pass"] else "failed"})
            del model, physical, masked
            tf.keras.backend.clear_session()
        except Exception as exc:
            failure = {"status": "unsupported_or_failed", "error_type": type(exc).__name__,
                       "error": str(exc)[:1200], "traceback_tail": traceback.format_exc()[-2500:]}
            repair["physical_channel_surgery"] = failure
            repair["failures"].append(failure)

        # Recompute MAC agreement on each original and new full-int8 model using one copy
        # of each layer in the selected backbone and classifier head.
        all_conversions = []
        original = json.loads((PILOT_DIR / "PILOT_RESULT.json").read_text(encoding="utf-8"))
        all_conversions.extend(original.get("capacity_int8_conversions", []))
        all_conversions.extend(conversion_rows)
        corrected_rows = []
        for conv in all_conversions:
            if conv.get("status") != "pass" or not conv.get("path"):
                continue
            alpha = float(conv["alpha"])
            seed = 76101 if alpha in (0.50, 0.75) else 20261101
            weight_path = None if alpha in (0.50, 0.75) else STUDY / "runs/D/survface" / ("sv_mnv1_a010_transfer_6" if alpha == 0.10 else "sv_mnv1_a025_uniform_select_6") / ("seed-%d" % seed) / "classifier.weights.h5"
            try:
                model, _, _ = load_model(alpha, seed, True, weight_path)
                interpreter = tf.lite.Interpreter(model_path=conv["path"], num_threads=2)
                interpreter.allocate_tensors()
                km = corrected_keras_macs(model)
                tm = tflite_operator_macs(interpreter)
                corrected_rows.append({"alpha": alpha, "path": conv["path"],
                                       "mac_accounting": compare_mac_accounts(km, tm),
                                       "keras_macs": km, "tflite_macs": tm})
                del model, interpreter
                tf.keras.backend.clear_session()
            except Exception as exc:
                corrected_rows.append({"alpha": alpha, "path": conv["path"], "status": "failed",
                                       "error_type": type(exc).__name__, "error": str(exc)[:1200]})
        repair["corrected_mac_accounting"] = corrected_rows
        repair["role_receipt"] = {"train_rows_for_representatives": len(root_rows),
                                  "calibration_known": len(data["cal_known"]),
                                  "calibration_unknown": len(data["cal_unknown"]),
                                  "development_known": len(data["dev_known"]),
                                  "development_unknown": len(data["dev_unknown"]),
                                  "final_role_accessed": False}

    repair["finished_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    repair["status"] = "complete" if not repair["failures"] and all(row["status"] == "pass" for row in repair["steps"]) else "partial"
    write_json(PILOT_DIR / "PILOT_REPAIRED.json", repair)
    print(json.dumps({"event": "pilot_repair_complete", "status": repair["status"],
                      "steps": repair["steps"], "report": str(PILOT_DIR / "PILOT_REPAIRED.json")}), flush=True)


if __name__ == "__main__":
    main()
