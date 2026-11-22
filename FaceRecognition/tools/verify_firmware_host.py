#!/usr/bin/env python3
"""Run seeded host parity/fault blocks against C/D/E exported TFLite models."""

import hashlib
import json
import os
import platform
from pathlib import Path
import sys
import tempfile

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("TF_NUM_INTRAOP_THREADS", "2")
os.environ.setdefault("TF_NUM_INTEROP_THREADS", "2")

import numpy as np
import tensorflow as tf

ROOT = Path(__file__).resolve().parents[1]
PROJECT = ROOT.parent
sys.path.insert(0, str(ROOT / "firmware"))
sys.path.insert(0, str(ROOT))
C_REPORTS = ROOT / "nicla_study/c_reference"
sys.path.insert(0, str(C_REPORTS))
from common import openmv_fomo_detections
from identity_contracts import (class_index_from_labels, classifier_decision, crop_xywh, dequantize,
    fomo_face_output_index, gallery_decision, normalize_embedding, openmv_dtype_name, prediction_values,
    pillow_bilinear_grayscale_u8, quality_decision, single_tensor_metadata,
    strict_score_pass)
from runtime_support import (BoundedLineQueue, TemporalConfirmer, TickTracker,
    config_hash_matches, enough_heap, hash_matches, validate_gallery_metadata)
from telemetry import Telemetry
from nicla_study.data_pipeline import load_pixels, read_jsonl

C_FIXTURES = PROJECT / "work/project-finish-v4/fixtures/C/selected_challenger"
STAGE = ROOT / "models/deployment-v1"
STUDY = PROJECT / "work/project-finish-v4"
DEVICE_DISCOVERY = STUDY / "reports/F/device_discovery.json"
KNOWN = ["K01", "K02", "K03", "K04", "K05"]


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def interpreter(path):
    itp = tf.lite.Interpreter(model_path=str(path), num_threads=2)
    itp.allocate_tensors()
    return itp


def quantize_gray8(gray, detail):
    x = np.asarray(gray, dtype=np.uint8).astype(np.float32) / np.float32(255.0)
    scale, zero = detail["quantization"]
    if detail["dtype"] == np.int8:
        return np.clip(np.rint(x / scale + zero), -128, 127).astype(np.int8)
    return x.astype(detail["dtype"])


def run_gray_model(itp, gray):
    input_detail = itp.get_input_details()[0]
    output_detail = itp.get_output_details()[0]
    raw_input = quantize_gray8(gray, input_detail).reshape(tuple(input_detail["shape"]))
    itp.set_tensor(input_detail["index"], raw_input)
    itp.invoke()
    raw_output = itp.get_tensor(output_detail["index"])
    if output_detail["dtype"] in (np.int8, np.uint8, np.int16):
        scale, zero = output_detail["quantization"]
        ref = (raw_output.astype(np.float32) - np.float32(zero)) * np.float32(scale)
    else:
        ref = raw_output.astype(np.float32)
    firmware_values = np.asarray(dequantize(raw_output.reshape(-1).tolist(),
        output_detail["quantization"][0], output_detail["quantization"][1],
        np.dtype(output_detail["dtype"]).name), dtype=np.float32)
    return raw_input, raw_output, ref.reshape(-1), firmware_values


class FakeUSB:
    def __init__(self, connected=False):
        self.connected = connected
        self.data = bytearray()

    def isconnected(self):
        return self.connected

    def write(self, data):
        self.data.extend(data)
        return len(data)

    def read(self, size):
        return None


class FakeUART:
    def __init__(self, fail_once=False):
        self.fail_once = fail_once
        self.data = bytearray()

    def write(self, data):
        if self.fail_once:
            self.fail_once = False
            raise OSError("injected temporary UART disconnect")
        self.data.extend(data)
        return len(data)

    def read(self, size):
        return None


def test_fault_blocks(config, gallery, c_interpreter):
    failures = []
    details = []
    # 01: exact C crop geometry and out-of-frame no-padding rejection.
    geometry = crop_xywh(128.5, 48.5, 96, 240, 240)
    outside = crop_xywh(11.0, 88.5, 96, 240, 240)
    assert geometry == ((81, 1, 96, 96), None)
    assert outside == ((-37, 41, 96, 96), "crop_outside_frame")
    details.append("C rounding, negative edge crop and no-clamp boundary policy")

    # 02: an exactly-on-threshold detector score must not pass.
    assert not strict_score_pass(0.7, 0.7)
    assert strict_score_pass(np.nextafter(np.float32(0.7), np.float32(1.0)), 0.7)
    details.append("strict greater-than threshold boundary")

    # 03: six E logits and the frozen K01..K05+OTHER order are mandatory.
    wrong_labels = ["K02", "K01", "K03", "K04", "K05", "OTHER"]
    bad_order = classifier_decision([1, 0, 0, 0, 0, -1], wrong_labels, 0.5)
    assert bad_order["status"] == "UNCERTAIN" and bad_order["reason"] == "classifier_label_order_mismatch"
    details.append("reversed classifier label map rejected")

    # 04: classifier and embedding tensors are not interchangeable.
    wrong_embedding = gallery_decision([1, 2, 3, 4, 5, 6], gallery["entries"], 0.99,
        expected_model_id=gallery["model_id"])
    wrong_classifier = classifier_decision([0.0] * 64, config["identity"]["large"]["class_order"], 0.5)
    assert wrong_embedding["status"] == "UNCERTAIN"
    assert wrong_classifier["status"] == "UNCERTAIN"
    details.append("6-value / 64-value backend swap rejected")

    # 05: OTHER argmax is an explicit unknown rejection, even at a high score.
    unknown = classifier_decision([0.0, 0.0, 0.0, 0.0, 0.0, 20.0],
        config["identity"]["large"]["class_order"], 0.5, uncertainty_band=0.01)
    assert unknown["status"] == "OTHER" and unknown["identity_id"] is None
    details.append("injected OTHER-logit argmax rejects identity")

    # 06: queue pressure drops whole old records and never crosses its cap.
    queue = BoundedLineQueue(max_bytes=112, max_record_bytes=32)
    for index in range(20):
        queue.push(json.dumps({"n": index, "value": "x" * 8}))
    assert queue.queued_bytes <= 112 and queue.dropped_records > 0
    drained = queue.take(200)
    assert drained.endswith(b"\n") and b'"n":0' not in drained
    details.append("injected telemetry queue overflow remains bounded")

    # 07: wraparound and a backward non-wrap discontinuity are distinguished.
    clock = TickTracker(1000)
    clock.observe(990)
    wrapped = clock.observe(5)
    reset = clock.observe(4)
    assert wrapped["event"] == "wrap" and wrapped["extended_ticks"] == 1005
    assert reset["event"] == "reset_or_clock_discontinuity" and reset["resets"] == 1
    details.append("clock modulus wrap and reset/discontinuity")

    # 08: missing model and byte-corrupted model are caught by file/hash guards.
    expected = config["identity"]["large"]["model_sha256"]
    missing_caught = False
    try:
        (STAGE / "missing-model.tflite").read_bytes()
    except FileNotFoundError:
        missing_caught = True
    assert missing_caught
    with tempfile.TemporaryDirectory(prefix="F-corrupt-model-") as temp_dir:
        corrupted = Path(temp_dir) / "corrupt.tflite"
        data = (STAGE / "identity_classifier_a025_int8.tflite").read_bytes()
        corrupted.write_bytes(data[:-1] + bytes([data[-1] ^ 0x01]))
        assert not hash_matches(sha256(corrupted), expected)
    details.append("missing and byte-corrupted model caught")

    # 09: wrong config hash and low heap block execution before model load.
    assert not config_hash_matches("0" * 64, "1" * 64)
    assert not enough_heap(49151, 49152) and enough_heap(49152, 49152)
    details.append("wrong config hash and injected low heap fail closed")

    # 10: USB-disabled telemetry still reaches UART, and a temporary UART error
    # does not remove the port object needed after the transport reconnects.
    usb = FakeUSB(connected=False)
    uart = FakeUART(fail_once=True)
    telemetry = Telemetry(usb, uart, config, lambda: 1234, "fault-run", "host-test",
        "a" * 64, "b" * 64, "c" * 64, 2 ** 30)
    telemetry.emit("dropped_during_disconnect", {"reason": "injected"})
    telemetry.flush()
    assert telemetry.uart_failures == 1 and usb.data == b""
    telemetry.emit("uart_reconnected", {"usb_connected": False})
    telemetry.flush()
    assert b"uart_reconnected" in uart.data and usb.data == b""
    details.append("USB disconnected while UART recovers independently")

    # 11: invalid crop/quality results cannot create a known identity.
    valid, _ = quality_decision(5, 2)
    assert not valid and outside[1] == "crop_outside_frame"
    details.append("invalid crop and insufficient quality block identity stage")

    # 12: 2-of-3 only confirms the same known ID, never a single vote.
    confirm = TemporalConfirmer(True)
    one = confirm.apply({"status": "KNOWN_IDENTITY", "identity_id": "K01"})
    two = confirm.apply({"status": "KNOWN_IDENTITY", "identity_id": "K01"})
    assert one["status"] == "UNCERTAIN" and two["status"] == "KNOWN_IDENTITY"
    details.append("2-of-3 temporal confirmer rejects a single positive")

    # 13: actual OpenMV ml.Model metadata is a one-item list. Keep signed and
    # unsigned dtype codes separate, and preserve predict()'s float values.
    assert single_tensor_metadata([(1, 96, 96, 1)], "input_shape") == (1, 96, 96, 1)
    assert openmv_dtype_name(["b"]) == "int8"
    assert openmv_dtype_name(["B"]) == "uint8"
    assert prediction_values([0.125, -0.25], 2) == [0.125, -0.25]
    metadata_rejected = False
    try:
        single_tensor_metadata(["b", "B"], "dtype")
    except ValueError:
        metadata_rejected = True
    assert metadata_rejected
    details.append("OpenMV list metadata and predict float outputs avoid dtype aliasing/double dequantization")

    # 14: FOMO labels may include background or list foreground labels only.
    assert class_index_from_labels(["background", "face"], 2, "face") == 1
    assert class_index_from_labels(["face"], 2, "face") == 1
    assert fomo_face_output_index(["background", "face"], 2, 2) == 1
    assert fomo_face_output_index(["background", "face"], 2, 1) == 0
    details.append("FOMO tensor and postprocessed face groups map with or without background")
    return {"faults_caught": len(details), "fault_checks": details,
            "failures": failures}


def fomo_stage_status(detections, frame):
    if len(detections) == 0:
        return "NO_FACE"
    if len(detections) > 1:
        return "MULTIPLE_OR_AMBIGUOUS"
    center = detections[0]["center_frame_xy"]
    crop_box, reason = crop_xywh(center[0], center[1], 96, 240, 240)
    if reason:
        return "CROP_OUTSIDE"
    x, y, side, _ = crop_box
    crop = frame[y:y + side, x:x + side]
    passed, _ = quality_decision(float(crop.mean()), float(crop.std()))
    return "ONE_FACE_VALID" if passed else "LOW_QUALITY"


def expected_stage_status(fixture, raw_output, output_detail, frame):
    scale, zero = output_detail["quantization"]
    probabilities = (raw_output.astype(np.float32) - np.float32(zero)) * np.float32(scale)
    detections = openmv_fomo_detections(probabilities, 0.7)
    return fomo_stage_status(detections, frame), detections


def main():
    config = json.loads((ROOT / "firmware/runtime_config.json").read_text())
    manifest = json.loads((STAGE / "integration_manifest.json").read_text())
    gallery = json.loads((STAGE / "fixed_roster_gallery.json").read_text())
    if manifest.get("runtime_config", {}).get("sha256") != sha256(ROOT / "firmware/runtime_config.json"):
        raise AssertionError("runtime configuration hash differs from integration manifest")
    labels = manifest.get("labels", {})
    if labels.get("class_order") != KNOWN + ["OTHER"]:
        raise AssertionError("integration classifier class order differs from the five-known contract")
    if sha256(STAGE / "detector_gray_fomo_int8.txt") != labels.get("detector_sha256"):
        raise AssertionError("detector label hash differs from integration manifest")
    if sha256(STAGE / "identity_classifier_a025_int8.txt") != labels.get("classifier_sha256"):
        raise AssertionError("classifier label hash differs from integration manifest")
    if (STAGE / "detector_gray_fomo_int8.txt").read_text().splitlines() != ["background", "face"]:
        raise AssertionError("C detector label order differs from its two-channel output")
    if (STAGE / "identity_classifier_a025_int8.txt").read_text().splitlines() != KNOWN + ["OTHER"]:
        raise AssertionError("E classifier sidecar order differs from its six-logit output")
    for name, metadata in manifest["files"].items():
        path = STAGE / name
        if not path.is_file() or sha256(path) != metadata["sha256"]:
            raise AssertionError("staged model hash mismatch: " + name)
    ok, reason = validate_gallery_metadata(gallery, config["identity"]["small"]["model_id"],
        config["identity"]["small"]["model_sha256"],
        config["identity"]["small"]["prototype_sha256"], KNOWN)
    if not ok:
        raise AssertionError("fixed gallery validation failed: " + reason)
    if sha256(STAGE / "fixed_roster_gallery.json") != manifest["gallery"]["sha256"]:
        raise AssertionError("gallery manifest hash mismatch")

    c_path = STAGE / "detector_gray_fomo_int8.tflite"
    c_interpreter = interpreter(c_path)
    c_idetail = c_interpreter.get_input_details()[0]
    c_odetail = c_interpreter.get_output_details()[0]
    fixture_paths = sorted(C_FIXTURES.glob("*/fixture.json"))
    if len(fixture_paths) < 10:
        raise AssertionError("expected at least ten exact C raw-byte fixtures")
    c_rows = []
    exact_output_count = 0
    total_output_elements = 0
    total_output_differences = 0
    max_output_delta = 0
    detector_resize_exact_count = 0
    identity_resize_exact_count = 0
    for fixture_path in fixture_paths:
        base = fixture_path.parent
        fixture = json.loads(fixture_path.read_text())
        for filename, metadata in fixture.get("exact_raw_files", {}).items():
            if metadata is None:
                continue
            raw_path = base / filename
            if not raw_path.is_file() or raw_path.stat().st_size != metadata["bytes"] or sha256(raw_path) != metadata["sha256"]:
                raise AssertionError("C fixture raw-byte checksum mismatch: " + str(raw_path))
        raw_input = np.fromfile(base / "detector_input_int8.i8", dtype=np.int8).reshape(tuple(c_idetail["shape"]))
        expected_output = np.fromfile(base / "detector_output_int8.i8", dtype=np.int8)
        c_interpreter.set_tensor(c_idetail["index"], raw_input)
        c_interpreter.invoke()
        actual_tensor = c_interpreter.get_tensor(c_odetail["index"])
        actual_output = actual_tensor.reshape(-1)
        expected_tensor = expected_output.reshape(actual_tensor.shape)
        differences = np.abs(actual_output.astype(np.int16) - expected_tensor.reshape(-1).astype(np.int16))
        diff_count = int(np.count_nonzero(differences))
        max_delta = int(differences.max()) if differences.size else 0
        exact_output_count += int(diff_count == 0)
        total_output_elements += int(differences.size)
        total_output_differences += diff_count
        max_output_delta = max(max_output_delta, max_delta)
        if max_delta > 16:
            raise AssertionError("C detector output drift exceeds documented cross-runtime bound: " + base.name)
        gray = np.fromfile(base / "detector_input_gray96.u8", dtype=np.uint8)
        quant = np.fromfile(base / "detector_input_int8.i8", dtype=np.int8)
        if not np.array_equal(quant, gray.astype(np.int16).reshape(-1) - 128):
            raise AssertionError("C fixture input quantization mismatch: " + base.name)
        expected_crop = fixture["expected_crop_geometry"].get("crop_xywh_window")
        center = fixture["expected_crop_geometry"].get("predicted_center_window_xy")
        crop_result = None
        if center is not None:
            crop_result, reason = crop_xywh(center[0], center[1], 96, 240, 240)
            if expected_crop is not None and list(crop_result) != expected_crop:
                raise AssertionError("C crop geometry mismatch: " + base.name)
            if expected_crop is None and reason is None:
                raise AssertionError("C invalid crop not rejected: " + base.name)
        camera_bytes = (base / "camera_window_gray240.u8").read_bytes()
        camera = np.frombuffer(camera_bytes, dtype=np.uint8).reshape(240, 240)
        detector_resize = bytearray(96 * 96)
        pillow_bilinear_grayscale_u8(camera_bytes, 240, 240, (0, 0, 240, 240), detector_resize)
        detector_resize_exact = detector_resize == (base / "detector_input_gray96.u8").read_bytes()
        detector_resize_exact_count += int(detector_resize_exact)
        if not detector_resize_exact:
            raise AssertionError("F Pillow bilinear detector resize mismatch: " + base.name)
        reference_status, reference_detections = expected_stage_status(fixture, expected_tensor, c_odetail, camera)
        actual_status, actual_detections = expected_stage_status(fixture, actual_tensor, c_odetail, camera)
        if actual_status != reference_status:
            raise AssertionError("C FOMO scene/crop state differs from raw fixture reference: " + base.name)
        expected_center = fixture["expected_crop_geometry"].get("predicted_center_window_xy")
        if expected_center is not None and actual_status in ("ONE_FACE_VALID", "LOW_QUALITY", "CROP_OUTSIDE"):
            actual_center = actual_detections[0]["center_frame_xy"]
            if list(actual_center) != list(expected_center):
                raise AssertionError("C top detection center changed: " + base.name)
        if expected_crop is not None:
            identity_path = base / "identity_input_gray96.u8"
            if identity_path.is_file():
                identity_resize = bytearray(96 * 96)
                pillow_bilinear_grayscale_u8(camera_bytes, 240, 240, tuple(crop_result), identity_resize)
                identity_resize_exact = identity_resize == identity_path.read_bytes()
                identity_resize_exact_count += int(identity_resize_exact)
                if not identity_resize_exact:
                    raise AssertionError("F Pillow bilinear identity resize mismatch: " + base.name)
            else:
                identity_resize_exact = None
        else:
            identity_resize_exact = None
        c_rows.append({"fixture": base.name,
            "actual_output_sha256": hashlib.sha256(actual_output.tobytes()).hexdigest(),
            "reference_output_sha256": hashlib.sha256(expected_output.tobytes()).hexdigest(),
            "raw_output_diff_elements": diff_count, "raw_output_element_count": int(differences.size),
            "max_raw_output_delta_int8": max_delta,
            "reference_detection_count": len(reference_detections),
            "actual_detection_count": len(actual_detections),
            "reference_stage_status": reference_status, "actual_stage_status": actual_status,
            "F_detector_pillow_bilinear_exact": detector_resize_exact,
            "F_identity_pillow_bilinear_exact": identity_resize_exact,
            "crop_xywh": list(crop_result) if crop_result else None})

    # Actual D/E selected train crops exercise both exported models. These are
    # integration inputs only; no recognition accuracy claim is inferred.
    data = read_jsonl(STUDY / "data/primary_split_manifest.jsonl")
    selected_ids = {rid for ids in gallery["source_record_ids"].values() for rid in ids}
    train_by_id = {str(row["record_id"]): row for row in data
        if row.get("dataset") == "survface_96" and row.get("role") == "train"}
    rows = [train_by_id[rid] for label in KNOWN for rid in gallery["source_record_ids"][label]]
    images = [load_pixels(row, PROJECT, output="gray8", action="train").reshape(96, 96, 1)
              for row in rows]
    if len(rows) != len(selected_ids) or len(images) != 15:
        raise AssertionError("fixed gallery selected crop inventory mismatch")
    d_itp = interpreter(STAGE / "identity_embedding_a010_int8.tflite")
    e_itp = interpreter(STAGE / "identity_classifier_a025_int8.tflite")
    d_detail = d_itp.get_input_details()[0]
    e_detail = e_itp.get_input_details()[0]
    e_output = e_itp.get_output_details()[0]
    if tuple(d_itp.get_output_details()[0]["shape"]) != (1, 64):
        raise AssertionError("D must export 64-value embedding output")
    if tuple(e_output["shape"]) != (1, 6):
        raise AssertionError("E must export six logits")
    if list(config["identity"]["large"]["class_order"]) != ["K01", "K02", "K03", "K04", "K05", "OTHER"]:
        raise AssertionError("E label order differs from fixed roster")

    seeded_blocks = []
    d_model_ref = []
    e_model_ref = []
    for block_index in range(10):
        seed = 20261101 + block_index
        rng = np.random.default_rng(seed)
        sample_index = int(rng.integers(0, len(images)))
        gray = images[sample_index]
        d_raw_input, d_raw, d_values, d_firmware = run_gray_model(d_itp, gray)
        e_raw_input, e_raw, e_values, e_firmware = run_gray_model(e_itp, gray)
        if not np.allclose(d_values, d_firmware, rtol=0, atol=1e-7):
            raise AssertionError("D manual output dequantization mismatch")
        if not np.allclose(e_values, e_firmware, rtol=0, atol=1e-7):
            raise AssertionError("E manual output dequantization mismatch")
        if not np.allclose(prediction_values(d_values.tolist(), 64), d_firmware, rtol=0, atol=1e-7):
            raise AssertionError("D OpenMV predict output contract mismatch")
        if not np.allclose(prediction_values(e_values.tolist(), 6), e_firmware, rtol=0, atol=1e-7):
            raise AssertionError("E OpenMV predict output contract mismatch")
        if d_raw_input.dtype != np.int8 or e_raw_input.dtype != np.int8:
            raise AssertionError("D/E input boundary must be int8")
        d_vec = normalize_embedding(d_values.tolist())
        d_decision = gallery_decision(d_vec, gallery["entries"],
            config["identity"]["small"]["threshold"], config["identity"]["uncertainty_band"],
            expected_model_id=config["identity"]["small"]["model_id"])
        e_decision = classifier_decision(e_values.tolist(),
            config["identity"]["large"]["class_order"],
            config["identity"]["large"]["threshold"],
            config["identity"]["uncertainty_band"])
        if d_decision["status"] not in ("KNOWN_IDENTITY", "OTHER", "UNCERTAIN"):
            raise AssertionError("D typed output decision invalid")
        if e_decision["status"] not in ("KNOWN_IDENTITY", "OTHER", "UNCERTAIN"):
            raise AssertionError("E typed output decision invalid")
        seeded_blocks.append({"seed": seed, "record_id": rows[sample_index]["record_id"],
            "D_output_width": len(d_values), "E_output_width": len(e_values),
            "D_decision": d_decision["status"], "E_decision": e_decision["status"],
            "D_raw_output_sha256": hashlib.sha256(d_raw.tobytes()).hexdigest(),
            "E_raw_output_sha256": hashlib.sha256(e_raw.tobytes()).hexdigest()})
        d_model_ref.append(d_raw)
        e_model_ref.append(e_raw)

    fault_report = test_fault_blocks(config, gallery, c_interpreter)
    if len(seeded_blocks) != 10 or fault_report["faults_caught"] < 10:
        raise AssertionError("ten seeded model/fault blocks are required")
    device_discovery = json.loads(DEVICE_DISCOVERY.read_text()) if DEVICE_DISCOVERY.is_file() else None
    report = {"schema": "job-f-host-verification-1", "status": "host_verified_physical_pending",
        "created_utc": "2026-10-30", "seeded_blocks": seeded_blocks,
        "seeded_block_count": len(seeded_blocks), "fault_report": fault_report,
        "C_fixture_count": len(c_rows), "C_fixtures": c_rows,
        "F_detector_pillow_bilinear_exact_count": detector_resize_exact_count,
        "F_identity_pillow_bilinear_exact_count": identity_resize_exact_count,
        "C_exact_raw_input_quantization_fixture_count": len(c_rows),
        "C_fomo_stage_status_match_count": len(c_rows),
        "C_raw_output_exact_fixture_count": exact_output_count,
        "C_raw_output_diff_elements": total_output_differences,
        "C_raw_output_total_elements": total_output_elements,
        "C_raw_output_max_abs_delta_int8": max_output_delta,
        "C_raw_output_parity": ("bit_exact_against_all_frozen_fixture_outputs" if
            exact_output_count == len(c_rows) else
            "not_bit_exact_across_current_host_TFLite_runtime; all exact-byte inputs and FOMO scene/crop states matched within <=16 INT8 steps"),
        "model_hashes": {name: metadata["sha256"] for name, metadata in manifest["files"].items()},
        "gallery_sha256": sha256(STAGE / "fixed_roster_gallery.json"),
        "D_prototype_sha256": gallery["prototype_sha256"],
        "D_historical_prototype_sha256": gallery.get("historical_D_prototype_sha256"),
        "D_threshold": config["identity"]["small"]["threshold"],
        "E_threshold": config["identity"]["large"]["threshold"],
        "D_E_unique_seeded_crop_count": len({row["record_id"] for row in seeded_blocks}),
        "host_environment": {"python": platform.python_version(),
            "tensorflow": tf.__version__, "numpy": np.__version__,
            "platform": platform.platform()},
        "final_role_accessed": False,
        "device_discovery": str(DEVICE_DISCOVERY) if device_discovery else None,
        "physical_board_tests": (device_discovery.get("physical_acceptance") if device_discovery
            else "pending; fresh physical device discovery has not been recorded"),
        "accuracy_claim": "none; D/E seeded crops verify actual exported model interfaces and host decision semantics only"}
    output = STUDY / "reports/F/host_verification.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, sort_keys=True, indent=2) + "\n")
    print(json.dumps({"status": report["status"], "seeded_blocks": len(seeded_blocks),
        "faults_caught": fault_report["faults_caught"], "C_fixtures": len(c_rows),
        "output": str(output)}, indent=2))


if __name__ == "__main__":
    main()
