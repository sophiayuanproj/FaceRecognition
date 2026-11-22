#!/usr/bin/env python3
"""Host-side contract and edge-case tests for MicroPython recognition helpers."""

from __future__ import annotations

import gc as host_gc
import importlib
import json
import math
import os
import time as host_time
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock


ROOT = Path(__file__).resolve().parents[3]
FIRMWARE = ROOT / "FaceRecognition" / "firmware"
sys.path.insert(0, str(FIRMWARE))


class FakeLED:
    def off(self):
        pass

    def on(self):
        pass


def install_micropython_stubs():
    csi = types.ModuleType("csi")
    csi.RGB565 = 0
    sys.modules["csi"] = csi

    pyb = types.ModuleType("pyb")
    class USBVCP:
        def isconnected(self):
            return False
    pyb.USB_VCP = USBVCP
    sys.modules["pyb"] = pyb

    ml = types.ModuleType("ml")
    ml.__path__ = []
    ml.postprocessing = types.ModuleType("ml.postprocessing")
    edgeimpulse = types.ModuleType("ml.postprocessing.edgeimpulse")
    class Fomo:
        def __init__(self, threshold=0.4):
            self.threshold = threshold
    edgeimpulse.Fomo = Fomo
    ml.postprocessing.edgeimpulse = edgeimpulse
    preprocessing = types.ModuleType("ml.preprocessing")
    class Normalization:
        def __init__(self, *args, **kwargs):
            self.args = args
            self.kwargs = kwargs
        def __call__(self, image):
            return image
    preprocessing.Normalization = Normalization
    ml.preprocessing = preprocessing
    sys.modules["ml"] = ml
    sys.modules["ml.postprocessing"] = ml.postprocessing
    sys.modules["ml.postprocessing.edgeimpulse"] = edgeimpulse
    sys.modules["ml.preprocessing"] = preprocessing

    machine = types.ModuleType("machine")
    machine.LED = lambda name: FakeLED()
    sys.modules["machine"] = machine

    import recognize_embedding
    import enroll_identity
    return recognize_embedding, enroll_identity


RUNTIME, ENROLL = install_micropython_stubs()

if not hasattr(host_gc, "mem_free"):
    host_gc.mem_free = lambda: 500000
if not hasattr(host_time, "ticks_us"):
    host_time.ticks_us = lambda: int(host_time.perf_counter() * 1000000)
if not hasattr(host_time, "ticks_ms"):
    host_time.ticks_ms = lambda: int(host_time.perf_counter() * 1000)
if not hasattr(host_time, "ticks_diff"):
    host_time.ticks_diff = lambda end, start: end - start
if not hasattr(host_time, "sleep_ms"):
    host_time.sleep_ms = lambda value: None


class FakeStats:
    def __init__(self, mean=100, stdev=30):
        self.mean = mean
        self.stdev = stdev


class FakeCrop:
    def __init__(self, mean=100, stdev=30):
        self.mean = mean
        self.stdev = stdev
    def get_statistics(self):
        return FakeStats(self.mean, self.stdev)


class FakeFrame:
    def __init__(self, w=240, h=240, mean=100, stdev=30):
        self.w = w
        self.h = h
        self.mean = mean
        self.stdev = stdev
    def width(self):
        return self.w
    def height(self):
        return self.h
    def to_grayscale(self, roi=None, copy=True):
        return FakeCrop(self.mean, self.stdev)


class FakeModel:
    def __init__(self, path, postprocess=None, output=None, labels=None, input_dtype=("b",),
                 input_shape=(1, 96, 96, 1), output_shape=(1, 64), output_dtype=("b",),
                 input_scale=(1.0 / 255.0,), input_zero=(-128,), vector=None):
        self.len = 1000
        self.ram = 2048
        self.input_shape = input_shape
        self.input_dtype = input_dtype
        self.input_scale = input_scale
        self.input_zero_point = input_zero
        self.output_shape = output_shape
        self.output_dtype = output_dtype
        self.output_scale = (0.01,)
        self.output_zero_point = (0,)
        self.labels = labels or ["background", "face"]
        self._output = output
        self._vector = vector or [1.0] + [0.0] * 63
    def predict(self, inputs):
        if self._output is not None:
            return self._output
        return [[self._vector]]


class FirmwareEdgeCaseTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.templates_path = self.root / "templates.json"
        self.calibration_path = self.root / "calibration.json"
        self.old_template = RUNTIME.cfg.TEMPLATE_FILE
        self.old_calibration = RUNTIME.cfg.CALIBRATION_FILE
        self.old_model = RUNTIME.cfg.EMBEDDING_MODEL
        self.old_low_mean = RUNTIME.cfg.LOW_QUALITY_MIN_MEAN
        self.old_high_mean = RUNTIME.cfg.LOW_QUALITY_MAX_MEAN
        self.old_low_std = RUNTIME.cfg.LOW_QUALITY_MIN_STDEV
        RUNTIME.cfg.TEMPLATE_FILE = str(self.templates_path)
        RUNTIME.cfg.CALIBRATION_FILE = str(self.calibration_path)
        RUNTIME.cfg.EMBEDDING_MODEL = "/test/model.tflite"
        self.old_ml_model = getattr(sys.modules["ml"], "Model", None)
        sys.modules["ml"].Model = FakeModel
        self.old_request_file = ENROLL.REQUEST_FILE
        self.old_detector_logged = RUNTIME.DETECTOR_METADATA_LOGGED
        self.old_embedding_logged = RUNTIME.EMBEDDING_METADATA_LOGGED

    def tearDown(self):
        RUNTIME.cfg.TEMPLATE_FILE = self.old_template
        RUNTIME.cfg.CALIBRATION_FILE = self.old_calibration
        RUNTIME.cfg.EMBEDDING_MODEL = self.old_model
        RUNTIME.cfg.LOW_QUALITY_MIN_MEAN = self.old_low_mean
        RUNTIME.cfg.LOW_QUALITY_MAX_MEAN = self.old_high_mean
        RUNTIME.cfg.LOW_QUALITY_MIN_STDEV = self.old_low_std
        if self.old_ml_model is None:
            delattr(sys.modules["ml"], "Model")
        else:
            sys.modules["ml"].Model = self.old_ml_model
        ENROLL.REQUEST_FILE = self.old_request_file
        RUNTIME.DETECTOR_METADATA_LOGGED = self.old_detector_logged
        RUNTIME.EMBEDDING_METADATA_LOGGED = self.old_embedding_logged
        self.temp.cleanup()

    def write_template(self, vector=None, identity="test_1"):
        vector = vector if vector is not None else [1.0] + [0.0] * 63
        self.templates_path.write_text(json.dumps({
            "version": 1, "entries": [{"identity_id": identity, "vector": vector}],
        }))

    def test_crop_bounds_exact_edges_and_one_pixel_outside(self):
        self.assertEqual(RUNTIME.valid_crop(FakeFrame(), (48, 48)), (0, 0, 96, 96))
        self.assertEqual(RUNTIME.valid_crop(FakeFrame(), (192, 192)), (144, 144, 96, 96))
        self.assertIsNone(RUNTIME.valid_crop(FakeFrame(), (47, 48)))
        self.assertIsNone(RUNTIME.valid_crop(FakeFrame(), (193, 192)))
        self.assertIsNone(RUNTIME.valid_crop(FakeFrame(), (48, 47)))
        self.assertIsNone(RUNTIME.valid_crop(FakeFrame(), (192, 193)))

    def test_face_output_index_label_layouts_and_rejects_ambiguous_layout(self):
        model = types.SimpleNamespace(labels=["background", "face"])
        self.assertEqual(RUNTIME.face_output_index(model, 2), 1)
        self.assertEqual(RUNTIME.face_output_index(model, 1), 0)
        model.labels = ["background", "person", "face"]
        self.assertEqual(RUNTIME.face_output_index(model, 2), 1)
        model.labels = []
        self.assertEqual(RUNTIME.face_output_index(model, 2), 1)
        model.labels = ["background", "person", "car"]
        with self.assertRaises(ValueError):
            RUNTIME.face_output_index(model, 3)

    def test_empty_detector_output_is_no_face_and_multiple_regions_are_capped(self):
        def detector(output):
            class Model(FakeModel):
                def __init__(self, path, postprocess=None):
                    super().__init__(path, postprocess, output=output,
                                     input_shape=(1, 96, 96, 3), output_shape=(1, 1))
            sys.modules["ml"].Model = Model

        stats = {}
        detector([])
        self.assertEqual(RUNTIME.detect_faces(FakeFrame(), stats), [])
        self.assertEqual(stats["face_count"], 0)

        detector([[], [((20, 20, 20, 20), 0.9), ((50, 50, 20, 20), 0.8),
                       ((80, 80, 20, 20), 0.7)]])
        stats = {}
        faces = RUNTIME.detect_faces(FakeFrame(), stats)
        self.assertEqual(stats["face_count"], 3)
        self.assertTrue(stats["face_records_truncated"])
        self.assertEqual(len(faces), 2)
        self.assertEqual(faces[0]["center"], (30, 30))


    def test_process_frame_covers_no_face_multiple_clipped_low_quality_and_error(self):
        with mock.patch.object(RUNTIME, "detect_faces", return_value=[]):
            self.assertEqual(RUNTIME.process_frame(FakeFrame(), {})["status"], "no_face")
        multi = [{"center": (100, 100), "score": 0.9},
                 {"center": (140, 140), "score": 0.8}]
        with mock.patch.object(RUNTIME, "detect_faces", return_value=multi):
            self.assertEqual(RUNTIME.process_frame(FakeFrame(), {})["status"], "multiple_faces")
        clipped = [{"center": (10, 10), "score": 0.9}]
        stats = {}
        with mock.patch.object(RUNTIME, "detect_faces", return_value=clipped):
            result = RUNTIME.process_frame(FakeFrame(), stats)
        self.assertEqual(result["status"], "uncertain")
        self.assertEqual(result["decision_reason"], "crop_clipped_by_frame_edge")
        self.assertEqual(stats["detection_score"], 0.9)
        centered = [{"center": (120, 120), "score": 0.9}]
        with mock.patch.object(RUNTIME, "detect_faces", return_value=centered):
            with mock.patch.object(RUNTIME, "embed_crop", return_value={
                    "status": "uncertain", "decision_reason": "low_quality_crop"}):
                self.assertEqual(RUNTIME.process_frame(FakeFrame(), {})["decision_reason"],
                                 "low_quality_crop")
        with mock.patch.object(RUNTIME, "detect_faces", side_effect=RuntimeError("bad detector")):
            error = RUNTIME.process_frame(FakeFrame(), {})
        self.assertEqual(error["status"], "pipeline_error")
        self.assertEqual(error["error"], "bad detector")

    def test_template_file_missing_is_empty_roster(self):
        self.assertEqual(RUNTIME.load_templates(), [])

    def test_template_valid_vector_normalizes_and_validates_identity(self):
        self.write_template([2.0] + [0.0] * 63, identity="person_A-2")
        entries = RUNTIME.load_templates()
        self.assertEqual(entries[0]["identity_id"], "person_A-2")
        self.assertAlmostEqual(entries[0]["vector"][0], 1.0)

    def test_template_rejects_malformed_and_unsafe_contents(self):
        bad_payloads = [
            [],
            {"version": 2, "entries": []},
            {"version": 1, "entries": {}},
            {"version": 1, "entries": [None]},
            {"version": 1, "entries": [{"identity_id": "../x", "vector": [1.0] * 64}]},
            {"version": 1, "entries": [{"identity_id": "x", "vector": [1.0] * 63}]},
            {"version": 1, "entries": [{"identity_id": "x", "vector": [0.0] * 64}]},
            {"version": 1, "entries": [{"identity_id": "x", "vector": [float("nan")] + [0.0] * 63}]},
            {"version": 1, "entries": [{"identity_id": "x", "vector": [float("inf")] + [0.0] * 63}]},
            {"version": 1, "entries": [{"identity_id": "x", "vector": ["1.0"] + [0.0] * 63}]},
            {"version": 1, "entries": [{"identity_id": "x", "vector": [True] + [0.0] * 63}]},
            {"version": 1, "entries": [{"identity_id": "same", "vector": [1.0] + [0.0] * 63},
                                        {"identity_id": "same", "vector": [1.0] + [0.0] * 63}]},
            {"version": 1, "entries": [{"identity_id": "x%d" % i, "vector": [1.0] + [0.0] * 63}
                                        for i in range(9)]},
        ]
        for payload in bad_payloads:
            with self.subTest(payload=repr(payload)[:90]):
                self.templates_path.write_text(json.dumps(payload))
                with self.assertRaises(ValueError):
                    RUNTIME.load_templates()
        self.templates_path.write_text("{bad json")
        with self.assertRaises(ValueError):
            RUNTIME.load_templates()

    def test_calibration_threshold_edges_and_rejects_nonfinite(self):
        self.assertIsNone(RUNTIME.calibration_threshold())
        for value in (-1.0, 0.0, 1.0):
            self.calibration_path.write_text(json.dumps({"threshold": value}))
            self.assertEqual(RUNTIME.calibration_threshold(), value)
        for value in (-1.001, 1.001, float("nan"), float("inf"), True, "0.8", None):
            with self.subTest(value=repr(value)):
                self.calibration_path.write_text(json.dumps({"threshold": value}))
                with self.assertRaises(ValueError):
                    RUNTIME.calibration_threshold()
        self.calibration_path.write_text("[]")
        with self.assertRaises(ValueError):
            RUNTIME.calibration_threshold()

    def test_low_quality_brightness_and_contrast_boundaries_short_circuit(self):
        class NoModel:
            def __init__(self, *args, **kwargs):
                raise AssertionError("low-quality input must not allocate a model")
        sys.modules["ml"].Model = NoModel
        for mean, stdev in ((11, 30), (244, 30), (100, 4)):
            with self.subTest(mean=mean, stdev=stdev):
                result = RUNTIME.embed_crop(FakeFrame(mean=mean, stdev=stdev),
                                            (72, 72, 96, 96), {})
                self.assertEqual(result["status"], "uncertain")
                self.assertEqual(result["decision_reason"], "low_quality_crop")
        # Exact configured boundaries are accepted by the crop-quality gate.
        sys.modules["ml"].Model = FakeModel
        result = RUNTIME.embed_crop(FakeFrame(mean=12, stdev=5),
                                    (72, 72, 96, 96), {})
        self.assertEqual(result["status"], "no_enrollment")

    def test_embedding_finite_zero_and_model_contract_guards(self):
        for vector in ([0.0] * 64, [float("nan")] + [0.0] * 63):
            class Model(FakeModel):
                def __init__(self, path):
                    super().__init__(path, vector=vector)
            sys.modules["ml"].Model = Model
            with self.assertRaises(ValueError):
                RUNTIME.embed_crop(FakeFrame(), (72, 72, 96, 96), {})
        class WrongShape(FakeModel):
            def __init__(self, path):
                super().__init__(path, input_shape=(1, 96, 96, 3))
        sys.modules["ml"].Model = WrongShape
        with self.assertRaises(ValueError):
            RUNTIME.embed_crop(FakeFrame(), (72, 72, 96, 96), {})

    def test_recognition_decision_known_unknown_uncertain_and_no_threshold(self):
        self.calibration_path.write_text(json.dumps({"threshold": 0.875}))
        self.write_template()
        sys.modules["ml"].Model = FakeModel
        self.assertEqual(RUNTIME.embed_crop(FakeFrame(), (72, 72, 96, 96), {})["status"], "known")

        self.write_template([-1.0] + [0.0] * 63)
        self.assertEqual(RUNTIME.embed_crop(FakeFrame(), (72, 72, 96, 96), {})["status"], "unknown")

        score = 0.86
        self.write_template([1.0] + [0.0] * 63)
        vector = [score, math.sqrt(1.0 - score * score)] + [0.0] * 62
        class NearThresholdModel(FakeModel):
            def __init__(self, path):
                super().__init__(path, vector=vector)
        sys.modules["ml"].Model = NearThresholdModel
        self.assertEqual(RUNTIME.embed_crop(FakeFrame(), (72, 72, 96, 96), {})["status"], "uncertain")

        self.calibration_path.unlink()
        self.assertEqual(RUNTIME.embed_crop(FakeFrame(), (72, 72, 96, 96), {})["status"], "uncertain")

    def test_enrollment_request_bounds_types_and_identifier_safety(self):
        ENROLL.REQUEST_FILE = str(self.root / "request.json")
        path = Path(ENROLL.REQUEST_FILE)
        path.write_text(json.dumps({"version": 1, "identity_id": "person_01"}))
        self.assertEqual(ENROLL.read_request(), ("person_01", 5, 30, False))
        path.write_text(json.dumps({"version": 1, "identity_id": "p-1", "min_samples": 3,
                                    "max_attempts": 3, "replace_existing": True}))
        self.assertEqual(ENROLL.read_request(), ("p-1", 3, 3, True))
        invalid = [
            [],
            {"version": 2, "identity_id": "x"},
            {"version": 1, "identity_id": "../x"},
            {"version": 1, "identity_id": "é"},
            {"version": 1, "identity_id": "x" * 25},
            {"version": 1, "identity_id": "x", "min_samples": 2},
            {"version": 1, "identity_id": "x", "min_samples": 21},
            {"version": 1, "identity_id": "x", "min_samples": 3.9},
            {"version": 1, "identity_id": "x", "min_samples": True},
            {"version": 1, "identity_id": "x", "min_samples": 5, "max_attempts": 4},
            {"version": 1, "identity_id": "x", "max_attempts": 61},
            {"version": 1, "identity_id": "x", "max_attempts": 30.8},
            {"version": 1, "identity_id": "x", "replace_existing": "false"},
        ]
        for payload in invalid:
            with self.subTest(payload=repr(payload)):
                path.write_text(json.dumps(payload))
                with self.assertRaises(ValueError):
                    ENROLL.read_request()

    def test_enrollment_mean_rejects_empty_wrong_shape_nonfinite_and_cancellation(self):
        with self.assertRaises(ValueError):
            ENROLL.normalized_mean([])
        with self.assertRaises(ValueError):
            ENROLL.normalized_mean([[1.0, 2.0]])
        with self.assertRaises(ValueError):
            ENROLL.normalized_mean([[float("nan")] + [0.0] * 63])
        with self.assertRaises(ValueError):
            ENROLL.normalized_mean([[1.0] + [0.0] * 63, [-1.0] + [0.0] * 63])
        vector = ENROLL.normalized_mean([[2.0] + [0.0] * 63, [1.0] + [0.0] * 63])
        self.assertAlmostEqual(vector[0], 1.0)

    def test_template_write_replacement_and_failed_commit_rollback(self):
        old_path = RUNTIME.cfg.TEMPLATE_FILE
        old_backup = ENROLL.TEMPLATE_BACKUP
        RUNTIME.cfg.TEMPLATE_FILE = str(self.templates_path)
        ENROLL.TEMPLATE_BACKUP = str(self.templates_path) + ".bak"
        try:
            old = {"version": 1, "entries": [{"identity_id": "old", "vector": [1.0] + [0.0] * 63}]}
            new = {"identity_id": "new", "vector": [1.0] + [0.0] * 63}
            self.templates_path.write_text(json.dumps(old))
            ENROLL.write_templates([new])
            self.assertEqual(json.loads(self.templates_path.read_text())["entries"][0]["identity_id"], "new")
            self.assertEqual(json.loads(Path(ENROLL.TEMPLATE_BACKUP).read_text()), old)
            self.templates_path.write_text(json.dumps(old))
            real_rename = os.rename
            def fail_temp_rename(src, dst):
                if str(src).endswith(".tmp"):
                    raise OSError("simulated power loss before commit")
                return real_rename(src, dst)
            with mock.patch("os.rename", side_effect=fail_temp_rename):
                with self.assertRaises(OSError):
                    ENROLL.write_templates([new])
            self.assertEqual(json.loads(self.templates_path.read_text()), old)
        finally:
            RUNTIME.cfg.TEMPLATE_FILE = old_path
            ENROLL.TEMPLATE_BACKUP = old_backup


if __name__ == "__main__":
    unittest.main(verbosity=2)

