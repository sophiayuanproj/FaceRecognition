"""Read-only/temporary-file MicroPython probes for the shipped edge guards.

Writes only uniquely named QA JSON files under /flash and deletes them in a
finally block. It never accesses the camera or saves image data.
"""

import json
import os
import math

import embedding_config as cfg
import qa_enroll_identity as enrollment
import recognize_embedding as runtime

TEMP_BASE = "/flash/qa_edge_guard_20261022"
TEMP_TEMPLATES = TEMP_BASE + "_templates.json"
TEMP_CALIBRATION = TEMP_BASE + "_calibration.json"
TEMP_REQUEST = TEMP_BASE + "_request.json"


def emit(event, fields=None):
    record = {"event": event}
    if fields:
        record.update(fields)
    print(json.dumps(record))


def write_json(path, value):
    with open(path, "w") as stream:
        json.dump(value, stream)
        stream.flush()


def expect_value_error(function):
    try:
        function()
    except ValueError:
        return True
    return False


class FakeDetector:
    labels = ["background", "face"]


class FakeFrame:
    def width(self):
        return 240
    def height(self):
        return 240


def main():
    paths = [TEMP_TEMPLATES, TEMP_CALIBRATION, TEMP_REQUEST,
             TEMP_TEMPLATES + ".tmp", TEMP_TEMPLATES + ".bak"]
    for path in paths:
        try:
            os.stat(path)
            raise RuntimeError("refusing to overwrite existing QA path " + path)
        except OSError:
            pass

    old_template = cfg.TEMPLATE_FILE
    old_calibration = cfg.CALIBRATION_FILE
    old_request = enrollment.REQUEST_FILE
    old_backup = enrollment.TEMPLATE_BACKUP
    passed = []

    try:
        checks = [
            runtime.finite_number(0) == 0.0,
            runtime.finite_number(1.25) == 1.25,
            runtime.finite_number(float("nan")) is None,
            runtime.finite_number(float("inf")) is None,
            runtime.valid_identity_id("target_1"),
            not runtime.valid_identity_id("../target"),
            not runtime.valid_identity_id("é"),
            runtime.valid_crop(FakeFrame(), (48, 48)) == (0, 0, 96, 96),
            runtime.valid_crop(FakeFrame(), (192, 192)) == (144, 144, 96, 96),
            runtime.valid_crop(FakeFrame(), (47, 48)) is None,
            runtime.face_output_index(FakeDetector(), 2) == 1,
            runtime.face_output_index(FakeDetector(), 1) == 0,
        ]
        if not all(checks):
            raise AssertionError("numeric, ASCII-ID, crop-boundary, or FOMO-index guard failed")
        passed.extend(["finite_number", "identity_id_ascii_bounds",
                       "crop_exact_bounds_and_outside", "FOMO_output_index_layouts"])

        cfg.TEMPLATE_FILE = TEMP_TEMPLATES
        cfg.CALIBRATION_FILE = TEMP_CALIBRATION
        enrollment.REQUEST_FILE = TEMP_REQUEST
        enrollment.TEMPLATE_BACKUP = TEMP_TEMPLATES + ".bak"

        vector = [1.0] + [0.0] * 63
        write_json(TEMP_TEMPLATES, {
            "version": 1,
            "entries": [{"identity_id": "target_1", "vector": vector}],
        })
        loaded = runtime.load_templates()
        if len(loaded) != 1 or abs(loaded[0]["vector"][0] - 1.0) > 1e-6:
            raise AssertionError("valid enrollment template did not normalize")

        invalid_templates = [
            [],
            {"version": 2, "entries": []},
            {"version": 1, "entries": {}},
            {"version": 1, "entries": [{"identity_id": "../x", "vector": vector}]},
            {"version": 1, "entries": [{"identity_id": "x", "vector": [1.0] * 63}]},
            {"version": 1, "entries": [{"identity_id": "x", "vector": [0.0] * 64}]},
            {"version": 1, "entries": [
                {"identity_id": "x", "vector": vector},
                {"identity_id": "x", "vector": vector},
            ]},
        ]
        for payload in invalid_templates:
            write_json(TEMP_TEMPLATES, payload)
            if not expect_value_error(runtime.load_templates):
                raise AssertionError("malformed/unsafe template unexpectedly accepted")
        passed.append("template_schema_duplicates_dimension_norm_and_path_traversal")

        cfg.CALIBRATION_FILE = TEMP_CALIBRATION
        for value in (-1.0, 0.0, 1.0):
            write_json(TEMP_CALIBRATION, {"threshold": value})
            if runtime.calibration_threshold() != value:
                raise AssertionError("valid calibration boundary rejected")
        for value in (-1.1, 1.1, True, "0.8", None):
            write_json(TEMP_CALIBRATION, {"threshold": value})
            if not expect_value_error(runtime.calibration_threshold):
                raise AssertionError("invalid calibration threshold unexpectedly accepted")
        passed.append("calibration_range_and_type_guards")

        enrollment.REQUEST_FILE = TEMP_REQUEST
        write_json(TEMP_REQUEST, {
            "version": 1, "identity_id": "target_1",
            "min_samples": 3, "max_attempts": 3, "replace_existing": False,
        })
        if enrollment.read_request() != ("target_1", 3, 3, False):
            raise AssertionError("valid bounded enrollment request failed")
        write_json(TEMP_REQUEST, {
            "version": 1, "identity_id": "target_1",
            "min_samples": 3.5, "max_attempts": 30,
        })
        if not expect_value_error(enrollment.read_request):
            raise AssertionError("fractional sample limit unexpectedly accepted")
        if abs(enrollment.normalized_mean([vector, vector])[0] - 1.0) > 1e-6:
            raise AssertionError("valid enrollment vector mean failed")
        if not expect_value_error(lambda: enrollment.normalized_mean(
                [vector, [-1.0] + [0.0] * 63])):
            raise AssertionError("cancelling enrollment vectors unexpectedly accepted")
        passed.append("enrollment_request_types_and_mean_edges")

        # Exercise the actual on-device atomic write/replace path using only QA files.
        enrollment.write_templates([{"identity_id": "target_1", "vector": vector}])
        if len(runtime.load_templates()) != 1:
            raise AssertionError("on-device template write did not round-trip")
        passed.append("on_device_template_write_round_trip")
        emit("edge_guard_probe", {
            "status": "passed",
            "checks": passed,
            "temporary_paths": paths,
            "camera_accessed": False,
            "images_saved": 0,
        })
    finally:
        cfg.TEMPLATE_FILE = old_template
        cfg.CALIBRATION_FILE = old_calibration
        enrollment.REQUEST_FILE = old_request
        enrollment.TEMPLATE_BACKUP = old_backup
        for path in paths:
            try:
                os.remove(path)
            except OSError:
                pass


try:
    main()
except Exception as error:
    emit("edge_guard_probe", {"status": "failed", "error": str(error)[:180]})
    raise

