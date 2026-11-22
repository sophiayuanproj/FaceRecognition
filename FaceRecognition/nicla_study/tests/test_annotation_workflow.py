from __future__ import annotations

import json
from pathlib import Path
import sys
import tempfile
import unittest

APP = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(APP))

from nicla_study.annotation_workflow import (create_annotation_template,
    validate_annotations)  # noqa: E402


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


class AnnotationWorkflowTest(unittest.TestCase):
    def test_development_event_joins_and_requires_independent_visibility_annotation(self):
        with tempfile.TemporaryDirectory(prefix="annotation 空間 ") as temp:
            root = Path(temp)
            events = root / "events.jsonl"
            template = root / "annotations.jsonl"
            validation = root / "validation.json"
            write_jsonl(events, [
                {"event": "session_start", "run_id": "run-1", "session_id": "session-1",
                 "block": "development", "monotonic_ns": 100},
                {"event": "stimulus_start", "run_id": "run-1", "trial_id": "run-1-000000",
                 "trial_index": 0, "condition": "known", "record_ids": ["record-1"],
                 "identity_ids": ["K01"], "roles": ["development"], "monotonic_ns": 200},
                {"event": "stimulus_end", "run_id": "run-1", "trial_id": "run-1-000000",
                 "record_ids": ["record-1"], "monotonic_ns": 900},
                {"event": "session_end", "run_id": "run-1", "monotonic_ns": 1000},
            ])
            receipt = create_annotation_template(events, template)
            self.assertEqual(receipt["status"], "template_created")
            row = json.loads(template.read_text(encoding="utf-8").splitlines()[0])
            row.update({
                "observed_face_count": 1,
                "observed_target_visible": True,
                "appearance_monotonic_ns": 240,
                "usable_interval_start_monotonic_ns": 260,
                "usable_interval_end_monotonic_ns": 800,
                "disappearance_monotonic_ns": 840,
                "appearance_time_source": "synchronized_optical_video",
                "clock_offset_to_controller_ms": 4.5,
                "clock_uncertainty_ms": 8.0,
                "camera_to_target_range_cm": 50.0,
                "range_measurement_method": "tape_measure",
                "camera_orientation_deg": 0.0,
                "lighting_description": "indoor diffuse",
                "wearer_motion": "stationary",
                "independent_observer_id": "observer-01",
                "observer_role": "independent_observer",
                "controller_outputs_seen_before_annotation": False,
            })
            write_jsonl(template, [row])
            result = validate_annotations(events, template)
            self.assertEqual(result["status"], "pass")
            self.assertFalse(result["final_role_opened"])
            row["controller_outputs_seen_before_annotation"] = True
            write_jsonl(template, [row])
            rejected = validate_annotations(events, template)
            self.assertEqual(rejected["status"], "failed_annotation_validation")
            self.assertTrue(any("without seeing controller outputs" in error for error in rejected["errors"]))

    def test_final_annotation_template_requires_sealed_freeze(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            events, output = root / "events.jsonl", root / "annotations.jsonl"
            write_jsonl(events, [
                {"event": "session_start", "run_id": "run-2", "session_id": "session-2",
                 "block": "final", "monotonic_ns": 100},
                {"event": "stimulus_start", "run_id": "run-2", "trial_id": "run-2-000000",
                 "trial_index": 0, "condition": "empty", "record_ids": [], "identity_ids": [],
                 "roles": [], "monotonic_ns": 200},
                {"event": "stimulus_end", "run_id": "run-2", "trial_id": "run-2-000000",
                 "record_ids": [], "monotonic_ns": 900},
            ])
            with self.assertRaises(PermissionError):
                create_annotation_template(events, output)


if __name__ == "__main__":
    unittest.main(verbosity=2)
