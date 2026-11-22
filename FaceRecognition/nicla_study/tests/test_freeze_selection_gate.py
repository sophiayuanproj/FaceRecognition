from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest

APP = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(APP))

from nicla_study.freeze_selection import freeze, verify_receipt_sources  # noqa: E402


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class FreezeSelectionGateTest(unittest.TestCase):
    def test_freeze_verifies_every_source_and_refuses_overwrite(self):
        with tempfile.TemporaryDirectory(prefix="freeze gate ") as temp:
            root = Path(temp)
            model = root / "model.tflite"
            model.write_bytes(b"model")
            config = root / "runtime.json"
            config.write_text(json.dumps({"identity": {"small": {"model_sha256": digest(model)}}}), encoding="utf-8")
            calibration = root / "calibration.json"
            calibration.write_text("{}\n", encoding="utf-8")
            roster = root / "roster.json"
            roster.write_text("{\"ids\":[\"K01\"]}\n", encoding="utf-8")
            controller = root / "controller.json"
            controller.write_text("{\"policies\":[\"H\"]}\n", encoding="utf-8")
            protocol = root / "protocol.json"
            protocol.write_text("{\"final_role_access\":false}\n", encoding="utf-8")
            firmware = "a" * 64
            candidate = {"model_id": "small-model", "data_role": "development", "used_final_data": False,
                "model_sha256": digest(model), "runtime_config_sha256": digest(config),
                "calibration_sha256": digest(calibration), "roster_sha256": digest(roster),
                "controller_sha256": digest(controller)}
            selection = root / "selection.json"
            selection.write_text(json.dumps({"status": "selected_development_candidate", "freeze_allowed": True,
                "data_role": "development_only", "used_final_data": False, "selected_candidate": candidate}),
                encoding="utf-8")
            receipt = root / "FROZEN_SELECTION.json"
            freeze(selection, config, model, calibration, roster, controller, protocol, firmware, receipt)
            verified = verify_receipt_sources(receipt)
            self.assertEqual(verified["status"], "pass")
            self.assertEqual(verified["source_file_count"], 7)
            with self.assertRaises(FileExistsError):
                freeze(selection, config, model, calibration, roster, controller, protocol, firmware, receipt)
            calibration.write_text("{\"changed\":true}\n", encoding="utf-8")
            with self.assertRaises(ValueError):
                verify_receipt_sources(receipt)


if __name__ == "__main__":
    unittest.main(verbosity=2)
