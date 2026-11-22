from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path
import tempfile
import unittest

from nicla_study.presentation_player import _hide_stimulus_labels, _load_trials


WORKSPACE = Path(__file__).resolve().parents[3]
STUDY = WORKSPACE / "work/project-finish-v4"


class JIKLifecyclePreparationTest(unittest.TestCase):
    def test_all_stimulus_labels_are_hidden_for_blank_and_close(self):
        class Label:
            def __init__(self):
                self.hidden = 0

            def place_forget(self):
                self.hidden += 1

        first, second = Label(), Label()
        _hide_stimulus_labels((first, second))
        self.assertEqual((first.hidden, second.hidden), (1, 1))

    def test_i_lifecycle_schedule_is_paired_randomized_and_complete(self):
        lifecycle = json.loads((STUDY / "config/I_LIFECYCLE_PLAN.json").read_text(encoding="utf-8"))
        matrix = json.loads((STUDY / "config/I_PHYSICAL_MATRIX.json").read_text(encoding="utf-8"))
        self.assertEqual(lifecycle["paired_block_count"], 10)
        self.assertEqual(lifecycle["planned_lifecycle_capture_count"], 110)
        self.assertEqual(len(lifecycle["cells"]), 110)
        self.assertEqual(len({cell["physical_block_seed"] for cell in lifecycle["cells"]}), 110)
        expected_stages = {(row["stage_id"], row["model_slot"]) for row in lifecycle["lifecycle_stages"]}
        self.assertEqual(len(expected_stages), 11)
        by_block = {}
        for cell in lifecycle["cells"]:
            by_block.setdefault(cell["block_id"], []).append(cell)
        self.assertEqual(len(by_block), 10)
        for rows in by_block.values():
            self.assertEqual(len(rows), 11)
            self.assertEqual({(row["stage_id"], row["model_slot"]) for row in rows}, expected_stages)
            self.assertEqual(len({row["paired_stimulus_seed"] for row in rows}), 1)
        self.assertEqual(matrix["training_root_count"], 10)
        self.assertEqual(matrix["prospective_pairwise_comparison_count"], 544)
        self.assertTrue(all(len(row["paired_artifacts"]) == 10
                            for row in matrix["all_pairwise_arm_format_comparisons"]))
        self.assertFalse(lifecycle["scope_limit"].endswith("complete"))

    def test_mixed_development_player_balances_four_scene_types(self):
        with tempfile.TemporaryDirectory(prefix="jik mixed scenes ") as temp:
            workspace = Path(temp)
            data = workspace / "work/project-finish-v4/data"
            data.mkdir(parents=True)
            rows = []
            for role, prefix in (("development", "K"), ("unknown_development", "U")):
                for index in range(3):
                    relative = "stimuli/%s%d.bin" % (prefix, index)
                    source = workspace / relative
                    source.parent.mkdir(parents=True, exist_ok=True)
                    content = ("%s-%d" % (prefix, index)).encode("ascii")
                    source.write_bytes(content)
                    row = {"record_id": "%s-%d" % (prefix, index),
                           "identity_id": "%s%d" % (prefix, index),
                           "role": role, "dataset": "survface_96",
                           "relative_path": relative,
                           "sha256": hashlib.sha256(content).hexdigest()}
                    rows.append(row)
            (data / "primary_split_manifest.jsonl").write_text(
                "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
            trials = _load_trials(workspace, "development", 1234, None,
                                  condition="mixed", trial_count=12)
            counts = Counter(row["condition"] for row in trials)
            self.assertEqual(counts, {"known": 3, "unknown": 3, "empty": 3, "ambiguous": 3})


if __name__ == "__main__":
    unittest.main()
