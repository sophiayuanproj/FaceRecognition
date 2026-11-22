from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from datetime import datetime, timezone

from nicla_study.artifacts import build_artifact_manifest, verify_artifact_manifest
from nicla_study.contracts import validate_model_output
from nicla_study.device_gates import require_fresh_device
from nicla_study.fs import atomic_write_json
from nicla_study.paths import resolve_registry
from nicla_study.roles import validate_manifest
from nicla_study.state import (complete_phase_from_receipt, initialize_state,
                               mark_stale_running_phases_interrupted, reconcile_stale_resource_leases,
                               resource_lease, resume_phase, start_phase, state_paths, transact)
from nicla_study.instruments.windows_fnirsi.instrument_adapter import (
    bounded_wait, reconnect_candidates,
)


class TenSeededFaultInputBlocks(unittest.TestCase):
    def test_ten_fault_blocks_and_resume(self):
        seeds = list(range(20261501, 20261511))
        rows = []
        with tempfile.TemporaryDirectory(prefix="study faults space λ ") as temp:
            base = Path(temp)
            workspace = base / "relocated workspace λ"
            workspace.mkdir()
            (workspace / "work/project-finish-v4").mkdir(parents=True)
            # 1: relocation into a path with spaces and Unicode.
            registry = {"workspace":{"mac_local_absolute":"/stale/mac/path"},
                        "paths":{"study":{"relative":"work/project-finish-v4"}}}
            resolved = resolve_registry(registry, workspace=workspace)
            self.assertTrue(resolved["paths"]["study"]["exists"])
            rows.append({"root_seed":seeds[0],"case":"path_relocation_spaces_unicode","status":"pass"})

            # 2: an artifact hash mismatch must be rejected.
            artifact = workspace / "artifact.bin"
            artifact.write_bytes(b"original")
            manifest = build_artifact_manifest(workspace, ["artifact.bin"])
            artifact.write_bytes(b"replaced")
            hash_result = verify_artifact_manifest(workspace, manifest)
            self.assertEqual(hash_result["status"],"failed")
            self.assertTrue(any("hash mismatch" in error for error in hash_result["errors"]))
            rows.append({"root_seed":seeds[1],"case":"artifact_hash_mismatch","status":"pass"})

            # 3: an interrupted write leaves the prior durable record intact.
            state_path = workspace / "durable.json"
            atomic_write_json(state_path, {"generation":1})
            with self.assertRaises(OSError):
                atomic_write_json(state_path, {"generation":2}, failpoint="before_replace")
            self.assertEqual(json.loads(state_path.read_text())["generation"],1)
            rows.append({"root_seed":seeds[2],"case":"interrupted_atomic_write","status":"pass"})

            # 4: partial receipts cannot complete a phase; restart survives and resumes.
            test_study = workspace / "work/project-finish-v4/test-state"
            phase_plan = {"phases":[{"id":"software_phase","host":"mac","job":"TEST",
                                     "requires":[],"resources":[]}]}
            initialize_state(test_study,phase_plan,"test/plan.json","fault-test-v1")
            start_phase(test_study,"software_phase")
            partial = workspace / "partial-receipt.json"
            partial.write_text(json.dumps({"status":"partial_external_dependency",
                                           "acceptance_checks":[{"id":"no","status":"pending"}]}))
            with self.assertRaises(RuntimeError):
                complete_phase_from_receipt(test_study,"software_phase",partial)
            self.assertEqual(json.loads(state_paths(test_study)[0].read_text())["phases"]["software_phase"]["status"],
                             "in_progress")
            self.assertEqual(mark_stale_running_phases_interrupted(test_study),["software_phase"])
            self.assertEqual(resume_phase(test_study,"software_phase")["status"],"ready")
            start_phase(test_study,"software_phase")
            verified = workspace / "software-receipt.json"
            verified.write_text(json.dumps({"status":"complete_verified",
                                            "evidence_tier":"software_unit",
                                            "acceptance_checks":[{"id":"local_software_only","status":"pass"}]}))
            self.assertEqual(complete_phase_from_receipt(test_study,"software_phase",verified)["status"],
                             "complete_verified")
            events = state_paths(test_study)[2].read_text().splitlines()
            self.assertGreaterEqual(len(events),6)
            rows.append({"root_seed":seeds[3],"case":"stale_partial_receipt_and_resume","status":"pass",
                         "software_phase_complete":True,"physical_success_created":False})

            # 5: invalid manifests, including unknown-person partition leakage, fail validation.
            bad_rows=[]
            for index, role in enumerate(("unknown_calibration","unknown_final")):
                bad_rows.append({"record_id":str(index),"identity_id":"same-unknown",
                                 "role":role,"sha256":"a"*64,"relative_path":"../escape.png"})
            result=validate_manifest(bad_rows)
            self.assertEqual(result["status"],"invalid")
            rows.append({"root_seed":seeds[4],"case":"invalid_manifest","status":"pass"})

            # 6: a classifier vector cannot be consumed as an embedding.
            with self.assertRaises(TypeError):
                validate_model_output({"model_type":"classifier_logits","values":[0.0]*6},
                                      "embedding",64)
            rows.append({"root_seed":seeds[5],"case":"wrong_model_type","status":"pass"})

            # 7: absent hardware stays pending; no success record is produced.
            discovery={"enumerated_utc":datetime.now(timezone.utc).isoformat(),
                       "hid_devices":[],"serial_ports":[]}
            gate=require_fresh_device(discovery,"nicla")
            self.assertEqual(gate["status"],"pending_external_dependency")
            self.assertFalse((workspace/"physical-success.json").exists())
            rows.append({"root_seed":seeds[6],"case":"absent_device_fails_closed","status":"pass",
                         "action_status":gate["status"]})

            # 8: a second process cannot acquire the same named lease; stale
            # state rows are reconciled only after the operating-system lock is free.
            code=("from nicla_study.state import resource_lease\nimport sys\n"
                  "try:\n with resource_lease(sys.argv[1],'meter',owner='child',timeout_s=0.15): pass\n"
                  "except TimeoutError:\n sys.exit(0)\n"
                  "sys.exit(17)\n")
            with resource_lease(test_study,"meter",owner="test-parent",timeout_s=1):
                proc=subprocess.run([sys.executable,"-c",code,str(test_study)],
                                    cwd=Path(__file__).resolve().parents[2],
                                    capture_output=True,text=True,timeout=5)
            self.assertEqual(proc.returncode,0,proc.stderr)
            state=json.loads(state_paths(test_study)[0].read_text())
            self.assertEqual(state["leases"]["active"],[])
            stale_record={"lease_id":"dead-owner","resource":"meter","owner":"interrupted-test","pid":-1}
            transact(test_study,lambda current:current["leases"]["active"].append(stale_record),
                     event_type="test_stale_lease_injected")
            self.assertEqual(reconcile_stale_resource_leases(test_study),[stale_record])
            rows.append({"root_seed":seeds[7],"case":"cross_process_resource_lease_contention_and_stale_recovery","status":"pass",
                         "lock_backend":"fcntl on this Mac","active_rows_after_release":0,"stale_rows_reconciled":1})

            # 9: bounded device operations time out without global signal handlers.
            with self.assertRaises(TimeoutError):
                bounded_wait(lambda:False,timeout_s=0.04,interval_s=0.005)
            rows.append({"root_seed":seeds[8],"case":"bounded_timeout","status":"pass"})

            # 10: reconnect follows a stable identity even when the COM number changes.
            prior={"serial_identity_sha256":"id-a","vid":0x2341,"pid":0x805A,"product":"Nano","manufacturer":"Arduino"}
            candidates=[{"device":"COM9","serial_identity_sha256":"id-a","vid":0x2341,"pid":0x805A,"product":"Nano","manufacturer":"Arduino"},
                        {"device":"COM8","serial_identity_sha256":"id-b","vid":0x2341,"pid":0x805A,"product":"Nano","manufacturer":"Arduino"}]
            matched=reconnect_candidates(prior,candidates)
            self.assertEqual([row["device"] for row in matched],["COM9"])
            rows.append({"root_seed":seeds[9],"case":"reconnect_same_identity_new_port","status":"pass"})

        self.assertEqual(len(rows),10)
        self.assertTrue(all(row["status"]=="pass" for row in rows))
        self.assertEqual(len(set(seeds)),10)
        report = {"schema":"cross-process-resumability-fault-blocks-1",
                  "status":"pass","seed_unit":"software input/fault block; no hardware trial implied",
                  "blocks":rows}
        root=Path(__file__).resolve().parents[3]
        out=root/"work/project-finish-v4/reports/fault-resumability-tests.json"
        out.parent.mkdir(parents=True,exist_ok=True)
        out.write_text(json.dumps(report,indent=2)+"\n")


if __name__ == "__main__":
    unittest.main(verbosity=2)
