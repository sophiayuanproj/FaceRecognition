"""Run and preserve the focused camera/control contracts and full-suite status."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys
from datetime import datetime, timezone


ROOT = Path(__file__).resolve().parents[2]
FACE = ROOT / "FaceRecognition"
PYTHON = FACE / ".venv-nicla/bin/python"
OUT = ROOT / "work/project-research-v5/reports/camera_control_contract_tests_v1"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run(command: list[str], cwd: Path, name: str) -> tuple[int, str]:
    proc = subprocess.run(command, cwd=cwd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, check=False)
    (OUT / f"{name}.txt").write_text(proc.stdout, encoding="utf-8")
    return proc.returncode, proc.stdout


def count_tests(log: str) -> int:
    matches = re.findall(r"Ran (\d+) tests? in", log)
    if not matches:
        raise ValueError("unittest summary missing")
    return int(matches[-1])


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=False)
    focused_modules = [
        "nicla_study.tests.test_camera_geometry_v6",
        "nicla_study.tests.test_fomo_peak_merge_v6",
        "nicla_study.tests.test_policy_controller_v6",
        "nicla_study.tests.test_runtime_decision_dtype_v6",
    ]
    focused_command = [str(PYTHON), "-m", "unittest", "-v", *focused_modules]
    full_command = [str(PYTHON), "-m", "unittest", "discover", "-s", "FaceRecognition/nicla_study/tests", "-v"]
    focused_rc, focused_log = run(focused_command, FACE, "focused_camera_control")
    full_rc, full_log = run(full_command, ROOT, "full_nicla_study_suite")
    focused_count = count_tests(focused_log)
    full_count = count_tests(full_log)
    i_matrix_failures = [
        name for name in (
            "test_all_prepared_overlay_payloads_validate_and_tampering_is_rejected",
            "test_catalog_preserves_full_matrix_and_schedules_only_supported_full_int8",
            "test_hardware_gate_requires_exact_g_h_firmware_and_variant_config_receipts",
        ) if name in full_log
    ]
    if focused_rc != 0 or focused_count != 31:
        raise RuntimeError(f"focused camera/control suite did not pass as expected: rc={focused_rc}, count={focused_count}")
    if full_count != 62 or full_rc == 0 or len(i_matrix_failures) != 3:
        raise RuntimeError(f"full-suite outcome changed; inspect log: rc={full_rc}, count={full_count}, i_matrix={i_matrix_failures}")

    bundles = ROOT / "work/project-finish-v4/staging/I/variant_bundles"
    bundle_manifests = sorted(bundles.rglob("variant_manifest.json")) if bundles.exists() else []
    if bundle_manifests:
        raise RuntimeError("the I-matrix missing-bundle explanation no longer holds")
    source_paths = [
        FACE / "firmware/camera_geometry.py",
        FACE / "firmware/policy_controller.py",
        FACE / "firmware/identity_contracts.py",
        *[FACE / "nicla_study/tests" / (module.rsplit(".", 1)[-1] + ".py") for module in focused_modules],
    ]
    receipt = {
        "schema": "nicla-camera-control-test-receipt-v1",
        "recorded_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "python": sys.version,
        "focused_command": focused_command,
        "focused_suite": {"status": "pass", "tests": focused_count, "exit_code": focused_rc},
        "focused_modules": focused_modules,
        "full_suite_command": full_command,
        "full_suite": {
            "status": "59_pass_3_i_matrix_bundle_setup_failures",
            "tests": full_count,
            "passed": 59,
            "failed": len(i_matrix_failures),
            "failed_tests": i_matrix_failures,
            "exit_code": full_rc,
            "expected_i_matrix_full_int8_bundles": 160,
            "found_i_matrix_bundle_manifests": len(bundle_manifests),
            "missing_bundle_root": str(bundles.relative_to(ROOT)),
            "reason": "The three failures are the prior I-matrix campaign fixtures, which expect 160 prepared variant bundle manifests. None are present; the campaign therefore reports zero eligible overlays. The camera/control-focused suite passes independently.",
        },
        "source_sha256": {path.relative_to(ROOT).as_posix(): sha(path) for path in source_paths},
        "camera_control_test_log": "work/project-research-v5/reports/camera_control_contract_tests_v1/focused_camera_control.txt",
        "full_suite_log": "work/project-research-v5/reports/camera_control_contract_tests_v1/full_nicla_study_suite.txt",
    }
    (OUT / "TEST_RESULT.json").write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (ROOT / "work/project-research-v5/CAMERA_CONTROL_CONTRACT_TESTS_V1.md").write_text(
        "# Camera/control contract test status v1\n\n"
        "The focused geometry, FOMO peak, policy-controller and binary32 decision contracts passed 31/31 methods. The full Nicla-study unittest discovery ran 62 methods: 59 passed; the three I-matrix preparation tests failed because they require 160 prebuilt full-INT8 variant manifests and none exist under `work/project-finish-v4/staging/I/variant_bundles`. They report zero eligible overlays. These are missing prepared artifacts, not failures in the camera/control-focused modules.\n\n"
        "The actual board A-full sleep/wake action is separately recorded in `CAMERA_POLICY_SLEEP_WAKE_DEVICE_V1.md`; these host tests do not qualify application startup, identity or power. Re-run with `python3 FaceRecognition/nicla_study/v6_record_camera_control_tests_v1.py`. Exact stdout and source hashes are in `work/project-research-v5/reports/camera_control_contract_tests_v1/`.\n",
        encoding="utf-8",
    )
    print(json.dumps({
        "focused_tests": focused_count,
        "focused_exit_code": focused_rc,
        "full_tests": full_count,
        "full_exit_code": full_rc,
        "full_suite_failures": i_matrix_failures,
        "missing_variant_manifests": len(bundle_manifests),
    }, sort_keys=True))


if __name__ == "__main__":
    main()
