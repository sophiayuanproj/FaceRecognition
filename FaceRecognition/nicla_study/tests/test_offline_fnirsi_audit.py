from __future__ import annotations

import json
from pathlib import Path
import sys
import tempfile
import unittest

APP = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(APP))

from nicla_study.instruments.windows_fnirsi.offline_audit import audit_run  # noqa: E402
from nicla_study.instruments.windows_fnirsi.upstream.power.fnirsi_protocol import crc8  # noqa: E402


def make_report(samples, *, header=b"\xaa\x04", reserved=0, corrupt_crc=False):
    payload = bytearray(64)
    payload[:2] = header
    for index, sample in enumerate(samples):
        offset = 2 + index * 15
        voltage, current, flag = sample
        payload[offset:offset + 4] = int(voltage).to_bytes(4, "little")
        payload[offset + 4:offset + 8] = int(current).to_bytes(4, "little")
        payload[offset + 8:offset + 12] = bytes(4)
        payload[offset + 12] = int(flag)
        payload[offset + 13:offset + 15] = (370).to_bytes(2, "little")
    payload[62] = reserved
    payload[63] = crc8(payload[1:-1])
    if corrupt_crc:
        payload[63] ^= 0x01
    return bytes(payload)


class OfflineFnirsiAuditTest(unittest.TestCase):
    def test_ten_seeded_raw_input_blocks_preserve_anomalies_and_reject_corruption(self):
        with tempfile.TemporaryDirectory(prefix="fnirsi audit 空間 ") as temp:
            root = Path(temp)
            for seed in range(10):
                run = root / ("run-%02d" % seed)
                run.mkdir()
                samples = [(520000, 4700, seed), (0, 4700, seed),
                           (6553587, 4700, seed), (519900, 419427, seed)]
                good = make_report(samples, reserved=seed)
                bad_crc = make_report(samples, reserved=seed, corrupt_crc=True)
                bad_type = make_report(samples, header=b"\xaa\x03", reserved=seed)
                rows = [
                    {"record_type": "measurement_report", "packet_status": "valid_measurement_report",
                     "raw_report_hex": good.hex(), "report_arrival_index": 10 + seed, "samples": []},
                    {"record_type": "invalid_report", "packet_status": "crc_mismatch",
                     "raw_report_hex": bad_crc.hex(), "report_arrival_index": 20 + seed, "samples": []},
                    {"record_type": "invalid_report", "packet_status": "wrong_header_or_report_type",
                     "raw_report_hex": bad_type.hex(), "report_arrival_index": 30 + seed, "samples": []},
                ]
                (run / "raw_packets.jsonl").write_text(
                    "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
                (run / "run_metadata.json").write_text(json.dumps({"run_id": "seed-%d" % seed}), encoding="utf-8")
                report = audit_run(run)
                self.assertEqual(report["status"], "complete_offline_audit")
                self.assertEqual(report["counts"]["decoded_crc_valid_reports"], 1)
                self.assertEqual(report["counts"]["crc_rejections"], 1)
                self.assertEqual(report["counts"]["wrong_header_or_type_rejections"], 1)
                self.assertEqual(report["counts"]["decoded_samples"], 4)
                self.assertEqual(report["slot_summaries"]["1"]["zero_voltage_count"], 1)
                self.assertEqual(report["slot_summaries"]["2"]["voltage_above_diagnostic_gate_count"], 1)
                self.assertEqual(report["slot_summaries"]["3"]["current_above_diagnostic_gate_count"], 1)
                self.assertEqual(report["claims"]["decoder_change"].split(";")[0], "none justified")
                self.assertEqual(report["counts"]["reserved_byte_62_counts"], {"%02x" % seed: 3})


if __name__ == "__main__":
    unittest.main(verbosity=2)
