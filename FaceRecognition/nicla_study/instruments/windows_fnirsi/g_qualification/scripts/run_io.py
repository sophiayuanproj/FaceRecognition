"""Versioned append-only run writer for raw HID packets and decoded samples."""

import csv
from datetime import datetime, timezone
import json
from pathlib import Path
import time
from typing import Dict, Optional

from fnirsi_protocol import PacketDecodeError, crc8, decode_report

CSV_COLUMNS = [
    "schema_version", "experiment_id", "run_id", "policy_id", "scenario_id", "root_seed",
    "meter_model_reported", "meter_firmware", "parser_version", "report_arrival_index",
    "sample_index_in_report", "host_receive_monotonic_ns", "host_receive_utc",
    "device_sample_tick", "estimated_device_sample_time_ns", "timing_basis",
    "voltage_v", "current_a", "power_w_calculated", "current_signedness",
    "sample_flag_raw_hex", "sample_flag_semantics", "dplus_v", "dminus_v", "temperature_c",
    "crc_status", "data_quality_flags", "raw_packet_file", "raw_report_hex",
]


def utc_now():
    return datetime.now(timezone.utc).isoformat(timespec="microseconds")


class RunWriter:
    def __init__(self, output_dir: Path, metadata: Dict[str, object]):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=False)
        self.metadata = dict(metadata)
        self.metadata.update({
            "schema_version": "power-run-1.0",
            "parser_version": "nicla-power-fnirsi-1.0.0",
            "status": "running",
            "started_utc": utc_now(),
            "started_monotonic_ns": time.monotonic_ns(),
            "raw_packet_file": "raw_packets.jsonl",
            "decoded_table_file": "decoded_samples.csv",
        })
        self.meta_path = self.output_dir / "run_metadata.json"
        self.raw_path = self.output_dir / "raw_packets.jsonl"
        self.csv_path = self.output_dir / "decoded_samples.csv"
        self.meta_path.open("x", encoding="utf-8").close()
        self.raw = self.raw_path.open("x", encoding="utf-8")
        self.csv_file = self.csv_path.open("x", newline="", encoding="utf-8")
        self.csv_writer = csv.DictWriter(self.csv_file, fieldnames=CSV_COLUMNS, extrasaction="ignore")
        self.csv_writer.writeheader()
        self.report_count = 0
        self.sample_count = 0
        self.invalid_count = 0
        self.timeout_count = 0
        self.previous_payload = None
        self.exact_duplicate_payload_count = 0
        self.write_metadata()
        self.event("run_start", {"host_monotonic_ns": time.monotonic_ns(), "host_utc": utc_now()})

    def write_metadata(self):
        temp = self.meta_path.with_suffix(".json.tmp")
        with temp.open("w", encoding="utf-8") as handle:
            json.dump(self.metadata, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
        temp.replace(self.meta_path)

    def event(self, event_type: str, fields: Optional[Dict[str, object]] = None):
        row = {"record_type": "event", "event": event_type,
               "experiment_id": self.metadata.get("experiment_id"),
               "run_id": self.metadata.get("run_id"),
               "policy_id": self.metadata.get("policy_id"),
               "scenario_id": self.metadata.get("scenario_id"),
               "root_seed": self.metadata.get("root_seed"),
               "host_receive_utc": utc_now(),
               "host_receive_monotonic_ns": time.monotonic_ns()}
        if fields:
            row.update(fields)
        self.raw.write(json.dumps(row, sort_keys=True, allow_nan=False) + "\n")
        self.raw.flush()

    def record_timeout(self, arrival_index: int):
        self.timeout_count += 1
        row = {"record_type": "timeout", "experiment_id": self.metadata.get("experiment_id"),
               "run_id": self.metadata.get("run_id"), "policy_id": self.metadata.get("policy_id"),
               "scenario_id": self.metadata.get("scenario_id"), "root_seed": self.metadata.get("root_seed"),
               "report_arrival_index": arrival_index,
               "host_receive_utc": utc_now(), "host_receive_monotonic_ns": time.monotonic_ns(),
               "data_quality_flags": ["hid_read_timeout"]}
        self.raw.write(json.dumps(row, sort_keys=True) + "\n")
        self.raw.flush()

    def record_report(self, payload: bytes, arrival_index: int, host_mono_ns: int, host_utc: str,
                      possible_gap: bool = False):
        exact_duplicate = self.previous_payload == payload if self.previous_payload is not None else False
        self.previous_payload = payload
        if exact_duplicate:
            self.exact_duplicate_payload_count += 1
        try:
            samples = decode_report(payload)
            crc_status = "pass"
            packet_status = "valid_measurement_report"
        except PacketDecodeError as exc:
            samples = []
            reason = str(exc)
            crc_status = "fail" if reason == "crc_mismatch" else "not_checked"
            packet_status = reason
            self.invalid_count += 1
        packet = {
            "record_type": "measurement_report" if samples else "invalid_report",
            "schema_version": "power-run-1.0",
            "experiment_id": self.metadata.get("experiment_id"),
            "run_id": self.metadata.get("run_id"),
            "policy_id": self.metadata.get("policy_id"),
            "scenario_id": self.metadata.get("scenario_id"),
            "root_seed": self.metadata.get("root_seed"),
            "report_arrival_index": arrival_index,
            "host_receive_monotonic_ns": host_mono_ns,
            "host_receive_utc": host_utc,
            "report_length_bytes": len(payload),
            "report_header_hex": payload[:2].hex(),
            "crc_status": crc_status,
            "packet_status": packet_status,
            "raw_report_hex": payload.hex(),
            "possible_exact_duplicate_payload": exact_duplicate,
            "duplicate_interpretation": "diagnostic_only; identical values alone do not prove stale acquisition",
            "samples": samples,
            "timing_basis": "host_report_reception_only; acquisition time not present/validated",
        }
        self.raw.write(json.dumps(packet, sort_keys=True, allow_nan=False) + "\n")
        self.raw.flush()
        if samples:
            self.report_count += 1
            for sample in samples:
                flags = ["sample_time_unknown", "flag_direction_semantics_unknown"]
                if possible_gap:
                    flags.append("report_gap_over_provisional_200ms")
                row = {
                    "schema_version": "power-run-1.0",
                    "experiment_id": self.metadata.get("experiment_id"),
                    "run_id": self.metadata.get("run_id"),
                    "policy_id": self.metadata.get("policy_id"),
                    "scenario_id": self.metadata.get("scenario_id"),
                    "root_seed": self.metadata.get("root_seed"),
                    "meter_model_reported": self.metadata.get("meter_model_reported", ""),
                    "meter_firmware": self.metadata.get("meter_firmware", ""),
                    "parser_version": self.metadata["parser_version"],
                    "report_arrival_index": arrival_index,
                    "sample_index_in_report": sample["sample_index_in_report"],
                    "host_receive_monotonic_ns": host_mono_ns,
                    "host_receive_utc": host_utc,
                    "device_sample_tick": "",
                    "estimated_device_sample_time_ns": "",
                    "timing_basis": "host_report_reception_only",
                    "voltage_v": sample["voltage_v"],
                    "current_a": sample["current_a"],
                    "power_w_calculated": sample["power_w_calculated"],
                    "current_signedness": sample["current_signedness"],
                    "sample_flag_raw_hex": sample["sample_flag_raw_hex"],
                    "sample_flag_semantics": sample["sample_flag_semantics"],
                    "dplus_v": sample["dplus_v"],
                    "dminus_v": sample["dminus_v"],
                    "temperature_c": sample["temperature_c"],
                    "crc_status": crc_status,
                    "data_quality_flags": ";".join(flags),
                    "raw_packet_file": "raw_packets.jsonl",
                    "raw_report_hex": payload.hex(),
                }
                self.csv_writer.writerow(row)
                self.sample_count += 1
            self.csv_file.flush()
        return packet

    def close(self, status: str, error: Optional[str] = None):
        if self.raw.closed:
            return
        self.event("run_end", {"status": status, "error": error,
                                "host_monotonic_ns": time.monotonic_ns(), "host_utc": utc_now()})
        self.raw.flush()
        self.csv_file.flush()
        self.raw.close()
        self.csv_file.close()
        self.metadata.update({
            "status": status,
            "error": error,
            "ended_utc": utc_now(),
            "ended_monotonic_ns": time.monotonic_ns(),
            "measurement_report_count": self.report_count,
            "decoded_sample_count": self.sample_count,
            "invalid_packet_count": self.invalid_count,
            "read_timeout_count": self.timeout_count,
            "possible_exact_duplicate_payload_count": self.exact_duplicate_payload_count,
        })
        self.write_metadata()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        status = "complete" if exc is None else "interrupted"
        self.close(status, None if exc is None else "%s: %s" % (exc_type.__name__, exc))
        return False
