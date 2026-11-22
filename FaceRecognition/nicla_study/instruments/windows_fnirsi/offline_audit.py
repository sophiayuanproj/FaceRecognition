"""Read-only anomaly audit for preserved FNIRSI JSONL captures.

This utility re-decodes raw HID payloads with the maintained parser and compares
the result with the packet's existing derived fields. It never edits a run or
removes anomalous samples.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import statistics

from .upstream.power.fnirsi_protocol import PacketDecodeError, decode_report


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_json(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("expected a JSON object: " + str(path))
    return value


def audit_run(run_dir: Path, *, voltage_diagnostic_max_v: float = 5.25,
              current_diagnostic_max_a: float = 0.20) -> dict:
    """Audit one immutable run directory without changing any input file."""
    run_dir = run_dir.resolve()
    raw_path = run_dir / "raw_packets.jsonl"
    metadata_path = run_dir / "run_metadata.json"
    if not raw_path.is_file() or not metadata_path.is_file():
        raise FileNotFoundError("run directory needs raw_packets.jsonl and run_metadata.json")
    metadata = _read_json(metadata_path)
    row_type_counts: Counter[str] = Counter()
    packet_status_counts: Counter[str] = Counter()
    parse_error_counts: Counter[str] = Counter()
    flag_counts: dict[str, Counter[str]] = {str(index): Counter() for index in range(4)}
    reserved_byte_counts: Counter[str] = Counter()
    duplicate_hash_counts: Counter[str] = Counter()
    samples_by_slot: dict[str, list[dict]] = {str(index): [] for index in range(4)}
    report_medians_v: list[float] = []
    anomaly_examples: dict[str, list[dict]] = {
        "zero_voltage": [], "voltage_above_diagnostic_gate": [],
        "current_above_diagnostic_gate": [], "parser_vs_logged_field_mismatch": [],
    }
    json_lines = 0
    malformed_json_lines = []
    raw_report_count = 0
    parsed_report_count = 0
    report_decoder_error_count = 0
    invalid_crc_count = 0
    invalid_length_count = 0
    invalid_header_type_count = 0
    logged_sample_mismatch_count = 0
    reports_with_any_diagnostic_value = 0

    with raw_path.open("r", encoding="utf-8-sig") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            json_lines += 1
            try:
                row = json.loads(line)
            except json.JSONDecodeError as error:
                malformed_json_lines.append({"line": line_number, "error": str(error)})
                continue
            if not isinstance(row, dict):
                malformed_json_lines.append({"line": line_number, "error": "row_not_object"})
                continue
            record_type = str(row.get("record_type", "missing"))
            row_type_counts[record_type] += 1
            packet_status_counts[str(row.get("packet_status", "missing"))] += 1
            payload_hex = row.get("raw_report_hex")
            if not isinstance(payload_hex, str):
                continue
            raw_report_count += 1
            try:
                payload = bytes.fromhex(payload_hex)
            except ValueError:
                parse_error_counts["invalid_hex"] += 1
                report_decoder_error_count += 1
                continue
            if len(payload) == 64:
                reserved_byte_counts["%02x" % payload[62]] += 1
            duplicate_hash_counts[hashlib.sha256(payload).hexdigest()] += 1
            try:
                decoded = decode_report(payload)
            except PacketDecodeError as error:
                report_decoder_error_count += 1
                reason = str(error)
                parse_error_counts[reason.split(":", 1)[0]] += 1
                if reason.startswith("crc_mismatch"):
                    invalid_crc_count += 1
                elif reason.startswith("wrong_length"):
                    invalid_length_count += 1
                elif reason.startswith("wrong_header_or_report_type"):
                    invalid_header_type_count += 1
                continue
            parsed_report_count += 1
            reported_samples = row.get("samples", [])
            report_has_diagnostic_value = False
            report_voltages = []
            for slot, sample in enumerate(decoded):
                slot_key = str(slot)
                samples_by_slot[slot_key].append(sample)
                flag_counts[slot_key][str(sample["sample_flag_raw_hex"])] += 1
                voltage = float(sample["voltage_v"])
                current = float(sample["current_a"])
                report_voltages.append(voltage)
                if voltage == 0.0:
                    report_has_diagnostic_value = True
                    if len(anomaly_examples["zero_voltage"]) < 12:
                        offset = 2 + slot * 15
                        anomaly_examples["zero_voltage"].append({
                            "report_arrival_index": row.get("report_arrival_index"),
                            "slot": slot,
                            "voltage_v": voltage,
                            "current_a": current,
                            "flag_hex": sample["sample_flag_raw_hex"],
                            "slot_bytes_hex": payload[offset:offset + 15].hex(),
                        })
                if voltage > voltage_diagnostic_max_v:
                    report_has_diagnostic_value = True
                    if len(anomaly_examples["voltage_above_diagnostic_gate"]) < 12:
                        offset = 2 + slot * 15
                        anomaly_examples["voltage_above_diagnostic_gate"].append({
                            "report_arrival_index": row.get("report_arrival_index"),
                            "slot": slot,
                            "voltage_v": voltage,
                            "current_a": current,
                            "flag_hex": sample["sample_flag_raw_hex"],
                            "slot_bytes_hex": payload[offset:offset + 15].hex(),
                        })
                if current > current_diagnostic_max_a:
                    report_has_diagnostic_value = True
                    if len(anomaly_examples["current_above_diagnostic_gate"]) < 12:
                        offset = 2 + slot * 15
                        anomaly_examples["current_above_diagnostic_gate"].append({
                            "report_arrival_index": row.get("report_arrival_index"),
                            "slot": slot,
                            "voltage_v": voltage,
                            "current_a": current,
                            "flag_hex": sample["sample_flag_raw_hex"],
                            "slot_bytes_hex": payload[offset:offset + 15].hex(),
                        })
                if slot < len(reported_samples):
                    logged = reported_samples[slot]
                    for field in ("voltage_raw_u32", "current_raw_u32", "sample_flag_raw_hex"):
                        if logged.get(field) != sample.get(field):
                            logged_sample_mismatch_count += 1
                            if len(anomaly_examples["parser_vs_logged_field_mismatch"]) < 12:
                                anomaly_examples["parser_vs_logged_field_mismatch"].append({
                                    "report_arrival_index": row.get("report_arrival_index"),
                                    "slot": slot, "field": field,
                                    "logged": logged.get(field), "redecoded": sample.get(field),
                                })
                            break
            if report_has_diagnostic_value:
                reports_with_any_diagnostic_value += 1
            if report_voltages:
                report_medians_v.append(float(statistics.median(report_voltages)))

    slot_summaries = {}
    for slot, samples in samples_by_slot.items():
        voltages = [float(sample["voltage_v"]) for sample in samples]
        currents = [float(sample["current_a"]) for sample in samples]
        slot_summaries[slot] = {
            "decoded_sample_count": len(samples),
            "flag_counts": dict(sorted(flag_counts[slot].items())),
            "zero_voltage_count": sum(value == 0.0 for value in voltages),
            "voltage_above_diagnostic_gate_count": sum(value > voltage_diagnostic_max_v for value in voltages),
            "current_above_diagnostic_gate_count": sum(value > current_diagnostic_max_a for value in currents),
            "max_voltage_v": max(voltages) if voltages else None,
            "max_current_a": max(currents) if currents else None,
        }
    duplicate_payload_extra_count = sum(count - 1 for count in duplicate_hash_counts.values() if count > 1)
    return {
        "schema": "fnirsi-offline-anomaly-audit-1",
        "status": "complete_offline_audit" if not malformed_json_lines else "complete_with_malformed_rows",
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "evidence_tier": "offline_raw_packet_redecode_no_physical_qualification",
        "run_id": metadata.get("run_id"),
        "input_paths": {"raw_packets": str(raw_path), "run_metadata": str(metadata_path)},
        "input_sha256": {"raw_packets": _sha256(raw_path), "run_metadata": _sha256(metadata_path)},
        "decoder": {"module": "nicla_study.instruments.windows_fnirsi.upstream.power.fnirsi_protocol",
                    "sha256": _sha256(Path(__file__).parent / "upstream/power/fnirsi_protocol.py"),
                    "report_length_bytes": 64, "header_and_type_hex": "aa04",
                    "sample_count_per_report": 4, "sample_bytes_per_slot": 15,
                    "voltage_current_scale": 100000, "dplus_dminus_scale": 1000,
                    "crc_poly": "0x39", "crc_initial": "0x42"},
        "diagnostic_rules_only_not_acceptance_gates": {
            "zero_voltage": "exactly 0.0 V",
            "voltage_above_diagnostic_gate_v": voltage_diagnostic_max_v,
            "current_above_diagnostic_gate_a": current_diagnostic_max_a,
            "interpretation": "inspection flags only; no samples were filtered or declared physically impossible by this audit"},
        "counts": {"jsonl_rows": json_lines, "record_type_counts": dict(sorted(row_type_counts.items())),
            "packet_status_counts": dict(sorted(packet_status_counts.items())),
            "raw_reports": raw_report_count, "decoded_crc_valid_reports": parsed_report_count,
            "decoder_rejections": report_decoder_error_count,
            "crc_rejections": invalid_crc_count, "wrong_length_rejections": invalid_length_count,
            "wrong_header_or_type_rejections": invalid_header_type_count,
            "malformed_json_rows": len(malformed_json_lines),
            "logged_raw_sample_field_mismatch_reports_or_fields": logged_sample_mismatch_count,
            "decoded_samples": sum(len(values) for values in samples_by_slot.values()),
            "reports_with_any_diagnostic_value": reports_with_any_diagnostic_value,
            "exact_duplicate_payload_repeats_after_first": duplicate_payload_extra_count,
            "reserved_byte_62_counts": dict(sorted(reserved_byte_counts.items()))},
        "slot_summaries": slot_summaries,
        "report_median_voltage_v": {
            "count": len(report_medians_v),
            "median": statistics.median(report_medians_v) if report_medians_v else None,
            "below_provisional_4p75_v_count": sum(value < 4.75 for value in report_medians_v),
            "within_provisional_4p75_to_5p25_v_count": sum(4.75 <= value <= 5.25 for value in report_medians_v),
            "above_provisional_5p25_v_count": sum(value > 5.25 for value in report_medians_v),
        },
        "anomaly_examples": anomaly_examples,
        "malformed_json_examples": malformed_json_lines[:12],
        "claims": {
            "normal_central_scaling": "compared against the maintained decoder; independent calibration requires reference measurements",
            "spike_and_zero_root_cause": "a raw replay can locate anomalous slots but cannot determine their physical cause",
            "flag_semantics": "unknown; reported raw values are retained and do not distinguish the observed anomalies",
            "sample_timestamp_basis": "host report-reception only; the four decoded slots are not assigned intra-report timestamps",
            "decoder_change": "none justified; preserve the existing parser and raw packet until Windows commissioning supplies documented corroboration",
            "energy_qualification": "requires separate instrument, topology, and time-window evidence"},
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    args = parser.parse_args(argv)
    result = audit_run(args.run_dir)
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    tmp = args.output_json.with_suffix(args.output_json.suffix + ".tmp")
    tmp.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    tmp.replace(args.output_json)
    print(json.dumps({"status": result["status"], "run_id": result["run_id"],
                      "counts": result["counts"], "output_json": str(args.output_json)}, indent=2))
    return 0 if result["status"] == "complete_offline_audit" else 2


if __name__ == "__main__":
    raise SystemExit(main())
