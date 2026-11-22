"""Conservative decoder for the FNIRSI FNB-C2/FNB58 64-byte HID reports.

The physical meter model is not identifiable from VID:PID 2e3c:5558 alone.
This module preserves unknown flag bytes and never invents device sample times.
"""

from typing import Dict, List, Union

VID = 0x2E3C
PID = 0x5558
REPORT_LEN = 64
SAMPLE_COUNT = 4
SAMPLE_LEN = 15
CRC_POLY = 0x39
CRC_INIT = 0x42


class PacketDecodeError(ValueError):
    pass


def crc8(data: Union[bytes, bytearray]) -> int:
    crc = CRC_INIT
    for byte in data:
        crc ^= byte
        for _ in range(8):
            if crc & 0x80:
                crc = ((crc << 1) ^ CRC_POLY) & 0xFF
            else:
                crc = (crc << 1) & 0xFF
    return crc


def make_command(code: int) -> bytes:
    if code not in (0x81, 0x82):
        raise ValueError("Only standard measurement init/read commands are allowed")
    report = bytearray(REPORT_LEN)
    report[0] = 0xAA
    report[1] = code
    report[-1] = crc8(report[1:-1])
    return bytes(report)


def decode_report(payload: bytes) -> List[Dict[str, object]]:
    """Decode a HIDAPI payload; returned sample times are deliberately absent."""
    if len(payload) != REPORT_LEN:
        raise PacketDecodeError("wrong_length:%d" % len(payload))
    if payload[0] != 0xAA or payload[1] != 0x04:
        raise PacketDecodeError("wrong_header_or_report_type")
    expected = crc8(payload[1:-1])
    if expected != payload[-1]:
        raise PacketDecodeError("crc_mismatch")

    samples = []
    for index in range(SAMPLE_COUNT):
        offset = 2 + index * SAMPLE_LEN
        voltage_raw = int.from_bytes(payload[offset:offset + 4], "little", signed=False)
        current_raw = int.from_bytes(payload[offset + 4:offset + 8], "little", signed=False)
        dplus_raw = int.from_bytes(payload[offset + 8:offset + 10], "little", signed=False)
        dminus_raw = int.from_bytes(payload[offset + 10:offset + 12], "little", signed=False)
        flag_raw = payload[offset + 12]
        temperature_raw = int.from_bytes(payload[offset + 13:offset + 15], "little", signed=False)
        voltage_v = voltage_raw / 100000.0
        current_a = current_raw / 100000.0
        dplus_v = dplus_raw / 1000.0
        dminus_v = dminus_raw / 1000.0
        temperature_c = temperature_raw / 10.0
        power_w = voltage_v * current_a
        samples.append({
            "sample_index_in_report": index,
            "voltage_raw_u32": voltage_raw,
            "current_raw_u32": current_raw,
            "current_signedness": "unsigned_protocol_field; reverse-direction semantics unverified",
            "voltage_v": voltage_v,
            "current_a": current_a,
            "power_w_calculated": power_w,
            "dplus_v": dplus_v,
            "dminus_v": dminus_v,
            "sample_flag_raw_hex": "%02x" % flag_raw,
            "sample_flag_semantics": "unknown_preserved",
            "temperature_c": temperature_c,
            "device_sample_tick": None,
            "estimated_device_sample_time_ns": None,
        })
    return samples


def summarize_report(samples: List[Dict[str, object]]) -> Dict[str, float]:
    if not samples:
        raise ValueError("cannot summarize empty report")
    n = float(len(samples))
    return {
        "voltage_mean_v": sum(float(row["voltage_v"]) for row in samples) / n,
        "current_mean_a": sum(float(row["current_a"]) for row in samples) / n,
        "power_mean_w": sum(float(row["power_w_calculated"]) for row in samples) / n,
    }
