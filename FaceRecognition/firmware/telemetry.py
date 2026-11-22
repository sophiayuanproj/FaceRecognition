"""Bounded JSONL telemetry, framed diagnostic packets and control input."""

import binascii
import gc
import hashlib
import json
import time
from runtime_support import BoundedLineQueue


_PACKET_MAGIC = b"NVPK1"
_PACKET_HEADER_MAX = 1024
_PACKET_PAYLOAD_MAX = 65536


def _encode_line(record):
    return json.dumps(record, separators=(",", ":")).encode("utf-8") + b"\n"


def _json_safe(value, depth=0):
    """Replace non-finite numbers before JSON so one NaN cannot poison a stream."""
    if depth > 10:
        return None
    if value is None or isinstance(value, (bool, int, str)):
        return value
    if isinstance(value, float):
        if value != value or value == float("inf") or value == -float("inf"):
            return None
        return value
    if isinstance(value, dict):
        return {str(key): _json_safe(item, depth + 1) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item, depth + 1) for item in value]
    if isinstance(value, (bytes, bytearray)):
        return {"_type": "bytes", "byte_count": len(value)}
    # OpenMV exposes a few numeric scalar wrappers which are float-convertible
    # but are not accepted directly by MicroPython's JSON encoder. Normalize
    # those values before encoding; retain only a bounded type label for any
    # unsupported diagnostic object instead of losing the whole event.
    try:
        converted = float(value)
        if converted != converted or converted == float("inf") or converted == -float("inf"):
            return None
        return converted
    except Exception:
        try:
            return "<%s>" % type(value).__name__
        except Exception:
            return "<unsupported>"


class Telemetry:
    def __init__(self, usb, uart, config, ticks_ms, run_id, reset_cause,
                 firmware_sha256, config_sha256, schema_sha256, ticks_modulus_ms=None):
        self.usb = usb
        self.uart = uart
        self.config = config
        self.ticks_ms = ticks_ms
        self.run_id = run_id
        self.reset_cause = reset_cause
        self.firmware_sha256 = firmware_sha256
        self.config_sha256 = config_sha256
        self.schema_sha256 = schema_sha256
        self.ticks_modulus_ms = ticks_modulus_ms
        max_bytes = int(config["telemetry"]["max_queue_bytes_per_sink"])
        max_record = int(config["telemetry"]["max_record_bytes"])
        self.usb_queue = BoundedLineQueue(max_bytes, max_record)
        self.uart_queue = BoundedLineQueue(max_bytes, max_record)
        self.sequence = 0
        self.usb_failures = 0
        self.uart_failures = 0

    def emit(self, event, fields=None):
        record = {"schema_version": 1, "schema_id": "nicla-face-telemetry-1",
                  "event": str(event), "run_id": self.run_id,
                  "reset_cause": self.reset_cause,
                  "sequence": self.sequence, "ticks_ms": int(self.ticks_ms()),
                  "ticks_unit": "ms", "ticks_modulus_ms": self.ticks_modulus_ms,
                  "model_config_sha256": self.config_sha256,
                  "firmware_sha256": self.firmware_sha256,
                  "schema_sha256": self.schema_sha256}
        self.sequence += 1
        if fields:
            for key, value in fields.items():
                if isinstance(key, str) and len(key) <= 40:
                    # The runtime's observation summary is constructed from
                    # JSON-native values. Reusing its nested objects avoids a
                    # second full recursive copy at the lowest-heap point of
                    # the camera cycle. Other event inputs still pass through
                    # the defensive normalizer.
                    record[key] = value if event == "observation_complete" else _json_safe(value)
        record["boot_id"] = self.run_id
        frame_id = record.get("frame_id", record.get("frame_index"))
        if isinstance(frame_id, int) and not isinstance(frame_id, bool):
            record["frame_id"] = frame_id
        if event == "observation_complete":
            record["schema_version"] = 2
            record["schema_id"] = "nicla-observation-v2"
        encode_error_type = None
        try:
            encoded = _encode_line(record)
        except Exception as error:
            encode_error_type = type(error).__name__
            encoded = None
        if encoded is None and event == "observation_complete":
            # Keep the frame-keyed execution result available when the rich
            # observation runs into a constrained heap. The exact payloads
            # have already been sent separately; remove explanatory-only
            # fields first, then preserve the complete same-frame stages.
            for key in ("trace", "qualification", "imu", "requested_state"):
                record.pop(key, None)
            record["record_truncated"] = True
            record["record_truncated_reason"] = "observation_memory_compaction"
            gc.collect()
            try:
                encoded = _encode_line(record)
                encode_error_type = None
            except Exception as error:
                encode_error_type = type(error).__name__
                encoded = None
        if encoded is None and event == "observation_complete":
            # Last-resort observation retains the exact frame, detector and
            # identity linkage, final decision, and timing while dropping
            # optional nested telemetry. It is still an honest, inspectable
            # observation rather than an unjoinable encode-error event.
            keep = ("schema_version", "schema_id", "event", "run_id", "reset_cause",
                    "sequence", "ticks_ms", "ticks_unit", "ticks_modulus_ms",
                    "model_config_sha256", "firmware_sha256", "schema_sha256", "boot_id",
                    "type", "source", "frame_id", "command_id", "frame", "camera",
                    "detection", "detector", "detector_input", "crop", "identity",
                    "identity_input", "decision", "timings_ms", "applied_state",
                    "model_sha256", "config_sha256", "failure_reason",
                    "record_truncated", "record_truncated_reason")
            compact = {key: record[key] for key in keep if key in record}
            gc.collect()
            try:
                encoded = _encode_line(compact)
                record = compact
                encode_error_type = None
            except Exception as error:
                encode_error_type = type(error).__name__
                encoded = None
        if encoded is None:
            record = {"schema_version": 1, "schema_id": "nicla-face-telemetry-1",
                      "event": "telemetry_encode_error", "run_id": self.run_id,
                      "sequence": self.sequence - 1, "ticks_ms": int(self.ticks_ms()),
                      "ticks_unit": "ms", "ticks_modulus_ms": self.ticks_modulus_ms,
                      "reset_cause": self.reset_cause,
                      "boot_id": self.run_id,
                      "model_config_sha256": self.config_sha256,
                      "firmware_sha256": self.firmware_sha256,
                      "schema_sha256": self.schema_sha256,
                      "failed_event": str(event)[:32],
                      "error_type": encode_error_type or "RuntimeError"}
            if isinstance(frame_id, int) and not isinstance(frame_id, bool):
                record["frame_id"] = frame_id
            encoded = _encode_line(record)
        maximum = int(self.config["telemetry"]["max_record_bytes"])
        if len(encoded) > maximum:
            # Keep control, lifecycle and decision facts first; discard only
            # optional diagnostics if an unusually large exception string lands.
            for key in ("decision", "face_records", "detector_labels", "acceleration_mg",
                        "angular_rate_mdps", "metadata", "class_probabilities", "message"):
                record.pop(key, None)
                encoded = _encode_line(record)
                if len(encoded) <= maximum:
                    break
        if len(encoded) > maximum:
            compact = {"schema_version": 1, "schema_id": "nicla-face-telemetry-1",
                "event": str(event)[:32], "run_id": self.run_id,
                "reset_cause": self.reset_cause,
                "sequence": record.get("sequence"), "ticks_ms": record.get("ticks_ms"),
                "ticks_unit": "ms", "ticks_modulus_ms": self.ticks_modulus_ms,
                "model_config_sha256": self.config_sha256,
                "firmware_sha256": self.firmware_sha256,
                "schema_sha256": self.schema_sha256, "record_truncated": True}
            if event == "control_ack" and fields:
                for key in ("source", "command", "control_status", "reason_code"):
                    if key in fields:
                        compact[key] = fields[key]
            elif event == "decision" and fields:
                for key in ("frame_index", "policy_id", "status", "identity_id",
                            "candidate_identity_id", "decision_reason", "backend",
                            "failure_reason"):
                    if key in fields:
                        compact[key] = fields[key]
            record = compact
            encoded = _encode_line(record)
        # Reserve queue capacity for ACKs without moving them ahead of records
        # that already have lower sequence numbers.
        queue_push = "push_priority" if event == "control_ack" else "push"
        # UART has its own queue and is populated regardless of native USB state.
        if self.uart is not None:
            getattr(self.uart_queue, queue_push)(encoded)
        try:
            connected = self.usb is not None and self.usb.isconnected()
        except Exception:
            connected = False
        if connected:
            getattr(self.usb_queue, queue_push)(encoded)
        return record

    def flush(self):
        telemetry = self.config["telemetry"]
        legacy_maximum = int(telemetry["flush_bytes_per_loop"])
        uart_maximum = int(telemetry.get("uart_flush_bytes_per_loop", legacy_maximum))
        usb_maximum = int(telemetry.get("usb_flush_bytes_per_loop", legacy_maximum))
        if self.uart is not None and self.uart_queue.queued_bytes:
            data = self.uart_queue.take(uart_maximum)
            try:
                written = self.uart.write(data)
                if written is not None and int(written) < len(data):
                    remainder = data[max(0, int(written)):]
                    if remainder:
                        self.uart_queue.push_front(remainder)
            except Exception:
                self.uart_failures += 1
        if self.usb is not None and self.usb_queue.queued_bytes:
            try:
                if self.usb.isconnected():
                    data = self.usb_queue.take(usb_maximum)
                    written = self.usb.write(data)
                    if written is not None and int(written) < len(data):
                        remainder = data[max(0, int(written)):]
                        if remainder:
                            self.usb_queue.push_front(remainder)
            except Exception:
                self.usb_failures += 1
        return {"uart_queue_bytes": self.uart_queue.queued_bytes,
                "usb_queue_bytes": self.usb_queue.queued_bytes,
                "uart_failures": self.uart_failures, "usb_failures": self.usb_failures}

    def usb_connected(self):
        try:
            return self.usb is not None and bool(self.usb.isconnected())
        except Exception:
            return False

    def _write_usb_all(self, payload, timeout_ms=2000):
        started = time.ticks_ms()
        offset = 0
        stalls = 0
        while offset < len(payload):
            stop = min(len(payload), offset + 512)
            chunk = payload[offset:stop]
            try:
                written = self.usb.write(chunk)
                written = len(chunk) if written is None else int(written)
            except Exception:
                return False, offset, "usb_write_error"
            if written < 0 or written > len(chunk):
                return False, offset, "usb_write_count_invalid"
            if written:
                offset += written
                stalls = 0
            else:
                stalls += 1
                if stalls > 400 or time.ticks_diff(time.ticks_ms(), started) >= timeout_ms:
                    return False, offset, "usb_write_timeout"
                time.sleep_ms(5)
        return True, offset, None

    def send_packet(self, kind, metadata, payload):
        """Send one bounded NVPK1 binary packet on USB CDC, never UART.

        JSON telemetry is drained first so packet framing cannot splice into a
        JSON line. A packet carries its own versioned header, byte count and
        SHA-256; the host rejects incomplete, oversized or mismatched payloads.
        """
        if not self.usb_connected():
            return {"sent": False, "reason": "usb_disconnected"}
        try:
            # Keep the original buffer (or a zero-copy view) so a model input
            # can be streamed from its callback before the model reuses it.
            # A bytes(payload) conversion here can require another 9 KiB on
            # the Nicla while the interpreter is live.
            if not isinstance(payload, (bytes, bytearray, memoryview)):
                payload = memoryview(payload)
        except Exception:
            return {"sent": False, "reason": "payload_not_buffer"}
        if len(payload) <= 0 or len(payload) > _PACKET_PAYLOAD_MAX:
            return {"sent": False, "reason": "payload_size_out_of_range",
                    "byte_count": len(payload)}
        digest = hashlib.sha256()
        digest.update(payload)
        checksum = binascii.hexlify(digest.digest()).decode("ascii")
        header = {"schema_id": "nicla-packet-v1", "packet_version": 1,
                  "kind": str(kind), "byte_count": len(payload),
                  "sha256": checksum}
        if isinstance(metadata, dict):
            for key, value in metadata.items():
                if isinstance(key, str) and len(key) <= 48:
                    header[key] = value
        try:
            encoded_header = json.dumps(header, separators=(",", ":")).encode("utf-8")
        except Exception:
            return {"sent": False, "reason": "packet_header_encode_error"}
        if len(encoded_header) <= 0 or len(encoded_header) > _PACKET_HEADER_MAX:
            return {"sent": False, "reason": "packet_header_size_out_of_range"}

        # Keep line records intact and ordered before the next binary packet.
        drain_started = time.ticks_ms()
        prior_bytes = -1
        while self.usb_queue.queued_bytes:
            self.flush()
            if self.usb_queue.queued_bytes == prior_bytes:
                time.sleep_ms(5)
            prior_bytes = self.usb_queue.queued_bytes
            if time.ticks_diff(time.ticks_ms(), drain_started) >= 2500:
                return {"sent": False, "reason": "telemetry_drain_timeout",
                        "queued_bytes": self.usb_queue.queued_bytes}
        prefix = _PACKET_MAGIC + bytes((len(encoded_header) & 255,
                                        (len(encoded_header) >> 8) & 255)) + encoded_header
        ok, header_bytes, reason = self._write_usb_all(prefix)
        if not ok:
            return {"sent": False, "reason": reason,
                    "header_bytes_written": header_bytes, "sha256": checksum}
        ok, payload_bytes, reason = self._write_usb_all(payload, timeout_ms=3000)
        if not ok:
            return {"sent": False, "reason": reason,
                    "header_bytes_written": header_bytes,
                    "payload_bytes_written": payload_bytes, "sha256": checksum}
        return {"sent": True, "byte_count": len(payload), "sha256": checksum,
                "packet_bytes": len(prefix) + len(payload)}


class LineInput:
    """Two bounded independent command sources, each accepting UTF-8 JSONL."""

    def __init__(self, usb, uart, max_line_bytes=192, max_read_bytes=64):
        self.sources = [("usb", usb), ("uart", uart)]
        self.max_line_bytes = int(max_line_bytes)
        self.max_read_bytes = int(max_read_bytes)
        self.buffers = {"usb": bytearray(), "uart": bytearray()}
        self.errors = {"usb": 0, "uart": 0}
        self.bytes_read_total = {"usb": 0, "uart": 0}

    def poll(self):
        commands = []
        for name, source in self.sources:
            if source is None:
                continue
            if name == "usb":
                try:
                    if not source.isconnected():
                        continue
                except Exception:
                    continue
            try:
                data = source.read(self.max_read_bytes)
            except Exception:
                self.errors[name] += 1
                continue
            if not data:
                continue
            self.bytes_read_total[name] += len(data)
            buffer = self.buffers[name]
            for byte in data:
                if byte in (10, 13):
                    if buffer:
                        try:
                            value = json.loads(bytes(buffer).decode("utf-8"))
                            if isinstance(value, dict):
                                commands.append((name, value))
                            else:
                                self.errors[name] += 1
                        except Exception:
                            self.errors[name] += 1
                        buffer[:] = b""
                elif len(buffer) < self.max_line_bytes:
                    buffer.append(byte)
                else:
                    buffer[:] = b""
                    self.errors[name] += 1
        return commands
