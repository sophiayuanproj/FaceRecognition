"""Raw-first bounded Nano serial capture paired with the Windows HID logger."""
from datetime import datetime, timezone
import json
from pathlib import Path
import time


METER_VID = 0x2E3C
METER_PID = 0x5558
MAX_UNTERMINATED_BYTES = 4096


def describe_port(port):
    hwid = getattr(port, "hwid", "") or ""
    safe_hwid = hwid.split(" SER=")[0]
    return {
        "device": getattr(port, "device", ""),
        "description": getattr(port, "description", "") or "",
        "hwid_without_serial": safe_hwid,
        "vid": getattr(port, "vid", None),
        "pid": getattr(port, "pid", None),
        "manufacturer": getattr(port, "manufacturer", None),
        "product": getattr(port, "product", None),
        "serial_number_present": bool(getattr(port, "serial_number", None)),
    }


class NanoSerialCapture:
    def __init__(self, run_dir, port_name, run_id, root_seed, *,
                 dtr=False, rts=False, port_info=None, serial_module=None,
                 list_ports_module=None, reconnect_interval_s=0.25):
        if root_seed is None or not 0 <= int(root_seed) <= 0xFFFFFFFF:
            raise ValueError("simultaneous Nano logging requires a recorded 32-bit --root-seed")
        self.run_dir = Path(run_dir)
        self.run_id = str(run_id)
        self.root_seed = int(root_seed)
        self.port_name = str(port_name).upper()
        self.dtr_requested = bool(dtr)
        self.rts_requested = bool(rts)
        if serial_module is None:
            import serial as serial_module
            import serial.tools.list_ports as list_ports_module
            matches = [p for p in list_ports_module.comports() if p.device.upper() == self.port_name]
            if len(matches) != 1:
                raise ValueError("Nano COM selection matched %d present ports" % len(matches))
            port_info = matches[0]
        if port_info is None:
            raise ValueError("port_info is required for a test-injected serial module")
        self.serial_module = serial_module
        self.list_ports_module = list_ports_module
        self.reconnect_interval_ns = int(float(reconnect_interval_s) * 1_000_000_000)
        self._last_reconnect_attempt_ns = 0
        self._last_reconnect_wait_record_ns = 0
        self.reconnect_count = 0
        self.connected = False
        self.descriptor = describe_port(port_info)
        if (self.descriptor["vid"], self.descriptor["pid"]) == (METER_VID, METER_PID):
            raise ValueError("refusing meter interface COM port as Nano serial; select a separately identified Nano port")
        if not getattr(port_info, "device", "") or self.descriptor["device"].upper() != self.port_name:
            raise ValueError("fresh COM descriptor does not match the requested port")
        self.path = self.run_dir / "nano_serial.jsonl"
        self.stream = self.path.open("x", encoding="utf-8", newline="\n")
        self._serial_identity_key = self._normalized_identity(getattr(port_info, "serial_number", None))
        self._port_vid_pid = (self.descriptor["vid"], self.descriptor["pid"])
        self.buffer = bytearray()
        self.previous_sequence = None
        self.previous_tick = None
        self.tick_wraps = 0
        self.seed_sent = False
        self.seed_acknowledged = False
        self.start_sent = False
        self.experiment_started = False
        self.experiment_completed = False
        self.closed = False
        self.opened_monotonic_ns = time.monotonic_ns()
        self.serial = None
        try:
            self._open_port(port_info, reconnect=False)
        except BaseException as exc:
            self._write_record("event", {"event": "serial_open_failed",
                                         "error": "%s: %s" % (type(exc).__name__, exc)})
            self.close("failed")
            raise

    @staticmethod
    def _normalized_identity(value):
        if not value:
            return None
        normalized = "".join(ch for ch in str(value).upper() if ch.isalnum())
        return normalized or None

    def _open_port(self, port_info, *, reconnect):
        descriptor = describe_port(port_info)
        if (descriptor["vid"], descriptor["pid"]) == (METER_VID, METER_PID):
            raise ValueError("refusing meter interface COM port during Nano re-enumeration")
        serial_obj = self.serial_module.Serial()
        serial_obj.port = descriptor["device"]
        serial_obj.baudrate = 115200
        serial_obj.timeout = 0
        serial_obj.write_timeout = 1
        serial_obj.dtr = self.dtr_requested
        serial_obj.rts = self.rts_requested
        self._write_record("event", {"event": "serial_reconnect_attempt" if reconnect else "serial_open_attempt",
                                     "port": descriptor,
                                     "baudrate": 115200,
                                     "dtr_requested_before_open": self.dtr_requested,
                                     "rts_requested_before_open": self.rts_requested,
                                     "dtr_may_reset_board": True,
                                     "settling_rule": "record all lines; no sample time inferred"})
        try:
            serial_obj.open()
        except BaseException:
            try:
                serial_obj.close()
            except Exception:
                pass
            raise
        self.serial = serial_obj
        self.port_name = descriptor["device"].upper()
        self.descriptor = descriptor
        self.connected = True
        self.opened_monotonic_ns = time.monotonic_ns()
        if reconnect:
            self.reconnect_count += 1
            # A device reboot resets the sketch sequence and seed state. Re-arm
            # this root on the fresh ready banner or the bounded fallback timer.
            self.seed_sent = False
            self.seed_acknowledged = False
            self.start_sent = False
            self.experiment_started = False
            self.experiment_completed = False
            self.previous_sequence = None
            self.previous_tick = None
            self.tick_wraps = 0
            self._write_record("event", {"event": "serial_reconnected",
                                         "port": self.descriptor,
                                         "reconnect_count": self.reconnect_count,
                                         "dtr_state_after_open": bool(serial_obj.dtr),
                                         "rts_state_after_open": bool(serial_obj.rts)})
        else:
            self._write_record("event", {"event": "serial_open",
                                         "dtr_state_after_open": bool(serial_obj.dtr),
                                         "rts_state_after_open": bool(serial_obj.rts)})

    def _find_reconnected_port(self):
        if self.list_ports_module is None:
            return None
        ports = list(self.list_ports_module.comports())
        ports = [p for p in ports if (getattr(p, "vid", None), getattr(p, "pid", None)) !=
                 (METER_VID, METER_PID)]
        if self._serial_identity_key:
            matches = [p for p in ports if self._normalized_identity(getattr(p, "serial_number", None)) ==
                       self._serial_identity_key]
        else:
            matches = [p for p in ports if (getattr(p, "vid", None), getattr(p, "pid", None)) ==
                       self._port_vid_pid]
        if len(matches) == 1:
            return matches[0]
        return None

    def _try_reconnect(self, now_ns):
        if self.list_ports_module is None or now_ns - self._last_reconnect_attempt_ns < self.reconnect_interval_ns:
            return
        self._last_reconnect_attempt_ns = now_ns
        port_info = self._find_reconnected_port()
        if port_info is None:
            return
        try:
            self._open_port(port_info, reconnect=True)
        except (OSError, PermissionError) as exc:
            self._write_record("event", {"event": "serial_reconnect_open_failed",
                                         "port": describe_port(port_info),
                                         "error": "%s: %s" % (type(exc).__name__, exc)})

    @staticmethod
    def _is_transport_error(exc):
        return isinstance(exc, OSError) or exc.__class__.__name__ == "SerialException"

    def _handle_disconnect(self, exc):
        self._write_record("event", {"event": "serial_disconnect",
                                     "port": self.descriptor,
                                     "error": "%s: %s" % (type(exc).__name__, exc),
                                     "reconnect_policy": "poll fresh serial descriptors every 250 ms; match the same private USB serial identity in memory"})
        if self.buffer:
            self._write_record("rx_partial_line", {"raw_hex": bytes(self.buffer).hex(),
                                                     "length_bytes": len(self.buffer),
                                                     "run_status": "disconnected"})
            self.buffer.clear()
        serial_obj = self.serial
        self.serial = None
        self.connected = False
        self._last_reconnect_attempt_ns = time.monotonic_ns()
        self.seed_sent = False
        self.seed_acknowledged = False
        self.start_sent = False
        self.previous_sequence = None
        self.previous_tick = None
        self.tick_wraps = 0
        if serial_obj is not None:
            try:
                serial_obj.close()
            except Exception:
                pass

    def _write_record(self, record_type, fields):
        row = {
            "record_type": record_type,
            "run_id": self.run_id,
            "root_seed": self.root_seed,
            "host_monotonic_ns": time.monotonic_ns(),
            "host_utc": datetime.now(timezone.utc).isoformat(timespec="microseconds"),
        }
        row.update(fields)
        self.stream.write(json.dumps(row, sort_keys=True, allow_nan=False) + "\n")
        self.stream.flush()

    def _send(self, text, reason):
        data = (text + "\n").encode("ascii")
        written = self.serial.write(data)
        self._write_record("tx", {"command_ascii": text,
                                  "command_hex": data.hex(),
                                  "write_return": int(written),
                                  "reason": reason})
        if written != len(data):
            raise OSError("short Nano serial write: %d/%d" % (written, len(data)))

    def _handle_line(self, raw_line):
        payload = raw_line.rstrip(b"\r\n")
        text = payload.decode("utf-8", "replace")
        fields = text.split(",")
        parsed = {"decoded_line": text}
        kind = fields[0] if fields else ""
        if kind in ("EV", "STAT", "PONG") and len(fields) >= 3:
            try:
                sequence = int(fields[1]) & 0xFFFFFFFF
                tick = int(fields[2]) & 0xFFFFFFFF
                parsed["sequence_number"] = sequence
                parsed["device_millis_raw"] = tick
                if self.previous_sequence is not None:
                    delta = (sequence - self.previous_sequence) & 0xFFFFFFFF
                    if delta != 1 and delta < 0x80000000:
                        self._write_record("sequence_gap", {
                            "previous_sequence": self.previous_sequence,
                            "current_sequence": sequence,
                            "missing_sequence_count": delta - 1,
                        })
                if self.previous_tick is not None and self.previous_tick > 0xF0000000 and tick < 0x0FFFFFFF:
                    self.tick_wraps += 1
                parsed["device_millis_wrap_count"] = self.tick_wraps
                parsed["device_millis_unwrapped"] = self.tick_wraps * (1 << 32) + tick
                self.previous_sequence = sequence
                self.previous_tick = tick
            except (ValueError, TypeError):
                parsed["parse_status"] = "bad_sequence_or_millis"
        self._write_record("rx_line", {"raw_hex": payload.hex(), **parsed})

        if kind == "EV" and len(fields) >= 4:
            if fields[3] == "experiment_start":
                self.experiment_started = True
            elif fields[3] == "experiment_end":
                self.experiment_completed = True

        if kind == "EV" and len(fields) >= 4 and fields[3] == "ready" and not self.seed_sent:
            self._send("SEED %d" % self.root_seed, "ready event observed")
            self.seed_sent = True
            self._send("PING win-%s" % self.run_id, "serial echo check")
        elif kind == "EV" and len(fields) >= 5 and fields[3] == "seed_set":
            try:
                accepted_seed = int(fields[4])
            except ValueError:
                accepted_seed = None
            if accepted_seed == self.root_seed and not self.start_sent:
                self.seed_acknowledged = True
                self._send("START", "matching seed acknowledgement observed")
                self.start_sent = True

    def poll(self):
        if self.closed:
            return
        if self.serial is None or not getattr(self.serial, "is_open", False):
            self.connected = False
            self._try_reconnect(time.monotonic_ns())
            if self.serial is None or not getattr(self.serial, "is_open", False):
                return
        try:
            waiting = int(self.serial.in_waiting)
            if waiting > 0:
                self.buffer.extend(self.serial.read(min(waiting, 4096)))
            while True:
                newline = self.buffer.find(b"\n")
                if newline < 0:
                    break
                raw_line = bytes(self.buffer[:newline + 1])
                del self.buffer[:newline + 1]
                self._handle_line(raw_line)
            if not self.seed_sent and time.monotonic_ns() - self.opened_monotonic_ns >= 1_000_000_000:
                self._send("SEED %d" % self.root_seed, "ready banner not observed after 1 s")
                self.seed_sent = True
                self._send("PING win-%s" % self.run_id, "serial echo check after ready-banner timeout")
        except Exception as exc:
            if not self._is_transport_error(exc):
                raise
            self._handle_disconnect(exc)
        if len(self.buffer) >= MAX_UNTERMINATED_BYTES:
            chunk = bytes(self.buffer)
            self.buffer.clear()
            self._write_record("rx_unterminated_chunk", {
                "raw_hex": chunk.hex(),
                "length_bytes": len(chunk),
                "reason": "host_line_buffer_cap",
            })

    def close(self, status):
        if self.closed:
            return
        try:
            if self.serial is not None and getattr(self.serial, "is_open", False):
                self.poll()
            if self.serial is not None and getattr(self.serial, "is_open", False):
                try:
                    self.serial.close()
                except Exception as exc:
                    self._write_record("event", {"event": "serial_close_error",
                                                 "error": "%s: %s" % (type(exc).__name__, exc)})
            if self.buffer:
                self._write_record("rx_partial_line", {"raw_hex": bytes(self.buffer).hex(),
                                                        "length_bytes": len(self.buffer),
                                                        "run_status": status})
                self.buffer.clear()
            self._write_record("event", {"event": "serial_close", "run_status": status,
                                         "seed_command_sent": self.seed_sent,
                                         "seed_acknowledged": self.seed_acknowledged,
                                         "start_command_sent": self.start_sent,
                                         "experiment_started": self.experiment_started,
                                         "experiment_completed": self.experiment_completed,
                                         "connected_at_close": self.connected,
                                         "reconnect_count": self.reconnect_count,
                                         "last_sequence_number": self.previous_sequence,
                                         "device_millis_wrap_count": self.tick_wraps})
        finally:
            self.stream.close()
            self.closed = True
