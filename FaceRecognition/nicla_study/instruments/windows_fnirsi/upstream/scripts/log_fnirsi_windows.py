"""Bounded native-Windows FNIRSI HID capture with fresh interface selection.

Raw payloads are stored before interpretation by the inherited parser. This is a
Nano diagnostic reader; valid model-energy claims require separate evidence.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import re
import sys
import time

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DEFAULT = SCRIPT_DIR.parents[1]
POWER_SCRIPTS = PROJECT_DEFAULT / "work" / "research-next" / "power" / "scripts"
sys.path.insert(0, str(POWER_SCRIPTS))
sys.path.insert(0, str(SCRIPT_DIR))

from device_selector_windows import choose_device, describe
from hardware_lock_windows import HardwareLock, default_lock_path
from fnirsi_protocol import PID, VID, make_command
from run_io import RunWriter
from nano_serial_capture_windows import NanoSerialCapture


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def build_parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--project-root", type=Path, default=PROJECT_DEFAULT)
    p.add_argument("--list", action="store_true", help="list this meter's fresh HID interfaces; no device is opened")
    p.add_argument("--interface-number", type=int,
                   help="required explicit interface selection for reads (current FNB-C2 candidate: 2)")
    p.add_argument("--path-fragment", help="optional private path discriminator if an interface number is duplicated")
    p.add_argument("--duration", type=float, default=30.0, help="capture seconds (0 < duration <= 600)")
    p.add_argument("--output-dir", type=Path, help="new immutable run directory; default project raw/run-id")
    p.add_argument("--run-id", default="windows-%s" % datetime.now().strftime("%Y%m%dT%H%M%S"))
    p.add_argument("--experiment-id", default="windows-fnirsi-nano-diagnostic")
    p.add_argument("--scenario-id", default="meter-in-path-pending-validation")
    p.add_argument("--root-seed", type=int)
    p.add_argument("--meter-label", default="FNB-C2 as observed in official app; HID descriptor string FNB-58")
    p.add_argument("--meter-firmware", default="not yet read from device")
    p.add_argument("--topology", default="not verified; record physical source and port path")
    p.add_argument("--load-description", default="Nano inline; exact board/port and measured source path pending")
    p.add_argument("--serial-port", help="freshly identified Nano COM port; meter's own COM interface is rejected")
    p.add_argument("--serial-dtr", action="store_true", help="assert CDC DTR on the Nano COM port; may reset the board")
    p.add_argument("--serial-rts", action="store_true", help="assert CDC RTS on the Nano COM port; may reset the board")
    p.add_argument("--lock-timeout", type=float, default=0.0)
    return p


def main(argv=None):
    args = build_parser().parse_args(argv)
    if not re.fullmatch(r"[A-Za-z0-9._-]{1,100}", args.run_id):
        build_parser().error("--run-id permits letters, numbers, dot, underscore, and hyphen only")
    if not args.list and (args.interface_number is None or not 0 < args.duration <= 600):
        build_parser().error("reads need --interface-number and --duration in (0, 600]")
    if args.serial_port and args.root_seed is None:
        build_parser().error("--serial-port requires --root-seed for the recorded Nano workload")
    if not args.serial_port and (args.serial_dtr or args.serial_rts):
        build_parser().error("--serial-dtr/--serial-rts require --serial-port")
    if args.root_seed is not None and not 0 <= args.root_seed <= 0xFFFFFFFF:
        build_parser().error("--root-seed must be an unsigned 32-bit integer")
    if args.lock_timeout < 0:
        build_parser().error("--lock-timeout must be nonnegative")
    root = args.project_root.resolve()
    power_scripts = root / "work" / "research-next" / "power" / "scripts"
    if not (power_scripts / "fnirsi_protocol.py").is_file():
        print("project root does not contain the inherited FNIRSI parser: %s" % power_scripts, file=sys.stderr)
        return 2

    try:
        import hid
    except ImportError as exc:
        print("import hid failed; activate the local venv and install requirements-windows.txt: %s" % exc,
              file=sys.stderr)
        return 2

    lock_path = default_lock_path(root)
    try:
        with HardwareLock(lock_path, args.run_id, args.lock_timeout):
            devices = hid.enumerate(VID, PID)
            descriptors = [describe(d, include_path=args.list) for d in devices]
            if args.list:
                print(json.dumps({"descriptor_only": True, "enumerated_utc": datetime.now(timezone.utc).isoformat(),
                                  "devices": descriptors}, indent=2))
                return 0
            try:
                selected = choose_device(devices, args.interface_number, args.path_fragment)
            except ValueError as exc:
                print("%s\nFresh devices: %s" % (exc, json.dumps(descriptors, indent=2)), file=sys.stderr)
                return 3

            selected_summary = describe(selected)
            run_id = args.run_id
            out_dir = args.output_dir or (root / "raw" / run_id)
            source_paths = {
                "windows_logger_sha256": Path(__file__).resolve(),
                "windows_lock_sha256": SCRIPT_DIR / "hardware_lock_windows.py",
                "device_selector_sha256": SCRIPT_DIR / "device_selector_windows.py",
                "nano_serial_capture_sha256": SCRIPT_DIR / "nano_serial_capture_windows.py",
                "protocol_sha256": power_scripts / "fnirsi_protocol.py",
                "run_writer_sha256": power_scripts / "run_io.py",
                "nano_sketch_sha256": root / "firmware" / "nano_meter_diagnostic" / "nano_meter_diagnostic.ino",
            }
            artifact_hashes = {key: sha256(path) for key, path in source_paths.items() if path.is_file()}
            metadata = {
                "experiment_id": args.experiment_id,
                "run_id": run_id,
                "policy_id": "diagnostic-only",
                "scenario_id": args.scenario_id,
                "root_seed": args.root_seed,
                "meter_vid_pid": "%04x:%04x" % (VID, PID),
                "meter_model_reported": args.meter_label,
                "meter_firmware": args.meter_firmware,
                "meter_hid_descriptor": selected_summary,
                "fresh_interface_selection": True,
                "topology": args.topology,
                "load_description": args.load_description,
                "source_input_paths_sha256": artifact_hashes,
                "host": {"platform": platform.platform(), "python": sys.version,
                         "python_bits": 64 if sys.maxsize > 2**32 else 32,
                         "hidapi_version": importlib.metadata.version("hidapi"),
                         "pyserial_version": importlib.metadata.version("pyserial")},
                "duration_requested_s": args.duration,
                "read_size_bytes": 64,
                "hid_write_report_id_prefix": 0,
                "measurement_claim": "diagnostic only until powered path, simultaneous Nano COM, display agreement, and timing pass",
                "acquisition_time_source": "host monotonic at HID read completion; no device timestamps assigned",
            }
            writer = None
            nano_serial = None
            device = hid.device()
            try:
                writer = RunWriter(out_dir, metadata)
                writer.event("meter_open_attempt", {"interface_number": selected_summary["interface_number"],
                                                     "usage_page": selected_summary["usage_page"],
                                                     "usage": selected_summary["usage"],
                                                     "path_sha256": selected_summary["enumerated_path_sha256"]})
                device.open_path(selected["path"])
                writer.event("meter_open", {"selected_interface": selected_summary,
                                            "protocol_commands": ["0x81 init", "0x82 read"],
                                            "protocol_source": "inherited commands; HID report conventions recorded during live capture"})
                if args.serial_port:
                    writer.event("nano_serial_open_attempt", {"port_requested": args.serial_port,
                                                               "baudrate": 115200,
                                                               "root_seed": args.root_seed})
                    nano_serial = NanoSerialCapture(out_dir, args.serial_port, run_id, args.root_seed,
                                                    dtr=args.serial_dtr, rts=args.serial_rts)
                    writer.metadata["nano_serial_descriptor"] = nano_serial.descriptor
                    writer.metadata["nano_serial_log_file"] = "nano_serial.jsonl"
                    writer.metadata["nano_serial_baudrate"] = 115200
                    writer.metadata["nano_serial_dtr_requested"] = bool(args.serial_dtr)
                    writer.metadata["nano_serial_rts_requested"] = bool(args.serial_rts)
                    writer.metadata["nano_serial_dtr_may_reset_board"] = True
                    writer.write_metadata()
                    writer.event("nano_serial_open", {"port": nano_serial.descriptor,
                                                       "log_file": "nano_serial.jsonl",
                                                       "dtr_requested": bool(args.serial_dtr),
                                                       "rts_requested": bool(args.serial_rts),
                                                       "dtr_rts_may_reset_board": True})
                init_packet = bytes(make_command(0x81))
                init_written = device.write([0] + list(init_packet))
                writer.event("hid_command", {"command": "0x81", "payload_len": len(init_packet),
                                             "hidapi_write_return": int(init_written),
                                             "report_id_prefix": 0})
                time.sleep(0.05)
                start = time.monotonic()
                deadline = start + args.duration
                prior_ns = None
                arrival_index = 0
                read_lengths = {}
                while time.monotonic() < deadline:
                    read_command = bytes(make_command(0x82))
                    written = device.write([0] + list(read_command))
                    if written not in (64, 65):
                        writer.event("hid_write_unexpected_length",
                                     {"command": "0x82", "hidapi_write_return": int(written),
                                      "expected_64_or_65": True})
                    payload = device.read(64, timeout_ms=250)
                    receive_ns = time.monotonic_ns()
                    receive_utc = datetime.now(timezone.utc).isoformat(timespec="microseconds")
                    if not payload:
                        writer.record_timeout(arrival_index)
                        if nano_serial:
                            nano_serial.poll()
                        continue
                    payload = bytes(payload)
                    read_lengths[str(len(payload))] = read_lengths.get(str(len(payload)), 0) + 1
                    gap = prior_ns is not None and receive_ns - prior_ns > 200_000_000
                    writer.record_report(payload, arrival_index, receive_ns, receive_utc, gap)
                    if len(payload) >= 2:
                        writer.event("hid_payload_observed", {"report_arrival_index": arrival_index,
                                                               "payload_length": len(payload),
                                                               "prefix_hex": payload[:4].hex(),
                                                               "write_return": int(written)})
                    prior_ns = receive_ns
                    arrival_index += 1
                    if nano_serial:
                        nano_serial.poll()
                if nano_serial:
                    nano_serial.close("complete" if read_lengths else "failed")
                    writer.metadata["nano_serial_status"] = (
                        "captured_in_same_locked_session_connected_at_end"
                        if nano_serial.connected else "captured_with_disconnect_no_reconnect_at_end")
                    writer.metadata["nano_serial_reconnect_count"] = nano_serial.reconnect_count
                    writer.metadata["nano_serial_experiment_started"] = nano_serial.experiment_started
                    writer.metadata["nano_serial_experiment_completed"] = nano_serial.experiment_completed
                writer.metadata["hid_read_length_counts"] = read_lengths
                writer.metadata["hid_read_convention"] = "raw HIDAPI payload preserved without stripping or adding a report-ID byte"
                writer.close("complete" if read_lengths else "failed",
                             None if read_lengths else "no HID payloads received during bounded session")
                status = "complete" if read_lengths else "failed"
                print(json.dumps({"output_dir": str(out_dir), "status": status,
                                  "reports": writer.report_count, "invalid_reports": writer.invalid_count,
                                  "timeouts": writer.timeout_count, "read_length_counts": read_lengths,
                                  "nano_serial_log_file": "nano_serial.jsonl" if nano_serial else None,
                                  "nano_serial_seed_acknowledged": bool(nano_serial and nano_serial.seed_acknowledged),
                                  "nano_serial_start_sent": bool(nano_serial and nano_serial.start_sent),
                                  "nano_serial_connected_at_end": bool(nano_serial and nano_serial.connected),
                                  "nano_serial_reconnect_count": int(nano_serial.reconnect_count) if nano_serial else 0,
                                  "nano_serial_experiment_started": bool(nano_serial and nano_serial.experiment_started),
                                  "nano_serial_experiment_completed": bool(nano_serial and nano_serial.experiment_completed),
                                  "measurement_claim": "diagnostic only; energy not qualified"}, indent=2))
                return 0 if writer.report_count > 0 else 7
            except KeyboardInterrupt:
                if nano_serial:
                    nano_serial.close("interrupted")
                if writer:
                    writer.close("interrupted", "KeyboardInterrupt; immutable raw partial capture retained")
                print("Interrupted; raw partial capture retained, run is not complete.", file=sys.stderr)
                return 130
            except Exception as exc:
                if nano_serial:
                    nano_serial.close("failed")
                if writer:
                    writer.event("logger_error", {"error": "%s: %s" % (type(exc).__name__, exc)})
                    writer.close("failed", "%s: %s" % (type(exc).__name__, exc))
                print("logger failed: %s: %s" % (type(exc).__name__, exc), file=sys.stderr)
                return 5
            finally:
                try:
                    device.close()
                except Exception:
                    pass
    except TimeoutError as exc:
        print(str(exc), file=sys.stderr)
        return 6


if __name__ == "__main__":
    raise SystemExit(main())
