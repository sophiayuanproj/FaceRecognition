#!/usr/bin/env python3
"""Run an explicitly selected OpenMV script and record a bounded local session.

Image capture is opt-in. The native OpenMV JPEG preview is decoded on the host
as RGB888 and saved as PNG; it is not a raw sensor dump or hardware validation.
"""

from __future__ import annotations

import argparse
import contextlib
import csv
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
from pathlib import Path
import re
import signal
import sys
import time


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MANIFEST_FIELDS = [
    "session_id", "person_id", "identity_role", "capture_camera", "lighting",
    "split", "source_file", "host_received_utc", "frame_index", "width",
    "height", "transport", "source_pixel_format_id", "wire_payload_bytes",
    "stored_encoding", "source_config", "notes",
]


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, default=str) + "\n", encoding="utf-8")


def positive_float(value: str) -> float:
    number = float(value)
    if not 0 < number < float("inf"):
        raise argparse.ArgumentTypeError("must be a finite positive number")
    return number


def positive_int(value: str) -> int:
    number = int(value)
    if number <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return number


class RecordingDeadline(BaseException):
    """Interrupt a synchronous SDK operation when the session duration ends."""


@contextlib.contextmanager
def duration_deadline(seconds: float):
    """Bound blocking OpenMV SDK calls on the project's POSIX host."""
    if not hasattr(signal, "setitimer"):
        yield
        return

    previous_handler = signal.getsignal(signal.SIGALRM)

    def expire(_signum: int, _frame: object) -> None:
        raise RecordingDeadline()

    signal.signal(signal.SIGALRM, expire)
    # Repeat during cleanup too, so a blocked best-effort remote stop cannot
    # keep the serial port open indefinitely after the deadline.
    signal.setitimer(signal.ITIMER_REAL, seconds, 0.5)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous_handler)


def identifier(value: str) -> str:
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", value):
        raise argparse.ArgumentTypeError("use letters, digits, dots, underscores, or hyphens")
    return value


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--list-ports", action="store_true", help="list ports without connecting")
    parser.add_argument("--port", help="explicit device path, e.g. /dev/cu.usbmodemXXXX")
    parser.add_argument("--script", type=Path, help="MicroPython script to run in RAM on the camera")
    parser.add_argument("--session", type=identifier, help="capture-session identifier")
    parser.add_argument("--person-id", type=identifier, help="declared subject ID; use none for no-face scenes")
    parser.add_argument("--identity-role", choices=["known", "other", "unknown", "no_face", "unlabeled"], default="unlabeled")
    parser.add_argument("--source-camera", default="arduino_nicla_vision", help="declared capture camera, not a model prediction")
    parser.add_argument("--lighting", default="unspecified", help="declared lighting condition")
    parser.add_argument("--split", choices=["training", "validation", "test", "unassigned"], default="unassigned")
    parser.add_argument("--source-config", type=Path, help="JSON metadata describing the camera script; does not change camera settings")
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "data" / "acquisition")
    parser.add_argument("--duration", type=positive_float, default=60.0, help="recording duration in seconds (default: 60)")
    parser.add_argument("--capture", action="store_true", help="explicitly enable local image collection from the JPEG preview stream")
    parser.add_argument("--frame-interval", type=positive_float, default=1.0, help="minimum host save interval in seconds (default: 1)")
    parser.add_argument("--max-frames", type=positive_int, default=60, help="stop after this many saved images when --capture is enabled")
    parser.add_argument("--quiet", action="store_true", help="keep camera stdout in files without echoing it")
    args = parser.parse_args()
    if args.list_ports:
        return args
    if not args.port or not args.session:
        parser.error("--port and --session are required; no camera is selected automatically")
    if args.capture and not args.person_id:
        parser.error("--capture requires --person-id so images are not silently mislabeled")
    if args.script and not args.script.is_file():
        parser.error("--script must name an existing MicroPython file")
    if args.source_config and not args.source_config.is_file():
        parser.error("--source-config must name an existing JSON file")
    # The project's docs directory is the public GitHub Pages source.
    output = args.output.resolve()
    if output == PROJECT_ROOT / "docs" or PROJECT_ROOT / "docs" in output.parents:
        parser.error("capture output cannot be inside the public docs/ website directory")
    args.output = output
    return args


def list_ports() -> int:
    try:
        from serial.tools import list_ports as serial_ports
    except ImportError:
        print("Install the project host dependencies first (openmv includes pyserial).", file=sys.stderr)
        return 2
    ports = list(serial_ports.comports())
    if not ports:
        print("No serial ports found. The Nicla need not be connected to prepare the project.")
    for port in ports:
        print(f"{port.device}\t{port.description}\tVID={port.vid!r}\tPID={port.pid!r}")
    return 0


def record(args: argparse.Namespace) -> int:
    try:
        from openmv import Camera
    except ImportError:
        print("Missing openmv. Use the project environment with openmv==1.0.7.", file=sys.stderr)
        return 2
    image_module = None
    if args.capture:
        try:
            from PIL import Image
            image_module = Image
        except ImportError:
            print("Image capture requires Pillow. Logging without --capture does not.", file=sys.stderr)
            return 2

    source_config = {}
    if args.source_config:
        source_config = json.loads(args.source_config.read_text(encoding="utf-8"))
        if not isinstance(source_config, dict):
            raise ValueError("source-config must contain a JSON object")
    script = args.script.read_text(encoding="utf-8") if args.script else None
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
    output = args.output / args.session / run_id
    output.mkdir(parents=True, exist_ok=False)
    if args.capture:
        (output / "images").mkdir()
    if script is not None:
        (output / "executed_script.py").write_text(script, encoding="utf-8")
    if args.source_config:
        write_json(output / "source_config.json", source_config)

    write_json(output / "session.json", {
        "session_id": args.session,
        "run_id": run_id,
        "started_utc": utc_now(),
        "port": args.port,
        "openmv_package_version": importlib.metadata.version("openmv"),
        "person_id": args.person_id,
        "identity_role": args.identity_role,
        "capture_camera": args.source_camera,
        "lighting": args.lighting,
        "split": args.split,
        "script_file": str(args.script.resolve()) if args.script else None,
        "script_sha256": hashlib.sha256(script.encode("utf-8")).hexdigest() if script else None,
        "declared_source_config": source_config,
        "capture_enabled": args.capture,
        "duration_limit_s": args.duration,
        "frame_interval_s": args.frame_interval,
        "frame_limit": args.max_frames if args.capture else None,
        "transport": "OpenMV JPEG preview, host-decoded RGB888" if args.capture else "stdout only",
        "frame_timestamp_kind": "host receive time; not sensor exposure time",
        "validation_status": "acquisition only; not a verified training dataset or hardware performance result",
        "notes": "Source configuration and subject labels are declarations. Camera streaming affects runtime and power.",
    })
    print(f"Recording to {output}")

    started = time.monotonic()
    saved = 0
    status = "completed"
    reason = "duration limit"
    error = None
    line_buffer = ""
    firmware_error = None
    try:
        with (output / "stdout.log").open("w", encoding="utf-8") as stdout_log, \
             (output / "telemetry.jsonl").open("w", encoding="utf-8") as telemetry_log, \
             (output / "manifest.csv").open("w", newline="", encoding="utf-8") as manifest:
            writer = csv.DictWriter(manifest, fieldnames=MANIFEST_FIELDS)
            writer.writeheader()

            def consume_stdout(text: str | None) -> None:
                nonlocal line_buffer, firmware_error
                if not text:
                    return
                stdout_log.write(text)
                stdout_log.flush()
                if not args.quiet:
                    sys.stdout.write(text)
                    sys.stdout.flush()
                line_buffer += text
                while "\n" in line_buffer:
                    line, line_buffer = line_buffer.split("\n", 1)
                    try:
                        event = json.loads(line)
                    except (json.JSONDecodeError, ValueError):
                        continue
                    if isinstance(event, dict) and event.get("event") in ("fatal_error", "recognition_halted"):
                        firmware_error = event
                    telemetry_log.write(json.dumps({
                        "host_received_utc": utc_now(),
                        "firmware_stdout_json": event,
                        "frame_association": "not synchronized with preview captures",
                    }) + "\n")
                    telemetry_log.flush()

            # Keep each synchronous protocol operation short so the duration
            # loop can regain control promptly if stdout goes quiet.
            with Camera(args.port, timeout=0.5, max_retry=1) as cam:
                write_json(output / "device.json", {
                    "version": cam.version(), "system_info": cam.system_info(),
                })
                streaming = False
                uploaded = False
                try:
                    if script is not None:
                        cam.stop()
                        cam.exec(script)
                        uploaded = True
                    if args.capture:
                        cam.streaming(True)  # Native SDK's default JPEG transport.
                        streaming = True
                    started = time.monotonic()
                    next_frame = started
                    with duration_deadline(args.duration):
                        try:
                            while time.monotonic() - started < args.duration:
                                consume_stdout(cam.read_stdout())
                                if firmware_error is not None:
                                    raise RuntimeError("Firmware halted: " + str(firmware_error.get("error", firmware_error)))
                                now = time.monotonic()
                                if args.capture and now >= next_frame:
                                    frame = cam.read_frame()
                                    if frame is not None:
                                        width, height = int(frame["width"]), int(frame["height"])
                                        pixels = bytes(frame["data"])
                                        if width <= 0 or height <= 0 or len(pixels) != width * height * 3:
                                            raise ValueError("OpenMV frame is not a complete host-decoded RGB888 image")
                                        received = utc_now()
                                        saved += 1
                                        filename = f"images/frame_{saved:06d}.png"
                                        image_module.frombytes("RGB", (width, height), pixels).save(output / filename)
                                        writer.writerow({
                                            "session_id": args.session, "person_id": args.person_id,
                                            "identity_role": args.identity_role, "capture_camera": args.source_camera,
                                            "lighting": args.lighting, "split": args.split,
                                            "source_file": filename, "host_received_utc": received,
                                            "frame_index": saved, "width": width, "height": height,
                                            "transport": "JPEG preview -> SDK RGB888",
                                            "source_pixel_format_id": frame["format"],
                                            "wire_payload_bytes": frame["raw_size"],
                                            "stored_encoding": "PNG RGB",
                                            "source_config": "source_config.json" if args.source_config else "",
                                            "notes": "Derived preview; not raw sensor pixels or an exposure timestamp",
                                        })
                                        manifest.flush()
                                        next_frame = time.monotonic() + args.frame_interval
                                        if saved >= args.max_frames:
                                            reason = "frame limit"
                                            break
                                time.sleep(0.05)
                            if firmware_error is not None:
                                raise RuntimeError("Firmware halted: " + str(firmware_error.get("error", firmware_error)))
                            if args.capture and saved == 0:
                                raise RuntimeError("image collection was requested, but no preview frames arrived")
                            write_json(output / "transport_stats.json", {
                                "host": cam.host_stats(), "device": cam.device_stats(),
                            })
                        finally:
                            if streaming:
                                with contextlib.suppress(Exception):
                                    cam.streaming(False)
                            if uploaded:
                                with contextlib.suppress(Exception):
                                    cam.stop()
                except RecordingDeadline:
                    status, reason = "completed", "duration limit; interrupted a blocking OpenMV SDK operation"
    except KeyboardInterrupt:
        status, reason = "interrupted", "user interrupted recording"
    except SystemExit as exc:
        # openmv 1.0.7 can call sys.exit(0) on an unexpected transport error.
        # Preserve that as failed acquisition, never a completed recording.
        status, reason, error = "failed", "OpenMV SDK exited", f"SystemExit: {exc.code}"
        print(error, file=sys.stderr)
    except Exception as exc:
        status, reason, error = "failed", "recording error", f"{type(exc).__name__}: {exc}"
        print(error, file=sys.stderr)
    finally:
        write_json(output / "summary.json", {
            "status": status, "reason": reason, "error": error,
            "finished_utc": utc_now(), "host_elapsed_s": time.monotonic() - started,
            "saved_frames": saved,
            "firmware_error": firmware_error,
            "hardware_validation": "not established by this recorder",
        })
    print(f"{status}: {saved} images saved; details in {output}")
    return 0 if status == "completed" else 130 if status == "interrupted" else 1


def main() -> int:
    args = parse_args()
    if args.list_ports:
        return list_ports()
    try:
        return record(args)
    except (OSError, ValueError) as exc:
        print(f"Cannot prepare recording: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
