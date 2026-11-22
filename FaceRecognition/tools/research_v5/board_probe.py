"""Bounded read-only Nicla raw-REPL discovery; no filesystem writes on target."""
import argparse
import json
from pathlib import Path
import time
import serial
from serial.tools import list_ports


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    ports = {p.device: p for p in list_ports.comports()}
    if args.port not in ports or ports[args.port].vid != 0x2341:
        raise SystemExit("Refusing a port not freshly identified as Arduino")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    result = {"schema": "nicla-v5-readonly-probe-1", "target_files_written": False,
              "vid": ports[args.port].vid, "pid": ports[args.port].pid, "steps": []}
    with serial.Serial(args.port, 115200, timeout=0.1, write_timeout=2, exclusive=True) as link:
        link.rts = False
        link.dtr = True
        time.sleep(0.2)
        initial = link.read(16384)
        args.output.with_suffix(".initial.bin").write_bytes(initial)
        link.write(b"\x03\x03")
        link.flush()
        time.sleep(0.1)
        link.write(b"\x01")
        link.flush()
        time.sleep(0.15)
        prompt = link.read(16384)
        result["steps"].append({"step": "raw_prompt", "bytes": len(prompt),
                                "text": prompt.decode("utf-8", "replace")[-1000:]})
        program = ("import sys,os,gc;print('V5_INFO',sys.implementation,os.uname(),"
                   "os.statvfs('/flash'),gc.mem_free());print('V5_ROOT',os.listdir('/flash'))")
        link.write(program.encode() + b"\x04")
        link.flush()
        end = time.monotonic() + 2.0
        raw = bytearray()
        while time.monotonic() < end:
            raw.extend(link.read(16384))
            if b"\x04\x04>" in raw:
                break
        args.output.with_suffix(".raw.bin").write_bytes(raw)
        result["steps"].append({"step": "info", "text": raw.decode("utf-8", "replace")})
        result["raw_control_verified"] = b"V5_INFO" in raw and b"Traceback" not in raw
        link.write(b"\x02")
        link.flush()
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps(result, allow_nan=False))
    return 0 if result["raw_control_verified"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
