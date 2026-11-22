"""Bounded, transcript-preserving standard-raw-REPL research executor.

Each block starts a fresh MicroPython VM. It is a warm/software reset, never
evidence of a physical power cycle. Does not own the port outside the command.
"""
import argparse
import hashlib
import json
from pathlib import Path
import time
from mpremote.transport_serial import SerialTransport


def connect(port):
    target = SerialTransport(port, timeout=10)
    target.serial.rts = False
    target.serial.dtr = True
    target.use_raw_paste = False
    target.enter_raw_repl(soft_reset=False)
    return target


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--port', required=True)
    p.add_argument('--program', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--blocks', type=int, default=1)
    p.add_argument('--timeout', type=int, default=90)
    p.add_argument('--continue-on-error', action='store_true', help='Only for deliberate fault studies; otherwise stop at first unexpected failure')
    args = p.parse_args()
    if not 1 <= args.blocks <= 30 or not 1 <= args.timeout <= 600:
        raise ValueError('Bounded blocks/timeouts required')
    source = args.program.read_bytes()
    args.output.mkdir(parents=True, exist_ok=True)
    summary = []
    for block in range(args.blocks):
        started = time.monotonic()
        target = None
        out = b""
        err = b""
        row = {'block': block, 'reset_kind': 'independent_soft_reset',
               'program_sha256': hashlib.sha256(source).hexdigest()}
        try:
            target = connect(args.port)
            target.enter_raw_repl(soft_reset=True)
            executed_source = ('BLOCK_INDEX=%d\n' % block).encode() + source
            out, err = target.exec_raw(executed_source, timeout=args.timeout)
            records = []
            for line in out.decode('utf-8', errors='replace').splitlines():
                try:
                    records.append(json.loads(line))
                except ValueError:
                    pass
            row.update(status='error' if err else 'executed', records=records,
                       stdout_sha256=hashlib.sha256(out).hexdigest(),
                       stderr=err.decode('utf-8', errors='replace'))
            target.exit_raw_repl()
        except Exception as error:
            row.update(status='transport_error', error=repr(error))
        finally:
            if target is not None:
                try:
                    target.close()
                except Exception as error:
                    # A USB detach or RTS cleanup error must not discard the
                    # captured output and partial run receipt.
                    row['close_warning'] = repr(error)
            (args.output / ('block_%02d.stdout.bin' % block)).write_bytes(out)
            (args.output / ('block_%02d.stderr.bin' % block)).write_bytes(err)
            row['host_elapsed_s'] = time.monotonic() - started
            summary.append(row)
            (args.output / 'RESULT.json').write_text(json.dumps(summary, indent=2, allow_nan=False)+'\n')
            print(json.dumps(row, allow_nan=False), flush=True)
        if row['status']!='executed' and not args.continue_on_error:
            break
    return 0 if all(row['status']=='executed' for row in summary) else 1


if __name__ == '__main__':
    raise SystemExit(main())
