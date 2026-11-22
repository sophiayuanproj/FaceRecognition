"""Cycle local public portraits and collect matched Nicla one-shot observations."""
import argparse
import hashlib
import json
from pathlib import Path
import time
import urllib.request
import uuid

ROOT = Path(__file__).resolve().parents[5]
OPTICAL = ROOT / 'work/project-research-v5/runs/optical'
CONTROL = OPTICAL / 'CONTROL.json'
RENDERS = OPTICAL / 'RENDER_EVENTS.jsonl'
BASE = 'http://127.0.0.1:8780'
DEFAULT_SCENES = (
    ('bush-original', 0, 90, 1.0, '#777', 50, 50),
    ('bush-dimmer', 0, 99, .68, '#343434', 50, 50),
    ('bush-frontal', 2, 99, .77, '#444', 50, 50),
    ('powell-turned', 3, 99, .77, '#444', 50, 50),
    ('blair-turned', 8, 99, .77, '#444', 50, 50),
    ('rumsfeld-looking-up', 11, 99, .77, '#444', 50, 50),
    ('schroeder-tilted', 13, 99, .77, '#444', 50, 50),
    ('jolie-turned', 33, 99, .77, '#444', 50, 50),
    ('kidd-forward', 19, 99, .77, '#444', 50, 50),
    ('bush-smaller-shifted', 2, 68, .77, '#444', 56, 49),
    ('checkerboard', 'calibration', 93, .68, '#444', 50, 50),
    ('blank', 'blank', 93, .77, '#666', 50, 50),
)


def digest(data):
    return hashlib.sha256(data).hexdigest()


def set_display(state):
    current = dict(state, version=time.time_ns() // 1_000_000)
    temp = CONTROL.with_name(f'.screen-trial-control-{uuid.uuid4().hex}.tmp')
    temp.write_text(json.dumps(current) + '\n', encoding='utf-8')
    try:
        for attempt in range(12):
            try:
                temp.replace(CONTROL)
                break
            except PermissionError:
                if attempt == 11:
                    raise
                time.sleep(.025)
    finally:
        temp.unlink(missing_ok=True)
    return current


def render_offset():
    try:
        return RENDERS.stat().st_size
    except FileNotFoundError:
        return 0


def wait_render(version, start_offset):
    deadline = time.monotonic() + 7
    while time.monotonic() < deadline:
        try:
            with RENDERS.open(encoding='utf-8') as stream:
                stream.seek(start_offset)
                for line in stream:
                    try:
                        render = json.loads(line)
                    except ValueError:
                        continue
                    if render.get('version') == version:
                        return render
        except FileNotFoundError:
            pass
        time.sleep(.12)
    raise RuntimeError(f'The target tab did not acknowledge display version {version}')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--repeats', type=int, default=3)
    parser.add_argument('--device-url', default=BASE)
    parser.add_argument('--scenes', type=Path, help='Optional JSON list with the same columns as DEFAULT_SCENES')
    parser.add_argument('--extra-sources', type=Path, help='Additional local sources indexed by token')
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    viewer = 'screen-test-' + uuid.uuid4().hex[:12]
    prior_display = json.loads(CONTROL.read_text())
    sources = json.loads((OPTICAL/'DISPLAY_SOURCES.json').read_text())['items']
    extra = ({row['token']: row for row in json.loads(args.extra_sources.read_text())['items']}
        if args.extra_sources else {})
    scenes = json.loads(args.scenes.read_text()) if args.scenes else DEFAULT_SCENES

    def call(route, body=None, binary=False):
        request = urllib.request.Request(args.device_url.rstrip('/')+route,
            data=None if body is None else json.dumps(body).encode(),
            headers={'X-Viewer-ID': viewer, 'Content-Type': 'application/json'})
        with urllib.request.urlopen(request, timeout=10) as response:
            return response.read() if binary else json.load(response)

    def device(route, body=None):
        return call('/api/v2/device/'+route, body)

    def capture(control, boot):
        command_id = uuid.uuid4().hex
        cursor = device('status')['event_seq']
        request = dict(control, schema='nicla-command-v2', command_id=command_id,
            cmd='one_shot', expected_boot_id=boot, timeout_ms=7000)
        reply = device('commands', request)
        if reply.get('status') != 'accepted':
            raise RuntimeError(reply)
        deadline = time.monotonic() + 8
        ack = observation = None
        while time.monotonic() < deadline:
            batch = device(f'events?after={cursor}&limit=100')
            cursor = batch['next_seq']
            for event in batch['events']:
                if event.get('command_id') != command_id:
                    continue
                if event.get('type') == 'observation_complete':
                    observation = event
                elif event.get('type') == 'command_ack' and event.get('control_status') != 'accepted':
                    ack = event
            if ack and (observation or ack.get('control_status') != 'completed'):
                break
            time.sleep(.1)
        if not ack or ack.get('control_status') != 'completed' or not observation:
            return {'command_id':command_id, 'ack_status':(ack or {}).get('control_status'),
                'frame_id':None}, None
        row = {key:observation.get(key) for key in ('boot_id','frame_id','host_received_utc',
            'source','camera','detection','crop','identity','decision','timings_ms','failure_reason')}
        row.update(command_id=command_id, ack_status=ack['control_status'], payload_checks={})
        jpeg = None
        for name in ('frame','detector_input','identity_input'):
            item = observation.get(name) or {}
            ref = item.get('frame_ref') or item.get('input_ref')
            if not ref:
                continue
            data = call(ref, binary=True)
            row['payload_checks'][name] = (item.get('sha256') == digest(data)
                and (item.get('boot_id'),item.get('frame_id')) == (boot,row['frame_id']))
            if name == 'frame':
                jpeg = data
            else:
                row[name] = {key:item.get(key) for key in ('shape','source','sha256')}
        return row, jpeg

    lease = device('controller/claim', {'viewer_id':viewer})
    if not lease.get('owned_by_viewer'):
        raise RuntimeError('Another dashboard viewer holds the controller lease')
    control = {'viewer_id':viewer, 'controller_id':lease['controller_id']}
    try:
        status = device('status')
        if not status['device_ready'] or status['running'] or status['sleeping']:
            raise RuntimeError('Camera must be connected, awake, and stopped')
        applied = status.get('applied_state') or {}
        if applied.get('policy_id') != 'S' or applied.get('model_override') is not None:
            raise RuntimeError(f'Expected fixed policy S and its default model, got {applied}')
        boot = status['boot_id']
        for name, source_id, size, opacity, background, x, y, *orientation in scenes:
            device('controller/heartbeat', control)
            if isinstance(source_id,int):
                source = sources[source_id]
            elif source_id in ('calibration','blank'):
                source = None
            else:
                source = extra[source_id]
            state = {'kind':'photo' if source else source_id, 'size_vmin':size,
                'opacity':opacity, 'background':background, 'x_pct':x, 'y_pct':y}
            if orientation:
                state['rotation_deg'] = orientation[0]
            if source:
                state['token'] = source['token']
            offset = render_offset()
            state = set_display(state)
            render = wait_render(state['version'], offset)
            time.sleep(2.2)
            for i in range(args.repeats):
                device('controller/heartbeat', control)
                row, jpeg = capture(control, boot)
                row.update(scene=name, repeat=i+1, display=state, render=render,
                    display_source=(source['record_id'] if source else None),
                    display_source_sha256=(source['sha256'] if source else None),
                    expected_face=source.get('expected_face',True) if source else False)
                if source:
                    for key in ('face_box_xywh_display_px','eye_points_display_px',
                                'source_camera','source_frame'):
                        if key in source:
                            row[key] = source[key]
                if jpeg:
                    path = args.output/f'{name}-{i+1}.jpg'
                    path.write_bytes(jpeg)
                    row['jpeg'] = path.name
                with (args.output/'captures.jsonl').open('a',encoding='utf-8') as stream:
                    stream.write(json.dumps(row,sort_keys=True)+'\n')
                print(name,i+1,row.get('frame_id'),(row.get('detection') or {}).get('status'),
                    (row.get('identity') or {}).get('status'),(row.get('decision') or {}).get('status'),flush=True)
    finally:
        set_display(prior_display)
        device('controller/release', control)


if __name__ == '__main__':
    main()
