"""Run identical video workloads on the Nicla and emit meter alignment receipts."""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import random
import re
from statistics import median
import time
import urllib.request
import uuid

from video_frame_evidence import (
    PrivateSamples, coverage_counts, episode_at, link_frames,
    media_at, media_knots, observation_capture_ns, projected_match,
)

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('run_screen_trial',HERE.parents[1]/'screen-variation/run_screen_trial.py')
optical = importlib.util.module_from_spec(spec)
spec.loader.exec_module(optical)
ARMS = {
    'capture-only': ('S','camera_capture_only','small'),
    'detector-only': ('S','detector_only','small'),
    'fixed-small': ('S','full_pipeline','small'),
    'scene-gated-small': ('T','full_pipeline','small'),
    'spatial-gated-small': ('V','full_pipeline','small'),
    'large-smoke': ('H','full_pipeline','large'),
    'fast-live-diagnostic': ('F','full_pipeline','small'),
}
PRIMARY = ('capture-only', 'detector-only', 'fixed-small', 'scene-gated-small')


def write_json(path, data):
    path.write_text(json.dumps(data,indent=2,allow_nan=False)+'\n',encoding='utf-8')


def trial_orders(arms, blocks, seed, shuffle_single_block=False):
    rng = random.Random(seed)
    orders = []
    for _ in range(blocks):
        order = list(arms)
        if blocks > 1 or shuffle_single_block:
            rng.shuffle(order)
        orders.append(order)
    return orders


class Device:
    def __init__(self,url):
        self.url=url.rstrip('/')
        self.viewer='video-power-'+uuid.uuid4().hex[:12]
        lease=self.call('controller/claim',{'viewer_id':self.viewer})
        if not lease.get('owned_by_viewer'):
            raise RuntimeError('Release dashboard control before running the video experiment.')
        self.control={'viewer_id':self.viewer,'controller_id':lease['controller_id']}

    def call(self,path,body=None):
        req=urllib.request.Request(self.url+'/api/v2/device/'+path,
            data=None if body is None else json.dumps(body).encode(),
            headers={'X-Viewer-ID':self.viewer,'Content-Type':'application/json'})
        with urllib.request.urlopen(req,timeout=7) as response:
            return json.load(response)

    def command(self,name,events,**fields):
        cid=uuid.uuid4().hex
        self.call('controller/heartbeat',self.control)
        cursor=self.call('status')['event_seq']
        sent=time.monotonic_ns()
        reply=self.call('commands',dict(self.control,schema='nicla-command-v2',cmd=name,command_id=cid,
            timeout_ms=7000,**fields))
        if reply.get('status')!='accepted':
            raise RuntimeError((name,reply))
        deadline=time.monotonic()+8
        while time.monotonic()<deadline:
            batch=self.call(f'events?after={cursor}&limit=100')
            cursor=batch['next_seq']
            events.extend(batch['events'])
            for event in batch['events']:
                if (event.get('type')=='command_ack' and event.get('command_id')==cid
                        and event.get('control_status')!='accepted'):
                    if event['control_status']!='completed':
                        raise RuntimeError((name,event))
                    return event,sent
            time.sleep(.06)
        raise RuntimeError(f'No completed acknowledgement for {name}')


def read_video_events(version,offset):
    try:
        with optical.RENDERS.open(encoding='utf-8') as stream:
            stream.seek(offset)
            for line in stream:
                try:
                    row=json.loads(line)
                except ValueError:
                    continue
                if row.get('version')==version and row.get('event'):
                    yield row
    except FileNotFoundError:
        return


def playback_receipt(render, geometry, boot):
    playback = [row for row in render if row.get('event') in ('video_started', 'video_tick', 'video_ended')]
    if len(playback) < 3 or playback[0]['event'] != 'video_started' or playback[-1]['event'] != 'video_ended':
        return {'verified': False, 'reason': 'missing_ordered_playback_receipts'}
    if any(row.get('page_visibility') != 'visible' for row in playback):
        return {'verified': False, 'reason': 'player_tab_hidden_or_renderer_outdated'}
    gaps_ms = [right['browser_ms'] - left['browser_ms'] for left, right in zip(playback, playback[1:])]
    largest_gap_ms = max(gaps_ms)
    if min(gaps_ms) <= 0 or largest_gap_ms > 2000:
        return {'verified': False, 'reason': 'playback_receipts_stalled', 'largest_gap_ms': largest_gap_ms}
    if geometry and (geometry.get('boot_id') != boot
                     or any(row['viewport'] != geometry.get('viewport') for row in playback)):
        return {'verified': False, 'reason': 'calibration_boot_or_viewport_changed'}
    return {'verified': True, 'largest_gap_ms': largest_gap_ms, 'receipt_count': len(playback),
            'focused_receipt_count': sum(row.get('page_has_focus') is True for row in playback)}


def summarize(events,render,video,policy,stage,boot,projection=None,linked_frames=None):
    start=next(r for r in render if r['event']=='video_started')
    end=next(r for r in render if r['event']=='video_ended')
    knots=media_knots(render)
    episodes=[dict(ep,name=ep['id'],end_ms=ep['end_ms_exclusive'],
        observations=0,detector_calls=0,identity_runs=0,detections=0,
        matched_faces=0,matched_identity_runs=0,first_matched_face_ms=None,
        capture_to_first_face_ms=None,first_detected_media_ms=None) for ep in video['episodes']]
    rows=[]
    seen=set()
    for row in events:
        if row.get('type')!='observation_complete' or row.get('boot_id')!=boot:
            continue
        frame_id=row['frame_id']
        if linked_frames is not None and frame_id not in linked_frames:
            continue
        if frame_id in seen:
            continue
        seen.add(frame_id)
        media = linked_frames[frame_id]['media_ms'] if linked_frames is not None else media_at(
            knots, observation_capture_ns(row))
        if media is None:
            continue
        info=episode_at({'episodes':episodes},media)
        applied=row.get('applied_state') or {}
        detect=(row.get('detection') or {}).get('status')
        identity=(row.get('identity') or {}).get('status')
        decision=row.get('decision') or {}
        total=(row.get('timings_ms') or {}).get('total')
        requested=row.get('requested_state') or {}
        status={'frame_id':frame_id,'estimated_media_ms':round(media,1),'episode':info['name'] if info else None,
            'detector_status':detect,'identity_status':identity,'model_id':(row.get('identity') or {}).get('model_id'),
            'policy_id':applied.get('policy_id'),'stage':applied.get('measurement_stage_id',requested.get('measurement_stage_id')),
            'decision':decision.get('status'),'reason':decision.get('reason'),'total_ms':total,
            'boot_id':boot,'frame_ref':(row.get('frame') or {}).get('frame_ref')}
        rows.append(status)
        if info:
            info['observations']+=1
            info['detector_calls']+=detect not in (None,'not_run','skipped')
            info['identity_runs']+=identity=='ran'
            info['detections']+=detect=='face'
            if detect=='face' and info['first_detected_media_ms'] is None:
                info['first_detected_media_ms']=round(media,1)
                info['capture_to_first_face_ms']=round(media-info['start_ms'],1)
            if projection and info['expected_face']:
                matched,coverage=projected_match(row,projection,info,video['fps'],media)
                status.update(matched_face=matched,face_coverage_in_crop=round(coverage,3))
                info['matched_faces']+=matched
                info['matched_identity_runs']+=matched and coverage>=.80 and identity=='ran'
                if matched and info['first_matched_face_ms'] is None:
                    info['first_matched_face_ms']=round(media-info['start_ms'],1)
    speeds=[r['total_ms'] for r in rows if isinstance(r['total_ms'],(int,float))]
    return {
        'meter_window_start_unix_ms':start['browser_wall_ms'],
        'meter_window_end_unix_ms':end['browser_wall_ms'],
        'window_host_monotonic_start_ns':start['host_receive_monotonic_ns'],
        'window_host_monotonic_end_ns':end['host_receive_monotonic_ns'],
        'episodes':episodes,'rows':rows,
        'observations':len(rows),
        'detector_calls':sum(r['detector_status'] not in (None,'not_run','skipped') for r in rows),
        'identity_runs':sum(r['identity_status']=='ran' for r in rows),
        'quiet_skips':sum(r['reason']=='SCENE_TRIGGER_SKIPPED_QUIET_CHECK' for r in rows),
        'identity_errors':sum(r['identity_status']=='error' for r in rows),
        'median_cycle_ms':median(speeds) if speeds else None,
        'policy_mismatches':sum(r['policy_id'] not in (None,policy) for r in rows),
        'stage_mismatches':sum(r['stage'] not in (None,stage) for r in rows),
    }


def run_one(device,video,layout,arm,block,out,optics_confirmed,geometry,expected,plan,private_samples_path=None):
    policy,stage,slot=ARMS[arm]
    status=device.call('status')
    boot=status['boot_id']
    events=[]
    render=[]
    private_samples=None
    projection=geometry['video_px_to_nicla_px'] if geometry else None
    event_gaps=[]
    result=dict(arm=arm,block=block,video_sha256=video['sha256'],video_token=video['token'],
        expected_duration_ms=video['duration_ms'],boot_id=boot,policy=policy,stage=stage,
        model_slot=slot,measurement_ready=False,**plan)
    try:
        if expected['boot_id'] and boot != expected['boot_id']:
            raise RuntimeError('Nicla restarted after meter preflight.')
        blank_offset=optical.render_offset()
        blank=optical.set_display(dict(layout,kind='blank'))
        optical.wait_render(blank['version'],blank_offset)
        device.command('set_model',events,model_slot='policy',expected_boot_id=boot)
        if slot=='small':
            result['preload_ack'],_=device.command('preload_model',events,model_slot='small',expected_boot_id=boot)
        result['start_ack'],_=device.command('start',events,policy_id=policy,
            measurement_stage_id=stage,model_slot=slot,expected_boot_id=boot)
        if private_samples_path is not None:
            private_samples=PrivateSamples(private_samples_path,device.url,boot,result['start_ack'],video,projection)
        cursor=device.call('status')['event_seq']
        offset=optical.render_offset()
        display=optical.set_display(dict(layout,kind='video',token=video['token'],
            play_at_unix_ms=time.time_ns()//1_000_000+1700))
        deadline=time.monotonic()+video['duration_ms']/1000+9
        renew=time.monotonic()+4
        while time.monotonic()<deadline:
            cursor=read_events(device,cursor,events,event_gaps)
            render=list(read_video_events(display['version'],offset))
            if private_samples is not None and len(render)>2:
                frames,_=link_frames(events,render,boot)
                private_samples.collect(frames)
            if any(r['event']=='video_error' for r in render):
                raise RuntimeError('Browser reported video_error')
            if any(r['event']=='video_ended' for r in render):
                break
            if time.monotonic()>=renew:
                device.call('controller/heartbeat',device.control)
                renew=time.monotonic()+4
            time.sleep(.11)
        else:
            raise RuntimeError('No video-ended receipt. Keep the player visible.')
        result['stop_ack'],_=device.command('stop',events,expected_boot_id=boot)
        read_events(device,cursor,events,event_gaps)
        frames,encode_errors=link_frames(events,render,boot)
        linked_frames={frame_id: row for frame_id,row in frames.items() if row['linked']}
        result['frame_linkage']=coverage_counts(frames,encode_errors,video)
        result['event_stream_gaps']=event_gaps
        result.update(summarize(events,render,video,policy,stage,boot,projection,linked_frames))
        window=result['meter_window_end_unix_ms']-result['meter_window_start_unix_ms']
        end_status=device.call('status')
        result['playback_receipt']=playback_receipt(render,geometry,boot)
        result['sequencing_verified']=bool(abs(window-video['duration_ms'])<600 and result['observations']>0
            and end_status['boot_id']==boot and not end_status.get('running')
            and not result['policy_mismatches'] and not result['stage_mismatches']
            and not result['frame_linkage']['whole_playback']['identity_errors'] and not event_gaps)
        result['measurement_ready']=bool(result['sequencing_verified'] and optics_confirmed
                                         and result['playback_receipt']['verified'] and not plan.get('diagnostic_only'))
        result['optics_confirmed']=optics_confirmed
        result['optical_geometry']=geometry
        hashes={(event.get('firmware_sha256'),event.get('model_config_sha256'))
            for event in events if event.get('type')=='observation_complete' and event.get('boot_id')==boot}
        result['firmware_config_sha256_pairs']=sorted([list(pair) for pair in hashes],key=str)
        if len(hashes)!=1 or not all(next(iter(hashes),())):
            result['measurement_ready']=False
        if any((expected['firmware_sha256'] and firmware != expected['firmware_sha256'])
               or (expected['config_sha256'] and config != expected['config_sha256'])
               for firmware,config in hashes):
            result['measurement_ready']=False
        print(arm,block,result['observations'],result['detector_calls'],result['identity_runs'],
            result['quiet_skips'],result['median_cycle_ms'],result['sequencing_verified'],flush=True)
    except Exception as error:
        result['error']=f'{type(error).__name__}: {error}'
        raise
    finally:
        if private_samples is not None:
            result['private_samples']={'local_folder':str(private_samples_path)}
            try:
                result['private_samples']['verified_counts']=private_samples.write_receipt()
            except OSError as error:
                result['private_samples']['receipt_error']=str(error)
        write_json(out/'RESULT.json',result)
        unique_events={row['event_seq']:row for row in events}
        write_json(out/'device_events.json',[unique_events[seq] for seq in sorted(unique_events)])
        write_json(out/'render_events.json',render)


def read_events(device,cursor,events,gaps):
    batch=device.call(f'events?after={cursor}&limit=100')
    if batch.get('truncated'):
        gaps.append(dict(after_seq=cursor,oldest_seq=batch.get('oldest_seq'),
                         skipped_count=batch.get('skipped_count')))
    events.extend(batch['events'])
    return batch['next_seq']


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',required=True,type=Path)
    parser.add_argument('--device-url',default='http://127.0.0.1:8780')
    parser.add_argument('--layout',type=Path)
    parser.add_argument('--geometry',type=Path,help='Projection JSON produced by calibrate_video_layout.py')
    parser.add_argument('--expected-boot-id', help='Require the same boot as the meter preflight')
    parser.add_argument('--expected-config-sha256', help='Require the deployment config used by the meter preflight')
    parser.add_argument('--expected-firmware-sha256', help='Require the firmware used by the meter preflight')
    parser.add_argument('--arms',nargs='+',choices=tuple(ARMS),default=list(PRIMARY))
    parser.add_argument('--blocks',type=int,default=1)
    parser.add_argument('--seed',type=int,default=4072)
    parser.add_argument('--physical-block-id', help='Unique identifier for one fresh physical start; requires --blocks 1')
    parser.add_argument('--shuffle-single-block',action='store_true',
        help='Shuffle even when --blocks is 1. Omit to use the declared --arms order for a single block.')
    parser.add_argument('--meter-run-id', help='Run ID expected in the meter CSV')
    parser.add_argument('--meter-csv',type=Path,help='Meter CSV to bind by relative path in each trial receipt')
    parser.add_argument('--topology-id', help='Predeclared measured supply and cable topology identifier')
    parser.add_argument('--optics-confirmed',action='store_true',
        help='Use only after checking on-camera playback and a matching checkerboard at this layout.')
    parser.add_argument('--private-frame-samples',type=Path,
        help='New local directory outside --output for up to two positive and two quiet linked JPEGs per arm.')
    args=parser.parse_args()
    if args.blocks<1 or len(set(args.arms))!=len(args.arms):
        parser.error('Use at least one block and unique arms.')
    if args.physical_block_id is not None and args.blocks != 1:
        parser.error('--physical-block-id identifies one fresh physical start; use --blocks 1.')
    fast_diagnostic='fast-live-diagnostic' in args.arms
    diagnostic=fast_diagnostic or args.private_frame_samples is not None
    if diagnostic and any(value is not None for value in (
            args.meter_run_id,args.meter_csv,args.topology_id,args.physical_block_id)):
        parser.error('Diagnostic runs and private frame samples require unmetered output.')
    if fast_diagnostic and (args.arms!=['fast-live-diagnostic'] or args.blocks!=1 or not args.optics_confirmed):
        parser.error('Use fast-live-diagnostic alone with --blocks 1 and --optics-confirmed.')
    if args.private_frame_samples is not None:
        private=args.private_frame_samples.resolve()
        output=args.output.resolve()
        if private.is_relative_to(output) or output.is_relative_to(private) or private.exists():
            parser.error('--private-frame-samples must be a new directory separate from --output.')
    for flag, value in (('--physical-block-id',args.physical_block_id),
                        ('--meter-run-id',args.meter_run_id),('--topology-id',args.topology_id)):
        if value is not None and not re.fullmatch(r'[A-Za-z0-9._-]{1,100}', value):
            parser.error(flag+' must contain 1 to 100 letters, digits, dots, underscores or hyphens.')
    expected={'boot_id':args.expected_boot_id,'config_sha256':args.expected_config_sha256,
              'firmware_sha256':args.expected_firmware_sha256}
    video=json.loads((HERE/'VIDEO_MANIFEST.json').read_text())['items'][0]
    if hashlib.sha256((optical.ROOT / video['path']).read_bytes()).hexdigest()!=video['sha256']:
        raise ValueError('Video hash mismatch')
    saved=json.loads(optical.CONTROL.read_text())
    layout=json.loads(args.layout.read_text()) if args.layout else {
        'size_vmin':96,'opacity':.76,'background':'#444','x_pct':50,'y_pct':50,'rotation_deg':-10}
    geometry=json.loads(args.geometry.read_text()) if args.geometry else None
    if args.optics_confirmed and not geometry:
        parser.error('--optics-confirmed requires --geometry from a linked calibration shot')
    if geometry and geometry.get('layout')!=layout:
        parser.error('Calibration layout differs from playback layout.')
    orders=trial_orders(args.arms,args.blocks,args.seed,args.shuffle_single_block)
    plan={'seed':args.seed,'order_method':'seeded_shuffle' if args.blocks>1 or args.shuffle_single_block else 'declared_arms_order'}
    if diagnostic:
        plan['diagnostic_only']=True
    for key in ('physical_block_id','meter_run_id','topology_id'):
        if getattr(args,key) is not None:
            plan[key]=getattr(args,key)
    if args.meter_csv is not None:
        try:
            meter_relative=os.path.relpath(args.meter_csv.resolve(),args.output.resolve())
        except ValueError:
            parser.error('--meter-csv and --output must be on the same drive for a relocatable receipt.')
        plan['meter_csv_path']=(Path('..')/meter_relative).as_posix()
    args.output.mkdir(parents=True,exist_ok=False)
    if args.private_frame_samples is not None:
        args.private_frame_samples.mkdir(parents=True,exist_ok=False)
    write_json(args.output/'PLAN.json',dict(plan,expected=expected,video_sha256=video['sha256'],orders=orders))
    device=Device(args.device_url)
    try:
        state=device.call('status')
        if not state['device_ready'] or state['running']:
            raise RuntimeError('Nicla needs to be connected and stopped.')
        if args.expected_boot_id and state['boot_id'] != args.expected_boot_id:
            raise RuntimeError('Nicla boot differs from meter preflight.')
        for field in ('config_sha256','firmware_sha256'):
            if expected[field] and (state.get('device') or {}).get(field) != expected[field]:
                raise RuntimeError(f'Nicla {field} differs from meter preflight.')
        for block, arms in enumerate(orders,1):
            for arm_index, arm in enumerate(arms,1):
                out=args.output/f'{block:02d}-{arm}'
                out.mkdir()
                run_one(device,video,layout,arm,block,out,args.optics_confirmed,geometry,expected,
                        dict(plan,arm_order=arms,arm_index=arm_index),
                        args.private_frame_samples/out.name if args.private_frame_samples is not None else None)
    finally:
        try:
            if device.call('status').get('running'):
                device.command('stop',[])
            device.command('set_policy',[],policy_id='S')
        finally:
            optical.set_display(saved)
            device.call('controller/release',device.control)


if __name__=='__main__':
    main()
