"""Ten actual paired optical blocks on frozen public development sources."""
import datetime,hashlib,json,random,time,urllib.request
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3];S=ROOT/'work/project-research-v5'

def rpc(action,obj):
    req=urllib.request.Request('http://127.0.0.1:8766/'+action,data=json.dumps(obj,allow_nan=False).encode(),headers={'Content-Type':'application/json'})
    return json.load(urllib.request.urlopen(req,timeout=65))

def show(state):
    p=S/'runs/optical/CONTROL.json';temp=p.with_suffix('.tmp');temp.write_text(json.dumps(state,allow_nan=False));temp.replace(p)
    end=time.monotonic()+8
    while time.monotonic()<end:
        log=S/'runs/optical/RENDER_EVENTS.jsonl'
        if log.exists():
            for line in reversed(log.read_text().splitlines()):
                try:row=json.loads(line)
                except ValueError:continue
                if row['version']==state['version']:return row
        time.sleep(.05)
    raise RuntimeError('Display did not acknowledge rendering the requested stimulus')

def main():
    session='optical-'+datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    out=S/'runs/optical'/session;out.mkdir()
    sources=json.loads((S/'runs/optical/DISPLAY_SOURCES.json').read_text())['items']
    known={k:[r for r in sources if r['label']==k] for k in ['K01','K02','K03','K04','K05']}
    unknown=[r for r in sources if not r['known']]
    if len({r['source_identity'] for r in unknown})!=30:raise ValueError('Need thirty actual distinct unknown IDs')
    if any('future_final' in r['role'] for r in sources):raise ValueError('Final role prohibited')
    conditions=[{'size_vmin':50,'x_pct':50},{'size_vmin':70,'x_pct':50},{'size_vmin':90,'x_pct':50},{'size_vmin':70,'x_pct':35},{'size_vmin':70,'x_pct':65}]*2
    random.Random(54000).shuffle(conditions);orders=[0,1]*5;random.Random(54011).shuffle(orders)
    blocks=[]
    for b in range(10):
        stimuli=[dict(known[k][(b+i)%3],truth=k) for i,k in enumerate(known)]
        stimuli.extend(dict(r,truth='OTHER') for r in unknown[3*b:3*b+3]);stimuli.append({'truth':'NO_FACE','kind':'blank','role':'generated_development_control'})
        random.Random(54001+b).shuffle(stimuli)
        blocks.append({'block':b,'seed':54001+b,'pair_order':orders[b],'condition':conditions[b],'stimuli':stimuli})
    protocol={'schema':'v5-actual-optical-paired-1','session':session,'blocks':blocks,
        'native_capture':'240x240 RGB565, exact raw bytes + actual signed FOMO96x96x3 input/output + Haar160 gray input',
        'physical_session':'one taped screen-camera setup; source/session dependence retained',
        'replication':'ten independently software-reset acquisition blocks, NOT ten power cycles or independent environments',
        'settling_after_render_ack_s':.7,'actual_ambient_light':'unmeasured; no lux claim','actual_range_cm':None,
        'ground_truth':'frozen source identity+role, independent of detector predictions; browser render acknowledgement is not sensor visibility proof',
        'known_presentations':50,'known_source_groups':15,'unknown_presentations':30,'unknown_identity_groups':30,
        'negative_controls':10,'engineering_targets_automatically_met':False,'future_final_used':False,
        'source_manifest_sha256':hashlib.sha256((S/'data/manifests/lfw_roles.jsonl').read_bytes()).hexdigest()}
    (out/'PROTOCOL.json').write_text(json.dumps(protocol,indent=2,allow_nan=False)+'\n');receipts=[];version=int(time.time()*1000)
    for block in blocks:
        reset=rpc('reset',{'pair_order':block['pair_order']});print(json.dumps({'block':block['block'],'reset':reset}),flush=True)
        # Calibration belongs to the same display size/position as this block.
        stimuli=[{'kind':'calibration','truth':'CALIBRATION','role':'generated_development_control'}]+block['stimuli']
        for i,source in enumerate(stimuli):
            version+=1;display=dict(block['condition'],version=version,kind=source.get('kind','photo'),opacity=.65,background='#777',y_pct=50)
            if 'token' in source:display['token']=source['token']
            render=show(display);time.sleep(.7)
            request={'capture_id':'%s-B%02d-T%02d'%(session,block['block'],i),'session':session,'block':block['block'],
                'physical_block_seed':block['seed'],'display':display,'render_ack':render,'source':source,'expected_truth':source['truth'],
                'reset':reset,'evidence_tier':'actual_optical_camera_development','operator_setup':'user confirmed taped Nicla pointed at Codex display'}
            result=rpc('capture',request)
            if 'error' in result:raise RuntimeError(result['error'])
            receipts.append(result)
            (out/'RECEIPTS.json').write_text(json.dumps(receipts,indent=2,allow_nan=False)+'\n')
            meta=result['metadata']
            if meta.get('fomo_error') or meta.get('haar_error'):raise RuntimeError('Stop before repeating unexpected device failure: '+str(meta))
        print(json.dumps({'completed_block':block['block'],'actual_captures':len(receipts),'source_faces_this_block':8,'native_reset_index':reset['reset_index']}),flush=True)
    (out/'COMPLETE.json').write_text(json.dumps({'status':'captured_not_yet_analyzed','session':session,'physical_blocks':10,'total_captures':len(receipts),
        'known_presentations':50,'unknown_presentations':30,'unknown_ids':30,'calibration_frames':10,'blank_frames':10,'future_final_used':False},indent=2)+'\n')
    (S/'runs/optical/LATEST_SESSION.txt').write_text(out.relative_to(ROOT).as_posix()+'\n')
    print(json.dumps({'status':'complete_capture','directory':str(out)}),flush=True)

if __name__=='__main__':main()
