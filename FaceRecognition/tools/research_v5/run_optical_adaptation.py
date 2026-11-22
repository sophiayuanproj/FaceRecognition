"""Collect separately frozen train/calibration camera parents in twenty blocks."""
import datetime,json,random,time
from run_optical import ROOT,S,rpc,show
def main():
    parents=[json.loads(l) for l in (S/'data/manifests/optical_adaptation_parents.jsonl').read_text().splitlines()]
    key='camera-adaptation-'+datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%SZ');out=S/'runs/optical'/key;out.mkdir()
    blocks=[];pools={k:[r for r in parents if r['role']=='train' and r['label']==k] for k in ['K01','K02','K03','K04','K05']};cursor={k:0 for k in pools}
    other=[r for r in parents if r['role']=='other_train']
    for b in range(10):
        items=[]
        for k in pools:items.append(pools[k][cursor[k]]);cursor[k]+=1
        k=list(pools)[b%5];items.append(pools[k][cursor[k]]);cursor[k]+=1;items+=other[2*b:2*b+2]
        random.Random(55001+b).shuffle(items);blocks.append({'block':b,'phase':'train','seed':55001+b,'items':items})
    known=[r for r in parents if r['role']=='threshold_calibration'];random.Random(56000).shuffle(known);unknown=[r for r in parents if r['role']=='unknown_calibration'];j=0
    for b in range(10):
        n=3 if b<5 else 2;items=known[j:j+n]+unknown[3*b:3*b+3];j+=n;random.Random(56001+b).shuffle(items);blocks.append({'block':b+10,'phase':'calibration','seed':56001+b,'items':items})
    (out/'PROTOCOL.json').write_text(json.dumps({'schema':'v5-camera-parent-capture-1','session':key,'blocks':blocks,'future_final_used':False,
        'physical_scope':'one taped setup; twenty independently reset acquisition blocks; no independent environmental/session claim'},indent=2)+'\n')
    receipts=[];version=int(time.time()*1000)
    for block in blocks:
        reset=rpc('reset',{'pair_order':block['block']%2})
        for i,source in enumerate(block['items']):
            version+=1;display={'version':version,'kind':'photo','token':source['token'],'size_vmin':70,'opacity':.65,'background':'#777','x_pct':50,'y_pct':50}
            render=show(display);time.sleep(.7)
            request={'capture_id':'%s-B%02d-T%02d'%(key,block['block'],i),'session':key,'block':block['block'],'seed':block['seed'],
                'display':display,'render_ack':render,'source':source,'expected_truth':source['label'],'phase':block['phase'],'reset':reset,
                'evidence_tier':'actual_camera_role_preserving_adaptation','operator_setup':'fixed taped setup confirmed by user'}
            result=rpc('capture',request)
            if 'error' in result or result['metadata'].get('fomo_error') or result['metadata'].get('haar_error'):raise RuntimeError('Capture failure: '+str(result))
            receipts.append(result);(out/'RECEIPTS.json').write_text(json.dumps(receipts,indent=2,allow_nan=False)+'\n')
        print(json.dumps({'completed_block':block['block'],'phase':block['phase'],'actual_captures':len(receipts)}),flush=True)
    (out/'COMPLETE.json').write_text(json.dumps({'status':'captured_not_trained','captures':len(receipts),'known_train':60,'other_train':20,'known_calibration':25,'unknown_calibration':30,'blocks':20,'future_final_used':False},indent=2)+'\n')
    (S/'runs/optical/LATEST_ADAPTATION.txt').write_text(out.relative_to(ROOT).as_posix()+'\n');print(json.dumps({'complete':str(out)}),flush=True)
if __name__=='__main__':main()
