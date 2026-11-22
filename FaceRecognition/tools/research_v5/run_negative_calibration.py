"""Acquire new calibration-only empty/checker controls in ten warm blocks."""
import datetime,json,random,time
from run_optical import ROOT,S,rpc,show

def main():
    key='negative-calibration-'+datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    out=S/'runs/optical'/key;out.mkdir()
    protocol={'schema':'v5-negative-calibration-1','blocks':10,'seed_start':57001,
        'stimuli':['blank','calibration'],'roles':'threshold_calibration only',
        'prospective_use':'detector threshold/crop development; never recognition final evaluation',
        'limitations':'two generated negative categories, one fixed physical setup; not representative clutter'}
    (out/'PROTOCOL.json').write_text(json.dumps(protocol,indent=2)+'\n')
    receipts=[];version=int(time.time()*1000)
    for b in range(10):
        reset=rpc('reset',{'pair_order':b%2});items=['blank','calibration'];random.Random(57001+b).shuffle(items)
        for i,kind in enumerate(items):
            version+=1;display={'version':version,'kind':kind,'size_vmin':70,'opacity':.65,'background':'#777','x_pct':50,'y_pct':50}
            ack=show(display);time.sleep(.7)
            req={'capture_id':'%s-B%02d-T%02d'%(key,b,i),'session':key,'block':b,'seed':57001+b,
                'display':display,'render_ack':ack,'source':{},'expected_truth':'NO_FACE','phase':'threshold_calibration',
                'reset':reset,'evidence_tier':'actual_camera_negative_calibration','operator_setup':'fixed taped user-confirmed setup'}
            r=rpc('capture',req)
            if 'error' in r or r['metadata'].get('fomo_error') or r['metadata'].get('haar_error'):raise RuntimeError(str(r))
            receipts.append(r);(out/'RECEIPTS.json').write_text(json.dumps(receipts,indent=2,allow_nan=False)+'\n')
        print(json.dumps({'completed_block':b,'negative_calibration_captures':len(receipts)}),flush=True)
    (out/'COMPLETE.json').write_text(json.dumps({'captures':20,'physical_reset_blocks':10,'future_final_used':False},indent=2)+'\n')
    (S/'runs/optical/LATEST_NEGATIVE_CALIBRATION.txt').write_text(out.relative_to(ROOT).as_posix()+'\n')

if __name__=='__main__':main()
