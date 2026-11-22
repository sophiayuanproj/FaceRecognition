"""Public development-only sensor-size replay fixtures, explicitly non-optical."""
import hashlib,json
from pathlib import Path
import cv2,numpy as np
ROOT=Path(__file__).resolve().parents[2];STUDY=ROOT/'work/project-research-v5'

def main():
    dest=STUDY/'staging/detection_fixtures';dest.mkdir(exist_ok=True)
    rows=[json.loads(l) for l in (STUDY/'data/manifests/lfw_roles.jsonl').read_text().splitlines()]
    chosen=[]
    for label in ['K01','K02','K03','K04','K05']:
        chosen.extend([r for r in rows if r['label']==label and r['role']=='stopping'][:3])
    ch=[json.loads(l) for l in (STUDY/'data/chokepoint_development/MANIFEST.jsonl').read_text().splitlines()]
    chosen.extend(ch[:15]);manifest=[]
    for i,row in enumerate(chosen):
        source=ROOT/row['path']
        if hashlib.sha256(source.read_bytes()).hexdigest()!=row['sha256']:raise ValueError('Source changed')
        image=cv2.imread(str(source),cv2.IMREAD_GRAYSCALE)
        # Fullscene aspect is letterboxed, retaining source geometry explicitly.
        h,w=image.shape;scale=min(240/w,240/h);rw,rh=round(w*scale),round(h*scale)
        frame=np.zeros((240,240),np.uint8);ox,oy=(240-rw)//2,(240-rh)//2
        frame[oy:oy+rh,ox:ox+rw]=cv2.resize(image,(rw,rh))
        name='frame_%02d.pgm'%i;cv2.imwrite(str(dest/name),frame)
        m={'fixture_id':'source-replay-%02d'%i,'file':name,'source_id':row.get('record_id',row.get('frame')),
            'source_sha256':row['sha256'],'role':row['role'],'source_width':w,'source_height':h,
            'scale':scale,'offset':[ox,oy],'evidence_tier':'source_image_replay_not_sensor',
            'file_sha256':hashlib.sha256((dest/name).read_bytes()).hexdigest()}
        if 'eyes' in row:
            m['eye_points_240']=[[x*scale+ox,y*scale+oy] for x,y in row['eyes']]
            m['eye_separation_240']=row['eye_separation_native_px']*scale
        manifest.append(m)
    for i,value in enumerate([0,64,128,255],start=len(manifest)):
        frame=np.full((240,240),value,np.uint8);name='frame_%02d.pgm'%i;cv2.imwrite(str(dest/name),frame)
        manifest.append({'fixture_id':'uniform-negative-%d'%value,'file':name,'role':'generated_development_control',
            'negative_kind':'uniform, not representative natural scene','evidence_tier':'generated_replay',
            'file_sha256':hashlib.sha256((dest/name).read_bytes()).hexdigest()})
    (dest/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(json.dumps({'fixtures':len(manifest),'future_final_used':False}))

if __name__=='__main__':main()
