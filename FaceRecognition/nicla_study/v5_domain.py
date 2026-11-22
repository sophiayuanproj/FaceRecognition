"""Bounded original-frame ChokePoint domain/geometry screen, no role fitting."""
from collections import defaultdict,Counter
import hashlib,json,math,tarfile,xml.etree.ElementTree as E
from pathlib import Path
import cv2,numpy as np
from PIL import Image,ImageDraw

ROOT=Path(__file__).resolve().parents[2];STUDY=ROOT/'work/project-research-v5'

def main():
    out=STUDY/'data/chokepoint_development';out.mkdir(parents=True,exist_ok=True)
    raw=STUDY/'data/raw';by=defaultdict(list)
    with tarfile.open(raw/'chokepoint_groundtruth.tar.xz') as a:
        root=E.fromstring(a.extractfile('groundtruth/P1E_S1_C1.xml').read())
    for frame in root:
        for person in frame:
            eyes=[]
            for eye in person:
                eyes.append([float(eye.attrib['x']),float(eye.attrib['y'])])
            if len(eyes)!=2:continue
            by[person.attrib['id']].append({'frame':frame.attrib['number'],'identity':person.attrib['id'],
                'eyes':eyes,'eye_separation_native_px':math.dist(eyes[0],eyes[1]),
                'event_group':'P1E_S1:'+person.attrib['id'],'view':'C1','role':'secondary_domain_development_only'})
    chosen=[]
    for identity,rows in sorted(by.items()):
        rows.sort(key=lambda x:x['eye_separation_native_px'])
        chosen.extend(rows[int((len(rows)-1)*q)] for q in [.1,.5,.9])
    targets={r['frame'] for r in chosen};found={}
    archive=raw/'chokepoint_P1E_S1_C1.tar.xz'
    with tarfile.open(archive) as a:
        for entry in a:
            if not entry.isfile() or not entry.name.endswith('.jpg'):continue
            stem=Path(entry.name).stem
            if stem not in targets:continue
            data=a.extractfile(entry).read();dest=out/(stem+'.jpg');dest.write_bytes(data)
            im=cv2.imdecode(np.frombuffer(data,np.uint8),cv2.IMREAD_COLOR)
            found[stem]={'path':dest.relative_to(ROOT).as_posix(),'sha256':hashlib.sha256(data).hexdigest(),
                         'width':im.shape[1],'height':im.shape[0]}
    for r in chosen:
        if r['frame'] not in found:raise ValueError('Ground truth frame absent')
        r.update(found[r['frame']])
        r['eye_separation_fullscene96_px']=r['eye_separation_native_px']*96/r['width']
    (out/'MANIFEST.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in chosen))
    tiles=[]
    for r in chosen[1::3]:
        im=Image.open(ROOT/r['path']).convert('RGB');x=np.array(r['eyes']).mean(0);side=max(32,int(r['eye_separation_native_px']*4))
        left=max(0,int(x[0]-side/2));top=max(0,int(x[1]-side*.4))
        crop=im.crop((left,top,min(im.width,left+side),min(im.height,top+side))).resize((96,96))
        tile=Image.new('RGB',(200,145),'white');im.thumbnail((100,100));tile.paste(im,(0,20));tile.paste(crop,(102,20));
        ImageDraw.Draw(tile).text((3,123),'ID'+r['identity']+' eye %.1fpx'%r['eye_separation_native_px'],fill='black');tiles.append(tile)
    sheet=Image.new('RGB',(1000,145*5),(215,215,215))
    for i,tile in enumerate(tiles):sheet.paste(tile,((i%5)*200,(i//5)*145))
    sheet.save(out/'CONTACT_SHEET.jpg')
    vals=np.array([r['eye_separation_native_px'] for rows in by.values() for r in rows])
    result={'source_event':'P1E_S1','views_available':3,'views_sampled':1,'sampled_identities':len(by),
        'annotated_face_frames':len(vals),'sampled_frames':len(found),'selected_face_rows':len(chosen),
        'eye_separation_native_percentiles':dict(zip(['p10','p50','p90'],np.percentile(vals,[10,50,90]).tolist())),
        'grouping':'All same-person frames and synchronized views in this walk belong together; no independent-session recognition claim.',
        'suitability':'Secondary full-frame localization/face-size challenge. 25 people in this event cannot supply five known plus90 disjoint calibration/development/future unknown identities.',
        'terms':'ChokePoint noncommercial scientific research; retain publisher README/acknowledgment, images remain source/derived research data.',
        'native_camera_equivalence':False,'future_final_used':False}
    (out/'RESULT.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n');print(json.dumps(result))

if __name__=='__main__':main()
