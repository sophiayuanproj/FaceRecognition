"""Ten seeded loader/role/augmentation checks, with nonfinal contact sheet."""
from collections import defaultdict,Counter
import hashlib,json
from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw

ROOT=Path(__file__).resolve().parents[2];STUDY=ROOT/'work/project-research-v5'

def main():
    manifest=STUDY/'data/manifests/lfw_roles.jsonl'
    rows=[json.loads(l) for l in manifest.read_text().splitlines()]
    seen={};unknown=defaultdict(set);known=set()
    for row in rows:
        group=row['source_group']
        if group in seen:raise ValueError('Duplicate or cross-role source group')
        seen[group]=row['role']
        if row['known']:known.add(row['source_identity'])
        else:unknown[row['role']].add(row['source_identity'])
    names=sorted(unknown)
    for i,name in enumerate(names):
        if known & unknown[name]:raise ValueError('Known/unknown identity overlap')
        for other in names[i+1:]:
            if unknown[name] & unknown[other]:raise ValueError('Unknown identity leakage')
    index=json.loads((STUDY/'runs/host_reference/nonfinal_index.json').read_text())
    cache=STUDY/'runs/host_reference/nonfinal_features.npz'
    with np.load(cache) as archive:
        data={k:archive[k] for k in ['detected','gray96']}
    if hashlib.sha256(cache.read_bytes()).hexdigest()!=index['cache_sha256']:raise ValueError('Cache hash mismatch')
    if any('future_final' in r['role'] for r in index['rows']):raise ValueError('Final cache leak')
    if data['gray96'].dtype!=np.uint8 or data['gray96'].shape[1:]!=(96,96,1):raise ValueError('Tensor contract')
    train=[i for i,r in enumerate(index['rows']) if r['role'] in ['train','other_train'] and data['detected'][i]]
    checks=[]
    for seed in range(53001,53011):
        rng=np.random.default_rng(seed);sample=rng.choice(train,24,replace=False)
        # Augmentations retain their immutable parent role, never create a new split.
        children=[]
        for i in sample:
            row=index['rows'][i];x=data['gray96'][i].astype(np.float32)/255
            y=np.clip(x*rng.uniform(.85,1.15)+rng.uniform(-.03,.03),0,1)
            if not np.isfinite(y).all():raise ValueError('Nonfinite augmentation')
            if row['role'] not in ['train','other_train']:raise ValueError('Augmentation role leakage')
            if hashlib.sha256((ROOT/row['path']).read_bytes()).hexdigest()!=row['sha256']:raise ValueError('Source mismatch')
            children.append({'parent':row['record_id'],'role':row['role'],'min':float(y.min()),'max':float(y.max())})
        checks.append({'seed':seed,'status':'passed','sampled_source_hashes':len(sample),'augmentation_parents':children})
    out=STUDY/'reports/M3';out.mkdir(exist_ok=True)
    tiles=[]
    for k in ['K01','K02','K03','K04','K05','OTHER']:
        indices=[i for i,r in enumerate(index['rows']) if r['label']==k and r['role'] in ['stopping','unknown_development'] and data['detected'][i]][:10]
        for i in indices:
            tile=Image.new('RGB',(100,118),'white');tile.paste(Image.fromarray(data['gray96'][i,:,:,0]).convert('RGB'),(2,0))
            ImageDraw.Draw(tile).text((3,100),k,fill='black');tiles.append(tile)
    sheet=Image.new('RGB',(1000,6*118),'#dddddd')
    for i,tile in enumerate(tiles):sheet.paste(tile,((i%10)*100,(i//10)*118))
    sheet.save(out/'LFW_GRAY96_CONTACT_SHEET.jpg')
    result={'status':'passed','fault_check_roots':10,'source_groups':len(seen),'role_counts':dict(Counter(r['role'] for r in rows)),
        'unknown_identities_per_role':{k:len(v) for k,v in unknown.items()},'checks':checks,
        'pixels_opened_for_validation':'nonfinal cache + train source hashes only; final role metadata checked for disjointness',
        'final_model_inference':False}
    (out/'DATA_CHECKS.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k!='checks'}))

if __name__=='__main__':main()
