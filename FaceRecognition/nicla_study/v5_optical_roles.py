"""Prospective camera-adaptation parents; no development/final training leakage."""
import json,random,hashlib
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[2];S=ROOT/'work/project-research-v5'
def main():
    idx=json.loads((S/'runs/host_reference/nonfinal_index.json').read_text());detected=np.load(S/'runs/host_reference/nonfinal_features.npz')['detected']
    usable=[r for i,r in enumerate(idx['rows']) if detected[i]];rng=random.Random(55100);selected=[]
    for role,n in [('train',12),('threshold_calibration',5)]:
        for k in ['K01','K02','K03','K04','K05']:
            pool=[r for r in usable if r['role']==role and r['label']==k];chosen=rng.sample(pool,n)
            selected.extend(dict(r,camera_role='camera_'+role) for r in chosen)
    for role,n in [('other_train',20),('unknown_calibration',30)]:
        by={}
        for r in usable:
            if r['role']==role:by.setdefault(r['source_identity'],[]).append(r)
        ids=sorted(by);rng.shuffle(ids)
        if len(ids)<n:raise ValueError('Insufficient separate unknown IDs')
        selected.extend(dict(rng.choice(by[identity]),camera_role='camera_'+role) for identity in ids[:n])
    if any(r['role'] in ['development','unknown_development'] or 'final' in r['role'] for r in selected):raise ValueError('Role leakage')
    for r in selected:r['token']=hashlib.sha256(r['record_id'].encode()).hexdigest()[:24]
    manifest=S/'data/manifests/optical_adaptation_parents.jsonl'
    if manifest.exists():raise ValueError('Do not silently replace frozen camera parent roles')
    manifest.write_text(''.join(json.dumps(r)+'\n' for r in selected))
    protocol={'schema':'camera-adaptation-parent-freeze-1','known_train':60,'other_train':20,'known_calibration':25,'unknown_calibration':30,
        'selection_seed':55100,'selection':'frozen source roles, strict-single-face source quality only, no recognition score selection',
        'source_groups_distinct':len({r['source_group'] for r in selected}),
        'manifest_sha256':hashlib.sha256(manifest.read_bytes()).hexdigest(),'future_final_used':False,
        'limits':'Same taped physical session as development; not independent-environment validation. Recaptures inherit parent role.'}
    (S/'data/OPTICAL_ADAPTATION_FREEZE.json').write_text(json.dumps(protocol,indent=2)+'\n')
    p=S/'runs/optical/DISPLAY_SOURCES.json';old=json.loads(p.read_text());merged={r['record_id']:r for r in old['items']}
    for r in selected:merged[r['record_id']]=r
    p.write_text(json.dumps({'items':list(merged.values()),'future_final_used':False},indent=2)+'\n');print(json.dumps(protocol))
if __name__=='__main__':main()
