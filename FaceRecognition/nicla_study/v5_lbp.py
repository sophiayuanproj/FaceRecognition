"""Native-compatible classical face histogram baseline with ten fitted galleries."""
import json,re,hashlib
from pathlib import Path
import numpy as np
from .v5_native import gray_native
ROOT=Path(__file__).resolve().parents[2];S=ROOT/'work/project-research-v5'
source=(S/'source_snapshots/upstream/lbp.c').read_text()
TABLE=np.array([int(x) for x in re.search(r'uniform_tbl\[256\] = \{(.*?)\};',source,re.S).group(1).replace('\n','').split(',') if x.strip()],np.uint8)
WEIGHTS=np.array([int(x) for x in re.search(r'lbp_weights\[49\] = \{(.*?)\};',source,re.S).group(1).replace('\n','').split(',') if x.strip()],np.int32)

def descriptor(im):
    if im.shape!=(91,91) or im.dtype!=np.uint8:raise ValueError('Exactly91x91 prevents non-divisible region overrun in upstream implementation')
    p=im[1:89,1:89];lbp=np.zeros((88,88),np.uint8)
    for bit,(y,x) in enumerate([(0,0),(0,1),(0,2),(1,2),(2,2),(2,1),(2,0),(1,0)]):
        lbp|=(im[y:y+88,x:x+88]>=p).astype(np.uint8)<<bit
    yy,xx=np.indices((88,88));regions=(yy//13)*7+xx//13
    hist=np.bincount((regions*59+TABLE[lbp]).ravel(),minlength=49*59)
    if hist.max()>255:raise ValueError('Native byte histogram would overflow')
    return hist.astype(np.uint8)

def distance(a,b):
    a=a.astype(np.int32);b=b.astype(np.int32)
    return (((a-b)**2//np.maximum(a+b,1))*np.repeat(WEIGHTS,59)).sum(axis=-1)

def main():
    out=S/'runs/lbp_native91';out.mkdir(exist_ok=True)
    idx=json.loads((S/'runs/host_reference/nonfinal_index.json').read_text());rows=idx['rows']
    with np.load(S/'runs/host_reference/nonfinal_features.npz') as ar:gray=ar['gray96'];detected=ar['detected']
    roles=np.array([r['role'] for r in rows]);labels=np.array([int(r['label'][1:])-1 if r['known'] else -1 for r in rows])
    desc=np.zeros((len(rows),49*59),np.uint8)
    for i in np.flatnonzero(detected):desc[i]=descriptor(gray_native(gray[i,:,:,0],width=91,height=91))
    np.save(out/'descriptors.npy',desc)
    results=[]
    for seed in range(53001,53011):
        rng=np.random.default_rng(seed);gallery=[]
        for k in range(5):
            ids=np.flatnonzero((roles=='train') & (labels==k) & detected)
            gallery.append(np.rint(np.median(desc[rng.choice(ids,10,replace=False)],axis=0)).astype(np.uint8))
        gallery=np.array(gallery);d=np.stack([distance(desc,g) for g in gallery],axis=1)
        confidence=-d.min(1);pred=d.argmin(1);cal=(roles=='unknown_calibration') & detected
        threshold=int(confidence[cal].max())+1;accept=detected & (confidence>=threshold)
        known=roles=='development';unknown=roles=='unknown_development'
        result={'seed':seed,'replication':'ten actually fitted sampled descriptor galleries, no neural gradient training',
            'known_correct_acceptance':float((accept & (pred==labels))[known].mean()),'known_coverage':float(accept[known].mean()),
            'known_top1_full':float((detected & (pred==labels))[known].mean()),'unknown_false':int((accept & unknown).sum()),'unknown_n':300,
            'threshold_negative_distance':threshold,'gallery_bytes':int(gallery.nbytes),'physical_qualified':False}
        results.append(result);np.save(out/('gallery_%d.npy'%seed),gallery)
    (out/'ROOT_RESULTS.json').write_text(json.dumps(results,indent=2,allow_nan=False)+'\n')
    (out/'PROTOCOL.json').write_text(json.dumps({'source':'https://github.com/openmv/openmv/blob/v5.0.1/lib/imlib/lbp.c',
        'source_sha256':hashlib.sha256(source.encode()).hexdigest(),'input':'GRAY91 safe divisible-by7 grid; native-bilinear box source',
        'metric':'upstream weighted integer chi-square','gallery':'median10 training descriptors/class, ten sample seeds',
        'safety':'Use91x91. Current C implementation requires safe region divisions; arbitrary96x96 can exceed49region indices.',
        'future_final_used':False},indent=2)+'\n')
    print(json.dumps({'roots':10,'known_mean':float(np.mean([r['known_correct_acceptance'] for r in results])),
        'known_top1_mean':float(np.mean([r['known_top1_full'] for r in results]))}))

if __name__=='__main__':main()
