"""Host OpenMV5.0.1 RGB565 fixed-point normalization emulator."""
import base64,json,math,hashlib
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[2];S=ROOT/'work/project-research-v5'

def unpack(pixels):
    p=np.asarray(pixels,dtype=np.uint16).astype(np.int32);r=(p>>11)&31;g=(p>>5)&63;b=p&31
    return np.stack([(r<<3)|(r>>2),(g<<2)|(g>>4),(b<<3)|(b>>2)],axis=-1).astype(np.uint8)

def pack(rgb):
    a=np.asarray(rgb,dtype=np.uint8).astype(np.uint16)
    return ((a[:,:,0]>>3)<<11)|((a[:,:,1]>>2)<<5)|(a[:,:,2]>>3)

def native_rgb(pixels,roi=None,width=64,height=64):
    p=np.asarray(pixels,dtype=np.uint16);h,w=p.shape;rx,ry,rw,rh=roi or (0,0,w,h)
    if min(rx,ry)<0 or min(rw,rh)<=3 or rx+rw>w or ry+rh>h:raise ValueError('Invalid native RGB ROI')
    sx=np.float32(width/rw);sy=np.float32(height/rh);dx=math.floor(float(np.float32(65536)/sx));dy=math.floor(float(np.float32(65536)/sy))
    ax=math.floor(float(np.float32(math.floor(float(np.float32(rx)*sx))*65536)/sx))-32768
    ay=math.floor(float(np.float32(math.floor(float(np.float32(ry)*sy))*65536)/sy))-32768
    if dx==65536 and dy==65536:return unpack(p[ry:ry+rh,rx:rx+rw])
    xx=ax+np.arange(width,dtype=np.int64)*dx;yy=ay+np.arange(height,dtype=np.int64)*dy
    x0=np.clip(xx>>16,rx,rx+rw-1);x1=np.clip((xx>>16)+1,rx,rx+rw-1);y0=np.clip(yy>>16,ry,ry+rh-1);y1=np.clip((yy>>16)+1,ry,ry+rh-1)
    wx=((xx>>11)&31)[None,:];wy=((yy>>11)&31)[:,None];channels=[]
    for shift,mask in [(11,31),(5,63),(0,31)]:
        a=((p>>shift)&mask).astype(np.int32)
        left=(a[y0[:,None],x0[None,:]]*(32-wy)+a[y1[:,None],x0[None,:]]*wy+16)>>5
        right=(a[y0[:,None],x1[None,:]]*(32-wy)+a[y1[:,None],x1[None,:]]*wy+16)>>5
        channels.append(((left*(32-wx)+right*wx+16)>>5).astype(np.uint16))
    result=(channels[0]<<11)|(channels[1]<<5)|channels[2]
    rgb=unpack(result)
    # The runtime floors float32 scaled extents.  Some ROI widths/heights
    # therefore cover only63 of64 output pixels; BLACK_BACKGROUND clears the
    # uncovered last row/column.  This is native geometry, not interpolation
    # error, and is retained rather than hidden by a tolerance.
    covered_w=min(width,math.floor(float(np.float32(sx*np.float32(rw)))))
    covered_h=min(height,math.floor(float(np.float32(sy*np.float32(rh)))))
    rgb[:,covered_w:]=0;rgb[covered_h:]=0
    return rgb

def main():
    folder=S/'reports/M2/rgb_native_fixtures';rows=json.loads((folder/'RESULT.json').read_text())[0]['records'];meta={r['seed']:r for r in rows if r.get('kind')=='rgb_fixture'}
    results=[];out=S/'data/native_fixtures_rgb';out.mkdir(exist_ok=True)
    for line in (folder/'block_00.stdout.bin').read_bytes().splitlines():
        if not line.startswith(b'RGBFIX '):continue
        _,seed,data=line.split(b' ',2);seed=int(seed);expected=base64.b64decode(data,validate=True)
        yy,xx=np.indices((113,137));pixels=(((xx+seed)%32)<<11)|(((2*yy+seed)%64)<<5)|((xx*yy+seed)%32)
        actual=native_rgb(pixels,meta[seed]['roi']);ref=np.frombuffer(expected,np.uint8).reshape(64,64,3);delta=np.abs(actual.astype(int)-ref.astype(int))
        results.append({'seed':seed,'exact':bool(np.array_equal(actual,ref)),'differing_channels':int((delta!=0).sum()),'maximum_abs_error':int(delta.max())})
        (out/('seed_%02d_native_rgb64.u8'%seed)).write_bytes(expected)
    report={'schema':'v5-native-rgb565-contract-1','status':'verified_exact' if len(results)==10 and all(r['exact'] for r in results) else 'failed',
        'results':results,'native_r5_to8':meta[0]['r5_to8'],'native_g6_to8':[v for pair in zip(meta[0]['g6_even_to8'],meta[1]['g6_odd_to8']) for v in pair],
        'contract':'RGB565 little-endian, 16.16 positions, 5-bit bilinear weights, per-stage rounding, bit-replicated RGB888 output',
        'physical_scope':'ten generated device fixtures; not ten photos or independent environments'}
    (S/'reports/M2/RGB_NATIVE_CONTRACT.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n');print(json.dumps(report))
    if report['status']!='verified_exact':raise RuntimeError('Do not use unverified RGB transform for parity claims')
if __name__=='__main__':main()
