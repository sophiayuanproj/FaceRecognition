"""Host emulator for verified OpenMV5.0.1 GRAY8 Normalization subset.

Derived from the pinned upstream draw.c fixed-point bilinear path. RGB565,
mirroring, arbitrary affine and letterbox are deliberately separate contracts.
"""
import hashlib,json,math
from pathlib import Path
import numpy as np

def gray_native(source,roi=None,width=96,height=96):
    if source.ndim!=2 or source.dtype!=np.uint8:raise ValueError('GRAY8 image required')
    h,w=source.shape
    rx,ry,rw,rh=roi or (0,0,w,h)
    if min(rx,ry)<0 or min(rw,rh)<=3 or rx+rw>w or ry+rh>h:raise ValueError('Invalid ROI')
    # Firmware computes in float32, then uses 16.16 positions / 8-bit weights.
    sx=np.float32(width/rw);sy=np.float32(height/rh)
    dx=math.floor(float(np.float32(65536)/sx));dy=math.floor(float(np.float32(65536)/sy))
    ax=math.floor(float(np.float32(math.floor(float(np.float32(rx)*sx))*65536)/sx))-32768
    ay=math.floor(float(np.float32(math.floor(float(np.float32(ry)*sy))*65536)/sy))-32768
    # The equal-size optimization clears bilinear and uses nearest copy.
    if dx==65536 and dy==65536:return source[ry:ry+rh,rx:rx+rw].copy()
    xx=ax+np.arange(width,dtype=np.int64)*dx;yy=ay+np.arange(height,dtype=np.int64)*dy
    x0=np.clip(xx>>16,rx,rx+rw-1);x1=np.clip((xx>>16)+1,rx,rx+rw-1)
    y0=np.clip(yy>>16,ry,ry+rh-1);y1=np.clip((yy>>16)+1,ry,ry+rh-1)
    wx=((xx>>8)&255)[None,:];wy=((yy>>8)&255)[:,None]
    src=source.astype(np.int32)
    left=(src[y0[:,None],x0[None,:]]*(256-wy)+src[y1[:,None],x0[None,:]]*wy+128)>>8
    right=(src[y0[:,None],x1[None,:]]*(256-wy)+src[y1[:,None],x1[None,:]]*wy+128)>>8
    return ((left*(256-wx)+right*wx+128)>>8).astype(np.uint8)

def main():
    root=Path(__file__).resolve().parents[2];study=root/'work/project-research-v5'
    fixtures=study/'private/pixel-fixtures';result=[]
    for seed in range(10):
        meta=json.loads((fixtures/('pattern_%02d.json'%seed)).read_text())
        source=np.frombuffer((fixtures/('pattern_%02d_source.u8'%seed)).read_bytes(),np.uint8).reshape(meta['height'],meta['width'])
        expected=np.frombuffer((fixtures/('pattern_%02d_native.u8'%seed)).read_bytes(),np.uint8).reshape(96,96)
        actual=gray_native(source,meta['roi'])
        diff=np.abs(expected.astype(int)-actual.astype(int))
        result.append({'seed':seed,'exact':bool(np.array_equal(actual,expected)),'differing_pixels':int((diff!=0).sum()),'max_error':int(diff.max()),'mae':float(diff.mean()),'roi':meta['roi']})
    source=np.frombuffer((fixtures/'scene_gray240.u8').read_bytes(),np.uint8).reshape(240,240)
    expected=np.frombuffer((fixtures/'scene_gray96.u8').read_bytes(),np.uint8).reshape(96,96)
    actual=gray_native(source);diff=np.abs(expected.astype(int)-actual.astype(int))
    result.append({'fixture':'private_actual_scene','exact':bool(np.array_equal(actual,expected)),
                  'differing_pixels':int((diff!=0).sum()),'max_error':int(diff.max()),'mae':float(diff.mean())})
    out=study/'reports/M2/native_emulator.json';out.write_text(json.dumps({'subset':'GRAY8 bilinear SCALE_ASPECT_IGNORE CENTER, boundedROI',
        'source_draw_c_sha256':hashlib.sha256((study/'source_snapshots/upstream/draw.c').read_bytes()).hexdigest(),
        'status':'verified_exact' if all(r['exact'] for r in result) else 'failed',
        'results':result},indent=2,allow_nan=False)+'\n')
    print(out.read_text())

if __name__=='__main__':main()
