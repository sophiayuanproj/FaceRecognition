# Exact image/tensor fixtures on Nicla. Captured scene bytes are private.
import gc,json,os,time,csi,image,binascii
from ml.preprocessing import Normalization
try:os.mkdir('/flash/research-v5-pixels')
except OSError:pass
def save(name,payload):
    with open('/flash/research-v5-pixels/'+name,'wb') as f:f.write(payload)
def metadata(name,obj):save(name,json.dumps(obj).encode())
cam=csi.CSI();cam.reset();cam.pixformat(csi.GRAYSCALE);cam.framesize(csi.QVGA);cam.window((240,240));cam.framebuffers(1);cam.snapshot(time=600)
frame=cam.snapshot();save('scene_gray240.u8',frame.bytearray())
buf=bytearray(96*96);Normalization()(frame)(buf,(1,96,96,1),ord('B'));save('scene_gray96.u8',buf)
metadata('scene.json',{'width':frame.width(),'height':frame.height(),'format':frame.format(),'block':BLOCK_INDEX,
    'context':'Uncontrolled actual sensor scene; private; not labeled optical recognition or range evidence',
    'stride':240,'gray_values_are_not_lux':True})
print(json.dumps({'kind':'saved_private_sensor_fixture','files':3,'block':BLOCK_INDEX}))
# Generated patterns are tensor diagnostics, never physical scene observations.
for seed in range(10):
    pat=image.Image(137,113,image.GRAYSCALE)
    raw=pat.bytearray()
    for y in range(113):
        for x in range(137):raw[y*137+x]=(x*17+y*31+seed*23+(x*y)%29)%256
    roi=(seed%5,seed%7,127-seed,103-seed)
    buf=bytearray(96*96);Normalization(roi=roi)(pat)(buf,(1,96,96,1),ord('B'))
    save('pattern_%02d_source.u8'%seed,raw);save('pattern_%02d_native.u8'%seed,buf)
    metadata('pattern_%02d.json'%seed,{'seed':seed,'width':137,'height':113,'roi':roi,'out':[96,96]})
    del raw,pat,buf;gc.collect()
print(json.dumps({'kind':'generated_fixture_complete','seeds':10}))
