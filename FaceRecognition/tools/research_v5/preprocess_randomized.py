import gc,json,time,csi,sys,random
from ml.preprocessing import Normalization
sys.path.insert(0,'/flash/releases/face-v1-a13bbcac1ccd-usb')
from identity_contracts import pillow_bilinear_model_input
random.seed(54001+BLOCK_INDEX)
cam=csi.CSI();cam.reset();cam.pixformat(csi.GRAYSCALE);cam.framesize(csi.QVGA);cam.window((240,240));cam.framebuffers(1);cam.snapshot(time=300);frame=cam.snapshot()
order=['native','legacy']
if random.getrandbits(1):order.reverse()
for name in order:
    out=bytearray(9216);gc.collect();t=time.ticks_us()
    if name=='native':Normalization()(frame)(out,(1,96,96,1),ord('B'))
    else:pillow_bilinear_model_input(frame)(out,(1,96,96,1),ord('B'))
    elapsed=time.ticks_diff(time.ticks_us(),t)
    print(json.dumps({'kind':'paired_preprocess','method':name,'elapsed_us':elapsed,'order':order,'block':BLOCK_INDEX,'sum':sum(out),
        'scene':'same captured uncontrolled stationary bench frame within block; not labeled optical recognition'}))
    del out
print(json.dumps({'kind':'complete','block':BLOCK_INDEX}))
