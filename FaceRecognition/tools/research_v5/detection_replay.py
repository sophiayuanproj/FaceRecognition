import json,gc,time,csi,image,ml,umalloc,random
from ml.postprocessing.edgeimpulse import Fomo
base='/flash/research-v5-detection'
with open(base+'/index.json') as f:manifest=json.load(f)
cam=csi.CSI();cam.reset();cam.framesize(csi.QVGA);cam.pixformat(csi.GRAYSCALE);cam.window((240,240));cam.framebuffers(1)
random.seed(54001+BLOCK_INDEX)
order=['fomo','haar']
if random.getrandbits(1):order.reverse()
for name in order:
    model=None;cascade=None;frame=None;gc.collect()
    if name=='fomo':model=ml.Model('/rom/fomo_face_detection.tflite',postprocess=Fomo(threshold=.5))
    else:cascade=image.HaarCascade('/rom/haarcascade_frontalface.cascade')
    for fixture in manifest:
        frame=image.Image(base+'/'+fixture['file'],copy_to_fb=True)
        start=time.ticks_us()
        if name=='fomo':result=model.predict([frame])
        else:result=frame.find_features(cascade,threshold=.6,scale=1.25)
        elapsed=time.ticks_diff(time.ticks_us(),start)
        print(json.dumps({'kind':'detection_replay','detector':name,'block':BLOCK_INDEX,
            'fixture_id':fixture['fixture_id'],'elapsed_us':elapsed,'output':str(result),
            'gc_free':gc.mem_free(),'allocator':umalloc.stats(0),'evidence_tier':'source/generated replay, not optical'}))
        frame=None;gc.collect()
    model=None;cascade=None;gc.collect()
print(json.dumps({'kind':'complete','block':BLOCK_INDEX}))
