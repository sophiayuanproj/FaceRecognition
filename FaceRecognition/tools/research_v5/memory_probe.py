import gc,json,time,csi,ml,umalloc,random,os
def state(stage):
    gc.collect();print(json.dumps({'kind':'memory','stage':stage,'gc_free':gc.mem_free(),'allocator':umalloc.stats(0)}))
state('idle')
cam=csi.CSI();cam.reset();cam.pixformat(csi.GRAYSCALE);cam.framesize(csi.QVGA);cam.window((240,240));cam.framebuffers(1);cam.snapshot(time=500)
frame=cam.snapshot();state('camera_and_frame')
paths=['/rom/v5_small.tflite','/rom/v4_large.tflite','/flash/research-v5/v5_small.tflite','/flash/research-v5/v4_large.tflite']
random.seed(54001+BLOCK_INDEX)
for i in range(len(paths)-1,0,-1):
    j=random.getrandbits(16)%(i+1);paths[i],paths[j]=paths[j],paths[i]
for path in paths:
    model=None
    try:
        state('before:'+path)
        start=time.ticks_us();model=ml.Model(path);load=time.ticks_diff(time.ticks_us(),start)
        print(json.dumps({'kind':'loaded','path':path,'load_us':load,'info':str(model)}));state('loaded:'+path)
        start=time.ticks_us();outputs=model.predict([frame]);duration=time.ticks_diff(time.ticks_us(),start)
        print(json.dumps({'kind':'inference','path':path,'duration_us':duration,'outputs':str(outputs),'semantic_qualification':False}))
        outputs=None
    except Exception as e:
        print(json.dumps({'kind':'error','path':path,'error':repr(e)}))
    model=None;gc.collect();state('unloaded:'+path)
for action in ['sleep','wake']:
    try:
        cam.sleep(action=='sleep')
        if action=='wake':cam.snapshot(time=200);cam.snapshot()
        state(action)
    except Exception as e:print(json.dumps({'kind':'sleep_error','action':action,'error':repr(e)}))
print(json.dumps({'kind':'complete','block':BLOCK_INDEX,'free_flash':os.statvfs('/flash')}))
