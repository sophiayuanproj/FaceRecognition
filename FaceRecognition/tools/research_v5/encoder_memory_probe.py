import gc,json,time,csi,ml,umalloc
def run(stage):
    gc.collect();print(json.dumps({'stage':stage,'kind':'before','free_gc':gc.mem_free(),'allocator':umalloc.stats(0)}))
    model=None
    try:
        t=time.ticks_us();model=ml.Model('/rom/face_encoder_int8.tflite')
        print(json.dumps({'stage':stage,'kind':'loaded','info':str(model),'load_us':time.ticks_diff(time.ticks_us(),t)}))
    except Exception as e:print(json.dumps({'stage':stage,'kind':'failure','error':repr(e)}))
    model=None;gc.collect()
run('no_camera')
cam=csi.CSI();cam.reset();cam.framesize(csi.QQVGA);cam.pixformat(csi.GRAYSCALE);cam.framebuffers(1);cam.snapshot(time=100)
run('small_camera_160x120')
cam.framesize(csi.QVGA);cam.window((240,240));cam.snapshot(time=100);frame=cam.snapshot()
run('camera_240x240')
print(json.dumps({'kind':'complete','block':BLOCK_INDEX}))
