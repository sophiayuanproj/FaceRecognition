# Executed ON the Nicla by board_lab.py, not by desktop Python.
import gc, json, os, sys, time
import csi, image, ml, umalloc
from ml.preprocessing import Normalization

def state(stage):
    gc.collect()
    print(json.dumps({'kind':'memory', 'stage':stage, 'gc_free':gc.mem_free(),
                      'allocator':umalloc.stats(0)}))

state('idle')
cam=csi.CSI()
cam.reset()
print(json.dumps({'kind':'camera', 'sensor_id':cam.cid(), 'api':dir(cam)}))
cam.pixformat(csi.GRAYSCALE)
cam.framesize(csi.QVGA)
cam.window((240,240))
cam.framebuffers(1)
cam.snapshot(time=1000)
state('camera_configured')
t=time.ticks_us(); frame=cam.snapshot(); capture=time.ticks_diff(time.ticks_us(),t)
print(json.dumps({'kind':'capture','width':frame.width(),'height':frame.height(),
                  'format':frame.format(),'capture_us':capture}))
state('captured')
buf=bytearray(96*96)
t=time.ticks_us(); Normalization()(frame)(buf,(1,96,96,1),ord('B')); native=time.ticks_diff(time.ticks_us(),t)
print(json.dumps({'kind':'preprocess','method':'native_normalization','elapsed_us':native,
                  'min':min(buf),'max':max(buf),'sum':sum(buf)}))
sys.path.insert(0,'/flash/releases/face-v1-a13bbcac1ccd-usb')
from identity_contracts import pillow_bilinear_model_input
old=bytearray(96*96)
t=time.ticks_us(); pillow_bilinear_model_input(frame)(old,(1,96,96,1),ord('B')); legacy=time.ticks_diff(time.ticks_us(),t)
print(json.dumps({'kind':'preprocess','method':'legacy_python','elapsed_us':legacy,
                  'min':min(old),'max':max(old),'sum':sum(old),
                  'mean_abs_difference':sum(abs(a-b) for a,b in zip(buf,old))/len(buf)}))
del old,buf
state('after_preprocessing')
from ml.postprocessing.edgeimpulse import Fomo
for name,post in [('fomo_face_detection', Fomo(threshold=.5))]:
    try:
        t=time.ticks_us(); model=ml.Model('/rom/'+name+'.tflite',postprocess=post)
        print(json.dumps({'kind':'model','name':name,'info':str(model),
                          'load_us':time.ticks_diff(time.ticks_us(),t)}))
        state(name+'_loaded')
        t=time.ticks_us(); result=model.predict([frame]); duration=time.ticks_diff(time.ticks_us(),t)
        print(json.dumps({'kind':'inference','name':name,'elapsed_us':duration,'output':str(result)}))
        del model,result
        state(name+'_unloaded')
    except Exception as error:
        print(json.dumps({'kind':'model_error','name':name,'error':repr(error)}))
try:
    cascade=image.HaarCascade('/rom/haarcascade_frontalface.cascade')
    t=time.ticks_us(); result=frame.find_features(cascade,threshold=.6,scale=1.25)
    print(json.dumps({'kind':'inference','name':'haar','elapsed_us':time.ticks_diff(time.ticks_us(),t),'output':result}))
except Exception as error:
    print(json.dumps({'kind':'model_error','name':'haar','error':repr(error)}))
try:
    from ml.postprocessing.mediapipe import BlazeFace
    model=ml.Model('/rom/blazeface_front_128.tflite',postprocess=BlazeFace(threshold=.5))
    print(json.dumps({'kind':'model','name':'blazeface','info':str(model)}))
    state('blazeface_loaded')
    t=time.ticks_us(); result=model.predict([frame])
    print(json.dumps({'kind':'inference','name':'blazeface','elapsed_us':time.ticks_diff(time.ticks_us(),t),'output':str(result)}))
except Exception as error:
    print(json.dumps({'kind':'model_error','name':'blazeface','error':repr(error)}))
state('end')
