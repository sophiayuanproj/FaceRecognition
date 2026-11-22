import csi,ml,gc,json,time,random,umalloc,hashlib,binascii
from ml.postprocessing.edgeimpulse import Fomo
def sha(path):
 h=hashlib.sha256()
 with open(path,'rb') as f:
  while True:
   b=f.read(2048)
   if not b:break
   h.update(b)
 gc.collect();return binascii.hexlify(h.digest()).decode()
cam=csi.CSI();cam.reset();cam.pixformat(csi.RGB565);cam.framesize(csi.QVGA);cam.window((240,240));cam.framebuffers(1);cam.snapshot(time=300);frame=cam.snapshot()
print(json.dumps({'kind':'camera','width':frame.width(),'height':frame.height(),'pixformat':frame.format(),'gc_free':gc.mem_free(),'allocator':umalloc.stats(0)}))
gc.collect();det=ml.Model('/rom/fomo_face_detection.tflite',postprocess=Fomo(threshold=.5));t=time.ticks_us();hits=det.predict([frame]);print(json.dumps({'kind':'detector','elapsed_us':time.ticks_diff(time.ticks_us(),t),'result':str(hits)}));det=None;hits=None;gc.collect()
paths=['/rom/operator_fused_encoder64.tflite','/rom/operator_fused_encoder56.tflite'];random.seed(54101+BLOCK_INDEX)
if random.getrandbits(1):paths.reverse()
for path in paths:
 model=None
 try:
  digest=sha(path);gc.collect();t=time.ticks_us();model=ml.Model(path);load=time.ticks_diff(time.ticks_us(),t)
  print(json.dumps({'kind':'loaded','path':path,'sha256':digest,'model':str(model),'load_us':load,'gc_free':gc.mem_free(),'allocator':umalloc.stats(0)}))
  t=time.ticks_us();output=model.predict([frame]);elapsed=time.ticks_diff(time.ticks_us(),t)
  vector=output[0][0];norm=sum(float(x)*float(x) for x in vector)**.5
  print(json.dumps({'kind':'encoder_inference','path':path,'elapsed_us':elapsed,'output_dimension':len(vector),'norm':norm,'identity_qualified':False,'input_contract':'default native transform; resource trial only, exact model LUT pending'}))
  output=None;vector=None
 except Exception as e:print(json.dumps({'kind':'failure','path':path,'error':repr(e)}))
 model=None;gc.collect();print(json.dumps({'kind':'unloaded','path':path,'gc_free':gc.mem_free(),'allocator':umalloc.stats(0)}))
print(json.dumps({'kind':'complete','block':BLOCK_INDEX}))
