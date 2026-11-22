import ml,gc,json,binascii
def flat(value):
 b=bytearray();stack=[value]
 while stack:
  x=stack.pop()
  if hasattr(x,'shape') and len(x.shape)>0:
   for i in range(x.shape[0]-1,-1,-1):stack.append(x[i])
  else:b.append(int(x)&255)
 return b
for name in ['prelu_3d.tflite','prelu_1d.tflite']:
 model=None
 try:
  gc.collect();model=ml.Model('/flash/research-v5-prelu/'+name)
  print(json.dumps({'kind':'loaded','model':name,'info':str(model)}))
  for seed in range(10):
   def fill(buffer,shape,dtype):
    for i in range(48):buffer[i]=(i*13+seed*23)%256
   def raw(m,inputs,outputs):return outputs
   out=model.predict([fill],callback=raw);print(json.dumps({'kind':'arithmetic','model':name,'seed':seed,'output_hex':binascii.hexlify(flat(out[0])).decode()}))
 except Exception as e:print(json.dumps({'kind':'failure','model':name,'error':repr(e)}))
 model=None;gc.collect()
print(json.dumps({'kind':'complete'}))
