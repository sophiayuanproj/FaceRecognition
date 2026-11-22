"""Single-owner live preview + exact-pixel optical experiment RPC on localhost."""
import argparse,base64,hashlib,json,queue,re,threading,time,os
from pathlib import Path
from http.server import BaseHTTPRequestHandler,ThreadingHTTPServer
from board_lab import connect
from live_camera import PAGE

PROGRAM='''import csi,pyb,time,gc,json,binascii,image,ml,umalloc
from ml.preprocessing import Normalization
from ml.postprocessing.edgeimpulse import Fomo
cam=csi.CSI();cam.reset();cam.pixformat(csi.RGB565);cam.framesize(csi.QVGA);cam.window((240,240));cam.framebuffers(1);cam.auto_exposure(True);cam.snapshot(time=600)
def dump(name,data):
 for i in range(0,len(data),2048):print('DATA '+name+' '+binascii.b2a_base64(memoryview(data)[i:i+2048]).decode().strip())
 gc.collect()
def flatten(value):
 out=bytearray();stack=[value]
 while stack:
  item=stack.pop()
  if hasattr(item,'shape') and len(item.shape)>0:
   for i in range(item.shape[0]-1,-1,-1):stack.append(item[i])
  else:out.append(int(item)&255)
 return out
PAIR_ORDER=0
def fomo_run(frame,meta):
 try:
  gc.collect();meta['fomo_before_gc_free']=gc.mem_free();start=time.ticks_us();model=ml.Model('/rom/fomo_face_detection.tflite',postprocess=Fomo(threshold=.5));meta['fomo_load_us']=time.ticks_diff(time.ticks_us(),start);meta['fomo_model']=str(model);meta['fomo_loaded_gc_free']=gc.mem_free()
  start=time.ticks_us();result=model.predict([frame]);meta['fomo_predict_us']=time.ticks_diff(time.ticks_us(),start);meta['fomo']=str(result)
  def bound(buffer,shape,dtype):
   Normalization()(frame)(buffer,shape,dtype);dump('fomo_input_i8',buffer)
  def raw_output(m,inputs,outputs):
   dump('fomo_output_i8',flatten(outputs[0]));return outputs
  model.predict([bound],callback=raw_output)
  model=None;result=None;bound=None;raw_output=None;gc.collect()
 except Exception as error:meta['fomo_error']=repr(error);model=None;gc.collect()
def haar_run(frame,meta):
 try:
  gray=image.Image(160,160,image.GRAYSCALE);start=time.ticks_us();gray.draw_image(frame,0,0,hint=image.BILINEAR|image.CENTER|image.SCALE_ASPECT_IGNORE|image.BLACK_BACKGROUND);meta['haar_preprocess_us']=time.ticks_diff(time.ticks_us(),start)
  dump('haar_input_gray160',gray.bytearray());cascade=image.HaarCascade('/rom/haarcascade_frontalface.cascade');start=time.ticks_us();boxes=gray.find_features(cascade,threshold=.6,scale=1.25);meta['haar_predict_us']=time.ticks_diff(time.ticks_us(),start);meta['haar_boxes_160']=boxes
  gray=None;cascade=None;boxes=None;gc.collect()
 except Exception as error:meta['haar_error']=repr(error);gray=None;cascade=None;gc.collect()
def shot(study):
 gc.collect();start=time.ticks_us();frame=cam.snapshot();capture=time.ticks_diff(time.ticks_us(),start)
 meta={'width':frame.width(),'height':frame.height(),'pixel_format':'RGB565','stride_bytes':frame.width()*2,'capture_us':capture,'ticks_ms':time.ticks_ms(),'auto_exposure_requested':True}
 if study:
  dump('camera_rgb565',frame.bytearray());gc.collect()
  order=('haar','fomo') if PAIR_ORDER else ('fomo','haar');meta['detector_order']=order
  for name in order:
   if name=='fomo':fomo_run(frame,meta)
   else:haar_run(frame,meta)
  meta['gc_free_after']=gc.mem_free();meta['allocator']=umalloc.stats(0)
 jpeg=frame.to_jpeg(quality=65);dump('preview_jpeg',jpeg.bytearray());print('META '+json.dumps(meta));jpeg=None;frame=None;gc.collect()
'''

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--port',required=True);p.add_argument('--http-port',type=int,default=8766);a=p.parse_args()
    root=Path(__file__).resolve().parents[3];out=root/'work/project-research-v5/private/optical-captures';out.mkdir(exist_ok=True)
    jobs=queue.Queue();state={'jpeg':None,'frames':0,'last':None,'error':None,'started':time.monotonic(),'resets':0};condition=threading.Condition()
    target=connect(a.port)
    def initialize(order=0):
        target.enter_raw_repl(soft_reset=True);target.exec(PROGRAM);target.exec('PAIR_ORDER=%d'%order);target.read_until(1,b'>');target.serial.timeout=.1;state['resets']+=1
    initialize()
    def exchange(command):
        code=command.encode()
        for i in range(0,len(code),256):target.serial.write(code[i:i+256]);time.sleep(.01)
        target.serial.write(b'\x04');ack=b'';end=time.monotonic()+2
        while len(ack)<2 and time.monotonic()<end:ack+=target.serial.read(2-len(ack))
        if ack!=b'OK':raise RuntimeError('Raw execution acknowledgement '+repr(ack))
        data=bytearray();end=time.monotonic()+45
        while time.monotonic()<end:
            chunk=target.serial.read(max(1,min(65536,target.serial.in_waiting)))
            if chunk:data.extend(chunk)
            if data.count(4)>=2 and data.endswith(b'>'):break
        else:raise TimeoutError('Bounded camera RPC timed out')
        first=data.find(4);second=data.find(4,first+1)
        if first<0 or second<0:raise RuntimeError('Missing raw execution terminator')
        stderr=bytes(data[first+1:second])
        if stderr:raise RuntimeError(stderr.decode(errors='replace'))
        stdout=bytes(data[:first]);blobs={};meta=None
        for line in stdout.splitlines():
            if line.startswith(b'DATA '):
                _,name,payload=line.split(b' ',2);blobs.setdefault(name.decode(),bytearray()).extend(base64.b64decode(payload,validate=True))
            elif line.startswith(b'META '):meta=json.loads(line[5:])
        if meta is None or 'preview_jpeg' not in blobs:raise ValueError('Incomplete camera response')
        return stdout,meta,{k:bytes(v) for k,v in blobs.items()}
    def worker():
        try:
            while True:
                try:job=jobs.get(timeout=.1)
                except queue.Empty:job=None
                try:
                    if job and job['action']=='reset':initialize(job['metadata'].get('pair_order',0));job['result']={'status':'reset','reset_index':state['resets']};continue
                    stdout,meta,blobs=exchange('shot(%d)'%(1 if job else 0))
                    with condition:state['jpeg']=blobs['preview_jpeg'];state['frames']+=1;state['last']=time.monotonic();condition.notify_all()
                    if job:
                        folder=out/job['metadata']['capture_id'];folder.mkdir(exist_ok=False)
                        (folder/'stdout.b64').write_bytes(stdout);files=[]
                        for name,payload in blobs.items():
                            filename=name+('.jpg' if name=='preview_jpeg' else '.bin');(folder/filename).write_bytes(payload);files.append({'name':filename,'bytes':len(payload),'sha256':hashlib.sha256(payload).hexdigest()})
                        receipt={'metadata':meta,'request':job['metadata'],'files':files,'reset_index':state['resets'],'host_completed_monotonic_ns':time.monotonic_ns(),
                            'program_sha256':hashlib.sha256(PROGRAM.encode()).hexdigest(),'directory':folder.relative_to(root).as_posix()}
                        (folder/'RECEIPT.json').write_text(json.dumps(receipt,indent=2,allow_nan=False)+'\n');job['result']=receipt
                except Exception as e:
                    if job:job['result']={'error':repr(e)}
                    else:state['error']=repr(e);time.sleep(.5)
                finally:
                    if job:job['done'].set()
        finally:target.close()
    class H(BaseHTTPRequestHandler):
        def log_message(self,*args):pass
        def do_GET(self):
            if self.path=='/':data=PAGE.encode();typ='text/html'
            elif self.path=='/status':
                now=time.monotonic();data=json.dumps({'frames':state['frames'],'fps':round(state['frames']/max(now-state['started'],.01),1),'age_s':round(now-state['last'],1) if state['last'] else None,'error':state['error'],'mode':'optical_lab','reset_index':state['resets']}).encode();typ='application/json'
            elif self.path=='/snapshot.jpg':data=state['jpeg'] or b'';typ='image/jpeg'
            elif self.path=='/live.mjpg':
                self.send_response(200);self.send_header('Content-Type','multipart/x-mixed-replace; boundary=frame');self.end_headers();last=-1
                try:
                    while True:
                        with condition:condition.wait_for(lambda:state['frames']!=last,timeout=5);jpeg=state['jpeg'];last=state['frames']
                        if jpeg:self.wfile.write(b'--frame\r\nContent-Type: image/jpeg\r\nContent-Length: '+str(len(jpeg)).encode()+b'\r\n\r\n'+jpeg+b'\r\n');self.wfile.flush()
                except (BrokenPipeError,ConnectionResetError):return
            else:self.send_error(404);return
            self.send_response(200);self.send_header('Content-Type',typ);self.send_header('Cache-Control','no-store');self.end_headers();self.wfile.write(data)
        def do_POST(self):
            if self.path not in ['/capture','/reset']:self.send_error(404);return
            n=int(self.headers.get('Content-Length','0'))
            if not 0<n<8192:self.send_error(400);return
            metadata=json.loads(self.rfile.read(n))
            if self.path=='/reset' and metadata.get('pair_order',0) not in [0,1]:self.send_error(400);return
            if self.path=='/capture' and not re.fullmatch(r'[A-Za-z0-9._-]{1,120}',metadata.get('capture_id','')):self.send_error(400);return
            job={'action':self.path[1:],'metadata':metadata,'done':threading.Event()};jobs.put(job)
            if not job['done'].wait(60):self.send_error(504);return
            data=json.dumps(job['result'],allow_nan=False).encode();self.send_response(500 if 'error' in job['result'] else 200);self.send_header('Content-Type','application/json');self.end_headers();self.wfile.write(data)
    server=ThreadingHTTPServer(('127.0.0.1',a.http_port),H);(out/'owner.pid').write_text(str(os.getpid()))
    threading.Thread(target=worker,daemon=True).start();print('OPTICAL_LAB_READY',flush=True);server.serve_forever()

if __name__=='__main__':main()
