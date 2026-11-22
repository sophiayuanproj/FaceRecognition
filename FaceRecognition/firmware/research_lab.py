"""Opt-in bounded candidate lab; never promotes unqualified identity outputs.

Uses the existing versioned loader and shared policy module. Public research
commands expose actual capture/model/sleep actions for development diagnostics.
"""
import gc,json,time,sys,os,math
import csi,ml,pyb,umalloc,imu
from machine import LED
from ml.preprocessing import Normalization
from ml.postprocessing.edgeimpulse import Fomo
from policy_controller import PolicyController
from camera_geometry import centered_roi_clamped, fomo_center_camera_px

ROOT=None
def configure_release_root(path):
    global ROOT
    ROOT=path

class Lab:
    def __init__(self):
        with open(ROOT+'/research_config.json') as f:self.config=json.load(f)
        self.usb=pyb.USB_VCP();self.line=b'';self.boot_id='%08x'%pyb.rng();self.seq=0
        self.red=LED('LED_RED');self.blue=LED('LED_BLUE');self.green=LED('LED_GREEN')
        self.camera=csi.CSI();self.camera.reset();self.camera.pixformat(csi.RGB565)
        self.camera.framesize(csi.QVGA);self.camera.window((240,240));self.camera.framebuffers(1)
        self.camera.snapshot(time=500);self.stage='detector_only';self.sleeping=False
        self.interval_ms=2000;self.block=None;self.last=time.ticks_ms();self.model=None;self.model_path=None
        self.policy=PolicyController('L',ticks_diff=time.ticks_diff)
        self.emit('hello',{'release':ROOT,'config':self.config,'status':'RESEARCH_COMPONENT_LAB','identity_qualified':False,
            'led_legend':'blue=capture, green=face candidate only, red=execution fault; no known-identity LED claim'})

    def emit(self,event,extra=None):
        row={'schema':'nicla-v5-lab-1','event':event,'sequence':self.seq,'boot_id':self.boot_id,'ticks_ms':time.ticks_ms()};self.seq+=1
        if extra:row.update(extra)
        if self.usb.isconnected():self.usb.write((json.dumps(row)+'\n').encode())

    def unload(self):self.model=None;self.model_path=None;gc.collect()
    def load(self,path,post=None):
        if path!=self.model_path:
            self.unload();self.model=ml.Model(path,postprocess=post);self.model_path=path
        return self.model

    def capture(self):
        if self.sleeping:self.camera.sleep(False);self.camera.snapshot(time=200);self.sleeping=False
        return self.camera.snapshot()

    def invoke(self):
        started=time.ticks_us();timings={};face_count=None;status='COMPONENT_DIAGNOSTIC';self.blue.on();self.red.off();self.green.off()
        if self.stage=='idle':self.blue.off();return {'stage':'idle','elapsed_us':time.ticks_diff(time.ticks_us(),started)}
        if self.stage=='sensor_sleep':
            if not self.sleeping:self.unload();self.camera.sleep(True);self.sleeping=True
            self.blue.off();return {'stage':'sensor_sleep','buffers_freed_claim':False,'allocator':umalloc.stats(0)}
        start=time.ticks_us();frame=self.capture();timings['capture_us']=time.ticks_diff(time.ticks_us(),start)
        if self.stage in ['preprocess_native','preprocess_legacy']:
            self.unload();buf=bytearray(9216);start=time.ticks_us()
            if self.stage=='preprocess_native':Normalization()(frame)(buf,(1,96,96,1),ord('B'))
            else:
                from identity_contracts import pillow_bilinear_model_input
                pillow_bilinear_model_input(frame)(buf,(1,96,96,1),ord('B'))
            timings['preprocess_us']=time.ticks_diff(time.ticks_us(),start);buf=None
        if self.stage in ['detector_only','pipeline_diagnostic']:
            start=time.ticks_us();model=self.load('/rom/fomo_face_detection.tflite',Fomo(threshold=.5));timings['detector_load_us']=time.ticks_diff(time.ticks_us(),start)
            start=time.ticks_us();result=model.predict([frame]);timings['detector_predict_us']=time.ticks_diff(time.ticks_us(),start)
            # Stock labels are background,face; preserve empty channels.
            faces=result[1] if len(result)>1 else []
            face_count=len(faces)
            if faces:self.green.on()
            status='NO_FACE' if not faces else ('FACE_CANDIDATE_UNQUALIFIED' if len(faces)==1 else 'MULTIPLE_OR_AMBIGUOUS')
            if self.stage=='pipeline_diagnostic' and face_count==1:
                rect=faces[0][0]
                center=fomo_center_camera_px(rect,frame.width(),frame.height())
                roi=centered_roi_clamped(center,160,frame.width(),frame.height())
                model=None;result=None;self.unload();start=time.ticks_us()
                model=self.load(self.config['models']['large']['path']);timings['identity_load_us']=time.ticks_diff(time.ticks_us(),start)
                start=time.ticks_us();output=model.predict([Normalization(roi=roi)(frame)]);timings['identity_predict_us']=time.ticks_diff(time.ticks_us(),start)
                status='IDENTITY_UNQUALIFIED';output=None
            model=None;result=None
        if self.stage in ['identity_small_only','identity_large_only']:
            slot='small' if self.stage=='identity_small_only' else 'large'
            start=time.ticks_us();model=self.load(self.config['models'][slot]['path']);timings['model_load_us']=time.ticks_diff(time.ticks_us(),start)
            roi=centered_roi_clamped((frame.width()/2,frame.height()/2),160,frame.width(),frame.height())
            start=time.ticks_us();output=model.predict([Normalization(roi=roi)(frame)]);timings['model_predict_us']=time.ticks_diff(time.ticks_us(),start)
            status='IDENTITY_UNQUALIFIED';model=None;output=None
        self.blue.off();frame=None
        if not self.config.get('reuse_model',False):self.unload()
        gc.collect()
        return {'stage':self.stage,'timings':timings,'elapsed_us':time.ticks_diff(time.ticks_us(),started),
            'status':status,'face_count':face_count,'gc_free':gc.mem_free(),'allocator':umalloc.stats(0),'identity_qualified':False}

    def command(self,row):
        cmd=row.get('cmd');request=row.get('request_id')
        if cmd=='ping':self.emit('ack',{'request_id':request,'ok':True});return
        if cmd=='info':self.emit('info',{'request_id':request,'release':ROOT,'config':self.config,'gc_free':gc.mem_free(),'allocator':umalloc.stats(0)});return
        if cmd=='stop':
            self.emit('block_end',{'request_id':request,'block':self.block,'reason':'requested_stop'});self.block=None;self.stage='idle';self.unload();return
        if cmd in ['configure','run_block']:
            stage=row.get('stage','detector_only')
            if stage not in ['idle','sensor_sleep','camera_capture_only','preprocess_native','preprocess_legacy','detector_only','identity_small_only','identity_large_only','pipeline_diagnostic']:raise ValueError('Unsupported research stage')
            interval=row.get('interval_ms',2000)
            if isinstance(interval,bool) or not isinstance(interval,int) or not 250<=interval<=5000:raise ValueError('Invalid bounded cadence')
            self.unload();self.stage=stage;self.interval_ms=interval
            if cmd=='run_block':
                duration=row.get('duration_ms',10000)
                if isinstance(duration,bool) or not isinstance(duration,int) or not 1000<=duration<=180000:raise ValueError('Invalid block duration')
                self.block={'id':row.get('block_id'),'started_ms':time.ticks_ms(),'duration_ms':duration,'completed_cycles':0,'errors':0}
                self.emit('block_start',{'request_id':request,'block':self.block,'stage':stage,'interval_ms':interval})
            else:self.emit('configured',{'request_id':request,'stage':stage,'interval_ms':interval})
            self.last=time.ticks_add(time.ticks_ms(),-interval);return
        if cmd=='once':
            self.emit('cycle',dict(self.invoke(),request_id=request));return
        if cmd=='imu':
            self.emit('imu',{'request_id':request,'acceleration_mg':imu.acceleration_mg(),'angular_rate_mdps':imu.angular_rate_mdps()});return
        raise ValueError('Unsupported command')

    def run(self):
        while True:
            if self.usb.any():
                data=self.usb.read(min(self.usb.any(),256))
                if data:
                    self.line+=data
                    if len(self.line)>2048:self.line=b'';self.emit('error',{'reason':'command_too_long'})
                    while b'\n' in self.line:
                        one,self.line=self.line.split(b'\n',1)
                        try:self.command(json.loads(one))
                        except Exception as e:self.emit('error',{'reason':repr(e),'request_id':None})
            now=time.ticks_ms()
            if self.block and time.ticks_diff(now,self.block['started_ms'])>=self.block['duration_ms']:
                self.emit('block_end',{'block':self.block,'reason':'duration_complete'});self.block=None;self.stage='idle';self.unload()
            if time.ticks_diff(now,self.last)>=self.interval_ms:
                self.last=now
                try:
                    result=self.invoke();self.emit('cycle',result)
                    if self.block:self.block['completed_cycles']+=1
                except Exception as e:
                    self.red.on();self.unload();self.emit('cycle_error',{'stage':self.stage,'error':repr(e)})
                    if self.block:self.block['errors']+=1
            time.sleep_ms(5)

def main():Lab().run()
