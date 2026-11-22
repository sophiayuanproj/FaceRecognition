"""Persistent local Nicla MJPEG preview; exclusive serial owner, RAM-only frames."""
import argparse,json,os,struct,threading,time
from pathlib import Path
from http.server import BaseHTTPRequestHandler,ThreadingHTTPServer
from board_lab import connect

PROGRAM='''import csi,pyb,time,gc
cam=csi.CSI();cam.reset();cam.pixformat(csi.RGB565);cam.framesize(csi.QVGA);cam.window((240,240));cam.framebuffers(1)
cam.snapshot(time=800)
usb=pyb.USB_VCP()
while True:
 frame=cam.snapshot()
 jpeg=frame.to_jpeg(quality=55)
 data=jpeg.bytearray()
 usb.write(b'NVJ1'+len(data).to_bytes(4,'little'))
 usb.write(data)
 data=None;jpeg=None;frame=None
 gc.collect()
 time.sleep_ms(80)
'''

PAGE='''<!doctype html><meta charset="utf-8"><title>Nicla live camera</title>
<style>body{margin:0;background:#111;color:#eee;font:16px system-ui;text-align:center}
header{padding:12px}main{height:calc(100vh - 105px);display:flex;align-items:center;justify-content:center;overflow:hidden}
img{width:min(75vh,85vw);height:min(75vh,85vw);object-fit:contain;image-rendering:auto}
button{padding:8px 14px;margin:0 6px;background:#333;color:#fff;border:1px solid #777;border-radius:5px}
#state{font-size:14px;color:#aaa}</style>
<header><b>Nicla Vision — live camera</b> <button onclick="angle=(angle+90)%360;camera.style.transform='rotate('+angle+'deg)'">Rotate preview</button>
<button onclick="document.documentElement.requestFullscreen()">Full screen</button><div id="state">Connecting…</div></header>
<main><img id="camera" src="/live.mjpg"></main><script>let angle=0;
setInterval(()=>fetch('/status').then(r=>r.json()).then(s=>state.textContent=s.error||('Live · '+s.frames+' frames · '+s.fps+' fps · '+s.age_s+' s since last frame')),1000);</script>'''

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--port',required=True);p.add_argument('--http-port',type=int,default=8766)
    p.add_argument('--pid-file',type=Path);a=p.parse_args()
    state={'jpeg':None,'frames':0,'last':None,'started':time.monotonic(),'error':None};condition=threading.Condition()
    target=connect(a.port);target.enter_raw_repl(soft_reset=True);target.exec_raw_no_follow(PROGRAM)
    target.serial.timeout=.1
    def capture():
        buffer=bytearray()
        try:
            while True:
                chunk=target.serial.read(max(1,min(65536,target.serial.in_waiting)))
                if not chunk:continue
                buffer.extend(chunk)
                while True:
                    start=buffer.find(b'NVJ1')
                    if start<0:
                        if b'Traceback' in buffer:raise RuntimeError(buffer.decode(errors='replace')[-1200:])
                        if len(buffer)>8192:del buffer[:-4]
                        break
                    if start:del buffer[:start]
                    if len(buffer)<8:break
                    n=struct.unpack('<I',buffer[4:8])[0]
                    if not 100<n<500000:del buffer[:4];continue
                    if len(buffer)<8+n:break
                    jpeg=bytes(buffer[8:8+n]);del buffer[:8+n]
                    if not jpeg.startswith(b'\xff\xd8'):raise ValueError('Invalid camera JPEG')
                    with condition:
                        state['jpeg']=jpeg;state['frames']+=1;state['last']=time.monotonic();condition.notify_all()
        except Exception as e:
            with condition:state['error']=str(e);condition.notify_all()
    class Handler(BaseHTTPRequestHandler):
        def log_message(self,*args):pass
        def do_GET(self):
            if self.path=='/':data=PAGE.encode();typ='text/html'
            elif self.path=='/status':
                now=time.monotonic();data=json.dumps({'frames':state['frames'],'fps':round(state['frames']/max(now-state['started'],.01),1),
                    'age_s':round(now-state['last'],1) if state['last'] else None,'error':state['error']}).encode();typ='application/json'
            elif self.path=='/snapshot.jpg':data=state['jpeg'] or b'';typ='image/jpeg'
            elif self.path=='/live.mjpg':
                self.send_response(200);self.send_header('Content-Type','multipart/x-mixed-replace; boundary=frame');self.send_header('Cache-Control','no-store');self.end_headers()
                delivered=-1
                try:
                    while True:
                        with condition:
                            condition.wait_for(lambda:state['frames']!=delivered or state['error'],timeout=5)
                            if state['error']:return
                            jpeg=state['jpeg'];delivered=state['frames']
                        if jpeg:
                            self.wfile.write(b'--frame\r\nContent-Type: image/jpeg\r\nContent-Length: '+str(len(jpeg)).encode()+b'\r\n\r\n'+jpeg+b'\r\n');self.wfile.flush()
                except (BrokenPipeError,ConnectionResetError):return
            else:self.send_error(404);return
            self.send_response(200);self.send_header('Content-Type',typ);self.send_header('Cache-Control','no-store');self.end_headers();self.wfile.write(data)
    server=ThreadingHTTPServer(('127.0.0.1',a.http_port),Handler)
    if a.pid_file:a.pid_file.parent.mkdir(parents=True,exist_ok=True);a.pid_file.write_text(str(os.getpid())+'\n')
    threading.Thread(target=capture,daemon=True).start()
    print('LIVE_NICLA http://127.0.0.1:%d/'%a.http_port,flush=True)
    try:server.serve_forever()
    finally:
        target.serial.write(b'\x03\x03\x02');target.close()

if __name__=='__main__':main()
