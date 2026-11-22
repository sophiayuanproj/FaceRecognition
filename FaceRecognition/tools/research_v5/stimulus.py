"""Loopback-only public nonfinal image player. Browser timing != optical onset."""
import argparse,hashlib,json,time
from pathlib import Path
from http.server import BaseHTTPRequestHandler,ThreadingHTTPServer

HTML='''<!doctype html><meta charset="utf-8"><title>Camera research target</title>
<style>html,body{margin:0;background:#080808;color:white;font:16px system-ui;height:100%;overflow:hidden}
#photo{position:absolute;left:50%;top:50%;transform:translate(-50%,-50%);width:420px;height:420px;object-fit:contain}
#controls{position:absolute;bottom:8px;left:8px}button{padding:8px}</style>
<img id="photo"><div id="controls"><button onclick="next()">Next public target</button>
<button onclick="document.documentElement.requestFullscreen()">Full screen</button><button onclick="blank()">Blank</button></div>
<script>let pos=-1,items=[];fetch('/plan').then(r=>r.json()).then(v=>{items=v;next()});
function mark(kind){fetch('/event',{method:'POST',body:JSON.stringify({kind,index:pos,browser_performance_ms:performance.now(),screen_width:screen.width,screen_height:screen.height,device_pixel_ratio:devicePixelRatio,viewport_width:innerWidth,viewport_height:innerHeight,image_display_css_px:420})})}
function next(){pos=(pos+1)%items.length;photo.onload=()=>requestAnimationFrame(()=>mark('browser_render_requested'));photo.src='/image/'+items[pos].token;photo.style.visibility='visible'}
function blank(){photo.style.visibility='hidden';mark('blank_requested')}
document.addEventListener('keydown',e=>{if(e.code==='Space'){e.preventDefault();next()}});</script>'''

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--port',type=int,default=8765)
    p.add_argument('--workspace',type=Path,default=Path(__file__).resolve().parents[3]);a=p.parse_args()
    root=a.workspace.resolve();study=root/'work/project-research-v5';out=study/'runs/stimulus';out.mkdir(exist_ok=True)
    rows=[json.loads(l) for l in (study/'data/manifests/lfw_roles.jsonl').read_text().splitlines()]
    plan=[]
    for label in ['K01','K02','K03','K04','K05','OTHER']:
        eligible=[r for r in rows if r['label']==label and r['role'] in ['development','unknown_development']]
        if label=='OTHER':
            seen=set();eligible=[r for r in eligible if not (r['source_identity'] in seen or seen.add(r['source_identity']))]
        for row in eligible[:3]:
            path=root/row['path']
            if hashlib.sha256(path.read_bytes()).hexdigest()!=row['sha256']:raise ValueError('Stimulus changed')
            plan.append(dict(row,token=hashlib.sha256(row['record_id'].encode()).hexdigest()[:24]))
    (out/'PLAN.json').write_text(json.dumps({'role':'development_only','items':plan,'camera_geometry':'not physically calibrated','browser_times_are_not_physical_onset':True},indent=2)+'\n')
    by={r['token']:r for r in plan}
    class Handler(BaseHTTPRequestHandler):
        def log_message(self,*args):pass
        def do_GET(self):
            if self.path=='/':data=HTML.encode();typ='text/html'
            elif self.path=='/plan':data=json.dumps([{'token':r['token']} for r in plan]).encode();typ='application/json'
            elif self.path.startswith('/image/') and self.path.split('/')[-1] in by:
                row=by[self.path.split('/')[-1]];data=(root/row['path']).read_bytes();typ='image/jpeg'
            else:self.send_error(404);return
            self.send_response(200);self.send_header('Content-Type',typ);self.send_header('Cache-Control','no-store');self.end_headers();self.wfile.write(data)
        def do_POST(self):
            if self.path!='/event':self.send_error(404);return
            n=int(self.headers.get('Content-Length','0'))
            if not 0<n<4096:self.send_error(400);return
            row=json.loads(self.rfile.read(n));row['host_receive_monotonic_ns']=time.monotonic_ns()
            i=row.get('index');row['source_record_id']=plan[i]['record_id'] if isinstance(i,int) and 0<=i<len(plan) else None
            row['physical_visibility_verified']=False
            with (out/'BROWSER_EVENTS.jsonl').open('a') as f:f.write(json.dumps(row,allow_nan=False)+'\n')
            self.send_response(204);self.end_headers()
    print('Loopback public development player http://127.0.0.1:%d'%a.port,flush=True)
    ThreadingHTTPServer(('127.0.0.1',a.port),Handler).serve_forever()

if __name__=='__main__':main()
