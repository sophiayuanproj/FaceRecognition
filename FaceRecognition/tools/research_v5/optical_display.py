"""Shared-state, loopback optical stimulus display; development sources only."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse


HTML = r"""<!doctype html><meta charset="utf-8"><title>Nicla test target</title>
<style>
html,body{margin:0;width:100%;height:100%;overflow:hidden;background:#777;cursor:none!important}
*{cursor:none!important}
#target,#clip{position:absolute;left:50%;top:50%;transform:translate(-50%,-50%);object-fit:contain;width:80vmin;height:80vmin;opacity:.65}
#clip{visibility:hidden}
#sync{position:fixed;left:12px;top:12px;width:220px;height:220px;z-index:9999;display:none;pointer-events:none;image-rendering:pixelated}
</style>
<img id="target" alt="optical target"><video id="clip" muted playsinline preload="auto" aria-label="optical workload video"></video><img id="sync" alt="">
<script>
let version=null;
const target=document.querySelector('#target');
const clip=document.querySelector('#clip');
const sync=document.querySelector('#sync');
let playbackTimer=null;
function place(element,s){
  element.style.width=(s.size_vmin||80)+'vmin';
  element.style.height=(s.size_vmin||80)+'vmin';
  element.style.left=(s.x_pct||50)+'%';
  element.style.top=(s.y_pct||50)+'%';
  const rotation=Number.isFinite(s.rotation_deg)?s.rotation_deg:0;
  element.style.transform='translate(-50%,-50%) rotate('+rotation+'deg)';
  element.style.opacity=s.opacity===undefined?.65:s.opacity;
}
function sendRender(s,element,markerId=null,extra={}){
  if(version!==s.version)return;
  const bounds=element.getBoundingClientRect();
  const markerBounds=markerId===null?null:sync.getBoundingClientRect();
  fetch('/render',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({
    version:s.version,kind:s.kind,token:s.token||null,marker_id:markerId,browser_ms:performance.now(),
    page_visibility:document.visibilityState,page_has_focus:document.hasFocus(),
    viewport:[innerWidth,innerHeight],dpr:devicePixelRatio,bounds:[bounds.x,bounds.y,bounds.width,bounds.height],
    marker_bounds:markerBounds?[markerBounds.x,markerBounds.y,markerBounds.width,markerBounds.height]:null,
    ...extra
  })}).catch(()=>{});
}
function stopClip(){
  clearTimeout(playbackTimer);
  clip.oncanplay=clip.onplaying=clip.ontimeupdate=clip.onended=clip.onerror=null;
  if(clip.hasAttribute('src')){
    clip.pause();
    clip.removeAttribute('src');
    clip.load();
  }
  clip.style.visibility='hidden';
}
function showClip(s){
  place(clip,s);
  target.style.visibility='hidden';
  sync.style.display='none';
  clip.style.visibility='visible';
  clip.muted=true;
  const report=(event,extra={})=>sendRender(s,clip,null,{
    event,media_time_s:clip.currentTime,browser_wall_ms:Date.now(),...extra
  });
  let lastTick=0;
  clip.oncanplay=()=>{
    clip.oncanplay=null;
    report('video_ready',{duration_s:clip.duration});
    const delay=Number.isFinite(s.play_at_unix_ms)?Math.max(0,s.play_at_unix_ms-Date.now()):0;
    playbackTimer=setTimeout(()=>{
      if(version!==s.version)return;
      clip.play().catch(error=>report('video_error',{message:error.name}));
    },delay);
  };
  clip.onplaying=()=>{
    clip.onplaying=null;
    lastTick=performance.now();
    report('video_started');
  };
  clip.ontimeupdate=()=>{
    const now=performance.now();
    if(clip.paused||now-lastTick<450)return;
    lastTick=now;
    report('video_tick');
  };
  clip.onended=()=>report('video_ended');
  clip.onerror=()=>report('video_error',{message:clip.error?.code||'media error'});
  clip.src='/video/'+encodeURIComponent(s.token)+'?v='+encodeURIComponent(s.version);
}
async function update(){
  try{
    const s=await(await fetch('/state',{cache:'no-store'})).json();
    if(s.version===version)return;
    version=s.version;
    document.body.style.background=s.background||'#777';
    stopClip();
    if(s.kind==='video'){
      showClip(s);
      return;
    }
    place(target,s);
    target.style.visibility=s.kind==='blank'?'hidden':'visible';
    const markerId=Number.isInteger(s.sync_marker_id)?s.sync_marker_id:null;
    sync.style.display=markerId===null?'none':'block';
    sync.style.left=(Number.isFinite(s.sync_marker_left_px)?s.sync_marker_left_px:12)+'px';
    sync.style.top=(Number.isFinite(s.sync_marker_top_px)?s.sync_marker_top_px:12)+'px';
    sync.style.width=(Number.isFinite(s.sync_marker_size_px)?s.sync_marker_size_px:220)+'px';
    sync.style.height=sync.style.width;
    const expected=markerId===null?1:2;
    let loaded=0;
    const report=()=>{
      loaded++;
      if(loaded!==expected)return;
      requestAnimationFrame(()=>sendRender(s,target,markerId));
    };
    target.onload=report;
    target.onerror=()=>{};
    if(markerId!==null){sync.onload=report;sync.onerror=()=>{};sync.src='/marker.svg?id='+markerId+'&v='+encodeURIComponent(s.version);}
    target.src=s.kind==='photo'?('/image/'+encodeURIComponent(s.token)+'?v='+encodeURIComponent(s.version)):('/cal.svg?v='+encodeURIComponent(s.version));
  }catch(e){}
}
async function poll(){
  await update();
  setTimeout(poll,150);
}
poll();
</script>"""


def make_marker_svg(matrix: list[str], marker_id: int) -> bytes:
    if len(matrix) != 6 or any(len(row) != 6 or set(row) - {"0", "1"} for row in matrix):
        raise ValueError("marker matrix must be six rows of six bits")
    rects = []
    for y, row in enumerate(matrix):
        for x, bit in enumerate(row):
            if bit == "1":
                rects.append(f'<rect x="{x + 1}" y="{y + 1}" width="1" height="1"/>')
    svg = (
        '<svg xmlns="http://www.w3.org/2000/svg" width="800" height="800" viewBox="0 0 8 8" '
        f'role="img" aria-label="source marker {marker_id}">'
        '<rect width="8" height="8" fill="#fff"/>'
        '<g fill="#000" shape-rendering="crispEdges">' + "".join(rects) + "</g></svg>"
    )
    return svg.encode("ascii")


def byte_range(value: str, size: int) -> tuple[int, int]:
    """Return the bounds of a single HTTP byte range, end exclusive."""
    match = re.fullmatch(r"bytes=(\d*)-(\d*)", value.strip())
    if not match or not any(match.groups()) or size == 0:
        raise ValueError("unsupported byte range")
    start_text, end_text = match.groups()
    if not start_text:
        count = int(end_text)
        if count == 0:
            raise ValueError("empty suffix range")
        return max(0, size - count), size
    start = int(start_text)
    stop = min(size, int(end_text) + 1) if end_text else size
    if start >= stop:
        raise ValueError("unsatisfiable byte range")
    return start, stop


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=8767)
    parser.add_argument("--extra-sources", type=Path, help="local manifest of additional optical test media")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[3]
    study = root / "work/project-research-v5"
    out = study / "runs/optical"
    out.mkdir(parents=True, exist_ok=True)
    sources = out / "DISPLAY_SOURCES.json"
    source_plan = json.loads((sources if sources.exists() else study / "runs/stimulus/PLAN.json").read_text())
    by_token = {row["token"]: row for row in source_plan["items"]}
    if args.extra_sources:
        extra_sources = json.loads(args.extra_sources.read_text())["items"]
        for row in extra_sources:
            if row["token"] in by_token:
                raise ValueError("duplicate optical source token: " + row["token"])
            by_token[row["token"]] = row
    control = out / "CONTROL.json"
    if not control.exists():
        control.write_text(json.dumps({"version": 0, "kind": "calibration", "opacity": 0.3, "size_vmin": 85}))

    codebook_path = out / "SYNC_MARKER_CODEBOOK_V1.json"
    marker_by_id: dict[int, dict] = {}
    if codebook_path.exists():
        codebook = json.loads(codebook_path.read_text(encoding="utf-8"))
        if codebook.get("schema") != "optical-sync-marker-codebook-v1" or codebook.get("source_index_sha256") != hashlib.sha256(sources.read_bytes()).hexdigest():
            raise RuntimeError("optical sync marker codebook does not match the current source index")
        marker_by_id = {int(row["marker_id"]): row for row in codebook.get("records", [])}

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_args):
            pass

        def do_GET(self):
            parsed = urlparse(self.path)
            query = parse_qs(parsed.query)
            path = parsed.path
            status = 200
            headers = {}
            if path == "/":
                data, content_type = HTML.encode("utf-8"), "text/html; charset=utf-8"
            elif path == "/state":
                data, content_type = control.read_bytes(), "application/json"
            elif path == "/cal.svg":
                squares = "".join(
                    '<rect x="%d" y="%d" width="50" height="50" fill="%s"/>'
                    % (x * 50, y * 50, "#ddd" if (x + y) % 2 else "#222")
                    for y in range(8) for x in range(8)
                )
                data = ('<svg xmlns="http://www.w3.org/2000/svg" width="400" height="400" viewBox="0 0 400 400">' + squares + '<circle cx="25" cy="25" r="17" fill="#f06"/></svg>').encode()
                content_type = "image/svg+xml"
            elif path.startswith(("/image/", "/video/")):
                token = path.rsplit("/", 1)[-1]
                row = by_token.get(token)
                video = path.startswith("/video/")
                expected_type = "video/webm" if video else "image/jpeg"
                if row is None or row.get("media_type", "image/jpeg") != expected_type:
                    self.send_error(404)
                    return
                image_path = root / row["path"]
                data = image_path.read_bytes()
                if hashlib.sha256(data).hexdigest() != row["sha256"]:
                    self.send_error(409)
                    return
                content_type = expected_type
                if video:
                    headers["Accept-Ranges"] = "bytes"
                    requested_range = self.headers.get("Range")
                    if requested_range:
                        size = len(data)
                        try:
                            start, stop = byte_range(requested_range, size)
                        except ValueError:
                            self.send_response(416)
                            self.send_header("Content-Range", f"bytes */{size}")
                            self.send_header("Content-Length", "0")
                            self.end_headers()
                            return
                        status = 206
                        headers["Content-Range"] = f"bytes {start}-{stop - 1}/{size}"
                        data = memoryview(data)[start:stop]
            elif path == "/marker.svg":
                try:
                    marker_id = int(query.get("id", [""])[0])
                    row = marker_by_id[marker_id]
                except (ValueError, KeyError, IndexError):
                    self.send_error(404)
                    return
                data = make_marker_svg(row["matrix_6x6_black_one"], marker_id)
                content_type = "image/svg+xml"
            else:
                self.send_error(404)
                return
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Cache-Control", "no-store")
            self.send_header("Content-Length", str(len(data)))
            for key, value in headers.items():
                self.send_header(key, value)
            self.end_headers()
            self.wfile.write(data)

        def do_POST(self):
            if urlparse(self.path).path != "/render":
                self.send_error(404)
                return
            length = int(self.headers.get("Content-Length", "0"))
            if not 0 < length < 4096:
                self.send_error(400)
                return
            row = json.loads(self.rfile.read(length))
            row["host_receive_monotonic_ns"] = time.monotonic_ns()
            row["physical_onset_verified"] = False
            with (out / "RENDER_EVENTS.jsonl").open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(row, allow_nan=False) + "\n")
            self.send_response(204)
            self.send_header("Content-Length", "0")
            self.end_headers()

    (out / "display.pid").write_text(str(os.getpid()))
    print(f"Optical display http://127.0.0.1:{args.port}/", flush=True)
    ThreadingHTTPServer(("127.0.0.1", args.port), Handler).serve_forever()


if __name__ == "__main__":
    main()
