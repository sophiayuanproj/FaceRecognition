"""Read-verified bounded research staging and artifact retrieval."""
import argparse,hashlib,json
from pathlib import Path
from board_lab import connect

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--port',required=True)
    p.add_argument('--upload-dir',type=Path);p.add_argument('--download-dir',type=Path)
    p.add_argument('--remote',default='/flash/research-v5');p.add_argument('--receipt',type=Path,required=True)
    a=p.parse_args()
    if not a.remote.startswith('/flash/research-v5') or '..' in a.remote:raise ValueError('Research staging path only')
    t=connect(a.port);rows=[]
    try:
        if a.upload_dir:
            t.exec("import os\ntry:os.mkdir(%r)\nexcept OSError:pass"%a.remote)
            for source in sorted(a.upload_dir.iterdir()):
                if not source.is_file():continue
                payload=source.read_bytes();path=a.remote+'/'+source.name
                t.fs_writefile(path,payload,chunk_size=4096)
                returned=bytes(t.fs_readfile(path,chunk_size=4096))
                if returned!=payload:raise ValueError('Stage readback mismatch')
                rows.append({'path':path,'bytes':len(payload),'sha256':hashlib.sha256(returned).hexdigest()})
        if a.download_dir:
            a.download_dir.mkdir(parents=True,exist_ok=True)
            for item in t.fs_listdir(a.remote):
                if item.st_mode & 0x4000:continue
                data=bytes(t.fs_readfile(a.remote+'/'+item.name,chunk_size=4096))
                (a.download_dir/item.name).write_bytes(data)
                rows.append({'path':a.remote+'/'+item.name,'bytes':len(data),'sha256':hashlib.sha256(data).hexdigest()})
        a.receipt.parent.mkdir(parents=True,exist_ok=True);a.receipt.write_text(json.dumps(rows,indent=2)+'\n')
        print(json.dumps({'verified_files':len(rows)}));t.exit_raw_repl()
    finally:t.close()

if __name__=='__main__':main()
