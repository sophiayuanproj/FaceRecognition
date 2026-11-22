"""Back up raw ROMFS, deploy aligned image, and read-verify exact partition.

Uses MicroPython mpremote's documented ROMFS ioctls after inspecting the actual
device. A verified raw original partition is retained before the first erase.
"""
import argparse,base64,hashlib,json,time
from pathlib import Path
from board_lab import connect

def device_hash(target,size):
    return target.exec("import hashlib,binascii\nh=hashlib.sha256()\nfor i in range(0,%d,16384):h.update(dev[i:min(i+16384,%d)])\nprint(binascii.hexlify(h.digest()).decode())"%(size,size)).decode().strip()

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--port',required=True)
    p.add_argument('--backup',type=Path,required=True);p.add_argument('--image',type=Path)
    p.add_argument('--receipt',type=Path,required=True);a=p.parse_args()
    t=connect(a.port)
    try:
        t.enter_raw_repl(soft_reset=True)
        t.exec('import vfs,binascii,gc;dev=vfs.rom_ioctl(2,0)')
        size=int(t.eval('len(dev)'))
        if size!=4194304:raise ValueError('Unexpected partition size')
        original=device_hash(t,size)
        if a.backup.exists():
            backup_hash=hashlib.sha256(a.backup.read_bytes()).hexdigest()
            side=a.backup.with_suffix('.json')
            if not side.exists() or json.loads(side.read_text())['sha256']!=backup_hash:raise ValueError('Original backup not verified')
        else:
            a.backup.parent.mkdir(parents=True,exist_ok=True)
            with a.backup.open('wb') as f:
                for offset in range(0,size,16384):
                    b=t.exec('print(binascii.b2a_base64(dev[%d:%d]).decode())'%(offset,min(offset+16384,size))).strip()
                    f.write(base64.b64decode(b,validate=True))
            backup_hash=hashlib.sha256(a.backup.read_bytes()).hexdigest()
            if backup_hash!=original:raise ValueError('Raw backup hash mismatch')
            a.backup.with_suffix('.json').write_text(json.dumps({'sha256':backup_hash,'bytes':size,'verified_against_device':True},indent=2)+'\n')
            print(json.dumps({'raw_backup_verified':True,'bytes':size}),flush=True)
        receipt={'observed_pre_deploy_partition_sha256':original,'original_backup_sha256':backup_hash,
                 'backup_scope':'original full raw4MiB, private; retained for rollback'}
        if a.image:
            payload=a.image.read_bytes()
            if not payload.startswith(b'\xd2\xcd\x31') or len(payload)>size:raise ValueError('Invalid/oversized ROMFS')
            t.exec("try:\n vfs.umount('/rom')\nexcept OSError:pass\ngc.collect()")
            # Whole-partition external-flash erase can legitimately exceed
            # mpremote.exec's fixed10s; extend this operation only, preserving
            # the explicit raw response and bounded90s deadline.
            out,err=t.exec_raw('print(vfs.rom_ioctl(3,0,%d))'%len(payload),timeout=90)
            if err:raise ValueError('ROMFS prepare: '+err.decode(errors='replace'))
            minimum=int(out.strip())
            if minimum<=0:raise ValueError('ROMFS prepare failed')
            chunk=max(4096,minimum)
            for offset in range(0,len(payload),chunk):
                data=payload[offset:offset+chunk];data+=bytes(chunk-len(data))
                encoded=base64.b64encode(data)
                t.exec('buf=binascii.a2b_base64(%r);vfs.rom_ioctl(4,0,%d,buf)'%(encoded,offset))
            t.eval('vfs.rom_ioctl(5,0)')
            t.exec('dev=vfs.rom_ioctl(2,0)')
            expected=hashlib.sha256(payload).hexdigest();actual=device_hash(t,len(payload))
            if actual!=expected:raise ValueError('ROMFS image readback mismatch')
            receipt.update(deployed_image_sha256=expected,readback_sha256=actual,image_bytes=len(payload),verified=True)
            # Remount with the built-in VfsRom after verified flush.
            t.exec("vfs.mount(vfs.VfsRom(dev),'/rom');print('ROM_REMOUNTED')")
        a.receipt.parent.mkdir(parents=True,exist_ok=True);a.receipt.write_text(json.dumps(receipt,indent=2)+'\n')
        print(json.dumps(receipt),flush=True);t.exit_raw_repl()
    finally:t.close()

if __name__=='__main__':main()
