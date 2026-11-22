import json,os,gc,binascii
import vfs
print(json.dumps({'vfs_api':dir(vfs),'csi_free':gc.mem_free(),'flash':os.statvfs('/flash')}))
if hasattr(vfs,'rom_ioctl'):
    print('ROM_PARTITIONS',vfs.rom_ioctl(1))
    dev=vfs.rom_ioctl(2,0)
    print('ROM_DEVICE',type(dev),len(dev),binascii.hexlify(memoryview(dev)[:32]))
    print('ROM_BLOCK',vfs.rom_ioctl(6,0))
import csi
print('CSI_CONSTANTS',[x for x in dir(csi) if any(t in x for t in ['FOV','READOUT','RGB','GRAY','VGA'])])
cam=csi.CSI();cam.reset()
for size in [csi.QQVGA,csi.QVGA,csi.VGA]:
    try:
        cam.framesize(size);cam.pixformat(csi.GRAYSCALE);cam.framebuffers(1)
        print('CAM_SIZE',cam.width(),cam.height(),cam.window(),cam.exposure_us(),cam.gain_db())
        for flag in ['IOCTL_GET_READOUT_WINDOW','IOCTL_GET_FOV_WIDE']:
            try:print(flag,cam.ioctl(getattr(csi,flag)))
            except Exception as e:print(flag,repr(e))
    except Exception as e:print('CAM_ERROR',repr(e))
