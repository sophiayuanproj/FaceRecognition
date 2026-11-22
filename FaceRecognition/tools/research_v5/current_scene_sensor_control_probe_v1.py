"""Run on Nicla with board_lab.py; checks GC2145 control support and resets defaults."""
import csi, json, time

cam=csi.CSI()
cam.reset()
cam.pixformat(csi.RGB565)
cam.framesize(csi.QVGA)
cam.window((240,240))
cam.framebuffers(1)
cam.snapshot(time=900)

def attempt(name, fn):
    try:
        value=fn()
        print(json.dumps({'probe':'sensor_control_api','name':name,'status':'supported','value':value}))
        return True
    except Exception as error:
        print(json.dumps({'probe':'sensor_control_api','name':name,'status':'unsupported_or_error','error':repr(error)}))
        return False

attempt('sensor_id',cam.cid)
attempt('width',cam.width)
attempt('height',cam.height)
attempt('exposure_readback_auto',cam.exposure_us)
attempt('gain_readback_auto',cam.gain_db)
attempt('auto_exposure_fixed_8000us',lambda:cam.auto_exposure(False,exposure_us=8000))
time.sleep_ms(250)
attempt('exposure_readback_after_fixed_request',cam.exposure_us)
attempt('auto_gain_fixed_0db',lambda:cam.auto_gain(False,gain_db=0.0))
attempt('gain_readback_after_fixed_request',cam.gain_db)
attempt('auto_whitebal_fixed_zero',lambda:cam.auto_whitebal(False,rgb_gain_db=(0.0,0.0,0.0)))
attempt('auto_exposure_restore',lambda:cam.auto_exposure(True))
attempt('auto_gain_restore',lambda:cam.auto_gain(True))
attempt('auto_whitebal_restore',lambda:cam.auto_whitebal(True))
cam.snapshot(time=500)
print(json.dumps({'probe':'sensor_control_api','name':'complete','status':'restored_auto_requested','width':cam.width(),'height':cam.height()}))
