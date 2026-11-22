import pyb

mode = pyb.usb_mode()
if mode not in ("VCP", "CDC"):
    if mode is not None:
        # Some OpenMV builds have already started the composite USB device.
        # The setter leaves an active device unchanged until it is stopped.
        pyb.usb_mode(None)
        pyb.delay(160)
    try:
        pyb.usb_mode("VCP", vid=0x2341, pid=0x045F, high_speed=True)
    except (OSError, ValueError, TypeError):
        pyb.usb_mode("VCP+MSC", vid=0x2341, pid=0x045F, high_speed=True)
