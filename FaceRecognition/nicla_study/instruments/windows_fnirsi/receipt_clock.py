"""Same-host UTC receipt clock for FNIRSI HID reports and read timeouts."""

from datetime import datetime, timezone
from functools import cache
import os
import time


UTC_SOURCE = "GetSystemTimePreciseAsFileTime" if os.name == "nt" else "time.time_ns"
FILETIME_UNIX_EPOCH = 116_444_736_000_000_000


@cache
def _windows_filetime_reader():
    from ctypes import POINTER, WinDLL, byref
    from ctypes.wintypes import FILETIME

    read_clock = WinDLL("kernel32").GetSystemTimePreciseAsFileTime
    read_clock.argtypes = (POINTER(FILETIME),)
    read_clock.restype = None

    def read_ns():
        value = FILETIME()
        read_clock(byref(value))
        ticks = (value.dwHighDateTime << 32) | value.dwLowDateTime
        return (ticks - FILETIME_UNIX_EPOCH) * 100

    return read_ns


def host_utc_ns():
    """Read the OS wall clock directly, with sub-tick resolution on Windows."""
    if os.name == "nt":
        return _windows_filetime_reader()()
    return time.time_ns()


def iso_utc_ns(utc_ns):
    seconds, nanoseconds = divmod(utc_ns, 1_000_000_000)
    return datetime.fromtimestamp(seconds, timezone.utc).replace(
        microsecond=nanoseconds // 1000
    ).isoformat(timespec="microseconds")
