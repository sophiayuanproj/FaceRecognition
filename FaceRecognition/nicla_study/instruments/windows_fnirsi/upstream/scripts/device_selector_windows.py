"""Resolve the meter HID interface from fresh enumeration, never a cached path."""
import hashlib


def path_bytes(info):
    path = info.get("path", b"")
    if isinstance(path, str):
        return path.encode("utf-8", "replace")
    return bytes(path)


def describe(info, include_path=False):
    raw_path = path_bytes(info)
    row = {
        "vendor_id": "%04x" % int(info.get("vendor_id") or 0),
        "product_id": "%04x" % int(info.get("product_id") or 0),
        "manufacturer": info.get("manufacturer_string") or "",
        "product": info.get("product_string") or "",
        "release_number": info.get("release_number"),
        "interface_number": info.get("interface_number"),
        "usage_page": info.get("usage_page"),
        "usage": info.get("usage"),
        "serial_number_present": bool(info.get("serial_number")),
        "enumerated_path_sha256": hashlib.sha256(raw_path).hexdigest(),
    }
    if include_path:
        row["path_private"] = raw_path.decode("utf-8", "backslashreplace")
    return row


def choose_device(devices, interface_number=None, path_fragment=None):
    if interface_number is None:
        raise ValueError("select the HID interface explicitly with --interface-number")
    matches = [d for d in devices if d.get("interface_number") == interface_number]
    if path_fragment:
        needle = path_fragment.casefold()
        matches = [d for d in matches if needle in path_bytes(d).decode("utf-8", "replace").casefold()]
    if len(matches) != 1:
        raise ValueError("interface selection matched %d HID paths; inspect --list and resolve explicitly" % len(matches))
    return matches[0]
