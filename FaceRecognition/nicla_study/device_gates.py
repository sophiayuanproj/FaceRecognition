"""Fail-closed fresh-device gates for actions that require physical hardware."""
from __future__ import annotations
from datetime import datetime, timezone


def require_fresh_device(discovery: dict, target: str, max_age_s: float = 60.0) -> dict:
    observed = discovery.get("enumerated_utc")
    if not observed:
        return {"status": "pending_external_dependency", "reason": "no fresh discovery timestamp"}
    try:
        stamp = datetime.fromisoformat(str(observed).replace("Z", "+00:00"))
        age = (datetime.now(timezone.utc) - stamp.astimezone(timezone.utc)).total_seconds()
    except Exception:
        return {"status": "pending_external_dependency", "reason": "invalid discovery timestamp"}
    if age < 0 or age > max_age_s:
        return {"status": "pending_external_dependency", "reason": "device discovery is stale"}
    devices = []
    if target == "meter":
        devices = [d for d in discovery.get("hid_devices", [])
                   if d.get("vendor_id") == 0x2E3C and d.get("product_id") == 0x5558]
    elif target == "nicla":
        devices = [d for d in discovery.get("serial_ports", [])
                   if "nicla" in str(d.get("description", "")).lower()
                   or "nicla" in str(d.get("product", "")).lower()]
    elif target in {"nano", "usb_uart"}:
        devices = discovery.get("serial_ports", [])
    else:
        raise ValueError("unknown physical-device target: " + target)
    if not devices:
        return {"status": "pending_external_dependency", "reason": target + " absent from fresh enumeration"}
    return {"status": "ready_for_owner_review", "device_count": len(devices),
            "reason": "presence observed only; model identity and physical semantics still require owner verification"}
