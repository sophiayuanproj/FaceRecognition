"""Shared, MicroPython-safe camera/FOMO geometry contract.

OpenMV's FOMO postprocessor returns rectangles in the camera image's pixel
space.  The detector tensor grid is an internal model space; it must not be
applied to these returned rectangles a second time.

All public coordinates are (x, y, width, height) or (x, y), in pixels.
"""

from identity_contracts import centered_roi_clamped, fomo_center_camera_px


FOMO_RECT_SPACE = "camera_xywh_px"
ROI_SPACE = "camera_xywh_px"


def roi_clip_plan(roi_xywh, frame_width, frame_height):
    """Return source/destination bounds and black-padding fraction for a ROI.

    Kept for explicit padded-crop investigations.  The corrected clamped
    policy uses fully in-frame ROIs, so its padding fraction is zero.
    """
    x, y, w, h = [int(v) for v in roi_xywh]
    if w <= 0 or h <= 0:
        raise ValueError("empty ROI")
    sx0, sy0 = max(0, x), max(0, y)
    sx1, sy1 = min(frame_width, x + w), min(frame_height, y + h)
    cw, ch = max(0, sx1 - sx0), max(0, sy1 - sy0)
    retained = float(cw * ch) / float(w * h)
    return {
        "source_xywh": (sx0, sy0, cw, ch),
        "destination_xywh": (sx0 - x, sy0 - y, cw, ch),
        "black_padding_fraction": 1.0 - retained,
    }


def frame_row_col(x_camera_px, y_camera_px):
    """Convert named Cartesian coordinates to array order (row, column)."""
    return (int(y_camera_px), int(x_camera_px))
