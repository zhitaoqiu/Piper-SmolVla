"""Image conversion helpers for camera readers."""

from __future__ import annotations

from typing import Any

import numpy as np


def read_rgb(camera: Any, key: str, *, threshold: float = 5.0) -> np.ndarray:
    ret, frame = camera.read()
    if not ret:
        raise RuntimeError(f"{key}: failed to read frame")
    if frame.ndim == 2:
        import cv2

        frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
    color_order = getattr(camera, "color_order", "bgr").lower()
    if frame.shape[-1] == 4:
        import cv2

        if color_order == "bgra":
            frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2RGB)
        elif color_order == "rgba":
            frame = cv2.cvtColor(frame, cv2.COLOR_RGBA2RGB)
        else:
            raise RuntimeError(f"{key}: unsupported 4-channel color order {color_order!r}")
    elif frame.shape[-1] == 3:
        if color_order == "bgr":
            import cv2

            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        elif color_order == "rgb":
            pass
        else:
            raise RuntimeError(f"{key}: unsupported 3-channel color order {color_order!r}")
    arr = np.asarray(frame, dtype=np.uint8)
    if arr.size == 0:
        raise RuntimeError(f"{key}: empty frame")
    if float(arr.mean()) < threshold:
        raise RuntimeError(f"{key}: appears black (mean={arr.mean():.1f} < {threshold})")
    return arr


def is_black_frame(image: Any, *, threshold: float = 5.0) -> bool:
    arr = np.asarray(image)
    if arr.size == 0:
        return True
    return float(arr.mean()) < threshold


_read_rgb = read_rgb
