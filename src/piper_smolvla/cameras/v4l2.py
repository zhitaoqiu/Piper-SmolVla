"""OpenCV/V4L2 camera driver for UVC-style USB cameras."""

from __future__ import annotations

import time
from typing import Any

import numpy as np

from piper_smolvla.camera_utils import normalize_video_device, video_index
from piper_smolvla.cameras.types import CameraControls


class V4L2Camera:
    color_order = "bgr"

    def __init__(
        self,
        device: str,
        config: Any,
        *,
        controls: CameraControls | None = None,
    ):
        import cv2

        device = normalize_video_device(device)
        self._cap = open_v4l2_capture(device, cv2, timeout_sec=config.read_timeout_sec * 2)

        self._cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, config.width)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.height)
        self._cap.set(cv2.CAP_PROP_FPS, config.fps)

        self.width = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.fps = self._cap.get(cv2.CAP_PROP_FPS)

        self._read_timeout_sec = config.read_timeout_sec
        self._max_consecutive_timeouts = config.max_consecutive_timeouts
        self._consecutive_timeouts = 0

        self._configure_controls(controls or CameraControls())

        frame = read_frame_with_timeout(
            self._cap,
            device,
            timeout_sec=self._read_timeout_sec * 2,
            warmup_frames=config.warmup_frames,
        )
        if frame is None:
            self.release()
            raise RuntimeError(f"V4L2 camera {device}: failed to read frame (timeout or no data)")
        if float(np.asarray(frame, dtype=np.uint8).mean()) < config.black_threshold:
            self.release()
            raise RuntimeError(
                f"V4L2 camera {device}: frame appears black "
                f"(mean={float(np.asarray(frame, dtype=np.uint8).mean()):.1f} < {config.black_threshold})"
            )

    def _configure_controls(self, controls: CameraControls) -> None:
        for name, prop_id, value in (
            ("power_line_frequency", getattr(__import__("cv2"), "CAP_PROP_POWERLINE_FREQUENCY", None), controls.power_line_frequency),
            ("exposure_time_absolute", getattr(__import__("cv2"), "CAP_PROP_EXPOSURE", None), controls.exposure_absolute),
            ("auto_exposure", getattr(__import__("cv2"), "CAP_PROP_AUTO_EXPOSURE", None), controls.auto_exposure),
            ("gain", getattr(__import__("cv2"), "CAP_PROP_GAIN", None), controls.gain),
            ("brightness", getattr(__import__("cv2"), "CAP_PROP_BRIGHTNESS", None), controls.brightness),
        ):
            if value is None or prop_id is None:
                continue
            try:
                ok = self._cap.set(prop_id, value)
                actual = self._cap.get(prop_id)
                if abs(actual - value) > 0.5 and ok:
                    print(f"  [V4L2] {name}: set {value}, got {actual} (driver may have clamped)")
            except Exception:
                print(f"  [V4L2] {name}: not supported by this device")

    @property
    def unhealthy(self) -> bool:
        return self._consecutive_timeouts >= self._max_consecutive_timeouts

    def read(self) -> tuple[bool, np.ndarray | None]:
        import threading

        result: dict[str, object] = {"ret": False, "frame": None}

        def _do_read() -> None:
            try:
                ret, frame = self._cap.read()
                result["ret"] = ret
                if ret and frame is not None:
                    result["frame"] = np.asarray(frame, dtype=np.uint8)
            except Exception:
                pass

        t = threading.Thread(target=_do_read, daemon=True)
        t.start()
        t.join(timeout=self._read_timeout_sec)
        if t.is_alive():
            self._consecutive_timeouts += 1
            print(
                f"  [V4L2] read timed out after {self._read_timeout_sec:.1f}s "
                f"(consecutive={self._consecutive_timeouts}/{self._max_consecutive_timeouts})"
            )
            if self.unhealthy:
                print(
                    f"  [V4L2] camera marked unhealthy after "
                    f"{self._consecutive_timeouts} consecutive timeouts"
                )
            return False, None

        self._consecutive_timeouts = 0
        if not result["ret"]:
            return False, None
        return True, result["frame"]

    def release(self) -> None:
        try:
            self._cap.release()
        except Exception:
            pass


def open_v4l2_capture(device: str, cv2_module: Any, *, timeout_sec: float = 4.0) -> Any:
    """Open a UVC camera robustly across OpenCV/V4L2 builds."""

    import threading

    sources: list[tuple[Any, int, str]] = []
    if device.startswith("/dev/video"):
        sources.append((video_index(device), cv2_module.CAP_V4L2, "index+CAP_V4L2"))
        sources.append((device, cv2_module.CAP_V4L2, "path+CAP_V4L2"))
        sources.append((video_index(device), cv2_module.CAP_ANY, "index+CAP_ANY"))
        sources.append((device, cv2_module.CAP_ANY, "path+CAP_ANY"))
    else:
        sources.append((device, cv2_module.CAP_V4L2, "device+CAP_V4L2"))
        sources.append((device, cv2_module.CAP_ANY, "device+CAP_ANY"))

    attempted: list[str] = []
    for _ in range(3):
        for source, backend, label in sources:
            result: dict[str, Any] = {"cap": None}
            exc_info: dict[str, BaseException | None] = {"exc": None}

            def _do_open(src: Any = source, be: int = backend) -> None:
                try:
                    result["cap"] = cv2_module.VideoCapture(src, be)
                except Exception as exc:
                    exc_info["exc"] = exc

            t = threading.Thread(target=_do_open, daemon=True)
            t.start()
            t.join(timeout=timeout_sec)
            if t.is_alive():
                attempted.append(f"{label}(timeout)")
                print(
                    f"  [V4L2] VideoCapture({source}, {label}) timed out "
                    f"after {timeout_sec:.1f}s"
                )
                continue

            if exc_info["exc"] is not None:
                attempted.append(f"{label}({type(exc_info['exc']).__name__})")
                continue

            cap = result["cap"]
            if cap is None:
                attempted.append(label)
                continue

            attempted.append(label)
            if cap.isOpened():
                return cap
            try:
                cap.release()
            except Exception:
                pass
        time.sleep(0.15)
    raise RuntimeError(f"cannot open V4L2 camera: {device}; attempted={attempted}")


def read_frame_with_timeout(
    cap: Any,
    device: str,
    *,
    timeout_sec: float = 4.0,
    warmup_frames: int = 5,
) -> np.ndarray | None:
    """Read a frame from a V4L2 capture with timeout, discarding warmup frames."""

    import threading

    result: dict[str, np.ndarray | None] = {"frame": None}

    def _do_read() -> None:
        try:
            for _ in range(max(1, warmup_frames)):
                cap.read()
            ret, frame = cap.read()
            if ret and frame is not None:
                result["frame"] = np.asarray(frame, dtype=np.uint8)
        except Exception:
            pass

    t = threading.Thread(target=_do_read, daemon=True)
    t.start()
    t.join(timeout=timeout_sec)
    if t.is_alive():
        print(f"  [V4L2] {device}: read timed out after {timeout_sec:.0f}s - "
              f"camera driver may be stuck; try re-plugging the USB cable")
        return None
    return result["frame"]


_V4L2Camera = V4L2Camera
_open_v4l2_capture = open_v4l2_capture
_read_frame_with_timeout = read_frame_with_timeout
