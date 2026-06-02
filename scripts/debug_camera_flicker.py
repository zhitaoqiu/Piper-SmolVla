#!/usr/bin/env python3
"""相机波纹/频闪诊断工具。

枚举 /dev/video* 设备，检测格式、分辨率、帧率，读取样帧，尝试设置
power_line_frequency 和固定曝光，帮助排查滚动条纹问题。

用法:
    python scripts/debug_camera_flicker.py                          # 全设备诊断
    python scripts/debug_camera_flicker.py --device /dev/video6    # 单设备详细诊断
    python scripts/debug_camera_flicker.py --device /dev/video6    # 尝试防频闪+固定曝光
        --try-power-line 50 --try-fixed-exposure
        --exposures 80 100 120 160
    python scripts/debug_camera_flicker.py --save-dir logs/flicker_debug
"""

from __future__ import annotations

import argparse
import ctypes
import fcntl
import os
import struct
import sys
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import cv2
import numpy as np

# ── V4L2 ioctl constants (from linux/videodev2.h) ──────────────────────────

_IOC_NRBITS = 8
_IOC_TYPEBITS = 8
_IOC_SIZEBITS = 14

_IOC_NRSHIFT = 0
_IOC_TYPESHIFT = _IOC_NRSHIFT + _IOC_NRBITS
_IOC_SIZESHIFT = _IOC_TYPESHIFT + _IOC_TYPEBITS
_IOC_DIRSHIFT = _IOC_SIZESHIFT + _IOC_SIZEBITS

_IOC_NONE = 0
_IOC_WRITE = 1
_IOC_READ = 2


def _IOC(dir_, type_, nr, size):
    return (
        (dir_ << _IOC_DIRSHIFT)
        | (ord(type_) << _IOC_TYPESHIFT)
        | (nr << _IOC_NRSHIFT)
        | (size << _IOC_SIZESHIFT)
    )


def _IOR(type_, nr, size):
    return _IOC(_IOC_READ, type_, nr, ctypes.sizeof(size))


def _IOWR(type_, nr, size):
    return _IOC(_IOC_READ | _IOC_WRITE, type_, nr, ctypes.sizeof(size))


V4L2_CID_BASE = 0x00980900
V4L2_CID_POWER_LINE_FREQUENCY = V4L2_CID_BASE + 24

V4L2_CID_CAMERA_CLASS_BASE = 0x009A0900
V4L2_CID_EXPOSURE_AUTO = V4L2_CID_CAMERA_CLASS_BASE + 1
V4L2_CID_EXPOSURE_ABSOLUTE = V4L2_CID_CAMERA_CLASS_BASE + 2

V4L2_EXPOSURE_MANUAL = 1

V4L2_CTRL_TYPE_INTEGER = 1
V4L2_CTRL_TYPE_MENU = 3
V4L2_CTRL_FLAG_DISABLED = 0x0001


class v4l2_queryctrl(ctypes.Structure):
    _fields_ = [
        ("id", ctypes.c_uint32),
        ("type", ctypes.c_uint32),
        ("name", ctypes.c_char * 32),
        ("minimum", ctypes.c_int32),
        ("maximum", ctypes.c_int32),
        ("step", ctypes.c_int32),
        ("default_value", ctypes.c_int32),
        ("flags", ctypes.c_uint32),
        ("_reserved", ctypes.c_uint32 * 2),
    ]


class v4l2_control(ctypes.Structure):
    _fields_ = [
        ("id", ctypes.c_uint32),
        ("value", ctypes.c_int32),
    ]


VIDIOC_QUERYCTRL = _IOWR("V", 36, v4l2_queryctrl)
VIDIOC_G_CTRL = _IOWR("V", 27, v4l2_control)
VIDIOC_S_CTRL = _IOWR("V", 28, v4l2_control)

# ── helpers ──────────────────────────────────────────────────────────────────


def video_index(dev_path: str) -> int:
    return int(Path(dev_path).name.replace("video", ""))


def device_name(dev_path: str) -> str:
    try:
        dev = Path(dev_path)
        return (Path("/sys/class/video4linux") / dev.name / "name").read_text(encoding="utf-8").strip()
    except OSError:
        return "unknown"


def device_group(dev_path: str) -> str:
    try:
        dev = Path(dev_path)
        resolved = os.path.realpath(Path("/sys/class/video4linux") / dev.name / "device")
        parts = Path(resolved).parts
        for part in reversed(parts):
            if "-" in part and ":" not in part and part[0].isdigit():
                return part
    except OSError:
        pass
    return dev_path


def list_video_devices() -> list[str]:
    return sorted(
        str(p) for p in Path("/dev").glob("video*") if p.exists()
    )


def is_realsense(dev_path: str) -> bool:
    return "realsense" in device_name(dev_path).lower()


def is_5mp(dev_path: str) -> bool:
    return "5mp" in device_name(dev_path).lower()


def has_capture_capability(dev_path: str) -> bool:
    dev_name = Path(dev_path).name
    caps_path = Path("/sys/class/video4linux") / dev_name / "device" / "video4linux" / dev_name / "capabilities"
    if not caps_path.exists():
        return True  # can't read, assume yes and let open fail later
    try:
        caps = caps_path.read_text(encoding="utf-8").strip()
        for line in caps.split("\n"):
            if ":" in line:
                key, val = line.split(":", 1)
                if key.strip() == "V4L2_CAP_VIDEO_CAPTURE" and val.strip() == "1":
                    return True
    except OSError:
        pass
    # Fallback: try to open with OpenCV
    idx = video_index(dev_path)
    cap = cv2.VideoCapture(idx, cv2.CAP_V4L2)
    ok = cap.isOpened()
    cap.release()
    return ok


# ── V4L2 control helpers (uses Python fcntl + ctypes) ────────────────────────


def _v4l2_ioctl(dev_path: str, request: int, arg: Any) -> int:
    with open(dev_path, "rb+", buffering=0) as f:
        return fcntl.ioctl(f.fileno(), request, arg)


def query_ctrl(dev_path: str, ctrl_id: int) -> v4l2_queryctrl | None:
    q = v4l2_queryctrl()
    q.id = ctrl_id
    try:
        _v4l2_ioctl(dev_path, VIDIOC_QUERYCTRL, q)
        return q
    except OSError:
        return None


def get_ctrl(dev_path: str, ctrl_id: int) -> int | None:
    c = v4l2_control()
    c.id = ctrl_id
    try:
        _v4l2_ioctl(dev_path, VIDIOC_G_CTRL, c)
        return c.value
    except OSError:
        return None


def set_ctrl(dev_path: str, ctrl_id: int, value: int) -> bool:
    c = v4l2_control()
    c.id = ctrl_id
    c.value = value
    try:
        _v4l2_ioctl(dev_path, VIDIOC_S_CTRL, c)
        return True
    except OSError:
        return False


def ctrl_name(ctrl_id: int) -> str:
    names = {
        V4L2_CID_POWER_LINE_FREQUENCY: "power_line_frequency",
        V4L2_CID_EXPOSURE_AUTO: "exposure_auto",
        V4L2_CID_EXPOSURE_ABSOLUTE: "exposure_absolute",
    }
    return names.get(ctrl_id, f"0x{ctrl_id:08x}")


# ── frame stats ──────────────────────────────────────────────────────────────


def frame_stats(frame: np.ndarray | None) -> dict[str, Any]:
    if frame is None:
        return {"ok": False, "mean": None, "min": None, "max": None, "is_black": True}
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if frame.ndim == 3 else frame
    mean = float(gray.mean())
    return {
        "ok": True,
        "shape": tuple(frame.shape),
        "mean": mean,
        "min": int(gray.min()),
        "max": int(gray.max()),
        "is_black": mean < 5.0,
    }


# ── device probe ─────────────────────────────────────────────────────────────


def probe_device(dev_path: str) -> dict[str, Any]:
    idx = video_index(dev_path)
    info: dict[str, Any] = {
        "path": dev_path,
        "name": device_name(dev_path),
        "group": device_group(dev_path),
        "is_realsense": is_realsense(dev_path),
        "is_5mp": is_5mp(dev_path),
        "v4l2_open_ok": False,
        "power_line_freq_support": None,
        "power_line_freq_current": None,
        "exposure_auto_support": None,
        "exposure_absolute_support": None,
        "frames": [],
    }

    # Check V4L2 controls via ioctl
    q = query_ctrl(dev_path, V4L2_CID_POWER_LINE_FREQUENCY)
    if q is not None and not (q.flags & V4L2_CTRL_FLAG_DISABLED):
        info["power_line_freq_support"] = {
            "min": q.minimum,
            "max": q.maximum,
            "default": q.default_value,
            "type": "menu",
        }
        info["power_line_freq_current"] = get_ctrl(dev_path, V4L2_CID_POWER_LINE_FREQUENCY)

    q = query_ctrl(dev_path, V4L2_CID_EXPOSURE_AUTO)
    if q is not None and not (q.flags & V4L2_CTRL_FLAG_DISABLED):
        info["exposure_auto_support"] = {
            "min": q.minimum,
            "max": q.maximum,
            "default": q.default_value,
            "type": "menu",
        }

    q = query_ctrl(dev_path, V4L2_CID_EXPOSURE_ABSOLUTE)
    if q is not None and not (q.flags & V4L2_CTRL_FLAG_DISABLED):
        info["exposure_absolute_support"] = {
            "min": q.minimum,
            "max": q.maximum,
            "default": q.default_value,
            "step": q.step,
            "type": "int",
        }

    # Try to open with OpenCV V4L2 and read frames
    cap = cv2.VideoCapture(idx, cv2.CAP_V4L2)
    if cap.isOpened():
        info["v4l2_open_ok"] = True
        # Read actual width/height/fps
        info["actual_width"] = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        info["actual_height"] = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        info["actual_fps"] = cap.get(cv2.CAP_PROP_FPS)
        info["fourcc"] = int(cap.get(cv2.CAP_PROP_FOURCC))
        info["fourcc_str"] = "".join(
            chr((info["fourcc"] >> (8 * i)) & 0xFF) for i in range(4)
        )

        # Read multiple frames
        for _ in range(5):
            ret, frame = cap.read()
            if ret:
                info["frames"].append(frame_stats(frame))
        cap.release()
    else:
        info["v4l2_open_ok"] = False

    return info


# ── exposure sweep ───────────────────────────────────────────────────────────


def restore_ctrl(dev_path: str, ctrl_id: int, original_value: int | None) -> None:
    if original_value is not None:
        set_ctrl(dev_path, ctrl_id, original_value)


def sweep_exposure(
    dev_path: str,
    exposures: list[int],
    power_line_freq: int | None,
    save_dir: Path,
) -> list[dict[str, Any]]:
    results = []
    idx = video_index(dev_path)

    original_plf = get_ctrl(dev_path, V4L2_CID_POWER_LINE_FREQUENCY)
    original_exposure_auto = get_ctrl(dev_path, V4L2_CID_EXPOSURE_AUTO)
    original_exposure_abs = get_ctrl(dev_path, V4L2_CID_EXPOSURE_ABSOLUTE)

    # Set power line frequency if requested
    if power_line_freq is not None:
        ok = set_ctrl(dev_path, V4L2_CID_POWER_LINE_FREQUENCY, power_line_freq)
        if ok:
            print(f"  power_line_frequency -> {power_line_freq} ({'50Hz' if power_line_freq == 1 else '60Hz'}) OK")
        else:
            print(f"  power_line_frequency -> {power_line_freq} FAILED")

    # Disable auto exposure
    set_ctrl(dev_path, V4L2_CID_EXPOSURE_AUTO, V4L2_EXPOSURE_MANUAL)
    time.sleep(0.1)

    for exp_val in exposures:
        set_ctrl(dev_path, V4L2_CID_EXPOSURE_ABSOLUTE, exp_val)
        time.sleep(0.15)

        cap = cv2.VideoCapture(idx, cv2.CAP_V4L2)
        frames_stats = []
        saved_path = ""
        for attempt in range(3):
            ret, frame = cap.read()
            if ret:
                stats = frame_stats(frame)
                frames_stats.append(stats)
                if attempt == 0:
                    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
                    fname = f"exp_{exp_val}_{ts}.jpg"
                    saved_path = str(save_dir / fname)
                    cv2.imwrite(saved_path, frame)
        cap.release()

        avg_mean = float(np.mean([s["mean"] for s in frames_stats])) if frames_stats else 0
        print(f"  exposure={exp_val:>5d}  mean={avg_mean:7.2f}  "
              f"min={min((s['min'] for s in frames_stats), default=0):>3d}  "
              f"max={max((s['max'] for s in frames_stats), default=0):>3d}  "
              f"saved={saved_path}")

        results.append({
            "exposure": exp_val,
            "frames": frames_stats,
            "saved_path": saved_path,
        })

    # Restore original settings
    restore_ctrl(dev_path, V4L2_CID_POWER_LINE_FREQUENCY, original_plf)
    restore_ctrl(dev_path, V4L2_CID_EXPOSURE_AUTO, original_exposure_auto)
    restore_ctrl(dev_path, V4L2_CID_EXPOSURE_ABSOLUTE, original_exposure_abs)

    return results


# ── main ──────────────────────────────────────────────────────────────────────


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Camera flicker diagnostic tool")
    p.add_argument("--device", default="", help="Target a single /dev/videoX device")
    p.add_argument("--try-power-line", type=int, default=None, choices=[1, 2],
                   help="Set power_line_frequency: 1=50Hz, 2=60Hz")
    p.add_argument("--try-fixed-exposure", action="store_true",
                   help="Sweep fixed exposure values")
    p.add_argument("--exposures", type=int, nargs="+",
                   default=[80, 100, 120, 160, 200, 300, 400],
                   help="Exposure values to try (device-specific units)")
    p.add_argument("--save-dir", default="outputs/camera_flicker_debug")
    p.add_argument("--full-sweep", action="store_true",
                   help="Also try alternate formats (YUYV, MJPG) at each exposure")
    return p.parse_args()


def main() -> int:
    args = parse_args()

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    if args.device:
        devices = [args.device]
    else:
        devices = list_video_devices()

    print("=" * 70)
    print("Camera Flicker Diagnostic")
    print("=" * 70)
    print(f"devices found: {len(devices)}")
    print(f"save dir: {save_dir}")
    print()

    # ── Phase 1: Enumerate all devices ──────────────────────────────────────
    all_info: list[dict] = []
    for dev in devices:
        info = probe_device(dev)
        all_info.append(info)

        status = "OK" if info["v4l2_open_ok"] else "NO-OPEN"
        plf = info.get("power_line_freq_current")
        plf_str = f"plf={plf}" if plf is not None else "plf=N/A"
        print(f"  {dev:14s}  {info['name'][:40]:40s}  {status:7s}  {plf_str}  "
              f"groups={info['group']}")

        if info["v4l2_open_ok"] and info.get("frames"):
            s = info["frames"][0]
            print(f"    {info['fourcc_str']}  {info['actual_width']}x{info['actual_height']}  "
                  f"fps={info['actual_fps']:.1f}  "
                  f"mean={s['mean']:.1f}  min={s['min']}  max={s['max']}  "
                  f"black={s['is_black']}")
        elif info["v4l2_open_ok"]:
            print(f"    {info['fourcc_str']}  {info['actual_width']}x{info['actual_height']}  "
                  f"fps={info['actual_fps']:.1f}  NO-FRAMES")

    print()

    # ── Phase 2: Identify working RGB cameras ───────────────────────────────
    rgb_devices = [
        d for d in all_info if d["v4l2_open_ok"] and d.get("frames")
        and not d["frames"][0]["is_black"]
    ]
    if not rgb_devices:
        print("ERROR: no working RGB devices found")
        return 1

    print("Working RGB cameras:")
    for d in rgb_devices:
        s = d["frames"][0]
        print(f"  {d['path']:14s}  mean={s['mean']:.1f}  "
              f"{d['fourcc_str']}  {d['actual_width']}x{d['actual_height']}  "
              f"fps={d['actual_fps']:.1f}")
        if d.get("power_line_freq_support"):
            plf = d.get("power_line_freq_current")
            print(f"    power_line_frequency: supported  current={plf}  "
                  f"({d['power_line_freq_support']})")
        if d.get("exposure_auto_support"):
            print(f"    exposure_auto: supported  {d['exposure_auto_support']}")
        if d.get("exposure_absolute_support"):
            print(f"    exposure_absolute: supported  {d['exposure_absolute_support']}")
    print()

    # ── Phase 3: diagnostic sweep on target device ──────────────────────────
    target = args.device if args.device else (
        next((d["path"] for d in rgb_devices if d["is_5mp"]), None)
        or next((d["path"] for d in rgb_devices if not d["is_realsense"]), None)
        or next((d["path"] for d in rgb_devices), None)
    )

    if not target:
        print("no device to run sweep on")
        return 0

    print(f"Detailed sweep on: {target}  ({device_name(target)})")
    print("-" * 70)

    # Check and try power_line_frequency
    info = next((d for d in all_info if d["path"] == target), {})
    plf_support = info.get("power_line_freq_support")
    if plf_support:
        test_freqs = [args.try_power_line] if args.try_power_line else [1, 2]
        print(f"\n--- power_line_frequency test on {target} ---")
        for freq in test_freqs:
            ok = set_ctrl(target, V4L2_CID_POWER_LINE_FREQUENCY, freq)
            time.sleep(0.2)
            cap = cv2.VideoCapture(video_index(target), cv2.CAP_V4L2)
            for i in range(3):
                ret, frame = cap.read()
                if ret:
                    s = frame_stats(frame)
                    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
                    fname = f"plf_{'50' if freq == 1 else '60'}hz_frame{i}_{ts}.jpg"
                    path = str(save_dir / fname)
                    cv2.imwrite(path, frame)
                    print(f"  freq={freq}({'50' if freq==1 else '60'}Hz)  "
                          f"frame={i}  mean={s['mean']:.1f}  saved={path}")
            cap.release()
        # Restore default
        def_freq = plf_support["default"]
        set_ctrl(target, V4L2_CID_POWER_LINE_FREQUENCY, def_freq)
        if args.try_power_line is None:
            print(f"  restored power_line_frequency to default={def_freq}")
    else:
        print(f"\n  {target}: power_line_frequency NOT supported (or already locked by driver)")

    # ── Phase 4: Fixed exposure sweep ───────────────────────────────────────
    exp_support = info.get("exposure_absolute_support")
    if args.try_fixed_exposure and exp_support:
        print(f"\n--- fixed exposure sweep on {target} ---")
        sweep_results = sweep_exposure(
            target,
            exposures=args.exposures,
            power_line_freq=args.try_power_line,
            save_dir=save_dir,
        )

        best = None
        for r in sweep_results:
            if r["frames"]:
                m = r["frames"][0]["mean"]
                if best is None or (100 < m < 180):
                    best = r

        if best:
            print(f"\n  Recommended exposure: {best['exposure']}  "
                  f"mean={best['frames'][0]['mean']:.1f}")
        else:
            print(f"\n  No clearly better exposure found; keep auto-exposure")
    elif args.try_fixed_exposure:
        print(f"\n  {target}: exposure_absolute NOT supported")

    # ── Phase 5: also sweep RealSense wrist camera ──────────────────────────
    wrist_device = None
    for d in rgb_devices:
        if d["path"] != target and d["is_realsense"]:
            wrist_device = d["path"]
            break

    if wrist_device and args.try_fixed_exposure:
        print(f"\n--- RealSense (wrist) info on {wrist_device} ---")
        wrist_info = next((d for d in all_info if d["path"] == wrist_device), {})
        print(f"  V4L2 open: {wrist_info.get('v4l2_open_ok')}")
        print(f"  power_line_freq: {wrist_info.get('power_line_freq_support')}")

        # For RealSense, we use librealsense SDK which bypasses V4L2
        # Show what the libsense backend configures
        try:
            import pyrealsense2 as rs
            pipe = rs.pipeline()
            cfg = rs.config()
            cfg.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
            profile = pipe.start(cfg)
            dev = profile.get_device()

            for sensor in dev.query_sensors():
                print(f"\n  Sensor: {sensor.get_info(rs.camera_info.name)}")
                for opt_name, opt_enum in [
                    ("power_line_frequency", rs.option.power_line_frequency),
                    ("exposure", rs.option.exposure),
                    ("enable_auto_exposure", rs.option.enable_auto_exposure),
                    ("auto_exposure_priority", rs.option.auto_exposure_priority),
                    ("emitter_enabled", rs.option.emitter_enabled),
                ]:
                    try:
                        if sensor.supports(opt_enum):
                            val = sensor.get_option(opt_enum)
                            rng = sensor.get_option_range(opt_enum)
                            print(f"    {opt_name}: current={val}  "
                                  f"range=[{rng.min}, {rng.max}]  default={rng.default}")
                        else:
                            print(f"    {opt_name}: NOT SUPPORTED")
                    except Exception:
                        print(f"    {opt_name}: ERROR reading")

            # Read a sample frame
            for _ in range(30):
                pipe.wait_for_frames()
            frames = pipe.wait_for_frames()
            color = frames.get_color_frame()
            if color:
                arr = np.asanyarray(color.get_data())
                ts = datetime.now().strftime("%Y%m%d_%H%M%S")
                path = str(save_dir / f"realsense_sample_{ts}.jpg")
                cv2.imwrite(path, arr)
                s = frame_stats(arr)
                print(f"\n  RealSense frame: mean={s['mean']:.1f}  saved={path}")

            pipe.stop()
        except ImportError:
            print("  pyrealsense2 not available in this env")
        except Exception as e:
            print(f"  RealSense probe error: {e}")

    # ── Final report ────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("FINAL REPORT")
    print("=" * 70)

    # Identify which is global (5MP) and which is wrist (RealSense)
    global_dev = next((d["path"] for d in all_info if d["is_5mp"] and d["v4l2_open_ok"]), None)
    wrist_dev = next((d["path"] for d in all_info if d["is_realsense"] and d["v4l2_open_ok"]), None)

    print(f"  Global camera: {global_dev or 'NOT FOUND'}  ({device_name(global_dev or 'N/A')})")
    print(f"  Wrist camera:  {wrist_dev or 'NOT FOUND'}  ({device_name(wrist_dev or 'N/A')})")
    print()

    # Recommended settings
    print("  Recommended settings:")
    print(f"    power_line_frequency: 50Hz (value=1, for China AC mains)")

    for d in all_info:
        if d["v4l2_open_ok"] and d.get("frames"):
            s = d["frames"][0]
            print(f"    {d['path']}: {d['fourcc_str']}  "
                  f"{d['actual_width']}x{d['actual_height']}  "
                  f"fps={d['actual_fps']:.1f}  mean={s['mean']:.1f}")

    print()
    print("  5MP USB camera V4L2 notes:")
    print("    - power_line_frequency control: usually NOT available on UVC cameras")
    print("    - exposure_absolute control: depends on camera firmware")
    print("    - Rolling bars on UVC cameras are often from auto-exposure hunting")
    print("    - If controls are unavailable, use lighting that is DC-powered or")
    print("      set the room lights to maximum brightness (reduces flicker depth)")
    print()
    print("  RealSense D435i notes:")
    print("    - V4L2 driver is BROKEN for color streaming (corrupted buffers)")
    print("    - Use pyrealsense2 SDK (bypasses V4L2)")
    print("    - emitter_enabled=0, power_line_frequency=1 (50Hz) set via SDK")
    print("    - auto_exposure_priority=0 may help, but NOT exposure lock")
    print()
    print(f"  Sample images saved to: {save_dir}")
    print()
    print("  Robot motion:   NO (no commands sent)")
    print("  Dataset write:  NO")
    print("  Training:       NO")
    print("  Policy:         NO")
    print("  Piper control:  NO")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
