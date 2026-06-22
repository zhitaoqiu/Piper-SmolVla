#!/usr/bin/env python3
"""RealSense D435i 传感器控件调试。

列出 RGB 传感器所有可调选项、当前值和范围，可交互式调节曝光/增益等。

用法:
    python scripts/debug_realsense.py                          # 列出选项和当前值
    python scripts/debug_realsense.py --auto-exposure 0 --exposure 15000 --gain 128  # 手动曝光
    python scripts/debug_realsense.py --interactive            # 交互式调节
"""

from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np
import cv2

cv2.setLogLevel(0)
os.environ.setdefault("OPENCV_LOG_LEVEL", "ERROR")


def main() -> int:
    args = parse_args()

    try:
        import pyrealsense2 as rs
    except ImportError:
        print("pyrealsense2 未安装: pip install pyrealsense2")
        return 1

    ctx = rs.context()
    devices = ctx.query_devices()
    if len(devices) == 0:
        print("No RealSense devices found")
        return 1

    # ── pick device ──────────────────────────────────────────────────────
    if args.list_devices:
        print(f"{len(devices)} RealSense device(s):")
        for i, d in enumerate(devices):
            print(f"  [{i}] {d.get_info(rs.camera_info.name)}  "
                  f"serial={d.get_info(rs.camera_info.serial_number)}  "
                  f"bus={d.get_info(rs.camera_info.usb_type_descriptor)}")
        return 0

    if args.device_index is not None:
        if args.device_index >= len(devices):
            print(f"Device index {args.device_index} out of range (found {len(devices)})")
            return 1
        target = devices[args.device_index]
    elif args.serial:
        target = None
        for d in devices:
            sn = d.get_info(rs.camera_info.serial_number)
            if sn == args.serial or args.serial in sn:
                target = d
                break
        if target is None:
            print(f"RealSense with serial containing '{args.serial}' not found.")
            print("Available devices:")
            for d in devices:
                print(f"  {d.get_info(rs.camera_info.name)}  serial={d.get_info(rs.camera_info.serial_number)}")
            return 1
    else:
        target = devices[0]

    name = target.get_info(rs.camera_info.name)
    serial = target.get_info(rs.camera_info.serial_number)
    print(f"Device: {name}  serial={serial}")

    # ── start pipeline ───────────────────────────────────────────────────
    pipe = rs.pipeline()
    cfg = rs.config()
    cfg.enable_device(serial)
    cfg.enable_stream(rs.stream.color, args.width, args.height, rs.format.bgr8, args.fps)

    try:
        profile = pipe.start(cfg)
    except RuntimeError as e:
        print(f"Failed to start stream: {e}")
        return 1

    dev = profile.get_device()
    try:
        rgb_sensor = dev.first_color_sensor()
    except RuntimeError:
        rgb_sensor = dev.first_depth_sensor()  # D405: color is from depth/stereo sensor
    depth_sensor = dev.first_depth_sensor()

    # ── list options ─────────────────────────────────────────────────────
    if args.list_options:
        print("\n=== RGB sensor options ===")
        _print_sensor_options(rgb_sensor, rs)
        print("\n=== Depth sensor options ===")
        _print_sensor_options(depth_sensor, rs)
        pipe.stop()
        return 0

    # ── apply settings ───────────────────────────────────────────────────
    changed = _set_realsense_option(rgb_sensor, rs, rs.option.enable_auto_exposure, args.auto_exposure)
    changed |= _set_realsense_option(rgb_sensor, rs, rs.option.exposure, args.exposure)
    changed |= _set_realsense_option(rgb_sensor, rs, rs.option.gain, args.gain)
    changed |= _set_realsense_option(rgb_sensor, rs, rs.option.brightness, args.brightness)
    changed |= _set_realsense_option(rgb_sensor, rs, rs.option.gamma, args.gamma)
    changed |= _set_realsense_option(rgb_sensor, rs, rs.option.contrast, args.contrast)
    changed |= _set_realsense_option(rgb_sensor, rs, rs.option.sharpness, args.sharpness)
    changed |= _set_realsense_option(rgb_sensor, rs, rs.option.saturation, args.saturation)
    changed |= _set_realsense_option(rgb_sensor, rs, rs.option.white_balance, args.white_balance)
    changed |= _set_realsense_option(rgb_sensor, rs, rs.option.enable_auto_white_balance,
                                     0 if args.white_balance is not None else None)
    changed |= _set_realsense_option(depth_sensor, rs, rs.option.emitter_enabled,
                                     0 if args.emitter_off else None)
    changed |= _set_realsense_option(depth_sensor, rs, rs.option.laser_power,
                                     args.laser_power)

    if not changed and not args.interactive:
        # No settings requested — just show current values
        print("\nCurrent RGB sensor state (no settings applied):")
        _print_key_options(rgb_sensor, rs)

    # ── interactive mode ─────────────────────────────────────────────────
    if args.interactive:
        return _interactive_loop(pipe, rgb_sensor, rs, args)

    # ── one-shot preview ─────────────────────────────────────────────────
    # Warmup
    for _ in range(10):
        pipe.wait_for_frames(timeout_ms=1000)

    print("\nPreview (press 'q' to quit, 's' to save snapshot)...")
    frame_count = 0
    fps_t0 = time.monotonic()
    fps_display = 0.0

    try:
        while True:
            try:
                frames = pipe.wait_for_frames(timeout_ms=1000)
            except RuntimeError:
                time.sleep(0.05)
                continue
            color = frames.get_color_frame()
            if not color:
                print("no frame")
                continue
            img = np.asanyarray(color.get_data())
            arr = np.asarray(img, dtype=np.float64)
            display = cv2.resize(img, (640, 480))
            cv2.putText(display, f"mean={arr.mean():.0f}  gain={_get_current(rgb_sensor, rs, rs.option.gain):.0f}",
                        (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            cv2.putText(display, f"exposure={_get_current(rgb_sensor, rs, rs.option.exposure):.0f}",
                        (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            cv2.putText(display, f"auto_exp={_get_current(rgb_sensor, rs, rs.option.enable_auto_exposure):.0f}  "
                        f"fps={fps_display:.1f}",
                        (10, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            cv2.imshow("RealSense Debug", display)

            frame_count += 1
            elapsed = time.monotonic() - fps_t0
            if elapsed >= 0.5:
                fps_display = frame_count / elapsed
                frame_count = 0
                fps_t0 = time.monotonic()

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            elif key == ord("s"):
                ts = time.strftime("%Y%m%d_%H%M%S")
                cv2.imwrite(f"outputs/realsense_{ts}.jpg", img)
                print(f"saved: outputs/realsense_{ts}.jpg")
    except KeyboardInterrupt:
        pass
    finally:
        pipe.stop()
        cv2.destroyAllWindows()

    return 0


# ── interactive mode ──────────────────────────────────────────────────────────

def _interactive_loop(pipe, rgb_sensor, rs, args) -> int:
    """Interactive keyboard-driven control."""
    import cv2

    gain = _get_current(rgb_sensor, rs, rs.option.gain) or 64
    exposure = _get_current(rgb_sensor, rs, rs.option.exposure) or 15000
    auto_exp = _get_current(rgb_sensor, rs, rs.option.enable_auto_exposure) or 1

    print("\nInteractive mode:")
    print("  a/d  — auto_exposure on/off")
    print("  up/down  — exposure +/- 1000")
    print("  left/right  — gain +/- 8")
    print("  r  — reset to defaults")
    print("  s  — print current values")
    print("  q  — quit\n")
    print("Warming up...")

    # Warmup — D405 needs more time for initial frames
    warmup_deadline = time.monotonic() + 3.0
    while time.monotonic() < warmup_deadline:
        try:
            pipe.wait_for_frames(timeout_ms=1000)
        except RuntimeError:
            time.sleep(0.1)
    print("Ready.\n")

    cv2.namedWindow("RealSense Debug (interactive)", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("RealSense Debug (interactive)", 640, 480)
    frame_count = 0
    print("Entering preview loop...")

    try:
        while True:
            try:
                frames = pipe.wait_for_frames(timeout_ms=1000)
            except RuntimeError:
                time.sleep(0.05)
                continue
            color = frames.get_color_frame()
            if not color:
                print("  (no color frame)")
                time.sleep(0.05)
                continue
            img = np.asanyarray(color.get_data())
            if frame_count == 0:
                print(f"  first frame: shape={img.shape}  dtype={img.dtype}")
                frame_count = 1
            arr = np.asarray(img, dtype=np.float64)
            display = cv2.resize(img, (640, 480))
            cv2.putText(display, f"auto_exp={auto_exp}  exposure={exposure}  gain={gain}",
                        (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            cv2.putText(display, f"mean={arr.mean():.0f}  min={arr.min():.0f}  max={arr.max():.0f}",
                        (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            cv2.putText(display, "a/d:auto  up/dn:exp  l/r:gain  s:print  r:reset  q:quit",
                        (10, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 2)
            cv2.imshow("RealSense Debug (interactive)", display)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            elif key == ord("a"):
                auto_exp = 1
                _set_realsense_option(rgb_sensor, rs, rs.option.enable_auto_exposure, 1)
            elif key == ord("d"):
                auto_exp = 0
                _set_realsense_option(rgb_sensor, rs, rs.option.enable_auto_exposure, 0)
            elif key == 82:  # up
                exposure += 1000
                _set_realsense_option(rgb_sensor, rs, rs.option.exposure, exposure)
            elif key == 84:  # down
                exposure = max(1, exposure - 1000)
                _set_realsense_option(rgb_sensor, rs, rs.option.exposure, exposure)
            elif key == 81:  # left
                gain = max(16, gain - 8)
                _set_realsense_option(rgb_sensor, rs, rs.option.gain, gain)
            elif key == 83:  # right
                gain = min(248, gain + 8)
                _set_realsense_option(rgb_sensor, rs, rs.option.gain, gain)
            elif key == ord("r"):
                auto_exp, exposure, gain = 1, 15000, 64
                _set_realsense_option(rgb_sensor, rs, rs.option.enable_auto_exposure, 1)
                _set_realsense_option(rgb_sensor, rs, rs.option.exposure, 15000)
                _set_realsense_option(rgb_sensor, rs, rs.option.gain, 64)
                print("reset to defaults")
            elif key == ord("s"):
                print(f"auto_exp={auto_exp}  exposure={exposure}  gain={gain}  "
                      f"mean={arr.mean():.0f}")
    except KeyboardInterrupt:
        pass
    finally:
        pipe.stop()
        cv2.destroyAllWindows()
    return 0


# ── helpers ───────────────────────────────────────────────────────────────────

def _print_sensor_options(sensor, rs) -> None:
    for opt in sorted(sensor.get_supported_options(), key=lambda o: str(o)):
        try:
            rng = sensor.get_option_range(opt)
            current = sensor.get_option(opt)
            desc = sensor.get_option_description(opt)
            print(f"  {str(opt):40s} = {current:>8.1f}   range=[{rng.min:.1f}, {rng.max:.1f}] step={rng.step:.1f}   "
                  f"# {desc}")
        except Exception:
            print(f"  {str(opt):40s}  (unable to read)")


def _print_key_options(sensor, rs) -> None:
    keys = [
        (rs.option.enable_auto_exposure, "enable_auto_exposure"),
        (rs.option.exposure, "exposure"),
        (rs.option.gain, "gain"),
        (rs.option.brightness, "brightness"),
        (rs.option.gamma, "gamma"),
        (rs.option.contrast, "contrast"),
        (rs.option.white_balance, "white_balance"),
    ]
    for opt, label in keys:
        try:
            val = sensor.get_option(opt)
            rng = sensor.get_option_range(opt)
            print(f"  {label:30s} = {val:>8.1f}   range=[{rng.min:.1f}, {rng.max:.1f}]")
        except Exception:
            pass


def _get_current(sensor, rs, option) -> float | None:
    try:
        return sensor.get_option(option)
    except Exception:
        return None


def _set_realsense_option(sensor, rs, option, value) -> bool:
    if value is None:
        return False
    try:
        if sensor.supports(option):
            sensor.set_option(option, float(value))
            print(f"  set {str(option):40s} = {value}")
            return True
        else:
            print(f"  SKIP {str(option):40s} (not supported)")
            return False
    except Exception as exc:
        print(f"  FAIL {str(option):40s} = {value}  ({exc})")
        return False


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="RealSense D435i sensor debug tool")
    p.add_argument("--serial", default="", help="Device serial number (substring match)")
    p.add_argument("--device-index", type=int, default=None, help="Device index (0, 1, ...)")
    p.add_argument("--list-devices", action="store_true", help="List all RealSense devices and exit")
    p.add_argument("--width", type=int, default=640)
    p.add_argument("--height", type=int, default=480)
    p.add_argument("--fps", type=int, default=30)
    p.add_argument("--list-options", action="store_true", help="List all sensor options and exit")
    p.add_argument("--auto-exposure", type=float, default=None, help="0=off, 1=on")
    p.add_argument("--exposure", type=float, default=None, help="Exposure in microseconds")
    p.add_argument("--gain", type=float, default=None, help="Gain (typically 16-248)")
    p.add_argument("--brightness", type=float, default=None, help="Brightness offset")
    p.add_argument("--gamma", type=float, default=None, help="Gamma (typically 100-500)")
    p.add_argument("--contrast", type=float, default=None, help="Contrast")
    p.add_argument("--sharpness", type=float, default=None, help="Sharpness (0-100)")
    p.add_argument("--saturation", type=float, default=None, help="Saturation (0-100)")
    p.add_argument("--white-balance", type=float, default=None, help="Manual white balance (2800-6500 K)")
    p.add_argument("--emitter-off", action="store_true", help="Turn off IR emitter")
    p.add_argument("--laser-power", type=float, default=None, help="Laser power")
    p.add_argument("--interactive", action="store_true", help="Interactive tuning mode")
    return p.parse_args()


if __name__ == "__main__":
    raise SystemExit(main())
