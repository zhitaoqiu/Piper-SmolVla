#!/usr/bin/env python3
"""相机调试预览：打开 global / wrist 摄像头，显示实时画面，按 q 退出。

单相机模式：只有一个相机时自动降级，或用 --single-camera 显式指定，
同一路图像同时充当 global 和 wrist。

用法:
    python scripts/debug_cameras.py                          # 自动检测
    python scripts/debug_cameras.py --single-camera          # 强制单相机模式
    python scripts/debug_cameras.py --list-only              # 列出设备（轻量探测）
    python scripts/debug_cameras.py --snapshot-only          # 只截图不显示窗口
    python scripts/debug_cameras.py --global-camera realsense:243222074879 --wrist-camera realsense:260322275595
"""

from __future__ import annotations

import _bootstrap  # noqa: F401

import argparse
import os
import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

# Suppress OpenCV internal V4L2 warnings during device probe
cv2.setLogLevel(0)
os.environ.setdefault("OPENCV_LOG_LEVEL", "ERROR")

from piper_smolvla.camera_utils import (
    is_explicit,
    is_realsense_device,
    list_realsense_physical_devices,
    list_video_devices,
    normalize_video_device,
    probe_readable_v4l2_devices_detailed,
    resolve_camera_pair,
    video_device_group,
    video_device_name,
)
from piper_smolvla.cameras import (
    CameraControls,
    DEFAULT_CAMERA_FPS,
    DEFAULT_CAMERA_HEIGHT,
    DEFAULT_CAMERA_WIDTH,
    DEFAULT_GLOBAL_CAMERA,
    DEFAULT_WRIST_CAMERA,
    RealCameraConfig,
    RealSenseCamera,
    V4L2Camera,
)
from piper_smolvla.cameras.presets import camera_control_defaults
from piper_smolvla.cameras.types import merge_camera_controls


# ── main ──────────────────────────────────────────────────────────────────────

def main() -> int:
    args = parse_args()

    if args.list_only:
        return _cmd_list_only()

    # ── resolve cameras ──────────────────────────────────────────────────
    single_camera = args.single_camera
    global_dev: str = ""
    wrist_dev: str = ""

    if is_explicit(args.global_camera) and is_explicit(args.wrist_camera):
        global_dev = normalize_video_device(args.global_camera)
        wrist_dev = normalize_video_device(args.wrist_camera)
        if single_camera:
            print("--single-camera: using global camera only, ignoring --wrist-camera")
            wrist_dev = global_dev
        print("camera assignment mode = explicit")
        print(f"  global: {global_dev}  name={video_device_name(global_dev)!r}  group={video_device_group(global_dev)}")
        if not single_camera:
            print(f"  wrist:  {wrist_dev}  name={video_device_name(wrist_dev)!r}  group={video_device_group(wrist_dev)}")
    else:
        print("probing /dev/video* devices (max 2.5 s per device)...\n")
        devices_info = probe_readable_v4l2_devices_detailed(verbose=True, skip_realsense=True)
        readable_v4l2 = [d.device for d in devices_info if d.shape is not None and not d.black and d.status == "ok"]
        realsense_info = list_realsense_physical_devices()
        realsense_specs = [d.spec for d in realsense_info]
        if realsense_info:
            print("\nRealSense physical device(s) via librealsense:")
            for info in realsense_info:
                nodes = ",".join(info.video_nodes) if info.video_nodes else "no /dev/video mapping"
                groups = ",".join(info.groups) if info.groups else "unknown-group"
                print(
                    f"  {info.spec}  name={info.name!r}  usb={info.usb_type or 'unknown'}  "
                    f"group={groups}  nodes={nodes}"
                )

        readable = readable_v4l2 + realsense_specs
        if not readable:
            all_devs = list_video_devices()
            print(f"\nNo readable /dev/video* found. {len(all_devs)} total devices exist:")
            for d in all_devs:
                print(f"  {d}  name={video_device_name(d)!r}  group={video_device_group(d)}")
            _print_explicit_hint(all_devs)
            return 1

        print(f"\n{len(readable)} readable device(s):")
        for d in readable:
            print(f"  {d}  name={video_device_name(d)!r}  group={video_device_group(d)}  "
                  f"{'REALSENSE' if is_realsense_device(d) else 'UVC'}")

        # Single physical camera detection: only 1 readable node, OR all
        # readable nodes share the same USB group (e.g. RealSense exposes
        # multiple /dev/video* from the same physical device).
        unique_groups = {video_device_group(d) for d in readable}
        all_same_group = len(unique_groups) == 1
        auto_single = (not single_camera) and (len(readable) == 1 or all_same_group)

        if single_camera or auto_single:
            single_camera = True
            # Prefer non-RealSense UVC first, then RealSense.
            def _single_cam_score(d: str) -> tuple[int, float]:
                name = video_device_name(d).lower()
                if "5mp" in name or "usb camera" in name:
                    return (0, 0.0)
                if "realsense" in name:
                    return (2, 0.0)
                return (1, 0.0)

            ranked = sorted(readable, key=_single_cam_score)
            global_dev = ranked[0]
            wrist_dev = global_dev

            if auto_single:
                if all_same_group and len(readable) > 1:
                    g = unique_groups.pop()
                    print(f"\nAll {len(readable)} readable nodes share USB group "
                          f"'{g}' — auto single-camera mode.")
                else:
                    print("\nOnly 1 readable camera — auto single-camera mode.")
            print(f"  single-camera mode: using {global_dev} for both global & wrist")
        else:
            try:
                global_dev, wrist_dev = resolve_camera_pair(args.global_camera, args.wrist_camera, devices=readable)
            except RuntimeError as exc:
                print(f"auto-resolve failed: {exc}")
                _print_explicit_hint(readable)
                return 1

    # ── same-device / same-group check (skip in single-camera mode) ──────
    if not single_camera:
        if global_dev == wrist_dev:
            print(f"ERROR: global and wrist resolved to same device: {global_dev}")
            return 1
        global_group = video_device_group(global_dev)
        wrist_group = video_device_group(wrist_dev)
        if global_group == wrist_group:
            print(f"ERROR: global ({global_dev}) and wrist ({wrist_dev}) share USB group '{global_group}'.")
            print("Refusing — use --global-camera /dev/videoX --wrist-camera /dev/videoY from different USB buses.")
            return 1

    # ── open cameras ─────────────────────────────────────────────────────
    readers: dict[str, V4L2Camera | RealSenseCamera] = {}
    opened_roles = ["global"] if single_camera else ["global", "wrist"]
    device_map: dict[str, str] = {"global": global_dev}
    if not single_camera:
        device_map["wrist"] = wrist_dev

    # Allow V4L2 probe handles to fully release before opening RealSense devices
    if any(is_realsense_device(d) for d in device_map.values()):
        time.sleep(0.5)

    try:
        for role in opened_roles:
            dev = device_map[role]
            w = args.width
            h = args.height
            fp = args.fps

            # Per-role overrides: wrist-specific args take priority, then common args
            backend = args.wrist_backend if role == "wrist" and args.wrist_backend != "auto" else args.backend
            exposure = args.wrist_exposure if role == "wrist" and args.wrist_exposure is not None else args.exposure
            auto_exp = args.wrist_auto_exposure if role == "wrist" and args.wrist_auto_exposure is not None else args.auto_exposure
            power_line = args.wrist_power_line if role == "wrist" and args.wrist_power_line is not None else args.power_line
            gain = args.wrist_gain if role == "wrist" and args.wrist_gain is not None else args.gain
            brightness = args.wrist_brightness if role == "wrist" and args.wrist_brightness is not None else args.brightness
            controls = merge_camera_controls(
                camera_control_defaults(dev, role=role),
                CameraControls(
                    exposure_absolute=exposure,
                    auto_exposure=auto_exp,
                    power_line_frequency=power_line,
                    gain=gain,
                    brightness=brightness,
                ),
            )

            use_realsense = is_realsense_device(dev) and backend != "v4l2"
            if backend == "realsense" and not is_realsense_device(dev):
                print(f"WARNING: {dev} is not a RealSense device, falling back to V4L2")

            if use_realsense:
                r = RealSenseCamera(device=dev, width=w, height=h, fps=fp, controls=controls)
            else:
                cam_cfg = RealCameraConfig(
                    width=w,
                    height=h,
                    fps=fp,
                )
                r = V4L2Camera(dev, cam_cfg, controls=controls)
            backend_label = "librealsense" if use_realsense else "V4L2"
            print(f"{role} camera: {dev}  {r.width}x{r.height}  fps={r.fps:.1f}  "
                  f"backend={backend_label}  "
                  f"name={video_device_name(dev)!r}")
            # Read first frame to report brightness immediately
            ok, frame = r.read()
            if ok and frame is not None:
                arr = np.asarray(frame, dtype=np.float64)
                print(f"  first frame: shape={arr.shape}  mean={arr.mean():.1f}  "
                      f"min={arr.min():.0f}  max={arr.max():.0f}")
            readers[role] = r
    except Exception as exc:
        print(f"camera open error: {type(exc).__name__}: {exc}")
        _release_all(readers)
        return 1

    # ── snapshot-only path ───────────────────────────────────────────────
    if args.snapshot_only:
        return _do_snapshots(readers, args, single_camera=single_camera)

    # ── GUI preview path ─────────────────────────────────────────────────
    if not os.environ.get("DISPLAY") and not os.environ.get("WAYLAND_DISPLAY"):
        print("No DISPLAY/WAYLAND_DISPLAY — falling back to --snapshot-only ...")
        return _do_snapshots(readers, args, single_camera=single_camera)

    import tkinter as tk
    from PIL import Image, ImageTk

    root = tk.Tk()
    root.title("Piper Camera (single)" if single_camera else "Piper Cameras (global | wrist)")

    if single_camera:
        main_label = tk.Label(root)
        main_label.pack(padx=4, pady=4)
    else:
        left_frame = tk.Frame(root)
        left_frame.pack(side=tk.LEFT, padx=4, pady=4)
        global_title = tk.Label(left_frame, text="global", font=("TkDefaultFont", 11, "bold"))
        global_title.pack()
        global_label = tk.Label(left_frame)
        global_label.pack()

        right_frame = tk.Frame(root)
        right_frame.pack(side=tk.LEFT, padx=4, pady=4)
        wrist_title = tk.Label(right_frame, text="wrist", font=("TkDefaultFont", 11, "bold"))
        wrist_title.pack()
        wrist_label = tk.Label(right_frame)
        wrist_label.pack()

    status_var = tk.StringVar(value="press q to quit, s to save snapshot")
    status_bar = tk.Label(root, textvariable=status_var, anchor=tk.W, relief=tk.SUNKEN)
    status_bar.pack(side=tk.BOTTOM, fill=tk.X)

    running = True
    saved = 0
    frame_count = 0
    fps_t0 = time.monotonic()
    fps_display = 0.0

    def on_key(event: tk.Event) -> None:
        nonlocal running, saved
        if event.char in ("q", "Q"):
            running = False
            root.destroy()
        elif event.char in ("s", "S"):
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            os.makedirs("outputs", exist_ok=True)
            if single_camera:
                ok, frame = readers["global"].read()
                if ok and frame is not None:
                    g_path = f"outputs/snapshot_{ts}.jpg"
                    cv2.imwrite(g_path, _display_frame(readers["global"], frame))
                    saved += 1
                    print(f"saved #{saved}: {g_path}")
            else:
                g_ok, g_frame = readers["global"].read()
                w_ok, w_frame = readers["wrist"].read()
                if g_ok and w_ok:
                    g_path = f"outputs/snapshot_global_{ts}.jpg"
                    w_path = f"outputs/snapshot_wrist_{ts}.jpg"
                    cv2.imwrite(g_path, _display_frame(readers["global"], g_frame))
                    cv2.imwrite(w_path, _display_frame(readers["wrist"], w_frame))
                    saved += 1
                    print(f"saved #{saved}: {g_path}  {w_path}")

    root.bind("<Key>", on_key)

    def update() -> None:
        nonlocal frame_count, fps_t0, fps_display
        if not running:
            return

        if single_camera:
            ok, frame = readers["global"].read()
            if ok and frame is not None:
                display = _display_frame(readers["global"], frame)
                arr = np.asarray(frame, dtype=np.float64)
                status = (f"mean={arr.mean():.0f}  min={arr.min():.0f}  max={arr.max():.0f}  "
                          f"fps={fps_display:.1f}  |  press q to quit, s to save")
                status_var.set(status)
                rgb = cv2.resize(display, (640, 480))
                pil_img = Image.fromarray(rgb[:, :, ::-1])  # BGR->RGB
                tk_img = ImageTk.PhotoImage(pil_img)
                main_label.configure(image=tk_img)
                main_label.image = tk_img
        else:
            g_ok, g_frame = readers["global"].read()
            w_ok, w_frame = readers["wrist"].read()
            if g_ok and w_ok and g_frame is not None and w_frame is not None:
                g_display = _display_frame(readers["global"], g_frame)
                w_display = _display_frame(readers["wrist"], w_frame)
                g_arr = np.asarray(g_frame, dtype=np.float64)
                w_arr = np.asarray(w_frame, dtype=np.float64)
                status = (f"global mean={g_arr.mean():.0f}  wrist mean={w_arr.mean():.0f}  "
                          f"fps={fps_display:.1f}  |  press q to quit, s to save")
                status_var.set(status)

                g_rgb = cv2.resize(g_display, (640, 480))
                w_rgb = cv2.resize(w_display, (640, 480))
                g_pil = Image.fromarray(g_rgb[:, :, ::-1])  # BGR->RGB
                w_pil = Image.fromarray(w_rgb[:, :, ::-1])
                g_tk = ImageTk.PhotoImage(g_pil)
                w_tk = ImageTk.PhotoImage(w_pil)
                global_label.configure(image=g_tk)
                global_label.image = g_tk
                wrist_label.configure(image=w_tk)
                wrist_label.image = w_tk

        frame_count += 1
        elapsed = time.monotonic() - fps_t0
        if elapsed >= 0.5:
            fps_display = frame_count / elapsed
            frame_count = 0
            fps_t0 = time.monotonic()

        root.after(10, update)

    root.after(100, update)
    root.mainloop()

    _release_all(readers)
    print("\ncameras released")

    return 0


# ── commands ──────────────────────────────────────────────────────────────────

def _cmd_list_only() -> int:
    print("=== /dev/video* devices ===\n")
    devices_info = probe_readable_v4l2_devices_detailed(verbose=True)
    print()
    ok = [d for d in devices_info if d.status == "ok"]
    bad = [d for d in devices_info if d.status != "ok"]
    print(f"Summary: {len(ok)} readable, {len(bad)} skipped/timeout/no-frame")
    if ok:
        print("Readable devices:")
        for d in ok:
            print(f"  {d.device}  name={d.name!r}  group={d.group}  shape={d.shape}  "
                  f"mean={d.mean:.1f}  {'REALSENSE' if d.realsense else 'UVC'}")
    if bad:
        print("Skipped/bad devices:")
        for d in bad:
            print(f"  {d.device}  status={d.status}  name={d.name!r}  group={d.group}")
    return 0


# ── snapshot ──────────────────────────────────────────────────────────────────

def _do_snapshots(
    readers: dict[str, V4L2Camera | RealSenseCamera],
    args: argparse.Namespace,
    *,
    single_camera: bool = False,
) -> int:
    save_dir = Path(args.snapshot_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    try:
        if single_camera:
            ok, frame = readers["global"].read()
            if not ok or frame is None:
                print("ERROR: failed to read snapshot")
                return 1
            arr = np.asarray(frame)
            print(f"  snapshot: shape={arr.shape} mean={arr.mean():.1f} "
                  f"min={arr.min()} max={arr.max()}")
            path = save_dir / f"snapshot_{ts}.jpg"
            cv2.imwrite(str(path), _display_frame(readers["global"], frame))
            print(f"  saved: {path}")
        else:
            frames: dict[str, np.ndarray] = {}
            for role in ("global", "wrist"):
                ok, frame = readers[role].read()
                if not ok or frame is None:
                    print(f"ERROR: failed to read {role} snapshot")
                    return 1
                frames[role] = frame
                arr = np.asarray(frame)
                print(f"  {role} snapshot: shape={arr.shape} mean={arr.mean():.1f} "
                      f"min={arr.min()} max={arr.max()}")
                path = save_dir / f"snapshot_{role}_{ts}.jpg"
                cv2.imwrite(str(path), _display_frame(readers[role], frame))
                print(f"  saved: {path}")
            pair_path = save_dir / f"snapshot_pair_global_wrist_{ts}.jpg"
            cv2.imwrite(
                str(pair_path),
                make_labeled_pair(
                    _display_frame(readers["global"], frames["global"]),
                    _display_frame(readers["wrist"], frames["wrist"]),
                ),
            )
            print(f"  saved labeled pair: {pair_path}")
    finally:
        _release_all(readers)
        print("cameras released after snapshot")
    return 0


# ── helpers ───────────────────────────────────────────────────────────────────

def _release_all(readers: dict) -> None:
    for r in readers.values():
        try:
            r.release()
        except Exception:
            pass


def _display_frame(reader: V4L2Camera | RealSenseCamera, frame: np.ndarray) -> np.ndarray:
    color_order = getattr(reader, "color_order", "bgr").lower()
    if frame.ndim == 3 and frame.shape[-1] == 3 and color_order == "rgb":
        return cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    if frame.ndim == 3 and frame.shape[-1] == 4 and color_order == "rgba":
        return cv2.cvtColor(frame, cv2.COLOR_RGBA2BGR)
    return frame


def _has_gui() -> bool:
    if bool(os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY")):
        return True
    for line in cv2.getBuildInformation().splitlines():
        if line.strip().startswith("GUI:"):
            return "NONE" not in line.upper()
    return False


def make_labeled_pair(global_frame: np.ndarray, wrist_frame: np.ndarray) -> np.ndarray:
    g_show = cv2.resize(global_frame, (640, 480))
    w_show = cv2.resize(wrist_frame, (640, 480))
    canvas = np.hstack((g_show, w_show))
    cv2.putText(canvas, "global", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.putText(canvas, "wrist", (650, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    return canvas


def _print_explicit_hint(devices: list[str]) -> None:
    print("\n请用显式设备号运行:")
    if len(devices) >= 2:
        global_cand = [d for d in devices if "realsense" not in video_device_name(d).lower()]
        wrist_cand = [d for d in devices if "realsense" in video_device_name(d).lower()]
        g = global_cand[0] if global_cand else devices[0]
        w = wrist_cand[0] if wrist_cand else devices[-1]
        if g == w and len(devices) >= 2:
            w = devices[1] if devices[1] != g else devices[-1]
        print(f"  --global-camera {g} --wrist-camera {w}")
    elif len(devices) == 1:
        print(f"  --global-camera {devices[0]}  (single camera — use --single-camera)")
    else:
        print("  (no /dev/video* devices detected)")

    print("\n所有 /dev/video* 设备:")
    for d in list_video_devices():
        print(f"  {d}  name={video_device_name(d)!r}  group={video_device_group(d)}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Open cameras for debug preview.")
    p.add_argument("--global-camera", default=DEFAULT_GLOBAL_CAMERA)
    p.add_argument("--wrist-camera", default=DEFAULT_WRIST_CAMERA)
    p.add_argument("--single-camera", action="store_true",
                   help="Use single camera for both global & wrist")
    p.add_argument("--width", type=int, default=DEFAULT_CAMERA_WIDTH)
    p.add_argument("--height", type=int, default=DEFAULT_CAMERA_HEIGHT)
    p.add_argument("--fps", type=int, default=DEFAULT_CAMERA_FPS)
    p.add_argument("--backend", choices=("auto", "v4l2", "realsense"), default="auto",
                   help="Force camera backend: auto (default), v4l2, or realsense")
    p.add_argument("--exposure", type=int, default=None,
                   help="Manual exposure absolute value (V4L2 CAP_PROP_EXPOSURE)")
    p.add_argument("--auto-exposure", type=int, default=None,
                   help="Auto exposure mode: 1=on, 0=off (V4L2 CAP_PROP_AUTO_EXPOSURE); "
                   "for RealSense: 1=on, 0=off")
    p.add_argument("--power-line", type=int, default=None,
                   help="Power line frequency: 1=50Hz, 2=60Hz")
    p.add_argument("--gain", type=float, default=None,
                   help="Sensor gain / ISO (higher = brighter, noisier)")
    p.add_argument("--brightness", type=float, default=None,
                   help="Image brightness offset")
    p.add_argument("--wrist-exposure", type=int, default=None,
                   help="Wrist camera manual exposure (overrides --exposure for wrist)")
    p.add_argument("--wrist-auto-exposure", type=int, default=None,
                   help="Wrist camera auto exposure: 1=on, 0=off")
    p.add_argument("--wrist-gain", type=float, default=None,
                   help="Wrist camera gain (overrides --gain for wrist)")
    p.add_argument("--wrist-brightness", type=float, default=None,
                   help="Wrist camera brightness (overrides --brightness for wrist)")
    p.add_argument("--wrist-power-line", type=int, default=None,
                   help="Wrist camera power line frequency: 1=50Hz, 2=60Hz")
    p.add_argument("--wrist-backend", choices=("auto", "v4l2", "realsense"), default="auto",
                   help="Wrist camera backend override")
    p.add_argument("--list-only", action="store_true")
    p.add_argument("--snapshot-only", action="store_true")
    p.add_argument("--snapshot-dir", default="outputs/camera_debug")
    return p.parse_args()


if __name__ == "__main__":
    raise SystemExit(main())
