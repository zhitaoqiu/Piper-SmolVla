#!/usr/bin/env python3
"""双摄调试预览：打开 global + wrist 摄像头，显示实时画面，按 q 退出。

用法:
    python scripts/debug_cameras.py                          # 自动检测
    python scripts/debug_cameras.py --list-only              # 列出设备（轻量探测）
    python scripts/debug_cameras.py --snapshot-only          # 只截图不显示窗口
    python scripts/debug_cameras.py --global-camera /dev/video6 --wrist-camera /dev/video2
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

from piper_smolvla.camera_utils import (
    _invalidated_bad_device_cache,
    is_explicit,
    is_realsense_device,
    list_video_devices,
    normalize_video_device,
    print_resolved_pair,
    probe_readable_v4l2_devices_detailed,
    resolve_camera_pair,
    video_device_group,
    video_device_name,
)
from piper_smolvla.real_sources import (
    RealCameraConfig,
    _RealSenseCamera,
    _V4L2Camera,
)


# ── main ──────────────────────────────────────────────────────────────────────

def main() -> int:
    args = parse_args()

    # ── list-only: light probe, print, exit ──────────────────────────────
    if args.list_only:
        return _cmd_list_only()

    # ── resolve cameras ──────────────────────────────────────────────────
    if is_explicit(args.global_camera) and is_explicit(args.wrist_camera):
        global_dev = normalize_video_device(args.global_camera)
        wrist_dev = normalize_video_device(args.wrist_camera)
        print(f"camera assignment mode = explicit")
        print(f"  global: {global_dev}  name={video_device_name(global_dev)!r}  group={video_device_group(global_dev)}")
        print(f"  wrist:  {wrist_dev}  name={video_device_name(wrist_dev)!r}  group={video_device_group(wrist_dev)}")
    else:
        print("probing /dev/video* devices (max 2.5 s per device)...")
        print()
        devices_info = probe_readable_v4l2_devices_detailed(verbose=True)
        readable = [d.device for d in devices_info if d.shape is not None and not d.black and d.status == "ok"]
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

        try:
            global_dev, wrist_dev = resolve_camera_pair(args.global_camera, args.wrist_camera, devices=readable)
        except RuntimeError as exc:
            print(f"auto-resolve failed: {exc}")
            _print_explicit_hint(readable)
            return 1

    # ── same-device / same-group check ───────────────────────────────────
    if global_dev == wrist_dev:
        print(f"ERROR: global and wrist resolved to same device: {global_dev}")
        return 1
    global_group = video_device_group(global_dev)
    wrist_group = video_device_group(wrist_dev)
    if global_group == wrist_group:
        print(f"ERROR: global ({global_dev}) and wrist ({wrist_dev}) share USB group '{global_group}'.")
        print("Refusing to continue — use --global-camera /dev/videoX --wrist-camera /dev/videoY from different USB buses.")
        return 1

    print(f"\nassigning: global={global_dev} (group={global_group})  wrist={wrist_dev} (group={wrist_group})\n")

    # ── open cameras ─────────────────────────────────────────────────────
    readers: dict[str, _V4L2Camera | _RealSenseCamera] = {}
    try:
        for role, dev in [("global", global_dev), ("wrist", wrist_dev)]:
            w = args.width
            h = args.height
            fp = args.global_fps if role == "global" else (args.wrist_fps or args.fps)

            if is_realsense_device(dev):
                r = _RealSenseCamera(device=dev, width=w, height=h, fps=fp)
            else:
                r = _V4L2Camera(dev, RealCameraConfig(width=w, height=h, fps=fp))
            print(f"{role} camera: {dev}  {r.width}x{r.height}  fps={r.fps:.1f}  "
                  f"backend={'librealsense' if is_realsense_device(dev) else 'V4L2'}  "
                  f"name={video_device_name(dev)!r}")
            readers[role] = r
    except Exception as exc:
        print(f"camera open error: {type(exc).__name__}: {exc}")
        _release_all(readers)
        return 1

    # ── snapshot-only path ───────────────────────────────────────────────
    if args.snapshot_only:
        return _do_snapshots(readers, args)

    # ── GUI preview path ─────────────────────────────────────────────────
    if not _has_gui():
        print("OpenCV GUI 不可用 (无 DISPLAY/WAYLAND_DISPLAY 或编译时无 GUI 支持)。")
        print("自动切换到 --snapshot-only ...")
        return _do_snapshots(readers, args)

    print("press 'q' to quit, 's' to save snapshot pair")
    print("NOTE: cameras return BGR frames; OpenCV displays BGR directly — colors are correct.\n")
    saved = 0

    try:
        while True:
            g_ok, g_frame = readers["global"].read()
            w_ok, w_frame = readers["wrist"].read()
            if not g_ok or not w_ok:
                missing = [r for r, ok in [("global", g_ok), ("wrist", w_ok)] if not ok]
                print(f"dropped frame: {missing}")
                time.sleep(0.02)
                continue

            g_show = cv2.resize(g_frame, (640, 480))
            w_show = cv2.resize(w_frame, (640, 480))
            canvas = np.hstack((g_show, w_show))
            cv2.putText(canvas, "global", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(canvas, "wrist", (650, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.imshow("Piper Cameras (global | wrist)", canvas)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            elif key == ord("s"):
                ts = datetime.now().strftime("%Y%m%d_%H%M%S")
                g_path = f"outputs/snapshot_global_{ts}.jpg"
                w_path = f"outputs/snapshot_wrist_{ts}.jpg"
                os.makedirs("outputs", exist_ok=True)
                cv2.imwrite(g_path, g_frame)
                cv2.imwrite(w_path, w_frame)
                saved += 1
                print(f"saved #{saved}: {g_path}  {w_path}")
    except KeyboardInterrupt:
        pass
    finally:
        _release_all(readers)
        cv2.destroyAllWindows()
        print("\ncameras released")

    return 0


# ── commands ──────────────────────────────────────────────────────────────────

def _cmd_list_only() -> int:
    """Light device listing: sysfs info + quick probe."""
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


# ── helpers ───────────────────────────────────────────────────────────────────

def _do_snapshots(
    readers: dict[str, _V4L2Camera | _RealSenseCamera], args: argparse.Namespace
) -> int:
    save_dir = Path(args.snapshot_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    frames: dict[str, np.ndarray] = {}
    try:
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
            cv2.imwrite(str(path), frame)
            print(f"  saved: {path}")
        pair_path = save_dir / f"snapshot_pair_global_wrist_{ts}.jpg"
        cv2.imwrite(str(pair_path), make_labeled_pair(frames["global"], frames["wrist"]))
        print(f"  saved labeled pair: {pair_path}")
    finally:
        _release_all(readers)
        print("cameras released after snapshot")
    return 0


def _release_all(readers: dict[str, Any]) -> None:
    for r in readers.values():
        try:
            r.release()
        except Exception:
            pass


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
        print(f"  --global-camera {devices[0]} --wrist-camera <手动指定>")
    else:
        print("  (no /dev/video* devices detected)")

    print("\n所有 /dev/video* 设备:")
    for d in list_video_devices():
        print(f"  {d}  name={video_device_name(d)!r}  group={video_device_group(d)}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Open two cameras for debug preview.")
    p.add_argument("--global-camera", default="auto")
    p.add_argument("--wrist-camera", default="auto")
    p.add_argument("--width", type=int, default=640)
    p.add_argument("--height", type=int, default=480)
    p.add_argument("--fps", type=int, default=30)
    p.add_argument("--global-fps", type=int, default=30)
    p.add_argument("--wrist-fps", type=int, default=30)
    p.add_argument("--list-only", action="store_true")
    p.add_argument("--snapshot-only", action="store_true")
    p.add_argument("--snapshot-dir", default="outputs/camera_debug")
    return p.parse_args()


if __name__ == "__main__":
    raise SystemExit(main())
