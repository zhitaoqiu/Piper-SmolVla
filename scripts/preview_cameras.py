#!/usr/bin/env python3
"""Preview the exact dual-camera stream used by collection and deployment."""

from __future__ import annotations

import _bootstrap  # noqa: F401

import argparse
import os
import time
from datetime import datetime
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageTk

from piper_smolvla.cameras import (
    DEFAULT_BLACK_FRAME_THRESHOLD,
    DEFAULT_CAMERA_FPS,
    DEFAULT_CAMERA_HEIGHT,
    DEFAULT_CAMERA_WIDTH,
    DEFAULT_GLOBAL_CAMERA,
    DEFAULT_WRIST_CAMERA,
    RealCameraConfig,
    RealCameraSource,
    camera_defaults_summary,
)
from piper_smolvla.schema import GLOBAL_IMAGE_KEY, WRIST_IMAGE_KEY


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Preview Piper global/wrist cameras with shared project defaults.")
    parser.add_argument("--global-camera", default=DEFAULT_GLOBAL_CAMERA)
    parser.add_argument("--wrist-camera", default=DEFAULT_WRIST_CAMERA)
    parser.add_argument("--width", type=int, default=DEFAULT_CAMERA_WIDTH)
    parser.add_argument("--height", type=int, default=DEFAULT_CAMERA_HEIGHT)
    parser.add_argument("--camera-fps", type=int, default=DEFAULT_CAMERA_FPS)
    parser.add_argument("--black-frame-threshold", type=float, default=DEFAULT_BLACK_FRAME_THRESHOLD)
    parser.add_argument("--snapshot-only", action="store_true")
    parser.add_argument("--snapshot-dir", default="outputs/camera_preview")
    parser.add_argument("--wrist-auto-exposure", type=int, default=None)
    parser.add_argument("--wrist-exposure", type=int, default=None)
    parser.add_argument("--wrist-gain", type=float, default=None)
    parser.add_argument("--wrist-brightness", type=float, default=None)
    parser.add_argument("--wrist-power-line", type=int, default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    print("camera_defaults:")
    for key, value in camera_defaults_summary().items():
        print(f"  {key}: {value}")

    source = RealCameraSource(
        RealCameraConfig(
            allow_hardware_readonly=True,
            global_camera=args.global_camera,
            wrist_camera=args.wrist_camera,
            width=args.width,
            height=args.height,
            fps=args.camera_fps,
            black_threshold=args.black_frame_threshold,
            wrist_auto_exposure=args.wrist_auto_exposure,
            wrist_exposure_absolute=args.wrist_exposure,
            wrist_gain=args.wrist_gain,
            wrist_brightness=args.wrist_brightness,
            wrist_power_line_frequency=args.wrist_power_line,
        )
    )

    try:
        source.connect()
        print(f"resolved_global={source.resolved_global}")
        print(f"resolved_wrist={source.resolved_wrist}")
        images = source.read_images()
        _print_image_stats(images)

        if args.snapshot_only or not (os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY")):
            _save_snapshot_pair(images, Path(args.snapshot_dir))
            return 0

        return _run_tk_preview(source, args)
    finally:
        source.close()
        print("cameras released")


def _run_tk_preview(source: RealCameraSource, args: argparse.Namespace) -> int:
    import tkinter as tk

    root = tk.Tk()
    root.title("Piper Cameras | global | wrist")
    label = tk.Label(root)
    label.pack()
    status_var = tk.StringVar(value="press q to quit, s to save snapshot")
    tk.Label(root, textvariable=status_var, anchor=tk.W, relief=tk.SUNKEN).pack(side=tk.BOTTOM, fill=tk.X)

    running = True
    frames = 0
    t0 = time.monotonic()
    display_fps = 0.0

    def on_key(event: tk.Event) -> None:
        nonlocal running
        if event.char in ("q", "Q") or event.keysym == "Escape":
            running = False
            root.destroy()
        elif event.char in ("s", "S"):
            _save_snapshot_pair(source.read_images(), Path(args.snapshot_dir))

    root.bind("<Key>", on_key)

    def update() -> None:
        nonlocal frames, t0, display_fps
        if not running:
            return
        try:
            images = source.read_images()
        except Exception as exc:  # noqa: BLE001
            status_var.set(f"camera read error: {type(exc).__name__}: {exc}")
            root.after(250, update)
            return

        canvas = _make_pair_image(images, width=640, height=480)
        photo = ImageTk.PhotoImage(canvas)
        label.configure(image=photo)
        label.image = photo

        frames += 1
        elapsed = time.monotonic() - t0
        if elapsed >= 0.5:
            display_fps = frames / elapsed
            frames = 0
            t0 = time.monotonic()
        g = np.asarray(images[GLOBAL_IMAGE_KEY], dtype=np.float32)
        w = np.asarray(images[WRIST_IMAGE_KEY], dtype=np.float32)
        status_var.set(f"global mean={g.mean():.0f}  wrist mean={w.mean():.0f}  fps={display_fps:.1f}")
        root.after(1, update)

    root.after(50, update)
    root.mainloop()
    return 0


def _make_pair_image(images: dict[str, np.ndarray], *, width: int, height: int) -> Image.Image:
    global_img = Image.fromarray(_as_rgb_uint8(images[GLOBAL_IMAGE_KEY])).resize((width, height))
    wrist_img = Image.fromarray(_as_rgb_uint8(images[WRIST_IMAGE_KEY])).resize((width, height))
    canvas = Image.new("RGB", (width * 2, height), (0, 0, 0))
    canvas.paste(global_img, (0, 0))
    canvas.paste(wrist_img, (width, 0))
    draw = ImageDraw.Draw(canvas)
    draw.text((10, 10), "global", fill=(0, 255, 0))
    draw.text((width + 10, 10), "wrist", fill=(0, 255, 0))
    return canvas


def _save_snapshot_pair(images: dict[str, np.ndarray], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    global_path = output_dir / f"global_{ts}.jpg"
    wrist_path = output_dir / f"wrist_{ts}.jpg"
    pair_path = output_dir / f"pair_global_wrist_{ts}.jpg"
    Image.fromarray(_as_rgb_uint8(images[GLOBAL_IMAGE_KEY])).save(global_path)
    Image.fromarray(_as_rgb_uint8(images[WRIST_IMAGE_KEY])).save(wrist_path)
    _make_pair_image(images, width=640, height=480).save(pair_path)
    print(f"saved_global={global_path}")
    print(f"saved_wrist={wrist_path}")
    print(f"saved_pair={pair_path}")


def _print_image_stats(images: dict[str, np.ndarray]) -> None:
    for label, key in (("global", GLOBAL_IMAGE_KEY), ("wrist", WRIST_IMAGE_KEY)):
        arr = np.asarray(images[key])
        print(f"{label}: shape={arr.shape} mean={arr.mean():.1f} min={arr.min()} max={arr.max()}")


def _as_rgb_uint8(image: np.ndarray) -> np.ndarray:
    arr = np.asarray(image)
    if arr.dtype != np.uint8:
        arr = np.clip(arr, 0, 255).astype(np.uint8)
    if arr.ndim != 3 or arr.shape[-1] != 3:
        raise ValueError(f"expected HWC RGB image, got shape={arr.shape}")
    return arr


if __name__ == "__main__":
    raise SystemExit(main())
