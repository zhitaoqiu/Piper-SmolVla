"""真实部署时的双摄预览和按键控制。

本模块只负责显示画面、读取键盘和等待操作员确认；它不连接 CAN，
也不会发送任何机器人动作。
"""

from __future__ import annotations

import os
import select
import sys
import termios
import time
import tty
from dataclasses import dataclass
from enum import Enum
from typing import Any

import numpy as np

from piper_smolvla.schema import GLOBAL_IMAGE_KEY, WRIST_IMAGE_KEY


class ControlCommand(Enum):
    NONE = "none"
    PAUSE = "pause"
    RESUME = "resume"
    QUIT = "quit"


class TerminalRolloutControl:
    """Non-blocking terminal keyboard control for --no-preview mode.

    Uses cbreak + select to read keystrokes without blocking the rollout
    loop.  If stdin is not a TTY (e.g. piped input), the control is
    automatically disabled and ``poll()`` always returns ``NONE``.
    """

    def __init__(self, *, enabled: bool = True):
        self._enabled = enabled and sys.stdin.isatty()
        self._old_settings: list[Any] | None = None
        if self._enabled:
            self._old_settings = termios.tcgetattr(sys.stdin.fileno())
            tty.setcbreak(sys.stdin.fileno())

    @property
    def enabled(self) -> bool:
        return self._enabled

    def poll(self) -> ControlCommand:
        """Return the next pending command, or NONE if no key was pressed."""
        if not self._enabled:
            return ControlCommand.NONE
        try:
            if select.select([sys.stdin], [], [], 0)[0]:
                ch = os.read(sys.stdin.fileno(), 1)
                if ch in (b" ", b"p", b"P"):
                    return ControlCommand.PAUSE
                if ch in (b"\n", b"\r", b"r", b"R"):
                    return ControlCommand.RESUME
                if ch in (b"q", b"Q"):
                    return ControlCommand.QUIT
        except (OSError, ValueError, TypeError):
            pass
        return ControlCommand.NONE

    def restore(self) -> None:
        if self._old_settings is not None:
            try:
                termios.tcsetattr(sys.stdin.fileno(), termios.TCSADRAIN, self._old_settings)
            except Exception:
                pass
            self._old_settings = None


@dataclass
class _DummyPreview:
    """Stand-in used when the real RolloutPreview is not available."""

    enabled: bool = False

    def show(self, **kwargs: Any) -> int:
        return -1

    def close(self) -> None:
        pass


class RolloutPreview:
    """ACT-style live dual-camera deployment preview."""

    def __init__(self, *, enabled: bool, window_name: str):
        self.enabled = enabled
        self.window_name = window_name
        self._cv2 = None
        if not enabled:
            return

        # Skip preview when no X11 display is available (e.g. SSH session).
        if not os.environ.get("DISPLAY"):
            print("preview_disabled: no DISPLAY (headless session); use --no-preview to silence this")
            self.enabled = False
            return

        import threading

        ok: dict[str, bool] = {"done": False}

        def _create_window() -> None:
            try:
                import cv2

                self._cv2 = cv2
                cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
                cv2.resizeWindow(self.window_name, 1280, 520)
                ok["done"] = True
            except Exception:
                pass

        t = threading.Thread(target=_create_window, daemon=True)
        t.start()
        t.join(timeout=3.0)
        if t.is_alive() or not ok["done"]:
            self.enabled = False
            self._cv2 = None
            print("preview_disabled: window creation timed out (no X11?); use --no-preview to silence this")

    def show(
        self,
        *,
        global_rgb: np.ndarray,
        wrist_rgb: np.ndarray,
        status: str,
        frame_idx: int | None = None,
        state: tuple[float, ...] | None = None,
        color: tuple[int, int, int] = (0, 255, 255),
    ) -> int:
        if not self.enabled or self._cv2 is None:
            return -1
        canvas = make_preview_canvas(global_rgb, wrist_rgb)
        lines = [status, "SPACE start/pause/resume | Q/ESC quit"]
        if frame_idx is not None:
            lines.insert(1, f"frame: {frame_idx}")
        if state is not None:
            lines.append(f"qpos: {' '.join(f'{value:.3f}' for value in state[:6])}  grip={state[6]:.4f}")
        for index, line in enumerate(lines):
            y = 470 - index * 26
            self._cv2.putText(canvas, line, (10, y), self._cv2.FONT_HERSHEY_SIMPLEX, 0.62, color, 2)
        self._cv2.imshow(self.window_name, canvas)
        return int(self._cv2.waitKey(1) & 0xFF)

    def close(self) -> None:
        if self.enabled and self._cv2 is not None:
            try:
                self._cv2.destroyWindow(self.window_name)
            except Exception:
                pass


def make_preview_canvas(global_rgb: np.ndarray, wrist_rgb: np.ndarray) -> np.ndarray:
    import cv2

    global_bgr = cv2.cvtColor(np.asarray(global_rgb), cv2.COLOR_RGB2BGR)
    wrist_bgr = cv2.cvtColor(np.asarray(wrist_rgb), cv2.COLOR_RGB2BGR)
    global_show = cv2.resize(global_bgr, (640, 480))
    wrist_show = cv2.resize(wrist_bgr, (640, 480))
    canvas = np.hstack((global_show, wrist_show))
    cv2.putText(canvas, "global", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.putText(canvas, "wrist", (650, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    return canvas


def is_start_or_pause_key(key: int) -> bool:
    return key in (ord(" "), 10, 13)


def is_quit_key(key: int) -> bool:
    return key in (ord("q"), ord("Q"), 27)


def wait_for_space_start(
    *,
    cameras: Any,
    backend: Any,
    preview: RolloutPreview,
    max_frames: int,
    terminal_control: TerminalRolloutControl | None = None,
) -> tuple[tuple[float, ...], tuple[np.ndarray, np.ndarray]] | None:
    _print_startup_controls(preview, terminal_control)

    use_terminal = terminal_control is not None and terminal_control.enabled and not preview.enabled

    if use_terminal:
        return _terminal_wait_for_start(cameras, backend, terminal_control, max_frames)

    if not preview.enabled:
        _print_terminal_unavailable()
        input("Press ENTER to start rollout, or Ctrl+C to abort ...")
        images = cameras.read_images()
        return tuple(backend.read_state()), (images[GLOBAL_IMAGE_KEY], images[WRIST_IMAGE_KEY])

    return _preview_wait_for_start(cameras, backend, preview, max_frames)


def _print_startup_controls(preview: RolloutPreview, terminal_control: TerminalRolloutControl | None) -> None:
    print()
    print("Controls:")
    if terminal_control is not None and terminal_control.enabled and not preview.enabled:
        print("  SPACE / P      start rollout")
        print("  Q              quit before rollout")
    elif preview.enabled:
        print("  SPACE / ENTER  start rollout")
        print("  Q / ESC        quit before rollout")
    else:
        print("  ENTER          start rollout")
    print("  Ctrl+C         emergency stop this script")
    print()


def _print_terminal_unavailable() -> None:
    print("[terminal-control] stdin is not a TTY — hotkeys disabled.")
    print("                    Use Ctrl+C to abort.")


def _terminal_wait_for_start(
    cameras: Any,
    backend: Any,
    terminal_control: TerminalRolloutControl,
    max_frames: int,
) -> tuple[tuple[float, ...], tuple[np.ndarray, np.ndarray]] | None:
    print(f"Waiting for SPACE to start rollout (max {max_frames} frames)...")
    while True:
        cmd = terminal_control.poll()
        if cmd == ControlCommand.PAUSE:
            images = cameras.read_images()
            return tuple(backend.read_state()), (images[GLOBAL_IMAGE_KEY], images[WRIST_IMAGE_KEY])
        if cmd == ControlCommand.QUIT:
            return None
        time.sleep(0.03)


def _preview_wait_for_start(
    cameras: Any,
    backend: Any,
    preview: RolloutPreview,
    max_frames: int,
) -> tuple[tuple[float, ...], tuple[np.ndarray, np.ndarray]] | None:
    while True:
        images = cameras.read_images()
        global_img = images[GLOBAL_IMAGE_KEY]
        wrist_img = images[WRIST_IMAGE_KEY]
        state = tuple(backend.read_state())
        key = preview.show(
            global_rgb=global_img,
            wrist_rgb=wrist_img,
            status=f"READY - SPACE start rollout (max {max_frames} frames)",
            state=state,
            color=(0, 255, 0),
        )
        if is_start_or_pause_key(key):
            return state, (global_img, wrist_img)
        if is_quit_key(key):
            return None
        time.sleep(0.03)


def wait_while_paused(
    *,
    cameras: Any,
    backend: Any,
    preview: RolloutPreview,
    frame_idx: int,
    max_frames: int,
    terminal_control: TerminalRolloutControl | None = None,
) -> tuple[str, tuple[np.ndarray, np.ndarray] | None]:
    use_terminal = terminal_control is not None and terminal_control.enabled and not preview.enabled

    if use_terminal:
        return _terminal_wait_while_paused(cameras, backend, terminal_control, frame_idx, max_frames)

    if not preview.enabled:
        print(f"  PAUSED {frame_idx}/{max_frames} — ENTER to resume, Ctrl+C to abort")
        input("Paused. Press ENTER to resume, or Ctrl+C to abort ...")
        images = cameras.read_images()
        return "resume", (images[GLOBAL_IMAGE_KEY], images[WRIST_IMAGE_KEY])

    return _preview_wait_while_paused(cameras, backend, preview, frame_idx, max_frames)


def _terminal_wait_while_paused(
    cameras: Any,
    backend: Any,
    terminal_control: TerminalRolloutControl,
    frame_idx: int,
    max_frames: int,
) -> tuple[str, tuple[np.ndarray, np.ndarray] | None]:
    print(f"  PAUSED {frame_idx}/{max_frames} — R/ENTER resume, Q quit")
    while True:
        cmd = terminal_control.poll()
        if cmd == ControlCommand.RESUME:
            images = cameras.read_images()
            return "resume", (images[GLOBAL_IMAGE_KEY], images[WRIST_IMAGE_KEY])
        if cmd == ControlCommand.QUIT:
            images = cameras.read_images()
            return "quit", (images[GLOBAL_IMAGE_KEY], images[WRIST_IMAGE_KEY])
        time.sleep(0.03)


def _preview_wait_while_paused(
    cameras: Any,
    backend: Any,
    preview: RolloutPreview,
    frame_idx: int,
    max_frames: int,
) -> tuple[str, tuple[np.ndarray, np.ndarray] | None]:
    print("  PAUSED - SPACE to resume, Q/ESC to quit")
    last_images = None
    while True:
        images = cameras.read_images()
        global_img = images[GLOBAL_IMAGE_KEY]
        wrist_img = images[WRIST_IMAGE_KEY]
        last_images = (global_img, wrist_img)
        try:
            state = tuple(backend.read_state())
        except Exception:
            state = None
        key = preview.show(
            global_rgb=global_img,
            wrist_rgb=wrist_img,
            status=f"PAUSED {frame_idx}/{max_frames}",
            frame_idx=frame_idx,
            state=state,
            color=(0, 165, 255),
        )
        if is_start_or_pause_key(key):
            return "resume", last_images
        if is_quit_key(key):
            return "quit", last_images
        time.sleep(0.03)
