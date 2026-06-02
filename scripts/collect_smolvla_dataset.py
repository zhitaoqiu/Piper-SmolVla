#!/usr/bin/env python3
"""SmolVLA 数据采集入口。

真实采集是 operator demonstration/read-only recording：只读 Piper qpos 和双摄，
按本项目镜像示教语义写 observation.state=prev_qpos、action=current_qpos。
"""

from __future__ import annotations

import _bootstrap  # noqa: F401

import argparse
import json
import select
import shutil
import sys
import time
from collections.abc import Callable
from datetime import datetime
from pathlib import Path

import numpy as np

from piper_smolvla.adapter import DryRunPiperIO, PiperSmolVLAAdapter, StaticImageSource
from piper_smolvla.collection import (
    CollectionConfig,
    create_lerobot_dataset,
    image_to_chw_uint8,
    make_readonly_transition_frame,
    write_episode,
)
from piper_smolvla.config import PiperSmolVLAAdapterConfig
from piper_smolvla.real_sources import RealCameraConfig, RealCameraSource, RealPiperStateConfig, RealPiperStateSource
from piper_smolvla.schema import (
    DEFAULT_TASK_INSTRUCTION,
    GLOBAL_IMAGE_KEY,
    START_GUARD_GRIPPER_OPEN_MIN_M,
    START_GUARD_ZONE_ARM_TOLERANCE_RAD,
    STATE_KEY,
    VERIFIED_START_QPOS,
    WRIST_IMAGE_KEY,
)
from piper_smolvla.validation import validate_state


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Collect Piper SmolVLA read-only operator demonstrations.")
    parser.add_argument("--allow-hardware-readonly", action="store_true")
    parser.add_argument("--can-port", default="can0")
    parser.add_argument("--global-camera", default="auto")
    parser.add_argument("--wrist-camera", default="auto")
    parser.add_argument("--output", default="")
    parser.add_argument(
        "--overwrite-output",
        action="store_true",
        help="Delete an existing --output directory before creating a new LeRobot dataset.",
    )
    parser.add_argument("--repo-id", default="piper/smolvla_cube_dual")
    parser.add_argument("--task", default=DEFAULT_TASK_INSTRUCTION)
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument("--fps", type=int, default=10, help="Dataset recording FPS.")
    parser.add_argument("--camera-fps", type=int, default=30, help="Camera capture FPS.")
    parser.add_argument("--operator-demo", action="store_true")
    parser.add_argument("--require-keyboard-start-stop", action="store_true")
    parser.add_argument("--max-duration-sec", type=float, default=60.0)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--state", default="0,1,-1,0,0,0,0.01")
    parser.add_argument(
        "--start-state",
        default="verified",
        help=(
            "Fixed start pose: 'verified' uses this project's verified Piper start, "
            "'current' freezes the precheck qpos, or pass comma-separated 7D qpos."
        ),
    )
    parser.add_argument(
        "--start-guard-mode",
        choices=("zone", "strict"),
        default="zone",
        help="zone uses per-joint tolerances and requires gripper open; strict uses scalar arm/gripper diffs.",
    )
    parser.add_argument("--start-state-tol-rad", type=float, default=0.08)
    parser.add_argument("--start-gripper-tol-m", type=float, default=0.015)
    parser.add_argument("--precheck-image-dir", default="outputs/collection_precheck")
    parser.add_argument("--no-save-precheck-images", action="store_true")
    parser.add_argument("--no-preview", action="store_true", help="Disable the live global/wrist preview window.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.episodes <= 0:
        raise SystemExit("--episodes must be positive")
    if not args.operator_demo:
        raise SystemExit("--operator-demo is required; scripted robot motion is not supported here")
    if not args.require_keyboard_start_stop:
        raise SystemExit("--require-keyboard-start-stop is required for operator-demo collection")
    preflight_output_dir(args)

    adapter, cleanup = build_adapter(args)
    preview = CollectionPreview(enabled=not args.no_preview)
    dataset = None
    dataset_finalized = False
    written_episodes = 0
    try:
        if hasattr(adapter, "image_source") and hasattr(adapter.image_source, "resolved_global"):
            print(f"camera_assignment_mode={adapter.image_source.assignment_mode}")
            print(f"global_camera_device={adapter.image_source.resolved_global}")
            print(f"wrist_camera_device={adapter.image_source.resolved_wrist}")
        print("read-only observation check ...")
        obs = adapter.read_observation(task=args.task)
        print(f"state={list(obs[STATE_KEY])}")
        print(f"global_ok shape={tuple(np.asarray(obs[GLOBAL_IMAGE_KEY]).shape)}")
        print(f"wrist_ok shape={tuple(np.asarray(obs[WRIST_IMAGE_KEY]).shape)}")
        assert_dual_view_observation(obs)
        preview.show(obs, status="PRECHECK: global | wrist")
        if not args.no_save_precheck_images:
            save_precheck_images(obs, args=args)
        start_state, start_source = resolve_start_state(args.start_state, obs)
        print(f"fixed_start_state={list(start_state)}")
        print(f"fixed_start_source={start_source}")
        print(f"start_guard_mode={args.start_guard_mode}")
        if args.start_guard_mode == "zone":
            print(f"zone_arm_tolerance_rad={list(START_GUARD_ZONE_ARM_TOLERANCE_RAD)}")
            print(f"zone_gripper_open_min_m={START_GUARD_GRIPPER_OPEN_MIN_M}")
        print("NO MOTION COMMAND SENT")

        if args.output and not args.dry_run:
            dataset = create_lerobot_dataset(
                root=Path(args.output),
                repo_id=args.repo_id,
                config=CollectionConfig(
                    fps=args.fps,
                    task=args.task,
                    image_shapes_chw=image_shapes_from_observation(obs),
                ),
            )

        for episode in range(args.episodes):
            wait_until_start_pose(
                adapter,
                args=args,
                task=args.task,
                start_state=start_state,
                episode=episode,
                preview=preview,
            )
            start_obs = wait_for_operator_start(adapter, task=args.task, episode=episode, preview=preview)
            frames = record_episode(
                adapter,
                task=args.task,
                fps=args.fps,
                max_duration_sec=args.max_duration_sec,
                preview=preview,
                initial_observation=start_obs,
            )
            print(f"episode {episode} frames={len(frames)}")
            if dataset is not None:
                episode_id = int(getattr(dataset, "num_episodes", 0))
                write_episode(dataset, frames)
                written_episodes += 1
                save_episode_metadata(
                    Path(args.output),
                    episode_id=episode_id,
                    args=args,
                    frame_count=len(frames),
                    start_state=start_state,
                )
            else:
                print("dry-run/no-output: episode not written")

        if dataset is not None:
            dataset.finalize()
            dataset_finalized = True
            print(f"dataset_finalized=True episodes_written={written_episodes}")
    finally:
        preview.close()
        if dataset is not None and not dataset_finalized:
            try:
                dataset.finalize()
            except Exception:
                pass
        cleanup()

    print("collection entrypoint finished")
    print("NO MOTION COMMAND SENT")
    return 0


def build_adapter(args: argparse.Namespace) -> tuple[PiperSmolVLAAdapter, Callable[[], None]]:
    if args.allow_hardware_readonly:
        state_source = RealPiperStateSource(
            RealPiperStateConfig(
                allow_hardware_readonly=True,
                can_port=args.can_port,
            )
        )
        camera_source = RealCameraSource(
            RealCameraConfig(
                allow_hardware_readonly=True,
                global_camera=args.global_camera,
                wrist_camera=args.wrist_camera,
                fps=args.camera_fps,
            )
        )

        def cleanup() -> None:
            camera_source.close()
            state_source.disconnect()

        return PiperSmolVLAAdapter(state_source=state_source, image_source=camera_source), cleanup

    if not args.dry_run:
        raise SystemExit("--allow-hardware-readonly is required unless --dry-run is used")
    state = validate_state(float(part.strip()) for part in args.state.split(","))
    image = np.zeros((480, 640, 3), dtype=np.uint8)
    io = DryRunPiperIO(state)
    camera_source = StaticImageSource({GLOBAL_IMAGE_KEY: image, WRIST_IMAGE_KEY: image})
    return PiperSmolVLAAdapter(state_source=io, image_source=camera_source, config=PiperSmolVLAAdapterConfig()), lambda: None


def preflight_output_dir(args: argparse.Namespace) -> None:
    """在连接硬件前检查输出目录。

    LeRobotDataset.create 要求 root 不存在；这里提前检查，避免相机和 Piper
    都打开后才因为目录冲突退出。
    """

    if args.dry_run or not args.output:
        return
    output = Path(args.output)
    if not output.exists():
        return
    if not args.overwrite_output:
        raise SystemExit(
            f"--output already exists: {output}\n"
            "Use a new --output path, or pass --overwrite-output to delete and recreate it."
        )
    assert_safe_overwrite_path(output)
    shutil.rmtree(output)
    print(f"overwrote_existing_output={output}")


def assert_safe_overwrite_path(path: Path) -> None:
    resolved = path.resolve()
    cwd = Path.cwd().resolve()
    forbidden = {Path("/").resolve(), Path.home().resolve(), cwd}
    if resolved in forbidden:
        raise SystemExit(f"refusing to overwrite unsafe output path: {resolved}")
    if not str(resolved).startswith(str(cwd / "data") + os_sep()):
        raise SystemExit(
            f"refusing to overwrite path outside project data directory: {resolved}\n"
            "Use a fresh --output path instead."
        )


def os_sep() -> str:
    import os

    return os.sep


def image_shapes_from_observation(observation: dict) -> dict[str, tuple[int, int, int]]:
    return {
        GLOBAL_IMAGE_KEY: tuple(int(dim) for dim in image_to_chw_uint8(observation[GLOBAL_IMAGE_KEY]).shape),
        WRIST_IMAGE_KEY: tuple(int(dim) for dim in image_to_chw_uint8(observation[WRIST_IMAGE_KEY]).shape),
    }


def resolve_start_state(text: str, observation: dict) -> tuple[tuple[float, ...], str]:
    value = text.strip().lower()
    if value in ("", "verified", "standard", "piper"):
        return validate_state(VERIFIED_START_QPOS), "local VERIFIED_START_QPOS"
    if value in ("current", "precheck"):
        return validate_state(observation[STATE_KEY]), "current precheck qpos"
    return parse_start_state(text), "--start-state explicit 7D qpos"


def parse_start_state(text: str) -> tuple[float, ...]:
    return validate_state(float(part.strip()) for part in text.split(","))


def assert_dual_view_observation(observation: dict) -> None:
    missing = [key for key in (GLOBAL_IMAGE_KEY, WRIST_IMAGE_KEY) if key not in observation]
    if missing:
        raise KeyError(f"missing camera views: {missing}")
    global_img = np.asarray(observation[GLOBAL_IMAGE_KEY])
    wrist_img = np.asarray(observation[WRIST_IMAGE_KEY])
    for key, image in ((GLOBAL_IMAGE_KEY, global_img), (WRIST_IMAGE_KEY, wrist_img)):
        if image.ndim != 3 or image.shape[-1] != 3:
            raise ValueError(f"{key} must be HWC RGB image, got shape={image.shape}")
        if image.size == 0:
            raise ValueError(f"{key} is empty")
    print(f"global_mean={float(global_img.mean()):.2f}")
    print(f"wrist_mean={float(wrist_img.mean()):.2f}")


def save_precheck_images(observation: dict, *, args: argparse.Namespace) -> None:
    import cv2

    output_name = Path(args.output).name if args.output else "dry_run"
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    root = Path(args.precheck_image_dir)
    root.mkdir(parents=True, exist_ok=True)
    global_rgb = np.asarray(observation[GLOBAL_IMAGE_KEY])
    wrist_rgb = np.asarray(observation[WRIST_IMAGE_KEY])
    global_path = root / f"{output_name}_global_{ts}.jpg"
    wrist_path = root / f"{output_name}_wrist_{ts}.jpg"
    pair_path = root / f"{output_name}_pair_global_wrist_{ts}.jpg"
    cv2.imwrite(str(global_path), cv2.cvtColor(global_rgb, cv2.COLOR_RGB2BGR))
    cv2.imwrite(str(wrist_path), cv2.cvtColor(wrist_rgb, cv2.COLOR_RGB2BGR))
    cv2.imwrite(str(pair_path), make_labeled_pair_bgr(global_rgb, wrist_rgb))
    print(f"saved_precheck_global={global_path}")
    print(f"saved_precheck_wrist={wrist_path}")
    print(f"saved_precheck_pair={pair_path}")


def make_labeled_pair_bgr(global_rgb: np.ndarray, wrist_rgb: np.ndarray) -> np.ndarray:
    import cv2

    global_bgr = cv2.cvtColor(global_rgb, cv2.COLOR_RGB2BGR)
    wrist_bgr = cv2.cvtColor(wrist_rgb, cv2.COLOR_RGB2BGR)
    g_show = cv2.resize(global_bgr, (640, 480))
    w_show = cv2.resize(wrist_bgr, (640, 480))
    canvas = np.hstack((g_show, w_show))
    cv2.putText(canvas, "global", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.putText(canvas, "wrist", (650, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    return canvas


class CollectionPreview:
    """双视角实时预览窗口。

    只显示图像和采集状态，不触发任何机械臂动作。
    """

    def __init__(self, *, enabled: bool, window_name: str = "Piper SmolVLA Collection | global | wrist"):
        self.enabled = enabled
        self.window_name = window_name
        self._cv2 = None
        if not enabled:
            return
        try:
            import cv2

            self._cv2 = cv2
            cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(self.window_name, 1280, 520)
        except Exception as exc:  # noqa: BLE001
            self.enabled = False
            self._cv2 = None
            print(f"preview_disabled: {exc}")

    def show(self, observation: dict, *, status: str, frame_count: int | None = None) -> int:
        if not self.enabled or self._cv2 is None:
            return -1
        global_rgb = np.asarray(observation[GLOBAL_IMAGE_KEY])
        wrist_rgb = np.asarray(observation[WRIST_IMAGE_KEY])
        canvas = make_labeled_pair_bgr(global_rgb, wrist_rgb)
        lines = [status, "SPACE/ENTER start or stop | Q/ESC quit window"]
        if frame_count is not None:
            lines.insert(1, f"frames: {frame_count}")
        for index, line in enumerate(lines):
            self._cv2.putText(
                canvas,
                line,
                (10, 470 - index * 26),
                self._cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (0, 255, 255),
                2,
            )
        self._cv2.imshow(self.window_name, canvas)
        return int(self._cv2.waitKey(1) & 0xFF)

    def close(self) -> None:
        if self.enabled and self._cv2 is not None:
            try:
                self._cv2.destroyWindow(self.window_name)
            except Exception:
                pass


def wait_until_start_pose(
    adapter: PiperSmolVLAAdapter,
    *,
    args: argparse.Namespace,
    task: str,
    start_state: tuple[float, ...],
    episode: int,
    preview: CollectionPreview,
) -> None:
    print(f"Episode {episode}: move robot to fixed start pose. Live preview is shown if GUI is available.")
    last_print = 0.0
    while True:
        obs = adapter.read_observation(task=task)
        current = validate_state(obs[STATE_KEY])
        ok, diffs = start_pose_ok(
            current,
            start_state,
            mode=args.start_guard_mode,
            joint_tol=args.start_state_tol_rad,
            gripper_tol=args.start_gripper_tol_m,
        )
        max_joint_diff = max(diffs[:6])
        status = (
            f"EP {episode} START GUARD {'OK' if ok else 'WAIT'} | "
            f"max joint diff {max_joint_diff:.4f} | grip diff {diffs[6]:.4f}"
        )
        key = preview.show(obs, status=status)
        if key in (ord("q"), ord("Q"), 27):
            raise KeyboardInterrupt("collection aborted from preview window")
        now = time.monotonic()
        if now - last_print > 1.0:
            print(
                f"episode={episode} start_guard_ok={ok} "
                f"max_joint_diff={max_joint_diff:.6f} gripper_diff={diffs[6]:.6f}"
            )
            if args.start_guard_mode == "zone":
                print(
                    f"zone_gripper_current={current[6]:.6f} "
                    f"zone_gripper_need_open_min={START_GUARD_GRIPPER_OPEN_MIN_M:.6f}"
                )
            last_print = now
        if ok:
            return
        if stdin_has_line():
            sys.stdin.readline()
            print(f"current_state={list(current)}")
            print(f"target_start_state={list(start_state)}")
            print(f"abs_diff={list(diffs)}")
        time.sleep(0.03)


def start_pose_ok(
    current: tuple[float, ...],
    target: tuple[float, ...],
    *,
    mode: str,
    joint_tol: float,
    gripper_tol: float,
) -> tuple[bool, tuple[float, ...]]:
    diffs = tuple(abs(a - b) for a, b in zip(current, target, strict=True))
    if mode == "zone":
        ok = all(
            diff <= tol for diff, tol in zip(diffs[:6], START_GUARD_ZONE_ARM_TOLERANCE_RAD, strict=True)
        ) and current[6] >= START_GUARD_GRIPPER_OPEN_MIN_M
    elif mode == "strict":
        ok = all(diff <= joint_tol for diff in diffs[:6]) and diffs[6] <= gripper_tol
    else:
        raise ValueError(f"unknown start guard mode: {mode}")
    return ok, diffs


def record_episode(
    adapter: PiperSmolVLAAdapter,
    *,
    task: str,
    fps: int,
    max_duration_sec: float,
    preview: CollectionPreview,
    initial_observation: dict | None = None,
) -> list[dict]:
    period = 1.0 / fps
    frames: list[dict] = []
    previous = initial_observation or adapter.read_observation(task=task)
    deadline = time.monotonic() + max_duration_sec
    print("recording... press ENTER in terminal or SPACE/ENTER in preview to stop")
    while time.monotonic() < deadline:
        t0 = time.monotonic()
        current = adapter.read_observation(task=task)
        frames.append(
            make_readonly_transition_frame(
                previous_state=previous[STATE_KEY],
                current_state=current[STATE_KEY],
                previous_images={GLOBAL_IMAGE_KEY: previous[GLOBAL_IMAGE_KEY], WRIST_IMAGE_KEY: previous[WRIST_IMAGE_KEY]},
                task=task,
            )
        )
        key = preview.show(current, status="RECORDING read-only operator demo", frame_count=len(frames))
        if key in (ord("q"), ord("Q"), 27):
            raise KeyboardInterrupt("collection aborted from preview window")
        if key in (ord(" "), 10, 13):
            break
        previous = current
        if stdin_has_line():
            sys.stdin.readline()
            break
        elapsed = time.monotonic() - t0
        if elapsed < period:
            time.sleep(period - elapsed)
    if not frames:
        raise RuntimeError("episode has no frames")
    return frames


def wait_for_operator_start(
    adapter: PiperSmolVLAAdapter,
    *,
    task: str,
    episode: int,
    preview: CollectionPreview,
) -> dict:
    print(f"Episode {episode}: start pose ok; press ENTER in terminal or SPACE/ENTER in preview to start recording")
    while True:
        obs = adapter.read_observation(task=task)
        key = preview.show(obs, status=f"EP {episode} READY: press SPACE/ENTER to record")
        if key in (ord(" "), 10, 13):
            return obs
        if key in (ord("q"), ord("Q"), 27):
            raise KeyboardInterrupt("collection aborted from preview window")
        if stdin_has_line():
            sys.stdin.readline()
            return obs
        time.sleep(0.03)


def save_episode_metadata(
    root: Path,
    *,
    episode_id: int,
    args: argparse.Namespace,
    frame_count: int,
    start_state: tuple[float, ...],
) -> None:
    metadata_dir = root / "meta" / "piper_smolvla_episode_metadata"
    metadata_dir.mkdir(parents=True, exist_ok=True)
    metadata = {
        "task": args.task,
        "camera_keys": [GLOBAL_IMAGE_KEY, WRIST_IMAGE_KEY],
        "state_dim": 7,
        "action_dim": 7,
        "timestamp": datetime.now().astimezone().isoformat(timespec="seconds"),
        "source": "piper_smolvla read-only operator demonstration",
        "hardware_readonly_mode": bool(args.allow_hardware_readonly),
        "action_semantics_note": "Read-only mirror demonstration: observation.state=previous qpos, action=current qpos.",
        "fixed_start_state": list(start_state),
        "fixed_start_state_source": args.start_state,
        "start_guard_mode": args.start_guard_mode,
        "start_state_tolerance_rad": args.start_state_tol_rad,
        "start_gripper_tolerance_m": args.start_gripper_tol_m,
        "camera_fps": args.camera_fps,
        "zone_arm_tolerance_rad": list(START_GUARD_ZONE_ARM_TOLERANCE_RAD),
        "zone_gripper_open_min_m": START_GUARD_GRIPPER_OPEN_MIN_M,
        "precheck_image_dir": args.precheck_image_dir,
        "frame_count": frame_count,
        "no_motion_command_sent": True,
    }
    (metadata_dir / f"episode_{episode_id:06d}.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )


def wait_for_enter(prompt: str) -> None:
    input(prompt + "\n")


def stdin_has_line() -> bool:
    if not sys.stdin.isatty():
        return False
    ready, _, _ = select.select([sys.stdin], [], [], 0.0)
    return bool(ready)


if __name__ == "__main__":
    raise SystemExit(main())
