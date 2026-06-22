#!/usr/bin/env python3
"""采集 left/right paper 语言条件诊断数据集。

数据格式严格走官方 LeRobotDataset v3 写入路径：
``LeRobotDataset.create/resume -> add_frame -> save_episode -> finalize``。

任务定义固定为 global camera 图像中的左/右：
- Place the object on the left paper.
- Place the object on the right paper.

本脚本只记录 operator demonstration。默认不写机器人动作，不 reset，不 enable，
不运行 policy。机械臂动作来自人工示教时的 qpos 变化，写入语义沿用本项目：
``observation.state = previous qpos``，``action = current qpos``。
"""

from __future__ import annotations

import _bootstrap  # noqa: F401

import argparse
import json
import os
import select
import shutil
import sys
import time
from collections import Counter
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

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
from piper_smolvla.cameras import RealCameraConfig, RealCameraSource
from piper_smolvla.real_sources import RealPiperStateConfig, RealPiperStateSource
from piper_smolvla.schema import (
    ACTION_KEY,
    GLOBAL_IMAGE_KEY,
    IMAGE_KEYS,
    START_GUARD_GRIPPER_OPEN_MIN_M,
    START_GUARD_ZONE_ARM_TOLERANCE_RAD,
    STATE_KEY,
    VERIFIED_START_QPOS,
    WRIST_IMAGE_KEY,
)
from piper_smolvla.validation import validate_action, validate_state


LEFT_TASK = "Place the object on the left paper."
RIGHT_TASK = "Place the object on the right paper."
TASKS = (LEFT_TASK, RIGHT_TASK)
TASK_TO_LABEL = {LEFT_TASK: "left", RIGHT_TASK: "right"}


@dataclass
class EpisodeQuality:
    frame_count: int
    duration_sec: float
    actual_fps: float
    global_read_failed: bool
    wrist_read_failed: bool
    black_frames: int
    finite_ok: bool
    big_jump_count: int
    gripper_min: float
    gripper_max: float
    has_close: bool
    has_release: bool
    has_open_close_release: bool
    too_short: bool

    @property
    def ok(self) -> bool:
        return (
            self.frame_count > 0
            and not self.global_read_failed
            and not self.wrist_read_failed
            and self.black_frames == 0
            and self.finite_ok
            and self.big_jump_count == 0
            and self.has_close
            and self.has_release
            and self.has_open_close_release
            and not self.too_short
        )


class PreviewWindow:
    """双摄预览。只显示画面，不触发任何机器人动作。"""

    def __init__(self, enabled: bool):
        self.enabled = enabled
        self.window_name = "Left/Right Paper Collection | global | wrist"
        self._cv2 = None
        if not enabled:
            return
        try:
            import cv2

            self._cv2 = cv2
            cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(self.window_name, 1280, 520)
        except Exception as exc:  # noqa: BLE001
            print(f"preview_disabled={type(exc).__name__}: {exc}")
            self.enabled = False
            self._cv2 = None

    def show(self, observation: dict[str, Any], *, status: str, frame_count: int | None = None) -> int:
        if not self.enabled or self._cv2 is None:
            return -1
        canvas = make_labeled_pair_bgr(
            np.asarray(observation[GLOBAL_IMAGE_KEY]),
            np.asarray(observation[WRIST_IMAGE_KEY]),
        )
        lines = [status, "SPACE/ENTER start/stop | Q/ESC quit preview"]
        if frame_count is not None:
            lines.insert(1, f"frames: {frame_count}")
        for index, line in enumerate(lines[:5]):
            self._cv2.putText(
                canvas,
                line,
                (10, 470 - index * 26),
                self._cv2.FONT_HERSHEY_SIMPLEX,
                0.62,
                (0, 255, 255),
                2,
            )
        self._cv2.imshow(self.window_name, canvas)
        return int(self._cv2.waitKey(1) & 0xFF)

    def close(self) -> None:
        if self.enabled and self._cv2 is not None:
            with suppress_all():
                self._cv2.destroyWindow(self.window_name)


class suppress_all:
    def __enter__(self):
        return None

    def __exit__(self, exc_type, exc, tb):
        return True


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Record left/right paper language-conditioned LeRobot dataset.")
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--repo-id")
    parser.add_argument("--num-left", type=int, default=30)
    parser.add_argument("--num-right", type=int, default=30)
    parser.add_argument("--fps", type=int, default=10)
    parser.add_argument("--camera-fps", type=int, default=30)
    parser.add_argument("--warmup-frames", type=int, default=5)
    parser.add_argument("--global-camera", default="/dev/video6")
    parser.add_argument("--wrist-camera", default="auto")
    parser.add_argument("--can-port", default="can0")
    parser.add_argument("--allow-hardware-readonly", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--preview", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--no-robot-write",
        action="store_true",
        default=True,
        help="Always true for this operator-demo script; kept as an explicit safety flag.",
    )
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--max-episode-seconds", type=float, default=60.0)
    parser.add_argument("--start-with", choices=("left", "right", "alternate"), default="alternate")
    parser.add_argument("--start-state", default="verified", help="'verified', 'current', or comma-separated 7D qpos.")
    parser.add_argument("--start-guard-mode", choices=("zone", "strict"), default="zone")
    parser.add_argument("--start-state-tol-rad", type=float, default=0.08)
    parser.add_argument("--start-gripper-tol-m", type=float, default=0.015)
    parser.add_argument("--min-frames", type=int, default=40)
    parser.add_argument("--max-delta-rad", type=float, default=0.30)
    parser.add_argument("--max-delta-gripper-m", type=float, default=0.03)
    parser.add_argument("--black-threshold", type=float, default=5.0)
    parser.add_argument("--overwrite-empty", action="store_true", help="Allow deleting an existing empty dataset root.")
    return parser.parse_args()


def inspect_project_interfaces() -> None:
    print("project_interfaces:")
    print("  robot_state: src/piper_smolvla/real_sources.py::RealPiperStateSource")
    print("  camera_init: src/piper_smolvla/cameras/source.py::RealCameraSource")
    print("  adapter: src/piper_smolvla/adapter.py::PiperSmolVLAAdapter")
    print("  dataset_writer: src/piper_smolvla/collection.py::create_lerobot_dataset/write_episode")
    print("  schema: src/piper_smolvla/schema.py")
    print("  camera_keys: observation.images.global_rgb, observation.images.wrist_rgb")
    print("  train_xvla_keys: raw canonical keys + LeRobot rename_map to observation.images.image/image2")


def main() -> int:
    args = parse_args()
    dataset_root = Path(args.dataset_root).expanduser()
    repo_id = args.repo_id or f"piper/{dataset_root.name}"

    print_safety_banner(args, dataset_root, repo_id)
    inspect_project_interfaces()

    if args.validate_only:
        return validate_dataset(dataset_root, repo_id=repo_id)

    if args.num_left < 0 or args.num_right < 0:
        raise SystemExit("--num-left/--num-right must be non-negative")
    if args.num_left + args.num_right <= 0:
        raise SystemExit("nothing to collect: --num-left + --num-right must be > 0")
    if not args.dry_run and not args.allow_hardware_readonly:
        raise SystemExit("--allow-hardware-readonly is required unless --dry-run is used")

    prepare_dataset_root(dataset_root, resume=args.resume, overwrite_empty=args.overwrite_empty, dry_run=args.dry_run)

    adapter, cleanup = build_adapter(args)
    preview = PreviewWindow(enabled=args.preview)
    dataset = None
    dataset_finalized = False
    try:
        precheck = adapter.read_observation(task=LEFT_TASK)
        assert_observation_health(precheck)
        preview.show(precheck, status="PRECHECK: global | wrist")
        print("camera_precheck=OK")
        print(f"global_shape={tuple(np.asarray(precheck[GLOBAL_IMAGE_KEY]).shape)}")
        print(f"wrist_shape={tuple(np.asarray(precheck[WRIST_IMAGE_KEY]).shape)}")

        start_state, start_source = resolve_start_state(args.start_state, precheck)
        print(f"fixed_start_state={list(start_state)}")
        print(f"fixed_start_source={start_source}")

        existing_counts = read_existing_counts(dataset_root, repo_id=repo_id) if dataset_root.exists() else Counter()
        print_counts("existing_counts", existing_counts)
        planned_tasks = build_task_plan(args, existing_counts)
        print(f"planned_new_episodes={len(planned_tasks)}")
        if not planned_tasks:
            print("target_counts_already_satisfied=True")
            return 0

        if args.dry_run:
            print("DRY_RUN=True: no dataset will be written")
        else:
            dataset = open_dataset_writer(
                dataset_root,
                repo_id=repo_id,
                fps=args.fps,
                resume=args.resume,
                observation=precheck,
            )

        for task_number, task in enumerate(planned_tasks, start=1):
            label = TASK_TO_LABEL[task]
            print(f"\n=== next_episode {task_number}/{len(planned_tasks)} label={label} ===")
            decision = prompt_before_episode(task, label)
            if decision == "q":
                break
            if decision == "s":
                continue

            wait_until_start_pose(adapter, args=args, task=task, target=start_state, preview=preview)
            episode_frames = record_one_episode(adapter, task=task, args=args, preview=preview)
            quality = validate_episode(episode_frames, task=task, fps=args.fps, args=args)
            print_episode_quality(quality)

            post = input("Save episode? [y save / r rerecord / d discard / q save existing and quit]: ").strip().lower()
            if post == "q":
                break
            if post == "r":
                print("discarded_for_rerecord=True")
                continue
            if post == "d":
                print("discarded=True")
                continue
            if post != "y":
                print("unknown input; discarded=True")
                continue

            if args.dry_run or dataset is None:
                print("dry_run_episode_not_written=True")
            else:
                episode_index = int(getattr(dataset, "num_episodes", 0))
                save_or_discard_episode(dataset, episode_frames, task=task)
                save_episode_sidecar(dataset_root, episode_index=episode_index, task=task, quality=quality)
                print(f"saved_episode_index={episode_index} task={task!r}")

        if dataset is not None:
            dataset.finalize()
            dataset_finalized = True
            print("dataset_finalized=True")
    finally:
        preview.close()
        if dataset is not None and not dataset_finalized:
            with suppress_all():
                dataset.finalize()
        cleanup()

    print("REAL_ACTIONS_SENT=NO")
    print("POLICY_ACTIONS_SENT=NO")
    return 0


def print_safety_banner(args: argparse.Namespace, dataset_root: Path, repo_id: str) -> None:
    print("=" * 72)
    print("LEFT/RIGHT PAPER OPERATOR DEMO COLLECTION")
    print(f"dataset_root={dataset_root}")
    print(f"repo_id={repo_id}")
    print(f"target_left={args.num_left} target_right={args.num_right}")
    print(f"global_camera={args.global_camera} wrist_camera={args.wrist_camera}")
    print(f"fps={args.fps} camera_fps={args.camera_fps}")
    print(f"dry_run={args.dry_run} validate_only={args.validate_only}")
    print("robot_write_enabled=NO")
    print("policy_rollout=NO")
    print("left/right frame=GLOBAL CAMERA IMAGE COORDINATES")
    print("Safety checklist:")
    print("  - desktop is clear")
    print("  - left and right white papers are fixed")
    print("  - object starts in the middle")
    print("  - emergency stop is available")
    print("  - global camera sees left paper, object, right paper")
    print("  - wrist camera can see object during grasp")
    print("=" * 72)


def build_adapter(args: argparse.Namespace) -> tuple[PiperSmolVLAAdapter, Any]:
    if args.dry_run:
        state = validate_state(VERIFIED_START_QPOS)
        image = np.zeros((480, 640, 3), dtype=np.uint8)
        io = DryRunPiperIO(state)
        images = StaticImageSource({GLOBAL_IMAGE_KEY: image, WRIST_IMAGE_KEY: image})
        return (
            PiperSmolVLAAdapter(
                state_source=io,
                image_source=images,
                config=PiperSmolVLAAdapterConfig(),
            ),
            lambda: None,
        )

    state_source = RealPiperStateSource(
        RealPiperStateConfig(allow_hardware_readonly=True, can_port=args.can_port)
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


def prepare_dataset_root(root: Path, *, resume: bool, overwrite_empty: bool, dry_run: bool) -> None:
    if dry_run:
        return
    if not root.exists():
        return
    if resume:
        if not (root / "meta" / "info.json").exists():
            raise SystemExit(f"--resume requested but dataset root is not a LeRobot dataset: {root}")
        return
    if overwrite_empty and not any(root.iterdir()):
        shutil.rmtree(root)
        return
    raise SystemExit(f"dataset root already exists: {root}. Use --resume or a new --dataset-root.")


def open_dataset_writer(
    root: Path,
    *,
    repo_id: str,
    fps: int,
    resume: bool,
    observation: dict[str, Any],
) -> Any:
    if resume:
        from lerobot.datasets.lerobot_dataset import LeRobotDataset

        return LeRobotDataset.resume(repo_id=repo_id, root=root)

    image_shapes = {
        key: tuple(int(dim) for dim in image_to_chw_uint8(observation[key]).shape)
        for key in IMAGE_KEYS
    }
    return create_lerobot_dataset(
        root=root,
        repo_id=repo_id,
        config=CollectionConfig(fps=fps, image_shapes_chw=image_shapes),
    )


def read_existing_counts(root: Path, *, repo_id: str) -> Counter[str]:
    counts: Counter[str] = Counter()
    if not (root / "meta" / "info.json").exists():
        return counts
    try:
        from lerobot.datasets.lerobot_dataset import LeRobotDataset

        dataset = LeRobotDataset(repo_id, root=str(root), tolerance_s=0.5)
    except Exception as exc:  # noqa: BLE001
        raise SystemExit(f"cannot read existing dataset via official LeRobotDataset: {exc}") from exc

    for episode_index in range(dataset.num_episodes):
        ep = dataset.meta.episodes[episode_index]
        tasks = [str(t) for t in ep.get("tasks", [])]
        for task in set(tasks):
            if task in TASK_TO_LABEL:
                counts[TASK_TO_LABEL[task]] += 1
    return counts


def print_counts(title: str, counts: Counter[str]) -> None:
    print(f"{title}: left={counts.get('left', 0)} right={counts.get('right', 0)}")


def build_task_plan(args: argparse.Namespace, existing_counts: Counter[str]) -> list[str]:
    remaining = {
        "left": max(0, args.num_left - int(existing_counts.get("left", 0))),
        "right": max(0, args.num_right - int(existing_counts.get("right", 0))),
    }
    plan: list[str] = []
    next_label = args.start_with if args.start_with in ("left", "right") else "left"
    while remaining["left"] > 0 or remaining["right"] > 0:
        if args.start_with == "alternate":
            candidates = (next_label, "right" if next_label == "left" else "left")
            chosen = next((label for label in candidates if remaining[label] > 0), None)
            next_label = "right" if next_label == "left" else "left"
        else:
            primary = args.start_with
            secondary = "right" if primary == "left" else "left"
            chosen = primary if remaining[primary] > 0 else secondary if remaining[secondary] > 0 else None
        if chosen is None:
            break
        remaining[chosen] -= 1
        plan.append(LEFT_TASK if chosen == "left" else RIGHT_TASK)
    return plan


def prompt_before_episode(task: str, label: str) -> str:
    print(f"prompt={task!r}")
    print(f"target={label.upper()} paper in GLOBAL CAMERA image")
    print("Confirm: object in middle, papers fixed, hand out of view, e-stop ready.")
    while True:
        text = input("Press ENTER to record, 's' skip, 'q' quit: ").strip().lower()
        if text == "":
            return "record"
        if text in ("s", "q"):
            return text


def wait_until_start_pose(
    adapter: PiperSmolVLAAdapter,
    *,
    args: argparse.Namespace,
    task: str,
    target: tuple[float, ...],
    preview: PreviewWindow,
) -> None:
    print("start_guard=waiting")
    while True:
        obs = adapter.read_observation(task=task)
        current = validate_state(obs[STATE_KEY])
        ok, diffs = start_pose_ok(
            current,
            target,
            mode=args.start_guard_mode,
            joint_tol=args.start_state_tol_rad,
            gripper_tol=args.start_gripper_tol_m,
        )
        status = (
            f"START GUARD {'OK' if ok else 'WAIT'} | "
            f"max_joint_diff={max(diffs[:6]):.4f} gripper_diff={diffs[6]:.4f}"
        )
        key = preview.show(obs, status=status)
        if key in (ord("q"), ord("Q"), 27):
            raise KeyboardInterrupt("aborted from preview")
        if ok:
            print("start_guard=OK")
            return
        if stdin_has_line():
            sys.stdin.readline()
            print(f"current_qpos={list(current)}")
            print(f"target_qpos={list(target)}")
            print(f"abs_diff={list(diffs)}")
        time.sleep(0.05)


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
    else:
        ok = all(diff <= joint_tol for diff in diffs[:6]) and diffs[6] <= gripper_tol
    return ok, diffs


def record_one_episode(
    adapter: PiperSmolVLAAdapter,
    *,
    task: str,
    args: argparse.Namespace,
    preview: PreviewWindow,
) -> list[dict[str, Any]]:
    print("warmup...")
    previous = adapter.read_observation(task=task)
    for _ in range(max(0, args.warmup_frames)):
        previous = adapter.read_observation(task=task)
        preview.show(previous, status=f"WARMUP\nTASK: {task}")
        time.sleep(1.0 / max(1, args.fps))

    print("recording: press ENTER in terminal or SPACE/ENTER in preview to stop")
    frames: list[dict[str, Any]] = []
    period = 1.0 / max(1, args.fps)
    deadline = time.monotonic() + args.max_episode_seconds
    while time.monotonic() < deadline:
        t0 = time.monotonic()
        current = adapter.read_observation(task=task)
        frames.append(
            make_readonly_transition_frame(
                previous_state=previous[STATE_KEY],
                current_state=current[STATE_KEY],
                previous_images={
                    GLOBAL_IMAGE_KEY: previous[GLOBAL_IMAGE_KEY],
                    WRIST_IMAGE_KEY: previous[WRIST_IMAGE_KEY],
                },
                task=task,
            )
        )
        key = preview.show(current, status=f"RECORDING\nTASK: {task}", frame_count=len(frames))
        if key in (ord("q"), ord("Q"), 27):
            raise KeyboardInterrupt("aborted from preview")
        if key in (ord(" "), 10, 13):
            break
        if stdin_has_line():
            sys.stdin.readline()
            break
        previous = current
        elapsed = time.monotonic() - t0
        if elapsed < period:
            time.sleep(period - elapsed)
    return frames


def validate_episode(
    frames: list[dict[str, Any]],
    *,
    task: str,
    fps: int,
    args: argparse.Namespace,
) -> EpisodeQuality:
    if not frames:
        return EpisodeQuality(0, 0.0, 0.0, True, True, 0, False, 0, 0.0, 0.0, False, False, False, True)

    actions = np.asarray([validate_action(frame[ACTION_KEY]) for frame in frames], dtype=np.float32)
    states = np.asarray([validate_state(frame[STATE_KEY]) for frame in frames], dtype=np.float32)
    finite_ok = bool(np.isfinite(actions).all() and np.isfinite(states).all())

    global_read_failed = any(GLOBAL_IMAGE_KEY not in frame for frame in frames)
    wrist_read_failed = any(WRIST_IMAGE_KEY not in frame for frame in frames)
    black_frames = 0
    for frame in frames:
        for key in IMAGE_KEYS:
            img = np.asarray(frame[key])
            if img.size == 0 or float(img.mean()) <= args.black_threshold:
                black_frames += 1

    deltas = np.abs(np.diff(actions, axis=0)) if len(actions) > 1 else np.zeros((0, 7), dtype=np.float32)
    arm_jump = (deltas[:, :6] > args.max_delta_rad).any(axis=1) if len(deltas) else np.array([], dtype=bool)
    grip_jump = (deltas[:, 6] > args.max_delta_gripper_m) if len(deltas) else np.array([], dtype=bool)
    big_jump_count = int((arm_jump | grip_jump).sum()) if len(deltas) else 0

    gripper = actions[:, 6]
    has_close = bool((gripper < 0.07).any())
    has_release = bool((gripper > 0.09).any())
    has_open_close_release = detect_open_close_release(gripper)
    frame_count = len(frames)
    duration_sec = frame_count / max(1, fps)
    actual_fps = frame_count / duration_sec if duration_sec > 0 else 0.0
    too_short = frame_count < args.min_frames

    for frame in frames:
        if frame.get("task") != task:
            raise ValueError(f"frame task mismatch: {frame.get('task')!r} != {task!r}")

    return EpisodeQuality(
        frame_count=frame_count,
        duration_sec=duration_sec,
        actual_fps=actual_fps,
        global_read_failed=global_read_failed,
        wrist_read_failed=wrist_read_failed,
        black_frames=black_frames,
        finite_ok=finite_ok,
        big_jump_count=big_jump_count,
        gripper_min=float(gripper.min()),
        gripper_max=float(gripper.max()),
        has_close=has_close,
        has_release=has_release,
        has_open_close_release=has_open_close_release,
        too_short=too_short,
    )


def print_episode_quality(q: EpisodeQuality) -> None:
    print("episode_quality:")
    print(f"  frames={q.frame_count} duration_sec={q.duration_sec:.2f} actual_fps={q.actual_fps:.2f}")
    print(f"  global_read_failed={q.global_read_failed} wrist_read_failed={q.wrist_read_failed}")
    print(f"  black_frames={q.black_frames} finite_ok={q.finite_ok}")
    print(f"  big_jump_count={q.big_jump_count}")
    print(f"  gripper_min={q.gripper_min:.4f} gripper_max={q.gripper_max:.4f}")
    print(f"  close={q.has_close} release={q.has_release} open_close_release={q.has_open_close_release}")
    print(f"  too_short={q.too_short} GOOD={q.ok}")


def save_or_discard_episode(dataset: Any, frames: list[dict[str, Any]], *, task: str) -> None:
    # 所有 dataset 写入集中在这里，方便审查。这里不写机器人动作。
    write_episode(dataset, frames)


def save_episode_sidecar(root: Path, *, episode_index: int, task: str, quality: EpisodeQuality) -> None:
    sidecar = root / "meta" / "left_right_paper_episode_metadata"
    sidecar.mkdir(parents=True, exist_ok=True)
    data = {
        "episode_index": episode_index,
        "task": task,
        "label": TASK_TO_LABEL[task],
        "timestamp": datetime.now().astimezone().isoformat(timespec="seconds"),
        "camera_keys": list(IMAGE_KEYS),
        "state_dim": 7,
        "action_dim": 7,
        "source": "left/right paper operator demonstration",
        "left_right_frame": "global camera image coordinates",
        "action_semantics_note": "observation.state=previous qpos, action=current qpos",
        "quality": quality.__dict__,
        "real_actions_sent": False,
        "policy_actions_sent": False,
    }
    (sidecar / f"episode_{episode_index:06d}.json").write_text(json.dumps(data, indent=2) + "\n")


def validate_dataset(root: Path, *, repo_id: str) -> int:
    if not root.is_dir():
        raise SystemExit(f"dataset root not found: {root}")
    try:
        from lerobot.datasets.lerobot_dataset import LeRobotDataset
    except Exception as exc:  # noqa: BLE001
        raise SystemExit(f"cannot import official LeRobotDataset: {exc}") from exc

    dataset = LeRobotDataset(repo_id, root=str(root), tolerance_s=0.5)
    errors: list[str] = []
    counts: Counter[str] = Counter()
    frame_counts: list[int] = []
    print(f"validate_dataset={root}")
    print(f"episodes={dataset.num_episodes} frames={dataset.num_frames}")
    print("tasks:")
    for task_text, row in dataset.meta.tasks.iterrows():
        print(f"  {int(row['task_index'])}: {task_text}")
        if str(task_text) not in TASKS:
            errors.append(f"unexpected task: {task_text}")

    features = dataset.meta.features
    for key in (STATE_KEY, ACTION_KEY, GLOBAL_IMAGE_KEY, WRIST_IMAGE_KEY):
        if key not in features:
            errors.append(f"missing feature: {key}")
    if tuple(features.get(STATE_KEY, {}).get("shape", ())) != (7,):
        errors.append(f"{STATE_KEY} shape is not (7,)")
    if tuple(features.get(ACTION_KEY, {}).get("shape", ())) != (7,):
        errors.append(f"{ACTION_KEY} shape is not (7,)")

    for ep_idx in range(dataset.num_episodes):
        ep = dataset.meta.episodes[ep_idx]
        tasks = [str(t) for t in ep.get("tasks", [])]
        length = int(ep["length"])
        frame_counts.append(length)
        if len(set(tasks)) != 1:
            errors.append(f"episode {ep_idx}: expected one task, got {tasks}")
            task = tasks[0] if tasks else "<missing>"
        else:
            task = tasks[0]
        if task in TASK_TO_LABEL:
            counts[TASK_TO_LABEL[task]] += 1
        else:
            errors.append(f"episode {ep_idx}: invalid task {task!r}")

        item = dataset[int(ep["dataset_from_index"])]
        if item.get("task") != task:
            errors.append(f"episode {ep_idx}: decoded task mismatch {item.get('task')!r} != {task!r}")
        for key in (STATE_KEY, ACTION_KEY):
            shape = tuple(getattr(item[key], "shape", ()))
            if shape != (7,):
                errors.append(f"episode {ep_idx}: {key} decoded shape {shape}")
        for key in IMAGE_KEYS:
            shape = tuple(getattr(item[key], "shape", ()))
            if len(shape) != 3 or 3 not in (shape[0], shape[-1]):
                errors.append(f"episode {ep_idx}: {key} decoded shape {shape}")

    print_counts("task_counts", counts)
    if frame_counts:
        print(f"frame_count_range={min(frame_counts)}..{max(frame_counts)}")

    if errors:
        print(f"errors={len(errors)}")
        for err in errors:
            print(f"  ERROR: {err}")
        print("LEFT_RIGHT_DATASET_READY=NO")
        return 1
    print("errors=0")
    print("LEFT_RIGHT_DATASET_READY=YES")
    return 0


def detect_open_close_release(gripper: np.ndarray) -> bool:
    open_indices = np.flatnonzero(gripper > 0.09)
    close_indices = np.flatnonzero(gripper < 0.07)
    if len(open_indices) == 0 or len(close_indices) == 0:
        return False
    first_open = int(open_indices[0])
    close_after_open = close_indices[close_indices > first_open]
    if len(close_after_open) == 0:
        return False
    first_close = int(close_after_open[0])
    release_after_close = open_indices[open_indices > first_close]
    return bool(len(release_after_close) > 0)


def assert_observation_health(observation: dict[str, Any]) -> None:
    validate_state(observation[STATE_KEY])
    for key in IMAGE_KEYS:
        image = np.asarray(observation[key])
        if image.ndim != 3 or image.shape[-1] != 3:
            raise ValueError(f"{key} must be HWC RGB, got {image.shape}")
        if image.size == 0:
            raise ValueError(f"{key} is empty")
        print(f"{key}_mean={float(image.mean()):.2f}")


def resolve_start_state(text: str, observation: dict[str, Any]) -> tuple[tuple[float, ...], str]:
    value = text.strip().lower()
    if value in ("", "verified", "standard"):
        return validate_state(VERIFIED_START_QPOS), "VERIFIED_START_QPOS"
    if value in ("current", "precheck"):
        return validate_state(observation[STATE_KEY]), "current precheck qpos"
    return validate_state(float(part.strip()) for part in text.split(",")), "explicit --start-state"


def make_labeled_pair_bgr(global_rgb: np.ndarray, wrist_rgb: np.ndarray) -> np.ndarray:
    import cv2

    global_bgr = cv2.cvtColor(global_rgb, cv2.COLOR_RGB2BGR)
    wrist_bgr = cv2.cvtColor(wrist_rgb, cv2.COLOR_RGB2BGR)
    left = cv2.resize(global_bgr, (640, 480))
    right = cv2.resize(wrist_bgr, (640, 480))
    canvas = np.hstack((left, right))
    cv2.putText(canvas, "global", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.putText(canvas, "wrist", (650, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    return canvas


def stdin_has_line() -> bool:
    if not sys.stdin.isatty():
        return False
    ready, _, _ = select.select([sys.stdin], [], [], 0.0)
    return bool(ready)


if __name__ == "__main__":
    raise SystemExit(main())
