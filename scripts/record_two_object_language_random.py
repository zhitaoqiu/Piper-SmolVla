#!/usr/bin/env python3
"""Record blue/green two-object language-conditioned Piper demonstrations.

This is a real-machine, read-only operator-demonstration recorder:

- robot state is read from the existing Piper state source;
- global/wrist images are read from the existing camera source;
- no robot action, reset, enable, or policy rollout is sent;
- frames are written through the official LeRobotDataset v3 path.
"""

from __future__ import annotations

import _bootstrap  # noqa: F401

import argparse
import csv
import json
import os
import random
import select
import shutil
import sys
import tempfile
import time
from collections import Counter
from contextlib import suppress
from dataclasses import asdict, dataclass, field
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
from piper_smolvla.cameras import (
    DEFAULT_BLACK_FRAME_THRESHOLD,
    DEFAULT_CAMERA_FPS,
    DEFAULT_DATASET_FPS,
    DEFAULT_GLOBAL_CAMERA,
    DEFAULT_WARMUP_FRAMES,
    DEFAULT_WRIST_CAMERA,
    RealCameraConfig,
    RealCameraSource,
)
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


BLUE_TARGET = "blue"
GREEN_TARGET = "green"
BLUE_TASK = "Pick up the blue object and put it into the box."
GREEN_TASK = "Pick up the green object and put it into the box."
TASK_BY_TARGET = {BLUE_TARGET: BLUE_TASK, GREEN_TARGET: GREEN_TASK}
TARGET_BY_TASK = {BLUE_TASK: BLUE_TARGET, GREEN_TASK: GREEN_TARGET}
EXPECTED_TASKS = (BLUE_TASK, GREEN_TASK)
SCHEDULE_VERSION = 1
DEFAULT_HF_DATASETS_CACHE = "/tmp/piper_smolvla_hf_cache/datasets"


@dataclass(frozen=True)
class TaskSchedule:
    path: Path
    targets: list[str]
    seed: int
    num_blue: int
    num_green: int
    source: str
    should_write: bool = False

    @property
    def total(self) -> int:
        return len(self.targets)


@dataclass
class EpisodeRecord:
    frames: list[dict[str, Any]]
    timestamps: list[float]
    timed_out: bool


@dataclass
class EpisodeQuality:
    episode_index: int
    target: str
    task: str
    frame_count: int
    duration_sec: float
    actual_fps: float
    global_frame_failures: int
    wrist_frame_failures: int
    black_frame_count: int
    finite_ok: bool
    state_shape_ok: bool
    action_shape_ok: bool
    duplicate_timestamp_count: int
    max_action_jump: float
    big_jump_count: int
    action_min: list[float]
    action_max: list[float]
    gripper_min: float
    gripper_max: float
    has_close: bool
    has_release: bool
    has_open_close_release: bool
    too_short: bool
    timed_out: bool
    status: str
    warnings: list[str] = field(default_factory=list)
    failures: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return self.status == "PASS"


class PreviewWindow:
    """Dual-camera preview only; it never sends robot actions."""

    def __init__(self, *, enabled: bool):
        self.enabled = enabled
        self.window_name = "Two Object Language Collection | global | wrist"
        self.backend = "none"
        self._cv2 = None
        self._tk = None
        self._tk_root = None
        self._tk_label = None
        self._tk_photo = None
        self._last_key = -1
        if not enabled:
            return
        try:
            import cv2

            self._cv2 = cv2
            cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(self.window_name, 1280, 520)
            self.backend = "opencv"
        except Exception as exc:  # noqa: BLE001
            print(f"opencv_preview_unavailable={type(exc).__name__}: {exc}")
            self._cv2 = None
            self._init_tk_preview()

    def _init_tk_preview(self) -> None:
        try:
            import tkinter as tk

            root = tk.Tk()
            root.title(self.window_name)
            root.geometry("1280x520")
            label = tk.Label(root)
            label.pack(fill=tk.BOTH, expand=True)

            def on_key(event: Any) -> None:
                if event.keysym == "Escape":
                    self._last_key = 27
                elif event.keysym in ("Return", "KP_Enter"):
                    self._last_key = 13
                elif event.char:
                    self._last_key = ord(event.char[0])

            root.bind("<Key>", on_key)
            self._tk = tk
            self._tk_root = root
            self._tk_label = label
            self.backend = "tk"
            print("preview_backend=tk")
        except Exception as exc:  # noqa: BLE001
            print(f"preview_disabled={type(exc).__name__}: {exc}")
            self.enabled = False

    def show(self, observation: dict[str, Any], *, status: str, frame_count: int | None = None) -> int:
        if not self.enabled:
            return -1
        lines = [status, "SPACE/ENTER start or stop | Q/ESC quit"]
        if frame_count is not None:
            lines.insert(1, f"frames: {frame_count}")
        global_rgb = image_to_hwc_uint8(observation[GLOBAL_IMAGE_KEY])
        wrist_rgb = image_to_hwc_uint8(observation[WRIST_IMAGE_KEY])
        if self.backend == "opencv" and self._cv2 is not None:
            canvas = make_labeled_pair_bgr(global_rgb, wrist_rgb)
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
        if self.backend == "tk":
            return self._show_tk(global_rgb, wrist_rgb, lines=lines)
        return -1

    def _show_tk(self, global_rgb: np.ndarray, wrist_rgb: np.ndarray, *, lines: list[str]) -> int:
        if self._tk_root is None or self._tk_label is None:
            return -1
        from PIL import Image, ImageDraw, ImageTk

        left = Image.fromarray(global_rgb).resize((640, 480))
        right = Image.fromarray(wrist_rgb).resize((640, 480))
        canvas = Image.new("RGB", (1280, 480), (0, 0, 0))
        canvas.paste(left, (0, 0))
        canvas.paste(right, (640, 0))
        draw = ImageDraw.Draw(canvas)
        draw.text((10, 10), "global", fill=(0, 255, 0))
        draw.text((650, 10), "wrist", fill=(0, 255, 0))
        y = 452
        for line in lines[:5]:
            for part in str(line).splitlines()[:2]:
                draw.text((10, y), part, fill=(255, 255, 0))
                y -= 18
        self._tk_photo = ImageTk.PhotoImage(canvas)
        self._tk_label.configure(image=self._tk_photo)
        self._tk_root.update_idletasks()
        self._tk_root.update()
        key = self._last_key
        self._last_key = -1
        return key

    def show_frame(self, frame: dict[str, Any], *, status: str) -> int:
        return self.show(
            {
                GLOBAL_IMAGE_KEY: frame[GLOBAL_IMAGE_KEY],
                WRIST_IMAGE_KEY: frame[WRIST_IMAGE_KEY],
            },
            status=status,
        )

    def close(self) -> None:
        if self.enabled and self._cv2 is not None:
            with suppress(Exception):
                self._cv2.destroyWindow(self.window_name)
        if self.enabled and self._tk_root is not None:
            with suppress(Exception):
                self._tk_root.destroy()


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Record a balanced blue/green two-object language-conditioned LeRobot v3 dataset."
    )
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--repo-id")
    parser.add_argument("--num-blue", type=int, default=100)
    parser.add_argument("--num-green", type=int, default=100)
    parser.add_argument("--fps", type=int, default=DEFAULT_DATASET_FPS)
    parser.add_argument("--camera-fps", type=int, default=DEFAULT_CAMERA_FPS)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--global-camera", default=DEFAULT_GLOBAL_CAMERA)
    parser.add_argument("--wrist-camera", default=DEFAULT_WRIST_CAMERA)
    parser.add_argument("--warmup-frames", type=int, default=DEFAULT_WARMUP_FRAMES)
    parser.add_argument("--preview", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--no-robot-write",
        action="store_true",
        default=True,
        help="Always true for this recorder; kept as an explicit safety flag.",
    )
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--max-episode-seconds", type=float, default=120.0)
    parser.add_argument("--max-frame-jump", type=float, default=0.30)
    parser.add_argument("--black-frame-threshold", type=float, default=DEFAULT_BLACK_FRAME_THRESHOLD)
    parser.add_argument("--schedule-file")
    parser.add_argument("--placement-metadata", action="store_true")
    parser.add_argument("--can-port", default="can0")
    parser.add_argument("--allow-hardware-readonly", action="store_true")
    parser.add_argument("--mock-hardware", action="store_true", help="Use static mock state/images for offline tests.")
    parser.add_argument("--start-state", default="verified", help="'verified', 'current', or comma-separated 7D qpos.")
    parser.add_argument("--start-guard-mode", choices=("zone", "strict"), default="zone")
    parser.add_argument("--start-state-tol-rad", type=float, default=0.08)
    parser.add_argument("--start-gripper-tol-m", type=float, default=0.015)
    parser.add_argument("--min-frames", type=int, default=20)
    parser.add_argument("--wrist-auto-exposure", type=int, default=None)
    parser.add_argument("--wrist-exposure", type=int, default=None)
    parser.add_argument("--wrist-gain", type=float, default=None)
    parser.add_argument("--wrist-brightness", type=float, default=None)
    parser.add_argument("--wrist-power-line", type=int, default=None)
    return parser.parse_args(argv)


def main() -> int:
    args = parse_args()
    dataset_root = Path(args.dataset_root).expanduser()
    repo_id = args.repo_id or f"piper/{dataset_root.name}"

    configure_hf_cache()
    interface_info = inspect_project_interfaces()

    if args.validate_only:
        schedule_path = resolve_schedule_path(args, dataset_root)
        return validate_dataset(dataset_root, repo_id=repo_id, args=args, schedule_path=schedule_path)

    validate_collection_args(args)
    recovered_schedule = recover_empty_partial_dataset_root(dataset_root, dry_run=args.dry_run)
    prepare_dataset_root(dataset_root, resume=args.resume, dry_run=args.dry_run)

    schedule = load_or_create_task_schedule(args, dataset_root, recovered_schedule=recovered_schedule)
    existing_targets = read_existing_targets(dataset_root, repo_id=repo_id)
    verify_schedule_prefix(schedule.targets, existing_targets)

    print_safety_banner(args, dataset_root, repo_id)

    adapter, cleanup = build_adapter(args)
    preview = PreviewWindow(enabled=args.preview)
    dataset = None
    dataset_finalized = False
    written_this_run = 0
    quit_requested = False

    try:
        precheck = adapter.read_observation(task=BLUE_TASK)
        assert_observation_health(precheck, threshold=args.black_frame_threshold)
        preview.show(precheck, status="PRECHECK global | wrist")
        start_state, start_source = resolve_start_state(args.start_state, precheck)
        print_ready_guard_precheck(precheck, args=args, target=start_state)

        print_startup_summary(
            args=args,
            dataset_root=dataset_root,
            repo_id=repo_id,
            interface_info=interface_info,
            schedule=schedule,
            existing_targets=existing_targets,
            observation=precheck,
            start_state=start_state,
            start_source=start_source,
        )

        if args.dry_run:
            check_dataset_writer_interface(precheck, fps=args.fps, repo_id=repo_id)
            print("DRY_RUN=True")
            print("dataset_write=NO")
            print("episode_recording=NO")
            print("REAL_ACTIONS_SENT=NO")
            return 0

        dataset = open_dataset_writer(
            dataset_root,
            repo_id=repo_id,
            fps=args.fps,
            resume=dataset_root.exists() and (dataset_root / "meta" / "info.json").exists(),
            observation=precheck,
        )
        if schedule.should_write:
            save_task_schedule(schedule)

        completed = len(existing_targets)
        while completed < schedule.total and not quit_requested:
            target = schedule.targets[completed]
            task = TASK_BY_TARGET[target]
            placement: dict[str, str] = {}
            attempt = 1
            while not quit_requested:
                post = ""
                decision = prompt_before_episode(
                    episode_number=completed + 1,
                    total=schedule.total,
                    target=target,
                    task=task,
                    attempt=attempt,
                )
                if decision == "q":
                    quit_requested = True
                    break
                if decision == "s":
                    print("skipped_preparation=True target_not_counted=True")
                    attempt += 1
                    continue

                placement = collect_placement_metadata(args.placement_metadata)
                wait_until_start_pose(adapter, args=args, task=task, target=start_state, preview=preview)
                print("\nReady! Press ENTER to START recording...")
                wait_for_record_start(adapter, task=task, preview=preview)
                record = record_one_episode(adapter, task=task, args=args, preview=preview)
                quality = validate_episode(
                    record,
                    task=task,
                    target=target,
                    episode_index=completed,
                    fps=args.fps,
                    args=args,
                )
                print_episode_quality(quality)
                print_manual_quality_reminder()

                while True:
                    post = ask_save_decision(quality)
                    if post == "p":
                        preview_episode_summary(record, quality=quality, preview=preview)
                        continue
                    if post == "q":
                        quit_requested = True
                        break
                    if post == "r":
                        print("discarded_for_rerecord=True target_not_counted=True")
                        attempt += 1
                        break
                    if post == "d":
                        print("discarded=True target_not_counted=True rerandomize_scene=True")
                        attempt += 1
                        break
                    if post == "y":
                        episode_index = int(getattr(dataset, "num_episodes", completed))
                        save_episode_checked(dataset, record.frames)
                        save_episode_sidecar(
                            dataset_root,
                            episode_index=episode_index,
                            schedule_index=completed,
                            target=target,
                            task=task,
                            quality=quality,
                            placement=placement,
                            args=args,
                        )
                        completed += 1
                        written_this_run += 1
                        print(f"saved_episode_index={episode_index} target={target} task={task!r}")
                        break

                if post == "y" or quit_requested:
                    break

        if dataset is not None:
            dataset.finalize()
            dataset_finalized = True
            print("dataset_finalized=True")
    except KeyboardInterrupt:
        print("\nkeyboard_interrupt=True")
        quit_requested = True
    finally:
        safe_shutdown(preview=preview, cleanup=cleanup, dataset=dataset, dataset_finalized=dataset_finalized)

    print(f"episodes_written_this_run={written_this_run}")
    print("REAL_ACTIONS_SENT=NO")
    print("POLICY_ACTIONS_SENT=NO")
    if quit_requested:
        print("collection_stopped_before_schedule_complete=True")
    return 0


def configure_hf_cache() -> None:
    cache = Path(os.environ.setdefault("HF_DATASETS_CACHE", DEFAULT_HF_DATASETS_CACHE)).expanduser()
    cache.mkdir(parents=True, exist_ok=True)


def inspect_project_interfaces() -> dict[str, str]:
    try:
        import lerobot
        from lerobot.datasets import lerobot_dataset as lrd
    except Exception as exc:  # noqa: BLE001
        raise SystemExit(f"cannot import installed LeRobot: {type(exc).__name__}: {exc}") from exc

    info = {
        "lerobot_version": str(getattr(lerobot, "__version__", "unknown")),
        "lerobot_file": str(Path(lerobot.__file__).resolve()),
        "lerobot_dataset_file": str(Path(lrd.__file__).resolve()),
        "dataset_format_version": str(getattr(lrd, "CODEBASE_VERSION", "unknown")),
        "robot_state": "src/piper_smolvla/real_sources.py::RealPiperStateSource",
        "camera_source": "src/piper_smolvla/cameras/source.py::RealCameraSource",
        "adapter": "src/piper_smolvla/adapter.py::PiperSmolVLAAdapter",
        "dataset_writer": "src/piper_smolvla/collection.py::create_lerobot_dataset/write_episode",
        "schema": "src/piper_smolvla/schema.py",
        "smolvla_loader": "src/piper_smolvla/policy_io.py::prepare_policy_batch",
        "xvla_launcher": "scripts/train_xvla.py::PIPER_TO_XVLA_RENAME_MAP",
    }
    print("project_interface_audit:")
    for key, value in info.items():
        print(f"  {key}: {value}")
    return info


def validate_collection_args(args: argparse.Namespace) -> None:
    if args.num_blue < 0 or args.num_green < 0:
        raise SystemExit("--num-blue/--num-green must be non-negative")
    if args.num_blue + args.num_green <= 0:
        raise SystemExit("nothing to collect: --num-blue + --num-green must be > 0")
    if args.fps <= 0:
        raise SystemExit("--fps must be positive")
    if args.camera_fps <= 0:
        raise SystemExit("--camera-fps must be positive")
    if args.max_episode_seconds <= 0:
        raise SystemExit("--max-episode-seconds must be positive")
    if args.max_frame_jump <= 0:
        raise SystemExit("--max-frame-jump must be positive")


def prepare_dataset_root(root: Path, *, resume: bool, dry_run: bool) -> None:
    if dry_run:
        return
    if not root.exists():
        return
    if resume and (root / "meta" / "info.json").exists():
        return
    if not resume and (root / "meta" / "info.json").exists():
        raise SystemExit(f"dataset root already exists: {root}. Use --resume to append.")
    raise SystemExit(
        f"dataset root exists but is not a LeRobot dataset: {root}. "
        "Use a fresh --dataset-root; this script will not delete existing files."
    )


def resolve_schedule_path(args: argparse.Namespace, dataset_root: Path) -> Path:
    if args.schedule_file:
        return Path(args.schedule_file).expanduser()
    return dataset_root / "meta" / "two_object_language_schedule.json"


def load_or_create_task_schedule(
    args: argparse.Namespace,
    dataset_root: Path,
    *,
    recovered_schedule: dict[str, Any] | None = None,
) -> TaskSchedule:
    path = resolve_schedule_path(args, dataset_root)
    if path.exists():
        data = json.loads(path.read_text(encoding="utf-8"))
        targets = targets_from_schedule_json(data)
        expected_counts = Counter(targets)
        if int(data.get("seed", args.seed)) != args.seed:
            print(f"WARNING: schedule seed={data.get('seed')} differs from --seed={args.seed}; using schedule file.")
        if expected_counts[BLUE_TARGET] != args.num_blue or expected_counts[GREEN_TARGET] != args.num_green:
            raise SystemExit(
                "schedule counts do not match requested counts: "
                f"file blue={expected_counts[BLUE_TARGET]} green={expected_counts[GREEN_TARGET]} "
                f"args blue={args.num_blue} green={args.num_green}"
            )
        return TaskSchedule(
            path=path,
            targets=targets,
            seed=int(data.get("seed", args.seed)),
            num_blue=expected_counts[BLUE_TARGET],
            num_green=expected_counts[GREEN_TARGET],
            source="existing",
            should_write=False,
        )

    if recovered_schedule is not None:
        targets = targets_from_schedule_json(recovered_schedule)
        counts = Counter(targets)
        if counts[BLUE_TARGET] != args.num_blue or counts[GREEN_TARGET] != args.num_green:
            raise SystemExit(
                "recovered schedule counts do not match requested counts: "
                f"file blue={counts[BLUE_TARGET]} green={counts[GREEN_TARGET]} "
                f"args blue={args.num_blue} green={args.num_green}"
            )
        return TaskSchedule(
            path=path,
            targets=targets,
            seed=int(recovered_schedule.get("seed", args.seed)),
            num_blue=counts[BLUE_TARGET],
            num_green=counts[GREEN_TARGET],
            source="recovered-partial-root",
            should_write=not args.dry_run,
        )

    targets = generate_task_schedule(args.num_blue, args.num_green, seed=args.seed)
    return TaskSchedule(
        path=path,
        targets=targets,
        seed=args.seed,
        num_blue=args.num_blue,
        num_green=args.num_green,
        source="generated",
        should_write=not args.dry_run,
    )


def generate_task_schedule(num_blue: int, num_green: int, *, seed: int) -> list[str]:
    targets = [BLUE_TARGET] * num_blue + [GREEN_TARGET] * num_green
    rng = random.Random(seed)
    rng.shuffle(targets)
    return targets


def targets_from_schedule_json(data: dict[str, Any]) -> list[str]:
    if "targets" in data:
        raw_targets = data["targets"]
    else:
        raw_targets = [row.get("target") for row in data.get("tasks", [])]
    targets = [normalize_target(value) for value in raw_targets]
    if not targets:
        raise ValueError("schedule file has no targets")
    return targets


def normalize_target(value: Any) -> str:
    target = str(value).strip().lower()
    if target not in (BLUE_TARGET, GREEN_TARGET):
        raise ValueError(f"invalid target in schedule: {value!r}")
    return target


def save_task_schedule(schedule: TaskSchedule) -> None:
    schedule.path.parent.mkdir(parents=True, exist_ok=True)
    data = {
        "version": SCHEDULE_VERSION,
        "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "seed": schedule.seed,
        "num_blue": schedule.num_blue,
        "num_green": schedule.num_green,
        "total": schedule.total,
        "prompts": {BLUE_TARGET: BLUE_TASK, GREEN_TARGET: GREEN_TASK},
        "targets": schedule.targets,
        "tasks": [
            {"schedule_index": index, "target": target, "task": TASK_BY_TARGET[target]}
            for index, target in enumerate(schedule.targets)
        ],
        "note": "Saved before the first episode; resume must reuse this file without reshuffling.",
    }
    schedule.path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
    print(f"task_schedule_saved={schedule.path}")


def recover_empty_partial_dataset_root(root: Path, *, dry_run: bool) -> dict[str, Any] | None:
    """Move aside an empty half-created LeRobot root and keep its schedule.

    LeRobotDataset.create writes ``meta/info.json`` very early. If startup is
    interrupted before the first episode, the root is not a readable dataset
    yet, and LeRobotDataset(repo_id, root=...) may try the Hub. For this
    recorder, the only safe automatic recovery is an empty root with no data,
    videos, tasks, episodes, or stats. The old directory is renamed, not
    deleted.
    """

    if dry_run or not root.exists():
        return None
    info = root / "meta" / "info.json"
    schedule = root / "meta" / "two_object_language_schedule.json"
    if not info.exists() or is_complete_lerobot_dataset(root):
        return None
    unsafe_paths = [
        root / "data",
        root / "videos",
        root / "meta" / "episodes",
        root / "meta" / "tasks.parquet",
        root / "meta" / "stats.json",
    ]
    if any(path.exists() for path in unsafe_paths):
        raise SystemExit(
            f"incomplete dataset root is not empty enough to auto-recover: {root}. "
            "Please inspect it manually before continuing."
        )
    recovered = json.loads(schedule.read_text(encoding="utf-8")) if schedule.exists() else None
    backup = unique_backup_path(root)
    shutil.move(str(root), str(backup))
    print(f"partial_empty_dataset_root_moved={backup}")
    if recovered is not None:
        print("partial_schedule_recovered=True")
    return recovered


def is_complete_lerobot_dataset(root: Path) -> bool:
    return (
        (root / "meta" / "info.json").exists()
        and (root / "meta" / "tasks.parquet").exists()
        and (root / "meta" / "episodes").exists()
        and (root / "data").exists()
    )


def unique_backup_path(root: Path) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base = root.with_name(f"{root.name}_partial_{timestamp}")
    candidate = base
    suffix = 1
    while candidate.exists():
        suffix += 1
        candidate = root.with_name(f"{base.name}_{suffix}")
    return candidate


def read_existing_targets(root: Path, *, repo_id: str) -> list[str]:
    if not (root / "meta" / "info.json").exists():
        return []
    try:
        from lerobot.datasets.lerobot_dataset import LeRobotDataset
    except Exception as exc:  # noqa: BLE001
        raise SystemExit(f"cannot import LeRobotDataset: {exc}") from exc

    try:
        dataset = LeRobotDataset(repo_id, root=str(root), tolerance_s=0.5)
    except Exception as exc:  # noqa: BLE001
        raise SystemExit(f"cannot read existing dataset via official LeRobotDataset: {exc}") from exc

    targets: list[str] = []
    for ep_idx in range(dataset.num_episodes):
        ep = dataset.meta.episodes[ep_idx]
        tasks = [str(task) for task in ep.get("tasks", [])]
        if len(set(tasks)) != 1:
            raise SystemExit(f"episode {ep_idx} must contain exactly one task, got {tasks}")
        task = tasks[0]
        if task not in TARGET_BY_TASK:
            raise SystemExit(f"episode {ep_idx} has unexpected task: {task!r}")
        targets.append(TARGET_BY_TASK[task])
    return targets


def verify_schedule_prefix(schedule_targets: list[str], existing_targets: list[str]) -> None:
    if len(existing_targets) > len(schedule_targets):
        raise SystemExit(
            f"existing dataset has {len(existing_targets)} episodes, schedule has only {len(schedule_targets)}"
        )
    for index, target in enumerate(existing_targets):
        if schedule_targets[index] != target:
            raise SystemExit(
                f"existing episode {index} target={target!r} does not match schedule target={schedule_targets[index]!r}"
            )


def print_safety_banner(args: argparse.Namespace, dataset_root: Path, repo_id: str) -> None:
    print("=" * 78)
    print("BLUE/GREEN TWO-OBJECT LANGUAGE COLLECTION")
    print(f"dataset_root={dataset_root}")
    print(f"repo_id={repo_id}")
    print(f"global_camera={args.global_camera} wrist_camera={args.wrist_camera}")
    print(f"fps={args.fps} camera_fps={args.camera_fps}")
    print(f"dry_run={args.dry_run} validate_only={args.validate_only}")
    print("robot_write_enabled=NO")
    print("policy_rollout=NO")
    print("Safety checklist:")
    print("  - 确认急停可用")
    print("  - 确认桌面无遮挡")
    print("  - 确认盒子固定")
    print("  - 确认蓝绿物体在安全随机框内")
    print("  - 确认 global camera 能同时看到两个物体和盒子")
    print("  - 确认 wrist camera 抓取时能看到目标")
    print("=" * 78)


def print_startup_summary(
    *,
    args: argparse.Namespace,
    dataset_root: Path,
    repo_id: str,
    interface_info: dict[str, str],
    schedule: TaskSchedule,
    existing_targets: list[str],
    observation: dict[str, Any],
    start_state: tuple[float, ...],
    start_source: str,
) -> None:
    counts = Counter(existing_targets)
    image_shapes = image_shapes_from_observation(observation)
    print("\nstartup_summary:")
    print(f"  dataset_root: {dataset_root}")
    print(f"  repo_id: {repo_id}")
    print(f"  lerobot_version: {interface_info['lerobot_version']}")
    print(f"  dataset_format_version: {interface_info['dataset_format_version']}")
    print(f"  existing_episodes: {len(existing_targets)}")
    print(f"  blue_done: {counts[BLUE_TARGET]}")
    print(f"  green_done: {counts[GREEN_TARGET]}")
    print(f"  blue_remaining: {schedule.num_blue - counts[BLUE_TARGET]}")
    print(f"  green_remaining: {schedule.num_green - counts[GREEN_TARGET]}")
    print(f"  state_shape: (7,)")
    print(f"  action_shape: (7,)")
    print(f"  camera_keys: {list(IMAGE_KEYS)}")
    print(f"  image_shapes_chw: {image_shapes}")
    print(f"  fps: {args.fps}")
    print(f"  robot_write_enabled: NO")
    print(f"  schedule_file: {schedule.path}")
    print(f"  schedule_source: {schedule.source}")
    print(f"  schedule_seed: {schedule.seed}")
    print(f"  schedule_total: {schedule.total}")
    print(f"  start_state_source: {start_source}")
    print(f"  start_state: {list(start_state)}")


def build_adapter(args: argparse.Namespace) -> tuple[PiperSmolVLAAdapter, Any]:
    if args.mock_hardware:
        state = validate_state(VERIFIED_START_QPOS)
        global_img = np.full((480, 640, 3), (96, 120, 150), dtype=np.uint8)
        wrist_img = np.full((480, 640, 3), (120, 110, 90), dtype=np.uint8)
        io = DryRunPiperIO(state)
        images = StaticImageSource({GLOBAL_IMAGE_KEY: global_img, WRIST_IMAGE_KEY: wrist_img})
        adapter = PiperSmolVLAAdapter(
            state_source=io,
            image_source=images,
            config=PiperSmolVLAAdapterConfig(),
        )
        return adapter, lambda: None

    state_source = RealPiperStateSource(
        RealPiperStateConfig(allow_hardware_readonly=True, can_port=args.can_port)
    )
    camera_source = RealCameraSource(
        RealCameraConfig(
            allow_hardware_readonly=True,
            global_camera=args.global_camera,
            wrist_camera=args.wrist_camera,
            fps=args.camera_fps,
            black_threshold=args.black_frame_threshold,
            warmup_frames=args.warmup_frames,
            wrist_auto_exposure=args.wrist_auto_exposure,
            wrist_exposure_absolute=args.wrist_exposure,
            wrist_gain=args.wrist_gain,
            wrist_brightness=args.wrist_brightness,
            wrist_power_line_frequency=args.wrist_power_line,
        )
    )

    def cleanup() -> None:
        camera_source.close()
        state_source.disconnect()

    return PiperSmolVLAAdapter(state_source=state_source, image_source=camera_source), cleanup


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

        return LeRobotDataset(repo_id, root=str(root), tolerance_s=0.5)

    image_shapes = image_shapes_from_observation(observation)
    return create_lerobot_dataset(
        root=root,
        repo_id=repo_id,
        config=CollectionConfig(fps=fps, image_shapes_chw=image_shapes),
    )


def check_dataset_writer_interface(observation: dict[str, Any], *, fps: int, repo_id: str) -> None:
    with tempfile.TemporaryDirectory(prefix="piper_two_object_lerobot_dry_") as tmp:
        root = Path(tmp) / "dataset"
        dataset = create_lerobot_dataset(
            root=root,
            repo_id=repo_id,
            config=CollectionConfig(fps=fps, image_shapes_chw=image_shapes_from_observation(observation)),
        )
        dataset.finalize()
    print("lerobot_writer_interface=OK")


def image_shapes_from_observation(observation: dict[str, Any]) -> dict[str, tuple[int, int, int]]:
    return {key: tuple(int(dim) for dim in image_to_chw_uint8(observation[key]).shape) for key in IMAGE_KEYS}


def assert_observation_health(observation: dict[str, Any], *, threshold: float) -> None:
    validate_state(observation[STATE_KEY])
    for key in IMAGE_KEYS:
        image = np.asarray(observation[key])
        if image.ndim != 3 or image.shape[-1] != 3:
            raise ValueError(f"{key} must be HWC RGB image, got shape={image.shape}")
        if image.size == 0:
            raise ValueError(f"{key} is empty")
        mean = float(image.mean())
        print(f"{key}_shape={tuple(image.shape)} mean={mean:.2f}")
        if mean <= threshold:
            raise ValueError(f"{key} looks black: mean={mean:.2f} <= threshold={threshold}")


def resolve_start_state(text: str, observation: dict[str, Any]) -> tuple[tuple[float, ...], str]:
    value = text.strip().lower()
    if value in ("", "verified", "standard", "piper"):
        return validate_state(VERIFIED_START_QPOS), "VERIFIED_START_QPOS"
    if value in ("current", "precheck"):
        return validate_state(observation[STATE_KEY]), "current precheck qpos"
    return validate_state(float(part.strip()) for part in text.split(",")), "explicit --start-state"


def print_ready_guard_precheck(
    observation: dict[str, Any],
    *,
    args: argparse.Namespace,
    target: tuple[float, ...],
) -> None:
    current = validate_state(observation[STATE_KEY])
    ok, diffs = start_pose_ok(
        current,
        target,
        mode=args.start_guard_mode,
        joint_tol=args.start_state_tol_rad,
        gripper_tol=args.start_gripper_tol_m,
    )
    print("\nstartup_ready_guard:")
    print("  recording_requires_ready: YES")
    print(f"  mode: {args.start_guard_mode}")
    print(f"  status: {'OK' if ok else 'WAIT'}")
    print(f"  current_qpos: {round_list(list(current), digits=4)}")
    print(f"  target_ready_qpos: {round_list(list(target), digits=4)}")
    print(f"  abs_diff: {round_list(list(diffs), digits=4)}")
    print(f"  max_joint_diff: {max(diffs[:6]):.6f}")
    print(f"  gripper_current_m: {current[6]:.6f}")
    if args.start_guard_mode == "zone":
        print(f"  zone_arm_tolerance_rad: {list(START_GUARD_ZONE_ARM_TOLERANCE_RAD)}")
        print(f"  zone_gripper_open_min_m: {START_GUARD_GRIPPER_OPEN_MIN_M}")
    else:
        print(f"  strict_joint_tol_rad: {args.start_state_tol_rad}")
        print(f"  strict_gripper_tol_m: {args.start_gripper_tol_m}")
    if not ok:
        print("  note: 正式采集时会停在 start_guard=waiting_for_READY，直到你手动回到 READY。")


def prompt_before_episode(*, episode_number: int, total: int, target: str, task: str, attempt: int) -> str:
    print("\n" + "-" * 78)
    print(f"Episode: {episode_number:03d} / {total}")
    print(f"Attempt: {attempt}")
    print(f"Target: {target.upper()}")
    print(f"Task: {task}")
    print("")
    print("请重新随机摆放蓝绿物体，并确认：")
    print("[ ] 两个物体同时出现在 global camera 中")
    print("[ ] 两物体没有靠得过近")
    print("[ ] 两物体都在安全方框内")
    print("[ ] 盒子位置未移动")
    print("[ ] 急停可用")
    print("")
    print("随机摆放提醒：每条 episode 都重新摆放；不要固定蓝左绿右；覆盖左/右/近/远。")
    print("按 Enter 后会先检测 Piper 是否在 READY 起始位姿；未通过不会开始录制。")
    while True:
        text = input("Enter: 开始 | s: 跳过这一轮准备 | q: 安全退出: ").strip().lower()
        if text == "":
            return "record"
        if text in ("s", "q"):
            return text


def collect_placement_metadata(enabled: bool) -> dict[str, str]:
    if not enabled:
        return {}
    print("placement_metadata: blank is allowed")
    return {
        "blue_side": prompt_choice("blue_side", ("left", "right", "center")),
        "green_side": prompt_choice("green_side", ("left", "right", "center")),
        "blue_depth": prompt_choice("blue_depth", ("near", "middle", "far")),
        "green_depth": prompt_choice("green_depth", ("near", "middle", "far")),
    }


def prompt_choice(name: str, choices: tuple[str, ...]) -> str:
    allowed = set(choices)
    while True:
        value = input(f"{name} [{'/'.join(choices)}/blank]: ").strip().lower()
        if value == "" or value in allowed:
            return value
        print(f"invalid {name}: {value!r}")


def wait_until_start_pose(
    adapter: PiperSmolVLAAdapter,
    *,
    args: argparse.Namespace,
    task: str,
    target: tuple[float, ...],
    preview: PreviewWindow,
) -> None:
    print("start_guard=waiting_for_READY")
    print("如果不在 READY，请用当前项目已有安全方式回到 READY；本脚本不会自动 reset/enable。")
    last_print = 0.0
    attempts = 0
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
        now = time.monotonic()
        if attempts == 0 or now - last_print > 1.0:
            print(
                f"start_guard_ok={ok} max_joint_diff={max(diffs[:6]):.6f} "
                f"gripper_diff={diffs[6]:.6f} current={[round(v, 4) for v in current]}"
            )
            if args.start_guard_mode == "zone":
                print(f"zone_gripper_open_min_m={START_GUARD_GRIPPER_OPEN_MIN_M}")
            last_print = now
        attempts += 1
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
    elif mode == "strict":
        ok = all(diff <= joint_tol for diff in diffs[:6]) and diffs[6] <= gripper_tol
    else:
        raise ValueError(f"unknown start guard mode: {mode}")
    return ok, diffs


def wait_for_record_start(
    adapter: PiperSmolVLAAdapter,
    *,
    task: str,
    preview: PreviewWindow,
) -> None:
    while True:
        obs = adapter.read_observation(task=task)
        key = preview.show(obs, status=f"READY: Press ENTER to record\nTASK: {task}")
        if key in (ord(" "), 10, 13):
            return
        if key in (ord("q"), ord("Q"), 27):
            raise KeyboardInterrupt("aborted from preview")
        if stdin_has_line():
            sys.stdin.readline()
            return
        time.sleep(0.05)


def record_one_episode(
    adapter: PiperSmolVLAAdapter,
    *,
    task: str,
    args: argparse.Namespace,
    preview: PreviewWindow,
) -> EpisodeRecord:
    print("RECORDING — press ENTER or SPACE to stop")
    previous = adapter.read_observation(task=task)
    frames: list[dict[str, Any]] = []
    timestamps: list[float] = []
    period = 1.0 / max(1, args.fps)
    preview_every = max(1, args.fps // 10)  # update preview at ~10 Hz
    start = time.monotonic()
    deadline = start + args.max_episode_seconds
    timed_out = True
    frame_idx = 0
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
        timestamps.append(t0 - start)
        frame_idx += 1
        if frame_idx % preview_every == 0:
            key = preview.show(current, status=f"RECORDING | TASK: {task}", frame_count=len(frames))
            if key in (ord("q"), ord("Q"), 27):
                raise KeyboardInterrupt("aborted from preview")
            if key in (ord(" "), 10, 13):
                timed_out = False
                break
        if stdin_has_line():
            sys.stdin.readline()
            timed_out = False
            break
        previous = current
        elapsed = time.monotonic() - t0
        if elapsed < period:
            time.sleep(period - elapsed)
    return EpisodeRecord(frames=frames, timestamps=timestamps, timed_out=timed_out)


def validate_episode(
    record: EpisodeRecord,
    *,
    task: str,
    target: str,
    episode_index: int,
    fps: int,
    args: argparse.Namespace,
) -> EpisodeQuality:
    frames = record.frames
    failures: list[str] = []
    warnings: list[str] = []
    if not frames:
        failures.append("empty episode")
        return EpisodeQuality(
            episode_index=episode_index,
            target=target,
            task=task,
            frame_count=0,
            duration_sec=0.0,
            actual_fps=0.0,
            global_frame_failures=1,
            wrist_frame_failures=1,
            black_frame_count=0,
            finite_ok=False,
            state_shape_ok=False,
            action_shape_ok=False,
            duplicate_timestamp_count=0,
            max_action_jump=0.0,
            big_jump_count=0,
            action_min=[],
            action_max=[],
            gripper_min=0.0,
            gripper_max=0.0,
            has_close=False,
            has_release=False,
            has_open_close_release=False,
            too_short=True,
            timed_out=record.timed_out,
            status="FAIL",
            warnings=warnings,
            failures=failures,
        )

    states: list[np.ndarray] = []
    actions: list[np.ndarray] = []
    global_failures = 0
    wrist_failures = 0
    black_frames = 0
    state_shape_ok = True
    action_shape_ok = True

    for index, frame in enumerate(frames):
        if frame.get("task") != task:
            failures.append(f"frame {index} task mismatch: {frame.get('task')!r}")
        try:
            state = np.asarray(validate_state(frame[STATE_KEY]), dtype=np.float32)
            states.append(state)
            state_shape_ok = state_shape_ok and state.shape == (7,)
        except Exception as exc:  # noqa: BLE001
            failures.append(f"frame {index} state invalid: {type(exc).__name__}: {exc}")
            state_shape_ok = False
        try:
            action = np.asarray(validate_action(frame[ACTION_KEY]), dtype=np.float32)
            actions.append(action)
            action_shape_ok = action_shape_ok and action.shape == (7,)
        except Exception as exc:  # noqa: BLE001
            failures.append(f"frame {index} action invalid: {type(exc).__name__}: {exc}")
            action_shape_ok = False

        for key in IMAGE_KEYS:
            if key not in frame:
                if key == GLOBAL_IMAGE_KEY:
                    global_failures += 1
                else:
                    wrist_failures += 1
                continue
            image = np.asarray(frame[key])
            if image.size == 0 or float(image.mean()) <= args.black_frame_threshold:
                black_frames += 1

    states_arr = np.asarray(states, dtype=np.float32) if states else np.empty((0, 7), dtype=np.float32)
    actions_arr = np.asarray(actions, dtype=np.float32) if actions else np.empty((0, 7), dtype=np.float32)
    finite_ok = bool(np.isfinite(states_arr).all() and np.isfinite(actions_arr).all())
    if not finite_ok:
        failures.append("state/action contains NaN or Inf")

    deltas = np.abs(np.diff(actions_arr, axis=0)) if len(actions_arr) > 1 else np.zeros((0, 7), dtype=np.float32)
    per_step_max = deltas.max(axis=1) if len(deltas) else np.array([], dtype=np.float32)
    max_jump = float(per_step_max.max()) if len(per_step_max) else 0.0
    big_jump_count = int((per_step_max > args.max_frame_jump).sum()) if len(per_step_max) else 0
    if big_jump_count:
        failures.append(f"action jump above --max-frame-jump: count={big_jump_count} max={max_jump:.4f}")

    gripper = actions_arr[:, 6] if len(actions_arr) else np.array([], dtype=np.float32)
    has_close = bool((gripper < 0.07).any()) if len(gripper) else False
    has_release = bool((gripper > 0.09).any()) if len(gripper) else False
    has_open_close_release = detect_open_close_release(gripper)
    if not has_close:
        warnings.append("no gripper close detected")
    if not has_release:
        warnings.append("no gripper release/open detected")
    if not has_open_close_release:
        warnings.append("no open -> close -> release sequence detected")

    frame_count = len(frames)
    duration_sec = (
        max(record.timestamps[-1] - record.timestamps[0] + (1.0 / max(1, fps)), 0.0)
        if len(record.timestamps) > 1
        else frame_count / max(1, fps)
    )
    actual_fps = frame_count / duration_sec if duration_sec > 0 else 0.0
    duplicate_timestamps = count_duplicate_timestamps(record.timestamps)
    too_short = frame_count < args.min_frames
    if too_short:
        failures.append(f"episode too short: frames={frame_count} min={args.min_frames}")
    if record.timed_out:
        warnings.append("episode reached --max-episode-seconds")
    if duplicate_timestamps:
        warnings.append(f"duplicate timestamps: {duplicate_timestamps}")
    if global_failures or wrist_failures:
        failures.append(f"camera frame failures global={global_failures} wrist={wrist_failures}")
    if black_frames:
        failures.append(f"black frames/images detected: {black_frames}")
    if not state_shape_ok:
        failures.append("state shape is not consistently (7,)")
    if not action_shape_ok:
        failures.append("action shape is not consistently (7,)")

    status = "FAIL" if failures else "WARNING" if warnings else "PASS"
    action_min = actions_arr.min(axis=0).astype(float).tolist() if len(actions_arr) else []
    action_max = actions_arr.max(axis=0).astype(float).tolist() if len(actions_arr) else []

    return EpisodeQuality(
        episode_index=episode_index,
        target=target,
        task=task,
        frame_count=frame_count,
        duration_sec=duration_sec,
        actual_fps=actual_fps,
        global_frame_failures=global_failures,
        wrist_frame_failures=wrist_failures,
        black_frame_count=black_frames,
        finite_ok=finite_ok,
        state_shape_ok=state_shape_ok,
        action_shape_ok=action_shape_ok,
        duplicate_timestamp_count=duplicate_timestamps,
        max_action_jump=max_jump,
        big_jump_count=big_jump_count,
        action_min=action_min,
        action_max=action_max,
        gripper_min=float(gripper.min()) if len(gripper) else 0.0,
        gripper_max=float(gripper.max()) if len(gripper) else 0.0,
        has_close=has_close,
        has_release=has_release,
        has_open_close_release=has_open_close_release,
        too_short=too_short,
        timed_out=record.timed_out,
        status=status,
        warnings=warnings,
        failures=failures,
    )


def print_episode_quality(q: EpisodeQuality) -> None:
    print("\nepisode_quality:")
    print(f"  episode_index={q.episode_index} target={q.target} status={q.status}")
    print(f"  task={q.task!r}")
    print(f"  frames={q.frame_count} duration_sec={q.duration_sec:.2f} actual_fps={q.actual_fps:.2f}")
    print(f"  global_frame_failures={q.global_frame_failures} wrist_frame_failures={q.wrist_frame_failures}")
    print(f"  black_frame_count={q.black_frame_count}")
    print(f"  finite_ok={q.finite_ok} state_shape_ok={q.state_shape_ok} action_shape_ok={q.action_shape_ok}")
    print(f"  duplicate_timestamp_count={q.duplicate_timestamp_count}")
    print(f"  max_action_jump={q.max_action_jump:.4f} big_jump_count={q.big_jump_count}")
    print(f"  action_min={round_list(q.action_min)}")
    print(f"  action_max={round_list(q.action_max)}")
    print(f"  gripper_min={q.gripper_min:.4f} gripper_max={q.gripper_max:.4f}")
    print(f"  open_close_release={q.has_open_close_release} close={q.has_close} release={q.has_release}")
    print(f"  too_short={q.too_short} timed_out={q.timed_out}")
    for warning in q.warnings:
        print(f"  WARNING: {warning}")
    for failure in q.failures:
        print(f"  FAIL: {failure}")


def print_manual_quality_reminder() -> None:
    print("manual_check_required:")
    print("  - 是否抓错颜色")
    print("  - 是否先靠近错误目标后再人工修正")
    print("  - 是否抓偏后反复补救")
    print("  - 是否碰倒干扰物体")
    print("  - 是否机械臂明显卡顿或长时间静止")


def ask_save_decision(quality: EpisodeQuality) -> str:
    suggestion = "save" if quality.status == "PASS" else "discard recommended"
    while True:
        text = input(
            f"Decision ({quality.status}, {suggestion}) "
            "[y save / r rerecord / d discard-new-scene / p preview-summary / q quit]: "
        ).strip().lower()
        if text in ("y", "r", "d", "p", "q"):
            return text
        print("please enter y/r/d/p/q")


def save_episode_checked(dataset: Any, frames: list[dict[str, Any]]) -> None:
    # All dataset writes are centralized here. This function does not write robot actions.
    write_episode(dataset, frames)


def save_episode_sidecar(
    root: Path,
    *,
    episode_index: int,
    schedule_index: int,
    target: str,
    task: str,
    quality: EpisodeQuality,
    placement: dict[str, str],
    args: argparse.Namespace,
) -> None:
    sidecar = root / "meta" / "two_object_language_episode_metadata"
    sidecar.mkdir(parents=True, exist_ok=True)
    data = {
        "episode_index": episode_index,
        "schedule_index": schedule_index,
        "target": target,
        "task": task,
        "timestamp": datetime.now().astimezone().isoformat(timespec="seconds"),
        "camera_keys": list(IMAGE_KEYS),
        "state_dim": 7,
        "action_dim": 7,
        "source": "two-object language read-only operator demonstration",
        "action_semantics_note": "observation.state=previous qpos, action=current qpos",
        "quality": asdict(quality),
        "placement": placement,
        "fps": args.fps,
        "camera_fps": args.camera_fps,
        "real_actions_sent": False,
        "policy_actions_sent": False,
        "robot_write_enabled": False,
    }
    (sidecar / f"episode_{episode_index:06d}.json").write_text(
        json.dumps(data, indent=2) + "\n", encoding="utf-8"
    )


def preview_episode_summary(record: EpisodeRecord, *, quality: EpisodeQuality, preview: PreviewWindow) -> None:
    print_episode_quality(quality)
    if not preview.enabled or not record.frames:
        return
    print("previewing first and last frame; press any key in preview to continue")
    for label, frame in (("first", record.frames[0]), ("last", record.frames[-1])):
        deadline = time.monotonic() + 1.0
        while time.monotonic() < deadline:
            key = preview.show_frame(frame, status=f"SUMMARY {label} | {quality.target.upper()} | {quality.status}")
            if key >= 0:
                return
            time.sleep(0.03)


def send_safe_action(adapter: PiperSmolVLAAdapter, action: Any, args: argparse.Namespace) -> None:
    """Central robot-write gate. This recorder intentionally never enables it."""
    if args.no_robot_write:
        raise RuntimeError("robot writes are disabled by --no-robot-write")
    raise RuntimeError("robot writes are not supported by record_two_object_language_random.py")


def safe_shutdown(*, preview: PreviewWindow, cleanup: Any, dataset: Any, dataset_finalized: bool) -> None:
    preview.close()
    if dataset is not None and not dataset_finalized:
        with suppress(Exception):
            dataset.finalize()
            print("dataset_finalized_in_shutdown=True")
    cleanup()


def validate_dataset(
    root: Path,
    *,
    repo_id: str,
    args: argparse.Namespace,
    schedule_path: Path,
) -> int:
    if not root.is_dir():
        raise SystemExit(f"dataset root not found: {root}")
    try:
        import lerobot
        from lerobot.datasets.lerobot_dataset import CODEBASE_VERSION, LeRobotDataset
    except Exception as exc:  # noqa: BLE001
        raise SystemExit(f"cannot import official LeRobotDataset: {exc}") from exc

    dataset = LeRobotDataset(repo_id, root=str(root), tolerance_s=0.5)
    errors: list[str] = []
    warnings: list[str] = []
    rows: list[dict[str, Any]] = []
    task_index_to_text = read_task_table(dataset, errors)
    features = dataset.meta.features
    schedule_targets: list[str] = []

    print(f"validate_dataset={root}")
    print(f"repo_id={repo_id}")
    print(f"lerobot_version={getattr(lerobot, '__version__', 'unknown')}")
    print(f"dataset_format_version={CODEBASE_VERSION}")
    print(f"episodes={dataset.num_episodes} frames={dataset.num_frames} fps={dataset.fps}")

    errors.extend(validate_features(features, dataset))
    if schedule_path.exists():
        schedule_targets = targets_from_schedule_json(json.loads(schedule_path.read_text(encoding="utf-8")))
    else:
        errors.append(f"missing schedule file: {schedule_path}")

    episode_targets: list[str] = []
    for ep_idx in range(dataset.num_episodes):
        row, row_errors, row_warnings = validate_dataset_episode(
            dataset,
            ep_idx=ep_idx,
            task_index_to_text=task_index_to_text,
            args=args,
        )
        rows.append(row)
        errors.extend(row_errors)
        warnings.extend(row_warnings)
        target = row.get("target", "")
        if target in (BLUE_TARGET, GREEN_TARGET):
            episode_targets.append(target)

    counts = Counter(episode_targets)
    if counts[BLUE_TARGET] != args.num_blue or counts[GREEN_TARGET] != args.num_green:
        errors.append(
            f"target counts mismatch: blue={counts[BLUE_TARGET]} green={counts[GREEN_TARGET]} "
            f"expected blue={args.num_blue} green={args.num_green}"
        )
    if set(task_index_to_text.values()) != set(EXPECTED_TASKS):
        errors.append(f"tasks must be exactly the two fixed prompts, got={sorted(task_index_to_text.values())}")
    if schedule_targets:
        if episode_targets != schedule_targets[: len(episode_targets)]:
            errors.append("saved episode targets do not match schedule prefix")
        if len(episode_targets) != len(schedule_targets):
            errors.append(
                f"schedule incomplete: saved={len(episode_targets)} schedule_total={len(schedule_targets)}"
            )

    write_validation_reports(root, rows=rows, errors=errors, warnings=warnings, counts=counts)
    print_validation_summary(rows=rows, counts=counts, errors=errors, warnings=warnings)
    return 1 if errors else 0


def read_task_table(dataset: Any, errors: list[str]) -> dict[int, str]:
    task_index_to_text: dict[int, str] = {}
    print("\ntasks:")
    for task_text, row in dataset.meta.tasks.iterrows():
        task_index = int(row["task_index"])
        task_index_to_text[task_index] = str(task_text)
        print(f"  {task_index}: {task_text}")
    if not task_index_to_text:
        errors.append("meta/tasks.parquet is empty")
    return task_index_to_text


def validate_features(features: dict[str, Any], dataset: Any) -> list[str]:
    errors: list[str] = []
    print("\nfeatures:")
    for key, feature in features.items():
        print(f"  {key}: dtype={feature.get('dtype')} shape={feature.get('shape')} names={feature.get('names')}")
    for key in (STATE_KEY, ACTION_KEY):
        if key not in features:
            errors.append(f"missing feature: {key}")
            continue
        if tuple(features[key].get("shape", ())) != (7,):
            errors.append(f"{key} shape must be (7,), got {features[key].get('shape')}")
    for key in IMAGE_KEYS:
        if key not in features:
            errors.append(f"missing camera feature: {key}")
            continue
        if features[key].get("dtype") not in ("video", "image"):
            errors.append(f"{key} dtype must be video/image, got {features[key].get('dtype')}")
    for renamed in ("observation.images.image", "observation.images.image2", "observation.image", "observation.image2"):
        if renamed in features:
            errors.append(f"raw dataset unexpectedly contains renamed policy key: {renamed}")
    camera_keys = tuple(getattr(dataset.meta, "camera_keys", ()) or ())
    for key in IMAGE_KEYS:
        if camera_keys and key not in camera_keys:
            errors.append(f"dataset.meta.camera_keys missing {key}: {camera_keys}")
    return errors


def validate_dataset_episode(
    dataset: Any,
    *,
    ep_idx: int,
    task_index_to_text: dict[int, str],
    args: argparse.Namespace,
) -> tuple[dict[str, Any], list[str], list[str]]:
    errors: list[str] = []
    warnings: list[str] = []
    ep = dataset.meta.episodes[ep_idx]
    start = int(ep["dataset_from_index"])
    end = int(ep["dataset_to_index"])
    length = int(ep["length"])
    ep_tasks = [str(task) for task in ep.get("tasks", [])]
    if end - start != length:
        errors.append(f"episode {ep_idx}: dataset_to-from={end - start} != length={length}")
    if len(set(ep_tasks)) != 1:
        errors.append(f"episode {ep_idx}: expected exactly one episode-level task, got {ep_tasks}")
    task = ep_tasks[0] if ep_tasks else ""
    target = TARGET_BY_TASK.get(task, "")
    if not target:
        errors.append(f"episode {ep_idx}: unexpected task {task!r}")

    frame_rows = dataset.hf_dataset.select(range(start, end))
    task_indices = [to_int(value) for value in frame_rows["task_index"]]
    frame_tasks = {task_index_to_text.get(index, f"<missing:{index}>") for index in task_indices}
    if frame_tasks != set(ep_tasks):
        errors.append(f"episode {ep_idx}: frame tasks {sorted(frame_tasks)} != episode tasks {ep_tasks}")

    states = values_to_2d_float(frame_rows[STATE_KEY])
    actions = values_to_2d_float(frame_rows[ACTION_KEY])
    if states.shape[1:] != (7,):
        errors.append(f"episode {ep_idx}: {STATE_KEY} shape {states.shape}")
    if actions.shape[1:] != (7,):
        errors.append(f"episode {ep_idx}: {ACTION_KEY} shape {actions.shape}")
    finite_ok = bool(np.isfinite(states).all() and np.isfinite(actions).all())
    if not finite_ok:
        errors.append(f"episode {ep_idx}: state/action contains NaN or Inf")

    timestamps = [float(to_float(value)) for value in frame_rows["timestamp"]]
    duplicate_timestamps = count_duplicate_timestamps(timestamps)
    if duplicate_timestamps:
        errors.append(f"episode {ep_idx}: duplicate timestamps={duplicate_timestamps}")
    fps_ok = validate_timestamp_fps(timestamps, fps=dataset.fps, tolerance=0.03)
    if not fps_ok:
        warnings.append(f"episode {ep_idx}: timestamp spacing deviates from fps={dataset.fps}")

    deltas = np.abs(np.diff(actions, axis=0)) if len(actions) > 1 else np.zeros((0, 7), dtype=np.float32)
    per_step_max = deltas.max(axis=1) if len(deltas) else np.array([], dtype=np.float32)
    max_jump = float(per_step_max.max()) if len(per_step_max) else 0.0
    big_jump_count = int((per_step_max > args.max_frame_jump).sum()) if len(per_step_max) else 0
    if big_jump_count:
        errors.append(f"episode {ep_idx}: action big jumps count={big_jump_count} max={max_jump:.4f}")

    gripper = actions[:, 6] if len(actions) else np.array([], dtype=np.float32)
    has_ocr = detect_open_close_release(gripper)
    if not has_ocr:
        warnings.append(f"episode {ep_idx}: no open-close-release sequence")
    too_short = length < args.min_frames
    if too_short:
        errors.append(f"episode {ep_idx}: too short length={length} min={args.min_frames}")

    missing_videos = missing_video_files(dataset, ep_idx)
    errors.extend(f"episode {ep_idx}: missing video {path}" for path in missing_videos)
    black_samples = count_black_decoded_samples(dataset, ep_idx, start, end, threshold=args.black_frame_threshold)
    if black_samples:
        errors.append(f"episode {ep_idx}: black decoded sample images={black_samples}")

    duration = (timestamps[-1] - timestamps[0] + 1.0 / max(1, dataset.fps)) if len(timestamps) > 1 else 0.0
    row = {
        "episode_index": ep_idx,
        "target": target,
        "task": task,
        "frame_count": length,
        "duration_sec": round(duration, 4),
        "actual_fps": round(length / duration, 4) if duration > 0 else 0.0,
        "finite_ok": finite_ok,
        "duplicate_timestamps": duplicate_timestamps,
        "fps_ok": fps_ok,
        "max_action_jump": round(max_jump, 6),
        "big_jump_count": big_jump_count,
        "gripper_min": round(float(gripper.min()), 6) if len(gripper) else 0.0,
        "gripper_max": round(float(gripper.max()), 6) if len(gripper) else 0.0,
        "open_close_release": has_ocr,
        "too_short": too_short,
        "black_sample_images": black_samples,
        "video_files_ok": not missing_videos,
        "status": "FAIL" if errors else "WARNING" if warnings else "PASS",
    }
    print(
        f"  ep={ep_idx:03d} target={target or '?':5s} frames={length:04d} "
        f"fps_ok={fps_ok} ocr={has_ocr} jump={max_jump:.4f} status={row['status']}"
    )
    return row, errors, warnings


def missing_video_files(dataset: Any, ep_idx: int) -> list[str]:
    missing: list[str] = []
    for key in IMAGE_KEYS:
        try:
            rel = dataset.meta.get_video_file_path(ep_idx, key)
        except Exception:  # noqa: BLE001
            continue
        path = dataset.root / rel
        if not path.exists():
            missing.append(str(rel))
    return missing


def count_black_decoded_samples(dataset: Any, ep_idx: int, start: int, end: int, *, threshold: float) -> int:
    del ep_idx
    if end <= start:
        return 0
    indices = sorted({start, (start + end - 1) // 2, end - 1})
    black = 0
    for index in indices:
        try:
            item = dataset[index]
        except Exception:
            black += len(IMAGE_KEYS)
            continue
        for key in IMAGE_KEYS:
            image = np.asarray(item[key])
            if image.size == 0 or float(image.mean()) <= threshold:
                black += 1
    return black


def write_validation_reports(
    root: Path,
    *,
    rows: list[dict[str, Any]],
    errors: list[str],
    warnings: list[str],
    counts: Counter[str],
) -> None:
    meta = root / "meta"
    meta.mkdir(parents=True, exist_ok=True)
    json_path = meta / "collection_validation_report.json"
    csv_path = meta / "collection_validation_report.csv"
    report = {
        "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "counts": dict(counts),
        "errors": errors,
        "warnings": warnings,
        "episodes": rows,
    }
    json_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    if rows:
        with csv_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
    else:
        csv_path.write_text("", encoding="utf-8")
    print(f"validation_report_json={json_path}")
    print(f"validation_report_csv={csv_path}")


def print_validation_summary(
    *,
    rows: list[dict[str, Any]],
    counts: Counter[str],
    errors: list[str],
    warnings: list[str],
) -> None:
    print("\nvalidation_summary:")
    print(f"  total_episodes={len(rows)}")
    print(f"  blue={counts[BLUE_TARGET]} green={counts[GREEN_TARGET]}")
    if rows:
        lengths = [int(row["frame_count"]) for row in rows]
        durations = [float(row["duration_sec"]) for row in rows]
        print(f"  frame_count_range={min(lengths)}..{max(lengths)}")
        print(f"  duration_sec_range={min(durations):.2f}..{max(durations):.2f}")
    print(f"  warnings={len(warnings)}")
    for warning in warnings[:30]:
        print(f"    WARNING: {warning}")
    if len(warnings) > 30:
        print(f"    ... {len(warnings) - 30} more warnings")
    print(f"  errors={len(errors)}")
    for error in errors[:50]:
        print(f"    ERROR: {error}")
    if len(errors) > 50:
        print(f"    ... {len(errors) - 50} more errors")
    print("TWO_OBJECT_LANGUAGE_DATASET_READY=" + ("NO" if errors else "YES"))


def detect_open_close_release(gripper: np.ndarray) -> bool:
    if len(gripper) == 0:
        return False
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


def count_duplicate_timestamps(timestamps: list[float], *, eps: float = 1e-9) -> int:
    if len(timestamps) < 2:
        return 0
    diffs = np.diff(np.asarray(timestamps, dtype=np.float64))
    return int((np.abs(diffs) <= eps).sum())


def validate_timestamp_fps(timestamps: list[float], *, fps: int, tolerance: float) -> bool:
    if len(timestamps) < 3:
        return True
    expected = 1.0 / max(1, fps)
    diffs = np.diff(np.asarray(timestamps, dtype=np.float64))
    return bool(np.all(np.abs(diffs - expected) <= tolerance))


def values_to_2d_float(values: Any) -> np.ndarray:
    rows = [to_numpy(value).astype(np.float32).reshape(-1) for value in values]
    if not rows:
        return np.empty((0, 0), dtype=np.float32)
    return np.stack(rows, axis=0)


def to_numpy(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    elif hasattr(value, "cpu") and hasattr(value, "numpy"):
        value = value.cpu().numpy()
    elif hasattr(value, "numpy"):
        value = value.numpy()
    return np.asarray(value)


def to_int(value: Any) -> int:
    arr = to_numpy(value)
    return int(arr.reshape(-1)[0])


def to_float(value: Any) -> float:
    arr = to_numpy(value)
    return float(arr.reshape(-1)[0])


def image_to_hwc_uint8(image: Any) -> np.ndarray:
    arr = np.asarray(image)
    if arr.ndim != 3:
        raise ValueError(f"image must be 3D, got shape {arr.shape}")
    if arr.shape[-1] == 3:
        hwc = arr
    elif arr.shape[0] == 3:
        hwc = np.moveaxis(arr, 0, -1)
    else:
        raise ValueError(f"image must have 3 channels, got shape {arr.shape}")
    if hwc.dtype != np.uint8:
        hwc = np.clip(hwc, 0, 255).astype(np.uint8)
    return np.ascontiguousarray(hwc)


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


def round_list(values: list[float], digits: int = 4) -> list[float]:
    return [round(float(value), digits) for value in values]


def stdin_has_line() -> bool:
    if not sys.stdin.isatty():
        return False
    ready, _, _ = select.select([sys.stdin], [], [], 0.0)
    return bool(ready)


if __name__ == "__main__":
    raise SystemExit(main())
