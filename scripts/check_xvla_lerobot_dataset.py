#!/usr/bin/env python3
"""只读检查 Piper 数据集是否符合官方 LeRobot / X-VLA 读取路径。

这个脚本故意不做训练、不做格式转换、不改数据。它用官方
``lerobot.datasets.lerobot_dataset.LeRobotDataset`` 加载数据集，并触发
``dataset[i]`` 解码路径，避免为了“能跑”而绕过 LeRobot 数据结构。
"""

from __future__ import annotations

import argparse
import os
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


REQUIRED_PIPER_FEATURES = {
    "observation.state": 7,
    "action": 7,
}
REQUIRED_CAMERA_KEYS = (
    "observation.images.global_rgb",
    "observation.images.wrist_rgb",
)
PIPER_TO_XVLA_RENAME_MAP = {
    "observation.images.global_rgb": "observation.images.image",
    "observation.images.wrist_rgb": "observation.images.image2",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Check a local LeRobot v3 dataset through the official LeRobotDataset reader."
    )
    parser.add_argument("--dataset", required=True, help="Local dataset root")
    parser.add_argument("--repo-id", help="LeRobot repo_id. Defaults to piper/<dataset_dir_name>.")
    parser.add_argument("--expected-episodes", type=int)
    parser.add_argument("--expected-tasks", nargs="*", help="Exact task strings expected in tasks.parquet")
    parser.add_argument("--sample-frames", type=int, default=3, help="Number of frames to decode via dataset[i]")
    parser.add_argument("--check-batch", action="store_true", help="Also build one torch DataLoader batch")
    parser.add_argument("--hf-datasets-cache", help="Writable Hugging Face datasets cache directory")
    return parser.parse_args()


def configure_cache(args: argparse.Namespace) -> None:
    if args.hf_datasets_cache:
        cache = Path(args.hf_datasets_cache).expanduser()
    else:
        cache = Path(os.environ.get("HF_DATASETS_CACHE", ".cache/huggingface/datasets")).expanduser()
    cache.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("HF_DATASETS_CACHE", str(cache))


def import_lerobot_dataset():
    try:
        import lerobot
        from lerobot.datasets.lerobot_dataset import LeRobotDataset
    except Exception as exc:  # noqa: BLE001
        raise SystemExit(
            "Cannot import official LeRobotDataset. Activate the LeRobot environment first, "
            "for example: conda activate piper_smolvla or conda activate lerobot_q. "
            f"Original error: {type(exc).__name__}: {exc}"
        ) from exc
    return lerobot, LeRobotDataset


def main() -> int:
    args = parse_args()
    dataset_root = Path(args.dataset).expanduser().resolve()
    if not dataset_root.is_dir():
        raise SystemExit(f"dataset not found: {dataset_root}")

    configure_cache(args)
    lerobot, LeRobotDataset = import_lerobot_dataset()

    repo_id = args.repo_id or f"piper/{dataset_root.name}"
    print(f"dataset_root={dataset_root}")
    print(f"repo_id={repo_id}")
    print(f"lerobot={Path(lerobot.__file__).resolve()}")
    print(f"HF_DATASETS_CACHE={os.environ.get('HF_DATASETS_CACHE')}")
    print("hardware_access=NO")
    print("training=NO")

    errors: list[str] = []
    warnings: list[str] = []

    try:
        dataset = LeRobotDataset(repo_id, root=str(dataset_root), tolerance_s=0.5)
    except Exception as exc:  # noqa: BLE001
        raise SystemExit(
            "OFFICIAL_LEROBOT_LOAD=FAIL\n"
            f"{type(exc).__name__}: {exc}\n"
            "Do not train this dataset until official LeRobotDataset can load it."
        ) from exc

    print("OFFICIAL_LEROBOT_LOAD=OK")
    print(f"episodes={dataset.num_episodes}")
    print(f"frames={dataset.num_frames}")
    print(f"fps={getattr(dataset.meta, 'fps', '?')}")
    print(f"codebase_version={metadata_value(dataset.meta.info, 'codebase_version', '?')}")

    if args.expected_episodes is not None and dataset.num_episodes != args.expected_episodes:
        errors.append(f"expected {args.expected_episodes} episodes, got {dataset.num_episodes}")

    errors.extend(check_file_layout(dataset_root))
    errors.extend(check_features(dataset))
    task_info = check_tasks(dataset, expected_tasks=args.expected_tasks, errors=errors, warnings=warnings)
    errors.extend(check_episode_tasks(dataset, task_info))
    errors.extend(check_decoded_samples(dataset, args.sample_frames))
    if args.check_batch:
        errors.extend(check_one_batch(dataset))

    print("\nXVLA_RENAME_MAP_EXPECTED:")
    for src, dst in PIPER_TO_XVLA_RENAME_MAP.items():
        print(f"  {src} -> {dst}")
    print("raw_dataset_keys_must_remain_canonical=True")

    print("\nSUMMARY:")
    if warnings:
        print(f"warnings={len(warnings)}")
        for warning in warnings:
            print(f"  WARN: {warning}")
    else:
        print("warnings=0")

    if errors:
        print(f"errors={len(errors)}")
        for error in errors:
            print(f"  ERROR: {error}")
        print("DATASET_XVLA_READY=NO")
        return 1

    print("errors=0")
    print("DATASET_XVLA_READY=YES")
    return 0


def check_file_layout(dataset_root: Path) -> list[str]:
    errors: list[str] = []
    required = [
        dataset_root / "meta" / "info.json",
        dataset_root / "meta" / "stats.json",
        dataset_root / "meta" / "tasks.parquet",
        dataset_root / "meta" / "episodes",
        dataset_root / "data",
    ]
    for path in required:
        if not path.exists():
            errors.append(f"missing LeRobot v3 path: {path.relative_to(dataset_root)}")
    return errors


def check_features(dataset: Any) -> list[str]:
    errors: list[str] = []
    features = dataset.meta.features
    print("\nFEATURES:")
    for key, feature in features.items():
        print(
            f"  {key}: dtype={feature.get('dtype', '?')} "
            f"shape={feature.get('shape', '?')} names={feature.get('names', None)}"
        )

    for key, dim in REQUIRED_PIPER_FEATURES.items():
        feature = features.get(key)
        if feature is None:
            errors.append(f"missing required feature: {key}")
            continue
        shape = tuple(feature.get("shape", ()))
        if shape != (dim,):
            errors.append(f"{key} shape must be ({dim},), got {shape}")
        dtype = feature.get("dtype")
        if dtype not in ("float32", "float64"):
            errors.append(f"{key} dtype should be float32/float64, got {dtype}")

    for key in REQUIRED_CAMERA_KEYS:
        feature = features.get(key)
        if feature is None:
            errors.append(f"missing camera feature: {key}")
            continue
        if feature.get("dtype") not in ("video", "image"):
            errors.append(f"{key} dtype must be video/image, got {feature.get('dtype')}")
        shape = tuple(feature.get("shape", ()))
        if len(shape) != 3 or 3 not in (shape[0], shape[-1]):
            errors.append(f"{key} must be RGB CHW or HWC, got shape={shape}")

    for renamed_key in PIPER_TO_XVLA_RENAME_MAP.values():
        if renamed_key in features:
            errors.append(
                f"raw dataset unexpectedly contains XVLA renamed key {renamed_key}; "
                "keep raw dataset canonical and use LeRobot rename_map at training time"
            )

    camera_keys = tuple(getattr(dataset.meta, "camera_keys", ()) or ())
    for key in REQUIRED_CAMERA_KEYS:
        if camera_keys and key not in camera_keys:
            errors.append(f"dataset.meta.camera_keys missing {key}: {camera_keys}")

    return errors


def check_tasks(
    dataset: Any,
    *,
    expected_tasks: list[str] | None,
    errors: list[str],
    warnings: list[str],
) -> dict[int, str]:
    tasks_df = dataset.meta.tasks
    task_index_to_text: dict[int, str] = {}
    print("\nTASKS:")
    for task_text, row in tasks_df.iterrows():
        task_index = int(row["task_index"])
        task_index_to_text[task_index] = str(task_text)
        print(f"  {task_index}: {task_text}")

    if not task_index_to_text:
        errors.append("tasks.parquet is empty")

    if expected_tasks is not None:
        expected = set(expected_tasks)
        actual = set(task_index_to_text.values())
        if actual != expected:
            errors.append(f"tasks mismatch: expected={sorted(expected)} actual={sorted(actual)}")

    if len(set(task_index_to_text)) != len(task_index_to_text):
        warnings.append("duplicate task_index values detected")

    return task_index_to_text


def check_episode_tasks(dataset: Any, task_index_to_text: dict[int, str]) -> list[str]:
    errors: list[str] = []
    episode_task_counts: Counter[str] = Counter()
    frame_task_counts: Counter[str] = Counter()
    frame_counts_by_episode: list[int] = []

    print("\nEPISODES:")
    hf_dataset = dataset.hf_dataset
    for ep_idx in range(dataset.num_episodes):
        ep = dataset.meta.episodes[ep_idx]
        ep_tasks = [str(task) for task in ep.get("tasks", [])]
        start = int(ep["dataset_from_index"])
        end = int(ep["dataset_to_index"])
        length = int(ep["length"])
        frame_counts_by_episode.append(length)

        if end - start != length:
            errors.append(f"episode {ep_idx}: dataset_to-from={end - start} != length={length}")
        if not ep_tasks:
            errors.append(f"episode {ep_idx}: no episode-level task")
        if len(set(ep_tasks)) != 1:
            errors.append(f"episode {ep_idx}: expected one task per episode, got {ep_tasks}")

        row_slice = hf_dataset.select(range(start, end))
        task_indices = [int(value) for value in row_slice["task_index"]]
        frame_tasks = {task_index_to_text.get(idx, f"<missing:{idx}>") for idx in task_indices}
        if set(ep_tasks) != frame_tasks:
            errors.append(
                f"episode {ep_idx}: episode tasks {ep_tasks} != frame-level tasks {sorted(frame_tasks)}"
            )

        for task in ep_tasks:
            episode_task_counts[task] += 1
        for task in frame_tasks:
            frame_task_counts[task] += length

        print(
            f"  ep={ep_idx:03d} frames={length:04d} "
            f"tasks={ep_tasks} frame_tasks={sorted(frame_tasks)}"
        )

    print("\nTASK_DISTRIBUTION_BY_EPISODE:")
    for task, count in sorted(episode_task_counts.items()):
        print(f"  {count:4d}  {task}")

    print("\nTASK_DISTRIBUTION_BY_FRAME:")
    for task, count in sorted(frame_task_counts.items()):
        print(f"  {count:4d}  {task}")

    if frame_counts_by_episode:
        print(
            "\nFRAME_COUNT_RANGE="
            f"{min(frame_counts_by_episode)}..{max(frame_counts_by_episode)}"
        )

    return errors


def check_decoded_samples(dataset: Any, sample_frames: int) -> list[str]:
    errors: list[str] = []
    indices = sample_indices(dataset.num_frames, sample_frames)
    print("\nDECODED_SAMPLES:")
    for index in indices:
        try:
            item = dataset[index]
        except Exception as exc:  # noqa: BLE001
            errors.append(f"dataset[{index}] decode failed: {type(exc).__name__}: {exc}")
            continue

        print(f"  index={index} keys={sorted(item.keys())}")
        for key, dim in REQUIRED_PIPER_FEATURES.items():
            value = item.get(key)
            shape = tuple(getattr(value, "shape", ()))
            dtype = getattr(value, "dtype", None)
            print(f"    {key}: shape={shape} dtype={dtype}")
            if shape != (dim,):
                errors.append(f"dataset[{index}] {key} shape must be ({dim},), got {shape}")
            if has_nan_or_inf(value):
                errors.append(f"dataset[{index}] {key} contains NaN/Inf")

        for key in REQUIRED_CAMERA_KEYS:
            value = item.get(key)
            shape = tuple(getattr(value, "shape", ()))
            dtype = getattr(value, "dtype", None)
            print(f"    {key}: shape={shape} dtype={dtype}")
            if len(shape) != 3 or 3 not in (shape[0], shape[-1]):
                errors.append(f"dataset[{index}] {key} must be RGB CHW/HWC, got {shape}")

        task = item.get("task")
        print(f"    task={task!r} task_index={item.get('task_index')}")
        if not isinstance(task, str) or not task.strip():
            errors.append(f"dataset[{index}] task string missing after official decode")

    return errors


def check_one_batch(dataset: Any) -> list[str]:
    errors: list[str] = []
    print("\nDATALOADER_BATCH:")
    try:
        from torch.utils.data import DataLoader

        loader = DataLoader(dataset, batch_size=min(2, max(1, dataset.num_frames)), shuffle=False)
        batch = next(iter(loader))
    except Exception as exc:  # noqa: BLE001
        return [f"DataLoader batch failed: {type(exc).__name__}: {exc}"]

    print(f"  keys={sorted(batch.keys())}")
    for key in ("observation.state", "action", *REQUIRED_CAMERA_KEYS):
        value = batch.get(key)
        shape = tuple(getattr(value, "shape", ()))
        print(f"  {key}: shape={shape} dtype={getattr(value, 'dtype', None)}")
        if not shape:
            errors.append(f"batch missing tensor-like key {key}")
    task = batch.get("task")
    print(f"  task={task!r}")
    if task is None:
        errors.append("batch missing task")
    return errors


def sample_indices(total_frames: int, count: int) -> list[int]:
    if total_frames <= 0 or count <= 0:
        return []
    if count == 1:
        return [0]
    raw = {0, total_frames - 1}
    if count > 2:
        for i in range(1, count - 1):
            raw.add(round(i * (total_frames - 1) / (count - 1)))
    return sorted(raw)


def has_nan_or_inf(value: Any) -> bool:
    try:
        import torch

        if isinstance(value, torch.Tensor):
            return bool((~torch.isfinite(value)).any().item())
    except Exception:  # noqa: BLE001
        pass
    try:
        import numpy as np

        return bool(~np.isfinite(value).all())
    except Exception:  # noqa: BLE001
        return False


def metadata_value(obj: Any, key: str, default: Any = None) -> Any:
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)


if __name__ == "__main__":
    raise SystemExit(main())
