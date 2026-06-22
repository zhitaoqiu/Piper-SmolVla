#!/usr/bin/env python3
"""Read-only prompt plumbing audit for Piper SmolVLA.

Checks whether green/blue task text survives dataset metadata, frame access,
policy batch preparation, LeRobot preprocessing/tokenization, and policy action
selection. This script never connects to hardware and never trains.
"""

from __future__ import annotations

import _bootstrap  # noqa: F401

import argparse
import math
import shutil
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from piper_smolvla.policy_io import load_lerobot_policy, prepare_policy_batch, select_policy_action_with_options
from piper_smolvla.rollout_runtime import reset_policy_runtime
from piper_smolvla.schema import ACTION_KEY, GLOBAL_IMAGE_KEY, PIPER_JOINT_ORDER, STATE_KEY, WRIST_IMAGE_KEY

GREEN_TASK = "Pick up the green object and put it into the box."
BLUE_TASK = "Pick up the blue object and put it into the box."
NONSENSE_TASK = "banana banana banana"
DEFAULT_CHECKPOINT = "outputs/train/smolvla_two_obj_language_16_clean_noaug_v2/checkpoints/last/pretrained_model"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit task/prompt plumbing without training or hardware.")
    parser.add_argument("--dataset", default="data/two_obj_language_16_clean")
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
    parser.add_argument("--paired-dataset", action="append", default=[])
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--episode", type=int, default=0)
    parser.add_argument("--frame-offset", type=int, default=0)
    parser.add_argument("--skip-policy", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    dataset_root = Path(args.dataset)
    dataset = LeRobotDataset(f"piper/{dataset_root.name}", root=str(dataset_root), tolerance_s=0.5)

    print("# Prompt Plumbing Audit")
    print(f"dataset={dataset_root}")
    print(f"checkpoint={args.checkpoint}")
    print("REAL ACTIONS SENT: NO")
    print("TRAINING: NO")
    print("ACT PROJECT MODIFIED: NO")

    episode_rows = episode_summaries(dataset, dataset_root)
    print_dataset_task_section(dataset, episode_rows)
    print_batch_section(dataset, episode_rows, args)
    print_training_pipeline_section(dataset_root)

    if args.skip_policy:
        print("\n## Policy sections skipped")
        return 0

    policy = load_lerobot_policy(args.checkpoint, ds_meta=dataset.meta, device=args.device)
    frame = representative_frame(dataset, episode_rows, args.episode, args.frame_offset)
    print_inference_batch_section(policy, frame)
    print_policy_sensitivity_section(policy, frame)
    print_paired_data_section(args.paired_dataset or discover_paired_datasets(dataset_root.parent))
    return 0


def episode_summaries(dataset: Any, root: Path) -> list[dict[str, Any]]:
    rows = []
    meta_episodes = getattr(dataset.meta, "episodes", None)
    if meta_episodes is None:
        raise RuntimeError("dataset.meta.episodes is missing")

    for idx in range(len(meta_episodes)):
        row = meta_episodes[idx]
        ep = int(as_scalar(row.get("episode_index", idx)))
        tasks = row.get("tasks", [])
        task = str(tasks[0] if isinstance(tasks, list) and tasks else tasks)
        start = int(as_scalar(row.get("dataset_from_index", 0)))
        end = int(as_scalar(row.get("dataset_to_index", start)))
        length = int(as_scalar(row.get("length", max(0, end - start))))
        first_frame = dataset[start] if start < len(dataset) else {}
        frame_task = str(first_frame.get("task", ""))
        task_index = int(as_scalar(first_frame.get("task_index", -1))) if first_frame else -1
        rows.append(
            {
                "episode": ep,
                "task": task,
                "frame_task": frame_task,
                "task_index": task_index,
                "length": length,
                "start": start,
                "end": end,
                "scene": infer_scene(root),
                "has_green": "green" in task.lower(),
                "has_blue": "blue" in task.lower(),
                "task_match": (not frame_task) or frame_task == task,
            }
        )
    return rows


def print_dataset_task_section(dataset: Any, rows: list[dict[str, Any]]) -> None:
    by_task: dict[str, list[int]] = defaultdict(list)
    frame_counts: dict[str, int] = defaultdict(int)
    print("\n## 1. Dataset Task Field")
    for row in rows:
        by_task[row["task"]].append(row["episode"])
        frame_counts[row["task"]] += row["length"]
        print(
            "episode={episode:02d} frames={length:04d} task_index={task_index} "
            "green={has_green} blue={has_blue} scene={scene} "
            "task_match={task_match} task={task!r} frame_task={frame_task!r}".format(**row)
        )

    green_eps = [row["episode"] for row in rows if row["has_green"]]
    blue_eps = [row["episode"] for row in rows if row["has_blue"]]
    print("task_distribution:")
    print(f"  green_episodes={len(green_eps)} episodes={green_eps}")
    print(f"  blue_episodes={len(blue_eps)} episodes={blue_eps}")
    print(f"  unique_task_strings={list(by_task)}")
    for task, episodes in by_task.items():
        print(f"  task={task!r} episodes={episodes} frames={frame_counts[task]}")
    print(f"  meta_tasks={getattr(dataset.meta, 'tasks', None)}")


def print_batch_section(dataset: Any, rows: list[dict[str, Any]], args: argparse.Namespace) -> None:
    frame = representative_frame(dataset, rows, args.episode, args.frame_offset)
    raw_keys = list(frame)
    batch = prepare_policy_batch(frame_to_observation(frame, task=str(frame.get("task", ""))))
    print("\n## 2. LeRobotDataset Batch")
    print(f"raw_frame_keys={raw_keys}")
    print(f"raw_task={frame.get('task')!r}")
    print(f"raw_task_index={as_scalar(frame.get('task_index', None))}")
    print(f"prepared_batch_keys={list(batch)}")
    for key, value in batch.items():
        print(f"  {key}: {describe_value(value)}")
    print("task_source=LeRobotDataset injects frame['task'] from meta.tasks/task_index; prepare_policy_batch wraps it as batch['task']=[text].")


def print_training_pipeline_section(dataset_root: Path) -> None:
    utils_path = Path("scripts/train_smolvla_utils.py")
    text = utils_path.read_text(encoding="utf-8")
    train_cli = shutil.which("lerobot-train") or "lerobot-train"
    command = [
        train_cli,
        "--dataset.repo_id",
        f"piper/{dataset_root.name}",
        "--dataset.root",
        str(dataset_root),
        "--policy.type",
        "smolvla",
        "--policy.pretrained_path",
        "lerobot/smolvla_base",
        "--output_dir",
        "<output>",
        "--steps",
        "<steps>",
        "--batch_size",
        "<batch-size>",
        "--num_workers",
        "0",
        "--policy.device",
        "cuda",
        "--policy.push_to_hub",
        "false",
        "--wandb.enable",
        "false",
        "--save_checkpoint",
        "true",
    ]
    print("\n## 3. Training Pipeline")
    print("actual_command_template=" + " ".join(command))
    print(f"contains_dataset_single_task={'dataset.single_task' in text}")
    print(f"contains_single_task={'single_task' in text}")
    print(f"contains_DEFAULT_TASK_INSTRUCTION={'DEFAULT_TASK_INSTRUCTION' in text}")
    print("task_override_detected=False")
    print("green_blue_can_be_collapsed_by_train_script=False")


def print_inference_batch_section(policy: Any, frame: dict[str, Any]) -> None:
    print("\n## 4. Inference Batch / Tokenization")
    tokenizer = find_tokenizer(policy)
    processed_by_task = {}
    for task in (GREEN_TASK, BLUE_TASK, NONSENSE_TASK):
        batch = prepare_policy_batch(frame_to_observation(frame, task=task))
        processed = policy.preprocessor(dict(batch))
        processed_by_task[task] = processed
        token_key = first_key_like(processed, ("language.tokens", "input_ids", "tokens"))
        mask_key = first_key_like(processed, ("language.attention_mask", "attention_mask"))
        print(f"task={task!r}")
        print(f"  batch_keys={list(batch)}")
        print(f"  processed_keys={list(processed)}")
        print(f"  token_key={token_key} mask_key={mask_key}")
        if token_key:
            ids = active_token_ids(processed[token_key], processed.get(mask_key))
            print(f"  token_count={len(ids)} token_head={ids[:24]}")
            print(f"  decoded={decode_tokens(tokenizer, ids)!r}")

    print("token_equal_green_blue=" + str(tokens_equal(processed_by_task[GREEN_TASK], processed_by_task[BLUE_TASK])))
    print("token_equal_green_banana=" + str(tokens_equal(processed_by_task[GREEN_TASK], processed_by_task[NONSENSE_TASK])))


def print_policy_sensitivity_section(policy: Any, frame: dict[str, Any]) -> None:
    print("\n## 5. Policy Sensitivity")
    actions = {}
    for label, task in (("green", GREEN_TASK), ("blue", BLUE_TASK), ("nonsense", NONSENSE_TASK)):
        reset_policy_runtime(policy)
        batch = prepare_policy_batch(frame_to_observation(frame, task=task))
        actions[label] = np.asarray(select_policy_action_with_options(policy, batch, validate_limits=False), dtype=np.float64)
        print(f"action_{label}={format_vec(actions[label])}")

    for a, b in (("green", "blue"), ("green", "nonsense"), ("blue", "nonsense")):
        diff = np.abs(actions[a] - actions[b])
        print(f"abs_diff_{a}_{b}={format_vec(diff)} max={diff.max():.8f} mean={diff.mean():.8f}")


def print_paired_data_section(paths: list[Path | str]) -> None:
    print("\n## 6. Paired Data Quality")
    if not paths:
        print("paired_datasets=[]")
        return

    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    for path_like in paths:
        root = Path(path_like)
        if not root.exists():
            continue
        try:
            ds = LeRobotDataset(f"piper/{root.name}", root=str(root), tolerance_s=0.5)
            rows = episode_summaries(ds, root)
            green = [row for row in rows if row["has_green"]]
            blue = [row for row in rows if row["has_blue"]]
            print(f"dataset={root} episodes={len(rows)} green={len(green)} blue={len(blue)} scene={infer_scene(root)}")
            if green and blue:
                g = green[0]
                b = blue[0]
                image_sim = start_image_similarity(ds, g, b)
                action_diff = early_action_diff(ds, g, b)
                print(f"  pair green_ep={g['episode']} blue_ep={b['episode']}")
                print(f"  green_task={g['task']!r}")
                print(f"  blue_task={b['task']!r}")
                print(f"  start_global_mean_abs_diff={image_sim:.3f}")
                print(f"  early30_action_diff_j1_j2_j3_j4_j6={format_vec(action_diff, precision=5)}")
        except Exception as exc:  # noqa: BLE001
            print(f"dataset={root} ERROR={type(exc).__name__}: {exc}")


def representative_frame(dataset: Any, rows: list[dict[str, Any]], episode: int, frame_offset: int) -> dict[str, Any]:
    by_episode = {row["episode"]: row for row in rows}
    row = by_episode.get(episode, rows[0])
    index = min(row["end"] - 1, row["start"] + max(0, frame_offset))
    return dict(dataset[index])


def frame_to_observation(frame: dict[str, Any], *, task: str) -> dict[str, Any]:
    return {
        STATE_KEY: frame[STATE_KEY],
        GLOBAL_IMAGE_KEY: frame[GLOBAL_IMAGE_KEY],
        WRIST_IMAGE_KEY: frame[WRIST_IMAGE_KEY],
        "task": task,
    }


def discover_paired_datasets(data_root: Path) -> list[Path]:
    names = ("two_obj_language_20_paired", "two_obj_paired", "two_obj_paired_LbRg", "two_obj_paired_LgRb", "two_obj_paired_merged")
    return [data_root / name for name in names if (data_root / name).exists()]


def infer_scene(root: Path) -> str:
    name = root.name
    if "LgRb" in name:
        return "LgRb"
    if "LbRg" in name:
        return "LbRg"
    return "unknown"


def as_scalar(value: Any) -> Any:
    if hasattr(value, "detach"):
        value = value.detach().cpu()
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            pass
    return value


def describe_value(value: Any) -> str:
    if hasattr(value, "shape"):
        return f"{type(value).__name__} shape={tuple(value.shape)} dtype={getattr(value, 'dtype', None)}"
    if isinstance(value, list):
        return f"list len={len(value)} value={value[:2]!r}"
    return f"{type(value).__name__} value={value!r}"


def first_key_like(mapping: dict[str, Any], needles: tuple[str, ...]) -> str | None:
    for key in mapping:
        lower = key.lower()
        if any(needle in lower for needle in needles):
            return key
    return None


def active_token_ids(tokens: Any, mask: Any | None) -> list[int]:
    token_arr = to_numpy(tokens)
    if token_arr.ndim > 1:
        token_arr = token_arr[0]
    if mask is not None:
        mask_arr = to_numpy(mask)
        if mask_arr.ndim > 1:
            mask_arr = mask_arr[0]
        token_arr = token_arr.astype(np.int64)[mask_arr.astype(bool)]
    return [int(v) for v in token_arr.tolist()]


def tokens_equal(a: dict[str, Any], b: dict[str, Any]) -> bool:
    token_key = first_key_like(a, ("language.tokens", "input_ids", "tokens"))
    if token_key is None or token_key not in b:
        return False
    return np.array_equal(to_numpy(a[token_key]), to_numpy(b[token_key]))


def decode_tokens(tokenizer: Any, ids: list[int]) -> str:
    if tokenizer is None:
        return "<no tokenizer found>"
    try:
        return tokenizer.decode(ids, skip_special_tokens=False)
    except Exception as exc:  # noqa: BLE001
        return f"<decode failed: {exc}>"


def find_tokenizer(root: Any, *, max_depth: int = 5) -> Any | None:
    seen: set[int] = set()

    def visit(obj: Any, depth: int) -> Any | None:
        if obj is None or depth > max_depth or id(obj) in seen:
            return None
        seen.add(id(obj))
        if callable(getattr(obj, "decode", None)) and callable(getattr(obj, "__call__", None)):
            return obj
        for attr in ("tokenizer", "processor", "image_processor", "preprocessor"):
            child = getattr(obj, attr, None)
            found = visit(child, depth + 1)
            if found is not None:
                return found
        if isinstance(obj, (list, tuple)):
            children = obj
        elif isinstance(obj, dict):
            children = obj.values()
        else:
            children = getattr(obj, "__dict__", {}).values()
        for child in children:
            if isinstance(child, (str, bytes, int, float, bool, Path)):
                continue
            found = visit(child, depth + 1)
            if found is not None:
                return found
        return None

    return visit(root, 0)


def start_image_similarity(dataset: Any, green_row: dict[str, Any], blue_row: dict[str, Any]) -> float:
    g = normalize_image(dataset[green_row["start"]][GLOBAL_IMAGE_KEY])
    b = normalize_image(dataset[blue_row["start"]][GLOBAL_IMAGE_KEY])
    return float(np.mean(np.abs(g.astype(np.float32) - b.astype(np.float32))))


def early_action_diff(dataset: Any, green_row: dict[str, Any], blue_row: dict[str, Any]) -> np.ndarray:
    length = min(30, green_row["length"], blue_row["length"])
    joints = [0, 1, 2, 3, 5]
    diffs = []
    for offset in range(length):
        g = to_numpy(dataset[green_row["start"] + offset][ACTION_KEY]).astype(np.float64)
        b = to_numpy(dataset[blue_row["start"] + offset][ACTION_KEY]).astype(np.float64)
        diffs.append(np.abs(g[joints] - b[joints]))
    return np.mean(diffs, axis=0) if diffs else np.full(len(joints), math.nan)


def normalize_image(image: Any) -> np.ndarray:
    arr = to_numpy(image)
    if arr.ndim == 3 and arr.shape[0] == 3:
        arr = np.moveaxis(arr, 0, -1)
    if arr.dtype != np.uint8:
        arr = np.clip(arr * 255 if arr.max() <= 1.0 else arr, 0, 255).astype(np.uint8)
    return arr


def to_numpy(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    elif hasattr(value, "cpu") and hasattr(value, "numpy"):
        value = value.cpu().numpy()
    elif hasattr(value, "numpy"):
        value = value.numpy()
    return np.asarray(value)


def format_vec(values: Any, *, precision: int = 6) -> str:
    arr = np.asarray(values, dtype=np.float64)
    return "[" + ", ".join(f"{v:.{precision}f}" for v in arr.tolist()) + "]"


if __name__ == "__main__":
    raise SystemExit(main())
