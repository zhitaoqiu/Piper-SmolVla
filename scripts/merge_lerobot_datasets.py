#!/usr/bin/env python3
"""Merge one-episode Piper SmolVLA LeRobot datasets into one dataset."""

from __future__ import annotations

import _bootstrap  # noqa: F401

import argparse
import os
from pathlib import Path
from typing import Any

import numpy as np

from piper_smolvla.collection import CollectionConfig, create_lerobot_dataset, image_to_chw_uint8
from piper_smolvla.schema import ACTION_KEY, GLOBAL_IMAGE_KEY, STATE_KEY, WRIST_IMAGE_KEY
from piper_smolvla.validation import validate_action, validate_state


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Merge Piper SmolVLA LeRobot datasets.")
    parser.add_argument("--input-glob", default="data/two_obj_L*_*_[1-4]")
    parser.add_argument("--input", action="append", default=[], help="Explicit input dataset root. Repeatable.")
    parser.add_argument("--output", required=True)
    parser.add_argument("--repo-id", default="")
    parser.add_argument("--fps", type=int, default=10)
    parser.add_argument("--expect-count", type=int, default=0)
    parser.add_argument("--no-videos", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    _force_writable_hf_cache()

    roots = _resolve_inputs(args)
    if args.expect_count and len(roots) != args.expect_count:
        raise SystemExit(f"expected {args.expect_count} input datasets, got {len(roots)}")

    output = Path(args.output)
    if output.exists():
        raise SystemExit(f"output already exists: {output}")

    repo_id = args.repo_id or f"piper/{output.name}"
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    out = create_lerobot_dataset(
        root=output,
        repo_id=repo_id,
        config=CollectionConfig(fps=args.fps, use_videos=not args.no_videos),
    )
    written_episodes = 0
    written_frames = 0
    try:
        for root in roots:
            src = LeRobotDataset(repo_id=f"piper/{root.name}", root=root, tolerance_s=0.5)
            ep_indices = np.array(src.hf_dataset["episode_index"])
            unique_eps = sorted(set(ep_indices))
            for ep in unique_eps:
                mask = ep_indices == ep
                ep_frames_idx = np.where(mask)[0]
                task = str(src[int(ep_frames_idx[0])].get("task", "")).strip()
                for idx in ep_frames_idx:
                    out.add_frame(_standard_frame(src[int(idx)], task_fallback=task))
                out.save_episode()
                written_episodes += 1
                written_frames += len(ep_frames_idx)
                print(f"  ep {ep:4d}  {task:>8s}  {len(ep_frames_idx):4d}f")
            print(f"merged {root}  episodes={len(unique_eps)}  frames={sum(1 for _ in ep_indices)}")
    finally:
        out.finalize()

    print(f"\nmerged_dataset={output}")
    print(f"episodes={written_episodes}")
    print(f"frames={written_frames}")
    print("NO HARDWARE ACCESSED")
    print("NO MOTION COMMAND SENT")
    return 0


def _resolve_inputs(args: argparse.Namespace) -> list[Path]:
    roots = [Path(value) for value in args.input]
    if not roots:
        roots = sorted(Path(".").glob(args.input_glob))
    roots = [root for root in roots if root.is_dir()]
    if not roots:
        raise SystemExit("no input datasets found")
    missing = [root for root in roots if not (root / "meta" / "info.json").exists()]
    if missing:
        raise SystemExit(f"input missing meta/info.json: {missing}")
    return roots


def _standard_frame(frame: dict[str, Any], *, task_fallback: str) -> dict[str, Any]:
    task = str(frame.get("task") or task_fallback).strip()
    if not task:
        raise ValueError("task is required")
    return {
        STATE_KEY: np.asarray(validate_state(frame[STATE_KEY]), dtype=np.float32),
        ACTION_KEY: np.asarray(validate_action(frame[ACTION_KEY]), dtype=np.float32),
        GLOBAL_IMAGE_KEY: image_to_chw_uint8(frame[GLOBAL_IMAGE_KEY]),
        WRIST_IMAGE_KEY: image_to_chw_uint8(frame[WRIST_IMAGE_KEY]),
        "task": task,
    }


def _dataset_task(dataset: Any) -> str:
    if len(dataset) <= 0:
        raise ValueError("dataset has no frames")
    task = str(dataset[0].get("task", "")).strip()
    if not task:
        raise ValueError("dataset first frame has no task")
    return task


def _force_writable_hf_cache() -> None:
    default = "/tmp/piper_smolvla_hf_cache"
    os.environ.setdefault("HF_HOME", default)
    os.environ.setdefault("HUGGINGFACE_HUB_CACHE", f"{default}/hub")


if __name__ == "__main__":
    raise SystemExit(main())
