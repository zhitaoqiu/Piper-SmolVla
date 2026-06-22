#!/usr/bin/env python3
"""Prepare a green-only two-object dataset with green on both left/right sides.

Offline only: this script reads existing clean two-object Piper SmolVLA data,
selects GOOD green-task episodes from both LgRb and LbRg scenes, and writes a
new LeRobot dataset. It never connects to hardware and never trains a policy.
"""

from __future__ import annotations

import _bootstrap  # noqa: F401

import argparse
import csv
import os
import shutil
from collections import Counter
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

from piper_smolvla.collection import CollectionConfig, create_lerobot_dataset, image_to_chw_uint8
from piper_smolvla.schema import ACTION_KEY, GLOBAL_IMAGE_KEY, STATE_KEY, WRIST_IMAGE_KEY
from piper_smolvla.validation import validate_action, validate_state


DEFAULT_EPISODE_TABLE = "outputs/data_quality/two_obj_language_48_20260604_160739/episode_table.csv"
DEFAULT_OUTPUT = "data/two_obj_green_only_left_right_24_clean"
GREEN_TASK = "Pick up the green object and put it into the box."


@dataclass(frozen=True)
class SelectedEpisode:
    global_ep: int
    source: Path
    source_ep: int
    source_start: int
    source_end: int
    scene: str
    task: str
    frame_count: int
    action_gripper_min: float
    action_gripper_max: float
    close_frames: int
    release_frames: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a green-only left/right two-object dataset.")
    parser.add_argument("--episode-table", default=DEFAULT_EPISODE_TABLE)
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    parser.add_argument("--report-dir", default="")
    parser.add_argument("--fps", type=int, default=10)
    parser.add_argument("--overwrite-output", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    force_writable_hf_cache()

    table = Path(args.episode_table)
    if not table.exists():
        raise SystemExit(f"episode table not found: {table}")

    output = Path(args.output)
    if output.exists():
        if not args.overwrite_output:
            raise SystemExit(f"output already exists: {output}")
        shutil.rmtree(output)

    selected = read_selected_episodes(table)
    assert_selection(selected)

    report_dir = Path(args.report_dir) if args.report_dir else Path("outputs/data_quality") / (
        "two_obj_green_only_lr_" + datetime.now().strftime("%Y%m%d_%H%M%S")
    )
    report_dir.mkdir(parents=True, exist_ok=True)

    write_dataset(output, selected, fps=args.fps)
    write_report(report_dir, selected, output)

    scene_counts = Counter(ep.scene for ep in selected)
    total_frames = sum(ep.frame_count for ep in selected)
    print(f"green_only_dataset={output}")
    print(f"episodes={len(selected)}")
    print(f"frames={total_frames}")
    print(f"LgRb_green_left={scene_counts.get('LgRb', 0)}")
    print(f"LbRg_green_right={scene_counts.get('LbRg', 0)}")
    print(f"report_dir={report_dir}")
    print("TRAINING: NO")
    print("REAL ACTIONS SENT: NO")
    print("ACT PROJECT MODIFIED: NO")
    return 0


def read_selected_episodes(table: Path) -> list[SelectedEpisode]:
    selected: list[SelectedEpisode] = []
    with table.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row["task_color"] != "green":
                continue
            if row["scene"] not in {"LgRb", "LbRg"}:
                continue
            if row["automated_status"] != "GOOD":
                continue
            task = row["task"].strip()
            if task != GREEN_TASK:
                raise ValueError(f"unexpected green task for ep {row['global_ep']}: {task!r}")
            selected.append(
                SelectedEpisode(
                    global_ep=int(row["global_ep"]),
                    source=Path(row["source"]),
                    source_ep=int(row["source_ep"]),
                    source_start=int(row["source_start"]),
                    source_end=int(row["source_end"]),
                    scene=row["scene"],
                    task=task,
                    frame_count=int(row["frame_count"]),
                    action_gripper_min=float(row["action_gripper_min"]),
                    action_gripper_max=float(row["action_gripper_max"]),
                    close_frames=int(row["close_frames"]),
                    release_frames=int(row["release_frames"]),
                )
            )
    return sorted(selected, key=lambda ep: (ep.scene, ep.global_ep))


def assert_selection(selected: list[SelectedEpisode]) -> None:
    counts = Counter(ep.scene for ep in selected)
    if counts.get("LgRb", 0) == 0 or counts.get("LbRg", 0) == 0:
        raise SystemExit(f"need both green-left and green-right scenes, got {dict(counts)}")
    if len(selected) != 24:
        raise SystemExit(f"expected 24 green GOOD episodes from all48, got {len(selected)}")
    if counts.get("LgRb", 0) != 12 or counts.get("LbRg", 0) != 12:
        raise SystemExit(f"expected 12 LgRb and 12 LbRg green episodes, got {dict(counts)}")
    bad_cycle = [ep.global_ep for ep in selected if ep.close_frames <= 0 or ep.release_frames <= 0]
    if bad_cycle:
        raise SystemExit(f"selected episodes missing close/release frames: {bad_cycle}")
    missing = [str(ep.source) for ep in selected if not ep.source.exists()]
    if missing:
        raise SystemExit(f"missing source datasets: {sorted(set(missing))}")


def write_dataset(output: Path, selected: list[SelectedEpisode], *, fps: int) -> None:
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    loaded: dict[Path, Any] = {}
    out = create_lerobot_dataset(
        root=output,
        repo_id=f"piper/{output.name}",
        config=CollectionConfig(fps=fps, use_videos=True),
    )
    try:
        for new_ep, ep in enumerate(selected):
            ds = loaded.get(ep.source)
            if ds is None:
                ds = LeRobotDataset(repo_id=f"piper/{ep.source.name}", root=ep.source, tolerance_s=0.5)
                loaded[ep.source] = ds
            for idx in range(ep.source_start, ep.source_end):
                out.add_frame(standard_frame(ds[int(idx)]))
            out.save_episode()
            side = "green_left" if ep.scene == "LgRb" else "green_right"
            print(f"merged_ep={new_ep:02d} source={ep.source.name}:{ep.source_ep} scene={ep.scene} {side} frames={ep.frame_count}")
    finally:
        out.finalize()


def standard_frame(frame: dict[str, Any]) -> dict[str, Any]:
    task = str(frame.get("task", "")).strip()
    if task != GREEN_TASK:
        raise ValueError(f"expected green task, got {task!r}")
    return {
        STATE_KEY: np.asarray(validate_state(frame[STATE_KEY]), dtype=np.float32),
        ACTION_KEY: np.asarray(validate_action(frame[ACTION_KEY]), dtype=np.float32),
        GLOBAL_IMAGE_KEY: image_to_chw_uint8(frame[GLOBAL_IMAGE_KEY]),
        WRIST_IMAGE_KEY: image_to_chw_uint8(frame[WRIST_IMAGE_KEY]),
        "task": task,
    }


def write_report(report_dir: Path, selected: list[SelectedEpisode], output: Path) -> None:
    episode_csv = report_dir / "episode_table.csv"
    with episode_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "new_episode",
                "global_ep",
                "source",
                "source_ep",
                "scene",
                "green_side",
                "task",
                "frame_count",
                "action_gripper_min",
                "action_gripper_max",
                "close_frames",
                "release_frames",
            ],
        )
        writer.writeheader()
        for new_ep, ep in enumerate(selected):
            writer.writerow(
                {
                    "new_episode": new_ep,
                    "global_ep": ep.global_ep,
                    "source": ep.source,
                    "source_ep": ep.source_ep,
                    "scene": ep.scene,
                    "green_side": "left" if ep.scene == "LgRb" else "right",
                    "task": ep.task,
                    "frame_count": ep.frame_count,
                    "action_gripper_min": ep.action_gripper_min,
                    "action_gripper_max": ep.action_gripper_max,
                    "close_frames": ep.close_frames,
                    "release_frames": ep.release_frames,
                }
            )

    counts = Counter(ep.scene for ep in selected)
    total_frames = sum(ep.frame_count for ep in selected)
    report = report_dir / "report.md"
    report.write_text(
        "\n".join(
            [
                "# Two-Object Green-Only Left/Right Dataset",
                "",
                f"- output: `{output}`",
                f"- episodes: {len(selected)}",
                f"- total frames: {total_frames}",
                f"- green left / LgRb: {counts.get('LgRb', 0)}",
                f"- green right / LbRg: {counts.get('LbRg', 0)}",
                f"- task: `{GREEN_TASK}`",
                f"- episode table: `{episode_csv}`",
                "",
                "This dataset is derived only from GOOD green episodes in the 48-demo clean two-object audit.",
                "",
                "TRAINING: NO",
                "REAL ACTIONS SENT: NO",
                "ACT PROJECT MODIFIED: NO",
                "",
            ]
        ),
        encoding="utf-8",
    )


def force_writable_hf_cache() -> None:
    default = "/tmp/piper_smolvla_hf_cache"
    os.environ.setdefault("HF_HOME", default)
    os.environ.setdefault("HUGGINGFACE_HUB_CACHE", f"{default}/hub")


if __name__ == "__main__":
    raise SystemExit(main())
