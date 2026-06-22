#!/usr/bin/env python3
"""Prepare the clean 40-demo single-cube line4pos dataset.

Offline only. This script merges the accepted position demos, replaces the
failed pos1_07 source episode with its retake, runs quality checks, and writes
position-difference statistics.
"""

from __future__ import annotations

import _bootstrap  # noqa: F401

import argparse
import csv
import json
import os
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from piper_smolvla.collection import CollectionConfig, create_lerobot_dataset, image_to_chw_uint8
from piper_smolvla.schema import ACTION_KEY, GLOBAL_IMAGE_KEY, STATE_KEY, WRIST_IMAGE_KEY
from piper_smolvla.validation import validate_action, validate_state

TASK = "Pick up the cube and put it into the box."
OUTPUT_DATASET = Path("data/single_cube_line4pos_40_clean")


@dataclass(frozen=True)
class EpisodeSpec:
    label: str
    position: str
    source: Path
    source_episode: int
    expected_status: str = "GOOD"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Merge and audit single_cube_line4pos 40 clean demos.")
    parser.add_argument("--output", default=str(OUTPUT_DATASET))
    parser.add_argument("--report-dir", default="")
    parser.add_argument("--fps", type=int, default=10)
    parser.add_argument("--no-videos", action="store_true")
    parser.add_argument("--frame-min", type=int, default=100)
    parser.add_argument("--frame-review-min", type=int, default=120)
    parser.add_argument("--frame-max", type=int, default=180)
    parser.add_argument("--close-threshold", type=float, default=0.07)
    parser.add_argument("--release-threshold", type=float, default=0.09)
    parser.add_argument("--black-mean-threshold", type=float, default=5.0)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    _force_writable_hf_cache()
    output = Path(args.output)
    if output.exists():
        raise SystemExit(f"output already exists: {output}")

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_dir = Path(args.report_dir) if args.report_dir else Path("outputs/data_quality") / "single_cube_line4pos" / f"line4pos_40_{stamp}"
    report_dir.mkdir(parents=True, exist_ok=True)
    samples_dir = report_dir / "samples"
    samples_dir.mkdir(parents=True, exist_ok=True)

    specs = build_manifest()
    if len(specs) != 40:
        raise SystemExit(f"internal manifest error: expected 40 specs, got {len(specs)}")

    loaded = load_sources(specs)
    rows = [inspect_episode(spec, loaded[spec.source], samples_dir=samples_dir, args=args) for spec in specs]
    bad = [row for row in rows if row["status"] == "BAD"]
    if bad:
        write_tables(report_dir, rows, [])
        raise SystemExit(f"refusing to merge because BAD episodes exist: {[row['label'] for row in bad]}")

    write_merged_dataset(output, specs, loaded, fps=args.fps, use_videos=not args.no_videos)
    position_stats = analyze_positions(rows, specs, loaded)
    write_tables(report_dir, rows, position_stats)
    write_manifest(report_dir, specs)
    write_report(report_dir / "report.md", rows, position_stats, output=output)

    print(f"clean_dataset={output}")
    print(f"report_dir={report_dir}")
    print(f"episodes={len(rows)}")
    print(f"frames={sum(int(row['frame_count']) for row in rows)}")
    print_status_counts(rows)
    print("TRAINING: NO")
    print("REAL ACTIONS SENT: NO")
    print("POLICY ACTIONS SENT: NO")
    print("ACT PROJECT MODIFIED: NO")
    return 0


def build_manifest() -> list[EpisodeSpec]:
    specs: list[EpisodeSpec] = [
        EpisodeSpec("pos1_01", "pos1", Path("data/single_cube_pos1_01"), 0),
        EpisodeSpec("pos1_02", "pos1", Path("data/single_cube_pos1_02"), 0),
    ]
    # data/single_cube_pos1_03_to_10 contains source episodes:
    # 0->pos1_03, 1->pos1_04, 2->pos1_05, 3->pos1_06, 4->bad pos1_07,
    # 5->pos1_08, 6->pos1_09, 7->pos1_10.
    for source_ep, label in ((0, "pos1_03"), (1, "pos1_04"), (2, "pos1_05"), (3, "pos1_06")):
        specs.append(EpisodeSpec(label, "pos1", Path("data/single_cube_pos1_03_to_10"), source_ep))
    specs.append(EpisodeSpec("pos1_07", "pos1", Path("data/single_cube_pos1_07_retake"), 0, expected_status="NEED REVIEW"))
    for source_ep, label in ((5, "pos1_08"), (6, "pos1_09"), (7, "pos1_10")):
        specs.append(EpisodeSpec(label, "pos1", Path("data/single_cube_pos1_03_to_10"), source_ep, expected_status="NEED REVIEW"))

    for pos, source in (
        ("pos2", Path("data/single_cube_pos2_10")),
        ("pos3", Path("data/single_cube_pos3_10")),
        ("pos4", Path("data/single_cube_pos4_10")),
    ):
        for source_ep in range(10):
            specs.append(EpisodeSpec(f"{pos}_{source_ep + 1:02d}", pos, source, source_ep))
    return specs


def load_sources(specs: list[EpisodeSpec]) -> dict[Path, Any]:
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    loaded: dict[Path, Any] = {}
    for source in sorted({spec.source for spec in specs}):
        if not source.exists():
            raise SystemExit(f"missing source dataset: {source}")
        loaded[source] = LeRobotDataset(repo_id=f"piper/{source.name}", root=source, tolerance_s=0.5)
    return loaded


def episode_indices(dataset: Any, source_episode: int) -> np.ndarray:
    ep_indices = np.asarray(dataset.hf_dataset["episode_index"])
    selected = np.where(ep_indices == source_episode)[0]
    if len(selected) == 0:
        raise ValueError(f"source episode not found: {source_episode}")
    return selected


def inspect_episode(spec: EpisodeSpec, dataset: Any, *, samples_dir: Path, args: argparse.Namespace) -> dict[str, Any]:
    indices = episode_indices(dataset, spec.source_episode)
    states = []
    actions = []
    global_means = []
    wrist_means = []
    black_frames = 0
    task = ""
    for index in indices:
        frame = dataset[int(index)]
        if not task:
            task = str(frame.get("task", ""))
        state = np.asarray(validate_state(frame[STATE_KEY]), dtype=np.float32)
        action = np.asarray(validate_action(frame[ACTION_KEY]), dtype=np.float32)
        states.append(state)
        actions.append(action)
        global_img = image_to_chw_uint8(frame[GLOBAL_IMAGE_KEY])
        wrist_img = image_to_chw_uint8(frame[WRIST_IMAGE_KEY])
        gm = float(global_img.mean())
        wm = float(wrist_img.mean())
        global_means.append(gm)
        wrist_means.append(wm)
        if gm < args.black_mean_threshold or wm < args.black_mean_threshold:
            black_frames += 1

    states_arr = np.asarray(states, dtype=np.float32)
    actions_arr = np.asarray(actions, dtype=np.float32)
    gripper = actions_arr[:, 6]
    close = np.where(gripper < args.close_threshold)[0]
    release = np.where(gripper > args.release_threshold)[0]
    close_first = int(close[0]) if len(close) else -1
    close_last = int(close[-1]) if len(close) else -1
    pre_open = np.where(gripper[:close_first] > args.release_threshold)[0] if close_first > 0 else np.asarray([], dtype=int)
    post_release = release[release > close_last] if len(close) else np.asarray([], dtype=int)
    has_cycle = len(pre_open) > 0 and len(close) > 0 and len(post_release) > 0

    sample_paths = export_samples(spec, dataset, indices, samples_dir=samples_dir, close_first=close_first)
    status, notes = classify(
        frame_count=len(indices),
        task=task,
        black_frames=black_frames,
        close_count=len(close),
        release_count=len(release),
        has_cycle=has_cycle,
        args=args,
    )
    return {
        "label": spec.label,
        "position": spec.position,
        "source": str(spec.source),
        "source_episode": spec.source_episode,
        "frame_count": int(len(indices)),
        "task": task,
        "state_dim": 7,
        "action_dim": 7,
        "black_frames": int(black_frames),
        "action_gripper_min": float(gripper.min()),
        "action_gripper_max": float(gripper.max()),
        "state_gripper_min": float(states_arr[:, 6].min()),
        "state_gripper_max": float(states_arr[:, 6].max()),
        "close_frames": int(len(close)),
        "release_frames": int(len(release)),
        "close_first": close_first,
        "close_last": close_last,
        "release_after_close_first": int(post_release[0]) if len(post_release) else -1,
        "open_close_release": bool(has_cycle),
        "close_j2": float(actions_arr[close_first, 1]) if close_first >= 0 else float("nan"),
        "close_j3": float(actions_arr[close_first, 2]) if close_first >= 0 else float("nan"),
        "close_gripper": float(actions_arr[close_first, 6]) if close_first >= 0 else float("nan"),
        "global_mean_min": float(min(global_means)),
        "global_mean_max": float(max(global_means)),
        "wrist_mean_min": float(min(wrist_means)),
        "wrist_mean_max": float(max(wrist_means)),
        "status": status,
        "expected_status": spec.expected_status,
        "notes": notes,
        **sample_paths,
    }


def classify(
    *,
    frame_count: int,
    task: str,
    black_frames: int,
    close_count: int,
    release_count: int,
    has_cycle: bool,
    args: argparse.Namespace,
) -> tuple[str, str]:
    bad = []
    review = []
    if task != TASK:
        bad.append("task_mismatch")
    if black_frames > 0:
        bad.append(f"black_frames={black_frames}")
    if close_count <= 0:
        bad.append("no_close_frames")
    if release_count <= 0:
        bad.append("no_release_frames")
    if not has_cycle:
        bad.append("no_open_close_release_cycle")
    if frame_count < args.frame_min:
        bad.append(f"too_short<{args.frame_min}")
    elif frame_count < args.frame_review_min or frame_count > args.frame_max:
        review.append(f"frame_count_outside_{args.frame_review_min}_{args.frame_max}")
    if bad:
        return "BAD", ";".join(bad + review)
    if review:
        return "NEED REVIEW", ";".join(review)
    return "GOOD", ""


def export_samples(spec: EpisodeSpec, dataset: Any, indices: np.ndarray, *, samples_dir: Path, close_first: int) -> dict[str, str]:
    out_dir = samples_dir / spec.label
    out_dir.mkdir(parents=True, exist_ok=True)
    local = {
        "start": 0,
        "mid": len(indices) // 2,
        "end": len(indices) - 1,
    }
    if close_first >= 0:
        local["close"] = close_first
    outputs: dict[str, str] = {}
    for tag, local_index in sorted(local.items(), key=lambda item: item[1]):
        frame = dataset[int(indices[local_index])]
        gp = out_dir / f"{tag}_global.jpg"
        wp = out_dir / f"{tag}_wrist.jpg"
        cv2.imwrite(str(gp), chw_rgb_to_bgr(image_to_chw_uint8(frame[GLOBAL_IMAGE_KEY])))
        cv2.imwrite(str(wp), chw_rgb_to_bgr(image_to_chw_uint8(frame[WRIST_IMAGE_KEY])))
        outputs[f"{tag}_global"] = str(gp)
        outputs[f"{tag}_wrist"] = str(wp)
    return outputs


def chw_rgb_to_bgr(chw: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(np.moveaxis(chw, 0, -1), cv2.COLOR_RGB2BGR)


def write_merged_dataset(output: Path, specs: list[EpisodeSpec], loaded: dict[Path, Any], *, fps: int, use_videos: bool) -> None:
    out = create_lerobot_dataset(
        root=output,
        repo_id=f"piper/{output.name}",
        config=CollectionConfig(fps=fps, use_videos=use_videos),
    )
    try:
        for spec in specs:
            ds = loaded[spec.source]
            indices = episode_indices(ds, spec.source_episode)
            task = str(ds[int(indices[0])].get("task", "") or TASK)
            for index in indices:
                out.add_frame(standard_frame(ds[int(index)], task_fallback=task))
            out.save_episode()
            print(f"merged {spec.label} source={spec.source.name}:{spec.source_episode} frames={len(indices)}")
    finally:
        out.finalize()


def standard_frame(frame: dict[str, Any], *, task_fallback: str) -> dict[str, Any]:
    return {
        STATE_KEY: np.asarray(validate_state(frame[STATE_KEY]), dtype=np.float32),
        ACTION_KEY: np.asarray(validate_action(frame[ACTION_KEY]), dtype=np.float32),
        GLOBAL_IMAGE_KEY: image_to_chw_uint8(frame[GLOBAL_IMAGE_KEY]),
        WRIST_IMAGE_KEY: image_to_chw_uint8(frame[WRIST_IMAGE_KEY]),
        "task": str(frame.get("task") or task_fallback or TASK),
    }


def analyze_positions(rows: list[dict[str, Any]], specs: list[EpisodeSpec], loaded: dict[Path, Any]) -> list[dict[str, Any]]:
    actions_by_pos: dict[str, list[np.ndarray]] = {pos: [] for pos in ("pos1", "pos2", "pos3", "pos4")}
    for spec in specs:
        actions_by_pos[spec.position].append(load_actions(loaded[spec.source], spec.source_episode))

    stats: list[dict[str, Any]] = []
    for left, right in (("pos1", "pos4"), ("pos2", "pos3")):
        for horizon in (30, 60):
            stats.append(compare_position_pair(left, right, horizon, actions_by_pos[left], actions_by_pos[right]))
    for position in ("pos1", "pos2", "pos3", "pos4"):
        pos_rows = [row for row in rows if row["position"] == position]
        stats.append(position_summary(position, pos_rows, actions_by_pos[position]))
    return stats


def load_actions(dataset: Any, source_episode: int) -> np.ndarray:
    indices = episode_indices(dataset, source_episode)
    return np.asarray([validate_action(dataset[int(index)][ACTION_KEY]) for index in indices], dtype=np.float32)


def compare_position_pair(left: str, right: str, horizon: int, left_actions: list[np.ndarray], right_actions: list[np.ndarray]) -> dict[str, Any]:
    joints = {"j1": 0, "j2": 1, "j3": 2, "j4": 3, "j6": 5, "gripper": 6}
    out: dict[str, Any] = {"kind": "pair_diff", "left": left, "right": right, "horizon": horizon}
    for name, idx in joints.items():
        diffs = []
        for a in left_actions:
            for b in right_actions:
                n = min(horizon, len(a), len(b))
                diffs.append(float(np.mean(np.abs(a[:n, idx] - b[:n, idx]))))
        out[f"{name}_mean_abs_diff"] = float(np.mean(diffs))
        out[f"{name}_max_pair_mean_abs_diff"] = float(np.max(diffs))
    return out


def position_summary(position: str, rows: list[dict[str, Any]], actions: list[np.ndarray]) -> dict[str, Any]:
    stacked = np.concatenate(actions, axis=0)
    out: dict[str, Any] = {
        "kind": "position_summary",
        "position": position,
        "episodes": len(rows),
        "frames": sum(int(row["frame_count"]) for row in rows),
        "close_first_min": min(int(row["close_first"]) for row in rows),
        "close_first_max": max(int(row["close_first"]) for row in rows),
        "close_j2_min": min(float(row["close_j2"]) for row in rows),
        "close_j2_max": max(float(row["close_j2"]) for row in rows),
        "close_j3_min": min(float(row["close_j3"]) for row in rows),
        "close_j3_max": max(float(row["close_j3"]) for row in rows),
        "close_gripper_min": min(float(row["close_gripper"]) for row in rows),
        "close_gripper_max": max(float(row["close_gripper"]) for row in rows),
    }
    for name, idx in (("j1", 0), ("j2", 1), ("j3", 2), ("j4", 3), ("j5", 4), ("j6", 5), ("gripper", 6)):
        out[f"action_{name}_min"] = float(stacked[:, idx].min())
        out[f"action_{name}_max"] = float(stacked[:, idx].max())
    return out


def write_tables(report_dir: Path, rows: list[dict[str, Any]], position_stats: list[dict[str, Any]]) -> None:
    write_csv(report_dir / "episode_table.csv", rows)
    if position_stats:
        keys = sorted({key for row in position_stats for key in row})
        write_csv(report_dir / "position_analysis.csv", position_stats, fieldnames=keys)


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str] | None = None) -> None:
    if not rows:
        return
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames or list(rows[0].keys()), extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def write_manifest(report_dir: Path, specs: list[EpisodeSpec]) -> None:
    payload = [
        {
            "label": spec.label,
            "position": spec.position,
            "source": str(spec.source),
            "source_episode": spec.source_episode,
            "expected_status": spec.expected_status,
        }
        for spec in specs
    ]
    (report_dir / "manifest.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")


def write_report(path: Path, rows: list[dict[str, Any]], position_stats: list[dict[str, Any]], *, output: Path) -> None:
    counts: dict[str, int] = {}
    pos_counts: dict[str, int] = {}
    for row in rows:
        counts[str(row["status"])] = counts.get(str(row["status"]), 0) + 1
        pos_counts[str(row["position"])] = pos_counts.get(str(row["position"]), 0) + 1
    lines = [
        "# single_cube_line4pos_40_clean Data Quality Report",
        "",
        f"- dataset: `{output}`",
        f"- episodes: {len(rows)}",
        f"- frames: {sum(int(row['frame_count']) for row in rows)}",
        f"- position counts: {pos_counts}",
        f"- status counts: {counts}",
        "- task: `Pick up the cube and put it into the box.`",
        "- training: NO",
        "- real actions sent by this script: NO",
        "- policy actions sent: NO",
        "",
        "## Position Analysis",
    ]
    for row in position_stats:
        lines.append(json.dumps(row, sort_keys=True))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def print_status_counts(rows: list[dict[str, Any]]) -> None:
    counts: dict[str, int] = {}
    pos_counts: dict[str, int] = {}
    for row in rows:
        counts[str(row["status"])] = counts.get(str(row["status"]), 0) + 1
        pos_counts[str(row["position"])] = pos_counts.get(str(row["position"]), 0) + 1
    print(f"position_counts={pos_counts}")
    print(f"status_counts={counts}")


def _force_writable_hf_cache() -> None:
    default = "/tmp/piper_smolvla_hf_cache"
    os.environ.setdefault("HF_HOME", default)
    os.environ.setdefault("HUGGINGFACE_HUB_CACHE", f"{default}/hub")
    os.environ.setdefault("HF_DATASETS_CACHE", "/tmp/piper_smolvla_datasets_cache")


if __name__ == "__main__":
    raise SystemExit(main())
