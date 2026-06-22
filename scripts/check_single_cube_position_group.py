#!/usr/bin/env python3
"""Check one single-cube position group.

This script is offline-only. It reads LeRobot datasets, groups frames by
episode, exports start/close/mid/end samples, and prints a compact quality
table for line4pos collection.
"""

from __future__ import annotations

import _bootstrap  # noqa: F401

import argparse
import csv
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

from piper_smolvla.collection import image_to_chw_uint8
from piper_smolvla.schema import ACTION_KEY, GLOBAL_IMAGE_KEY, STATE_KEY, WRIST_IMAGE_KEY

EXPECTED_TASK = "Pick up the cube and put it into the box."


@dataclass(frozen=True)
class EpisodeRef:
    label: str
    path: Path
    source_episode: int
    frame_indices: list[int]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Offline sanity check for one single-cube line4pos group.")
    parser.add_argument("--position", required=True, help="Position label, e.g. pos1.")
    parser.add_argument("--datasets", nargs="+", required=True, help="LeRobot dataset roots to check.")
    parser.add_argument("--output-dir", default="", help="Quality output dir. Defaults under outputs/data_quality.")
    parser.add_argument("--expected-task", default=EXPECTED_TASK)
    parser.add_argument("--frame-min", type=int, default=120)
    parser.add_argument("--frame-max", type=int, default=180)
    parser.add_argument("--close-threshold", type=float, default=0.07)
    parser.add_argument("--release-threshold", type=float, default=0.09)
    parser.add_argument("--black-mean-threshold", type=float, default=5.0)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path(args.output_dir) if args.output_dir else Path("outputs/data_quality") / "single_cube_line4pos" / f"{args.position}_{stamp}"
    samples_dir = out_dir / "samples"
    samples_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, object]] = []
    refs = collect_episode_refs(args.datasets, args.position)
    for ref in refs:
        rows.append(check_episode(ref, samples_dir=samples_dir, args=args))

    out_dir.mkdir(parents=True, exist_ok=True)
    table_path = out_dir / "episode_table.csv"
    write_csv(table_path, rows)
    write_report(out_dir / "report.md", rows, args=args, table_path=table_path)
    print_summary(rows, out_dir=out_dir, table_path=table_path)
    print("TRAINING: NO")
    print("REAL ACTIONS SENT: NO")
    print("POLICY ACTIONS SENT: NO")
    return 0


def collect_episode_refs(dataset_paths: list[str], position: str) -> list[EpisodeRef]:
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    refs: list[EpisodeRef] = []
    global_episode = 1
    for dataset_path in dataset_paths:
        root = Path(dataset_path)
        ds = LeRobotDataset(repo_id=f"check_{root.name}", root=root)
        per_ep: dict[int, list[int]] = {}
        for idx in range(len(ds)):
            frame = ds[idx]
            ep = int(np.asarray(frame["episode_index"]).reshape(-1)[0])
            per_ep.setdefault(ep, []).append(idx)
        for source_ep in sorted(per_ep):
            label = f"{position}_{global_episode:02d}"
            refs.append(EpisodeRef(label=label, path=root, source_episode=source_ep, frame_indices=per_ep[source_ep]))
            global_episode += 1
    return refs


def check_episode(ref: EpisodeRef, *, samples_dir: Path, args: argparse.Namespace) -> dict[str, object]:
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    ds = LeRobotDataset(repo_id=f"check_{ref.path.name}", root=ref.path)
    states = []
    actions = []
    global_means = []
    wrist_means = []
    task = ""
    black_frames = 0
    state_dim = -1
    action_dim = -1

    for idx in ref.frame_indices:
        frame = ds[idx]
        if not task:
            task = str(frame.get("task", ""))
        state = np.asarray(frame[STATE_KEY], dtype=np.float32)
        action = np.asarray(frame[ACTION_KEY], dtype=np.float32)
        state_dim = int(state.reshape(-1).shape[0])
        action_dim = int(action.reshape(-1).shape[0])
        states.append(state.reshape(-1))
        actions.append(action.reshape(-1))

        global_img = image_to_chw_uint8(frame[GLOBAL_IMAGE_KEY])
        wrist_img = image_to_chw_uint8(frame[WRIST_IMAGE_KEY])
        global_mean = float(global_img.mean())
        wrist_mean = float(wrist_img.mean())
        global_means.append(global_mean)
        wrist_means.append(wrist_mean)
        if global_mean < args.black_mean_threshold or wrist_mean < args.black_mean_threshold:
            black_frames += 1

    states_arr = np.asarray(states, dtype=np.float32)
    actions_arr = np.asarray(actions, dtype=np.float32)
    gripper = actions_arr[:, 6]
    close_indices = np.where(gripper < args.close_threshold)[0]
    release_indices = np.where(gripper > args.release_threshold)[0]
    close_first = int(close_indices[0]) if len(close_indices) else -1
    close_last = int(close_indices[-1]) if len(close_indices) else -1
    release_after_close = release_indices[release_indices > close_last] if len(close_indices) else np.asarray([], dtype=int)
    pre_open = np.where(gripper[:close_first] > args.release_threshold)[0] if close_first > 0 else np.asarray([], dtype=int)
    has_cycle = len(pre_open) > 0 and len(close_indices) > 0 and len(release_after_close) > 0

    sample_paths = export_samples(ref, ds, samples_dir=samples_dir, close_first=close_first)
    status, notes = classify(
        frame_count=len(ref.frame_indices),
        state_dim=state_dim,
        action_dim=action_dim,
        task=task,
        black_frames=black_frames,
        close_count=len(close_indices),
        release_count=len(release_indices),
        has_cycle=has_cycle,
        args=args,
    )

    return {
        "label": ref.label,
        "dataset": str(ref.path),
        "source_episode": ref.source_episode,
        "frame_count": len(ref.frame_indices),
        "task": task,
        "state_dim": state_dim,
        "action_dim": action_dim,
        "black_frames": black_frames,
        "action_gripper_min": float(gripper.min()),
        "action_gripper_max": float(gripper.max()),
        "state_gripper_min": float(states_arr[:, 6].min()),
        "state_gripper_max": float(states_arr[:, 6].max()),
        "close_frames": int(len(close_indices)),
        "release_frames": int(len(release_indices)),
        "close_first": close_first,
        "close_last": close_last,
        "release_after_close_first": int(release_after_close[0]) if len(release_after_close) else -1,
        "open_close_release": bool(has_cycle),
        "global_mean_min": float(min(global_means)),
        "global_mean_max": float(max(global_means)),
        "wrist_mean_min": float(min(wrist_means)),
        "wrist_mean_max": float(max(wrist_means)),
        "status": status,
        "notes": notes,
        **sample_paths,
    }


def classify(
    *,
    frame_count: int,
    state_dim: int,
    action_dim: int,
    task: str,
    black_frames: int,
    close_count: int,
    release_count: int,
    has_cycle: bool,
    args: argparse.Namespace,
) -> tuple[str, str]:
    bad = []
    review = []
    if state_dim != 7:
        bad.append(f"state_dim={state_dim}")
    if action_dim != 7:
        bad.append(f"action_dim={action_dim}")
    if task != args.expected_task:
        bad.append("task_mismatch")
    if black_frames > 0:
        bad.append(f"black_frames={black_frames}")
    if close_count <= 0:
        bad.append("no_close_frames")
    if release_count <= 0:
        bad.append("no_release_frames")
    if not has_cycle:
        bad.append("no_open_close_release_cycle")
    if frame_count < args.frame_min or frame_count > args.frame_max:
        review.append(f"frame_count_outside_{args.frame_min}_{args.frame_max}")
    if bad:
        return "BAD", ";".join(bad + review)
    if review:
        return "NEED REVIEW", ";".join(review)
    return "GOOD", ""


def export_samples(ref: EpisodeRef, ds: object, *, samples_dir: Path, close_first: int) -> dict[str, str]:
    episode_dir = samples_dir / ref.label
    episode_dir.mkdir(parents=True, exist_ok=True)
    local_indices = {
        "start": 0,
        "mid": len(ref.frame_indices) // 2,
        "end": len(ref.frame_indices) - 1,
    }
    if close_first >= 0:
        local_indices["close"] = close_first

    outputs: dict[str, str] = {}
    for tag, local_index in sorted(local_indices.items(), key=lambda item: item[1]):
        frame = ds[ref.frame_indices[local_index]]
        global_path = episode_dir / f"{tag}_global.jpg"
        wrist_path = episode_dir / f"{tag}_wrist.jpg"
        cv2.imwrite(str(global_path), chw_rgb_to_bgr(image_to_chw_uint8(frame[GLOBAL_IMAGE_KEY])))
        cv2.imwrite(str(wrist_path), chw_rgb_to_bgr(image_to_chw_uint8(frame[WRIST_IMAGE_KEY])))
        outputs[f"{tag}_global"] = str(global_path)
        outputs[f"{tag}_wrist"] = str(wrist_path)
    return outputs


def chw_rgb_to_bgr(chw: np.ndarray) -> np.ndarray:
    rgb = np.moveaxis(chw, 0, -1)
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_report(path: Path, rows: list[dict[str, object]], *, args: argparse.Namespace, table_path: Path) -> None:
    counts: dict[str, int] = {}
    for row in rows:
        counts[str(row["status"])] = counts.get(str(row["status"]), 0) + 1
    total_frames = sum(int(row["frame_count"]) for row in rows)
    with path.open("w") as handle:
        handle.write(f"# {args.position} Single-Cube Sanity Report\n\n")
        handle.write(f"- episodes: {len(rows)}\n")
        handle.write(f"- total frames: {total_frames}\n")
        handle.write(f"- status counts: {counts}\n")
        handle.write(f"- table: `{table_path}`\n")
        handle.write("- training: NO\n")
        handle.write("- real actions sent by this check: NO\n")


def print_summary(rows: list[dict[str, object]], *, out_dir: Path, table_path: Path) -> None:
    print(f"report_dir={out_dir}")
    print(f"episode_table={table_path}")
    print(f"episodes={len(rows)}")
    print(f"total_frames={sum(int(row['frame_count']) for row in rows)}")
    for row in rows:
        print(
            "{label} status={status} frames={frame_count} grip={gmin:.4f}..{gmax:.4f} "
            "close={close_frames} release={release_frames} cycle={cycle} black={black_frames} notes={notes}".format(
                label=row["label"],
                status=row["status"],
                frame_count=row["frame_count"],
                gmin=float(row["action_gripper_min"]),
                gmax=float(row["action_gripper_max"]),
                close_frames=row["close_frames"],
                release_frames=row["release_frames"],
                cycle=row["open_close_release"],
                black_frames=row["black_frames"],
                notes=row["notes"],
            )
        )


if __name__ == "__main__":
    raise SystemExit(main())
