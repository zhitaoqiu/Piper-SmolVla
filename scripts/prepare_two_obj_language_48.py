#!/usr/bin/env python3
"""Prepare and audit the 48-demo two-object language dataset.

This is an offline data-prep script. It never connects to Piper hardware, never
trains a policy, and never modifies the ACT project. It creates:

- data/two_obj_language_48_all_clean
- data/two_obj_language_strict_paired_clean
- a frozen raw-data manifest and quality report
"""

from __future__ import annotations

import _bootstrap  # noqa: F401

import argparse
import csv
import hashlib
import json
import os
import shutil
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

from piper_smolvla.collection import CollectionConfig, create_lerobot_dataset, image_to_chw_uint8
from piper_smolvla.schema import ACTION_KEY, GLOBAL_IMAGE_KEY, STATE_KEY, WRIST_IMAGE_KEY
from piper_smolvla.validation import validate_action, validate_state

SOURCES = (
    "data/two_obj_paired_LgRb",
    "data/two_obj_paired_LbRg",
    "data/two_obj_paired_LgRb_b2",
    "data/two_obj_paired_LbRg_b2",
    "data/two_obj_paired_LgRb_b3",
    "data/two_obj_paired_LbRg_b3",
)
CRITICAL_JOINTS = (0, 1, 2, 3, 5)
CRITICAL_NAMES = ("j1", "j2", "j3", "j4", "j6")


@dataclass
class EpisodeAudit:
    global_ep: int
    source: str
    source_ep: int
    source_start: int
    source_end: int
    pair_id: str
    pair_local_index: int
    scene: str
    task: str
    task_color: str
    frame_count: int
    state_dim: int
    action_dim: int
    has_global_rgb: bool
    has_wrist_rgb: bool
    black_frames: int
    action_gripper_min: float
    action_gripper_max: float
    state_gripper_min: float
    state_gripper_max: float
    close_frames: int
    release_frames: int
    open_close_release: bool
    automated_status: str
    review_notes: str


@dataclass
class PairAudit:
    pair_id: str
    source: str
    scene: str
    green_global_ep: int
    blue_global_ep: int
    start_global_mean_abs_diff: float
    early30_diff: np.ndarray
    early60_diff: np.ndarray
    strict_status: str
    notes: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Freeze, merge, and audit the 48-demo two-object language data.")
    parser.add_argument("--all-output", default="data/two_obj_language_48_all_clean")
    parser.add_argument("--strict-output", default="data/two_obj_language_strict_paired_clean")
    parser.add_argument("--report-dir", default="")
    parser.add_argument("--fps", type=int, default=10)
    parser.add_argument("--strict-start-image-threshold", type=float, default=12.0)
    parser.add_argument("--strict-mean-diff-threshold", type=float, default=0.03)
    parser.add_argument("--strict-max-diff-threshold", type=float, default=0.06)
    parser.add_argument("--overwrite-outputs", action="store_true")
    parser.add_argument("--no-samples", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    force_writable_hf_cache()

    source_roots = [Path(src) for src in SOURCES]
    missing = [str(root) for root in source_roots if not root.exists()]
    if missing:
        raise SystemExit(f"missing source datasets: {missing}")

    all_output = Path(args.all_output)
    strict_output = Path(args.strict_output)
    prepare_output_dir(all_output, overwrite=args.overwrite_outputs)
    prepare_output_dir(strict_output, overwrite=args.overwrite_outputs)

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_dir = Path(args.report_dir) if args.report_dir else Path("outputs/data_quality") / f"two_obj_language_48_{stamp}"
    report_dir.mkdir(parents=True, exist_ok=True)
    samples_dir = report_dir / "samples"
    if not args.no_samples:
        samples_dir.mkdir(parents=True, exist_ok=True)

    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    loaded = [(root, LeRobotDataset(repo_id=f"piper/{root.name}", root=root, tolerance_s=0.5)) for root in source_roots]
    episodes = audit_episodes(loaded, samples_dir=None if args.no_samples else samples_dir)
    pairs = audit_pairs(loaded, episodes, args)

    assert_expected_48(episodes, pairs)
    write_freeze_manifest(source_roots, episodes, pairs, report_dir)

    strict_pair_ids = {pair.pair_id for pair in pairs if pair.strict_status == "STRICT"}
    write_merged_dataset(all_output, loaded, episodes, keep_pair_ids=None, fps=args.fps)
    write_merged_dataset(strict_output, loaded, episodes, keep_pair_ids=strict_pair_ids, fps=args.fps)

    write_reports(report_dir, episodes, pairs, all_output, strict_output)

    print(f"all_clean_dataset={all_output}")
    print(f"strict_paired_dataset={strict_output}")
    print(f"report_dir={report_dir}")
    print(f"episodes={len(episodes)}")
    print(f"pairs={len(pairs)}")
    print(f"strict_pairs={len(strict_pair_ids)} strict_episodes={len(strict_pair_ids) * 2}")
    print("TRAINING: NO")
    print("REAL ACTIONS SENT: NO")
    print("ACT PROJECT MODIFIED: NO")
    return 0


def audit_episodes(
    loaded: list[tuple[Path, Any]],
    *,
    samples_dir: Path | None,
) -> list[EpisodeAudit]:
    episodes: list[EpisodeAudit] = []
    global_ep = 0
    for root, ds in loaded:
        ep_indices = np.asarray(ds.hf_dataset["episode_index"])
        scene = infer_scene(root)
        unique_eps = sorted(int(ep) for ep in set(ep_indices.tolist()))
        for local_order, ep in enumerate(unique_eps):
            frame_indices = np.where(ep_indices == ep)[0]
            pair_local_index = local_order // 2
            pair_id = f"{scene}_{root.name}_pair{pair_local_index:02d}"
            episode = audit_one_episode(
                ds,
                root=root,
                ep=ep,
                global_ep=global_ep,
                frame_indices=frame_indices,
                pair_id=pair_id,
                pair_local_index=pair_local_index,
                scene=scene,
                samples_dir=samples_dir,
            )
            episodes.append(episode)
            global_ep += 1
    return episodes


def audit_one_episode(
    ds: Any,
    *,
    root: Path,
    ep: int,
    global_ep: int,
    frame_indices: np.ndarray,
    pair_id: str,
    pair_local_index: int,
    scene: str,
    samples_dir: Path | None,
) -> EpisodeAudit:
    states = []
    actions = []
    black_frames = 0
    has_global = True
    has_wrist = True
    task = ""
    state_dim = -1
    action_dim = -1

    for offset, idx in enumerate(frame_indices):
        frame = ds[int(idx)]
        if offset == 0:
            task = str(frame.get("task", "")).strip()
            state_dim = len(frame[STATE_KEY])
            action_dim = len(frame[ACTION_KEY])
            has_global = GLOBAL_IMAGE_KEY in frame
            has_wrist = WRIST_IMAGE_KEY in frame
        state = np.asarray(validate_state(frame[STATE_KEY]), dtype=np.float64)
        action = np.asarray(validate_action(frame[ACTION_KEY]), dtype=np.float64)
        states.append(state)
        actions.append(action)
        global_mean = float(image_to_chw_uint8(frame[GLOBAL_IMAGE_KEY]).mean())
        wrist_mean = float(image_to_chw_uint8(frame[WRIST_IMAGE_KEY]).mean())
        if global_mean < 5.0 or wrist_mean < 5.0:
            black_frames += 1

    states_arr = np.asarray(states)
    actions_arr = np.asarray(actions)
    grip = actions_arr[:, 6]
    state_grip = states_arr[:, 6]
    close_mask = grip < 0.07
    release_mask = grip > 0.09
    open_close_release = has_open_close_release(grip)
    task_color = color_from_task(task)
    notes: list[str] = []
    status = "GOOD"
    if not task_color:
        status = "BAD"
        notes.append("task_missing_green_blue")
    if state_dim != 7 or action_dim != 7:
        status = "BAD"
        notes.append("bad_state_or_action_dim")
    if not has_global or not has_wrist:
        status = "BAD"
        notes.append("missing_camera_key")
    if black_frames > 0:
        status = "BAD"
        notes.append(f"black_frames={black_frames}")
    if int(close_mask.sum()) == 0:
        status = "BAD"
        notes.append("no_close_frames")
    if int(release_mask.sum()) == 0:
        status = "BAD"
        notes.append("no_release_open_frames")
    if not open_close_release:
        status = "NEED REVIEW" if status == "GOOD" else status
        notes.append("no_clear_open_close_release")
    if status == "GOOD":
        notes.append("actual_grasp_color_not_encoded_manual_sample_review_recommended")

    if samples_dir is not None:
        save_episode_samples(ds, frame_indices, grip, samples_dir / f"ep_{global_ep:03d}_{scene}_{task_color or 'unknown'}")

    return EpisodeAudit(
        global_ep=global_ep,
        source=str(root),
        source_ep=ep,
        source_start=int(frame_indices[0]),
        source_end=int(frame_indices[-1]) + 1,
        pair_id=pair_id,
        pair_local_index=pair_local_index,
        scene=scene,
        task=task,
        task_color=task_color or "unknown",
        frame_count=len(frame_indices),
        state_dim=state_dim,
        action_dim=action_dim,
        has_global_rgb=has_global,
        has_wrist_rgb=has_wrist,
        black_frames=black_frames,
        action_gripper_min=float(grip.min()),
        action_gripper_max=float(grip.max()),
        state_gripper_min=float(state_grip.min()),
        state_gripper_max=float(state_grip.max()),
        close_frames=int(close_mask.sum()),
        release_frames=int(release_mask.sum()),
        open_close_release=open_close_release,
        automated_status=status,
        review_notes=";".join(notes),
    )


def audit_pairs(loaded: list[tuple[Path, Any]], episodes: list[EpisodeAudit], args: argparse.Namespace) -> list[PairAudit]:
    by_pair: dict[str, list[EpisodeAudit]] = {}
    source_to_ds = {str(root): ds for root, ds in loaded}
    for ep in episodes:
        by_pair.setdefault(ep.pair_id, []).append(ep)

    pairs: list[PairAudit] = []
    for pair_id, eps in sorted(by_pair.items(), key=lambda item: min(ep.global_ep for ep in item[1])):
        eps_sorted = sorted(eps, key=lambda ep: ep.global_ep)
        green = next((ep for ep in eps_sorted if ep.task_color == "green"), None)
        blue = next((ep for ep in eps_sorted if ep.task_color == "blue"), None)
        notes = []
        if len(eps_sorted) != 2 or green is None or blue is None:
            notes.append("pair_not_exactly_green_blue")
            dummy = np.full(len(CRITICAL_JOINTS), np.nan)
            pairs.append(
                PairAudit(pair_id, eps_sorted[0].source, eps_sorted[0].scene, -1, -1, np.nan, dummy, dummy, "WEAK", ";".join(notes))
            )
            continue
        ds = source_to_ds[green.source]
        start_diff = start_image_diff(ds, green, blue)
        early30 = early_action_diff(ds, green, blue, horizon=30)
        early60 = early_action_diff(ds, green, blue, horizon=60)
        both_good = green.automated_status == "GOOD" and blue.automated_status == "GOOD"
        mean_diff = float(max(np.nanmean(early30), np.nanmean(early60)))
        max_diff = float(max(np.nanmax(early30), np.nanmax(early60)))
        strict = (
            both_good
            and start_diff <= args.strict_start_image_threshold
            and mean_diff >= args.strict_mean_diff_threshold
            and max_diff >= args.strict_max_diff_threshold
        )
        if not both_good:
            notes.append("one_or_both_episodes_not_good")
        if start_diff > args.strict_start_image_threshold:
            notes.append(f"start_image_diff>{args.strict_start_image_threshold}")
        if mean_diff < args.strict_mean_diff_threshold:
            notes.append(f"mean_action_diff<{args.strict_mean_diff_threshold}")
        if max_diff < args.strict_max_diff_threshold:
            notes.append(f"max_action_diff<{args.strict_max_diff_threshold}")
        pairs.append(
            PairAudit(
                pair_id=pair_id,
                source=green.source,
                scene=green.scene,
                green_global_ep=green.global_ep,
                blue_global_ep=blue.global_ep,
                start_global_mean_abs_diff=start_diff,
                early30_diff=early30,
                early60_diff=early60,
                strict_status="STRICT" if strict else "WEAK",
                notes=";".join(notes),
            )
        )
    return pairs


def write_merged_dataset(
    output: Path,
    loaded: list[tuple[Path, Any]],
    episodes: list[EpisodeAudit],
    *,
    keep_pair_ids: set[str] | None,
    fps: int,
) -> None:
    if output.exists():
        raise RuntimeError(f"output already exists: {output}")
    source_to_ds = {str(root): ds for root, ds in loaded}
    out = create_lerobot_dataset(
        root=output,
        repo_id=f"piper/{output.name}",
        config=CollectionConfig(fps=fps, use_videos=True),
    )
    try:
        for ep in episodes:
            if keep_pair_ids is not None and ep.pair_id not in keep_pair_ids:
                continue
            ds = source_to_ds[ep.source]
            for idx in range(ep.source_start, ep.source_end):
                out.add_frame(standard_frame(ds[idx]))
            out.save_episode()
    finally:
        out.finalize()


def standard_frame(frame: dict[str, Any]) -> dict[str, Any]:
    task = str(frame.get("task", "")).strip()
    if not task:
        raise ValueError("task is required")
    return {
        STATE_KEY: np.asarray(validate_state(frame[STATE_KEY]), dtype=np.float32),
        ACTION_KEY: np.asarray(validate_action(frame[ACTION_KEY]), dtype=np.float32),
        GLOBAL_IMAGE_KEY: image_to_chw_uint8(frame[GLOBAL_IMAGE_KEY]),
        WRIST_IMAGE_KEY: image_to_chw_uint8(frame[WRIST_IMAGE_KEY]),
        "task": task,
    }


def write_reports(report_dir: Path, episodes: list[EpisodeAudit], pairs: list[PairAudit], all_output: Path, strict_output: Path) -> None:
    episode_csv = report_dir / "episode_table.csv"
    pair_csv = report_dir / "pair_table.csv"
    write_episode_csv(episode_csv, episodes)
    write_pair_csv(pair_csv, pairs)

    counts = count_groups(episodes)
    status_counts = count_statuses(episodes)
    strict_pairs = [pair for pair in pairs if pair.strict_status == "STRICT"]
    weak_pairs = [pair for pair in pairs if pair.strict_status == "WEAK"]
    total_frames = sum(ep.frame_count for ep in episodes)
    report = report_dir / "report.md"
    with report.open("w", encoding="utf-8") as f:
        f.write("# Two-Object Language 48 Data Quality Report\n\n")
        f.write(f"Generated: {datetime.now().isoformat()}\n\n")
        f.write("## Outputs\n\n")
        f.write(f"- all48: `{all_output}`\n")
        f.write(f"- strict paired: `{strict_output}`\n")
        f.write(f"- episode table: `{episode_csv}`\n")
        f.write(f"- pair table: `{pair_csv}`\n\n")
        f.write("## Summary\n\n")
        f.write(f"- episodes: {len(episodes)}\n")
        f.write(f"- pairs: {len(pairs)}\n")
        f.write(f"- total frames: {total_frames}\n")
        f.write(f"- strict pairs: {len(strict_pairs)} ({len(strict_pairs) * 2} episodes)\n")
        f.write(f"- weak pairs: {len(weak_pairs)} ({len(weak_pairs) * 2} episodes)\n\n")
        f.write("## Scene / Task Counts\n\n")
        for key in ("LgRb green", "LgRb blue", "LbRg green", "LbRg blue"):
            f.write(f"- {key}: {counts.get(key, 0)}\n")
        f.write("\n## Status Counts\n\n")
        for key, value in sorted(status_counts.items()):
            f.write(f"- {key}: {value}\n")
        f.write("\n## Notes\n\n")
        f.write("- Automated checks verify schema, images, black frames, gripper cycle, and pair action separation.\n")
        f.write("- Actual grasp color is not encoded in metadata; close/release sample images were exported for manual visual review.\n")
        f.write("- GOOD means automated checks passed, not that visual target color has been human-certified.\n\n")
        f.write("## Recommendation\n\n")
        recommend_all = status_counts.get("BAD", 0) == 0 and len(episodes) == 48
        recommend_strict = len(strict_pairs) >= 8
        f.write(f"- recommend_train_all48: {recommend_all}\n")
        f.write(f"- recommend_train_strict_paired_subset: {recommend_strict}\n")
        f.write("- all48 training suggestion: `steps=20000`, `batch_size=1`, `output=outputs/train/smolvla_two_obj_language_48_all_clean_v1`\n")
        f.write("- strict subset training suggestion: `steps=12000`, `batch_size=1`, `output=outputs/train/smolvla_two_obj_language_strict_paired_clean_v1`\n")
        f.write("\nREAL ACTIONS SENT: NO\nTRAINING: NO\nACT PROJECT MODIFIED: NO\n")


def write_episode_csv(path: Path, episodes: list[EpisodeAudit]) -> None:
    fields = list(EpisodeAudit.__dataclass_fields__)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for ep in episodes:
            writer.writerow({field: getattr(ep, field) for field in fields})


def write_pair_csv(path: Path, pairs: list[PairAudit]) -> None:
    fields = [
        "pair_id",
        "source",
        "scene",
        "green_global_ep",
        "blue_global_ep",
        "start_global_mean_abs_diff",
        "early30_j1",
        "early30_j2",
        "early30_j3",
        "early30_j4",
        "early30_j6",
        "early60_j1",
        "early60_j2",
        "early60_j3",
        "early60_j4",
        "early60_j6",
        "strict_status",
        "notes",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for pair in pairs:
            row = {
                "pair_id": pair.pair_id,
                "source": pair.source,
                "scene": pair.scene,
                "green_global_ep": pair.green_global_ep,
                "blue_global_ep": pair.blue_global_ep,
                "start_global_mean_abs_diff": pair.start_global_mean_abs_diff,
                "strict_status": pair.strict_status,
                "notes": pair.notes,
            }
            for prefix, values in (("early30", pair.early30_diff), ("early60", pair.early60_diff)):
                for name, value in zip(CRITICAL_NAMES, values, strict=True):
                    row[f"{prefix}_{name}"] = float(value)
            writer.writerow(row)


def write_freeze_manifest(source_roots: list[Path], episodes: list[EpisodeAudit], pairs: list[PairAudit], report_dir: Path) -> None:
    manifest = {
        "created_at": datetime.now().isoformat(),
        "source_roots": [str(root) for root in source_roots],
        "episodes": len(episodes),
        "pairs": len(pairs),
        "files": [],
        "note": "Raw source datasets were not modified; this manifest records file size, mtime, and sha256.",
    }
    for root in source_roots:
        for path in sorted(root.rglob("*")):
            if path.is_file():
                manifest["files"].append(
                    {
                        "path": str(path),
                        "size": path.stat().st_size,
                        "mtime_ns": path.stat().st_mtime_ns,
                        "sha256": sha256_file(path),
                    }
                )
    manifest_path = Path("data/two_obj_language_48_raw_freeze_manifest.json")
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    shutil.copy2(manifest_path, report_dir / manifest_path.name)


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            h.update(chunk)
    return h.hexdigest()


def assert_expected_48(episodes: list[EpisodeAudit], pairs: list[PairAudit]) -> None:
    if len(episodes) != 48:
        raise RuntimeError(f"expected 48 episodes, got {len(episodes)}")
    if len(pairs) != 24:
        raise RuntimeError(f"expected 24 pairs, got {len(pairs)}")
    bad_pairs = [pair.pair_id for pair in pairs if pair.green_global_ep < 0 or pair.blue_global_ep < 0]
    if bad_pairs:
        raise RuntimeError(f"non green/blue pairs found: {bad_pairs}")


def has_open_close_release(grip: np.ndarray) -> bool:
    close = np.where(grip < 0.07)[0]
    if len(close) == 0:
        return False
    pre_open = np.where(grip[: close[0]] > 0.09)[0]
    post_release = np.where(grip[close[-1] + 1 :] > 0.09)[0]
    return len(pre_open) > 0 and len(post_release) > 0


def save_episode_samples(ds: Any, frame_indices: np.ndarray, grip: np.ndarray, out_dir: Path) -> None:
    import cv2

    out_dir.mkdir(parents=True, exist_ok=True)
    close = np.where(grip < 0.07)[0]
    release_after = np.array([], dtype=np.int64)
    if len(close) > 0:
        release_after = np.where(grip[close[-1] + 1 :] > 0.09)[0] + close[-1] + 1
    samples = {
        "start": 0,
        "middle": len(frame_indices) // 2,
        "end": len(frame_indices) - 1,
    }
    if len(close) > 0:
        samples["close"] = int(close[0])
    if len(release_after) > 0:
        samples["release"] = int(release_after[0])
    for label, local_idx in samples.items():
        frame = ds[int(frame_indices[local_idx])]
        for key, suffix in ((GLOBAL_IMAGE_KEY, "global"), (WRIST_IMAGE_KEY, "wrist")):
            rgb = chw_to_hwc_uint8(frame[key])
            bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
            cv2.imwrite(str(out_dir / f"{label}_{suffix}.jpg"), bgr)


def start_image_diff(ds: Any, green: EpisodeAudit, blue: EpisodeAudit) -> float:
    g = chw_to_hwc_uint8(ds[green.source_start][GLOBAL_IMAGE_KEY]).astype(np.float32)
    b = chw_to_hwc_uint8(ds[blue.source_start][GLOBAL_IMAGE_KEY]).astype(np.float32)
    return float(np.mean(np.abs(g - b)))


def early_action_diff(ds: Any, green: EpisodeAudit, blue: EpisodeAudit, *, horizon: int) -> np.ndarray:
    length = min(horizon, green.frame_count, blue.frame_count)
    diffs = []
    for offset in range(length):
        g = np.asarray(ds[green.source_start + offset][ACTION_KEY], dtype=np.float64)
        b = np.asarray(ds[blue.source_start + offset][ACTION_KEY], dtype=np.float64)
        diffs.append(np.abs(g[list(CRITICAL_JOINTS)] - b[list(CRITICAL_JOINTS)]))
    return np.mean(diffs, axis=0) if diffs else np.full(len(CRITICAL_JOINTS), np.nan)


def chw_to_hwc_uint8(image: Any) -> np.ndarray:
    arr = image_to_chw_uint8(image)
    return np.moveaxis(arr, 0, -1)


def color_from_task(task: str) -> str:
    lower = task.lower()
    if "green" in lower:
        return "green"
    if "blue" in lower:
        return "blue"
    return ""


def infer_scene(root: Path) -> str:
    name = root.name
    if "LgRb" in name:
        return "LgRb"
    if "LbRg" in name:
        return "LbRg"
    return "unknown"


def count_groups(episodes: list[EpisodeAudit]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for ep in episodes:
        key = f"{ep.scene} {ep.task_color}"
        counts[key] = counts.get(key, 0) + 1
    return counts


def count_statuses(episodes: list[EpisodeAudit]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for ep in episodes:
        counts[ep.automated_status] = counts.get(ep.automated_status, 0) + 1
    return counts


def prepare_output_dir(path: Path, *, overwrite: bool) -> None:
    if path.exists():
        if not overwrite:
            raise SystemExit(f"output already exists: {path}; pass --overwrite-outputs to replace generated outputs")
        shutil.rmtree(path)


def force_writable_hf_cache() -> None:
    default = "/tmp/piper_smolvla_hf"
    os.environ.setdefault("HF_HOME", default)
    os.environ.setdefault("HUGGINGFACE_HUB_CACHE", f"{default}/hub")
    os.environ.setdefault("HF_DATASETS_CACHE", "/tmp/piper_smolvla_datasets_cache")


if __name__ == "__main__":
    raise SystemExit(main())
