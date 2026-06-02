#!/usr/bin/env python3
"""Sanity check two-object language-conditioned demos."""
import _bootstrap  # noqa: F401

import sys
from pathlib import Path

import cv2
import numpy as np

from piper_smolvla.collection import image_to_chw_uint8
from piper_smolvla.schema import STATE_KEY, ACTION_KEY, GLOBAL_IMAGE_KEY, WRIST_IMAGE_KEY


def _to_bgr_uint8(img: np.ndarray) -> np.ndarray:
    """Convert dataset image (float32 [0,1] CHW or HWC uint8) to BGR uint8 for imwrite."""
    arr = np.asarray(img)
    if arr.dtype == np.float32 or arr.dtype == np.float64:
        arr = (arr * 255).clip(0, 255).astype(np.uint8)
    elif arr.dtype != np.uint8:
        arr = arr.astype(np.uint8)
    if arr.shape[0] == 3 and arr.shape[-1] != 3:
        arr = np.moveaxis(arr, 0, -1)
    if arr.shape[-1] == 3:
        arr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
    return arr


def check_demo(label: str, path: str) -> dict:
    r = {"label": label, "path": path, "errors": []}

    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    ds = LeRobotDataset(repo_id="check", root=Path(path))
    r["total_frames"] = len(ds)
    r["num_episodes"] = int(getattr(ds, "num_episodes", 0) or 0)

    states = []
    actions = []
    global_means = []
    wrist_means = []
    task_str = None
    black_frames = 0

    for i in range(len(ds)):
        f = ds[i]
        if i == 0:
            task_str = str(f.get("task", "MISSING"))
            r["task"] = task_str

        s = np.asarray(f[STATE_KEY])
        a = np.asarray(f[ACTION_KEY])
        g = image_to_chw_uint8(f[GLOBAL_IMAGE_KEY])
        w = image_to_chw_uint8(f[WRIST_IMAGE_KEY])

        states.append(s)
        actions.append(a)

        gm = float(g.mean())
        wm = float(w.mean())
        global_means.append(gm)
        wrist_means.append(wm)
        if gm < 5.0 or wm < 5.0:
            black_frames += 1

    states = np.array(states)
    actions = np.array(actions)
    grip = actions[:, 6]

    r["state_gripper_min"] = float(states[:, 6].min())
    r["state_gripper_max"] = float(states[:, 6].max())
    r["action_gripper_min"] = float(grip.min())
    r["action_gripper_max"] = float(grip.max())

    close_idxs = np.where(grip < 0.07)[0]
    open_idxs = np.where(grip > 0.09)[0]
    release_idxs = np.where(grip > 0.095)[0]
    r["close_first"] = int(close_idxs[0]) if len(close_idxs) > 0 else -1
    r["open_first"] = int(open_idxs[0]) if len(open_idxs) > 0 else -1
    r["release_first"] = int(release_idxs[0]) if len(release_idxs) > 0 else -1
    r["close_count"] = len(close_idxs)
    r["open_count"] = len(open_idxs)
    r["release_count"] = len(release_idxs)
    r["black_frames"] = black_frames
    r["global_mean_min"] = float(min(global_means))
    r["global_mean_max"] = float(max(global_means))
    r["wrist_mean_min"] = float(min(wrist_means))
    r["wrist_mean_max"] = float(max(wrist_means))

    # open -> close -> release trend
    has_trend = False
    for thr in [0.07, 0.072, 0.075]:
        c = np.where(grip < thr)[0]
        rel = np.where(grip > 0.095)[0]
        if len(c) > 0 and len(rel) > 0:
            pre_open = np.where(grip[:c[0]] > 0.09)[0]
            post_release = rel[rel > c[-1]] if len(c) > 0 else np.array([])
            if len(pre_open) > 0 and len(post_release) > 0:
                has_trend = True
                break
    r["has_open_close_release"] = has_trend

    # save sample frames
    samples_dir = Path(path) / "sanity_samples"
    samples_dir.mkdir(exist_ok=True)

    total = len(ds)
    sample_idxs = [0, total // 2, total - 1]
    if len(close_idxs) > 0:
        sample_idxs.insert(1, int(close_idxs[0]))
    sample_idxs = sorted(set(sample_idxs))
    sample_labels = {0: "start", total // 2: "mid", total - 1: "end"}
    if len(close_idxs) > 0:
        sample_labels[int(close_idxs[0])] = "close"

    r["samples"] = []
    for si in sample_idxs:
        tag = sample_labels.get(si, f"frame_{si}")
        f = ds[si]
        g_rgb = np.asarray(f[GLOBAL_IMAGE_KEY])
        w_rgb = np.asarray(f[WRIST_IMAGE_KEY])

        g_bgr = _to_bgr_uint8(g_rgb)
        w_bgr = _to_bgr_uint8(w_rgb)

        g_path = samples_dir / f"{tag}_global.jpg"
        w_path = samples_dir / f"{tag}_wrist.jpg"
        cv2.imwrite(str(g_path), g_bgr)
        cv2.imwrite(str(w_path), w_bgr)
        r["samples"].append((tag, str(g_path), str(w_path)))

    # verdict
    verdicts = []
    if not task_str or "green" in task_str.lower():
        expected_color = "green"
    else:
        expected_color = "blue"

    if r["total_frames"] < 10:
        verdicts.append("BAD DEMO: too few frames")
    if black_frames > len(ds) * 0.1:
        verdicts.append("BAD DEMO: >10% black frames")
    if r["close_count"] == 0:
        verdicts.append("BAD DEMO: no close detected")
    if r["release_count"] == 0:
        verdicts.append("BAD DEMO: no release detected")
    r["verdict"] = "; ".join(verdicts) if verdicts else "OK (NEED REVIEW for object color via manual inspection)"

    return r


def main() -> int:
    green = check_demo("GREEN", "data/smolvla_two_object_left_green_right_blue_green1")
    blue = check_demo("BLUE", "data/smolvla_two_object_left_green_right_blue_blue1")

    for r in [green, blue]:
        print(f"===== {r['label']} DEMO =====")
        print(f"  path: {r['path']}")
        print(f"  task: {r['task']}")
        print(f"  frames: {r['total_frames']}")
        print(f"  episodes: {r['num_episodes']}")
        print(f"  state_gripper: {r['state_gripper_min']:.6f} ~ {r['state_gripper_max']:.6f}")
        print(f"  action_gripper: {r['action_gripper_min']:.6f} ~ {r['action_gripper_max']:.6f}")
        print(f"  close_first: frame {r['close_first']} (count={r['close_count']})")
        print(f"  open_first: frame {r['open_first']} (count={r['open_count']})")
        print(f"  release_first: frame {r['release_first']} (count={r['release_count']})")
        print(f"  open->close->release: {r['has_open_close_release']}")
        print(f"  global_mean: {r['global_mean_min']:.1f} ~ {r['global_mean_max']:.1f}")
        print(f"  wrist_mean: {r['wrist_mean_min']:.1f} ~ {r['wrist_mean_max']:.1f}")
        print(f"  black_frames: {r['black_frames']}")
        print(f"  verdict: {r['verdict']}")
        print(f"  samples:")
        for tag, gp, wp in r['samples']:
            print(f"    {tag}: {gp}")
            print(f"    {tag}: {wp}")
        print()

    print("===== MANUAL CHECKS REQUIRED =====")
    print("1. Open sanity_samples/start_global.jpg in each demo dir -- confirm operator arm not visible yet")
    print("2. Open sanity_samples/close_global.jpg -- confirm gripper approaching correct color object")
    print("3. GREEN demo must show green object being grabbed (left side)")
    print("4. BLUE demo must show blue object being grabbed (right side)")
    print("5. Check wrist samples show clear grip view")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
