#!/usr/bin/env python3
"""Evaluate SmolVLA model on its own training demos — three diagnostic questions.

Q1: Does the model replicate training demo actions?
Q2: Does the model learn gripper open → close → release?
Q3: Does the model use the prompt to distinguish green vs blue?

No hardware, no training, no data collection.
"""

from __future__ import annotations

import _bootstrap  # noqa: F401

import argparse
import time
from pathlib import Path

import numpy as np

from piper_smolvla.policy_io import load_lerobot_policy, prepare_policy_batch, select_policy_action_with_options
from piper_smolvla.schema import PIPER_JOINT_ORDER, STATE_KEY

BLUE_TASK = "Pick up the blue object and put it into the box."
GREEN_TASK = "Pick up the green object and put it into the box."


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Training-set evaluation with prompt discrimination test.")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--dataset", default="data/two_obj_language_16_clean")
    p.add_argument("--device", default="cuda")
    p.add_argument("--cross-prompt-stride", type=int, default=5,
                   help="Evaluate cross-prompt on every Nth frame (0=skip)")
    return p.parse_args()


def run_episode(ds, ep_idx_all, ti_all, ep, task_str: str, policy) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Run policy on every frame of one episode with the given task prompt.

    Returns (gt_actions, pred_actions, pred_raw_j2).
    """
    mask = ep_idx_all == ep
    indices = np.where(mask)[0]
    n = len(indices)

    gt = np.zeros((n, 7), dtype=np.float32)
    pred = np.zeros((n, 7), dtype=np.float32)
    raw_j2 = np.zeros(n, dtype=np.float32)

    for i, idx in enumerate(indices):
        frame = ds[int(idx)]
        gt[i] = [float(v) for v in frame["action"]]
        obs_state = tuple(float(v) for v in frame["observation.state"])

        observation = {
            STATE_KEY: obs_state,
            "observation.images.global_rgb": frame["observation.images.global_rgb"],
            "observation.images.wrist_rgb": frame["observation.images.wrist_rgb"],
            "task": task_str,
        }
        batch = prepare_policy_batch(observation)
        raw = select_policy_action_with_options(policy, batch, validate_limits=False)
        pred[i] = raw
        raw_j2[i] = raw[1]

    return gt, pred, raw_j2


def main() -> int:
    args = parse_args()

    print("=" * 70)
    print("TRAINING SET EVALUATION")
    print(f"checkpoint: {args.checkpoint}")
    print(f"dataset:    {args.dataset}")
    print("=" * 70)

    # ── load dataset ──────────────────────────────────────────────────────
    print("\n[1/3] loading dataset...")
    from lerobot.datasets import LeRobotDataset

    ds_root = Path(args.dataset)
    ds_name = ds_root.name
    ds = LeRobotDataset(f"piper/{ds_name}", root=str(ds_root), tolerance_s=0.5)
    hf = ds.hf_dataset

    ep_idx_all = np.array(hf["episode_index"])
    ti_all = np.array(hf["task_index"])
    unique_eps = sorted(set(ep_idx_all))

    blue_eps = [e for e in unique_eps if int(ti_all[np.where(ep_idx_all == e)[0][0]]) == 0]
    green_eps = [e for e in unique_eps if int(ti_all[np.where(ep_idx_all == e)[0][0]]) == 1]
    print(f"       blue episodes:  {blue_eps}")
    print(f"       green episodes: {green_eps}")

    # ── load policy ───────────────────────────────────────────────────────
    print("\n[2/3] loading policy...")
    policy = load_lerobot_policy(args.checkpoint, ds_meta=ds.meta, device=args.device)
    print(f"       policy: {type(policy.policy).__name__}  device={args.device}")

    # ═══════════════════════════════════════════════════════════════════════
    # Q1 + Q2: per-episode action replication + gripper cycle
    # ═══════════════════════════════════════════════════════════════════════
    print(f"\n[3/3] Q1+Q2: action replication + gripper cycle...")
    t0 = time.perf_counter()

    all_results: list[dict] = []

    for ep in unique_eps:
        ti_val = int(ti_all[np.where(ep_idx_all == ep)[0][0]])
        task_str = BLUE_TASK if ti_val == 0 else GREEN_TASK
        gt, pred, pred_j2 = run_episode(ds, ep_idx_all, ti_all, ep, task_str, policy)

        n = len(gt)
        errors = pred - gt
        mae = np.abs(errors).mean(axis=0)
        rmse = np.sqrt((errors ** 2).mean(axis=0))
        j2_corr = np.corrcoef(pred_j2, gt[:, 1])[0, 1] if n > 2 else float("nan")

        # gripper phase analysis
        gt_grip = gt[:, 6]
        pred_grip = pred[:, 6]
        # open phase: frames where gt grip > 0.095 (open)
        # close phase: frames where gt grip < 0.085 (close)
        # release phase: frames where gt grip returns > 0.09 after close
        gt_open_mask = gt_grip > 0.095
        gt_close_mask = gt_grip < 0.085
        gt_close_idx = np.where(gt_close_mask)[0]
        gt_first_close = int(gt_close_idx[0]) if len(gt_close_idx) > 0 else -1
        # release: after first close, does gt grip go back above 0.09?
        if gt_first_close > 0:
            post_close = gt_grip[gt_first_close:]
            gt_release_mask = post_close > 0.09
            gt_releases = gt_release_mask.any()
        else:
            gt_releases = False

        pred_close_idx = np.where(pred_grip < 0.085)[0]
        pred_first_close = int(pred_close_idx[0]) if len(pred_close_idx) > 0 else -1

        # gripper MAE per phase
        grip_open_mae = np.abs(pred_grip[gt_open_mask] - gt_grip[gt_open_mask]).mean() if gt_open_mask.any() else float("nan")
        grip_close_mae = np.abs(pred_grip[gt_close_mask] - gt_grip[gt_close_mask]).mean() if gt_close_mask.any() else float("nan")

        all_results.append({
            "ep": ep, "ti": ti_val, "n": n,
            "mae": mae, "rmse": rmse,
            "j2_corr": j2_corr,
            "grip_open_mae": grip_open_mae, "grip_close_mae": grip_close_mae,
            "gt_close": gt_first_close, "pred_close": pred_first_close,
            "gt_releases": gt_releases,
            "gt_grip": gt_grip, "pred_grip": pred_grip,
        })

        status = (
            f"  ep {ep:2d} ti={ti_val} {n:3d}f  "
            f"j2_mae={mae[1]:.4f}  j2_corr={j2_corr:+.3f}  "
            f"grip_mae={mae[6]:.4f}  grip_open_mae={grip_open_mae:.4f}  grip_close_mae={grip_close_mae:.4f}  "
            f"gt_close={gt_first_close:3d}  pred_close={pred_first_close}"
        )
        if gt_releases:
            status += "  [release]"
        print(status)

    elapsed = time.perf_counter() - t0
    print(f"\n       {len(all_results)} episodes in {elapsed:.1f}s")

    # ── Q1 summary ────────────────────────────────────────────────────────
    all_errors = np.concatenate([r["mae"].reshape(1, -1) for r in all_results], axis=0)
    # Actually, compute frame-level MAE
    all_e_frames = []
    for r in all_results:
        # reconstruct errors: we have mae but need frame-level for proper aggregate
        pass

    print(f"\n{'─' * 70}")
    print("Q1: ACTION REPLICATION")
    print(f"{'─' * 70}")
    print(f"  per-episode (mean over 8 blue + 8 green):")
    print(f"  {'joint':>8s}  {'mae ± std':>16s}  {'rmse ± std':>16s}")
    for ti_label, eps in [("blue", blue_eps), ("green", green_eps)]:
        eps_results = [r for r in all_results if r["ep"] in eps]
        for j, name in enumerate(PIPER_JOINT_ORDER):
            unit = "m" if j == 6 else "rad"
            maes = [r["mae"][j] for r in eps_results]
            rmses = [r["rmse"][j] for r in eps_results]
            mae_m, mae_s = np.mean(maes), np.std(maes)
            rmse_m, rmse_s = np.mean(rmses), np.std(rmses)
            print(f"  [{ti_label:>5s}] {name:>8s}  {mae_m:8.4f}±{mae_s:.4f}{unit}  {rmse_m:8.4f}±{rmse_s:.4f}{unit}")

    j2_corrs = [r["j2_corr"] for r in all_results if not np.isnan(r["j2_corr"])]
    print(f"\n  j2 trajectory correlation: mean={np.mean(j2_corrs):+.3f}  "
          f"min={np.min(j2_corrs):+.3f}  max={np.max(j2_corrs):+.3f}")

    # ── Q2 summary ────────────────────────────────────────────────────────
    print(f"\n{'─' * 70}")
    print("Q2: GRIPPER OPEN → CLOSE → RELEASE CYCLE")
    print(f"{'─' * 70}")
    for ti_label, eps in [("blue", blue_eps), ("green", green_eps)]:
        eps_results = [r for r in all_results if r["ep"] in eps]
        n_gt_close = sum(1 for r in eps_results if r["gt_close"] >= 0)
        n_pred_close = sum(1 for r in eps_results if r["pred_close"] >= 0)
        n_gt_release = sum(1 for r in eps_results if r["gt_releases"])
        print(f"  [{ti_label:>5s}] gt closes: {n_gt_close}/{len(eps_results)}  "
              f"pred closes: {n_pred_close}/{len(eps_results)}  "
              f"gt releases: {n_gt_release}/{len(eps_results)}")
        # close timing error
        timing_errors = []
        for r in eps_results:
            if r["gt_close"] >= 0 and r["pred_close"] >= 0:
                timing_errors.append(r["pred_close"] - r["gt_close"])
        if timing_errors:
            print(f"         close timing Δ (pred-gt): mean={np.mean(timing_errors):.1f}f  "
                  f"min={np.min(timing_errors)}f  max={np.max(timing_errors)}f")

        grip_open_maes = [r["grip_open_mae"] for r in eps_results if not np.isnan(r["grip_open_mae"])]
        grip_close_maes = [r["grip_close_mae"] for r in eps_results if not np.isnan(r["grip_close_mae"])]
        print(f"         gripper MAE — open phase: {np.mean(grip_open_maes):.4f}m  "
              f"close phase: {np.mean(grip_close_maes):.4f}m")

    # ═══════════════════════════════════════════════════════════════════════
    # Q3: cross-prompt discrimination test
    # ═══════════════════════════════════════════════════════════════════════
    if args.cross_prompt_stride > 0:
        print(f"\n{'─' * 70}")
        print("Q3: PROMPT DISCRIMINATION (cross-prompt test)")
        print(f"{'─' * 70}")

        t0 = time.perf_counter()
        cross_results: list[dict] = []

        for ep in unique_eps:
            ti_val = int(ti_all[np.where(ep_idx_all == ep)[0][0]])
            correct_task = BLUE_TASK if ti_val == 0 else GREEN_TASK
            wrong_task = GREEN_TASK if ti_val == 0 else BLUE_TASK

            mask = ep_idx_all == ep
            indices = np.where(mask)[0]
            sample_indices = indices[::args.cross_prompt_stride]
            if len(sample_indices) < 3:
                continue

            correct_preds = np.zeros((len(sample_indices), 7), dtype=np.float32)
            wrong_preds = np.zeros((len(sample_indices), 7), dtype=np.float32)
            gt_acts = np.zeros((len(sample_indices), 7), dtype=np.float32)

            for i, idx in enumerate(sample_indices):
                frame = ds[int(idx)]
                obs_state = tuple(float(v) for v in frame["observation.state"])
                gt_acts[i] = [float(v) for v in frame["action"]]

                obs = {
                    STATE_KEY: obs_state,
                    "observation.images.global_rgb": frame["observation.images.global_rgb"],
                    "observation.images.wrist_rgb": frame["observation.images.wrist_rgb"],
                }
                batch_correct = prepare_policy_batch({**obs, "task": correct_task})
                batch_wrong = prepare_policy_batch({**obs, "task": wrong_task})

                correct_preds[i] = select_policy_action_with_options(policy, batch_correct, validate_limits=False)
                wrong_preds[i] = select_policy_action_with_options(policy, batch_wrong, validate_limits=False)

            # difference between correct-prompt and wrong-prompt output
            prompt_diff = np.abs(correct_preds - wrong_preds).mean(axis=0)
            # is correct-prompt closer to GT than wrong-prompt?
            correct_err = np.abs(correct_preds - gt_acts).mean(axis=0)
            wrong_err = np.abs(wrong_preds - gt_acts).mean(axis=0)

            cross_results.append({
                "ep": ep, "ti": ti_val, "n_samples": len(sample_indices),
                "prompt_diff": prompt_diff,
                "correct_mae": correct_err,
                "wrong_mae": wrong_err,
            })

        elapsed_q3 = time.perf_counter() - t0
        print(f"       {len(cross_results)} episodes, every {args.cross_prompt_stride}th frame  ({elapsed_q3:.1f}s)")

        # aggregate
        all_prompt_diffs = np.array([r["prompt_diff"] for r in cross_results])
        all_correct_mae = np.array([r["correct_mae"] for r in cross_results])
        all_wrong_mae = np.array([r["wrong_mae"] for r in cross_results])

        print(f"\n  per-joint |correct_prompt - wrong_prompt| (mean over episodes):")
        print(f"  {'joint':>8s}  {'prompt_diff':>14s}  {'correct_mae':>14s}  {'wrong_mae':>14s}  {'Δ':>10s}")
        for j, name in enumerate(PIPER_JOINT_ORDER):
            unit = "m" if j == 6 else "rad"
            pdiff = all_prompt_diffs[:, j].mean()
            cmae = all_correct_mae[:, j].mean()
            wmae = all_wrong_mae[:, j].mean()
            delta = wmae - cmae  # positive = correct prompt is better
            print(f"  {name:>8s}  {pdiff:10.4f}{unit}    {cmae:10.4f}{unit}    {wmae:10.4f}{unit}    {delta:+10.4f}")

        # key diagnostic: does using the WRONG prompt change the gripper prediction?
        grip_prompt_diff = all_prompt_diffs[:, 6]
        print(f"\n  gripper prompt sensitivity: mean_diff={grip_prompt_diff.mean():.4f}m  "
              f"max_diff={grip_prompt_diff.max():.4f}m")
        if grip_prompt_diff.mean() < 0.001:
            print("  → model does NOT meaningfully differentiate prompts for gripper")
        else:
            print("  → model DOES respond to prompt in gripper output")

        # per-task breakdown
        for ti_label, eps in [("blue", blue_eps), ("green", green_eps)]:
            cross_eps = [r for r in cross_results if r["ep"] in eps]
            if cross_eps:
                diffs = np.array([r["prompt_diff"] for r in cross_eps])
                print(f"\n  [{ti_label:>5s}] n={len(cross_eps)}  "
                      f"grip_prompt_diff={diffs[:,6].mean():.4f}m  "
                      f"j2_prompt_diff={diffs[:,1].mean():.4f}rad")

    # ── final ─────────────────────────────────────────────────────────────
    print(f"\n{'=' * 70}")
    print(f"  TRAINING: NO")
    print(f"  ACT PROJECT MODIFIED: NO")
    print(f"{'=' * 70}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
