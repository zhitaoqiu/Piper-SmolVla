#!/usr/bin/env python3
"""Replay ground-truth actions from a training demo on real hardware.

Reads an episode from the dataset, sends each action through the same
validation / limiter / gripper pipeline as the full policy rollout, and
reports per-joint tracking error.

No policy inference, no training, no data collection.
"""

from __future__ import annotations

import _bootstrap  # noqa: F401

import argparse
import math
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np

from piper_smolvla.hardware import OfficialPiperSdkBackend, PiperHardwareConfig
from piper_smolvla.rollout_preview import ControlCommand, TerminalRolloutControl
from piper_smolvla.rollout_runtime import (
    ACTION_J2_MAX,
    ACTION_J2_MIN,
    ACTION_SMOOTH_ALPHA,
    J2_DELTA_STOP_RAD,
    J2_DELTA_WARN_RAD,
    PIPER_GRIPPER_MAX_M,
    JOINT_CLAMP_RAD,
    JOINT_LIMIT_STOP_RAD,
    WRIST_DELTA_RATIO,
    WRIST_FREEZE_J2,
    RolloutActionLimiter,
    RolloutSafetyConfig,
    fmt_vec,
)
from piper_smolvla.schema import PIPER_JOINT_ORDER
from piper_smolvla.validation import validate_action, LimitValidationError


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Replay ground-truth demo actions on real hardware.")
    p.add_argument("--dataset", default="data/two_obj_language_16_clean")
    p.add_argument("--episode", type=int, default=0, help="Episode index to replay")
    p.add_argument("--task-index", type=int, default=0, help="task_index filter (0=blue, 1=green)")
    p.add_argument("--can-port", default="can0")
    p.add_argument("--velocity-pct", type=int, default=15)
    p.add_argument("--max-frames", type=int, default=40, help="Number of demo frames to replay")
    p.add_argument("--rate-hz", type=float, default=5.0)
    p.add_argument("--max-delta-rad", type=float, default=0.06,
                   help="Per-step arm-joint delta limit for replay (looser than policy default)")
    p.add_argument("--replay-wrist-delta", type=float, default=0.03,
                   help="Per-step wrist-joint delta limit for replay")
    p.add_argument("--max-delta-gripper-m", type=float, default=0.004)
    p.add_argument("--action-smooth", type=float, default=ACTION_SMOOTH_ALPHA)
    p.add_argument("--disable-wrist-freeze", action="store_true")
    p.add_argument("--skip-home", action="store_true",
                   help="Skip homing to demo start position before replay")
    p.add_argument("--home-tolerance-rad", type=float, default=0.02,
                   help="Convergence tolerance for homing phase")
    p.add_argument("--home-max-steps", type=int, default=200,
                   help="Max steps for homing phase")
    p.add_argument("--save-rollout", action="store_true")
    p.add_argument("--output-dir", default="outputs/replays")
    p.add_argument("--allow-hardware-action", action="store_true")
    p.add_argument("--confirm-replay", default="", help="Must be the literal string REPLAY_DEMO")
    p.add_argument("--dry-run", action="store_true",
                   help="Run pipeline (read state, limiter, log) without sending robot commands.")
    return p.parse_args()


@dataclass
class ReplayFrame:
    frame: int
    demo_action: tuple[float, ...]
    current_state: tuple[float, ...]
    sent_target: tuple[float, ...]
    wrist_frozen: bool
    clamp_joints: list[int]
    grip_clamped: bool


def main() -> int:
    args = parse_args()

    if not args.dry_run:
        if not args.allow_hardware_action:
            raise SystemExit("ERROR: --allow-hardware-action is required.")
        if args.confirm_replay != "REPLAY_DEMO":
            raise SystemExit("ERROR: --confirm-replay must be REPLAY_DEMO.")

    # ── load demo actions ─────────────────────────────────────────────────
    print(f"\n[1/4] loading demo episode {args.episode} (task_index={args.task_index})...")
    from lerobot.datasets import LeRobotDataset

    ds_root = Path(args.dataset)
    ds_name = ds_root.name
    ds = LeRobotDataset(f"piper/{ds_name}", root=str(ds_root), tolerance_s=0.5)
    hf = ds.hf_dataset

    ep_idx = np.array(hf["episode_index"])
    ti = np.array(hf["task_index"])

    # filter to matching episodes
    candidates = sorted(set(ep_idx[ti == args.task_index]))
    if args.episode >= len(candidates):
        raise SystemExit(f"Episode {args.episode} out of range; task_index={args.task_index} has {len(candidates)} episodes")
    ep = candidates[args.episode]
    mask = ep_idx == ep
    frames_in_ep = int(mask.sum())
    indices = np.where(mask)[0]
    replay_count = min(args.max_frames, frames_in_ep)

    demo_actions = [tuple(float(v) for v in hf[int(idx)]["action"]) for idx in indices[:replay_count]]
    demo_actions = [tuple(max(0.0, a[1]) if i == 1 else v for i, v in enumerate(a)) for a in demo_actions]
    demo_states = [tuple(float(v) for v in hf[int(idx)]["observation.state"]) for idx in indices[:replay_count]]
    print(f"       episode {ep}: {frames_in_ep} total frames, replaying first {replay_count}")
    print(f"       demo j2 range: {min(a[1] for a in demo_actions):.4f} ~ {max(a[1] for a in demo_actions):.4f}")
    print(f"       demo gripper range: {min(a[6] for a in demo_actions):.4f} ~ {max(a[6] for a in demo_actions):.4f}")
    # find close frame in demo
    demo_close_frame = next((i for i, a in enumerate(demo_actions) if a[6] < 0.085), -1)
    if demo_close_frame >= 0:
        print(f"       demo gripper close @ frame {demo_close_frame}  grip={demo_actions[demo_close_frame][6]:.4f}")

    # ── hardware setup ────────────────────────────────────────────────────
    print("\n[2/4] creating hardware handles...")
    arm_delta = args.max_delta_rad
    wrist_delta = args.replay_wrist_delta

    demo_start = list(demo_states[0])
    demo_start[1] = max(0.0, demo_start[1])  # clamp j2 >= 0 to pass joint-limit validation
    print(f"       demo start state (raw): {[round(v, 4) for v in demo_states[0]]}")
    print(f"       demo start state (clamped): {[round(v, 4) for v in demo_start]}")

    hw_cfg = PiperHardwareConfig(
        can_port=args.can_port,
        enable_on_connect=False,
        disable_on_disconnect=False,
        call_master_slave_config=False,
        velocity_pct=args.velocity_pct,
    )
    backend = OfficialPiperSdkBackend(hw_cfg)

    action_limiter = RolloutActionLimiter(
        RolloutSafetyConfig(
            arm_delta_rad=arm_delta,
            wrist_delta_rad=wrist_delta,
            gripper_delta_m=args.max_delta_gripper_m,
            action_smooth_alpha=args.action_smooth,
            wrist_freeze_enabled=not args.disable_wrist_freeze,
        )
    )

    print("=" * 70)
    print("DEMO ACTION REPLAY")
    if args.dry_run:
        print("DRY-RUN — NO REAL ACTIONS SENT")
    else:
        print("REAL HARDWARE ACTION")
    print(f"episode:    {ep}  (task_index={args.task_index})")
    print(f"frames:     {replay_count}")
    print(f"rate:       {args.rate_hz} Hz")
    print(f"delta arm:  {arm_delta:.4f} rad  wrist: {wrist_delta:.4f} rad")
    print(f"delta grip: {args.max_delta_gripper_m} m")
    print(f"velocity:   {args.velocity_pct}%")
    print(f"ema alpha:  {args.action_smooth}")
    print(f"wrist freeze @ J2>{WRIST_FREEZE_J2}: {not args.disable_wrist_freeze}")
    print(f"j2 guard:   hard=[{ACTION_J2_MIN},{ACTION_J2_MAX}]  delta_warn={J2_DELTA_WARN_RAD}  delta_stop={J2_DELTA_STOP_RAD}")
    print(f"can:        {args.can_port}")
    print("=" * 70)

    # ── output setup ──────────────────────────────────────────────────────
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / f"replay_{ts}.csv"
    records: list[ReplayFrame] = []

    # ── main loop ─────────────────────────────────────────────────────────
    last_smoothed: np.ndarray | None = None
    last_sent_j2: float | None = None
    stop_reason = "max_frames"
    frame_idx = 0
    paused = False

    terminal_control = TerminalRolloutControl()

    print("\nControls:")
    print("  SPACE / P   pause | R / ENTER  resume | Q  quit | Ctrl+C  emergency stop\n")

    try:
        backend.connect()
        print(f"\n[3/4] CAN connected: {args.can_port}")

        state = backend.read_state()
        current_state = tuple(float(v) for v in state)
        print(f"       initial state: {[round(v, 4) for v in current_state]}")
        for i, v in enumerate(current_state):
            if not math.isfinite(v):
                print(f"FATAL: state[{i}] is non-finite: {v}")
                stop_reason = "non_finite_initial_state"
                break

        if stop_reason != "max_frames":
            return 1

        # ── home to demo start position ───────────────────────────────────
        if not args.skip_home:
            print(f"\n[home] moving to demo start position...")
            print(f"       target: {[round(v, 4) for v in demo_start]}")
            target = np.array(demo_start, dtype=np.float64)
            home_step = 0
            _home_limiter = RolloutActionLimiter(
                RolloutSafetyConfig(
                    arm_delta_rad=0.03,
                    wrist_delta_rad=0.012,
                    gripper_delta_m=args.max_delta_gripper_m,
                    action_smooth_alpha=0.0,
                    wrist_freeze_enabled=False,
                )
            )
            while home_step < args.home_max_steps:
                state = backend.read_state()
                current = np.array([float(v) for v in state], dtype=np.float64)
                errors = target - current
                joint_done = all(abs(errors[i]) < args.home_tolerance_rad for i in range(6))
                grip_done = abs(errors[6]) < args.max_delta_gripper_m
                if joint_done and grip_done:
                    print(f"       homed at step {home_step}: {[round(v, 4) for v in current]}\n")
                    break

                # use limiter to compute safe step toward target
                limited = _home_limiter.limit(
                    current_state=tuple(current.tolist()),
                    raw_action=tuple(target.tolist()),
                    last_smoothed=None,
                )
                sent = backend.write_action(limited.sent_target)
                err_str = " ".join(f"{errors[i]:+.4f}" for i in range(6))
                home_step += 1
                time.sleep(0.05)

            if home_step >= args.home_max_steps:
                errors = target - np.array([float(v) for v in backend.read_state()], dtype=np.float64)
                max_err = max(abs(e) for e in errors)
                print(f"       [WARN] homing did not converge: max_err={max_err:.4f} after {args.home_max_steps} steps\n")

        print(f"\n[4/4] replaying {replay_count} frames @ {args.rate_hz} Hz...\n")
        header = (
            f"{'frame':>5s} {'demo_j2':>8s} {'sent_j2':>8s} {'obs_j2':>8s} {'err_j2':>8s} "
            f"{'freeze':>5s} {'grip_d':>8s} {'grip_s':>8s} {'grip_o':>8s} {'grip_e':>8s}"
        )
        print(header)
        print("-" * 100)

        interval = 1.0 / max(1.0, args.rate_hz)

        for demo_idx in range(replay_count):
            loop_start = time.monotonic()

            # ── terminal control ──────────────────────────────────────────
            cmd = terminal_control.poll()
            if cmd == ControlCommand.QUIT:
                print(f"\nuser quit at frame {frame_idx}")
                stop_reason = "user_quit"
                break
            if cmd == ControlCommand.PAUSE or paused:
                if not paused:
                    print(f"\n  PAUSED at frame {frame_idx} — R/ENTER resume, Q quit")
                    paused = True
                while paused:
                    cmd = terminal_control.poll()
                    if cmd == ControlCommand.RESUME:
                        paused = False
                        last_smoothed = None
                        last_sent_j2 = None
                        print("  RESUMED — limiter state reset, no action skipped")
                        break
                    if cmd == ControlCommand.QUIT:
                        print(f"\nuser quit while paused at frame {frame_idx}")
                        stop_reason = "user_quit"
                        break
                    time.sleep(0.03)
                if stop_reason != "max_frames":
                    break
                continue

            demo_action = demo_actions[demo_idx]

            # read current state
            try:
                state = backend.read_state()
                current_state = tuple(float(v) for v in state)
            except Exception as e:
                print(f"\nFATAL: state read failed at frame {frame_idx}: {e}")
                stop_reason = "state_read_failure"
                break

            # validate state
            state_finite = all(math.isfinite(v) for v in current_state)
            if not state_finite:
                print(f"\nFATAL: non-finite state at frame {frame_idx}")
                stop_reason = "non_finite_state"
                break

            # J2 range check on demo action (informational)
            raw_j2 = demo_action[1]
            if raw_j2 < ACTION_J2_MIN or raw_j2 > ACTION_J2_MAX:
                print(f"  [INFO] demo action J2={raw_j2:.4f} outside soft guard "
                      f"[{ACTION_J2_MIN},{ACTION_J2_MAX}] — frame {frame_idx}")

            # pass through limiter (same pipeline as policy rollout)
            limited = action_limiter.limit(
                current_state=current_state,
                raw_action=demo_action,
                last_smoothed=last_smoothed,
            )
            sent_target = limited.sent_target

            # J2 delta guard on sent target
            sent_j2 = sent_target[1]
            if last_sent_j2 is not None and frame_idx >= 2:
                j2_delta = abs(sent_j2 - last_sent_j2)
                if j2_delta > J2_DELTA_STOP_RAD:
                    print(f"\n  [STOP] J2 delta too fast: "
                          f"|sent_j2 {last_sent_j2:.4f} -> {sent_j2:.4f}| = {j2_delta:.4f} > {J2_DELTA_STOP_RAD}")
                    stop_reason = "j2_delta_too_fast"
                    break
                if j2_delta > J2_DELTA_WARN_RAD:
                    print(f"  [WARN] J2 delta large: "
                          f"|{last_sent_j2:.4f} -> {sent_j2:.4f}| = {j2_delta:.4f} > {J2_DELTA_WARN_RAD}")
            last_sent_j2 = sent_j2

            # joint limit check
            if np.any(np.abs(sent_target[:6]) > JOINT_LIMIT_STOP_RAD):
                print(f"\n  [STOP] Joint limit violation: {fmt_vec(sent_target)}")
                stop_reason = "joint_limit"
                break

            # send or dry-run
            if args.dry_run:
                sent = tuple(sent_target.tolist())
            else:
                sent = backend.write_action(sent_target)

            last_smoothed = sent_target.copy()

            records.append(ReplayFrame(
                frame=frame_idx,
                demo_action=demo_action,
                current_state=current_state,
                sent_target=sent,
                wrist_frozen=limited.wrist_frozen,
                clamp_joints=limited.clamp_joints,
                grip_clamped=limited.grip_clamped,
            ))

            # per-frame print
            state_arr = np.asarray(current_state, dtype=np.float32)
            err_j2 = sent[1] - state_arr[1]
            err_grip = sent[6] - state_arr[6]
            freeze_str = "FRZ" if limited.wrist_frozen else "·"
            print(f"{frame_idx:5d} {demo_action[1]:8.4f} {sent[1]:8.4f} {state_arr[1]:8.4f} {err_j2:8.4f} "
                  f"{freeze_str:>5s} {demo_action[6]:8.4f} {sent[6]:8.4f} {state_arr[6]:8.4f} {err_grip:8.4f}")

            frame_idx += 1

            elapsed = time.monotonic() - loop_start
            if elapsed < interval:
                time.sleep(interval - elapsed)

    except KeyboardInterrupt:
        print(f"\n\nstopped by user at frame {frame_idx}")
        stop_reason = "keyboard_interrupt"
    except LimitValidationError as e:
        print(f"\nFATAL: joint limit violation at frame {frame_idx}: {e}")
        stop_reason = "limit_violation"
    except Exception as e:
        print(f"\nFATAL: unexpected error at frame {frame_idx}: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        stop_reason = f"exception_{type(e).__name__}"
    finally:
        terminal_control.restore()
        try:
            backend.disconnect()
        except Exception as exc:
            print(f"CAN disconnect error: {exc}")
        print("hardware released (NOT disabled)")

    # ── save CSV ──────────────────────────────────────────────────────────
    if records and args.save_rollout:
        import csv
        with open(csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(
                ["frame"]
                + [f"demo_{name}" for name in PIPER_JOINT_ORDER]
                + [f"state_{name}" for name in PIPER_JOINT_ORDER]
                + [f"sent_{name}" for name in PIPER_JOINT_ORDER]
                + ["wrist_frozen", "clamp_joints", "grip_clamped"]
            )
            for r in records:
                writer.writerow(
                    [r.frame]
                    + list(r.demo_action)
                    + list(r.current_state)
                    + list(r.sent_target)
                    + [1 if r.wrist_frozen else 0,
                       ",".join(str(j) for j in r.clamp_joints) if r.clamp_joints else "",
                       1 if r.grip_clamped else 0]
                )
        print(f"\nCSV saved: {csv_path}")

    # ── tracking error report ─────────────────────────────────────────────
    print(f"\n{'=' * 70}")
    print("REPLAY SUMMARY")
    print(f"  stop reason:        {stop_reason}")
    print(f"  frames executed:    {len(records)}")
    if records:
        demo = np.array([r.demo_action for r in records])
        state_obs = np.array([r.current_state for r in records])
        sent_arr = np.array([r.sent_target for r in records])

        # per-joint tracking error: |sent - observed|
        tracking_errors = np.abs(sent_arr - state_obs)
        print(f"\n  per-joint tracking error (|sent - observed|):")
        print(f"  {'joint':>8s} {'mean_err':>10s} {'max_err':>10s} {'rmse':>10s}")
        for j, name in enumerate(PIPER_JOINT_ORDER):
            mean_e = tracking_errors[:, j].mean()
            max_e = tracking_errors[:, j].max()
            rmse = np.sqrt((tracking_errors[:, j] ** 2).mean())
            unit = " m" if j == 6 else " rad"
            print(f"  {name:>8s} {mean_e:10.4f}{unit} {max_e:10.4f}{unit} {rmse:10.4f}{unit}")

        # gripper analysis
        grip_demo = demo[:, 6]
        grip_sent = sent_arr[:, 6]
        grip_obs = state_obs[:, 6]
        grip_close_in_demo = (grip_demo < 0.085).any()
        grip_close_in_sent = (grip_sent < 0.085).any()
        grip_close_in_obs = (grip_obs < 0.085).any()
        print(f"\n  gripper close in demo: {grip_close_in_demo}")
        print(f"  gripper close in sent: {grip_close_in_sent}")
        print(f"  gripper close in obs:  {grip_close_in_obs}")
        if grip_close_in_obs:
            close_obs_frame = int(np.where(grip_obs < 0.085)[0][0])
            print(f"  gripper close frame (obs): {close_obs_frame}")

        # delta clamp summary
        clamped_frames = sum(1 for r in records if r.clamp_joints or r.grip_clamped)
        frozen_frames = sum(1 for r in records if r.wrist_frozen)
        print(f"\n  wrist frozen frames:     {frozen_frames}")
        print(f"  frames with delta clamp: {clamped_frames}")

        if stop_reason == "max_frames":
            total_demo = np.sqrt(((demo[:, :6] - demo[0, :6]) ** 2).sum(axis=1)).max()
            total_obs = np.sqrt(((state_obs[:, :6] - state_obs[0, :6]) ** 2).sum(axis=1)).max()
            print(f"\n  total arm displacement [demo]: {total_demo:.4f} rad")
            print(f"  total arm displacement [obs]:  {total_obs:.4f} rad")

    if args.dry_run:
        print(f"\n  DRY-RUN — NO REAL ACTIONS SENT")
    else:
        print(f"\n  REAL ACTIONS SENT: {'YES' if records else 'NO'}")
    print(f"  TRAINING: NO")
    print(f"  ACT PROJECT MODIFIED: NO")
    print(f"{'=' * 70}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
