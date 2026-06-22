#!/usr/bin/env python3
"""SmolVLA full policy rollout on real Piper hardware.

Safety mechanisms adapted from the ACT Piper deployment:
  - Per-joint delta clamp: arm (J1-J3) vs wrist (J4-J6) with ACT ratio (0.4x)
  - Wrist freeze: lock J4-J6 when J2 > WRIST_FREEZE_J2 (1.45 rad)
  - EMA smoothing: alpha=0.5 for sent targets
  - Joint-limit safety stop: halt if any joint exceeds 3.0 rad
  - Action sanity: halt if policy J2 outside [-0.1, 1.8]
  - Stagnation detection: halt if 20 consecutive steps < 0.0008 rad before 70%
  - Ready stop: halt if J2 > 1.65 for 5+ consecutive steps after 70%

Usage:
  PYTHONPATH=src python scripts/deploy_smolvla.py \\
    --checkpoint <path> --task "<prompt>" \\
    --can-port can0 \\
    --rate-hz 20 --max-frames 360 \\
    --save-rollout --save-final-images \\
    --allow-hardware-action --confirm-policy-rollout ROLLOUT \\
    2>&1 | tee logs_smolvla_lgrb_blue_full_rollout_001.txt
"""

from __future__ import annotations

import _bootstrap  # noqa: F401

import argparse
import math
import time
from datetime import datetime
from pathlib import Path

import numpy as np

from piper_smolvla.hardware import OfficialPiperSdkBackend, PiperHardwareConfig
from piper_smolvla.policy_io import load_lerobot_policy, prepare_policy_batch, select_policy_action_with_options
from piper_smolvla.cameras import (
    DEFAULT_CAMERA_FPS,
    DEFAULT_GLOBAL_CAMERA,
    DEFAULT_ROLLOUT_RATE_HZ,
    DEFAULT_WRIST_CAMERA,
    RealCameraConfig,
    RealCameraSource,
)
from piper_smolvla.rollout_preview import (
    ControlCommand,
    RolloutPreview,
    TerminalRolloutControl,
    is_quit_key,
    is_start_or_pause_key,
    wait_for_space_start,
    wait_while_paused,
)
from piper_smolvla.rollout_runtime import (
    ACTION_J2_MAX,
    ACTION_J2_MIN,
    ACTION_J2_SOFT_MARGIN,
    ACTION_SMOOTH_ALPHA,
    GRIPPER_CLOSE_ONSET,
    GRIPPER_OPEN_M,
    J2_DELTA_STOP_RAD,
    J2_DELTA_WARN_RAD,
    JOINT_LIMIT_STOP_RAD,
    READY_COUNT_MIN,
    READY_J2,
    STAGNATION_STEPS,
    STAGNATION_THRESHOLD,
    WRIST_DELTA_RATIO,
    WRIST_FREEZE_J2,
    RolloutActionLimiter,
    RolloutSafetyConfig,
    fmt_vec,
    max_abs_diff,
    require_policy_rollout_confirmation,
    reset_policy_runtime,
    save_image,
    write_rollout_csv,
)
from piper_smolvla.schema import (
    GLOBAL_IMAGE_KEY,
    PIPER_JOINT_ORDER,
    START_GUARD_GRIPPER_OPEN_MIN_M,
    STATE_KEY,
    VERIFIED_START_QPOS,
    WRIST_IMAGE_KEY,
)
from piper_smolvla.validation import LimitValidationError

DEMO_MATCHED_JOINT_DELTA_RAD = (0.035, 0.075, 0.040, 0.006, 0.030, 0.006)
DEMO_MATCHED_GRIPPER_DELTA_M = 0.006


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Full policy rollout on real Piper hardware.")

    # ── required ──────────────────────────────────────────────────────────
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--task", default="Pick up the cube and put it into the box.")

    # ── hardware ──────────────────────────────────────────────────────────
    p.add_argument("--can-port", default="can0")
    p.add_argument("--global-camera", default=DEFAULT_GLOBAL_CAMERA)
    p.add_argument("--wrist-camera", default=DEFAULT_WRIST_CAMERA)
    p.add_argument("--camera-fps", type=int, default=DEFAULT_CAMERA_FPS,
                   help="Camera capture FPS; keep aligned with collection.")
    p.add_argument("--wrist-auto-exposure", type=int, default=None,
                   help="Wrist camera auto exposure: 1=on, 0=off.")
    p.add_argument("--wrist-exposure", type=int, default=None,
                   help="Wrist camera manual exposure value.")
    p.add_argument("--wrist-gain", type=float, default=None,
                   help="Wrist camera gain/ISO; higher is brighter and noisier.")
    p.add_argument("--wrist-brightness", type=float, default=None,
                   help="Wrist camera brightness offset.")
    p.add_argument("--wrist-power-line", type=int, default=None,
                   help="Wrist camera power line frequency: 1=50Hz, 2=60Hz.")

    # ── rollout control ───────────────────────────────────────────────────
    p.add_argument("--rate-hz", type=float, default=DEFAULT_ROLLOUT_RATE_HZ)
    p.add_argument("--max-frames", type=int, default=360)
    p.add_argument("--control-profile", choices=("safe", "demo_matched"), default="safe",
                   help=(
                       "Deployment limiter profile. 'safe' uses the scalar --max-delta-rad. "
                       "'demo_matched' uses per-joint qpos deltas measured from the clean demos "
                       "so J2/J3 can descend at a sampled-trajectory-like pace while wrist joints stay conservative."
                   ))
    p.add_argument("--max-delta-rad", type=float, default=0.03,
                   help="Safe-profile per-step arm-joint (J1-J3) delta limit; wrist gets 0.4x this")
    p.add_argument("--max-delta-gripper-m", type=float, default=None,
                   help=(
                       "Per-step gripper delta. Defaults to 0.004 for safe profile and "
                       f"{DEMO_MATCHED_GRIPPER_DELTA_M} for demo_matched profile."
                   ))
    p.add_argument("--velocity-pct", type=int, default=25)
    p.add_argument("--action-smooth", type=float, default=ACTION_SMOOTH_ALPHA,
                   help="EMA smoothing factor (0=disable)")
    p.add_argument("--gripper-open-until-frame", type=int, default=0,
                   help="Force the gripper command to stay open until this rollout frame. Default disables the gate.")
    p.add_argument("--gripper-close-confirm-frames", type=int, default=1,
                   help="Require this many consecutive policy close frames before allowing close. Default is immediate.")
    p.add_argument("--gripper-close-j2-min", type=float, default=None,
                   help="If set, hold gripper open until current J2 reaches this value.")
    p.add_argument("--disable-wrist-freeze", action="store_true")
    p.add_argument("--disable-ready-stop", action="store_true")
    p.add_argument("--disable-stagnation-stop", action="store_true")

    # ── dataset ───────────────────────────────────────────────────────────
    p.add_argument("--dataset", default="data/two_obj_language_48_all_clean")

    # ── output ────────────────────────────────────────────────────────────
    p.add_argument("--save-rollout", action="store_true")
    p.add_argument("--save-final-images", action="store_true")
    p.add_argument("--output-dir", default="outputs/rollouts")
    p.add_argument("--no-preview", action="store_true", help="Disable live global/wrist preview window.")
    p.add_argument("--preview-window", default="Piper SmolVLA Deployment | global | wrist")

    # ── safety gates ──────────────────────────────────────────────────────
    p.add_argument("--allow-hardware-action", action="store_true")
    p.add_argument("--confirm-policy-rollout", default="")
    p.add_argument("--dry-run", action="store_true",
                   help="Run full pipeline (cameras, inference, logging) without sending robot commands.")
    p.add_argument("--auto-start", action="store_true",
                   help="Skip the operator wait and start rollout immediately (useful with --dry-run).")
    p.add_argument("--loop", action="store_true",
                   help="Persistent loop mode: load once, then run multiple rollouts with new task prompts.")
    p.add_argument("--disable-start-guard", action="store_true",
                   help="Disable start qpos guard. Not recommended for real rollout.")
    p.add_argument("--start-qpos-tol", type=float, default=0.12,
                   help="Max abs radian difference from dataset/schema start state for J1-J6.")
    p.add_argument("--start-gripper-open-min", type=float, default=START_GUARD_GRIPPER_OPEN_MIN_M,
                   help="Minimum current gripper opening required before rollout.")
    p.add_argument("--start-guard-source", choices=("dataset_mean", "schema"), default="dataset_mean",
                   help="Reference qpos for start guard.")

    return p.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if args.dry_run:
        return
    try:
        require_policy_rollout_confirmation(
            allow_hardware_action=args.allow_hardware_action,
            confirm_policy_rollout=args.confirm_policy_rollout,
        )
    except PermissionError as exc:
        raise SystemExit(f"ERROR: {exc}") from exc


def main() -> int:
    args = parse_args()
    validate_args(args)

    # httpx crashes on socks:// proxy configured via GNOME; disable all proxies
    import os
    for key in ("HTTP_PROXY", "HTTPS_PROXY", "http_proxy", "https_proxy", "ALL_PROXY", "all_proxy"):
        os.environ[key] = ""
    os.environ["NO_PROXY"] = "*"
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ.setdefault("HF_HOME", str(Path.home() / ".cache" / "huggingface"))

    control = resolve_control_profile(args)
    arm_delta = control["arm_delta_rad"]
    wrist_delta = control["wrist_delta_rad"]
    gripper_delta = control["gripper_delta_m"]
    joint_delta_rad = control["joint_delta_rad"]

    print("=" * 70)
    if args.dry_run:
        print("FULL POLICY ROLLOUT — DRY-RUN (no robot action)")
    else:
        print("FULL POLICY ROLLOUT — REAL HARDWARE")
    print(f"task:       {args.task}")
    print(f"checkpoint: {args.checkpoint}")
    print(f"rate:       {args.rate_hz} Hz")
    print(f"max frames: {args.max_frames}")
    print(f"profile:    {args.control_profile}")
    if joint_delta_rad is None:
        print(f"delta arm:  {arm_delta:.4f} rad  wrist: {wrist_delta:.4f} rad")
    else:
        delta_str = " ".join(f"{name}={value:.4f}" for name, value in zip(PIPER_JOINT_ORDER[:6], joint_delta_rad))
        print(f"delta joint:{delta_str}")
    print(f"delta grip: {gripper_delta} m")
    print(f"velocity:   {args.velocity_pct}%")
    print(f"ema alpha:  {args.action_smooth}")
    print(f"grip gate:  open_until_frame={args.gripper_open_until_frame}  "
          f"confirm_frames={args.gripper_close_confirm_frames}  "
          f"j2_min={args.gripper_close_j2_min}")
    print(f"wrist freeze @ J2>{WRIST_FREEZE_J2}: {not args.disable_wrist_freeze}")
    print(f"ready stop  @ J2>{READY_J2}: {not args.disable_ready_stop}")
    print(f"stagnation stop: {not args.disable_stagnation_stop}")
    print(f"j2 guard:   hard=[{ACTION_J2_MIN},{ACTION_J2_MAX}]  soft_margin={ACTION_J2_SOFT_MARGIN}"
          f"  delta_warn={J2_DELTA_WARN_RAD}  delta_stop={J2_DELTA_STOP_RAD}")
    print(f"start guard: {'disabled' if args.disable_start_guard else args.start_guard_source}"
          f"  tol={args.start_qpos_tol}  gripper_open_min={args.start_gripper_open_min}")
    print(f"can:        {args.can_port}")
    print(f"cameras:    {args.global_camera} / {args.wrist_camera}")
    print("=" * 70)

    # ── load policy ────────────────────────────────────────────────────────
    print("\n[1/5] loading policy...")
    ds_meta = None
    dataset_for_start_guard = None
    if args.dataset:
        from lerobot.datasets.lerobot_dataset import LeRobotDataset
        ds_root = Path(args.dataset)
        ds_name = ds_root.name
        dataset_for_start_guard = LeRobotDataset(f"piper/{ds_name}", root=str(ds_root), tolerance_s=0.5)
        ds_meta = dataset_for_start_guard.meta
        # Patch missing 'names' on image features (older dataset format compat)
        for key, ft in ds_meta.features.items():
            if ft.get("dtype") in ("image", "video") and "names" not in ft:
                ft["names"] = ["channels", "height", "width"]
        print(f"       dataset: {ds_root}")
    policy = load_lerobot_policy(args.checkpoint, ds_meta=ds_meta)
    print(f"       policy loaded: {type(policy.policy).__name__}")

    # ── create hardware handles ──────────────────────────────────────────
    print("\n[2/5] creating hardware handles...")
    cam_cfg = RealCameraConfig(
        allow_hardware_readonly=True,
        global_camera=args.global_camera,
        wrist_camera=args.wrist_camera,
        fps=args.camera_fps,
        wrist_auto_exposure=args.wrist_auto_exposure,
        wrist_exposure_absolute=args.wrist_exposure,
        wrist_gain=args.wrist_gain,
        wrist_brightness=args.wrist_brightness,
        wrist_power_line_frequency=args.wrist_power_line,
    )
    cameras = RealCameraSource(cam_cfg)

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
            gripper_delta_m=gripper_delta,
            joint_delta_rad=joint_delta_rad,
            action_smooth_alpha=args.action_smooth,
            wrist_freeze_enabled=not args.disable_wrist_freeze,
        )
    )

    # ── output setup ───────────────────────────────────────────────────────
    print("\n[3/5] output setup...")
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / f"rollout_{ts}.csv"

    frame_records: list[dict] = []
    frame_idx = 0
    current_state: tuple[float, ...] = ()
    last_images: tuple = (None, None)
    global_img: np.ndarray = np.zeros((1,), dtype=np.uint8)
    wrist_img: np.ndarray = np.zeros((1,), dtype=np.uint8)

    # ── stateful trackers ──────────────────────────────────────────────────
    last_smoothed: np.ndarray | None = None
    last_state: tuple[float, ...] | None = None
    stagnation_count = 0
    ready_count = 0
    stop_reason = "max_frames"
    close_detected = False
    close_frame: int | None = None
    raw_close_count = 0
    gripper_gate_count = 0

    # ── j2 guard trackers ─────────────────────────────────────────────────
    last_sent_j2: float | None = None
    j2_soft_warnings = 0
    j2_hard_violations = 0
    j2_delta_warn_count = 0
    j2_delta_stop_frame: int | None = None

    # Safe fallback: a disabled preview is always safe to .close().
    # The real preview (which calls cv2.namedWindow) is created inside the
    # try block so any X11 hang still triggers finally cleanup.
    preview = RolloutPreview(enabled=False, window_name="")
    terminal_control = TerminalRolloutControl()

    # ═══════════════════════════════════════════════════════════════════════
    # All hardware connect / rollout / release is inside a single try/finally
    # so that Camera + CAN are always released on ANY exit path.
    # ═══════════════════════════════════════════════════════════════════════
    try:
        # ── create preview window (may block on X11) ───────────────────────
        preview = RolloutPreview(enabled=not args.no_preview, window_name=args.preview_window)

        # ── connect cameras ────────────────────────────────────────────────
        cameras.connect()
        print(f"       cameras: global={cameras.resolved_global}  wrist={cameras.resolved_wrist}")

        # ── connect CAN ────────────────────────────────────────────────────
        backend.connect()
        print(f"       CAN connected: {args.can_port}")

        # ── verify initial state ───────────────────────────────────────────
        print("\n[4/5] verifying initial state...")
        images = cameras.read_images()
        global_img = images[GLOBAL_IMAGE_KEY]
        wrist_img = images[WRIST_IMAGE_KEY]
        state = backend.read_state()
        print(f"       state={[round(v, 4) for v in state]}")
        print(f"       global={global_img.shape} mean={global_img.mean():.1f}")
        print(f"       wrist={wrist_img.shape} mean={wrist_img.mean():.1f}")

        for i, v in enumerate(state):
            if not math.isfinite(v):
                print(f"FATAL: state[{i}] ({PIPER_JOINT_ORDER[i]}) is non-finite: {v}")
                stop_reason = "non_finite_initial_state"
        if stop_reason == "non_finite_initial_state":
            print("       state finite: FAILED — aborting before rollout")
        else:
            print("       state finite: OK")

        # ── start qpos guard ───────────────────────────────────────────────
        if stop_reason == "max_frames":
            guard_ok, guard_expected, guard_diff, guard_reason = check_start_guard(
                current_state=state,
                dataset=dataset_for_start_guard,
                source=args.start_guard_source,
                qpos_tol=args.start_qpos_tol,
                gripper_open_min=args.start_gripper_open_min,
                disabled=args.disable_start_guard,
            )
            if args.disable_start_guard:
                print("       start guard: DISABLED")
            else:
                print("       start guard reference="
                      f"{[round(float(v), 4) for v in guard_expected]}")
                print("       start guard diff="
                      f"{[round(float(v), 4) for v in guard_diff]}")
                print(f"       start guard: {'OK' if guard_ok else 'FAILED'} ({guard_reason})")
            if not guard_ok:
                print("       Refusing rollout before any action. Manually align Piper to the recorded start pose.")
                print("       No reset, no return-to-start, and no gripper command was sent.")
                stop_reason = "start_guard_failed"

        # ── rollout loop ───────────────────────────────────────────────────
        if stop_reason == "max_frames":
            current_state = tuple(state)
            last_images = (global_img, wrist_img)

            print(f"\n[5/5] waiting for operator start ({args.max_frames} frames max, {args.rate_hz} Hz)...")
            print("       SPACE starts rollout; SPACE during rollout pauses before the next command\n")

            header_parts = [
                f"{'frame':>5s} {'pred_grip':>10s} {'sent_grip':>10s}",
                f"{'j2':>8s} {'gate':>4s} {'freeze':>5s} {'ready':>5s} {'stag':>4s}",
                f"{'inf_ms':>7s}",
                f"{'j1':>8s} {'j2':>8s} {'j3':>8s} {'j4':>8s} {'j5':>8s} {'j6':>8s}",
            ]
            print(" ".join(header_parts))
            print("-" * 120)

            interval = 1.0 / max(1.0, args.rate_hz)

            if args.auto_start:
                print("\n       --auto-start: skipping operator wait, starting rollout immediately\n")
                reset_policy_runtime(policy)
                last_sent_j2 = None
                print("       policy runtime reset")
                print("       rollout started\n")
            else:
                start_result = wait_for_space_start(
                    cameras=cameras,
                    backend=backend,
                    preview=preview,
                    max_frames=args.max_frames,
                    terminal_control=terminal_control,
                )
                if start_result is None:
                    print("\nrollout cancelled before start")
                    stop_reason = "cancelled_before_start"
                else:
                    current_state, last_images = start_result
                    reset_policy_runtime(policy)
                    last_sent_j2 = None
                    print("       policy runtime reset")
                    print("       rollout started\n")

            while frame_idx < args.max_frames and stop_reason == "max_frames":
                loop_start = time.monotonic()

                # ── read cameras ───────────────────────────────────────────
                t0 = time.perf_counter()
                try:
                    images = cameras.read_images()
                    global_img = images[GLOBAL_IMAGE_KEY]
                    wrist_img = images[WRIST_IMAGE_KEY]
                    last_images = (global_img, wrist_img)
                except Exception as e:
                    print(f"\nFATAL: camera read failed at frame {frame_idx}: {e}")
                    stop_reason = "camera_failure"
                    break
                cam_ms = (time.perf_counter() - t0) * 1000

                # ── read state ─────────────────────────────────────────────
                try:
                    current_state = tuple(backend.read_state())
                except Exception as e:
                    print(f"\nFATAL: state read failed at frame {frame_idx}: {e}")
                    stop_reason = "state_read_failure"
                    break

                # ── validate state ─────────────────────────────────────────
                state_finite = True
                for i, v in enumerate(current_state):
                    if not math.isfinite(v):
                        print(f"\nFATAL: state[{i}] ({PIPER_JOINT_ORDER[i]}) is non-finite: {v}")
                        state_finite = False
                        break
                if not state_finite:
                    stop_reason = "non_finite_state"
                    break

                # ── terminal / preview control ──────────────────────────
                cmd = terminal_control.poll()
                if cmd == ControlCommand.QUIT:
                    print(f"\nuser quit at frame {frame_idx}")
                    stop_reason = "user_quit"
                    break

                if cmd == ControlCommand.PAUSE:
                    pause_status, pause_images = wait_while_paused(
                        cameras=cameras,
                        backend=backend,
                        preview=preview,
                        frame_idx=frame_idx,
                        max_frames=args.max_frames,
                        terminal_control=terminal_control,
                    )
                    if pause_images is not None:
                        last_images = pause_images
                    if pause_status == "quit":
                        print(f"\nuser quit while paused at frame {frame_idx}")
                        stop_reason = "user_quit"
                        break
                    reset_policy_runtime(policy)
                    last_smoothed = None
                    last_sent_j2 = None
                    print("  RESUMED - policy runtime reset, no command sent while paused")
                    continue

                key = preview.show(
                    global_rgb=global_img,
                    wrist_rgb=wrist_img,
                    status=f"RUNNING {frame_idx}/{args.max_frames} - SPACE pause",
                    frame_idx=frame_idx,
                    state=current_state,
                    color=(0, 0, 255),
                )
                if is_quit_key(key):
                    print(f"\nuser quit at frame {frame_idx}")
                    stop_reason = "user_quit"
                    break
                if is_start_or_pause_key(key):
                    pause_status, pause_images = wait_while_paused(
                        cameras=cameras,
                        backend=backend,
                        preview=preview,
                        frame_idx=frame_idx,
                        max_frames=args.max_frames,
                        terminal_control=terminal_control,
                    )
                    if pause_images is not None:
                        last_images = pause_images
                    if pause_status == "quit":
                        print(f"\nuser quit while paused at frame {frame_idx}")
                        stop_reason = "user_quit"
                        break
                    reset_policy_runtime(policy)
                    last_smoothed = None
                    last_sent_j2 = None
                    print("  RESUMED - policy runtime reset, no command sent while paused")
                    continue

                # ── build observation & run inference ──────────────────────
                observation = {
                    STATE_KEY: current_state,
                    GLOBAL_IMAGE_KEY: global_img,
                    WRIST_IMAGE_KEY: wrist_img,
                    "task": args.task,
                }
                batch = prepare_policy_batch(observation)
                t1 = time.perf_counter()
                raw_action = select_policy_action_with_options(policy, batch)
                inf_ms = (time.perf_counter() - t1) * 1000

                # ── NaN/Inf check on raw action ────────────────────────────
                action_finite = True
                for i, v in enumerate(raw_action):
                    if not math.isfinite(v):
                        print(f"\nFATAL: raw_action[{i}] ({PIPER_JOINT_ORDER[i]}) non-finite: {v}")
                        action_finite = False
                        break
                if not action_finite:
                    stop_reason = "non_finite_action"
                    break

                # ── action sanity: J2 range check with soft margin ──────
                raw_j2 = raw_action[1]
                if raw_j2 < ACTION_J2_MIN - ACTION_J2_SOFT_MARGIN or raw_j2 > ACTION_J2_MAX + ACTION_J2_SOFT_MARGIN:
                    j2_hard_violations += 1
                    print(f"\n  [STOP] Action J2 hard violation: raw J2={raw_j2:.4f} "
                          f"(hard [{ACTION_J2_MIN},{ACTION_J2_MAX}]  margin={ACTION_J2_SOFT_MARGIN})")
                    stop_reason = "action_j2_hard_ood"
                    break
                if raw_j2 < ACTION_J2_MIN or raw_j2 > ACTION_J2_MAX:
                    j2_soft_warnings += 1
                    print(f"  [WARN] soft J2: raw={raw_j2:.4f} outside [{ACTION_J2_MIN},{ACTION_J2_MAX}]"
                          f" but within soft margin — frame {frame_idx}")

                gated_action, gripper_gate_active, raw_close_count = apply_gripper_timing_gate(
                    frame_idx=frame_idx,
                    current_state=current_state,
                    raw_action=raw_action,
                    raw_close_count=raw_close_count,
                    open_until_frame=args.gripper_open_until_frame,
                    close_confirm_frames=args.gripper_close_confirm_frames,
                    close_j2_min=args.gripper_close_j2_min,
                )
                if gripper_gate_active:
                    gripper_gate_count += 1

                limited = action_limiter.limit(
                    current_state=current_state,
                    raw_action=gated_action,
                    last_smoothed=last_smoothed,
                )
                sent_target = limited.sent_target

                # ── J2 delta guard (frame-to-frame sent target) ───────────
                sent_j2 = sent_target[1]
                if last_sent_j2 is not None and frame_idx >= 2:
                    j2_delta = abs(sent_j2 - last_sent_j2)
                    if j2_delta > J2_DELTA_STOP_RAD:
                        j2_delta_stop_frame = frame_idx
                        print(f"\n  [STOP] J2 delta too fast: "
                              f"|sent_j2 {last_sent_j2:.4f} -> {sent_j2:.4f}| = {j2_delta:.4f} > {J2_DELTA_STOP_RAD}")
                        stop_reason = "j2_delta_too_fast"
                        break
                    if j2_delta > J2_DELTA_WARN_RAD:
                        j2_delta_warn_count += 1
                        print(f"  [WARN] J2 delta large: "
                              f"|{last_sent_j2:.4f} -> {sent_j2:.4f}| = {j2_delta:.4f} > {J2_DELTA_WARN_RAD}")
                last_sent_j2 = sent_j2

                # ── Step 5: joint limit safety stop ────────────────────────
                if np.any(np.abs(sent_target[:6]) > JOINT_LIMIT_STOP_RAD):
                    print(f"\n  [STOP] Joint limit violation: target={fmt_vec(sent_target)}")
                    stop_reason = "joint_limit"
                    break

                # ── send action ────────────────────────────────────────────
                if args.dry_run:
                    sent = tuple(sent_target.tolist())
                else:
                    sent = backend.write_action(sent_target)

                # ── close detection ────────────────────────────────────────
                if not close_detected and sent[6] < GRIPPER_CLOSE_ONSET:
                    close_detected = True
                    close_frame = frame_idx
                    print(f"\n  [CLOSE] Gripper closing detected at frame {frame_idx}: grip={sent[6]:.4f}")

                # ── update trackers ────────────────────────────────────────
                last_smoothed = sent_target.copy()

                # ── record ─────────────────────────────────────────────────
                state_arr = np.asarray(current_state, dtype=np.float32)
                raw_delta_j = sent_target - state_arr
                record = {
                    "frame": frame_idx,
                    "timestamp": time.time(),
                    "state": current_state,
                    "raw_action": raw_action,
                    "clamped_action": tuple(limited.clamped_action.tolist()),
                    "sent_action": tuple(sent),
                    "sent_delta": tuple(raw_delta_j.tolist()),
                    "wrist_frozen": limited.wrist_frozen,
                    "clamp_joints": limited.clamp_joints,
                    "grip_clamped": limited.grip_clamped,
                    "inf_ms": inf_ms,
                    "cam_ms": cam_ms,
                    "ready_count": ready_count,
                    "stagnation_count": stagnation_count,
                }
                frame_records.append(record)

                # ── print per-frame status ─────────────────────────────────
                grip_status = "CLOSE" if sent[6] < 0.07 else ("OPEN" if sent[6] > 0.09 else "HOLD")
                freeze_str = "FRZ" if limited.wrist_frozen else "·"
                gate_str = "GATE" if gripper_gate_active else "·"
                ready_str = f"R{ready_count}" if ready_count > 0 else "·"
                stag_str = f"S{stagnation_count}" if stagnation_count > 0 else "·"
                j_str = " ".join(f"{sent[i]:8.4f}" for i in range(6))
                print(f"{frame_idx:5d} {raw_action[6]:10.4f} {sent[6]:10.4f} "
                      f"{state_arr[1]:8.4f} {gate_str:>4s} {freeze_str:>5s} {ready_str:>5s} {stag_str:>4s} "
                      f"{inf_ms:7.1f} {j_str}")

                # ── ready stop ─────────────────────────────────────────────
                near_end = frame_idx > args.max_frames * 0.7
                if not args.disable_ready_stop and state_arr[1] > READY_J2 and near_end:
                    ready_count += 1
                else:
                    ready_count = max(0, ready_count - 1)
                if ready_count >= READY_COUNT_MIN:
                    print(f"\n  [DONE] Task complete: J2 > {READY_J2} for {READY_COUNT_MIN} consecutive steps")
                    stop_reason = "ready_stop"
                    break

                # ── stagnation detection ────────────────────────────────────
                if not args.disable_stagnation_stop:
                    state_diff = max_abs_diff(current_state, last_state)
                    if not near_end and last_state is not None and state_diff < STAGNATION_THRESHOLD:
                        stagnation_count += 1
                    else:
                        stagnation_count = 0
                    if stagnation_count >= STAGNATION_STEPS:
                        print(f"\n  [STOP] Stagnation: {STAGNATION_STEPS} steps with diff < {STAGNATION_THRESHOLD} before 70%")
                        print(f"    frame={frame_idx}/{args.max_frames}  state_diff={state_diff:.6f}")
                        stop_reason = "stagnation"
                        break

                last_state = current_state
                frame_idx += 1

                # ── sleep to maintain rate ─────────────────────────────────
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
        # ── save CSV ───────────────────────────────────────────────────────
        if frame_records and args.save_rollout:
            try:
                write_rollout_csv(csv_path, frame_records)
                print(f"\nCSV saved: {csv_path}")
            except Exception as exc:
                print(f"\nCSV save failed: {exc}")

        # ── save final images ──────────────────────────────────────────────
        if args.save_final_images and last_images[0] is not None:
            try:
                global_path = out_dir / f"final_global_{ts}.png"
                wrist_path = out_dir / f"final_wrist_{ts}.png"
                save_image(last_images[0], global_path)
                save_image(last_images[1], wrist_path)
                print(f"final global image: {global_path}")
                print(f"final wrist image:  {wrist_path}")
            except Exception as exc:
                print(f"image save failed: {exc}")

        # ── release hardware ───────────────────────────────────────────────
        terminal_control.restore()
        try:
            preview.close()
        except Exception:
            pass
        try:
            cameras.close()
        except Exception as exc:
            print(f"camera close error: {exc}")
        try:
            backend.disconnect()
        except Exception as exc:
            print(f"CAN disconnect error: {exc}")
        print("hardware released (NOT disabled)")

    # ── summary ────────────────────────────────────────────────────────────
    print(f"\n{'=' * 70}")
    print("ROLLOUT SUMMARY")
    print(f"  stop reason:         {stop_reason}")
    print(f"  frames executed:     {len(frame_records)}")
    if frame_records:
        sent = np.array([r["sent_action"] for r in frame_records])
        raw = np.array([r["raw_action"] for r in frame_records])
        grip = sent[:, 6]
        close_count = int(np.sum(grip < 0.07))
        open_count = int(np.sum(grip > 0.09))
        frozen_count = sum(1 for r in frame_records if r["wrist_frozen"])
        delta_clamped = sum(1 for r in frame_records if r["clamp_joints"] or r["grip_clamped"])
        inf_avg = np.mean([r["inf_ms"] for r in frame_records])
        cam_avg = np.mean([r["cam_ms"] for r in frame_records])

        print(f"  gripper: min={grip.min():.4f} max={grip.max():.4f} mean={grip.mean():.4f}")
        print(f"  close frames: {close_count}  open frames: {open_count}")
        print(f"  wrist frozen frames: {frozen_count}")
        print(f"  gripper gate frames: {gripper_gate_count}")
        print(f"  frames with delta clamp: {delta_clamped}")
        print(f"  close detected: {close_detected} (frame {close_frame})")
        print(f"  avg inference: {inf_avg:.1f} ms  avg camera read: {cam_avg:.1f} ms")
        print(f"  NaN: {bool(np.any(np.isnan(sent)))}  Inf: {bool(np.any(np.isinf(sent)))}")
        print(f"  ── j2 safety ──")
        print(f"  raw soft limit warnings:  {j2_soft_warnings}")
        print(f"  raw hard violations:      {j2_hard_violations}")
        print(f"  sent hard violations:     {sum(1 for r in frame_records if r['sent_action'][1] < ACTION_J2_MIN or r['sent_action'][1] > ACTION_J2_MAX)}")
        print(f"  j2_delta_warn_count:      {j2_delta_warn_count}")
        print(f"  j2_delta_stop_frame:      {j2_delta_stop_frame}")

        print(f"\n  per-joint sent action range:")
        for j, name in enumerate(PIPER_JOINT_ORDER):
            print(f"    {name}: {sent[:, j].min():.4f} ~ {sent[:, j].max():.4f}  "
                  f"(raw: {raw[:, j].min():.4f} ~ {raw[:, j].max():.4f})")

        if close_detected:
            print(f"\n  post-close J2 range: "
                  f"{sent[close_frame:, 1].min():.4f} ~ {sent[close_frame:, 1].max():.4f}")

    if args.dry_run:
        print(f"\n  DRY-RUN — NO REAL ACTIONS SENT")
    else:
        print(f"\n  REAL ACTIONS SENT: {'YES' if frame_records else 'NO'}")
    print(f"  TRAINING: NO")
    print(f"  ACT PROJECT MODIFIED: NO")
    print(f"{'=' * 70}")

    if not args.loop:
        return 0

    print()
    next_task = input(f"Next task (q=quit, Enter=repeat \"{current_task}\"): ").strip()
    if next_task.lower() == 'q':
        print("Exiting loop mode.")
        return 0
    if next_task:
        current_task = next_task
    print(f"\n=== New task: {current_task} ===\n")
    reset_policy_runtime(policy)


def check_start_guard(
    *,
    current_state: tuple[float, ...] | list[float],
    dataset,
    source: str,
    qpos_tol: float,
    gripper_open_min: float,
    disabled: bool,
) -> tuple[bool, np.ndarray, np.ndarray, str]:
    current = np.asarray(current_state, dtype=np.float64)
    expected = resolve_start_guard_reference(dataset=dataset, source=source)
    diff = np.abs(current - expected)

    if disabled:
        return True, expected, diff, "disabled"
    if not np.isfinite(current).all():
        return False, expected, diff, "current_state_non_finite"
    if np.any(diff[:6] > qpos_tol):
        bad = [PIPER_JOINT_ORDER[i] for i in range(6) if diff[i] > qpos_tol]
        return False, expected, diff, "arm_qpos_out_of_start_zone:" + ",".join(bad)
    if float(current[6]) < gripper_open_min:
        return False, expected, diff, f"gripper_not_open:{current[6]:.4f}<{gripper_open_min:.4f}"
    return True, expected, diff, "ok"


def resolve_start_guard_reference(*, dataset, source: str) -> np.ndarray:
    if source == "schema" or dataset is None:
        return np.asarray(VERIFIED_START_QPOS, dtype=np.float64)
    try:
        episode_index = np.asarray(dataset.hf_dataset["episode_index"])
        starts = []
        for ep in sorted(set(int(value) for value in episode_index.tolist())):
            indices = np.where(episode_index == ep)[0]
            if len(indices) == 0:
                continue
            starts.append(np.asarray(dataset[int(indices[0])][STATE_KEY], dtype=np.float64))
        if starts:
            return np.mean(np.stack(starts, axis=0), axis=0)
    except Exception as exc:  # noqa: BLE001
        print(f"       start guard dataset reference failed ({type(exc).__name__}: {exc}); using schema reference")
    return np.asarray(VERIFIED_START_QPOS, dtype=np.float64)


def resolve_control_profile(args: argparse.Namespace) -> dict:
    """Resolve deployment limiter settings without touching hardware.

    The clean single-cube demonstrations have large J2/J3 moves during the
    descent phase, but very small wrist deltas. A scalar clamp like 0.02 rad
    makes the real arm lag far behind the model's absolute qpos targets, so
    demo_matched uses conservative per-joint deltas derived from the demo
    distribution instead of one global arm value.
    """

    if args.control_profile == "demo_matched":
        gripper_delta = (
            DEMO_MATCHED_GRIPPER_DELTA_M
            if args.max_delta_gripper_m is None
            else float(args.max_delta_gripper_m)
        )
        return {
            "arm_delta_rad": float(max(DEMO_MATCHED_JOINT_DELTA_RAD[:3])),
            "wrist_delta_rad": float(max(DEMO_MATCHED_JOINT_DELTA_RAD[3:])),
            "gripper_delta_m": gripper_delta,
            "joint_delta_rad": DEMO_MATCHED_JOINT_DELTA_RAD,
        }

    gripper_delta = 0.004 if args.max_delta_gripper_m is None else float(args.max_delta_gripper_m)
    return {
        "arm_delta_rad": float(args.max_delta_rad),
        "wrist_delta_rad": float(args.max_delta_rad * WRIST_DELTA_RATIO),
        "gripper_delta_m": gripper_delta,
        "joint_delta_rad": None,
    }


def apply_gripper_timing_gate(
    *,
    frame_idx: int,
    current_state: tuple[float, ...],
    raw_action: tuple[float, ...] | np.ndarray,
    raw_close_count: int,
    open_until_frame: int,
    close_confirm_frames: int,
    close_j2_min: float | None,
) -> tuple[np.ndarray, bool, int]:
    """Optionally hold gripper open until a close command is credible.

    This is a deployment-only timing gate for cases where the policy starts
    closing before the gripper is physically aligned with the object. Defaults
    are no-op: open_until_frame=0, confirm_frames=1, close_j2_min=None.
    """

    action = np.asarray(raw_action, dtype=np.float32).copy()
    policy_wants_close = float(action[6]) < GRIPPER_CLOSE_ONSET
    raw_close_count = raw_close_count + 1 if policy_wants_close else 0

    hold = False
    if frame_idx < open_until_frame:
        hold = True
    if close_j2_min is not None and float(current_state[1]) < close_j2_min:
        hold = True
    if policy_wants_close and raw_close_count < max(1, close_confirm_frames):
        hold = True

    gate_active = hold and float(action[6]) < GRIPPER_OPEN_M
    if gate_active:
        action[6] = GRIPPER_OPEN_M
    return action, gate_active, raw_close_count


if __name__ == "__main__":
    raise SystemExit(main())
