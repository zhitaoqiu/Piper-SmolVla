#!/usr/bin/env bash
# =============================================================================
# SmolVLA 全流程部署 —— 双摄，真实机械臂
# =============================================================================
# 用法:
#   bash scripts/run_smolvla_deploy.sh             默认参数
#   TASK="pick blue" bash scripts/run_smolvla_deploy.sh
#   DRY_RUN=1 bash scripts/run_smolvla_deploy.sh   仅推演不下发
# =============================================================================

set -euo pipefail

CHECKPT="${CHECKPT:-/home/huatec/models/050000/pretrained_model}"
TASK="${TASK:-Pick up the cube and put it into the box.}"
CAN_PORT="${CAN_PORT:-can0}"
GLOBAL_CAM="${GLOBAL_CAM:-realsense:243222074879}"
WRIST_CAM="${WRIST_CAM:-realsense:260322275595}"
CAMERA_FPS="${CAMERA_FPS:-30}"

RATE_HZ="${RATE_HZ:-20}"
MAX_FRAMES="${MAX_FRAMES:-5000}"

SAVE_FLAGS="--save-rollout --save-final-images"
HARDWARE="--allow-hardware-action --confirm-policy-rollout ROLLOUT"
NO_RETURN="$(if [[ "${NO_RETURN:-0}" == "1" ]]; then echo "--no-return-to-start"; fi)"
DRY_FLAG="$(if [[ "${DRY_RUN:-0}" == "1" ]]; then echo "--dry-run"; fi)"
AUTO_FLAG="$(if [[ "${AUTO_START:-0}" == "1" ]]; then echo "--auto-start"; fi)"

CAMERA_CONTROL_FLAGS=()
if [[ -n "${WRIST_AUTO_EXPOSURE:-}" ]]; then
    CAMERA_CONTROL_FLAGS+=(--wrist-auto-exposure "$WRIST_AUTO_EXPOSURE")
fi
if [[ -n "${WRIST_EXPOSURE:-}" ]]; then
    CAMERA_CONTROL_FLAGS+=(--wrist-exposure "$WRIST_EXPOSURE")
fi
if [[ -n "${WRIST_GAIN:-}" ]]; then
    CAMERA_CONTROL_FLAGS+=(--wrist-gain "$WRIST_GAIN")
fi
if [[ -n "${WRIST_BRIGHTNESS:-}" ]]; then
    CAMERA_CONTROL_FLAGS+=(--wrist-brightness "$WRIST_BRIGHTNESS")
fi
if [[ -n "${WRIST_POWER_LINE:-}" ]]; then
    CAMERA_CONTROL_FLAGS+=(--wrist-power-line "$WRIST_POWER_LINE")
fi

PYTHONPATH="src" python scripts/deploy_smolvla.py \
    --checkpoint "$CHECKPT" \
    --task "$TASK" \
    --can-port "$CAN_PORT" \
    --global-camera "$GLOBAL_CAM" \
    --wrist-camera "$WRIST_CAM" \
    --camera-fps "$CAMERA_FPS" \
    --rate-hz "$RATE_HZ" \
    --max-frames "$MAX_FRAMES" \
    $SAVE_FLAGS \
    $HARDWARE \
    $NO_RETURN \
    $DRY_FLAG \
    $AUTO_FLAG \
    "${CAMERA_CONTROL_FLAGS[@]}" \
    "$@"
