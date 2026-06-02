#!/usr/bin/env bash
# 用法:
#   bash scripts/collect.sh green 2
#   bash scripts/collect.sh blue 3
set -euo pipefail

COLOR="${1:?usage: collect.sh <green|blue> <N>}"
N="${2:?usage: collect.sh <green|blue> <N>}"
OUTPUT="data/smolvla_two_object_left_green_right_blue_${COLOR}${N}"

cd "$(dirname "$0")/.."

PYTHONPATH=src python scripts/collect_smolvla_dataset.py \
  --allow-hardware-readonly --can-port can0 \
  --global-camera /dev/video6 --wrist-camera /dev/video4 \
  --output "$OUTPUT" \
  --task "Pick up the ${COLOR} object and put it into the box." \
  --episodes 1 --operator-demo --require-keyboard-start-stop
