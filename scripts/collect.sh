#!/usr/bin/env bash
# 用法:
#   bash scripts/collect.sh <场景> <颜色> <编号>
#   场景: LgRb (左绿右蓝) | LbRg (左蓝右绿)
#   颜色: green | blue
#   编号: 1~4
#
# 示例:
#   bash scripts/collect.sh LgRb green 1
#   bash scripts/collect.sh LbRg blue 3
set -euo pipefail

SCENE="${1:?usage: collect.sh <LgRb|LbRg> <green|blue> <N>}"
COLOR="${2:?usage: collect.sh <scene> <green|blue> <N>}"
N="${3:?usage: collect.sh <scene> <green|blue> <N>}"
OUTPUT="data/two_obj_${SCENE}_${COLOR}_${N}"

cd "$(dirname "$0")/.."

PYTHONPATH=src python scripts/collect_smolvla_dataset.py \
  --allow-hardware-readonly --can-port can0 \
  --output "$OUTPUT" --overwrite-output \
  --task "Pick up the ${COLOR} object and put it into the box." \
  --episodes 1 --operator-demo --require-keyboard-start-stop
