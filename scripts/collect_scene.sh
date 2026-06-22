#!/usr/bin/env bash
# 批量采集一个场景的 8 条 demo（绿蓝交替，4 轮）
# 用法:
#   bash scripts/collect_scene.sh LgRb    # 左绿右蓝
#   bash scripts/collect_scene.sh LbRg    # 左蓝右绿
set -euo pipefail

SCENE="${1:?usage: collect_scene.sh <LgRb|LbRg>}"

echo "============================================"
echo " 开始采集场景: ${SCENE}"
echo " 每轮采集 green + blue 各一条，共 4 轮"
echo "============================================"
echo ""

for N in 1 2 3 4; do
  echo "--- 第 ${N} 轮: 轮到 green ---"
  bash "$(dirname "$0")/collect.sh" "${SCENE}" green "${N}"

  echo ""
  echo "--- 第 ${N} 轮: 轮到 blue ---"
  bash "$(dirname "$0")/collect.sh" "${SCENE}" blue "${N}"

  echo ""
done

echo "============================================"
echo " 场景 ${SCENE} 8 条全部采集完毕"
echo "============================================"
