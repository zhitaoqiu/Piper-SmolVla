#!/usr/bin/env bash
set -euo pipefail

OUTPUT="${1:-data/demo_showcase}"
TASK="${2:-Pick up the green object.}"

cd "$(dirname "$0")/.."

python scripts/collect_smolvla_dataset.py \
  --task "$TASK" \
  --output "$OUTPUT" \
  --episodes 1 \
  --fps 20 \
  --operator-demo \
  --require-keyboard-start-stop \
  --allow-hardware-readonly

echo ""
echo "replay:"
echo "  python scripts/replay_demo.py --dataset $OUTPUT --max-frames 200 --rate-hz 20 --action-smooth 0.0 --max-delta-rad 0.08 --allow-hardware-action --confirm-replay REPLAY_DEMO"
