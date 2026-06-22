#!/usr/bin/env bash
set -euo pipefail

# SmolVLA fine-tune wrapper for the 170-episode blue/green two-object dataset.
#
# Default mode is a read-only precheck plus SmolVLA dry-run:
#   bash scripts/run_train_4090_smolvla_twoobj200.sh
#
# Start the real training run explicitly on the 4090 server:
#   START_TRAINING=1 bash scripts/run_train_4090_smolvla_twoobj200.sh
#
# Useful overrides:
#   STEPS=80000 BATCH_SIZE=4 TASK_NAME=my_run START_TRAINING=1 bash ...

if [ -z "${Q_WS:-}" ]; then
  Q_WS=/home/huatecserver/q_ws
fi

if [ -z "${PYTHON_BIN:-}" ]; then
  if [ -x /home/huatecserver/miniconda3/envs/piper_smolvla_q/bin/python ]; then
    PYTHON_BIN=/home/huatecserver/miniconda3/envs/piper_smolvla_q/bin/python
  elif [ -x /home/huatecserver/miniconda3/envs/lerobot_q/bin/python ]; then
    PYTHON_BIN=/home/huatecserver/miniconda3/envs/lerobot_q/bin/python
  else
    PYTHON_BIN=python3
  fi
fi

export PATH="/home/huatecserver/miniconda3/envs/piper_smolvla_q/bin:/home/huatecserver/miniconda3/envs/lerobot_q/bin:$PATH"
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}
export HF_HOME=${HF_HOME:-/home/huatecserver/.cache/huggingface}
export HF_DATASETS_CACHE=${HF_DATASETS_CACHE:-/home/huatecserver/.cache/huggingface/datasets}
export HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-1}
export TRANSFORMERS_OFFLINE=${TRANSFORMERS_OFFLINE:-1}

cd "$Q_WS"

TASK_NAME=${TASK_NAME:-smolvla_twoobj200_170ep_20hz_100k_b4_4090}
DATASET=${DATASET:-$Q_WS/datasets/two_obj_language_200}
MODEL_PATH=${MODEL_PATH:-$Q_WS/models/smolvla_base}
TRAIN_ENTRY=${TRAIN_ENTRY:-$Q_WS/scripts/q_smolvla_train.py}
OUT_DIR=${OUT_DIR:-$Q_WS/outputs/$TASK_NAME}
LOG_DIR=${LOG_DIR:-$Q_WS/logs}
LOG_FILE=${LOG_FILE:-$LOG_DIR/${TASK_NAME}.log}

EXPECTED_EPISODES=${EXPECTED_EPISODES:-170}
EXPECTED_FRAMES=${EXPECTED_FRAMES:-32876}
EXPECTED_TASKS=${EXPECTED_TASKS:-2}
EXPECTED_BLUE=${EXPECTED_BLUE:-85}
EXPECTED_GREEN=${EXPECTED_GREEN:-85}
EXPECTED_FPS=${EXPECTED_FPS:-20}

# Keep this aligned with the real collection rate. SmolVLA deployment later
# should also start at 20Hz for this dataset.
STEPS=${STEPS:-100000}
BATCH_SIZE=${BATCH_SIZE:-4}
SAVE_FREQ=${SAVE_FREQ:-10000}
LOG_FREQ=${LOG_FREQ:-10}
CHUNK_SIZE=${CHUNK_SIZE:-50}
N_ACTION_STEPS=${N_ACTION_STEPS:-50}
GRADIENT_CHECKPOINTING=${GRADIENT_CHECKPOINTING:-0}
START_TRAINING=${START_TRAINING:-0}
MIN_FREE_GPU_MB=${MIN_FREE_GPU_MB:-18000}
RUN_PREFLIGHT_SMOKE=${RUN_PREFLIGHT_SMOKE:-1}
PREFLIGHT_ONLY=${PREFLIGHT_ONLY:-0}
SMOKE_STEPS=${SMOKE_STEPS:-1}
SMOKE_EPISODES=${SMOKE_EPISODES:-4}
SMOKE_BATCH_SIZE=${SMOKE_BATCH_SIZE:-1}
SMOKE_SAVE_FREQ=${SMOKE_SAVE_FREQ:-9999}
SMOKE_LOG_FREQ=${SMOKE_LOG_FREQ:-1}
SMOKE_DIR=${SMOKE_DIR:-/tmp/${TASK_NAME}_preflight_$(date +%Y%m%d_%H%M%S)_$$}

if [ ! -d "$DATASET" ]; then
  echo "ERROR: missing dataset: $DATASET" >&2
  exit 1
fi
if [ ! -d "$MODEL_PATH" ]; then
  echo "ERROR: missing SmolVLA base model: $MODEL_PATH" >&2
  exit 1
fi
if [ ! -f "$TRAIN_ENTRY" ]; then
  echo "ERROR: missing training entry: $TRAIN_ENTRY" >&2
  exit 1
fi
if [ ! -x "$PYTHON_BIN" ] && ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
  echo "ERROR: python executable not found: $PYTHON_BIN" >&2
  exit 1
fi
if [ "$START_TRAINING" = "1" ] && [ "$PREFLIGHT_ONLY" != "1" ] && [ -e "$OUT_DIR" ]; then
  echo "ERROR: output directory already exists: $OUT_DIR" >&2
  echo "Choose a new TASK_NAME/OUT_DIR or move the old output before training." >&2
  exit 1
fi
if [ "$PREFLIGHT_ONLY" = "1" ] && [ "$RUN_PREFLIGHT_SMOKE" != "1" ]; then
  echo "ERROR: PREFLIGHT_ONLY=1 requires RUN_PREFLIGHT_SMOKE=1" >&2
  exit 1
fi

mkdir -p "$LOG_DIR" "$HF_DATASETS_CACHE" "$(dirname "$OUT_DIR")"
exec > >(tee -a "$LOG_FILE") 2>&1

echo "=== $TASK_NAME started at $(date) ==="
echo "Q_WS=$Q_WS"
echo "PYTHON_BIN=$PYTHON_BIN"
echo "DATASET=$DATASET"
echo "MODEL_PATH=$MODEL_PATH"
echo "TRAIN_ENTRY=$TRAIN_ENTRY"
echo "OUT_DIR=$OUT_DIR"
echo "EXPECTED_EPISODES=$EXPECTED_EPISODES EXPECTED_FRAMES=$EXPECTED_FRAMES EXPECTED_TASKS=$EXPECTED_TASKS EXPECTED_FPS=$EXPECTED_FPS"
echo "EXPECTED_BLUE=$EXPECTED_BLUE EXPECTED_GREEN=$EXPECTED_GREEN"
echo "STEPS=$STEPS BATCH_SIZE=$BATCH_SIZE SAVE_FREQ=$SAVE_FREQ LOG_FREQ=$LOG_FREQ"
echo "CHUNK_SIZE=$CHUNK_SIZE N_ACTION_STEPS=$N_ACTION_STEPS"
echo "GRADIENT_CHECKPOINTING=$GRADIENT_CHECKPOINTING"
echo "START_TRAINING=$START_TRAINING MIN_FREE_GPU_MB=$MIN_FREE_GPU_MB"
echo "RUN_PREFLIGHT_SMOKE=$RUN_PREFLIGHT_SMOKE PREFLIGHT_ONLY=$PREFLIGHT_ONLY"
echo "SMOKE_STEPS=$SMOKE_STEPS SMOKE_EPISODES=$SMOKE_EPISODES SMOKE_BATCH_SIZE=$SMOKE_BATCH_SIZE"
echo "SMOKE_DIR=$SMOKE_DIR"
echo "image_transforms=false"
echo "no_hardware_access=true"

"$PYTHON_BIN" -m py_compile "$TRAIN_ENTRY"
if ! grep -q "lerobot_feature_compat_patch" "$TRAIN_ENTRY"; then
  echo "ERROR: training entry is missing the LeRobot feature compatibility patch: $TRAIN_ENTRY" >&2
  exit 1
fi

DATASET="$DATASET" \
MODEL_PATH="$MODEL_PATH" \
EXPECTED_EPISODES="$EXPECTED_EPISODES" \
EXPECTED_FRAMES="$EXPECTED_FRAMES" \
EXPECTED_TASKS="$EXPECTED_TASKS" \
EXPECTED_BLUE="$EXPECTED_BLUE" \
EXPECTED_GREEN="$EXPECTED_GREEN" \
EXPECTED_FPS="$EXPECTED_FPS" \
"$PYTHON_BIN" - <<'PY'
from collections import Counter, defaultdict
from pathlib import Path
import json
import os

try:
    import pyarrow.parquet as pq
except Exception as exc:
    raise SystemExit(f"ERROR: pyarrow is required for precheck: {type(exc).__name__}: {exc}") from exc

dataset = Path(os.environ["DATASET"])
model_path = Path(os.environ["MODEL_PATH"])
expected_episodes = int(os.environ["EXPECTED_EPISODES"])
expected_frames = int(os.environ["EXPECTED_FRAMES"])
expected_tasks = int(os.environ["EXPECTED_TASKS"])
expected_blue = int(os.environ["EXPECTED_BLUE"])
expected_green = int(os.environ["EXPECTED_GREEN"])
expected_fps = int(os.environ["EXPECTED_FPS"])

info = json.loads((dataset / "meta" / "info.json").read_text())
checks = {
    "total_episodes": expected_episodes,
    "total_frames": expected_frames,
    "total_tasks": expected_tasks,
    "fps": expected_fps,
}
for key, expected in checks.items():
    got = int(info.get(key, -1))
    if got != expected:
        raise SystemExit(f"ERROR: {key}={got} expected={expected}")

features = info.get("features", {})
required = [
    "observation.state",
    "action",
    "observation.images.global_rgb",
    "observation.images.wrist_rgb",
]
missing = [name for name in required if name not in features]
if missing:
    raise SystemExit(f"ERROR: missing features: {missing}")
if list(features["observation.state"].get("shape", [])) != [7]:
    raise SystemExit(f"ERROR: observation.state shape={features['observation.state'].get('shape')} expected=[7]")
if list(features["action"].get("shape", [])) != [7]:
    raise SystemExit(f"ERROR: action shape={features['action'].get('shape')} expected=[7]")

task_table = pq.read_table(dataset / "meta" / "tasks.parquet")
tasks = {}
for row in task_table.to_pylist():
    if "task" in row:
        task_text = row["task"]
    elif "__index_level_0__" in row:
        task_text = row["__index_level_0__"]
    else:
        raise SystemExit(
            f"ERROR: cannot find task text column in tasks.parquet columns={task_table.column_names}"
        )
    tasks[int(row["task_index"])] = str(task_text)
if len(tasks) != expected_tasks:
    raise SystemExit(f"ERROR: tasks.parquet has {len(tasks)} tasks expected={expected_tasks}")

frame_counts = defaultdict(int)
task_index_by_episode = {}
for parquet_path in sorted((dataset / "data").glob("**/*.parquet")):
    table = pq.read_table(parquet_path, columns=["episode_index", "task_index"])
    for row in table.to_pylist():
        episode_index = int(row["episode_index"])
        frame_counts[episode_index] += 1
        task_index_by_episode.setdefault(episode_index, int(row["task_index"]))

episodes = list(range(expected_episodes))
missing_episodes = [episode for episode in episodes if episode not in frame_counts]
if missing_episodes:
    raise SystemExit(f"ERROR: missing episodes: {missing_episodes}")

selected_frames = sum(frame_counts[episode] for episode in episodes)
if selected_frames != expected_frames:
    raise SystemExit(f"ERROR: selected_frames={selected_frames} expected={expected_frames}")

color_counts = Counter()
for episode in episodes:
    task = tasks.get(task_index_by_episode[episode], "").lower()
    if "blue" in task:
        color_counts["blue"] += 1
    elif "green" in task:
        color_counts["green"] += 1
    else:
        color_counts["other"] += 1
if (
    color_counts["blue"] != expected_blue
    or color_counts["green"] != expected_green
    or color_counts["other"] != 0
):
    raise SystemExit(f"ERROR: unexpected task colors: {dict(color_counts)}")

for rel in [
    "model.safetensors",
    "config.json",
    "policy_preprocessor.json",
    "policy_postprocessor.json",
]:
    path = model_path / rel
    if not path.exists():
        raise SystemExit(f"ERROR: missing model file: {path}")

model_cfg = json.loads((model_path / "config.json").read_text())
print("dataset_ok=True")
print("episode_count=", len(episodes))
print("total_frames=", selected_frames)
print("fps=", info.get("fps"))
print("blue_count=", color_counts["blue"])
print("green_count=", color_counts["green"])
print("state_shape=", features["observation.state"].get("shape"))
print("action_shape=", features["action"].get("shape"))
print("global_shape=", features["observation.images.global_rgb"].get("shape"))
print("wrist_shape=", features["observation.images.wrist_rgb"].get("shape"))
print("base_type=", model_cfg.get("type"))
print("base_chunk_size=", model_cfg.get("chunk_size"))
print("base_n_action_steps=", model_cfg.get("n_action_steps"))
print("base_freeze_vision_encoder=", model_cfg.get("freeze_vision_encoder"))
print("base_train_expert_only=", model_cfg.get("train_expert_only"))
PY

start_args=()
checkpoint_args=()
if [ "$GRADIENT_CHECKPOINTING" = "1" ]; then
  checkpoint_args+=(--gradient-checkpointing)
fi

if [ "$START_TRAINING" = "1" ]; then
  if command -v nvidia-smi >/dev/null 2>&1; then
    free_mb=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -n 1 | tr -d ' ')
    used_mb=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -n 1 | tr -d ' ')
    echo "gpu_memory_free_mb=$free_mb"
    echo "gpu_memory_used_mb=$used_mb"
    echo "--- current GPU compute apps ---"
    nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader,nounits || true
    if [ -n "$free_mb" ] && [ "$free_mb" -lt "$MIN_FREE_GPU_MB" ]; then
      echo "ERROR: not enough free GPU memory: ${free_mb}MiB < ${MIN_FREE_GPU_MB}MiB" >&2
      exit 1
    fi
  fi

  if [ "$RUN_PREFLIGHT_SMOKE" = "1" ]; then
    echo "preflight_smoke=START"
    "$PYTHON_BIN" -u "$TRAIN_ENTRY" \
      --dataset "$DATASET" \
      --output "$SMOKE_DIR" \
      --model-path "$MODEL_PATH" \
      --steps "$SMOKE_STEPS" \
      --batch-size "$SMOKE_BATCH_SIZE" \
      --save-freq "$SMOKE_SAVE_FREQ" \
      --log-freq "$SMOKE_LOG_FREQ" \
      --episodes "$SMOKE_EPISODES" \
      --chunk-size "$CHUNK_SIZE" \
      --n-action-steps "$N_ACTION_STEPS" \
      "${checkpoint_args[@]}" \
      --start-training
    echo "preflight_smoke=OK output=$SMOKE_DIR"
  fi

  if [ "$PREFLIGHT_ONLY" = "1" ]; then
    echo "preflight_only_done=true"
    echo "=== $TASK_NAME finished at $(date) ==="
    exit 0
  fi

  if [ -e "$OUT_DIR" ]; then
    echo "ERROR: output directory already exists after preflight: $OUT_DIR" >&2
    echo "Choose a new TASK_NAME/OUT_DIR or move the old output before training." >&2
    exit 1
  fi

  start_args+=(--start-training)
else
  echo "training_not_started_by_script=true"
fi

echo "training_command=$PYTHON_BIN $TRAIN_ENTRY --dataset $DATASET --output $OUT_DIR --model-path $MODEL_PATH --steps $STEPS --batch-size $BATCH_SIZE --save-freq $SAVE_FREQ --log-freq $LOG_FREQ --episodes $EXPECTED_EPISODES --chunk-size $CHUNK_SIZE --n-action-steps $N_ACTION_STEPS ${checkpoint_args[*]} ${start_args[*]}"

"$PYTHON_BIN" -u "$TRAIN_ENTRY" \
  --dataset "$DATASET" \
  --output "$OUT_DIR" \
  --model-path "$MODEL_PATH" \
  --steps "$STEPS" \
  --batch-size "$BATCH_SIZE" \
  --save-freq "$SAVE_FREQ" \
  --log-freq "$LOG_FREQ" \
  --episodes "$EXPECTED_EPISODES" \
  --chunk-size "$CHUNK_SIZE" \
  --n-action-steps "$N_ACTION_STEPS" \
  "${checkpoint_args[@]}" \
  "${start_args[@]}"

echo "=== $TASK_NAME finished at $(date) ==="
