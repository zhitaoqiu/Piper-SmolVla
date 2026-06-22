#!/usr/bin/env python3
"""SmolVLA training entry point — server version (Python API, not subprocess).

Patches SmolVLAConfig.input_features before make_policy() so that the renamed
batch keys (observation.image, observation.image2) match self.config.image_features.

Usage:
  python q_smolvla_train.py \\
      --dataset ~/q_ws/datasets/single_cube_line4pos_40_clean \\
      --output ~/q_ws/outputs/smolvla_smoke \\
      --steps 500 --episodes 4

  # Dry-run is the default; add --start-training to launch.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
RENAME_MAP = {
    "observation.images.global_rgb": "observation.image",
    "observation.images.wrist_rgb": "observation.image2",
}
EMPTY_CAMERAS = 1
MAX_STATE_DIM = 32
MAX_ACTION_DIM = 32
CHUNK_SIZE = 50
N_ACTION_STEPS = 50
RESIZE = (512, 512)
DEFAULT_MODEL_PATH = os.path.expanduser("~/q_ws/models/smolvla_base")
VLM_CACHE_SNAPSHOT = os.path.expanduser(
    "~/.cache/huggingface/hub/models--HuggingFaceTB--SmolVLM2-500M-Video-Instruct/"
    "snapshots/7b375e1b73b11138ff12fe22c8f2822d8fe03467"
)
REQUIRED_FEATURES = (
    "observation.state",
    "observation.images.global_rgb",
    "observation.images.wrist_rgb",
    "action",
)


def _feature_shape(features, key: str) -> tuple[int, ...]:
    if key not in features:
        raise SystemExit(f"dataset missing required feature: {key}")
    shape = features[key].get("shape")
    if shape is None:
        raise SystemExit(f"dataset feature has no shape: {key}")
    return tuple(int(dim) for dim in shape)


# ---------------------------------------------------------------------------
# Patch: set input_features on SmolVLAConfig so make_policy() does not
# override them with raw dataset keys.
# ---------------------------------------------------------------------------
def patch_smolvla_features(policy_cfg, dataset_root: Path):
    """Ensure policy_cfg.input_features uses *renamed* keys matching the batch.

    Called before ``train()`` so that when ``make_policy()`` runs later it sees
    ``cfg.input_features`` is already populated and skips the dataset override.

    Returns the policy_cfg (mutated in place).
    """
    from lerobot.configs.types import FeatureType, PolicyFeature
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    repo_id = f"piper/{dataset_root.name}"
    ds = LeRobotDataset(repo_id, root=str(dataset_root), tolerance_s=0.5)
    features = ds.meta.features

    missing = [key for key in REQUIRED_FEATURES if key not in features]
    if missing:
        raise SystemExit(f"dataset missing required features: {missing}")

    # Collect original shapes
    state_shape = _feature_shape(features, "observation.state")
    action_shape = _feature_shape(features, "action")
    global_shape = _feature_shape(features, "observation.images.global_rgb")
    wrist_shape = _feature_shape(features, "observation.images.wrist_rgb")
    if state_shape != (7,):
        raise SystemExit(f"unexpected observation.state shape: {state_shape}, expected (7,)")
    if action_shape != (7,):
        raise SystemExit(f"unexpected action shape: {action_shape}, expected (7,)")
    for key, shape in (
        ("observation.images.global_rgb", global_shape),
        ("observation.images.wrist_rgb", wrist_shape),
    ):
        if len(shape) != 3:
            raise SystemExit(f"unexpected {key} shape: {shape}, expected 3 dimensions")

    policy_cfg.input_features = {
        "observation.state": PolicyFeature(type=FeatureType.STATE, shape=state_shape),
        "observation.image": PolicyFeature(type=FeatureType.VISUAL, shape=global_shape),
        "observation.image2": PolicyFeature(type=FeatureType.VISUAL, shape=wrist_shape),
    }
    policy_cfg.output_features = {
        "action": PolicyFeature(type=FeatureType.ACTION, shape=action_shape),
    }

    return policy_cfg


def install_lerobot_feature_compat_patch() -> None:
    """Patch LeRobot feature conversion for v3 datasets without ``names``.

    Some LeRobot v3 datasets store image/video feature ``shape`` but omit the
    optional ``names`` field. LeRobot's training factory still reads
    ``ft["names"]`` while creating a policy, even when the policy config already
    has patched input_features. Keep this local to the training entrypoint so we
    do not edit site-packages on the 4090 server.
    """
    from lerobot.configs.types import FeatureType, PolicyFeature
    import lerobot.datasets.utils as dataset_utils
    import lerobot.policies.factory as policy_factory

    def dataset_to_policy_features_compat(features):
        policy_features = {}
        for key, ft in features.items():
            shape = tuple(ft["shape"])
            dtype = ft.get("dtype")
            if dtype in ["image", "video"]:
                feature_type = FeatureType.VISUAL
                if len(shape) != 3:
                    raise ValueError(f"Number of dimensions of {key} != 3 (shape={shape})")

                names = ft.get("names")
                if names is not None and len(names) >= 3:
                    if names[2] in ["channel", "channels"]:
                        shape = (shape[2], shape[0], shape[1])
                elif shape[-1] in (1, 3, 4) and shape[0] not in (1, 3, 4):
                    shape = (shape[2], shape[0], shape[1])
            elif key == dataset_utils.OBS_ENV_STATE:
                feature_type = FeatureType.ENV
            elif key.startswith(dataset_utils.OBS_STR):
                feature_type = FeatureType.STATE
            elif key.startswith(dataset_utils.ACTION):
                feature_type = FeatureType.ACTION
            else:
                continue

            policy_features[key] = PolicyFeature(type=feature_type, shape=shape)

        return policy_features

    dataset_utils.dataset_to_policy_features = dataset_to_policy_features_compat
    policy_factory.dataset_to_policy_features = dataset_to_policy_features_compat
    print("lerobot_feature_compat_patch=missing_names_ok")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser(
        description="SmolVLA training (Python API, patched input_features)."
    )
    p.add_argument("--dataset", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--steps", type=int, default=10000)
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--save-freq", type=int, default=None)
    p.add_argument("--log-freq", type=int, default=10)
    p.add_argument("--episodes", type=int, default=0)

    p.add_argument("--model-path", default=DEFAULT_MODEL_PATH)
    p.add_argument("--chunk-size", type=int, default=CHUNK_SIZE)
    p.add_argument("--n-action-steps", type=int, default=N_ACTION_STEPS)
    p.add_argument("--max-state-dim", type=int, default=MAX_STATE_DIM)
    p.add_argument("--max-action-dim", type=int, default=MAX_ACTION_DIM)
    p.add_argument("--empty-cameras", type=int, default=EMPTY_CAMERAS)
    p.add_argument("--no-freeze-vision-encoder", dest="freeze_vision_encoder",
                   action="store_false", default=True)
    p.add_argument("--no-train-expert-only", dest="train_expert_only",
                   action="store_false", default=True)
    p.add_argument("--gradient-checkpointing", action="store_true", default=False)

    p.add_argument("--skip-dataset-check", action="store_true")
    p.add_argument("--start-training", action="store_true")
    p.add_argument("--allow-model-download", action="store_true")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    dataset_root = Path(args.dataset).expanduser().resolve()
    output_dir = Path(args.output).expanduser().resolve()
    model_path = Path(args.model_path).expanduser().resolve()

    # ---- Guardrails ----
    if not dataset_root.is_dir():
        raise SystemExit(f"dataset not found: {dataset_root}")
    if not model_path.is_dir():
        raise SystemExit(f"model not found: {model_path}")
    vlm_cache = Path(VLM_CACHE_SNAPSHOT)
    if not vlm_cache.is_dir():
        raise SystemExit(f"VLM cache not found: {vlm_cache}")
    if output_dir.exists() and args.start_training:
        raise SystemExit(f"output already exists: {output_dir}")
    if args.steps <= 0 or args.batch_size <= 0:
        raise SystemExit("--steps and --batch-size must be > 0")
    if args.episodes < 0:
        raise SystemExit("--episodes must be >= 0")

    save_freq = args.save_freq or args.steps
    episodes = list(range(args.episodes)) if args.episodes > 0 else None

    # ---- Dataset quick-check ----
    n_ep = 0
    if not args.skip_dataset_check:
        try:
            from lerobot.datasets.lerobot_dataset import LeRobotDataset
            ds = LeRobotDataset(f"piper/{dataset_root.name}", root=str(dataset_root), tolerance_s=0.5)
            meta = ds.meta
            n_ep = int(getattr(meta, "total_episodes", 0) or 0)
            n_frames = int(getattr(ds, "num_frames", len(ds)) or 0)
            print(f"dataset_ok=True episodes={n_ep} frames={n_frames}")
        except Exception as exc:
            print(f"dataset_check_failed={type(exc).__name__}: {exc}")
            if args.start_training:
                raise SystemExit(1) from exc
    else:
        print("dataset_check_skipped=True")
    if args.episodes > 0 and n_ep > 0 and args.episodes > n_ep:
        raise SystemExit(f"--episodes={args.episodes} exceeds dataset episodes={n_ep}")

    # ---- Build configs ----
    from lerobot.configs.train import TrainPipelineConfig
    from lerobot.configs.default import DatasetConfig, WandBConfig
    from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig

    # 1) SmolVLA policy config – default-init, then patch
    policy_cfg = SmolVLAConfig()
    policy_cfg.pretrained_path = str(model_path)
    policy_cfg.vlm_model_name = str(VLM_CACHE_SNAPSHOT)  # bypass HF Hub, use local cache
    policy_cfg.device = "cuda"
    policy_cfg.dtype = "bfloat16"
    policy_cfg.chunk_size = args.chunk_size
    policy_cfg.n_action_steps = args.n_action_steps
    policy_cfg.max_state_dim = args.max_state_dim
    policy_cfg.max_action_dim = args.max_action_dim
    policy_cfg.empty_cameras = args.empty_cameras
    policy_cfg.resize_imgs_with_padding = RESIZE
    policy_cfg.freeze_vision_encoder = args.freeze_vision_encoder
    policy_cfg.train_expert_only = args.train_expert_only
    policy_cfg.push_to_hub = False
    if args.gradient_checkpointing:
        policy_cfg.gradient_checkpointing = True

    # PATCH: set input_features BEFORE train() calls make_policy()
    policy_cfg = patch_smolvla_features(policy_cfg, dataset_root)

    # Print confirmation
    print("SmolVLA patched input_features:")
    for k in sorted(policy_cfg.input_features):
        print(f"  {k}")
    print("SmolVLA patched output_features:")
    for k in sorted(policy_cfg.output_features):
        print(f"  {k}")
    print("image_features:")
    for k in sorted(policy_cfg.image_features):
        print(f"  {k}")

    # 2) Dataset config
    dataset_cfg = DatasetConfig(
        repo_id=f"piper/{dataset_root.name}",
        root=str(dataset_root),
    )
    if episodes is not None:
        dataset_cfg.episodes = episodes

    # 3) Training config
    job_name = output_dir.name
    cfg = TrainPipelineConfig(
        dataset=dataset_cfg,
        policy=policy_cfg,
        output_dir=output_dir,
        job_name=job_name,
        steps=args.steps,
        batch_size=args.batch_size,
        num_workers=0,
        save_checkpoint=True,
        save_freq=save_freq,
        log_freq=args.log_freq,
        rename_map=dict(RENAME_MAP),
        wandb=WandBConfig(enable=False),
    )

    # ---- Dry-run / Launch ----
    print(f"output_dir={output_dir}")
    print(f"steps={args.steps} batch_size={args.batch_size} save_freq={save_freq}")
    print("no_hardware_access=True")

    if not args.start_training:
        print("training_not_started=True (pass --start-training to launch)")
        return 0

    # ---- Environment (set directly on os.environ BEFORE any lerobot imports) ----
    if not args.allow_model_download:
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
        os.environ.pop("HF_ENDPOINT", None)
    else:
        os.environ.pop("HF_HUB_OFFLINE", None)
        os.environ.pop("TRANSFORMERS_OFFLINE", None)
        os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
    for key in ("HTTP_PROXY", "HTTPS_PROXY", "http_proxy", "https_proxy",
                "ALL_PROXY", "all_proxy"):
        os.environ.setdefault(key, "")
    os.environ.setdefault("NO_PROXY", "*")

    # ---- Run ----
    install_lerobot_feature_compat_patch()
    print("launching training...")
    from lerobot.scripts.lerobot_train import train
    train(cfg)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
