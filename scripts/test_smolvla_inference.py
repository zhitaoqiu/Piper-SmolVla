#!/usr/bin/env python3
"""Real-data inference test for SmolVLA checkpoints."""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import torch
import numpy as np

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

VLM_CACHE = os.path.expanduser(
    "~/.cache/huggingface/hub/models--HuggingFaceTB--SmolVLM2-500M-Video-Instruct/"
    "snapshots/7b375e1b73b11138ff12fe22c8f2822d8fe03467"
)
DATASET = os.path.expanduser("~/piper-smallvla/data/single_cube_line4pos_40_clean")


def load_config(ckpt_path: Path):
    from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig
    from lerobot.configs.types import FeatureType, PolicyFeature

    with open(ckpt_path / "config.json") as f:
        raw = json.load(f)

    for key in list(raw.keys()):
        if key in ("type", "use_amp", "license", "repo_id", "private",
                    "compile_mode", "compile_model", "pretrained_path",
                    "rtc_config", "use_peft", "peft_config", "tags",
                    "load_vlm_weights", "add_image_special_tokens",
                    "use_policy_training_preset"):
            del raw[key]

    for key in ("input_features", "output_features"):
        if key in raw and isinstance(raw[key], dict):
            raw[key] = {
                k: PolicyFeature(
                    type=FeatureType(v["type"]) if isinstance(v["type"], str) else v["type"],
                    shape=tuple(v["shape"]),
                )
                for k, v in raw[key].items()
            }

    cfg = SmolVLAConfig(**raw)
    cfg.vlm_model_name = VLM_CACHE
    cfg.device = "cuda"
    cfg.dtype = "float32"
    return cfg


def test_checkpoint(ckpt_path: Path, dataset_path: Path):
    from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    RENAME_MAP = {
        "observation.images.global_rgb": "observation.image",
        "observation.images.wrist_rgb": "observation.image2",
    }

    print(f"\n{'='*50}")
    print(f"Checkpoint: {ckpt_path.name}")

    cfg = load_config(ckpt_path)
    print(f"inputs: {list(cfg.input_features.keys())}")

    # Load policy
    t0 = time.time()
    policy = SmolVLAPolicy.from_pretrained(ckpt_path, config=cfg)
    policy = policy.to(dtype=torch.float32)
    policy.eval()
    print(f"loaded: {time.time() - t0:.1f}s")

    # Load dataset
    ds = LeRobotDataset(f"piper/{dataset_path.name}", root=str(dataset_path), tolerance_s=0.5)
    print(f"dataset: {ds.meta.total_episodes} episodes, {len(ds)} frames")

    # Build batch from 4 real frames across different episodes
    B = 4
    batch = {}
    for name, feat in cfg.input_features.items():
        batch[name] = []

    sample_indices = [0, 120, 240, 360]  # spread across episodes
    for idx in sample_indices[:B]:
        item = ds[idx]
        for k_old, k_new in RENAME_MAP.items():
            if k_old in item:
                item[k_new] = item.pop(k_old)
        for name in cfg.input_features:
            if name in item:
                batch[name].append(torch.as_tensor(item[name]))

    for name in batch:
        if batch[name]:
            batch[name] = torch.stack(batch[name]).to(dtype=torch.float32, device="cuda")
        elif "empty_camera" in name:
            # Empty camera — fill with zeros
            feat = cfg.input_features[name]
            shape = tuple(feat.shape) if hasattr(feat, 'shape') else tuple(feat.get("shape", []))
            batch[name] = torch.zeros(B, *shape, dtype=torch.float32, device="cuda")

    # Task
    task_text = "Pick up the cube and put it into the box."
    batch["task"] = [task_text] * B

    # Language tokens (needed by both predict_action_chunk and forward)
    from lerobot.utils.constants import OBS_LANGUAGE_TOKENS, OBS_LANGUAGE_ATTENTION_MASK
    tokenizer = policy.model.vlm_with_expert.processor.tokenizer
    encoded = tokenizer([task_text], return_tensors="pt", padding=True)
    batch[OBS_LANGUAGE_TOKENS] = encoded["input_ids"].expand(B, -1).to("cuda")
    batch[OBS_LANGUAGE_ATTENTION_MASK] = encoded["attention_mask"].bool().expand(B, -1).to("cuda")

    print(f"batch: {B} samples, keys: {list(batch.keys())}")
    for k, v in batch.items():
        if isinstance(v, torch.Tensor):
            print(f"  {k}: {list(v.shape)}")

    # Inference: predict_action_chunk
    t0 = time.time()
    with torch.inference_mode():
        action_chunk = policy.predict_action_chunk(batch)
    elapsed = time.time() - t0

    print(f"\n--- Inference ---")
    print(f"action_chunk shape: {list(action_chunk.shape)}")
    print(f"action_chunk[0, 0, :]: {action_chunk[0, 0, :].cpu().numpy().round(4)}")
    print(f"action_chunk[0, -1, :]: {action_chunk[0, -1, :].cpu().numpy().round(4)}")
    print(f"action range: [{action_chunk.min().item():.4f}, {action_chunk.max().item():.4f}]")
    print(f"inference: {elapsed:.3f}s")

    # Warmup
    times = []
    for _ in range(10):
        t0 = time.time()
        with torch.inference_mode():
            policy.predict_action_chunk(batch)
        times.append(time.time() - t0)
    print(f"avg (10 runs): {np.mean(times):.3f}s ± {np.std(times):.3f}s")

    # Also measure forward loss for comparison
    print(f"\n--- Forward loss (training mode, real state+image, real action) ---")
    for idx in sample_indices[:B]:
        pass  # action needs to be chunk_size=50, load from dataset properly

    # Load actions for these frames (chunk_size=50 window)
    from collections import defaultdict
    action_samples = defaultdict(list)
    for idx in sample_indices[:B]:
        # Get up to cfg.chunk_size future actions from this frame
        for t in range(min(cfg.chunk_size, len(ds) - idx)):
            item = ds[idx + t]
            action_samples[idx].append(torch.as_tensor(item["action"]).float())

    actions_list = []
    for idx in sample_indices[:B]:
        acts = torch.stack(action_samples[idx])
        if acts.shape[0] < cfg.chunk_size:
            pad = torch.zeros(cfg.chunk_size - acts.shape[0], acts.shape[1])
            acts = torch.cat([acts, pad])
        actions_list.append(acts)
    batch["action"] = torch.stack(actions_list).to(device="cuda")
    print(f"  action: {list(batch['action'].shape)}")

    t0 = time.time()
    with torch.inference_mode():
        loss, loss_dict = policy.forward(batch)
    print(f"loss: {loss.item():.6f}")
    print(f"forward: {time.time() - t0:.3f}s")

    del policy
    torch.cuda.empty_cache()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("checkpoints", nargs="+")
    p.add_argument("--dataset", default=DATASET)
    args = p.parse_args()

    ds = Path(args.dataset)
    if not ds.is_dir():
        raise SystemExit(f"dataset not found: {ds}")

    for cp in args.checkpoints:
        try:
            test_checkpoint(Path(cp), ds)
        except Exception as e:
            print(f"FAILED: {e}")
            import traceback
            traceback.print_exc()

    print("\nDone.")


if __name__ == "__main__":
    main()
