"""Tests for the XVLA training launcher (scripts/train_xvla.py)."""

import importlib
import json
import os
import re
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "train_xvla.py"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _import_train_xvla():
    """Import the launcher module so we can unit-test its functions."""
    spec = importlib.util.spec_from_file_location("train_xvla", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["train_xvla"] = mod
    spec.loader.exec_module(mod)
    return mod


def _parse_args(argv: list[str] | None = None):
    mod = _import_train_xvla()
    if argv is None:
        argv = ["--dataset", "/tmp/fake_ds", "--output", "/tmp/fake_out"]
    return mod.parse_args(argv)


def _build_cmd(argv: list[str] | None = None):
    mod = _import_train_xvla()
    if argv is None:
        argv = ["--dataset", "/tmp/fake_ds", "--output", "/tmp/fake_out"]
    args = mod.parse_args(argv)
    return mod.build_command(args)


def _run_dryrun(*extra_args: str) -> subprocess.CompletedProcess:
    env = os.environ.copy()
    env["PYTHONPATH"] = str(ROOT / "src")
    return subprocess.run(
        [sys.executable, str(SCRIPT), "--dataset", str(ROOT), "--output", "/tmp/fake_out",
         "--skip-dataset-check", *extra_args],
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )


# ---------------------------------------------------------------------------
# 1.  No huggingface_hub dependency
# ---------------------------------------------------------------------------

def test_no_huggingface_hub_import():
    """The launcher module must not import huggingface_hub at load time."""
    src = SCRIPT.read_text()
    assert "huggingface_hub" not in src, (
        "train_xvla.py should never import huggingface_hub"
    )


def test_no_load_florence_config_json_function():
    """load_florence_config_json must be completely removed."""
    src = SCRIPT.read_text()
    assert "load_florence_config_json" not in src, (
        "load_florence_config_json() must be deleted"
    )


def test_dry_run_does_not_import_huggingface_hub():
    """A dry-run invocation must succeed without huggingface_hub installed."""
    result = _run_dryrun()
    assert result.returncode == 0, result.stderr
    assert "DRY RUN" in result.stdout


# ---------------------------------------------------------------------------
# 2.  Required policy parameters
# ---------------------------------------------------------------------------

def test_policy_path():
    cmd = _build_cmd()
    assert "--policy.path=lerobot/xvla-base" in cmd


def test_policy_action_mode_auto():
    cmd = _build_cmd()
    assert "--policy.action_mode=auto" in cmd


def test_policy_max_action_dim_20():
    cmd = _build_cmd()
    assert "--policy.max_action_dim=20" in cmd


# ---------------------------------------------------------------------------
# 3.  Forbidden / removed parameters
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("forbidden", [
    "policy.pretrained_path",
    "franka_joint7",
    "policy.florence_config",
    "policy.max_state_dim",
    "policy.optimizer_lr",
])
def test_forbidden_params_absent(forbidden: str):
    cmd = _build_cmd()
    joined = " ".join(cmd)
    assert forbidden not in joined, f"'{forbidden}' must not appear in the training command"


# ---------------------------------------------------------------------------
# 4.  Defaults
# ---------------------------------------------------------------------------

def test_default_steps_500():
    mod = _import_train_xvla()
    args = mod.parse_args(["--dataset", "d", "--output", "o"])
    assert args.steps == 500


def test_default_batch_size_1():
    mod = _import_train_xvla()
    args = mod.parse_args(["--dataset", "d", "--output", "o"])
    assert args.batch_size == 1


def test_user_can_override_steps():
    cmd = _build_cmd(["--dataset", "d", "--output", "o", "--steps", "1234"])
    assert "--steps=1234" in cmd


def test_user_can_override_batch_size():
    cmd = _build_cmd(["--dataset", "d", "--output", "o", "--batch-size", "16"])
    assert "--batch_size=16" in cmd


# ---------------------------------------------------------------------------
# 5.  Rename map
# ---------------------------------------------------------------------------

def test_rename_map_direction():
    """Piper data keys map to XVLA expected keys."""
    mod = _import_train_xvla()
    mapping = mod.PIPER_TO_XVLA_RENAME_MAP
    assert mapping["observation.images.global_rgb"] == "observation.images.image"
    assert mapping["observation.images.wrist_rgb"] == "observation.images.image2"


def test_rename_map_in_command():
    cmd = _build_cmd()
    joined = " ".join(cmd)
    assert "--rename_map=" in joined
    # verify the JSON content
    for part in cmd:
        if part.startswith("--rename_map="):
            val = part.split("=", 1)[1]
            mapping = json.loads(val)
            assert mapping == {
                "observation.images.global_rgb": "observation.images.image",
                "observation.images.wrist_rgb": "observation.images.image2",
            }


# ---------------------------------------------------------------------------
# 6.  Remaining correct settings preserved
# ---------------------------------------------------------------------------

def test_required_settings_present():
    cmd = _build_cmd()
    joined = " ".join(cmd)

    required = [
        "--policy.num_image_views=3",
        "--policy.empty_cameras=1",
        "--policy.dtype=bfloat16",
        "--policy.device=cuda",
        "--policy.freeze_vision_encoder=false",
        "--policy.freeze_language_encoder=false",
        "--policy.train_policy_transformer=true",
        "--policy.train_soft_prompts=true",
        "--policy.push_to_hub=false",
        "--wandb.enable=false",
        "--policy.type=xvla",
    ]
    for r in required:
        assert r in joined, f"Missing required arg: {r}"


# ---------------------------------------------------------------------------
# 7.  build_env offline / online logic
# ---------------------------------------------------------------------------

class _FakeArgs:
    allow_model_download = False


def test_build_env_offline_clears_proxy_and_sets_offline_mode():
    mod = _import_train_xvla()
    args = _FakeArgs()
    args.allow_model_download = False
    env = mod.build_env(args)
    assert env.get("HF_HUB_OFFLINE") == "1"
    assert env.get("TRANSFORMERS_OFFLINE") == "1"
    for key in ("HTTP_PROXY", "HTTPS_PROXY", "http_proxy", "https_proxy"):
        assert env.get(key, "MISSING") == ""


def test_build_env_online_preserves_existing_proxy():
    mod = _import_train_xvla()
    args = _FakeArgs()
    args.allow_model_download = True
    # seed the env with a proxy value beforehand
    os.environ["HTTP_PROXY"] = "http://test-proxy:8080"
    try:
        env = mod.build_env(args)
        assert env.get("HF_HUB_OFFLINE", "") != "1"
        assert env.get("TRANSFORMERS_OFFLINE", "") != "1"
        # proxy keys should still be set (not blanked out)
        assert env.get("HTTP_PROXY") == "http://test-proxy:8080"
    finally:
        del os.environ["HTTP_PROXY"]


# ---------------------------------------------------------------------------
# 8.  Dry-run integration
# ---------------------------------------------------------------------------

def test_dry_run_prints_key_params():
    result = _run_dryrun()
    assert result.returncode == 0
    out = result.stdout

    must_contain = [
        "--policy.path=lerobot/xvla-base",
        "--policy.action_mode=auto",
        "--policy.max_action_dim=20",
        "--policy.empty_cameras=1",
        "--rename_map=",
    ]
    for token in must_contain:
        assert token in out, f"dry-run output missing: {token}"

    must_not = [
        "--policy.pretrained_path",
        "--policy.action_mode=franka_joint7",
        "--policy.florence_config",
        "--policy.max_state_dim",
        "--policy.optimizer_lr",
    ]
    for token in must_not:
        assert token not in out, f"dry-run output should not contain: {token}"


def test_dry_run_returns_zero():
    result = _run_dryrun()
    assert result.returncode == 0


def test_help():
    env = os.environ.copy()
    env["PYTHONPATH"] = str(ROOT / "src")
    result = subprocess.run(
        [sys.executable, str(SCRIPT), "--help"],
        env=env,
        text=True,
        capture_output=True,
    )
    assert result.returncode == 0
    assert "usage:" in result.stdout.lower()


# ---------------------------------------------------------------------------
# 9.  Edge cases
# ---------------------------------------------------------------------------

def test_missing_dataset_path_exits():
    result = _run_dryrun("--dataset", "/nonexistent/path_12345")
    assert result.returncode != 0
    assert "not found" in result.stderr


def test_output_exists_blocks_training():
    env = os.environ.copy()
    env["PYTHONPATH"] = str(ROOT / "src")
    result = subprocess.run(
        [sys.executable, str(SCRIPT),
         "--dataset", str(ROOT),
         "--output", str(ROOT),
         "--skip-dataset-check",
         "--start-training"],
        env=env,
        text=True,
        capture_output=True,
    )
    assert result.returncode != 0
    assert ("already exists" in result.stderr or "already exists" in result.stdout)


def test_skip_dataset_check_flag():
    result = _run_dryrun("--skip-dataset-check")
    assert result.returncode == 0
    assert "Dataset check skipped" in result.stdout


def test_chunk_and_n_action_steps_in_command():
    cmd = _build_cmd(["--dataset", "d", "--output", "o",
                      "--chunk-size", "25", "--n-action-steps", "20"])
    joined = " ".join(cmd)
    assert "--policy.chunk_size=25" in joined
    assert "--policy.n_action_steps=20" in joined
