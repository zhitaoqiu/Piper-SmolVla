import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def run_help(script: str):
    env = os.environ.copy()
    env["PYTHONPATH"] = str(ROOT / "src")
    return subprocess.run(
        [sys.executable, str(ROOT / "scripts" / script), "--help"],
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )


def test_required_cli_help_commands():
    scripts = [
        "train_smolvla_smoke.py",
        "train_smolvla_full.py",
        "collect_smolvla_dataset.py",
        "check_collection_readiness.py",
        "check_piper_readonly.py",
        "check_existing_data_fullflow.py",
        "infer_smolvla_policy.py",
        "deploy_smolvla.py",
        "preview_cameras.py",
        "merge_lerobot_datasets.py",
        "reset_to_start.py",
        "shadow_test.py",
        "audit_prompt_plumbing.py",
        "record_two_object_language_random.py",
        "smoke_test_dataset_for_s_and_x.py",
    ]
    for script in scripts:
        result = run_help(script)
        assert result.returncode == 0, (script, result.stdout, result.stderr)
        assert "usage:" in result.stdout.lower()


def test_reset_to_start_requires_hardware_action_confirmation():
    env = os.environ.copy()
    env["PYTHONPATH"] = str(ROOT / "src")
    result = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "reset_to_start.py")],
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode != 0
    assert "--allow-hardware-action" in result.stderr


def test_shadow_test_requires_hardware_readonly_confirmation():
    env = os.environ.copy()
    env["PYTHONPATH"] = str(ROOT / "src")
    result = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "shadow_test.py"),
            "--checkpoint",
            "missing-checkpoint",
            "--task",
            "Pick up the blue object and put it into the box.",
        ],
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode != 0
    assert "--allow-hardware-readonly" in result.stderr
