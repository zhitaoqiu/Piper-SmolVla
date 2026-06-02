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
    ]
    for script in scripts:
        result = run_help(script)
        assert result.returncode == 0, (script, result.stdout, result.stderr)
        assert "usage:" in result.stdout.lower()
