# Documentation Index

This directory keeps the longer design notes and audits separate from the GitHub
front page.

## Documents

| File | Purpose |
| --- | --- |
| [`PIPER_SMOLVLA_REFERENCE_AUDIT.md`](PIPER_SMOLVLA_REFERENCE_AUDIT.md) | Reference audit for AgileX Piper, LeRobot, SmolVLA, and compatible community implementations. |
| [`PIPER_SMOLVLA_PHASE0_REPORT.md`](PIPER_SMOLVLA_PHASE0_REPORT.md) | Initial schema, units, limits, validation, and no-hardware adapter work. |
| [`PIPER_SMOLVLA_ADAPTER_FRAMEWORK.md`](PIPER_SMOLVLA_ADAPTER_FRAMEWORK.md) | Adapter framework structure and module-level design notes. |

## Current Entrypoints

| Entrypoint | Purpose |
| --- | --- |
| `scripts/preview_cameras.py` | Preview or snapshot the global/wrist cameras with shared defaults. |
| `scripts/record_two_object_language_random.py` | Record blue/green two-object language-conditioned demonstrations. |
| `scripts/run_train_4090_smolvla_twoobj200.sh` | Fine-tune SmolVLA on the 4090 server workspace. |
| `scripts/deploy_smolvla.py` | Run SmolVLA policy rollout with dry-run and hardware action gates. |

For the short operating guide, use the root [`README.md`](../README.md).
