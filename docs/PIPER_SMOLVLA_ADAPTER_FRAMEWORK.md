# Piper + SmolVLA Adapter Framework

Date: 2026-06-01

This document describes the current no-hardware adapter framework built on top of the Phase 0 schema, unit, limit, and validation layer.

## Scope

Implemented:

- canonical SmolVLA observation assembly
- canonical 7D action validation
- LeRobot feature metadata helpers
- ACT/V2-style flat motor-key bridge for compatibility
- dry-run in-memory Piper state/action endpoint
- small dependency protocols for future hardware/camera adapters
- tests for schema, units, validation, features, and adapter behavior

Not implemented:

- no real Piper SDK client
- no CAN connection
- no `MasterSlaveConfig`
- no `send_action` to hardware
- no reset behavior
- no gripper movement
- no training
- no deployment loop
- no ACT project edits

## Local ACT/V2 References Read

The following local files were inspected as references only:

```text
/home/huatec/piper_diffusion_bottle_grasp-master/adapter_v2/schema.py
/home/huatec/piper_diffusion_bottle_grasp-master/adapter_v2/piper_bus.py
/home/huatec/piper_diffusion_bottle_grasp-master/adapter_v2/start_pose.py
/home/huatec/piper_diffusion_bottle_grasp-master/inference/deploy_adapter_v2.py
/home/huatec/piper_act_bottle_grasp/hardware/piper_wrapper.py
/home/huatec/piper_act_bottle_grasp/hardware/config_piper.py
```

Useful lessons borrowed:

- validated ACT/V2 uses a 7D qpos/state vector
- flat compatibility keys can be `j1.pos` through `j6.pos` and `gripper.pos`
- canonical state/action remains `[j1, j2, j3, j4, j5, j6, gripper]`
- deployment safety should keep action writes behind an explicit opt-in
- gripper range from ACT/V2 is a local measured deployment fact, not a universal Piper truth

Not copied:

- no ACT `deploy.py`
- no diffusion/ACT rollout loop
- no hard-coded ACT gripper max as this project's global truth
- no real hardware wrapper import
- no reset/start-pose motion logic
- no MIT/position control code

## Modules

### `src/piper_smolvla/config.py`

Defines adapter configuration:

```text
PiperSmolVLAAdapterConfig
ImageFeatureConfig
validate_adapter_config
```

Important defaults:

```text
can_topology_policy = "preserve_existing"
call_master_slave_config = False
allow_action_sink = False
require_task = True
```

The adapter therefore cannot write actions through a sink unless the caller explicitly opts in.

### `src/piper_smolvla/interfaces.py`

Defines small protocols:

```text
StateSource.read_state()
ImageSource.read_images()
ActionSink.write_action()
```

These keep the framework independent from `piper_sdk`, RealSense/OpenCV, and LeRobot runtime classes.

### `src/piper_smolvla/features.py`

Defines bridge helpers:

```text
MOTOR_POS_KEYS = ("j1.pos", "j2.pos", "j3.pos", "j4.pos", "j5.pos", "j6.pos", "gripper.pos")
vector_to_motor_pos_dict(...)
motor_pos_dict_to_vector(...)
build_lerobot_feature_spec(...)
required_observation_keys()
```

This is the compatibility layer for ACT/V2-style flat keys. It is not the canonical dataset schema.

Canonical dataset/policy keys remain:

```text
observation.state
observation.images.global_rgb
observation.images.wrist_rgb
action
```

### `src/piper_smolvla/adapter.py`

Defines:

```text
PiperSmolVLAAdapter
DryRunPiperIO
StaticImageSource
AdapterError
MissingImageError
MissingTaskError
ActionSinkDisabledError
```

Main behavior:

- `read_observation(task=...)` returns the canonical SmolVLA frame
- `prepare_policy_batch(...)` validates an existing frame before policy use
- `validate_policy_action(...)` validates a 7D action
- `send_action(...)` validates action first, then refuses by default unless `allow_action_sink=True`

Dry-run behavior:

- `DryRunPiperIO` stores state in memory
- enabling `allow_action_sink=True` can update only that in-memory state
- no hardware path exists in this module

## Current Feature Contract

Observation:

```python
{
    "observation.state": (j1, j2, j3, j4, j5, j6, gripper),
    "observation.images.global_rgb": ...,
    "observation.images.wrist_rgb": ...,
    "task": "...",
}
```

Action:

```python
(j1, j2, j3, j4, j5, j6, gripper)
```

Units:

```text
joints: radians
gripper: meters
```

## Verification

Test command:

```bash
/home/huatec/miniconda3/bin/conda run -n piper_act env PYTHONPATH=/home/huatec/piper-smallvla/src PYTHONDONTWRITEBYTECODE=1 python -m pytest tests
```

Result:

```text
32 passed in 0.05s
```

Smoke command:

```bash
/home/huatec/miniconda3/bin/conda run -n piper_act env PYTHONPATH=/home/huatec/piper-smallvla/src PYTHONDONTWRITEBYTECODE=1 python -c "from piper_smolvla.adapter import DryRunPiperIO, PiperSmolVLAAdapter, StaticImageSource; from piper_smolvla.config import PiperSmolVLAAdapterConfig; from piper_smolvla.schema import GLOBAL_IMAGE_KEY, WRIST_IMAGE_KEY, STATE_KEY; io = DryRunPiperIO((0.0, 1.0, -1.0, 0.0, 0.0, 0.0, 0.01)); adapter = PiperSmolVLAAdapter(state_source=io, image_source=StaticImageSource({GLOBAL_IMAGE_KEY: 'global', WRIST_IMAGE_KEY: 'wrist'}), config=PiperSmolVLAAdapterConfig()); obs = adapter.read_observation(task='pick bottle'); assert obs[STATE_KEY][6] == 0.01; print('adapter smoke ok')"
```

Result:

```text
adapter smoke ok
```

## Next Safe Step

The next safe layer is a hardware-read-only client skeleton:

- import `piper_sdk` only in one module
- connect only under an explicit read-only command
- read joint/gripper state
- convert SDK raw values to radians/meters
- never send `JointCtrl`, `GripperCtrl`, reset, or `MasterSlaveConfig` by default

Before that, confirm:

- project measured gripper max/min in meters
- whether this hardware topology should ever opt into follower feedback mode
- exact camera source implementation for `global_rgb` and `wrist_rgb`

