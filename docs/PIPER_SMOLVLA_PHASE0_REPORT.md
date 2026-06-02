# Piper + SmolVLA Phase 0 Report

Date: 2026-05-29

## Scope Completed

Phase -1 and Phase 0 were completed only within the approved boundary.

No hardware path was implemented or exercised:

- no CAN connection
- no `piper_sdk` import
- no `send_action`
- no reset
- no gripper movement
- no training
- no deployment
- no ACT project changes

## Phase -1 Git Status

The workspace had an empty read-only `.git` placeholder directory, so normal `git init` could not write to `.git`, and the directory could not be moved because the environment reported it as busy.

To still create an initial commit, a separate Git store was initialized:

```bash
git init --bare .git-store
git --git-dir=.git-store --work-tree=. add docs/PIPER_SMOLVLA_REFERENCE_AUDIT.md
git --git-dir=.git-store --work-tree=. commit -m "Initial Piper SmolVLA audit"
```

Initial commit:

```text
942861f Initial Piper SmolVLA audit
```

Use this form for local git commands in this workspace:

```bash
git --git-dir=.git-store --work-tree=. status
```

## Files Added

```text
src/piper_smolvla/schema.py
src/piper_smolvla/units.py
src/piper_smolvla/limits.py
src/piper_smolvla/validation.py
tests/test_schema.py
tests/test_units.py
tests/test_validation.py
docs/PIPER_SMOLVLA_PHASE0_REPORT.md
```

## Locked Schema

Implemented in `src/piper_smolvla/schema.py`:

```text
PIPER_JOINT_ORDER = ("j1", "j2", "j3", "j4", "j5", "j6", "gripper")
STATE_DIM = 7
ACTION_DIM = 7
JOINT_UNIT = "radians"
GRIPPER_UNIT = "meters"
STATE_KEY = "observation.state"
GLOBAL_IMAGE_KEY = "observation.images.global_rgb"
WRIST_IMAGE_KEY = "observation.images.wrist_rgb"
ACTION_KEY = "action"
```

CAN topology defaults are also locked to no mode change:

```text
DEFAULT_CAN_TOPOLOGY_POLICY = "preserve_existing"
DEFAULT_CALL_MASTER_SLAVE_CONFIG = False
REFERENCE_FOLLOWER_MASTER_SLAVE_CONFIG = (0xFC, 0, 0, 0)
```

Phase 0 does not call `MasterSlaveConfig` or any CAN API.

## Unit Conversion

Implemented in `src/piper_smolvla/units.py`:

- joint radians -> SDK `0.001 degree`
- SDK `0.001 degree` -> joint radians
- gripper meters -> SDK `0.001 mm`
- SDK `0.001 mm` -> gripper meters
- 7D state/action vector conversion helpers that preserve `[j1, j2, j3, j4, j5, j6, gripper]`

Conversion formulas:

```text
joint_sdk_units = round(joint_rad * 180 / pi * 1000)
joint_rad = joint_sdk_units / 1000 * pi / 180

gripper_sdk_units = round(gripper_m * 1_000_000)
gripper_m = gripper_sdk_units / 1_000_000
```

## Limits

Implemented in `src/piper_smolvla/limits.py`.

Arm joint limits are the AgileX SDK/ROS limits in radians:

```text
j1: [-2.6179,   2.6179]
j2: [ 0.0,      3.14  ]
j3: [-2.967,    0.0   ]
j4: [-1.745,    1.745 ]
j5: [-1.22,     1.22  ]
j6: [-2.09439,  2.09439]
```

Gripper limits are configurable:

```text
gripper_min_m = 0.0 by default
gripper_max_m = None by default
```

Important: the gripper max is intentionally not hard-coded as a universal truth. It still needs to be confirmed with this project's measured Piper gripper range before hardware control or rollout.

## Validation

Implemented in `src/piper_smolvla/validation.py`:

- validates state shape `(7,)`
- validates action shape `(7,)`
- rejects strings/non-iterables as vectors
- rejects NaN
- rejects Inf
- rejects non-numeric values
- rejects arm joint limit violations
- rejects gripper min/max violations according to configurable `LimitConfig`
- allows limit checks to be disabled only after shape and finite checks pass

Validation exceptions:

```text
ValidationError
ShapeValidationError
FiniteValidationError
LimitValidationError
```

## Test Results

`pytest` was not initially installed in the `piper_act` conda environment, so it was installed there before running acceptance tests.

Command:

```bash
conda run -n piper_act env PYTHONPATH=/home/huatec/piper-smallvla/src python -m pytest tests
```

Result:

```text
18 passed in 0.02s
```

Smoke import command:

```bash
conda run -n piper_act env PYTHONPATH=/home/huatec/piper-smallvla/src python -c "from piper_smolvla.schema import PIPER_JOINT_ORDER, STATE_KEY, ACTION_KEY; from piper_smolvla.units import joint_rad_to_sdk_units, gripper_m_to_sdk_units; from piper_smolvla.validation import validate_action; assert PIPER_JOINT_ORDER == ('j1','j2','j3','j4','j5','j6','gripper'); assert STATE_KEY == 'observation.state'; assert ACTION_KEY == 'action'; assert joint_rad_to_sdk_units(0.0) == 0; assert gripper_m_to_sdk_units(0.001) == 1000; validate_action((0.0, 1.0, -1.0, 0.0, 0.0, 0.0, 0.01)); print('smoke import ok')"
```

Result:

```text
smoke import ok
```

## Phase 0 Acceptance

- `pytest` passes in `piper_act`: yes
- smoke import passes in `piper_act`: yes
- normal 7D state/action validation passes: yes
- wrong shape reports validation error: yes
- NaN reports validation error: yes
- Inf reports validation error: yes
- out-of-limit joint values report validation error: yes
- configurable gripper max reports validation error when exceeded: yes
- no hardware/CAN calls: yes
- no ACT modifications: yes

## Stop Point

Phase 0 is complete. Do not continue to Phase 1 until reviewed.

Recommended next review items:

- confirm this project's measured gripper min/max in meters
- confirm whether any run should ever opt into `MasterSlaveConfig(0xFC, 0, 0, 0)`
- decide whether Phase 0 files should be committed as a second commit
- decide whether Phase 1 should start with a dry-run adapter or hardware-read-only client

