# Piper + SmolVLA Reference Audit

Audit date: 2026-05-29

This audit is for building a LeRobot-compatible AgileX Piper adapter for SmolVLA while preserving this project's verified Piper control semantics.

## Non-negotiable Project Invariants

- Do not overwrite or restructure the existing ACT project.
- Do not change the known-good action/state schema without explicit approval.
- Keep the action/state order exactly:

  ```text
  [j1, j2, j3, j4, j5, j6, gripper]
  ```

- Keep units exactly:
  - joints: radians
  - gripper: meters
- Keep camera keys unless there is a reviewed, explicit reason to change:
  - `observation.images.global_rgb`
  - `observation.images.wrist_rgb`
- Do not adopt gripper scale from a random reference repo. If a repo uses another scale, document the difference only.
- Do not assume another Piper repo matches our hardware topology. Our hardware may use a custom single-CAN / mirror-teaching workflow.

## References Inspected

### Official AgileX Piper

- AgileX Piper SDK: https://github.com/agilexrobotics/piper_sdk
- SDK V2 interface docs: https://github.com/agilexrobotics/piper_sdk/blob/master/asserts/V2/INTERFACE_V2.MD
- SDK V2 joint demo: https://github.com/agilexrobotics/piper_sdk/blob/master/piper_sdk/demo/V2/piper_ctrl_joint.py
- SDK V2 gripper demo: https://github.com/agilexrobotics/piper_sdk/blob/master/piper_sdk/demo/V2/piper_ctrl_gripper.py
- AgileX Piper ROS: https://github.com/agilexrobotics/piper_ros
- Piper URDF: https://github.com/agilexrobotics/piper_ros/blob/noetic/src/piper_description/urdf/piper_description.urdf
- Piper ROS single-arm control node: https://github.com/agilexrobotics/piper_ros/blob/noetic/src/piper/scripts/piper_ctrl_single_node.py
- Piper MoveIt joint limits: https://github.com/agilexrobotics/piper_ros/blob/noetic/src/piper_moveit/piper_with_gripper_moveit/config/joint_limits.yaml
- Piper ROS English README: https://github.com/agilexrobotics/piper_ros/blob/noetic/README%28EN%29.md

### LeRobot / SmolVLA

- LeRobot issue requesting AgileX Piper support: https://github.com/huggingface/lerobot/issues/1335
- LeRobot PR "Add Piper follower support": https://github.com/huggingface/lerobot/pull/1481
- LeRobot SmolVLA docs: https://github.com/huggingface/lerobot/blob/main/docs/source/smolvla.mdx
- SmolVLA model card: https://huggingface.co/lerobot/smolvla_base
- SmolVLA base config: https://huggingface.co/lerobot/smolvla_base/blob/main/config.json
- SmolVLA tutorial example: https://github.com/huggingface/lerobot/blob/main/examples/tutorial/smolvla/using_smolvla_example.py
- LeRobot dataset v3 docs: https://huggingface.co/docs/lerobot/main/lerobot-dataset-v3
- LeRobot real-world robot recording docs: https://huggingface.co/docs/lerobot/il_robots

### Community Piper + LeRobot

- AgRoboticsResearch `lerobot_robot_piper`: https://github.com/AgRoboticsResearch/lerobot_robot_piper
- WeGo Robotics `lerobot_piper` branch: https://github.com/WeGo-Robotics/lerobot_piper/tree/piper
- MINT-SJTU Evo-RL LeRobot fork with Piper/Piper-X support: https://github.com/MINT-SJTU/Evo-RL
- Additional search terms checked:
  - `lerobot_piper`
  - `lerobot_piper2`
  - `lerobot_piper3`
  - `lerobot_robot_piper`
  - `AgileX Piper LeRobot`

Exact-name searches for `lerobot_piper2` and `lerobot_piper3` did not turn up an authoritative public implementation in this pass. The useful public Piper/LeRobot references were the LeRobot PR, AgRoboticsResearch package, WeGo fork, and Evo-RL fork.

## Official AgileX Piper SDK Findings

The SDK is the only reference that should define low-level CAN command semantics.

Useful facts:

- CAN setup uses SocketCAN. The AgileX docs and examples use `can0` at `1000000` baud.
- The main Python interface is `C_PiperInterface_V2` in current V2 examples. Older examples and ROS wrappers may use `C_PiperInterface`.
- Typical connection flow is:
  - construct interface with CAN name and optional flags
  - call `ConnectPort()`
  - set control mode before sending commands
  - enable the arm explicitly
- Readback methods include `GetArmJointMsgs()`, `GetArmGripperMsgs()`, `GetArmStatus()`, and `GetArmEnableStatus()`.
- The SDK docs note that joint feedback requires motion-output/follower-style feedback mode in some setups, commonly via `MasterSlaveConfig(0xFC, 0, 0, 0)`.

Important API calls:

```text
ConnectPort()
MotionCtrl_2(ctrl_mode=0x01, move_mode=0x01, move_spd_rate_ctrl=..., is_mit_mode=0x00)
EnableArm(7) / EnablePiper()
DisableArm(7) / DisablePiper()
JointCtrl(joint_1, joint_2, joint_3, joint_4, joint_5, joint_6)
GripperCtrl(gripper_angle, gripper_effort, gripper_code, set_zero)
EmergencyStop(0x01) / EmergencyStop(0x02)
ResetPiper()
```

SDK raw units:

- `JointCtrl` takes integer joint positions in `0.001 degree`.
- `GetArmJointMsgs()` reports joint positions in the same raw convention.
- `GripperCtrl` takes `gripper_angle` in `0.001 mm`.
- `GetArmGripperMsgs()` reports `grippers_angle` in the same gripper raw convention.

Project conversion rules should be:

```text
joint_sdk_int = round(joint_rad * 180 / pi * 1000)
joint_rad = joint_sdk_int / 1000 * pi / 180

gripper_sdk_int = round(gripper_m * 1_000_000)
gripper_m = gripper_sdk_int / 1_000_000
```

Official SDK joint limits, converted to project units:

```text
j1: [-2.6179,   2.6179] rad
j2: [ 0.0,      3.14  ] rad
j3: [-2.967,    0.0   ] rad
j4: [-1.745,    1.745 ] rad
j5: [-1.22,     1.22  ] rad
j6: [-2.09439,  2.09439] rad
```

Gripper control codes:

```text
0x00: disable
0x01: enable
0x02: disable and clear error
0x03: enable and clear error
0xAE: set current gripper position as zero, passed in set_zero
```

Safety notes:

- `EnablePiper()`/`EnableArm(7)` should be explicit and should check readback status with a timeout.
- `DisablePiper()`/`DisableArm(7)` should be available but should not be hidden in unrelated code paths.
- `EmergencyStop(0x01)` and resume `EmergencyStop(0x02)` should be explicit operator-level actions.
- `ResetPiper()` must not be called automatically. AgileX docs and ROS docs warn that reset immediately removes power and the arm can fall.

What can be borrowed:

- CAN connect order and explicit `ConnectPort()` usage.
- Unit conversions at the SDK boundary.
- Use of `MotionCtrl_2(... move_mode=0x01 ...)` for joint control.
- Enable/disable status checks.
- Official joint limits.

What must not be copied blindly:

- Any demo loop timing, sleep values, or go-to-zero pose.
- Any default call to `MasterSlaveConfig(0xFC, ...)` unless our hardware mode has been explicitly selected for that run.
- Any automatic reset or disconnect "safe pose" from examples.

## Official AgileX Piper ROS Findings

The ROS repo is useful for naming, URDF limits, MoveIt conventions, and AgileX's own ROS mapping. It should not define our LeRobot dataset schema.

Official joint and link conventions:

- Arm joints are `joint1` through `joint6`.
- The gripper URDF uses a fixed `joint6_to_gripper_base`, then two prismatic finger joints:
  - `joint7`: `0` to `0.035` m
  - `joint8`: `-0.035` to `0` m
- The ROS single-arm control node publishes a compact joint state name list:

  ```text
  ["joint1", "joint2", "joint3", "joint4", "joint5", "joint6", "gripper"]
  ```

ROS conversion behavior:

- The ROS control node converts incoming arm joint positions from radians to SDK raw `0.001 degree`.
- The same node converts gripper command meters to SDK raw `0.001 mm`.
- The node reads gripper feedback by dividing `grippers_angle` by `1_000_000`.

ROS gripper differences:

- The URDF models two finger joints with `0.035` m per side.
- The ROS README also discusses RViz gripper scaling because RViz `joint7` control may be multiplied to represent the real opening.
- The single-arm node contains gripper clamps around `0.08` m in some paths.
- ROS service paths mention different practical ranges, for example around `0.07` m.

Decision: treat all ROS gripper range values as reference notes only. The project should keep gripper in meters and use this project's verified gripper min/max, not a ROS/RViz-derived multiplier.

What can be borrowed:

- Official joint names for diagnostics and optional metadata.
- Official joint limits and velocity-limit context.
- Confirmation that the compact arm+gripper control surface is six revolute joints plus one scalar gripper.
- The exact SDK unit conversions.

What must not be copied:

- ROS topics, ROS node lifecycle, or MoveIt-specific abstractions.
- RViz gripper multiplier behavior.
- Any ROS-specific clamp as our project gripper scale.
- Any reset behavior exposed by ROS services as a normal adapter operation.

## LeRobot Piper Issue and PR Findings

Issue #1335 is the upstream request to add AgileX Piper support. PR #1481 is the concrete upstream work-in-progress reference. As of this audit, the PR page still shows the PR as open, so it should be treated as a reference, not a stable upstream contract.

Useful upstream LeRobot patterns:

- Piper is modeled as a follower robot.
- `piper-sdk` is added as an optional dependency in the PR.
- The PR uses normal LeRobot robot methods such as:
  - `connect()`
  - `disconnect()`
  - `get_observation()`
  - `send_action()`
  - `observation_features`
  - `action_features`
- The PR demonstrates how LeRobot teleop, record, visualize, train, and policy-control workflows expect a robot wrapper to plug in.

Important differences from this project:

- The PR maps an SO100/SO101 leader shape to Piper, including a fixed Piper joint, rather than preserving a native 7-value Piper action/state vector.
- The PR uses flat feature names like `joint_0.pos`/`joint_1.pos` or leader-style names such as `shoulder_pan.pos`.
- It uses percent or range-normalized action concepts in places.
- It discusses converting `-100` to `100` style leader values into Piper joint ranges.
- It uses a gripper interpretation that does not match our hard rule of gripper meters.

Decision: borrow the LeRobot wrapper shape, not the action mapping or units.

What can be borrowed:

- Registration/config pattern for a LeRobot `Robot`.
- LeRobot lifecycle method names.
- Optional dependency declaration idea.
- The idea that Piper support should be follower-oriented for policy rollout.

What must not be copied:

- SO100/SO101 alias mapping.
- Fixed joint insertion.
- Percent-normalized control.
- PR gripper scale.
- Any code path that sends a hard-coded disconnect pose or reset without project review.

## Community Project Findings

Community repos are useful for adapter structure and operational lessons. They are not authority for our schema, units, or hardware topology.

### AgRoboticsResearch/lerobot_robot_piper

Useful for:

- Out-of-tree LeRobot plugin/package layout.
- Piper config separation.
- CAN interface configuration.
- Camera config pass-through.
- Lazy SDK wrapper structure.
- Real command examples for record and policy control.

Differences:

- Uses feature names such as `joint_1.pos` through `joint_6.pos` plus `gripper.pos`.
- Includes SO101-style aliasing and sign flips.
- Has optional degree/radian handling, with defaults that can differ from this project.
- Treats gripper in mm or SDK-adjacent scale in places.
- Uses gripper scale choices such as 10 mm that must not be adopted here.

Borrow:

- Package/plugin layout and config ideas.
- Separation between hardware SDK wrapper and LeRobot-facing robot class.
- Camera config pass-through.

Do not copy:

- Sign flips.
- Alias mapping.
- Gripper scale.
- Degree-mode defaults.
- Any calibration values.

### WeGo-Robotics/lerobot_piper branch piper

Useful for:

- Leader/follower split.
- A Piper motor-bus abstraction.
- `max_relative_target`-style motion safety idea.
- Passing camera config through LeRobot.
- Master/follower mode concepts, including `0xFC` follower and `0xFA` master references.

Differences:

- Uses normalized range modes such as `RANGE_M100_100` and `RANGE_0_100`.
- Calibration ranges are in SDK raw units, not project radians/meters.
- Uses flat keys like `joint1.pos` and `gripper.pos`.
- Sends gripper raw values with its own scale/range assumptions.
- Calls master/slave configuration in ways that may not match our single-CAN mirror-teaching topology.

Borrow:

- Safety concept of limiting relative target jumps.
- Explicit leader/follower separation as a conceptual model.
- Awareness of master/follower CAN modes.

Do not copy:

- Normalization scheme.
- Motor-bus schema.
- Key names.
- Gripper range.
- Topology setup as a default.

### MINT-SJTU/Evo-RL

Useful for:

- A current LeRobot fork with Piper/Piper-X support.
- Mature follower and leader classes.
- `lerobot-setup-can` style operational tooling.
- Mode guards around follower/motion-output mode.
- Handling of text/task input for VLA-style policies.
- Practical warnings that some Piper modes cannot accept external commands.

Differences:

- Uses flat LeRobot motor features, not our array schema.
- Converts through its own `milli_to_unit`/`unit_to_milli` utilities.
- Uses calibration values tied to its own setup.
- Assumes firmware/mode behavior that may not match our hardware.
- Includes bi-arm/Piper-X conventions outside this project's scope.

Borrow:

- Mode-readback checks.
- Refusing partial joint actions.
- Optional mode refresh.
- Configurable speed ratio.
- Clear errors only through explicit, reviewed behavior.

Do not copy:

- Dataset schema.
- Calibration data.
- Unit helpers without adapting to our radians/meters contract.
- Firmware/topology assumptions.

## LeRobot / SmolVLA Schema Findings

SmolVLA is a vision-language-action policy. The official docs and model card describe inputs as:

- multiple camera views
- current sensorimotor/proprioceptive state
- natural-language task instruction

It outputs continuous actions, usually as action chunks.

LeRobot policy configs encode feature names. The SmolVLA base config uses feature keys like:

```text
observation.state
observation.images.camera1
observation.images.camera2
observation.images.camera3
action
```

The exact image key names are not universal. They are part of the dataset/model feature contract. The official tutorial warns that camera keys and resolutions must match what the model was trained with.

For this project, the correct SmolVLA-facing feature contract should be:

```text
observation.state: shape (7,), float32
  [j1, j2, j3, j4, j5, j6, gripper]
  joints in radians, gripper in meters

observation.images.global_rgb: RGB image
observation.images.wrist_rgb: RGB image

action: shape (7,), float32
  [j1, j2, j3, j4, j5, j6, gripper]
  joints in radians, gripper in meters

task: natural-language task string used during recording, training, and rollout
```

Dataset notes:

- LeRobotDataset v3 stores multimodal time-series data, videos/images, sensorimotor features, and episode/task metadata.
- Recording commands use `--dataset.single_task="..."`.
- SmolVLA fine-tuning should use the same task descriptions and feature keys expected at rollout time.
- Dataset finalization is required before pushing/loading v3 datasets.

Decision: use LeRobot-compatible feature names at the dataset/policy boundary, but preserve this project's exact state/action vector and camera keys.

## Differences That Matter

| Area | Project invariant | Reference differences | Decision |
| --- | --- | --- | --- |
| Joint order | `[j1, j2, j3, j4, j5, j6, gripper]` | ROS uses `joint1` names; community repos use `joint_1.pos`, `joint1.pos`, or SO leader aliases | Keep project order. Names are metadata only. |
| Joint units | radians | SDK uses `0.001 degree`; some repos use degrees or percent normalization | Convert only at SDK boundary. |
| Gripper units | meters | SDK uses `0.001 mm`; ROS/RViz has doubled-finger conventions; community repos use mm, percent, 10 mm, 68 mm, or 80 mm assumptions | Keep meters. Use project-verified gripper limits only. |
| Camera keys | `observation.images.global_rgb`, `observation.images.wrist_rgb` | LeRobot examples use `front`, `camera1`, `camera2`, `cam_1`, etc. | Keep project keys and train/configure SmolVLA with those keys. |
| State/action schema | array feature keys `observation.state` and `action` | LeRobot robot wrappers often expose flat motor keys | Adapter may internally translate, but dataset/policy contract stays array-based. |
| Robot topology | custom single-CAN / mirror-teaching may exist | Community repos assume follower/master modes such as `0xFC`/`0xFA` | Make topology mode explicit config. Do not set master/slave modes by default. |
| Safety | preserve verified control semantics | Some references reset, clear errors, or move to safe pose automatically | No automatic reset or hard-coded park pose. Operator-reviewed safety actions only. |

## Recommended Final Adapter Design

Build a small layered adapter instead of merging a community robot wrapper wholesale.

### Layer 1: PiperHardwareClient

Responsibility: one narrow wrapper around the official AgileX SDK.

Suggested behavior:

- Own `C_PiperInterface_V2` construction.
- Connect using configured CAN interface, normally `can0`.
- Optionally initialize CAN only when explicitly configured.
- Expose explicit methods:
  - `connect()`
  - `disconnect()`
  - `enable(timeout_s=...)`
  - `disable()`
  - `read_state() -> np.ndarray shape (7,)`
  - `send_action(action: np.ndarray shape (7,))`
  - `emergency_stop()`
  - `resume_after_emergency_stop()`
- Convert SDK raw units to project units on read.
- Convert project units to SDK raw units on write.
- Clamp arm joints to official limits.
- Clamp gripper to this project's verified gripper limits.
- Check finite numeric values before sending.
- Reject partial joint commands.
- Rate-limit or max-delta-limit targets as a configurable safety layer.
- Never call `ResetPiper()` automatically.
- Never send `MasterSlaveConfig(...)` unless an explicit config requests that mode for the current hardware run.

### Layer 2: PiperSmolVLASchemaAdapter

Responsibility: preserve the project schema and expose a LeRobot-compatible frame.

Frame shape:

```python
{
    "observation.state": np.ndarray((7,), dtype=np.float32),
    "observation.images.global_rgb": np.ndarray(..., dtype=np.uint8),
    "observation.images.wrist_rgb": np.ndarray(..., dtype=np.uint8),
    "task": str,
}
```

Action shape:

```python
{
    "action": np.ndarray((7,), dtype=np.float32)
}
```

If a LeRobot `Robot` class must expose flat motor keys for compatibility with generic LeRobot utilities, keep that translation local and lossless:

```text
Internal LeRobot motor keys, if needed:
j1.pos, j2.pos, j3.pos, j4.pos, j5.pos, j6.pos, gripper.pos

Canonical dataset/policy keys:
observation.state
action
```

The canonical dataset and policy path should remain `observation.state` and `action`.

### Layer 3: LeRobot Recording / Training / Rollout Bridge

Responsibility: integrate with LeRobot tooling without changing robot semantics.

Suggested behavior:

- Dataset features:
  - `observation.state`: `STATE`, shape `[7]`
  - `action`: `ACTION`, shape `[7]`
  - `observation.images.global_rgb`: `VISUAL`
  - `observation.images.wrist_rgb`: `VISUAL`
- Camera configs should use names that produce the exact keys above.
- Record with a single task instruction, then keep the same instruction style for training and rollout.
- Train/fine-tune SmolVLA on this dataset's feature names, not on upstream SO101 camera names.
- At rollout, verify the loaded policy config expects the same feature keys and action dimension before connecting to hardware.

## Safety Requirements For Implementation

- Do a dry-run mode that exercises schema, cameras, and policy I/O without sending CAN commands.
- Do a hardware-read-only mode that connects and logs state/images without sending actions.
- Add explicit operator-controlled enable.
- Fail closed if:
  - SDK connection fails
  - enable readback does not confirm all required motors
  - action has wrong shape
  - action contains NaN/Inf
  - action exceeds official joint limits beyond configured tolerance
  - gripper target exceeds project-verified limits
  - policy feature keys do not match adapter feature keys
- Keep `EmergencyStop` accessible from the runtime loop.
- Do not call `ResetPiper()` as recovery.
- Do not clear gripper errors or set gripper zero unless explicitly requested.
- Log raw SDK command values alongside project-unit values during early validation.

## Recommended Implementation Steps

1. Add a constants module for the locked schema:
   - `PIPER_JOINT_ORDER = ("j1", "j2", "j3", "j4", "j5", "j6", "gripper")`
   - `STATE_KEY = "observation.state"`
   - `ACTION_KEY = "action"`
   - `GLOBAL_IMAGE_KEY = "observation.images.global_rgb"`
   - `WRIST_IMAGE_KEY = "observation.images.wrist_rgb"`

2. Add a unit conversion module with tests:
   - radians <-> SDK `0.001 degree`
   - meters <-> SDK `0.001 mm`
   - shape and finite-value validation
   - official joint-limit clipping/rejection

3. Add `PiperHardwareClient` as the only code that imports `piper_sdk`.
   - Keep it independent from LeRobot.
   - Add dry-run and read-only modes.
   - Add explicit enable/disable and emergency stop methods.

4. Add camera capture/config glue that preserves:
   - `observation.images.global_rgb`
   - `observation.images.wrist_rgb`

5. Add the LeRobot-facing adapter.
   - `get_observation()` returns the locked observation schema.
   - `send_action()` accepts only a 7-vector action in project units.
   - If LeRobot generic utilities require flat motor features, translate only inside this adapter.

6. Add dataset feature metadata for SmolVLA.
   - `observation.state` shape `[7]`
   - `action` shape `[7]`
   - both camera keys with their actual resolution
   - task instruction metadata

7. Add validation scripts before policy rollout:
   - schema-only dry run
   - camera key/resolution check
   - hardware read-only check
   - one-step command in a low-speed supervised mode
   - short replay using recorded actions, not policy output

8. Add SmolVLA rollout only after the above pass.
   - Verify policy config feature keys and dimensions.
   - Verify normalization stats correspond to the same units.
   - Start with low speed and max-delta limits.

## Open Items To Confirm Before Code

- The current project's verified gripper min/max in meters.
- Whether our hardware should ever call `MasterSlaveConfig(0xFC, 0, 0, 0)`, or whether the single-CAN mirror-teaching workflow must leave mode configuration alone.
- Exact CAN interface name and whether CAN activation is managed externally.
- Camera device paths, resolutions, and frame formats for `global_rgb` and `wrist_rgb`.
- Whether the LeRobot version in this environment supports a robot returning array-valued `observation.state` directly, or whether a thin flat-key translation is needed for generic record/rollout commands.
- Whether existing ACT datasets are LeRobot v2.1 or v3.0, and whether any migration is desired. No migration should be performed without review.

## Final Recommendation

Use the official AgileX SDK and ROS repos only for hardware semantics, unit conversions, joint names, joint limits, and safety warnings. Use LeRobot and SmolVLA docs for feature-key and policy I/O expectations. Use community projects only for adapter organization patterns and operational caution.

The final adapter should be native to this project:

```text
Piper hardware raw SDK units
  <-> PiperHardwareClient unit conversion
  <-> project canonical 7-vector radians/meters schema
  <-> LeRobot/SmolVLA dataset and policy bridge
```

Do not adopt community Piper scales, leader aliases, percent normalization, camera names, or CAN topology defaults. The safest path is a thin adapter that treats our existing ACT/Piper schema as the source of truth and makes LeRobot/SmolVLA conform to it.
