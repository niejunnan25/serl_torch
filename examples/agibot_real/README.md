# AgiBot Real Residual RL

`examples/agibot_real` is the repo-local AgiBot real-robot residual RL example.

The important boundary is:

- this repo owns the residual RL actor / learner / eval flow
- this repo owns the AgiBot env wrapper and robot-service bootstrap needed by
  that flow
- this repo does not import, source, or execute anything from
  `/vla/users/niejunnan/codebase/serl_torch/reference`
- the reference Tangyili inference code is only used as a semantic reference for
  observation/action contracts
- there is no standalone inference entrypoint in this example tree

The base policy server is external by design. For local real-robot residual RL,
assume it is already serving websocket policy requests at `127.0.0.1:9001`.

## What Is Included

- Local AgiBot env implementation:
  - [`env/task_env.py`](env/task_env.py)
  - [`env/controller.py`](env/controller.py)
  - [`env/factory.py`](env/factory.py)
- Robot-facing runtime helpers:
  - [`robot/interface.py`](robot/interface.py)
  - [`robot/retargeter.py`](robot/retargeter.py)
  - [`robot/hooks.py`](robot/hooks.py)
  - [`robot/init_positions.py`](robot/init_positions.py)
  - [`robot/reset_hooks.py`](robot/reset_hooks.py)
- Repo-local vendored SDK/bootstrap:
  - [`robot/sdk_bootstrap.py`](robot/sdk_bootstrap.py)
  - `vendor/a2d_sdk/wheels/*.whl`
- Repo-local robot-service bootstrap files:
  - [`robot_service/env.sh`](robot_service/env.sh)
  - [`robot_service/conf/copilot.pbtxt`](robot_service/conf/copilot.pbtxt)
  - [`robot_service/scripts/ros_env_wrapper.sh`](robot_service/scripts/ros_env_wrapper.sh)
  - [`scripts/services/start_robot_service.py`](scripts/services/start_robot_service.py)
- Residual RL runtime/data bindings:
  - [`runtime/runtime_bindings.py`](runtime/runtime_bindings.py)
  - [`runtime/data_bindings.py`](runtime/data_bindings.py)
  - [`runtime/obs_adapter.py`](runtime/obs_adapter.py)
  - [`runtime/policy_adapter.py`](runtime/policy_adapter.py)
  - [`runtime/controller_actor.py`](runtime/controller_actor.py)
- Training and eval entrypoints:
  - [`scripts/train/run_actor.py`](scripts/train/run_actor.py)
  - [`scripts/train/run_learner.py`](scripts/train/run_learner.py)
  - [`scripts/eval/evaluate_checkpoint.py`](scripts/eval/evaluate_checkpoint.py)
  - [`scripts/eval/process_eval_queue.py`](scripts/eval/process_eval_queue.py)
- Runtime preparation helper:
  - [`tools/prepare_robot_runtime.sh`](tools/prepare_robot_runtime.sh)

## Observation And Action Contract

The current real-robot residual RL implementation intentionally keeps a narrow
execution contract:

- control mode: `camera_position`
- env action dimension: `14`
- executed action layout:
  - left arm pose: `0:6`
  - left gripper: `6`
  - right arm pose: `7:13`
  - right gripper: `13`
- proprio state sent to base policy:
  - OpenPI / PI05: 14D `state/pose`
  - JoyRA: 18D `state/pose + state/head + state/waist`
- policy images:
  - head image
  - left wrist image
  - right wrist image

JoyRA may return an 18D action chunk. The current residual RL runtime keeps the
first 14 dimensions before composing residual actions, because the current
`AgiBotTaskEnv` only executes 14D `camera_position` actions. This is consistent
with the current real-robot execution path where head/waist control is disabled,
but it is not full 18D JoyRA parity.

No interpolation is added in this repo-local residual RL flow. The actor consumes
the base policy action chunk, composes residual actions, and enqueues the final
chunk into the controller-driven env.

## Runtime Dependency Boundary

For true robot residual RL, this repo no longer needs anything under
`/vla/users/niejunnan/codebase/serl_torch/reference` at runtime.

The AgiBot-specific runtime pieces that used to live outside the repo are now
bundled here:

- `a2d_sdk` wheel
- `genie_msgs_pb` wheel
- `cosine_bus` wheels for `x86_64` and `aarch64`
- repo-local `robot_service` env/config/scripts

At runtime, [`robot/sdk_bootstrap.py`](robot/sdk_bootstrap.py) extracts the
vendored wheels into `examples/agibot_real/vendor/a2d_sdk/_site/` and prepends
that directory to `sys.path`.

The ROS forwarder bundle is treated as a runtime asset, not a Git-tracked source
file. When robot-service is started with ROS/forwarder enabled, the repo-local
bootstrap resolves it in this order:

1. `AGIBOT_FORWARDER_DIR=/path/to/extracted/forwarder`
2. `AGIBOT_FORWARDER_TAR=/path/to/forwarder_x86_v1.7.0.tar.gz`
3. existing `examples/agibot_real/robot_service/forwarder`
4. local cache `examples/agibot_real/vendor/a2d_sdk/forwarder_x86_v1.7.0.tar.gz`

Important:

- the vendored `cosine_bus` wheel is CPython 3.10-specific, so the robot-side
  Python interpreter must be `3.10`
- kinematics/runtime deps such as `scipy`, `ruckig`, and the packages needed by
  [`robot/retargeter.py`](robot/retargeter.py) still need to exist in the active
  Python env
- base policy serving, such as JoyRA or OpenPI, remains external and is assumed
  to be local at `127.0.0.1:9001`

## Directory Layout

```text
examples/agibot_real/
  assets/
  conf/
  env/
  robot/
  robot_service/
  runtime/
  scripts/
    services/
    train/
    eval/
  tools/
  vendor/
```

## Start The Real-Robot Residual RL Flow

The intended startup order is:

1. Start the external base policy server at `127.0.0.1:9001`.
2. Prepare the ROS forwarder runtime once per machine if you need repo-local ROS startup.
3. Start repo-local AgiBot robot-service if it is not already running.
4. Start the learner.
5. Start the actor. The actor terminal owns the live robot env and controller.

### Terminal A: Base Policy Server

For JoyRA, start your external JoyRA server in its own environment. Example:

```bash
cd /workspace/codebase/JoyRA
conda activate joyra
export CUDA_VISIBLE_DEVICES=0
export PYTHONPATH=/workspace/codebase/JoyRA:$PYTHONPATH
python /workspace/codebase/JoyRA/deployment/real_infer/server.py \
  --host 0.0.0.0 \
  --port 9001 \
  --ckpt-path /path/to/run_dir/checkpoints/steps_xxx.pt
```

This command is intentionally outside `serl_torch`: the model server is a base
policy dependency, not part of the residual RL runtime.

If you want to launch the same JoyRA server shape from this repo, the wrapper
now follows that exact pattern:

```bash
cd /vla/users/niejunnan/codebase/serl_torch/examples/agibot_real
JOYRA_ROOT=/workspace/codebase/JoyRA \
JOYRA_CKPT_PATH=/workspace/codebase/JoyRA/outputs/.../checkpoints/steps_30000_pytorch_model.pt \
JOYRA_CONDA_ENV=joyra \
bash tools/serve_joyra.sh --port 9001
```

`tools/serve_joyra.sh` now:

1. activates the JoyRA conda env
2. `cd`s into `JOYRA_ROOT`
3. exports `PYTHONPATH=$JOYRA_ROOT:$PYTHONPATH`
4. runs `$JOYRA_ROOT/deployment/real_infer/server.py`

This matches the validated container-side JoyRA startup command more closely than
the older wrapper behavior.

### Terminal B: Prepare Robot Runtime

If you want the repo-local robot-service wrapper to start ROS/forwarder for you,
prepare the forwarder runtime once on that machine.

Option 1: use an already extracted forwarder directory.

```bash
cd /vla/users/niejunnan/codebase/serl_torch/examples/agibot_real
bash tools/prepare_robot_runtime.sh --from-dir /path/to/forwarder
```

Option 2: use a local tarball.

```bash
cd /vla/users/niejunnan/codebase/serl_torch/examples/agibot_real
bash tools/prepare_robot_runtime.sh --from-tar /path/to/forwarder_x86_v1.7.0.tar.gz
```

Option 3: download once from an internal artifact URL.

```bash
cd /vla/users/niejunnan/codebase/serl_torch/examples/agibot_real
AGIBOT_FORWARDER_URL=https://internal.example/path/forwarder_x86_v1.7.0.tar.gz \
bash tools/prepare_robot_runtime.sh
```

This prepares a repo-local `robot_service/forwarder` directory, or reuses an
existing one. The local cache path
`examples/agibot_real/vendor/a2d_sdk/forwarder_x86_v1.7.0.tar.gz` is intentionally
not tracked in Git.

If your machine already has a system-managed forwarder path, or you plan to run
robot-service with `--no-ros`, you can skip this step.

### Terminal C: Robot Service

Start the robot service from this repo:

```bash
cd /vla/users/niejunnan/codebase/serl_torch/examples/agibot_real
conda activate serl_torch
source robot_service/env.sh
python scripts/services/start_robot_service.py \
  -s \
  -c robot_service/conf/copilot.pbtxt
```

What this does:

1. extracts vendored SDK wheels from `vendor/a2d_sdk/wheels/`
2. resolves forwarder assets from `AGIBOT_FORWARDER_DIR`, `AGIBOT_FORWARDER_TAR`,
   `robot_service/forwarder`, or the local cache tarball
3. prepends the extracted wheel site to `sys.path`
4. imports `a2d_sdk.tools.robot_service`
5. delegates to `a2d_sdk.tools.robot_service.main()`

That replaces the old external-style startup:

```bash
cd /path/to/external/a2d_sdk
source env.sh
robot-service -s -c ./conf/copilot.pbtxt
```

If `robot-service` is already running, do not start another one.

If you do not want repo-local ROS/forwarder startup, use:

```bash
cd /vla/users/niejunnan/codebase/serl_torch/examples/agibot_real
conda activate serl_torch
source robot_service/env.sh
python scripts/services/start_robot_service.py \
  -s \
  -c robot_service/conf/copilot.pbtxt \
  --no-ros
```

or equivalently:

```bash
cd /vla/users/niejunnan/codebase/serl_torch/examples/agibot_real
AGIBOT_NO_ROS=1 bash tools/start_robot_service.sh
```

### Terminal D: Learner

Recommended wrapper:

```bash
cd /vla/users/niejunnan/codebase/serl_torch/examples/agibot_real
mkdir -p outputs/agibot_real/train_default
SERL_CONDA_ENV=serl_torch \
bash tools/run_learner.sh \
  conf/train_residual_sac.yaml \
  --bootstrap outputs/agibot_real/train_default/agentlace_bootstrap.pkl \
  policy.type=joyra \
  joyra.host=127.0.0.1 \
  joyra.port=9001 \
  hydra.run.dir=outputs/agibot_real/train_default/learner
```

Equivalent direct command:

```bash
cd /vla/users/niejunnan/codebase/serl_torch/examples/agibot_real
conda activate serl_torch
mkdir -p outputs/agibot_real/train_default
export CUDA_VISIBLE_DEVICES=0
python scripts/train/run_learner.py \
  --config-dir conf \
  --config-name train_residual_sac \
  policy.type=joyra \
  joyra.host=127.0.0.1 \
  joyra.port=9001 \
  hydra.run.dir=outputs/agibot_real/train_default/learner \
  ++training.async.enabled=true \
  ++training.async.backend=agentlace \
  ++training.async.agentlace.bootstrap_file=/vla/users/niejunnan/codebase/serl_torch/examples/agibot_real/outputs/agibot_real/train_default/agentlace_bootstrap.pkl
```

### Terminal E: Actor

Recommended wrapper:

```bash
cd /vla/users/niejunnan/codebase/serl_torch/examples/agibot_real
SERL_CONDA_ENV=serl_torch \
bash tools/run_actor.sh \
  conf/train_residual_sac.yaml \
  --bootstrap outputs/agibot_real/train_default/agentlace_bootstrap.pkl \
  policy.type=joyra \
  joyra.host=127.0.0.1 \
  joyra.port=9001 \
  hydra.run.dir=outputs/agibot_real/train_default/actor
```

Equivalent direct command:

```bash
cd /vla/users/niejunnan/codebase/serl_torch/examples/agibot_real
conda activate serl_torch
export CUDA_VISIBLE_DEVICES=0
python scripts/train/run_actor.py \
  --config-dir conf \
  --config-name train_residual_sac \
  policy.type=joyra \
  joyra.host=127.0.0.1 \
  joyra.port=9001 \
  hydra.run.dir=outputs/agibot_real/train_default/actor \
  ++training.async.enabled=true \
  ++training.async.backend=agentlace \
  ++training.async.agentlace.spawn_local_worker=false \
  ++training.async.agentlace.bootstrap_file=/vla/users/niejunnan/codebase/serl_torch/examples/agibot_real/outputs/agibot_real/train_default/agentlace_bootstrap.pkl
```

Notes:

- learner and actor must use the same `--bootstrap` path
- learner and actor must agree on `policy.type` and backend host/port
- the actor terminal owns the live robot env, action queue, and manual
  controller
- `controller.enabled=true` and `chunk_step.enabled=true` are the expected
  default training mode for real robot
- for a base-policy-only safety check, override `residual.alpha=0.0` before
  letting the residual policy affect the robot

## What The Actor Does

The real-robot actor loop is controller-driven:

1. read the latest robot observation from `AgiBotTaskEnv`
2. build a canonical policy input through `runtime/policy_adapter.py`
3. call the base policy server, such as JoyRA at `127.0.0.1:9001`
4. truncate the base action chunk to the current 14D env action contract
5. sample the residual SAC policy
6. compose `final_action = base_action + alpha * residual_delta`
7. enqueue the final action chunk into the env controller
8. collect executed transitions from the robot control thread
9. insert those transitions into replay for learner updates

The actor is therefore the real deployment entrypoint for residual RL. There is
no separate repo-local inference loop to run.

## JoyRA Websocket Contract

The repo-local JoyRA backend in
[`serl_launcher/policy/joyra`](../../serl_launcher/serl_launcher/policy/joyra)
is aligned to the current JoyRA websocket server shape.

Request fields sent to JoyRA:

- `observation/image`
- `observation/wrist_left_image`
- `observation/wrist_right_image`
- `observation/state`
- `prompt`

State layout:

- preferred JoyRA layout: 18D `pose + head + waist`
- fallback layout: 14D `pose` if the observation payload does not carry
  `state/head` and `state/waist`

Expected response fields:

- `actions`
- optional `policy_timing.infer_ms`
- optional `server_timing.infer_ms`

JoyRA requests are resized locally to `224x224` with padding before they are
sent. OpenPI/PI05 is not changed by this JoyRA-specific preprocessing.

## OpenPI / PI05

OpenPI remains supported through the existing OpenPI backend:

```bash
cd /vla/users/niejunnan/codebase/serl_torch/examples/agibot_real
OPENPI_ROOT=/path/to/openpi \
OPENPI_CONDA_ENV=my_openpi_env \
POLICY_CONFIG=pi05_agibot \
POLICY_DIR=/path/to/pi05_agibot/checkpoint \
bash tools/serve_openpi.sh --port 9001 --gpu-id 0
```

Then run learner/actor with:

```bash
policy.type=openpi openpi.host=127.0.0.1 openpi.port=9001
```

## Controller Keys

When `controller.enabled=true`, the actor/eval terminal accepts:

- `g`: ready / resume
- `p`: pause
- `r`: reset / truncate the current episode
- `s`: mark success
- `f`: mark failure
- `h`: print help

Current reset behavior is intentionally simple: `r` ends the episode and clears
queued actions, but it does not move the robot back to a home pose. Task-specific
reset motion should live in reset hooks.

## Evaluate A Checkpoint

Start the base policy server first, make sure the robot service is running, and
then run:

```bash
cd /vla/users/niejunnan/codebase/serl_torch/examples/agibot_real
SERL_CONDA_ENV=serl_torch \
bash tools/eval.sh \
  policy.type=joyra \
  joyra.host=127.0.0.1 \
  joyra.port=9001 \
  eval.checkpoint_path=outputs/checkpoint_2500.pt \
  hydra.run.dir=outputs/agibot_real/eval_default
```

With controller mode enabled, the eval terminal is also the operator console.

## Reference Notes

The reference implementation used for semantic comparison is:

- `reference/tangyili/tangyili/code/agibot/inference_camera_position.py`
- `reference/tangyili/tangyili/run_agibot.sh`

The comparison informed this repo in these ways:

- JoyRA should receive 18D state: `state/pose + state/head + state/waist`
- PI05/OpenPI should continue to receive 14D `state/pose`
- JoyRA policy images should be padded/resized to `224x224`
- the current real-robot execution path can stay 14D while head/waist control is
  disabled
- interpolation/extrapolation from the reference inference script is not part of
  the current residual RL flow

TODO / questions to confirm before declaring full JoyRA parity:

- Is dropping the final 4 JoyRA action dimensions acceptable long term, or
  should `AgiBotTaskEnv` eventually support the full 18D action path?
- The reference script applies gripper postprocessing on dimensions `6` and
  `13` with `/0.9` then clip to `[0, 1]`. This behavior is intentionally not
  ported yet; confirm the reason before adding it to residual RL.
- If JoyRA sometimes returns fewer actions than `residual.chunk_horizon`, the
  shared chunk helper currently pads by repeating the final action. Confirm
  whether that is acceptable for the real robot, or whether AgiBot should execute
  the shorter raw chunk directly.

## Removed Paths

The example-local offline demo conversion and online prefill collection paths
were removed from `examples/agibot_real`.

The shared residual training framework still has generic offline / prefill
support, so the AgiBot config keeps disabled compatibility stubs for
`cfg.offline` and `cfg.training.online_prefill`. The AgiBot actor and learner
entrypoints reject enabling those paths.

## Current Limitations

- only `camera_position` mode is implemented
- only 14D residual actions are executed
- success/reward logic is hook-driven; there is no baked-in AgiBot task reward
- normalization is disabled by default
- async eval launches a separate eval process, so keep
  `training.async_eval.enabled=false` unless that worker can own the robot
  exclusively

That is intentional for this version: the goal is a clean, repo-local AgiBot
residual RL runtime without dragging unrelated reference training/model code into
`examples/agibot_real`.
