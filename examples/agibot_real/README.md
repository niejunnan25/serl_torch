# AgiBot Real Residual RL

`examples/agibot_real` is the self-contained AgiBot real-robot residual RL example tree.

It is modeled on `examples/libero`, but it does not depend on
`reference/VLAPipeline_RL_BY_Niejunnan` at runtime. The reference repo was used only
to extract the minimum real-robot inference/runtime ideas that were needed here.

## What is included

- AgiBot real env wrappers:
  - local env: [task_env.py](/vla/users/niejunnan/codebase/serl_torch/examples/agibot_real/env_wrappers/task_env.py)
  - remote env: [remote_task_env.py](/vla/users/niejunnan/codebase/serl_torch/examples/agibot_real/env_wrappers/remote_task_env.py)
- Robot-facing runtime helpers:
  - [interface.py](/vla/users/niejunnan/codebase/serl_torch/examples/agibot_real/robot/interface.py)
  - [retargeter.py](/vla/users/niejunnan/codebase/serl_torch/examples/agibot_real/robot/retargeter.py)
  - [hooks.py](/vla/users/niejunnan/codebase/serl_torch/examples/agibot_real/robot/hooks.py)
- Residual RL runtime/data bindings:
  - [runtime_bindings.py](/vla/users/niejunnan/codebase/serl_torch/examples/agibot_real/runtime/runtime_bindings.py)
  - [data_bindings.py](/vla/users/niejunnan/codebase/serl_torch/examples/agibot_real/runtime/data_bindings.py)
  - [obs_adapter.py](/vla/users/niejunnan/codebase/serl_torch/examples/agibot_real/runtime/obs_adapter.py)
  - [policy_adapter.py](/vla/users/niejunnan/codebase/serl_torch/examples/agibot_real/runtime/policy_adapter.py)
- End-to-end scripts:
  - `scripts/train`
  - `scripts/eval`
  - `scripts/data`
  - `scripts/services`
- Shell tools:
  - `tools/*.sh`

## Robot/action assumptions

The current AgiBot implementation is intentionally narrow and opinionated:

- control mode: `camera_position`
- env action dimension: `14`
- state sent to OpenPI: 14D pose-only state
- images sent to OpenPI:
  - head image
  - left wrist image
  - right wrist image

That matches the reference `inference_pi05_camera_position.py` flow, but is reimplemented
inside this repo.

## External runtime dependencies

This example tree is self-contained inside this repo, but real execution still expects
the external runtime dependencies to be installed in the active environment:

- AgiBot SDK package providing `a2d_sdk.robot`
- kinematics dependencies used by [retargeter.py](/vla/users/niejunnan/codebase/serl_torch/examples/agibot_real/robot/retargeter.py)
- OpenPI serving environment for the base policy

Those are external packages/runtime services, not dependencies on the reference repo.

## Directory layout

```text
examples/agibot_real/
  assets/
  conf/
  env_wrappers/
  robot/
  runtime/
  scripts/
    train/
    eval/
    data/
    services/
  tools/
```

## Main entrypoints

- Train actor:
  - [run_actor.py](/vla/users/niejunnan/codebase/serl_torch/examples/agibot_real/scripts/train/run_actor.py)
- Train learner:
  - [run_learner.py](/vla/users/niejunnan/codebase/serl_torch/examples/agibot_real/scripts/train/run_learner.py)
- Launch async train stack:
  - [launch_async_train.py](/vla/users/niejunnan/codebase/serl_torch/examples/agibot_real/scripts/train/launch_async_train.py)
- Serve remote real env:
  - [serve_env.py](/vla/users/niejunnan/codebase/serl_torch/examples/agibot_real/scripts/services/serve_env.py)
- Evaluate checkpoint:
  - [evaluate_checkpoint.py](/vla/users/niejunnan/codebase/serl_torch/examples/agibot_real/scripts/eval/evaluate_checkpoint.py)
- Process async eval queue:
  - [process_eval_queue.py](/vla/users/niejunnan/codebase/serl_torch/examples/agibot_real/scripts/eval/process_eval_queue.py)
- Prepare offline demos:
  - [prepare_offline_demos.py](/vla/users/niejunnan/codebase/serl_torch/examples/agibot_real/scripts/data/prepare_offline_demos.py)
- Collect online prefill:
  - [collect_online_prefill.py](/vla/users/niejunnan/codebase/serl_torch/examples/agibot_real/scripts/data/collect_online_prefill.py)

## Typical workflow

### 1. Start an AgiBot env server

```bash
cd /vla/users/niejunnan/codebase/serl_torch/examples/agibot_real
bash tools/serve_env.sh --host 127.0.0.1 --port 32000
```

### 2. Start OpenPI for the base policy

```bash
cd /vla/users/niejunnan/codebase/serl_torch/examples/agibot_real
POLICY_CONFIG=pi05_agibot \
POLICY_DIR=/path/to/pi05_agibot/checkpoint \
bash tools/serve_openpi.sh --port 30001 --gpu-id 0
```

### 3. Launch async residual training

```bash
cd /vla/users/niejunnan/codebase/serl_torch/examples/agibot_real
bash tools/launch_async_train.sh \
  conf/train_residual_sac.yaml \
  hydra.run.dir=/vla/users/niejunnan/codebase/serl_torch/examples/agibot_real/outputs/train_default
```

### 4. Evaluate a residual checkpoint

```bash
cd /vla/users/niejunnan/codebase/serl_torch/examples/agibot_real
bash tools/eval.sh \
  conf/eval_residual_fast.yaml \
  eval.checkpoint_path=/path/to/checkpoint_2500.pt \
  hydra.run.dir=/vla/users/niejunnan/codebase/serl_torch/examples/agibot_real/outputs/eval_default
```

### 5. Convert offline demos into residual-training PKLs

```bash
cd /vla/users/niejunnan/codebase/serl_torch/examples/agibot_real
python scripts/data/prepare_offline_demos.py \
  --task_name agibot_pick_place \
  --prompt "Pick up the object with the right hand and place it at the target location." \
  --input_dir /path/to/demo_pkls \
  --chunk_horizon 5 \
  --policy_type openpi \
  --policy_id pi05_agibot \
  --openpi_host 127.0.0.1 \
  --openpi_port 30001 \
  --residual_alpha 0.2 \
  --output_dir /vla/users/niejunnan/codebase/serl_torch/examples/agibot_real/data/residual_training/offline
```

### 6. Collect online warmup/prefill data

```bash
cd /vla/users/niejunnan/codebase/serl_torch/examples/agibot_real
python scripts/data/collect_online_prefill.py \
  conf/train_residual_sac.yaml \
  --episodes 20 \
  --output_dir /vla/users/niejunnan/codebase/serl_torch/examples/agibot_real/data/residual_training/online \
  env.backend=remote \
  env.remote.host=127.0.0.1 \
  env.remote.port=32000 \
  openpi.host=127.0.0.1 \
  openpi.port=30001
```

## Hooks

Task-specific real-robot logic should be plugged in through hook functions instead of
hard-coding task behavior into the environment wrapper.

Supported hook fields in the config:

- `task.reset_hook`
- `task.success_hook`
- `task.expert_precheck_hook`

Each hook can be specified as:

- `module:function`
- `module.function`

See [hooks.py](/vla/users/niejunnan/codebase/serl_torch/examples/agibot_real/robot/hooks.py) for
the expected return conventions.

## Current limitations

- only `camera_position` mode is implemented
- only 14D pose-only residual actions are supported
- success/reward logic is hook-driven; there is no baked-in AgiBot task reward
- normalization is disabled by default

That is intentional for the first version: the goal here is to provide a clean,
repo-local AgiBot residual RL runtime foundation without dragging unrelated reference
training/model code into `examples/agibot_real`.
