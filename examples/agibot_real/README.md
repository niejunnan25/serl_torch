# AgiBot Real Residual RL

`examples/agibot_real` is the repo-local AgiBot real-robot residual RL example tree.

It is modeled on `examples/libero`, but it does not import or execute code from
`reference/VLAPipeline_RL_BY_Niejunnan` at runtime. The reference repo was only
used to extract the minimum real-robot inference ideas needed here.

## What is included

- AgiBot env wrappers:
  - local env: [`env_wrappers/task_env.py`](env_wrappers/task_env.py)
  - remote env: [`env_wrappers/remote_task_env.py`](env_wrappers/remote_task_env.py)
- Robot-facing runtime helpers:
  - [`robot/interface.py`](robot/interface.py)
  - [`robot/retargeter.py`](robot/retargeter.py)
  - [`robot/hooks.py`](robot/hooks.py)
- Residual RL runtime/data bindings:
  - [`runtime/runtime_bindings.py`](runtime/runtime_bindings.py)
  - [`runtime/data_bindings.py`](runtime/data_bindings.py)
  - [`runtime/obs_adapter.py`](runtime/obs_adapter.py)
  - [`runtime/policy_adapter.py`](runtime/policy_adapter.py)
- End-to-end scripts:
  - `scripts/train`
  - `scripts/eval`
  - `scripts/data`
  - `scripts/services`
- Shell tools:
  - `tools/*.sh`

## Robot/action assumptions

The current AgiBot implementation is intentionally narrow:

- control mode: `camera_position`
- env action dimension: `14`
- state sent to OpenPI: 14D pose-only state
- images sent to OpenPI:
  - head image
  - left wrist image
  - right wrist image

That matches the reference `inference_pi05_camera_position.py` flow, but is
reimplemented inside this repo.

## External runtime dependencies

This example tree is self-contained inside this repo, but real execution still
expects these external runtime dependencies in the active environment:

- AgiBot SDK package providing `a2d_sdk.robot`
- kinematics dependencies used by [`robot/retargeter.py`](robot/retargeter.py)
- an OpenPI serving environment for the base policy

These are external packages/runtime services, not dependencies on the reference
repo.

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

## Path convention

The examples below intentionally use repo-relative paths or user-supplied
relative paths.

- If you `cd examples/agibot_real`, then paths like `conf/train_residual_sac.yaml`
  or `data/residual_training/offline` are relative to that directory.
- For external assets that live outside this repo, pass them explicitly with
  env vars or CLI args instead of editing hard-coded absolute paths into files.

## Main entrypoints

- Train actor: [`scripts/train/run_actor.py`](scripts/train/run_actor.py)
- Train learner: [`scripts/train/run_learner.py`](scripts/train/run_learner.py)
- Launch async train stack: [`scripts/train/launch_async_train.py`](scripts/train/launch_async_train.py)
  for non-interactive debugging only
- Serve remote real env: [`scripts/services/serve_env.py`](scripts/services/serve_env.py)
- Evaluate checkpoint: [`scripts/eval/evaluate_checkpoint.py`](scripts/eval/evaluate_checkpoint.py)
- Process async eval queue: [`scripts/eval/process_eval_queue.py`](scripts/eval/process_eval_queue.py)
- Prepare offline demos: [`scripts/data/prepare_offline_demos.py`](scripts/data/prepare_offline_demos.py)
- Collect online prefill: [`scripts/data/collect_online_prefill.py`](scripts/data/collect_online_prefill.py)

## Typical workflows

### Environment selection

The shell wrappers are written to work across different machines and conda
layouts:

- If your target env is already active, the wrappers reuse the current shell
  environment.
- To switch envs per command without changing your current shell, set env vars:
  - `SERL_CONDA_ENV` or `SERL_CONDA_PREFIX` for training/eval/data scripts
  - `AGIBOT_CONDA_ENV` or `AGIBOT_CONDA_PREFIX` for the real env server
  - `OPENPI_CONDA_ENV` or `OPENPI_CONDA_PREFIX` for OpenPI serving
- If you are not using conda, point the wrappers at a Python binary:
  - `SERL_PYTHON_BIN`
  - `AGIBOT_PYTHON_BIN`

These env vars are consumed by the `tools/*.sh` wrappers. If you invoke
`python ...` directly, activate the target environment yourself first.

Examples:

```bash
cd examples/agibot_real
SERL_CONDA_ENV=my_serl_env bash tools/eval.sh eval.checkpoint_path=outputs/checkpoint_2500.pt
```

```bash
cd examples/agibot_real
AGIBOT_CONDA_ENV=my_robot_env bash tools/serve_env.sh --host 127.0.0.1 --port 32000
```

```bash
cd examples/agibot_real
OPENPI_CONDA_ENV=my_openpi_env \
OPENPI_ROOT=relative/path/to/openpi \
bash tools/serve_openpi.sh --port 30001 --gpu-id 0
```

For the vision backbone, the configs default to Hugging Face model id
`microsoft/resnet-18`. If you want a local mirror or an offline snapshot, set:

```bash
export SERL_RESNET_MODEL=relative/path/to/resnet-18
```

### 1. Required real-robot training workflow: start components separately

For real-robot training, start the stack in separate terminals.

This is the required workflow for `examples/agibot_real`. Do not treat the
one-shot launcher as the standard real-robot entrypoint.

Reason:

- real-robot training often needs human-in-the-loop `precheck` and
  `success_hook`
- controller-driven training needs the env process to stay attached to the
  operator terminal
- the env process should stay independently visible and controllable
- debugging reset/safety/manual-success behavior is much easier when env,
  OpenPI, learner, and actor are split apart

Prepare a shared bootstrap path that both learner and actor can see:

```bash
cd examples/agibot_real
mkdir -p outputs/agibot_real/train_default
```

Terminal A: start the real env server.

```bash
cd examples/agibot_real
AGIBOT_CONDA_ENV=my_robot_env \
bash tools/serve_env.sh --host 127.0.0.1 --port 32000
```

If `controller.enabled=true`, Terminal A is also the operator console. The env
server listens for single-key commands:

- `g`: ready / resume
- `p`: pause
- `r`: mark the current episode as reset / truncated
- `s`: mark the current episode as success
- `f`: mark the current episode as fail
- `h`: print the key help again

The first version of controller mode assumes reset is human-operated. Pressing
`r` stops the current episode and clears queued actions, but it does not move
the robot back to a home pose automatically.

Terminal B: start OpenPI for the base policy.

```bash
cd examples/agibot_real
OPENPI_ROOT=relative/path/to/openpi \
OPENPI_CONDA_ENV=my_openpi_env \
POLICY_CONFIG=pi05_agibot \
POLICY_DIR=relative/path/to/pi05_agibot/checkpoint \
bash tools/serve_openpi.sh --port 30001 --gpu-id 0
```

`tools/serve_openpi.sh` respects `OPENPI_CONDA_PREFIX` or `OPENPI_CONDA_ENV`.
`OPENPI_ROOT` must point to your local OpenPI checkout. It can be a relative
path from the current working directory or an absolute path if you prefer.

Terminal C: start the learner.

```bash
cd examples/agibot_real
SERL_CONDA_ENV=my_serl_env \
bash tools/run_learner.sh \
  conf/train_residual_sac.yaml \
  --bootstrap outputs/agibot_real/train_default/agentlace_bootstrap.pkl \
  hydra.run.dir=outputs/agibot_real/train_default/learner
```

Terminal D: start the actor.

```bash
cd examples/agibot_real
SERL_CONDA_ENV=my_serl_env \
bash tools/run_actor.sh \
  conf/train_residual_sac.yaml \
  --bootstrap outputs/agibot_real/train_default/agentlace_bootstrap.pkl \
  hydra.run.dir=outputs/agibot_real/train_default/actor
```

Notes:

- learner and actor must point to the same `--bootstrap` path
- actor is the live robot rollout process; keep it in the foreground
- if you use manual terminal input or any manual success interface, attach it to
  the env side, not the learner side
- with `controller.enabled=true`, actor runs in controller-driven chunk mode and
  expects `chunk_step.enabled=true`

### Controller-driven training semantics

When `controller.enabled=true`, the training stack uses the env-side controller
as the runtime source of truth.

- `reset()` starts a new episode record and returns to `WAIT_READY`
- pressing `g` transitions the env into `RUNNING`
- actor only enqueues a chunk when the controller is in `RUNNING`
- the env executes queued actions at robot control frequency in its own control
  loop
- `p` pauses without ending the episode
- `r` ends the episode as a truncated reset
- `s` ends the episode with reward `1`
- `f` ends the episode with reward `0`
- step-limit timeout ends the episode as truncated with reward `0`

This mode is intentionally different from the old synchronous step/step-chunk
runtime. The goal is to avoid a "step once, wait once" control pattern on the
real robot.

### 2. Evaluate a residual checkpoint

```bash
cd examples/agibot_real
SERL_CONDA_ENV=my_serl_env \
bash tools/eval.sh \
  eval.checkpoint_path=outputs/checkpoint_2500.pt \
  hydra.run.dir=outputs/agibot_real/eval_default
```

This expects the env server and OpenPI server to already be running unless you
are evaluating inside another orchestration flow.

### 3. Convert offline demos into residual-training PKLs

```bash
cd examples/agibot_real
SERL_CONDA_ENV=my_serl_env \
bash tools/prepare_offline_demos.sh \
  --task_name agibot_pick_place \
  --prompt "Pick up the object with the right hand and place it at the target location." \
  --input_dir data/raw_demos \
  --chunk_horizon 5 \
  --policy_type openpi \
  --policy_id pi05_agibot \
  --openpi_host 127.0.0.1 \
  --openpi_port 30001 \
  --residual_alpha 0.2 \
  --output_dir data/residual_training/offline
```

By default, exported offline data now records `clip_residual_to_unit=true` so it
matches the default training config. If you intentionally want unclipped
projection metadata, pass `--no_clip_residual_to_unit` and make the training
config match.

### 4. Collect online warmup/prefill data

```bash
cd examples/agibot_real
SERL_CONDA_ENV=my_serl_env \
bash tools/collect_online_prefill.sh \
  conf/train_residual_sac.yaml \
  --episodes 20 \
  --output_dir data/residual_training/online \
  env.backend=remote \
  env.remote.host=127.0.0.1 \
  env.remote.port=32000 \
  openpi.host=127.0.0.1 \
  openpi.port=30001
```

### 5. One-shot launcher

`tools/launch_async_train.sh` still exists, but treat it as a convenience
launcher for non-interactive debugging only.

It is not the recommended real-robot training path because it backgrounds the
env server and OpenPI into managed subprocesses. That is a poor fit for
human-in-the-loop `precheck`, manual success confirmation, and safety/debugging
workflows.

If you still use the one-shot launcher, pass `controller.enabled=false`.
The launcher now refuses `controller.enabled=true` because the env server would
otherwise start without an operator-attached TTY.

## Hooks

Task-specific real-robot logic should be plugged in through hook functions
instead of hard-coding task behavior into the environment wrapper.

Supported hook fields in the config:

- `task.reset_hook`
- `task.success_hook`
- `task.expert_precheck_hook`

Each hook can be specified as:

- `module:function`
- `module.function`

See [`robot/hooks.py`](robot/hooks.py) for the expected return conventions.

## Current limitations

- only `camera_position` mode is implemented
- only 14D pose-only residual actions are supported
- success/reward logic is hook-driven; there is no baked-in AgiBot task reward
- normalization is disabled by default

That is intentional for the first version: the goal here is to provide a clean,
repo-local AgiBot residual RL runtime foundation without dragging unrelated
reference training/model code into `examples/agibot_real`.
