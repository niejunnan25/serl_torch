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
- Serve optional remote env bridge: [`scripts/services/serve_env.py`](scripts/services/serve_env.py)
- Evaluate checkpoint: [`scripts/eval/evaluate_checkpoint.py`](scripts/eval/evaluate_checkpoint.py)
- Process async eval queue: [`scripts/eval/process_eval_queue.py`](scripts/eval/process_eval_queue.py)
- Prepare offline demos: [`scripts/data/prepare_offline_demos.py`](scripts/data/prepare_offline_demos.py)
- Collect online prefill: [`scripts/data/collect_online_prefill.py`](scripts/data/collect_online_prefill.py)

## How To Use

### TL;DR

If you are doing normal real-robot work, use the direct local-env path:

- `env.backend=local`
- start OpenPI in one terminal
- start `tools/run_learner.sh` in one terminal
- start `tools/run_actor.sh` in one terminal
- operate the robot from the actor terminal

Only use `tools/serve_env.sh` when you intentionally want a separate remote env
bridge. Only use `tools/launch_async_train.sh` for non-interactive debugging.

### 1. Choose the workflow

There are three ways to run `examples/agibot_real`:

1. Recommended real-robot path: direct local env.
   `env.backend=local`, no env server, actor/eval/prefill owns the robot env.
2. Optional remote bridge path: separate env server.
   `env.backend=remote`, `tools/serve_env.sh` owns the robot env and controller
   terminal.
3. Debug-only one-shot launcher.
   `tools/launch_async_train.sh` manages subprocesses for you, forces remote env
   defaults, and does not support interactive controller mode.

The default configs already point at the recommended path:

- `conf/train_residual_sac.yaml`: `env.backend=local`
- `conf/eval_residual_fast.yaml`: `env.backend=local`

### 2. Pick the wrapper you actually want

Training wrappers are split into two families:

- `tools/run_actor.sh` / `tools/run_learner.sh`
  Default training entrypoints. They are aliases to the agentlace split-process
  wrappers and are what you should use for the normal actor + learner workflow.
- `tools/run_actor_agentlace.sh` / `tools/run_learner_agentlace.sh`
  Explicit agentlace versions of the same split-process workflow.
- `tools/run_actor_generic.sh`
  Raw config-driven actor wrapper. It does not force `training.async.*`.
- `tools/run_learner_generic.sh`
  Raw config-driven learner wrapper. The current learner entrypoint is still
  agentlace-only, so this is mainly for advanced debugging and explicit config
  experiments.

Important:

- `tools/run_actor.sh` and `tools/run_learner.sh` intentionally override
  `training.async.*` to run the split actor/learner flow.
- `conf/train_residual_sac.yaml` itself still has `training.async.enabled=false`.
  That is fine. The default wrappers turn on agentlace for the split workflow.
- If you use the generic wrappers, you are responsible for making
  `training.async.*` consistent with the mode you actually want.

### 3. Environment selection

The shell wrappers can reuse the current shell env, or switch envs per command.

- `SERL_CONDA_ENV` or `SERL_CONDA_PREFIX`
  Used by training, eval, and data wrappers.
- `AGIBOT_CONDA_ENV` or `AGIBOT_CONDA_PREFIX`
  Used only by `tools/serve_env.sh` in the remote-bridge workflow.
- `OPENPI_CONDA_ENV` or `OPENPI_CONDA_PREFIX`
  Used by `tools/serve_openpi.sh`.
- `SERL_PYTHON_BIN` / `AGIBOT_PYTHON_BIN`
  Use these if you are not using conda.

For the recommended local-env workflow, the `SERL` env should be a merged env
that contains both:

- SERL training/runtime dependencies
- AgiBot runtime dependencies such as `a2d_sdk.robot` and kinematics packages

For the optional remote bridge workflow, keep:

- `SERL_CONDA_ENV` for actor/learner/eval/prefill
- `AGIBOT_CONDA_ENV` for the env server

If you want a local mirror for the vision backbone, set:

```bash
export SERL_RESNET_MODEL=relative/path/to/resnet-18
```

### 4. Recommended training workflow: direct local env

This is the main real-robot path and the one you should start from.

Use it when:

- actor runs on the robot machine
- you can maintain one merged Python env for SERL + AgiBot runtime
- you want the shortest control path

Prepare a shared bootstrap output path:

```bash
cd examples/agibot_real
mkdir -p outputs/agibot_real/train_default
```

Terminal A: start OpenPI.

```bash
cd examples/agibot_real
OPENPI_ROOT=relative/path/to/openpi \
OPENPI_CONDA_ENV=my_openpi_env \
POLICY_CONFIG=pi05_agibot \
POLICY_DIR=relative/path/to/pi05_agibot/checkpoint \
bash tools/serve_openpi.sh --port 30001 --gpu-id 0
```

Terminal B: start the learner.

```bash
cd examples/agibot_real
SERL_CONDA_ENV=my_robot_serl_env \
bash tools/run_learner.sh \
  conf/train_residual_sac.yaml \
  --bootstrap outputs/agibot_real/train_default/agentlace_bootstrap.pkl \
  hydra.run.dir=outputs/agibot_real/train_default/learner
```

Terminal C: start the actor. This terminal is also the controller console.

```bash
cd examples/agibot_real
SERL_CONDA_ENV=my_robot_serl_env \
bash tools/run_actor.sh \
  conf/train_residual_sac.yaml \
  --bootstrap outputs/agibot_real/train_default/agentlace_bootstrap.pkl \
  hydra.run.dir=outputs/agibot_real/train_default/actor
```

Notes:

- learner and actor must use the same `--bootstrap` path
- there is no env server in this workflow
- `controller.enabled=true` by default in the training config
- `chunk_step.enabled=true` by default in the training config
- keep the actor terminal in the foreground because it owns the live rollout

### 5. Controller keys

When `controller.enabled=true`, the terminal that owns the env accepts:

- `g`: ready / resume
- `p`: pause
- `r`: reset / truncate the current episode
- `s`: mark success
- `f`: mark failure
- `h`: print help again

The owner terminal depends on the backend:

- `env.backend=local`: actor, eval, or prefill terminal
- `env.backend=remote`: `tools/serve_env.sh` terminal

Current reset behavior is intentionally simple: `r` ends the episode and clears
queued actions, but it does not move the robot back to a home pose.

### 6. Optional training workflow: remote env bridge

Use this only when you explicitly want the robot env in a separate process, for
example:

- robot SDK dependencies must stay outside the SERL env
- you want a separate operator console
- you want the env process isolated from the actor process

Terminal A: start the env bridge.

```bash
cd examples/agibot_real
AGIBOT_CONDA_ENV=my_robot_env \
bash tools/serve_env.sh --host 127.0.0.1 --port 32000
```

Terminal B: start OpenPI.

```bash
cd examples/agibot_real
OPENPI_ROOT=relative/path/to/openpi \
OPENPI_CONDA_ENV=my_openpi_env \
POLICY_CONFIG=pi05_agibot \
POLICY_DIR=relative/path/to/pi05_agibot/checkpoint \
bash tools/serve_openpi.sh --port 30001 --gpu-id 0
```

Terminal C: start the learner.

```bash
cd examples/agibot_real
SERL_CONDA_ENV=my_serl_env \
bash tools/run_learner.sh \
  conf/train_residual_sac.yaml \
  --bootstrap outputs/agibot_real/train_default/agentlace_bootstrap.pkl \
  hydra.run.dir=outputs/agibot_real/train_default/learner
```

Terminal D: start the actor with remote overrides.

```bash
cd examples/agibot_real
SERL_CONDA_ENV=my_serl_env \
bash tools/run_actor.sh \
  conf/train_residual_sac.yaml \
  --bootstrap outputs/agibot_real/train_default/agentlace_bootstrap.pkl \
  hydra.run.dir=outputs/agibot_real/train_default/actor \
  env.backend=remote \
  env.remote.host=127.0.0.1 \
  env.remote.port=32000
```

In this workflow, Terminal A is the operator console.

### 7. Evaluate a checkpoint

For normal local-env evaluation, start OpenPI first and then run:

```bash
cd examples/agibot_real
SERL_CONDA_ENV=my_robot_serl_env \
bash tools/eval.sh \
  eval.checkpoint_path=outputs/checkpoint_2500.pt \
  hydra.run.dir=outputs/agibot_real/eval_default
```

`tools/eval.sh` respects the backend in the config or CLI overrides. For the
optional remote bridge path, add:

```bash
env.backend=remote env.remote.host=127.0.0.1 env.remote.port=32000
```

With controller mode enabled, the terminal that owns the env is also the
operator console.

### 8. Collect online prefill data

For normal local-env prefill, start OpenPI first and then run:

```bash
cd examples/agibot_real
SERL_CONDA_ENV=my_robot_serl_env \
bash tools/collect_online_prefill.sh \
  conf/train_residual_sac.yaml \
  --episodes 20 \
  --output_dir data/residual_training/online
```

`tools/collect_online_prefill.sh` also respects the backend in the config or
CLI overrides. For the optional remote bridge path, add:

```bash
env.backend=remote env.remote.host=127.0.0.1 env.remote.port=32000
```

With controller mode enabled, the terminal that owns the env is also the
operator console.

### 9. Convert offline demos

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

By default, exported offline data records `clip_residual_to_unit=true` so it
matches the default training config. If you want unclipped projection metadata,
pass `--no_clip_residual_to_unit` and make the training config match.

### 10. One-shot launcher

`tools/launch_async_train.sh` is a convenience launcher for non-interactive
debugging only.

It is not the recommended real-robot path because it backgrounds the env server
and OpenPI into managed subprocesses. That is a poor fit for human-in-the-loop
precheck, manual success confirmation, and safety/debugging workflows.

Current launcher behavior:

- defaults to `env.backend=remote`
- defaults to `controller.enabled=false`
- rejects `controller.enabled=true`

If you care about real operator interaction, use the explicit multi-terminal
workflow instead of the one-shot launcher.

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
