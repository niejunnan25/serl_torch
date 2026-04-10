# AgiBot Real Residual RL

`examples/agibot_real` is the repo-local AgiBot real-robot residual RL example
tree.

It is modeled on `examples/libero`, but it does not import or execute code from
`reference/VLAPipeline_RL_BY_Niejunnan` at runtime. The reference repo was only
used to extract the minimum real-robot inference ideas needed here.

This tree is now local-only:

- the robot env always lives in the process that runs actor, eval, or prefill
- there is no remote env bridge
- there is no separate env server workflow

## What is included

- Local AgiBot env wrapper:
  - [`env_wrappers/task_env.py`](env_wrappers/task_env.py)
  - [`env_wrappers/controller.py`](env_wrappers/controller.py)
  - [`env_wrappers/factory.py`](env_wrappers/factory.py)
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
- Evaluate checkpoint: [`scripts/eval/evaluate_checkpoint.py`](scripts/eval/evaluate_checkpoint.py)
- Process async eval queue: [`scripts/eval/process_eval_queue.py`](scripts/eval/process_eval_queue.py)
- Prepare offline demos: [`scripts/data/prepare_offline_demos.py`](scripts/data/prepare_offline_demos.py)
- Collect online prefill: [`scripts/data/collect_online_prefill.py`](scripts/data/collect_online_prefill.py)

## TL;DR

For normal real-robot work:

1. Start OpenPI in one terminal.
2. Start `tools/run_learner.sh` in one terminal.
3. Start `tools/run_actor.sh` in one terminal.
4. Operate the robot from the actor terminal.

The actor, eval, and prefill flows all own the live robot env directly.

## Wrapper selection

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

## Environment selection

The shell wrappers can reuse the current shell env, or switch envs per command.

- `SERL_CONDA_ENV` or `SERL_CONDA_PREFIX`
  Used by training, eval, and data wrappers.
- `OPENPI_CONDA_ENV` or `OPENPI_CONDA_PREFIX`
  Used by [`tools/serve_openpi.sh`](tools/serve_openpi.sh).
- `SERL_PYTHON_BIN`
  Use this if you are not using conda.

The `SERL` env should be a merged env that contains both:

- SERL training/runtime dependencies
- AgiBot runtime dependencies such as `a2d_sdk.robot` and kinematics packages

If you want a local mirror for the vision backbone, set:

```bash
export SERL_RESNET_MODEL=relative/path/to/resnet-18
```

## Recommended training workflow

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
- `controller.enabled=true` by default in the training config
- `chunk_step.enabled=true` by default in the training config
- keep the actor terminal in the foreground because it owns the live rollout

## Controller keys

When `controller.enabled=true`, the terminal that owns the env accepts:

- `g`: ready / resume
- `p`: pause
- `r`: reset / truncate the current episode
- `s`: mark success
- `f`: mark failure
- `h`: print help again

Current reset behavior is intentionally simple: `r` ends the episode and clears
queued actions, but it does not move the robot back to a home pose.

## Evaluate a checkpoint

Start OpenPI first and then run:

```bash
cd examples/agibot_real
SERL_CONDA_ENV=my_robot_serl_env \
bash tools/eval.sh \
  eval.checkpoint_path=outputs/checkpoint_2500.pt \
  hydra.run.dir=outputs/agibot_real/eval_default
```

With controller mode enabled, the eval terminal is also the operator console.

## Collect online prefill data

Start OpenPI first and then run:

```bash
cd examples/agibot_real
SERL_CONDA_ENV=my_robot_serl_env \
bash tools/collect_online_prefill.sh \
  conf/train_residual_sac.yaml \
  --episodes 20 \
  --output_dir data/residual_training/online
```

With controller mode enabled, the prefill terminal is also the operator console.

## Convert offline demos

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

## Async eval

[`scripts/eval/process_eval_queue.py`](scripts/eval/process_eval_queue.py) is
still available, but it now launches a separate local eval process. Keep
`training.async_eval.enabled=false` unless that eval worker can own the robot
exclusively.

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
