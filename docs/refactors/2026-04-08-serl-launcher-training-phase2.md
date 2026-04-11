# SERL Launcher Training Phase 2

## Goal

Move residual training orchestration out of `serl_launcher.residual.runtime`
into a clearer `serl_launcher.residual.train` package while leaving the Phase 3
mixed infrastructure files in place.

## Moved Files

### Residual Train Core

- `serl_launcher/serl_launcher/residual/runtime/bindings.py`
  -> `serl_launcher/serl_launcher/residual/train/bindings.py`
- `serl_launcher/serl_launcher/residual/runtime/async_eval.py`
  -> `serl_launcher/serl_launcher/residual/train/async_eval.py`
- `serl_launcher/serl_launcher/residual/runtime/pretrain.py`
  -> `serl_launcher/serl_launcher/residual/train/pretrain.py`
- `serl_launcher/serl_launcher/residual/runtime/schedules.py`
  -> `serl_launcher/serl_launcher/residual/train/schedules.py`
- `serl_launcher/serl_launcher/residual/runtime/step_chunk_replay.py`
  -> `serl_launcher/serl_launcher/residual/train/step_chunk_replay.py`
- `serl_launcher/serl_launcher/residual/runtime/obs_utils.py`
  -> `serl_launcher/serl_launcher/residual/train/obs_utils.py`

### Actor Orchestration

- `serl_launcher/serl_launcher/residual/runtime/actor_runtime.py`
  -> `serl_launcher/serl_launcher/residual/train/actor/runtime.py`
- `serl_launcher/serl_launcher/residual/runtime/actor_setup.py`
  -> `serl_launcher/serl_launcher/residual/train/actor/setup.py`
- `serl_launcher/serl_launcher/residual/runtime/actor_support.py`
  -> `serl_launcher/serl_launcher/residual/train/actor/support.py`
- `serl_launcher/serl_launcher/residual/runtime/actor_warmup.py`
  -> `serl_launcher/serl_launcher/residual/train/actor/warmup.py`
- `serl_launcher/serl_launcher/residual/runtime/actor_loop.py`
  -> `serl_launcher/serl_launcher/residual/train/actor/loop.py`
- `serl_launcher/serl_launcher/residual/runtime/actor_episode.py`
  -> `serl_launcher/serl_launcher/residual/train/actor/episode.py`
- `serl_launcher/serl_launcher/residual/runtime/actor_episode_shared.py`
  -> `serl_launcher/serl_launcher/residual/train/actor/episode_shared.py`
- `serl_launcher/serl_launcher/residual/runtime/actor_episode_chunk.py`
  -> `serl_launcher/serl_launcher/residual/train/actor/episode_chunk.py`
- `serl_launcher/serl_launcher/residual/runtime/actor_episode_step.py`
  -> `serl_launcher/serl_launcher/residual/train/actor/episode_step.py`

### Learner Orchestration

- `serl_launcher/serl_launcher/residual/runtime/learner_service.py`
  -> `serl_launcher/serl_launcher/residual/train/learner/service.py`

## Package Shape After Phase 2

- `serl_launcher.residual.runtime_agent`
  holds residual DRQ/SAC runtime helpers and snapshot/sync utilities.
- `serl_launcher.residual.train`
  holds residual training orchestration and residual-train-specific helpers.
- `serl_launcher.training`
  holds generic training infrastructure introduced in Phase 1.
- `serl_launcher.residual.runtime`
  now only holds the Phase 3 mixed files that still need splitting:
  `agentlace_bridge.py`, `async_learning.py`, `config_utils.py`,
  `tb_metrics.py`, and `train_loop_utils.py`.

## Review

I checked:

- `python -m py_compile` on the moved train files and the updated LIBERO
  entrypoints
- `conda run -n serl_torch python examples/libero/scripts/train_residual_sac.py --help`
- `conda run -n serl_torch python examples/libero/scripts/run_learner.py --help`
- `conda run -n serl_torch python examples/libero/scripts/eval_residual_fast.py --help`
- `git diff --check`
- `rg` to confirm there are no remaining Python imports that still reference the
  moved `serl_launcher.residual.runtime.*` train modules

## Outcome

Phase 2 finishes the package move for residual training orchestration without
touching the Phase 3 mixed infrastructure files. The residual package shape is
now much closer to the intended split:

- residual semantics
- residual runtime helpers
- residual training orchestration

instead of a single overloaded `residual.runtime` bucket.
