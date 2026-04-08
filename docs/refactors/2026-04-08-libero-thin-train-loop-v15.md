# V15: Split Episode Execution Out of `actor_loop.py`

## Goal

Continue shrinking `actor_loop.py` so it reads as phase orchestration rather than a
single monolithic rollout implementation.

This stage focuses on one boundary:

- `actor_loop.py` orchestrates phases / episode lifecycle / summary handling
- a dedicated helper module owns per-episode execution

## What Changed

Added:

- `serl_launcher/serl_launcher/residual/runtime/actor_episode.py`

Key change:

- Moved the per-episode execution body out of
  `serl_launcher/serl_launcher/residual/runtime/actor_loop.py`
  into:
  - `run_policy_episode(...)`
  - `EpisodeSpec`
  - `EpisodeResult`

The extracted helper now owns:

- probing rollout
- chunk-mode episode execution
- step-mode episode execution
- per-step logging
- replay insertion and learner update triggering
- per-step TB / profiling flush behavior
- in-episode checkpoint / async timing hooks

`actor_loop.py` now mainly owns:

- warmup handoff
- phase iteration
- reset / seed / init-state selection
- episode-level logging and async-eval trigger decisions
- final summary / shutdown

## Outcome

Line counts after this stage:

- `actor_loop.py`: `2150 -> 891`
- new `actor_episode.py`: `1212`

This is intentionally not the final split. The point of this stage is to make the main
actor loop read as orchestration, while preserving current runtime behavior.

## Review

Checks run:

- `python -m py_compile` on:
  - `actor_loop.py`
  - `actor_episode.py`
- `git diff --check`
- `conda run -n serl_torch python examples/libero/scripts/train_residual_sac.py --help`
- `conda run -n serl_torch python examples/libero/scripts/run_learner.py --help`
- AST parse smoke test for:
  - `actor_loop.py`
  - `actor_episode.py`

## Notes

- During implementation, the new helper initially still referenced actor-loop-local
  state for async update accounting and profiling counters. That was corrected before
  finalizing this stage by explicitly threading:
  - `advance_async_update_calls(...)`
  - `timer_train_episode_id`

## Residual Risk

- `actor_episode.py` is still large because chunk-mode and step-mode execution remain in
  the same module.
- The next split should separate:
  - chunk episode execution
  - step episode execution
  - shared in-episode update / metric helpers

## Next Step

The most natural continuation is:

- `V16`: split `actor_episode.py` into chunk-mode / step-mode helpers

That will let `actor_loop.py` stay orchestration-focused while also shrinking the new
episode module into clearer execution slices.
