# LIBERO Thin Train Loop V2

## Modification Summary

- Added [serl_launcher/serl_launcher/residual/runtime/agentlace_bridge.py](/vla/users/niejunnan/codebase/serl_torch/serl_launcher/serl_launcher/residual/runtime/agentlace_bridge.py) to isolate actor-side agentlace concerns from the rollout runtime.
- Moved these responsibilities out of [actor_runtime.py](/vla/users/niejunnan/codebase/serl_torch/serl_launcher/serl_launcher/residual/runtime/actor_runtime.py):
  - actor bootstrap file generation
  - `_AgentlaceAsyncLearner` construction/startup
  - timer stats payload generation and send cadence
  - bounded-lag update target bookkeeping and wait logic
- Replaced repeated inline agentlace bootstrap / learner-construction blocks with shared bridge helpers.
- Centralized agentlace communication state into:
  - `AgentlaceBridgeConfig`
  - `AgentlaceBridgeState`

## Review Notes

- `py_compile` passed for:
  - [serl_launcher/serl_launcher/residual/runtime/agentlace_bridge.py](/vla/users/niejunnan/codebase/serl_torch/serl_launcher/serl_launcher/residual/runtime/agentlace_bridge.py)
  - [serl_launcher/serl_launcher/residual/runtime/actor_runtime.py](/vla/users/niejunnan/codebase/serl_torch/serl_launcher/serl_launcher/residual/runtime/actor_runtime.py)
  - [examples/libero/scripts/train_residual_sac.py](/vla/users/niejunnan/codebase/serl_torch/examples/libero/scripts/train_residual_sac.py)
  - [examples/libero/runtime/runtime_bindings.py](/vla/users/niejunnan/codebase/serl_torch/examples/libero/runtime/runtime_bindings.py)
- During extraction I corrected one important detail:
  - bounded-lag target update budgeting must still use `training.update_every` and `training.updates_per_step`
  - it must not silently switch to `training.async.update_frequency`

## Commit Summary

- Theme: separate actor-side agentlace communication from the residual rollout runtime.
- Result:
  - `actor_runtime.py` is less entangled with bootstrap, remote learner connection, and bounded-lag state
  - agentlace-specific logic now has a dedicated module and state model
