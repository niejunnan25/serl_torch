# V14: Residual Runtime Minimal Abstraction

## Goal

Introduce a minimal residual runtime abstraction so actor / learner runtime no longer
directly depends on the concrete DRQ/SAC implementation.

This stage keeps the current training behavior and only changes the ownership boundary:

- runtime depends on `ResidualAgentRuntime`
- DRQ/SAC becomes one implementation of that runtime surface

## What Changed

Added:

- `serl_launcher/serl_launcher/residual/runtime/algorithm.py`
- `serl_launcher/serl_launcher/residual/runtime/sac_algorithm.py`

Key runtime changes:

- `actor_setup.py`
  - builds one `ResidualAgentRuntime`
  - uses it to construct actor / learner agents
  - passes it into async learner paths and agentlace bootstrap
- `actor_support.py`
  - uses algorithm snapshot / sync helpers instead of direct SAC helpers
- `agentlace_bridge.py`
  - accepts `ResidualAgentRuntime`
  - stores runtime-produced snapshot payloads in actor bootstrap
- `learner_service.py`
  - builds learner agent through the algorithm
  - applies / emits snapshot payloads through the algorithm
- `async_learning.py`
  - thread/process/agentlace async paths now all use algorithm methods for:
    - learner agent construction
    - actor action sampling
    - learner updates
    - actor sync
    - snapshot apply / snapshot export

## Why This Stage Matters

Before this stage, runtime still assumed:

- DRQ-SAC agent construction
- `sample_actions(...)`
- `update_high_utd(...)`
- direct checkpoint snapshot helpers

That meant future residual methods would still require invasive runtime edits.

After this stage:

- policy backend is one axis
- bindings are one axis
- residual runtime is now a third axis

This keeps the runtime aligned with the longer-term goal of supporting:

- `LIBERO`
- future `franka_real`
- multiple chunk-policy backends
- future non-SAC residual methods

## Review

Checks run:

- `python -m py_compile` on changed runtime files
- `git diff --check`
- `conda run -n serl_torch python examples/libero/scripts/run_learner.py --help`
- `conda run -n serl_torch python examples/libero/scripts/train_residual_sac.py --help`

Additional verification:

- searched runtime for remaining direct SAC coupling
- confirmed concrete SAC references now live in `sac_algorithm.py` plus the low-level
  helper implementations it intentionally wraps

## Residual Risk

- This stage does not yet define multiple residual algorithm implementations; it only
  introduces the interface and migrates current runtime to the SAC implementation.
- `checkpoint.py` and `config_utils.py` still contain low-level SAC-specific helpers,
  which is expected because `sac_algorithm.py` currently wraps them.
- No end-to-end training run was executed in this stage; validation is static plus
  entry-point smoke testing.

## Next Step

With policy, bindings, and residual algorithm all separated, the next highest-value work
is to continue shrinking `actor_loop.py` by splitting orchestration from chunk/step
episode execution paths.
