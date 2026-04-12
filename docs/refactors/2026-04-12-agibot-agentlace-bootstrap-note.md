# AgiBot Agentlace Bootstrap Note

## Context

We compared the original SERL Agentlace examples under
`reference/serl/examples/*` with the current `examples/agibot_real`
Agentlace path.

The specific question was whether `agibot_real` still needs:

- `serl_launcher.utils.agentlace_io.resolve_agentlace_bootstrap_path`
- `serl_launcher.utils.agentlace_io.save_agentlace_bootstrap`
- `serl_launcher.utils.agentlace_io.wait_for_agentlace_bootstrap`

or whether we can remove them and switch back to a more direct SERL-style
actor/learner handshake.

## What Original SERL Does

The original SERL examples do not use `agentlace_io` or a bootstrap file.

Their handshake is purely network-based:

- actor creates a `QueuedDataStore`
- actor creates a `TrainerClient(..., wait_for_server=True)`
- learner creates a `TrainerServer(...)`
- learner registers the replay/data store
- learner publishes network parameters with `publish_network(...)`
- actor receives parameter updates through `recv_network_callback(...)`
- actor sends online data through `TrainerClient.update()`

This works because both sides can independently construct the same training
contract from local code and config:

- observation structure
- action structure
- replay schema
- agent initialization inputs

In other words, original SERL does not need an out-of-band file handshake
because the actor and learner already agree on the tensor shapes and replay
layout before they connect.

## Why `agibot_real` Still Uses `agentlace_io`

`agibot_real` is not in that state yet.

Today, the learner side still depends on actor-produced bootstrap data before
it can finish initialization. The actor currently materializes and writes:

- `sample_obs`
- `state_core_dim`
- `env_action_dim`
- `step_action_dim`
- `agent_action_dim`
- `critic_action_dim`
- `image_keys`
- `action_transform`
- `chunk_step_enabled`
- `chunk_horizon`
- `state_mode`
- `initial_agent_payload`

The critical blockers are not the scalar config values. Most of those can be
re-derived from config.

The real blockers are:

- `sample_obs`
- `state_core_dim`

Right now, `sample_obs` is produced by the actor through:

1. `env.reset()`
2. `build_residual_step_obs(...)`
3. `build_residual_step_core(...)`

The learner then consumes that bootstrap payload before it can build:

- the residual agent
- the chunk replay buffer
- the observation space template

So the current learner is not yet self-sufficient in the same way the original
SERL learner is.

## Current Judgment

For `agibot_real`, `agentlace_io` should stay for now.

Removing it immediately would break learner startup because the learner still
expects actor-generated template data before it can initialize replay and the
agent.

So the answer is:

- original SERL: does not need bootstrap files
- current `agibot_real`: still needs them

## Desired Direction

The long-term direction is still to move closer to the original SERL style.

The right way to get there is not to make the learner touch the real robot
environment. Instead, we should introduce a shared pure helper that both actor
and learner can call locally, for example:

- build AgiBot residual sample observation template from config
- build `state_core_dim` from config/observation contract
- build replay/action-transform template from config

After that exists:

- actor and learner can independently build the same sample template
- `initial_agent_payload` can be replaced by normal `publish_network(...)`
  synchronization
- the bootstrap file can likely be removed

## Deferred Decision

We are not changing this now.

Current decision:

- keep `agentlace_io` for `agibot_real`
- do not attempt to remove bootstrap file flow yet
- revisit only after the residual sample template becomes a shared pure
  contract instead of an actor-materialized runtime artifact
