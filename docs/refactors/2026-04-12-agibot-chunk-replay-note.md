# AgiBot Chunk And Replay Note

## Context

This note records the current discussion around chunk handling in
`examples/agibot_real`, especially the difference between:

- the original SERL examples under `reference/serl/examples/*`
- the current AgiBot residual training path

The goal is not to force an immediate code change. The goal is to clearly
capture what the current design is doing, why it feels heavier than the
reference examples, and what tradeoffs exist if we later change it.

## Main Questions

We wanted to answer these questions:

1. Are the current `serl_launcher.residual.action` helpers really residual
   logic, or are some of them just generic chunk helpers?
2. How does original SERL handle chunking?
3. Why does AgiBot chunk logic feel much heavier than original SERL?
4. Why was replay originally implemented as step-level storage?
5. Should AgiBot continue to store executed steps, or switch to chunk-level
   replay?

## Current Action Helpers

In `examples/agibot_real/scripts/train/run_actor_reference_style_residual.py` we
currently import:

- `compose_residual_action_chunk`
- `reshape_flat_action_to_chunk`
- `select_action_chunk_window`

These are not all the same kind of logic.

### Truly Residual Logic

`compose_residual_action_chunk(...)` is genuinely residual-specific.

It does the core residual operation:

- take a `base_chunk`
- take a `residual_chunk`
- apply residual limits / indices / alpha
- produce the final executable chunk

This is the real:

- `base + residual -> final`

contract.

### Generic Chunk / Shape Logic

The other two helpers are not really residual-specific:

- `reshape_flat_action_to_chunk(...)`
- `select_action_chunk_window(...)`

Their job is more generic:

- reshape a flat action vector into `[chunk_horizon, action_dim]`
- take a returned action chunk and keep only the first `horizon` steps

These helpers are chunk utilities more than residual utilities.

### Current Judgment

If we later clean up module boundaries, a reasonable split would be:

- keep `compose_residual_action*` under a residual module
- move `reshape_flat_action_to_chunk(...)` and
  `select_action_chunk_window(...)` into a shared chunk/action helper module

## How Original SERL Handles Chunking

Original SERL has a `ChunkingWrapper`:

- `serl_launcher/serl_launcher/wrappers/chunking.py`

This wrapper supports two concepts:

- `obs_horizon`
- `act_exec_horizon`

### `obs_horizon`

When `obs_horizon > 1`, the wrapper stacks recent observations over time.

This is observation history stacking, not action chunking.

### `act_exec_horizon`

When `act_exec_horizon is not None`, one high-level `env.step(action_chunk)`
executes multiple low-level actions internally.

This means:

- one RL step
- multiple executed environment actions inside that step

It does **not** mean the policy is re-sampled multiple times.

More precisely:

- the policy makes one high-level decision
- that decision contains a chunk of actions
- the wrapper executes the first `act_exec_horizon` actions in order

### What The Reference Examples Actually Do

The important observation is that the reference examples we inspected
generally do **not** turn on action chunk execution.

For example:

- `reference/serl/examples/async_bin_relocation_fwbw_drq/async_drq_randomized.py`

uses:

```python
env = ChunkingWrapper(env, obs_horizon=1, act_exec_horizon=None)
```

So in practice the reference actor loop is still:

- sample one action
- call `env.step(action)`
- write one standard transition

This is the key reason the reference loop looks much simpler.

It is not that the reference repo solved the full chunk problem elegantly.
It is that the reference examples mostly avoid enabling action chunk execution
in the first place.

## Why AgiBot Chunk Logic Feels Much Heavier

The AgiBot path is not just "chunking". It combines several layers of
complexity:

1. A base policy produces a `base_chunk`
2. A residual policy produces a `residual_chunk`
3. Both are composed into a `final_chunk`
4. The environment executes through `env.step_chunk(...)`
5. The execution returns a stream of executed steps / rewards / terminal info
6. Replay logic reconstructs training samples from that execution stream

So AgiBot is not dealing with the same problem as reference SERL.

Reference examples look like:

- `action -> env.step(action) -> one transition`

AgiBot looks like:

- `base_chunk + residual_chunk -> final_chunk -> env.step_chunk(final_chunk) -> executed-step stream -> replay logic`

This is why AgiBot feels much heavier.

## Current Replay Design In AgiBot

The current replay implementation is:

- step-level storage
- chunk-level sampling

The main class is:

- `serl_launcher/serl_launcher/residual/train/step_chunk_replay.py`

Its class docstring says:

> Store step-level rollout data and sample chunk windows with env-step stride.

That is exactly what it does.

### What Gets Inserted

The buffer stores one record per executed step, including fields such as:

- `obs_core`
- `base_action`
- `base_action_norm`
- `actions`
- `rewards`
- `dones`
- `alpha`
- `episode_id`
- `episode_step`

So it does **not** directly store one replay record per high-level chunk
decision.

### What Gets Sampled

At sample time, the buffer does not sample isolated steps.

Instead, it:

- chooses a valid `start_step_id`
- walks forward for up to `chunk_horizon` consecutive executed steps
- verifies episode continuity and stride constraints
- rebuilds:
  - chunk-level observations
  - flattened chunk actions
  - discounted chunk reward
  - next observation after the chunk

So the real training sample is still chunk-shaped.

The design is:

- **step-level storage**
- **chunk-level training samples**

## Why It Was Probably Implemented This Way

The current design strongly suggests that step-level storage was chosen to
support sliding-window chunk learning.

That means:

- the actor stores the executed step stream
- the learner can later sample chunk windows starting from many different step
  positions

For example, if the executed stream is:

- `s0 -> s1 -> s2 -> s3 -> s4 -> s5 -> ...`

and `chunk_horizon = 5`, then the learner can build chunk samples from:

- start at `s0`
- start at `s1`
- start at `s2`
- etc.

This gives more overlapping chunk samples from one real rollout.

### What This Buys You

The main benefits are:

- higher data reuse from the same rollout
- natural sliding-window chunk sampling
- better handling of early termination inside a chunk
- more detailed chunk reward / terminal reconstruction

### What This Costs

The main costs are:

- more complex actor code
- more complex replay schema
- more complex learner sampling logic
- more cognitive mismatch, because the policy outputs chunks but replay is
  written one executed step at a time

## Alternative Design: Store Replay By Chunk

A simpler alternative is:

- one high-level decision
- one replay record

Under that design, one replay item would directly represent:

- current observation
- base chunk
- residual chunk
- final chunk
- next observation after executing the chunk
- chunk reward
- done / truncated / info

### Why This Is Attractive

This would make the system much closer to the reference examples:

- sample action
- step environment
- store one transition

It also matches the current policy contract more naturally, because the actor
already outputs chunk-shaped actions.

### What Would Be Lost

The main thing we would lose is sliding-window reuse of the executed step
stream.

We would also lose some fine-grained chunk-internal detail unless we keep a
summary inside `info`.

## Tradeoff Summary

### Step-Level Storage + Chunk Sampling

Pros:

- better data reuse
- supports sliding windows naturally
- preserves more executed-step detail
- flexible around early termination inside a chunk

Cons:

- heavier actor logic
- heavier learner / replay logic
- training contract is less obvious at a glance

### Direct Chunk-Level Replay

Pros:

- much clearer actor loop
- much closer to original SERL style
- action semantics match policy output more directly
- easier replay contract to reason about

Cons:

- fewer overlapping training samples per rollout
- less fine-grained chunk-internal detail
- loses some of the benefits of step-stream reconstruction

## Current Judgment

At this point, the design tension is clear:

- if the priority is **sample efficiency / sliding-window reuse**, then the
  current step-level storage with chunk sampling makes sense
- if the priority is **clarity / reference-style simplicity / cleaner actor
  contract**, then chunk-level replay is more attractive

For the current AgiBot residual path, chunk-level replay appears more natural
from a high-level modeling perspective because:

- the actor decision is chunk-level
- the policy output is chunk-level
- the base policy output is chunk-level
- the residual composition contract is chunk-level

But the current implementation was likely shaped by the desire to keep
step-stream information and reuse it through sliding-window chunk sampling.

So the current system is not "wrong". It is a particular tradeoff:

- more complicated system
- potentially more useful samples per rollout

## Deferred Decision

We are not changing this immediately.

For now, the main conclusion is:

- the current step-level replay design was likely chosen to support sliding
  chunk windows over executed steps
- original SERL examples are simpler largely because they do not actually
  enable action chunk execution in the inspected training scripts
- if we later want to simplify the AgiBot training loop, replay storage policy
  is one of the highest-leverage design choices to revisit
