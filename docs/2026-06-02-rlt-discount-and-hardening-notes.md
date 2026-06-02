# RLT Discount Semantics and Hardening Notes

Date: 2026-06-02

## Issue

The first runnable LIBERO RLT integration used `chunk_size=10` but executed only
the first 5 actions before replanning. The actor therefore stored replay
transitions of the form:

```text
s_t -- execute 5 environment steps --> s_{t+5}
```

The critic target previously used:

```python
discount_chunk = gamma ** chunk_size
target = reward + (1 - done) * discount_chunk * Q(next_state, next_action)
```

With `gamma=0.99` and `chunk_size=10`, this bootstrapped with `0.99 ** 10`,
even though the replay transition usually advanced only 5 environment steps.
That made the bootstrap discount inconsistent with the stored transition.

## Chosen Fix

RLT now separates the model output horizon from the execution horizon:

- `rlt.chunk_size`: number of actions predicted by the actor.
- `rlt.execute_horizon`: number of predicted actions executed before replanning.

The actor stores per-transition metadata:

```python
executed_steps = number_of_environment_steps_actually_executed
discounts = gamma ** executed_steps
```

The learner uses `discounts` from the replay batch when building the TD target:

```python
target = reward + (1 - done) * discounts * Q(next_state, next_action)
```

For normal non-terminal chunks this is equivalent to
`gamma ** rlt.execute_horizon`. If a chunk terminates early, the stored discount
matches the shorter executed length. Terminal transitions still disable
bootstrapping through the `done` flag.

Reward aggregation remains the raw sum over executed steps for now. A stricter
SMDP implementation could store discounted reward sums, but that is a separate
algorithm change and should be evaluated independently.

## Hardening Checklist

- Keep `rlt.execute_horizon` visible in configs and validate it is no larger
  than `rlt.chunk_size`.
- Keep actor and learner in the same run directory, but only let the learner
  write Hydra metadata. The launch script passes `hydra.output_subdir=null` to
  the actor.
- Let the actor send `actor_finished=true` when it exits normally so the
  learner can stop once it catches up to the final environment step.
- Skip next-state VLA inference on terminal transitions because terminal TD
  targets do not bootstrap from the next state.
- Keep async-eval checkpoint retention wired to
  `training.async_eval.checkpoint.keep`; a non-positive value keeps all eval
  checkpoints.
- Keep launch-time experiment changes as Hydra overrides after `--`, rather
  than copying configs for every step count or warmup setting.
- Keep rollout video saving disabled by default; use `logging.save_videos=true`
  only for debugging.
- Normalize VLA feature-client outputs to writable `float32` arrays before
  passing them into torch.
- Avoid smoke/debug code in the training path, including duplicate stats
  requests and dependency-disabling hacks.
- Keep generated files out of git status: `__pycache__`, `*.pyc`, Hydra
  outputs, wandb runs, videos, and transient `*.INFO` logs.
