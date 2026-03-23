# LIBERO Chunk-Step Review And PLD Alignment Notes

Date: 2026-03-23

This note summarizes the current residual RL implementation after the latest review and refactor. It covers:

1. what was changed,
2. why those changes were necessary,
3. how the current step-level and chunk-step behavior works,
4. what is still not fully aligned with PLD Stage 1,
5. why the implementation now uses `critic(final action)` + `actor(residual action)`.

Update:
- The chunk-step path has since been switched back to step-stream replay with
  sample-time chunk-window assembly.
- `chunk_step.sample_stride` is now interpreted in env-step units, matching the
  `Subsampling Action Chunks` behavior described in RLT.

## 1. Executive Summary

There are now two clearly separated interpretations in this codebase.

1. `train_residual_sac.yaml`
   This remains the general step-level trainer.

2. `train_residual_sac_pld_stage1_strict.yaml`
   This is the new PLD-facing step-level baseline config.
   It keeps the step-level code path and aligns the setup more closely with Algorithm 1:
   - no probing,
   - offline base-success bootstrap enabled,
   - symmetric online/offline replay enabled,
   - Cal-QL critic pretrain enabled,
   - warm-up base-only collection enabled,
   - `xi_scheduler` enabled.

3. `chunk_step`
   This is now implemented as a PLD-consistent extension, not as a strict reproduction of Stage 1.
   The current replay path stores step transitions and assembles chunk windows at sample time.
   This matches the RLT interpretation of chunk subsampling better than the earlier direct-chunk replay path.

This matters because the old chunk-step path mixed two semantics:
- collection happened at chunk-decision granularity,
- replay still pretended that the primitive unit was env-step.

After the latest refactor, online replay and offline bootstrap both write the same
step stream, and chunk replay windows are built from that shared representation.

## 2. What Changed

### 2.1 Observation encoding for chunk-step

Updated file:
- `examples/libero/policy/observation.py`

Changes:
1. The encoded `state` now includes:
   - normalized proprio state,
   - normalized current `base_action`,
   - normalized `base_action_chunk` when `chunk_step.enabled=true`,
   - scalar `xi`.
2. Raw `base_action`, raw `base_action_chunk`, and raw `xi` are still preserved in the observation dict.

Why:
The chunk actor outputs a full residual chunk, so it must see the full base chunk that defines how those residuals map into final executed actions.

### 2.2 Chunk replay is now decision-level, not sample-time step assembly

Updated file:
- `examples/libero/utils/step_chunk_replay.py`

Changes:
1. The replay buffer now stores one transition per chunk decision.
2. Each transition directly stores:
   - `observations`,
   - flattened final chunk action,
   - `action_mask`,
   - `next_observations`,
   - discounted chunk reward sum,
   - bootstrap mask,
   - `dones`,
   - `chunk_steps`.
3. The old sample-time reconstruction logic is gone.
4. `sample_stride` is preserved by storing `decision_id` metadata and filtering eligible chunk transitions at sampling time.

Why:
This makes chunk replay semantically consistent with SMDP-style decision learning and lets:
- online replay,
- offline bootstrap replay,
- Cal-QL pretrain,
- symmetric online/offline replay
all use the same transition type.

### 2.3 Warmup and online chunk collection now insert one chunk transition

Updated file:
- `examples/libero/scripts/train_residual_sac.py`

Changes:
1. Separate warmup collection under `chunk_step.enabled=true` now inserts one transition per chunk decision instead of one per env-step.
2. Main chunk RL collection now also inserts one transition per chunk decision.
3. The stored chunk action is the executed final chunk, padded to horizon when necessary.
4. `action_mask` marks the executed prefix and zeroes the padded tail.
5. `next_observations` is now built explicitly at the next decision boundary.

Why:
The old implementation executed chunk decisions but still wrote replay at env-step granularity, which was not the right training object for a chunk actor / chunk critic.

### 2.4 Chunk offline bootstrap now aligns with the same replay semantics

Updated file:
- `examples/libero/data/offline_bootstrap.py`

Changes:
1. Step mode still writes step transitions.
2. Chunk mode now writes direct chunk transitions.
3. Offline bootstrap in chunk mode stores base-policy final chunks and their chunk-level returns.

Why:
If online replay is chunk-level but offline bootstrap remains step-level, then Cal-QL pretrain and symmetric replay would still be mismatched.

### 2.5 Chunk training updates now trigger on decision steps

Updated file:
- `examples/libero/scripts/train_residual_sac.py`

Changes:
1. In chunk mode, sync updates now trigger from `global_policy_step` (decision clock), not from env-step crossings inside the chunk.
2. `training_starts` is therefore effectively interpreted in units of chunk decisions for chunk replay.

Why:
Once the primitive replay unit becomes the chunk decision, updating on env-step crossings would over-count collection progress relative to replay semantics.

### 2.6 New PLD-facing step baseline config

Added file:
- `examples/libero/conf/train_residual_sac_pld_stage1_strict.yaml`

Key settings:
- `chunk_step.enabled=false`
- `residual.xi=0.5`
- `replay.batch_size=256`
- `offline.enabled=true`
- `offline.dataset_paths=[]`
- `offline.bootstrap_base.enabled=true`
- `offline.bootstrap_base.success_episodes=50`
- `training.warmup_base_episodes=100`
- `training.max_online_env_steps=250000`
- `training.enable_base_probing=false`
- `training.calql_pretrain.enabled=true`
- `training.xi_scheduler.enabled=true`

Why:
This gives a clean “PLD Stage 1 baseline / strict mode” config without forcing a large code-path rewrite of the step-level trainer.

## 3. Current Behavior Logic

## 3.1 Step-level mode

This is the Stage-1 baseline path.

Flow:
1. OpenPI produces `base_chunk`.
2. The trainer uses `base_chunk[0]` for the current env step.
3. The residual actor outputs one residual step action.
4. Final executed action is composed as:
   - `a_final = a_base + xi * bounded_residual * limits`
5. Replay stores one step transition.
6. Critic learns on the final executed action.

## 3.2 Chunk-step mode

This is now a chunk-decision RL formulation.

Flow:
1. OpenPI produces `base_chunk`.
2. Observation encoding includes the full `base_action_chunk` and current `xi`.
3. The residual actor outputs one residual chunk.
4. The trainer composes:
   - `final_chunk = compose(base_chunk, residual_chunk, xi)`
5. Environment executes `env.step_chunk(final_chunk_prefix)`.
6. Replay stores one chunk transition for that decision.

Stored chunk transition fields:
- `observations`: decision-start observation
- `actions`: flattened final chunk, padded to `chunk_horizon`
- `action_mask`: `1` on executed prefix, `0` on padded tail
- `next_observations`: next decision observation, or zeros if terminal
- `rewards`: `sum_i gamma^i r_i`
- `masks`: `gamma^(k-1)` if non-terminal, else `0`
- `dones`
- `chunk_steps`: executed prefix length `k`

## 3.3 Online / offline replay semantics

Step mode:
- online replay and offline replay are both step transitions.

Chunk mode:
- online replay and offline bootstrap replay are both chunk transitions.
- external offline dataset paths are currently ignored in chunk mode.
- the chunk offline buffer is intended to stay PLD-style: successful base-policy rollouts only.

## 4. Why `critic(final)` + `actor(residual)` Is The Right Pairing

This is now the intended design.

Actor:
- input: observation plus base-policy context
- output: residual action or residual chunk

Critic:
- input: observation plus final executed action or final executed chunk
- output: scalar Q-value

Why this is better than the older `critic(residual)` implementation:

1. It matches the policy that actually interacts with the environment.
   The environment never sees a raw residual action by itself. It sees the combined action.

2. It matches Algorithm 1 more closely.
   PLD defines a combined policy and performs environment steps with the combined action.
   The value function is therefore semantically attached to that combined policy.

3. It keeps the actor parameterization efficient.
   The actor only needs to search in the residual space around the base policy, which is the whole point of PLD-style residual RL.

4. It handles `xi` and action limits correctly.
   If the critic only sees residual action, then changes in `xi`, control indices, and action limits change the actual executed action without changing the critic input semantics enough. Feeding the final action removes that ambiguity.

So the current split is:
- actor parameterization in residual space,
- critic semantics in executed-action space.

That is the cleaner design.

## 5. What Still Does Not Fully Align With PLD Stage 1

### 5.1 Chunk-step is still an extension, not strict PLD Stage 1

Algorithm 1 is step-wise:
- sample one combined action,
- call `env.step(a_bar)`.

Chunk-step instead learns over chunk decisions and chunk transitions.
So chunk-step should be described as a PLD-consistent extension, not a strict reproduction.

### 5.2 Exact network backbone still differs from the paper

The paper appendix reports a ResNetV1-10 encoder.
This codebase currently uses the available ResNet-18 path in the existing infrastructure.
So the new strict YAML aligns hyperparameters and training structure better, but not the exact encoder variant.

### 5.3 `xi_scheduler` shape is an implementation choice

The paper states that `xi` is scheduler-controlled, but it does not fully pin down the exact scheduler parameters used in this repo.
The new strict YAML enables the scheduler, but the linear schedule shape remains an implementation choice.

### 5.4 Chunk legacy knobs are mostly gone, but `sample_stride` is preserved

In chunk direct-replay mode:
- `chunk_step.sample_stride` is still active, now implemented as decision-id based sampling on direct chunk transitions.
- `chunk_step.require_full_horizon`
- `chunk_step.pad_action_to_horizon`

The last two were meaningful in the old sample-time assembly implementation, but not in the new direct chunk replay path.

### 5.5 Remaining risk: entropy still covers padded tail dimensions

Even after the replay refactor, the actor distribution log-prob is still computed over the full residual chunk dimension.
For truncated chunk transitions:
- critic inputs are masked,
- but actor entropy still includes padded tail dimensions.

This is the main remaining known issue in chunk mode.
It is smaller than the old replay-semantics bug, but it is still real.

## 6. Final Practical Takeaway

If you want the closest thing to PLD Stage 1 in this repo today, use:
- `examples/libero/conf/train_residual_sac_pld_stage1_strict.yaml`

If you want the chunk extension, the chunk path is now much more principled than before:
- actor sees full base chunk,
- replay is decision-level,
- offline bootstrap matches online replay semantics,
- critic learns on executed final chunk actions.

That means the current chunk implementation is no longer “step replay with chunk-flavored control”; it is now actually a chunk-transition RL formulation.
