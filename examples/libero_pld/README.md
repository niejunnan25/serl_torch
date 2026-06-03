# LIBERO PLD Stage-1

This example is a narrow implementation of the PLD Stage-1 residual RL recipe
for LIBERO. It intentionally excludes the later experimental branches in
`examples/libero`, including KeyRL gating, Robo-Dopamine rewards, processor
pipelines, and staged reward models.

The training loop uses the existing remote LIBERO environment and OpenPI policy
servers. Stage 1 keeps the base policy frozen and trains a residual Gaussian
policy with

```text
final_action = base_action + alpha * residual_action
```

For paper-faithful PLD reproduction, keep `residual.chunk_horizon: 1`. Larger
chunk horizons are an engineering variant, not strict per-step residual PLD.

## What Is Included

- Base-policy success replay collection with Monte Carlo return-to-go fields.
- Offline/online mixed replay for RLPD-style updates.
- Cal-QL-style critic pretraining using a CQL conservative term and optional
  MC-return lower-bound calibration when prepared replay contains
  `mc_returns` and `mc_returns_valid`.
- Actor/learner training via the existing SERL transport and checkpoint stack.

Generated replay, Hydra outputs, checkpoints, videos, and logs are not source
artifacts. Store them under ignored paths such as `examples/libero_pld/data/`,
`examples/libero_pld/outputs/`, or an external output root.

## Prepare Base-Success Replay

Start the LIBERO env server and the OpenPI policy server with ports matching the
config, then collect successful base-policy episodes:

```bash
cd /vla/users/niejunnan/codebase/serl_torch

python examples/libero_pld/scripts/collect_base_success_replay.py \
  --config-path examples/libero_pld/configs \
  --config-name pld_libero_spatial_task4 \
  pld.base_success.target_successes=50
```

To use another base checkpoint, override `policy_dir`, `policy.id`, the policy
server port, and `offline.prepared_path` so the prepared replay is separated by
base policy. For example, pi0_60000 should use its own output path and policy
server script.

## Train With Existing Services

If env and policy services are already running:

```bash
cd /vla/users/niejunnan/codebase/serl_torch

bash examples/libero_pld/tools/launch_pld.sh \
  --config examples/libero_pld/configs/pld_libero_spatial_task4.yaml \
  --session pld_spatial4 \
  --gpu 0
```

The launcher starts a tmux session with learner and actor windows and uses the
ports/checkpoint paths from the YAML config.

## Collect Then Train

For unattended runs, use the collect-then-train launcher:

```bash
cd /vla/users/niejunnan/codebase/serl_torch

bash examples/libero_pld/tools/launch_pld_after_collect.sh \
  --config examples/libero_pld/configs/pld_libero_spatial_task4.yaml \
  --session pld_spatial4 \
  --gpu 0 \
  --learner-gpu 1 \
  --target-successes 50 \
  --policy-script examples/libero/tools/serve_openpi_10000_policy.sh
```

For pi0_60000, pass the matching config overrides and policy launcher, for
example:

```bash
bash examples/libero_pld/tools/launch_pld_after_collect.sh \
  --config examples/libero_pld/configs/pld_libero_spatial_task4.yaml \
  --session pld_spatial4_pi0_60000 \
  --gpu 6 \
  --learner-gpu 7 \
  --target-successes 50 \
  --policy-script examples/libero/tools/serve_openpi_60000_policy.sh
```

When changing policy checkpoints, also override `policy_dir`, `policy.id`,
ports, and `offline.prepared_path` so replay and services do not collide.
