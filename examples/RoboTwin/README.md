# RoboTwin Residual RL (OpenPI + DrQ-SAC)

This folder implements paper-aligned Stage-1/2 residual RL for RoboTwin:
- Stage-1: freeze base VLA/OpenPI and train residual policy.
- Stage-2: base probing + residual takeover for hybrid rollouts.
- Stage-3 distillation is out of scope.

## Main Files

- `scripts/train_residual_sac.py`: Stage-1 residual RL training entry.
- `scripts/eval_residual_fast.py`: Stage-2/eval rollout and data collection entry.
- `core/common.py`: shared environment/action/IO utilities.
- `conf/train_residual_sac.yaml`: main training config.
- `conf/eval_residual_fast.yaml`: main eval config.
- `conf/train_demo.yaml` / `conf/eval_demo.yaml`: minimal runnable demos.
- `scripts/run_stage12_repro.sh`: 3-seed paper-style run helper.
- `scripts/aggregate_eval_ci.py`: mean + 95% CI aggregation.

## Implemented Paper-Critical Details

- Residual policy is step-wise (`a_delta_t`), not residual chunk output.
- Residual observation is `images_t + state_t + base_action_t`.
- Action composition is `a = a_b + a_delta`.
- Residual amplitude is constrained by `xi`, and `xi` is scheduler-driven (`training.xi_scheduler`).
- Warmup uses episode semantics (`training.warmup_base_episodes`, default 100).
- Base probing supports paper form `T_base ~ U(0, alpha * T)` (`probing_alpha`, default 0.6).
- Probing prefix is not inserted into replay.
- Offline/online replay supports symmetric `1:1` mixing.
- Critic pretrain uses Cal-QL/CQL-style conservative critic updates (`training.calql_pretrain`).
- OTF TD backup supports multi-sample next-action bootstrap (`sac.otf_num_samples`).
- Default critic:actor update ratio is `2:1` (`sac.utd_ratio=2`).
- Target entropy default is `-act_dim/2`.
- Async collection-learning loop is supported (`training.async.*`) with periodic actor parameter sync.

## Paper-Aligned Defaults

Current defaults in `conf/train_residual_sac.yaml`:
- `residual.xi=0.5`
- `training.xi_scheduler.enabled=true`
- `training.async.enabled=true`
- `training.async.update_frequency=30`
- `sac.encoder_type=resnet-pretrained`
- `sac.policy_hidden_dims=[256,256,256]`
- `sac.critic_hidden_dims=[256,256,256]`
- `sac.temperature_init=1.0`
- `sac.optimizer.type=adamw`, `grad_clip_norm=1.0`
- `replay.capacity=250000`, `offline.capacity=250000`
- `offline.bootstrap_base.success_episodes=50`
- `training.max_online_env_steps=250000`

## Output And Checkpoints

- Train run root: `hydra.run.dir` (default: `outputs/train_residual_sac/<date>/<time>`).
- Checkpoint directory: `<hydra.run.dir>/<training.checkpoint_dir>` (default subdir `checkpoints`).
- Save interval: every `training.checkpoint_period` policy steps.
- Keep latest N checkpoints: `training.keep_checkpoints`.
- Eval summary/logs are written under eval `hydra.run.dir`.

## Quick Start

From `examples/RoboTwin`:

```bash
python scripts/train_residual_sac.py
```

Evaluate:

```bash
python scripts/eval_residual_fast.py \
  eval.checkpoint_path=/abs/path/to/checkpoint_xxxxx.pt
```

Collect Stage-2 hybrid rollouts:

```bash
python scripts/eval_residual_fast.py \
  eval.checkpoint_path=/abs/path/to/checkpoint_xxxxx.pt \
  eval.enable_base_probing=true \
  eval.probing_alpha=0.6 \
  eval.collect_dataset_path=stage2_hybrid_rollouts.pkl
```

Run 3-seed paper-style protocol:

```bash
bash scripts/run_stage12_repro.sh
```

## Docs

- `docs/README.md`: docs index.
- `docs/alignment/STAGE12_ALIGNMENT.md`: Stage-1/2 alignment overview.
- `docs/alignment/PAPER_IMPLEMENTATION_COMPARISON.md`: paper-vs-code comparison.
- `docs/guides/DATAFLOW_DIMENSIONS.md`: dataflow + dimensions.
- `docs/guides/YAML_USAGE_GUIDE.md`: YAML field guide and examples.
- `docs/reviews/IMPLEMENTATION_REVIEW.md`: static review and run risks.

## Notes

- Default RoboTwin root resolves to `../../RoboTwin`; override with `robo_root=/abs/path/to/RoboTwin`.
- Train step logs include `a_base`, `a_res`, `a_final`, `is_probing`, `residual_scale`, `xi`, reward/success, and latency fields.
- Eval step logs include the same core fields except `xi` (constant in eval summary).
- For custom controlled dimensions, prefer `residual.action_indices`.
- For old synchronous loop behavior: set `training.async.enabled=false`.
