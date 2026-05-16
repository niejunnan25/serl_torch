# Stage-1/2 论文对齐实现说明（当前代码）

本文档总结 RoboTwin Stage-1/2 论文对齐后的当前实现状态。

## 1. Critic 预训练：SAC warm-start -> Cal-QL/CQL-style

- 新增 conservative critic 预训练路径：
  - `SACAgent.update_critics_calql(...)`
  - `DrQAgent.update_critics_calql(...)`
- 训练入口增加 `training.calql_pretrain.*`，在 offline 成功轨迹上先预训练 critic。

## 2. OTF TD backup：支持多采样

- `sac.otf_num_samples` 已接入 critic TD target。
- `=1` 保持默认逻辑；`>1` 时对 next residual action 多采样并聚合目标 Q。

## 3. Warmup 口径：episodes 优先

- 新增 `training.warmup_base_episodes`（论文口径）。
- `training.warmup_base_steps` 保留兼容。

## 4. Base probing horizon：支持 U(0, alpha*T)

- 新增：
  - `training.probing_alpha`
  - `eval.probing_alpha`
- probing 步数优先按 `U(0, alpha*T)` 采样，min/max 保留 fallback。

## 5. ξ 幅度语义落地

- 新增 `residual.xi`。
- 在线动作组合前将 residual 限制到 `[-xi, xi]`。
- 离线 expert->residual 反解同步使用 `xi`：
  - `(a_expert - a_base) / (limits * xi * expert_reference_scale)`。

## 6. 论文语义进一步对齐：xi 调度

- 新增 `training.xi_scheduler.*`（linear/cosine）。
- 每个 policy step 计算 `xi_step`，并用于当前步动作组合。
- `training.residual_scale_scheduler.*` 保留为历史兼容项。

## 7. 优化器与梯度裁剪

- 训练/评估统一支持 `sac.optimizer`：
  - `type` (`adam` / `adamw`)
  - `weight_decay`
  - `grad_clip_norm`
  - `warmup_steps`, `cosine_decay_steps`

## 8. 论文默认超参落地（配置）

主配置已使用论文对齐默认值：
- `sac.utd_ratio=2`
- `sac.temperature_init=1.0`
- `replay.capacity=250000` / `offline.capacity=250000`
- `offline.enabled=true`
- `offline.bootstrap_base.enabled=true`
- `offline.bootstrap_base.success_episodes=50`
- `training.warmup_base_episodes=100`
- `training.max_online_env_steps=250000`
- `sac.encoder_type=resnet-pretrained`
- `policy/critic hidden_dims=[256,256,256]`

## 9. 在线预算终止条件

- 新增 `training.max_online_env_steps` 全局步数上限。
- 训练可在 phase 中途达到预算时提前收敛并写入 summary。

## 10. 异步采样-学习（SERL 风格）

- 新增 `training.async.*`：
  - `enabled`
  - `update_frequency`
  - `idle_sleep_sec`
- 启用后：
  - actor 线程负责采样与写 replay
  - learner 线程持续更新
  - 每隔 `update_frequency` 次更新同步参数给 actor

## 11. 实验协议脚本

- 新增：
  - `scripts/run_stage12_repro.sh`
  - `scripts/aggregate_eval_ci.py`
- 支持 3 seeds + mean/95% CI 聚合。

## 12. 文档同步

- 已同步更新：
  - `README.md`
  - `docs/alignment/STAGE12_ALIGNMENT.md`
  - `docs/guides/YAML_USAGE_GUIDE.md`
  - `docs/guides/DATAFLOW_DIMENSIONS.md`

## 13. 静态完整性检查（本地）

- `python -m py_compile`：
  - `examples/RoboTwin/scripts/train_residual_sac.py`
  - `examples/RoboTwin/scripts/eval_residual_fast.py`
  - `examples/RoboTwin/env_wrappers/task_env.py`
  - `examples/RoboTwin/policy/action.py`
  - `examples/RoboTwin/policy/observation.py`
  - `serl_launcher/serl_launcher/agents/continuous/sac.py`
  - `serl_launcher/serl_launcher/agents/continuous/drq.py`
  - `examples/RoboTwin/scripts/aggregate_eval_ci.py`
- `bash -n examples/RoboTwin/scripts/run_stage12_repro.sh`
