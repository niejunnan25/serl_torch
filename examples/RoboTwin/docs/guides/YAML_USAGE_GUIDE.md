# RoboTwin YAML 配置说明与示例（当前版本）

本文档说明 `examples/RoboTwin/conf` 下训练与评估 YAML 的关键字段、影响范围和使用方式。

## 1. 推荐配置入口

- 训练主配置：`conf/train_residual_sac.yaml`
- 评估主配置：`conf/eval_residual_fast.yaml`
- 快速验证时直接使用主配置并通过命令行 override 缩小预算。

## 2. 输出目录与权重保存

训练时：
- run 根目录：`hydra.run.dir`
- 权重目录：`<hydra.run.dir>/<training.checkpoint_dir>`
- 保存间隔：`training.checkpoint_period`（按 `global_policy_step`）
- 最多保留：`training.keep_checkpoints`

默认主配置下：
- `hydra.run.dir=outputs/train_residual_sac/<date>/<time>`
- `training.checkpoint_dir=checkpoints`
- 所以默认权重路径形如：
  - `outputs/train_residual_sac/2026-03-03/12-34-56/checkpoints/checkpoint_1000.pt`

评估时：
- 通过 `eval.checkpoint_path` 指向训练输出的 checkpoint 文件。

## 3. 字段说明（训练）

### 3.1 任务与 OpenPI

- `task.name/task.task_config/task.prompt`
  - 决定 RoboTwin 任务实例和文本指令。
- `task.seed_base`
  - 控制 episode seed 起点。
- `openpi.host/openpi.port`
  - OpenPI 服务地址。

### 3.2 残差动作与观测

- `residual.image_keys`
  - 残差策略读取哪些相机（1/2/3 路）。
- `residual.action_dim` 或 `residual.action_indices`
  - 残差输出维度（`action_indices` 优先）。
- `residual.xi`
  - 残差幅度上限（论文关键超参）。
- `residual.chunk_horizon`
  - VLA chunk 执行窗口大小。

### 3.3 SAC/DrQ

- `sac.encoder_type`
  - `small/resnet/resnet-pretrained`。
- `sac.policy_hidden_dims` / `sac.critic_hidden_dims`
  - actor/critic MLP 结构。
- `sac.learning_rate` + `sac.optimizer.*`
  - 学习率、Adam/AdamW、weight decay、grad clip。
- `sac.utd_ratio`
  - critic:actor 更新比（`2` 即 `2:1`）。
- `sac.otf_num_samples`
  - OTF TD backup 采样数。

### 3.4 Replay 与离线初始化

- `replay.capacity` / `replay.batch_size`
  - online buffer 容量与 batch。
- `offline.enabled`
  - 是否启用 offline buffer。
- `offline.dataset_paths`
  - 外部离线数据路径（可空）。
- `offline.bootstrap_base.*`
  - 自动收集 base 成功轨迹。
- `offline.symmetric_replay` / `offline.ratio`
  - offline/online 混采策略。

### 3.5 训练流程控制

- `training.max_online_env_steps`
  - 在线交互总预算（硬停止条件）。
- `training.warmup_base_episodes`
  - warmup 期间 base-only。
- `training.enable_base_probing` + `training.probing_alpha`
  - base probing：`T_base ~ U(0, alpha*T)`。
- `training.calql_pretrain.*`
  - critic Cal-QL/CQL-style 预训练。

### 3.6 调度与异步

- `training.xi_scheduler.*`
  - 对 `residual.xi` 做线性/余弦调度（论文语义）。
- `training.residual_scale_scheduler.*`
  - 旧兼容项，仅调 `residual_scale`。
- `training.async.*`
  - 异步采样-学习：
  - `enabled`: 是否开启异步 learner
  - `update_frequency`: learner 更新多少次后同步一次参数给 actor
  - `idle_sleep_sec`: learner 空转等待间隔

## 4. 字段说明（评估）

- `eval.episodes`
  - 评估 episode 数。
- `eval.checkpoint_path`
  - 要加载的残差 checkpoint。
- `eval.deterministic`
  - 残差策略是否确定性推理。
- `eval.residual_scale`
  - 评估时残差全局缩放。
- `eval.enable_base_probing` + `eval.probing_alpha`
  - 评估也可启用 Stage-2 probing。
- `eval.collect_dataset_path`
  - 导出轨迹数据路径（可用于 PLD/分析）。

## 5. 使用示例

在 `examples/RoboTwin` 目录运行。

### 5.1 小预算训练 smoke test

```bash
python scripts/train_residual_sac.py --config-name train_residual_sac \
  training.max_online_env_steps=20000 \
  training.checkpoint_period=200 \
  hydra.run.dir=outputs/train_smoke
```

常用覆写：

```bash
python scripts/train_residual_sac.py --config-name train_residual_sac \
  training.max_online_env_steps=20000 \
  training.checkpoint_period=200 \
  hydra.run.dir=outputs/train_smoke_local
```

### 5.2 小预算评估 smoke test

```bash
python scripts/eval_residual_fast.py --config-name eval_residual_fast \
  eval.checkpoint_path=/abs/path/to/checkpoint_5000.pt
```

### 5.3 主配置（论文预算）

```bash
python scripts/train_residual_sac.py --config-name train_residual_sac
python scripts/eval_residual_fast.py --config-name eval_residual_fast \
  eval.checkpoint_path=/abs/path/to/checkpoint_xxxxx.pt
```

## 6. 从 demo 切到论文设定的最小覆盖

- `sac.encoder_type=resnet-pretrained`
- `offline.bootstrap_base.success_episodes=50`
- `training.warmup_base_episodes=100`
- `training.max_online_env_steps=250000`
- `eval.episodes=50`
- 保持 `training.xi_scheduler.enabled=true`
- 保持 `training.async.enabled=true`
