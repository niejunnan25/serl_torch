# RoboTwin Stage1/2 对齐设计（VLA + Residual RL）

本文档说明当前 `examples/RoboTwin` 与论文
**Self-Improving Vision-Language-Action Models with Data Generation via Residual RL**
在 Stage-1/2 的实现对齐（不含 Stage-3 蒸馏）。

## 1. 总体目标

- Stage-1：冻结 base VLA（OpenPI），训练 task-specific residual policy。
- Stage-2：base probing + residual takeover，生成 hybrid rollout。
- 配置入口：`conf/train_residual_sac.yaml`、`conf/eval_residual_fast.yaml`。

## 2. 当前对齐要点

### 2.1 逐步 residual（非 chunk residual）

- base 仍按 chunk 推理：`base_chunk = OpenPI(obs, prompt)`。
- residual 按步推理：输入 `image_t + state_t + base_action_t`，输出一步 residual action。
- 执行：`a_t = a_b_t + a_delta_t`。

### 2.2 ξ 约束与调度

- `residual.xi` 为残差幅度上限。
- 执行前先把残差约束到 `[-xi_t, xi_t]`，再映射到机器人动作增量。
- `training.xi_scheduler` 对 `xi_t` 按训练步调度（linear/cosine）。

### 2.3 Cal-QL critic 预训练

- `training.calql_pretrain.*` 启用 offline critic 预训练。
- 用 base 成功轨迹初始化 critic，再进入在线 residual RL。

### 2.4 OTF TD backup

- `sac.otf_num_samples` 控制 TD backup 的 next action 多采样。
- `=1` 为默认论文设定；`>1` 可提升前期样本效率（计算更重）。

### 2.5 Warmup 口径

- `training.warmup_base_episodes`（默认 100）是论文主口径。
- `warmup_base_steps` 保留兼容，默认为 0。

### 2.6 Probing 口径

- `training.probing_alpha` / `eval.probing_alpha` 支持 `U(0, alpha*T)`。
- probing prefix 不写入 replay，仅用于初始化状态分布。

### 2.7 Replay 与更新

- `offline.symmetric_replay=true` 时 online/offline 1:1 混采。
- 默认 `sac.utd_ratio=2`，对应 critic:actor=2:1。
- Polyak target update 率 `0.005`。

### 2.8 Async 采样-学习解耦

- 默认 `training.async.enabled=true`。
- actor 与 learner 并行，按 `training.async.update_frequency` 同步参数。
- 如需要旧同步行为：`training.async.enabled=false`。

### 2.9 论文默认超参（配置层）

- `sac.encoder_type=resnet-pretrained`
- `policy/critic hidden_dims=[256,256,256]`
- `sac.temperature_init=1.0`
- `optimizer=adamw` + `grad_clip_norm=1.0`
- `replay/offline capacity=250000`
- `offline.bootstrap_base.success_episodes=50`
- `training.max_online_env_steps=250000`

## 3. 推荐用法

### 3.1 Stage-1 训练

```bash
cd examples/RoboTwin
python scripts/train_residual_sac.py
```

### 3.2 Stage-2 评估/采集

```bash
cd examples/RoboTwin
python scripts/eval_residual_fast.py \
  eval.checkpoint_path=/abs/path/to/checkpoint_xxxxx.pt
```

### 3.3 3-seed 论文协议

```bash
cd examples/RoboTwin
bash scripts/run_stage12_repro.sh
```

## 4. 说明

- Stage-3（把 hybrid 数据蒸馏回 VLA）未纳入本目录实现。
- 更细的实现改动见：`PAPER_STAGE12_ALIGNMENT_IMPLEMENTATION.md`。
- 论文对照见：`PAPER_IMPLEMENTATION_COMPARISON.md`。
