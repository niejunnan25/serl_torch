# 下一步调试计划

基于 `2026-03-03 16:21:53` 训练结果分析（详见 `training_analysis_20260303.md`），  
本文档列出按优先级排列的修改项、具体操作步骤和验证方法。

---

## 修改清单总览

| 步骤 | 修改项 | 优先级 | 需改代码？ | 需改配置？ |
|------|--------|--------|-----------|-----------|
| 1 | 确认 ResNet 编码器修复生效 | 致命 | 已修复 | 已修复 |
| 2 | 启用 observation normalization | 高 | 否 | 是 |
| 3 | 去掉 warmup，用 xi_scheduler 替代 | 高 | 否 | 是 |
| 4 | 开启 backup_entropy | 低 | 否 | 是 |
| 5 | 运行训练并观察前 50 个 episode | — | 否 | 否 |

---

## 步骤 1：确认 ResNet 编码器修复已生效

上一次训练使用的是旧版 `encoder_type: resnet-pretrained`，ResNet 权重未加载。  
当前 YAML 已改为：

```yaml
sac:
  encoder_type: resnet
  resnet:
    model_name: /vla/users/niejunnan/codebase/serl_torch/pretrained_models/microsoft--resnet-18
    pretrained: true
    freeze_backbone: true
    pooling_method: spatial_learned_embeddings
    num_spatial_blocks: 8
    bottleneck_dim: 256
```

### 验证方法

训练启动后，在日志中应看到：

```
[ResNetEncoder] Loaded pretrained weights: /vla/users/niejunnan/codebase/serl_torch/pretrained_models/microsoft--resnet-18
```

**如果看到 `Random init from architecture`，说明有问题。**

同时确认本地模型文件存在：

```bash
ls /vla/users/niejunnan/codebase/serl_torch/pretrained_models/microsoft--resnet-18/model.safetensors
```

---

## 步骤 2：启用 observation normalization

### 修改 `conf/train_residual_sac.yaml`

```yaml
# 修改前
normalization:
  enabled: false
  stats_dir: null

# 修改后
normalization:
  enabled: true
  stats_dir: null    # null 时自动查找 data/stats/{task_name}.json
```

### 验证方法

统计文件已存在：

```bash
ls data/stats/place_a2b_left.json
```

训练启动后日志中应看到：

```
State/action normalizer loaded for task=place_a2b_left
```

**如果看到 `Normalization disabled`，说明没有生效。**

### 为什么重要

`place_a2b_left` 任务中，14 维 state 各维度的 std 差异达 **67.7 倍**（0.015 vs 1.042）。不归一化时，小尺度维度的信息会被网络忽略。

---

## 步骤 3：去掉 warmup，用 xi_scheduler 替代

### 问题回顾

- 旧配置：`warmup_base_episodes: 100` + `xi_scheduler.anneal_steps: 2000`
- 实际效果：warmup 100 个 episode ≈ 16,000 env steps，全零残差数据写入 replay；xi 在 step 2000 就退火完成，warmup 结束后残差直接满幅注入

### 修改 `conf/train_residual_sac.yaml`

```yaml
# 修改前
training:
  warmup_base_episodes: 100
  warmup_base_steps: 0
  # ...
  xi_scheduler:
    enabled: true
    type: linear
    min_xi: 0.1
    warmup_steps: 0
    anneal_steps: 2000

# 修改后
training:
  warmup_base_episodes: 0      # 不要硬性 warmup
  warmup_base_steps: 0
  # ...
  xi_scheduler:
    enabled: true
    type: linear
    min_xi: 0.05               # 从很小的残差开始（比之前的 0.1 更保守）
    warmup_steps: 0
    anneal_steps: 20000        # 缓慢退火，约 100 个 episode 才到满幅
```

### 原理

用 xi 的渐进退火替代硬性 warmup：

| policy_step | xi 值 | 残差幅度 | 效果 |
|-------------|-------|---------|------|
| 0 | 0.05 | 极小 | 接近纯基策略，但已有微量残差开始学习 |
| 5,000 | 0.16 | 小 | 残差缓慢增大，不破坏基策略 |
| 10,000 | 0.28 | 中等 | Critic 已学到一定经验，可接受更大残差 |
| 20,000 | 0.50 | 满幅 | 正式全力修正 |

### 优势

- 从第 1 步就有非零残差，Critic 从一开始就在有效数据上学习
- 不会有 16,000 条全零残差数据污染 replay buffer
- 残差幅度渐进增大，基策略不会被突然打断

---

## 步骤 4：开启 backup_entropy（可选）

### 修改 `conf/train_residual_sac.yaml`

```yaml
# 修改前
sac:
  backup_entropy: false

# 修改后
sac:
  backup_entropy: true
```

### 为什么

标准 SAC 中，Critic 的 TD target 应为：

```
target_q = reward + gamma * (min_next_q - alpha * log_pi(next_a))
```

`backup_entropy: false` 时缺少熵项，Critic 和 Actor 对"好动作"的度量不一致。开启后二者对齐，训练更稳定。

---

## 步骤 5：运行训练并观察

### 启动命令

与之前相同的启动方式，YAML 配置已修改。

### 关键日志检查点

训练启动后，依次确认以下日志输出：

1. **ResNet 加载成功**：
   ```
   [ResNetEncoder] Loaded pretrained weights: ...microsoft--resnet-18
   ```

2. **归一化器加载成功**：
   ```
   State/action normalizer loaded for task=place_a2b_left
   ```

3. **无 warmup**（不应看到连续的全零残差 episode）：
   - step_logs 中第一个 episode 就应有 `"a_res": [非零值, ...]`
   - `"xi"` 应为 0.05（起始值）

### 成功标准

| 指标 | 旧训练（参考） | 期望值 |
|------|--------------|--------|
| episode 0–20 成功率 | 70%（纯基策略） | ≥ 60%（残差很小，接近基策略） |
| episode 50–100 成功率 | 开始下降 | **≥ 70%（不应下降）** |
| episode 100–200 成功率 | 65% → 60% 下降 | **≥ 70%（稳定或上升）** |
| episode 200+ 趋势 | 持续下降到 52% | **稳定或缓慢上升** |

**核心判断**：如果 episode 100 之后成功率**不再下降**，说明修复有效。如果成功率**超过基策略峰值（75%）并持续上升**，说明残差策略真正学到了有用的修正。

### 如果仍然不收敛怎么办？

按以下顺序排查：

1. **检查 step_logs 中 `a_res` 的幅度**：
   - 如果每维 |a_res| 仍然 ≈ 0.008（像旧训练一样），说明 Actor 没有学到有效信号
   - 可能原因：学习率太小、Critic 估计不准

2. **检查 Critic loss**：
   - 如果 Critic loss 居高不下或震荡，考虑降低学习率或增大 `utd_ratio`

3. **考虑加入离线数据预训练**：
   ```yaml
   offline:
     enabled: true
     dataset_paths:
       - /path/to/lerobot_datasets/single_task_clean_place_a2b_left
   training:
     calql_pretrain:
       enabled: true
       steps: 1000
   ```

4. **考虑解冻 ResNet 后几层**：
   ```yaml
   sac:
     resnet:
       freeze_backbone: false   # 让视觉编码器也可以微调
   ```
   注意：这会大幅增加可训练参数量和训练时间

---

## 配置变更汇总 (diff)

以下是需要修改的 `conf/train_residual_sac.yaml` 行：

```
sac:
  backup_entropy: true          # 原值 false

training:
  warmup_base_episodes: 0       # 原值 100
  xi_scheduler:
    min_xi: 0.05                # 原值 0.1
    anneal_steps: 20000         # 原值 2000

normalization:
  enabled: true                 # 原值 false
```

共 **4 行配置修改**，不需要改任何代码。
