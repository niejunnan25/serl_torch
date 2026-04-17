# 训练结果分析：2026-03-03 16:21:53 运行

**任务**: `place_a2b_left`（残差 SAC + OpenPI 基策略）  
**日志路径**: `outputs/train_residual_sac/2026-03-03/16-21-53/`  
**训练时长**: 40.5 小时，434 个 episode，75,916 env steps，49,318 policy steps

---

## 一、核心现象：成功率持续下降

| 区间 | Episode | 20-ep 窗口成功率 | 累计成功率 |
|------|---------|------------------|------------|
| Warmup 早期 | 0–19 | **70%** | 70.0% |
| Warmup 峰值 | 20–39 | **80%** | 75.0% |
| Warmup 中期 | 40–59 | 70% | 70.0% |
| Warmup 后期 | 80–99 | 55% | 68.0% |
| 残差介入 | 100–119 | 55% | 66.4% |
| 稳定下降 | 200–219 | 45% | 60.7% |
| 严重退化 | 260–279 | **35%** | 57.0% |
| 后期回弹 | 300–319 | 55% | 55.9% |
| 末期低谷 | 380–399 | **30%** | 53.2% |
| 训练结束 | 420–433 | 43% | **52.1%** |

**关键结论**：
- **Warmup 期间（纯基策略）峰值 80%**，说明 OpenPI 基策略本身在该任务上就有较好表现
- **残差策略介入后（ep ≥ 100），成功率从 68% 持续下降到 52%**
- 残差策略不但没有改善基策略，反而造成了性能退化
- 最后 50 个 episode 成功率仅 **36%**

---

## 二、根因分析

### 问题 1（致命）：视觉编码器使用的是随机权重

本次训练使用的是 **旧版** `encoder_type: resnet-pretrained`，对应的 `ResNetEncoder` 存在 P0 bug：

- 实际构建的是 `torchvision.resnet18(weights=None)`，即**随机初始化**
- 从 JAX/Flax 格式 `.pkl` 加载权重时 key 完全不匹配，`strict=False` 静默跳过
- `freeze_backbone=True` 冻结了这些**随机权重**

这意味着三路相机图像（cam_high、cam_left_wrist、cam_right_wrist）经过 ResNet 后输出的是**随机特征**，视觉信息完全没有被利用。Policy MLP 的 768 维视觉输入全是噪声。

> **影响**：残差策略几乎无法从视觉中学到有用信息，只能依赖 64 维的本体感知（proprio + base_action），信息量严重不足。

### 问题 2（严重）：xi_scheduler 退火在 warmup 期间被浪费

配置：
```yaml
xi_scheduler:
  enabled: true
  min_xi: 0.1
  warmup_steps: 0      # 从 step 0 开始退火
  anneal_steps: 2000    # 2000 步完成退火
warmup_base_episodes: 100  # 前 100 个 episode 残差为 0
```

实际行为：
- xi 从 0.1 线性退火到 0.5，在 policy_step ≈ 2000 时完成（约 episode 11）
- 但 warmup 持续到 episode 100（global_env_step ≈ 16,447）
- **warmup 结束时 xi 已经是 0.5（最大值）**，残差一步到位到满幅，没有渐进过程

残差动作在 ep 100 第一次出现时 L1 范数就已经 ≈ 0.12，导致策略行为突变。

### 问题 3（中等）：warmup 零动作数据占满 replay buffer

- 前 100 个 episode 的 ~16,447 条 transition，`a_res` 全为零向量
- 这些数据写入 replay buffer（容量 250,000），占比约 22%（16,447/75,916）
- Critic 在大量"残差=0"的数据上训练，对非零残差的 Q 值估计不准
- warmup 结束后策略输出非零残差时，Q 值估计与实际 return 脱节

### 问题 4（中等）：残差动作幅度极小且无结构

后续训练中残差的统计特征：

| 指标 | 值 |
|------|-----|
| L1 均值 | 0.143 |
| L1 最大值 | 0.248 |
| 每维平均绝对值 | 0.008 |
| 夹爪维度（dim 6, 13）| 0.023（稍大） |

每维残差约 0.008 弧度 ≈ 0.5°，**对关节位置几乎没有实际影响**。这表明：
- 残差策略没有学到有意义的修正信号
- 可能原因：视觉特征为噪声 → Critic 估计不准 → Actor 梯度质量差 → 策略输出近似随机噪声
- 少量随机扰动反而干扰了基策略的执行

### 问题 5（轻微）：probing 机制稀释了训练样本

后 warmup 阶段步级统计：
- Probing（纯基策略，无残差）：20,562 步（34.5%）
- 非 probing（有残差）：39,037 步（65.5%）

约 1/3 的时间花在 probing 上（探测基策略是否能单独完成），这些步骤产生的 transition 中 `a_res=0`，进一步稀释了有效学习样本。

### 问题 6（配置）：其他待优化项

- **`backup_entropy: false`**：Critic TD target 不含熵项，与 Actor 目标不一致
- **`utd_ratio: 2`**：UTD 偏低，可能在 off-policy 效率上有提升空间
- **训练速度慢**：每 episode 平均 5.6 分钟，1,876 env steps/hour

---

## 三、数据证据

### 成功率在 warmup（基策略）vs 训练（残差策略）的对比

```
Episodes   0–99  (warmup, 纯基策略):  68% 成功率
Episodes 100–433 (残差策略介入):      47% 成功率
最后 50 episodes:                     36% 成功率
```

**残差策略使成功率下降了约 20 个百分点。**

### 残差动作首次出现

```
Episode 100, global_env_step = 16,490, global_policy_step = 10,321
残差 L1 ≈ 0.12, xi = 0.5 (已满幅)
```

### Expert Precheck 跳过率

```
总共尝试 515 个 seed
跑了 434 个 episode
跳过 81 个 (15.7% precheck 失败)
```

约 15.7% 的 seed 被跳过，说明基策略（OpenPI）本身在这些 seed 上就无法完成任务。

---

## 四、改进方向

### 必须修复（P0）

| # | 改进 | 预期效果 |
|---|------|----------|
| 1 | **使用已修复的 HuggingFace ResNet 编码器**（`encoder_type: resnet`，配合 `sac.resnet` 配置块） | 视觉特征从随机噪声变为有意义的 ImageNet 预训练特征，残差策略可以利用视觉信息 |
| 2 | **`load_state_dict` 使用 `strict=True`**（已修复） | 任何权重加载失败立即报错，杜绝静默失败 |

### 强烈建议（P1）

| # | 改进 | 预期效果 |
|---|------|----------|
| 3 | **修复 xi_scheduler 退火时机**：`xi_scheduler.warmup_steps` 设置为与 warmup 对齐，或 `effective_step = max(0, policy_step - warmup_end_step)` | warmup 结束后残差从小到大渐进放大，避免策略突变 |
| 4 | **减少或消除 warmup 零动作对 replay 的污染**：warmup 数据不入 replay，或缩短 warmup 到 10–20 episodes | Critic 从一开始就在有效数据上训练 |

### 可选优化（P2/P3）

| # | 改进 | 预期效果 |
|---|------|----------|
| 6 | 设置 `backup_entropy: true` | Critic target 与 Actor 目标一致，标准 SAC |
| 7 | 增大 `utd_ratio`（如 4 或 8） | 提高 off-policy 数据利用率 |
| 8 | 调整 probing 参数：降低 `probing_alpha` 或 `probing_max_steps` | 减少 probing 对训练样本的稀释 |
| 9 | 缩短 `warmup_base_episodes`（如 20–30） | 更早开始残差训练，总训练时间更短 |
| 10 | 考虑加入离线数据预训练（`offline.enabled: true`） | 用专家数据初始化 Critic，改善早期 Q 估计 |

---

## 五、建议的下一步实验配置

```yaml
sac:
  encoder_type: resnet
  resnet:
    model_name: /home/hello/codebase/serl_torch/pretrained_models/microsoft--resnet-18
    pretrained: true
    freeze_backbone: true
    pooling_method: spatial_learned_embeddings
    num_spatial_blocks: 8
    bottleneck_dim: 256
  backup_entropy: true

training:
  warmup_base_episodes: 20        # 大幅缩短 warmup
  xi_scheduler:
    enabled: true
    min_xi: 0.1
    warmup_steps: 4000            # 对齐 warmup 结束后再开始退火
    anneal_steps: 10000           # 更缓慢地放大残差幅度

  enabled: true
  stats_dir: /path/to/precomputed/stats
```

---

## 六、总结

本次训练失败的**根本原因**是 P0 bug：ResNet 视觉编码器权重未成功加载，视觉信息等价于随机噪声。残差策略无法从视觉中学习，反而以微小随机扰动干扰了原本表现不错的基策略（warmup 期间 80% 峰值成功率）。修复 P0 后，配合 xi_scheduler 时机修正和 warmup 策略优化，预期残差策略能够有效利用视觉信息改善基策略表现。
