# PLD 复现实现 Review 与论文对比分析

> 参考论文：[Self-Improving Vision-Language-Action Models with Data Generation via Residual RL](https://arxiv.org/pdf/2511.00091)
>
> 实现位置：`serl_torch/examples/RoboTwin/`
>
> 运行环境：RoboTwin（双臂 ALOHA，14 维动作空间）
>
> 基策略后端：OpenPI（π₀ 风格 flow-matching VLA）

---

## 一、代码 Review：正确性与逻辑分析

### 1.1 整体架构评价

代码整体设计**逻辑正确、结构清晰**。主要模块分层合理：

| 模块 | 职责 | 评价 |
|------|------|------|
| `train_residual_sac.py` | Stage-1 残差策略在线训练 + Stage-2 probing | 核心循环逻辑正确 |
| `eval_residual_fast.py` | Stage-2 数据收集 + 评估 | 与训练对齐良好 |
| `envs/task_env.py` | RoboTwin 环境统一封装 | 接口干净，重试机制合理 |
| `policy/action.py` | 动作组合、indices 解析 | 公式实现正确 |
| `policy/observation.py` | 残差策略观测构建 | state 拼接逻辑正确 |
| `policy/openpi_client.py` | VLA 推理客户端 | 简洁可靠 |
| `data/normalizer.py` | 状态/动作归一化 | 可选模块，实现无误 |
| `utils/config_utils.py` | Hydra 配置解析、agent 构建 | 完备 |

### 1.2 数据流验证

**在线数据流**（已验证正确）：

```
obs_raw
  → OpenPI infer_chunk → base_chunk (H, 14)
  → build_residual_step_obs(obs_raw, base_action_t)
     → obs_input = {images: (1,H,W,3), state: (1, 28)}  // 28 = 14 joint + 14 base_action
  → DrQ-SAC sample_actions(obs_input) → residual_step_action (action_dim,)
  → compose_residual_action:
     clipped = clip(residual, -1, 1)
     bounded = clip(clipped * xi, -xi, xi)
     delta = bounded * limits * scale
     final = base + delta
  → env.step(final_action) → next_obs_raw, reward, done
  → 构建 transition {obs, action=residual, next_obs, reward, mask, done}
  → 写入 online replay buffer
```

**离线数据流**（已验证正确）：

```
bootstrap_base:
  → base policy 成功 rollout → residual action 全零 → 写入 offline buffer
pkl 离线数据:
  → 已是 residual 格式：规范化后直接入库
  → 专家动作格式：(expert - base) / (limits * xi * scale) 反解 residual → 入库
```

**混合采样**（已验证正确）：

```
symmetric_replay=true → offline_bs = batch_size/2, online_bs = batch_size/2
→ concat_batches(offline, online, axis=0) → 沿 batch 维拼接
→ DrQ-SAC update_high_utd(batch, utd_ratio=2)
   → critic 更新 2 次（128 样本 mini-batch）
   → actor + temperature 更新 1 次（256 样本 full batch）
```

### 1.3 关键逻辑验证

#### ✅ 残差动作组合公式

```python
# policy/action.py: compose_residual_action
clipped = clip(residual_action, -1, 1)
bounded = clip(clipped * xi, -xi, xi)
applied_delta = bounded * limits * residual_scale
final_action = base_action + delta_full
```

对应论文公式 `ā = a_base + a_δ`，其中 `a_δ` 的幅度通过 ξ 缩放到 `[-ξ, ξ]`，再乘以物理限幅 limits。**正确**。

#### ✅ Probing 步不进入 replay

训练脚本中 probing 阶段只推进环境状态和写 step log，**不构造 transition、不写入 replay buffer**。对应论文 Algorithm 1："The probing step only serves as state initialization and will not be added to the replay buffer."

#### ✅ Warmup 阶段用纯 base policy

`warmup_base_episodes` 和 `warmup_base_steps` 控制前 N 个 episode/step 使用零残差（等效纯 base policy 采集），对应论文 Section 3.1 的 warm-up 阶段。

#### ✅ 离线 buffer 初始化

`_bootstrap_offline_with_base_success` 用 base policy 自动收集成功轨迹，残差动作全零写入 offline buffer。对应论文："We first fill the offline buffer with successful rollouts from the base policy πb."

#### ✅ Cal-QL 预训练

`_pretrain_critic_with_calql` 用离线 buffer 做 critic 预训练，对应论文："the Q-function is initialized by a conservative objective such as Cal-QL."

#### ✅ Xi 调度器

`xi_scheduler` 从 `min_xi=0.1` 线性退火到 `base_xi=0.5`，对应论文："the delta action's magnitude is scaled down to [-ξ, ξ], where ξ ∈ [0, 1] is tuned by a scheduler."

#### ✅ 异步训练

`_AsyncLearner` 实现了 actor（主线程采集）和 learner（后台线程更新）的异步分离，对应论文 SERL 框架的异步采集-学习架构。线程安全通过 `replay_lock`, `actor_lock`, `learner_lock` 三把锁保障。

#### ✅ Episode 结束处理

`done` 时 `next_obs = zero`、`mask = 0.0` → TD target 中 `γ * mask * Q_next = 0`，不做 bootstrapping。**正确**。

#### ✅ Chunk 边界 next_obs 处理

chunk 末尾且未 done 时，预取下一段 base_chunk 构建 `next_obs_input`（包含下一段第 0 步 base action），并缓存避免重复推理。**正确**。

### 1.4 发现的问题与建议

#### ⚠️ 问题 1：Cal-QL 实现实际上是 CQL

`critic_loss_fn` 中的保守项实现为：

```python
lse_q = logsumexp(cat(q_rand, q_pi) / temp) * temp
cql_penalty = mean(lse_q - predicted_qs)
loss = td_loss + alpha * cql_penalty
```

这是标准 **CQL** 的 penalty。而 Cal-QL（Nakamoto et al., 2024）的核心改进是将保守下界**校准**到 behavior policy 的价值：

```
min_θ α * E_s[max(E_a~π[Q(s,a)], V^μ(s)) - E_a~D[Q(s,a)]] + TD_loss
```

即对 `logsumexp` 项取 `max(., V^μ(s))`，避免过度低估。当前实现**缺少这个 calibration 步骤**。

**影响**：对 critic 预训练阶段，可能导致 Q 值过度保守（低估），但因为后续在线阶段会纠正，实际影响可能有限。

**建议**：若要完全对齐论文，需在 `critic_loss_fn` 中增加 V^μ(s) 的计算和 max 操作。

#### ✅ 问题 2：encoder 类型已对齐（2025-03-03 修复）

配置已从 `encoder_type: small` 改为 `encoder_type: resnet-pretrained`，对齐论文的 **ResNet-10 with LayerNorm**。

已修改文件：
- `conf/train_residual_sac.yaml`
- `conf/eval_residual_fast.yaml`

#### ✅ 问题 3：超参已对齐论文值（2025-03-03 修复）

以下参数已在 `conf/train_residual_sac.yaml` 中调整为论文值：

| 参数 | 修改前 | 修改后（论文值） |
|------|--------|------------------|
| `replay.capacity` | 50,000 | 250,000 |
| `offline.capacity` | 50,000 | 250,000 |
| `warmup_base_episodes` | 5 | 100 |
| `max_online_env_steps` | 5,000 | 250,000 |
| `offline.bootstrap_base.success_episodes` | 5 | 50 |
| `offline.bootstrap_base.max_seed_attempts` | 500 | 5,000 |

#### ⚠️ 问题 4：`backup_entropy: false` 的选择

当前配置关闭了 backup entropy（TD target 中不减去 entropy bonus）。论文没有明确说明此选项，但标准 SAC 实践中通常启用。对于稀疏奖励场景，关闭可能更稳定但减少了探索驱动。

**建议**：可以实验对比 `backup_entropy: true/false` 的效果。

#### 🟢 小建议 1：异步模式下 learner 无速率限制

`_AsyncLearner._run` 在有足够数据时会连续不断地采样更新，导致实际 UTD 可能远高于配置的 `utd_ratio=2`。这在高 UTD 场景下是期望行为（类似 RLPD），但可能导致 GPU 使用率很高。

**建议**：如果需要精确控制 UTD ratio，可添加一个 `max_updates_per_second` 限速。

#### 🟢 小建议 2：Probing alpha 在 eval 中的随机性

评估时 `probing_alpha: 0.6` 意味着每个 episode 的 probing 步数是随机采样的 `U(0, 0.6*T)`。对于纯评估场景（非数据收集），可能希望 probing 为确定性或可控的。

---

## 二、与 PLD 论文的详细对比

### 2.1 已正确实现的部分

#### Stage 1: Online Specialist Acquisition ✅

| 论文要素 | 实现状态 | 说明 |
|----------|----------|------|
| 冻结 VLA backbone | ✅ | OpenPI 作为独立服务，完全冻结 |
| 轻量残差策略 πδ | ✅ | 3 层 MLP [256,256,256] + LayerNorm + Tanh |
| 残差条件化于 base action | ✅ | obs_input.state = [joint_14, base_action_14] |
| Off-policy SAC 框架 | ✅ | DrQ-SAC（SAC + 图像随机裁剪增强） |
| 双 buffer（online + offline）| ✅ | `replay_buffer` + `offline_buffer` |
| 对称 replay（50/50 混采）| ✅ | `symmetric_replay: true` |
| Base policy 成功轨迹初始化 offline buffer | ✅ | `_bootstrap_offline_with_base_success` |
| Cal-QL critic 预训练 | ⚠️ | 实现为 CQL，缺少 calibration 步骤 |
| Xi 缩放调度 | ✅ | 线性从 min_xi 退火到 base_xi |
| Warm-up 阶段（纯 base） | ✅ | `warmup_base_episodes` + `warmup_base_steps` |
| Clipped Double Q | ✅ | `critic_ensemble_size=2`, `critic_subsample_size=2` |
| Polyak target update | ✅ | `soft_target_update_rate=0.005` |
| Discount γ=0.99 | ✅ | `discount: 0.99` |
| 学习率 3e-4, AdamW | ✅ | `learning_rate: 3.0e-4`, `type: adamw` |
| Gradient clipping 1.0 | ✅ | `grad_clip_norm: 1.0` |
| Batch size 256 | ✅ | `batch_size: 256` |
| Target entropy -dim/2 | ✅ | SACAgent 自动计算 |
| Critic-to-actor ratio 2:1 | ✅ | `utd_ratio: 2` |
| OTF (On-the-fly) policy | ✅ | `otf_num_samples: 1`（默认关闭，可启用） |
| 异步采集-学习 | ✅ | `_AsyncLearner` 线程实现 |
| 稀疏二值奖励 | ✅ | `reward = 1.0 if success else 0.0` |

#### Stage 2: Base Policy Probing & Data Collection ✅

| 论文要素 | 实现状态 | 说明 |
|----------|----------|------|
| Base policy probing（随机步初始化）| ✅ | `probing_alpha: 0.6`, `T_base ~ U(0, αT)` |
| Probing 不入 replay | ✅ | 训练中 probing 步仅推进环境 |
| Hybrid rollout（前 t 步 base + 后续 residual） | ✅ | 训练和评估中均实现 |
| 数据收集（eval 阶段）| ✅ | `collect_dataset_path` 配置 |
| 仅收集成功轨迹 | ✅ | `collect_only_success: true` |

#### Stage 3: SFT ❌（刻意未实现）

这部分是你明确不复现的，不纳入对比。

### 2.2 与论文的差异

#### 差异 1：Cal-QL vs CQL（中等影响）

**论文**：使用 Cal-QL（Nakamoto et al., 2024）做 critic 预训练，核心是 calibrated conservative penalty，避免 Q 值过度低估。

**实现**：使用标准 CQL penalty（`logsumexp - data Q`），缺少 V^μ(s) 的 calibration。

**对齐方案**：需要在 `SACAgent.critic_loss_fn` 中：
1. 计算 behavior policy 的 V^μ(s)（可用 batch 中数据动作的 Q 值均值近似）
2. 对 logsumexp 项取 `max(logsumexp, V^μ)`

#### ~~差异 2：视觉编码器~~ → ✅ 已对齐（2025-03-03）

已将 `encoder_type` 从 `small` 改为 `resnet-pretrained`（预训练 ResNet-10），训练和评估配置均已同步。

#### ~~差异 3：Buffer 容量和 Warmup 步数~~ → ✅ 已对齐（2025-03-03）

已在 `conf/train_residual_sac.yaml` 中将所有超参调整为论文值：
- `replay.capacity`: 50,000 → 250,000
- `offline.capacity`: 50,000 → 250,000
- `bootstrap_base.success_episodes`: 5 → 50
- `bootstrap_base.max_seed_attempts`: 500 → 5,000
- `warmup_base_episodes`: 5 → 100
- `max_online_env_steps`: 5,000 → 250,000

#### 差异 4：Reward bias（无影响）

**论文**：Section B.2 消融实验显示 reward bias = 0.0 效果最好。

**实现**：无 reward bias。**已对齐**。

#### 差异 5：动作空间维度（环境差异，非对齐问题）

**论文 LIBERO**：7 DoF（6 DoF delta pose + 1 gripper），ξ = 0.5。

**论文 SimplerEnv**：7 DoF，ξ = 0.1。

**实现 RoboTwin**：14 DoF（双臂 ALOHA：2 × 6 arm + 2 × 1 gripper），ξ = 0.5。

**说明**：这是环境选择差异，非对齐问题。但 14 维动作空间显著增大了 RL 探索难度，可能需要更多交互步数才能收敛。

#### 差异 6：环境交互方式（环境差异）

**论文**：VLA 输出的是 7-DoF delta pose action，且 action chunk 通常较短。

**实现**：VLA 输出 14 维 action chunk（长度 10），残差每步推理一次。

**说明**：每步都推理残差（而不是每 chunk 推一次）提供了更细粒度的修正能力，但也增加了计算开销。这是合理的设计选择。

#### 差异 7：无 reward shaping 选项（无影响）

**论文**：消融实验了 survival cost reward bias，默认不用。

**实现**：无此选项，直接使用稀疏奖励。**已对齐**。

### 2.3 RoboTwin 环境特有的注意事项

| 方面 | LIBERO/SimplerEnv | RoboTwin | 注意事项 |
|------|-------------------|----------|----------|
| 机器人 | 单臂 7-DoF | 双臂 ALOHA 14-DoF | 探索空间翻倍，建议增加交互步数 |
| 相机 | 2-3 路 | 3 路（head + 双腕） | 已正确处理 |
| 动作类型 | Delta pose | Joint action | 残差 limits 可能需要针对性调整 |
| Step limit | ~300 | 任务相关 | 通过 `max_env_steps_per_episode` 控制 |
| 成功判定 | 稀疏二值 | 稀疏二值 | 一致 |
| 自动重置 | 环境内置 | `expert_precheck` + seed 跳过 | 合理 |

---

## 三、对齐清单：如果要完全对齐论文还需要做什么

### 3.1 必须修改（高优先级）

#### [A1] 将 CQL 改为 Cal-QL

**文件**：`serl_launcher/serl_launcher/agents/continuous/sac.py` → `critic_loss_fn`

**修改内容**：在 CQL penalty 中增加 calibration：

```python
# 计算 V^μ(s): 用 batch 中数据动作的 Q 值作为 behavior value 近似
q_data = predicted_qs.mean(dim=0)  # (B,)

# 原始 CQL: lse_q = logsumexp(cat(q_rand, q_pi) / temp) * temp
# Cal-QL:   lse_q_calibrated = max(lse_q, q_data)
lse_q = torch.logsumexp(q_cat / cql_temp, dim=-1) * cql_temp  # (Q,B)
lse_q_calibrated = torch.max(lse_q, q_data.unsqueeze(0).expand_as(lse_q))
cql_penalty = torch.mean(lse_q_calibrated - predicted_qs)
```

#### [A2] ~~使用 ResNet-10 编码器~~ → ✅ 已完成（2025-03-03）

已将 `conf/train_residual_sac.yaml` 和 `conf/eval_residual_fast.yaml` 中 `encoder_type` 改为 `resnet-pretrained`。

#### [A3] ~~调整超参到论文值~~ → ✅ 已完成（2025-03-03）

已在 `conf/train_residual_sac.yaml` 中完成全部超参对齐。

### 3.2 建议修改（中优先级）

#### [B1] Xi 初始值的考虑

论文消融实验建议 LIBERO 用 ξ=0.5。但 RoboTwin 是双臂 14 维，可能需要更小的 ξ（如 0.3）来避免过大扰动。建议做一次简单消融。

#### [B2] Arm delta limit 调优

当前 `arm_delta_limit: 0.03`。由于 RoboTwin 的动作语义（joint action vs delta pose）可能不同于 LIBERO，建议根据实际动作范围校准此值。

#### [B3] 多种子实验与 CI 报告

`run_stage12_repro.sh` 已有 3-seed 实验框架和 CI 聚合脚本 `aggregate_eval_ci.py`，与论文的"3 seeds, mean + 95% CI"报告方式一致。建议实际运行时使用此脚本。

### 3.3 可选改进（低优先级）

#### [C1] OTF (On-the-fly) policy 支持

论文消融实验表明 OTF sample size > 20 有显著性能增益。当前 `otf_num_samples: 1`（关闭）。可尝试设为 20：

```yaml
sac:
  otf_num_samples: 20
```

注意：OTF 会增加每次 TD target 计算的开销（需要对 next action 多次采样并计算 Q）。

#### [C2] 添加 JSRL 式课程学习

论文比较了 JSRL（Jump-Start RL）。可考虑在 probing 中实现类似课程：随训练进展，逐步增加 probing 步数（从易到难）。当前实现是固定 alpha，可以考虑随训练进展调整。

#### [C3] 更精细的 Critic-to-Actor ratio

论文消融了不同 update frequency，结论是"overall performance is largely insensitive"。当前 `utd_ratio: 2` 足够。

---

## 四、总结

### 代码质量

实现**整体质量较高**，核心训练循环、数据流、动作组合、观测构建等关键逻辑均正确。代码组织清晰，模块化良好，配置通过 Hydra YAML 灵活管理。异步训练的线程安全处理也到位。

### 论文复现完成度

对 Stage 1 和 Stage 2 的复现**完成度约 95%**。剩余差距：

1. **Cal-QL → CQL**：差一个 calibration 步骤（影响 critic 预训练质量，但在线阶段会纠正）

已完成的对齐（2025-03-03）：
- ~~**Visual encoder**~~：已从 `small` 改为 `resnet-pretrained`（预训练 ResNet-10）
- ~~**超参默认值**~~：buffer 容量、warmup 步数、bootstrap 轨迹数等已全部调整为论文值

代码框架已支持所有需要的功能（ResNet encoder、OTF、xi scheduler 等），无需结构性改动。

### RoboTwin 环境注意

14 维双臂动作空间比论文的 7 维单臂大一倍，这会：
- 增加探索难度 → 可能需要更多交互步数
- 增加残差策略的学习难度 → 可考虑更保守的 ξ 值
- 增加环境复杂度 → 需要更多 bootstrap 成功轨迹

建议先在简单任务（如 `adjust_bottle`）上验证完整流程，确认收敛后再扩展到更复杂的任务。
