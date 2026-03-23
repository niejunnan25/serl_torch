# LIBERO Chunk Steps（`step_chunk`）实现说明（SMDP 版本）

本文面向 `examples/libero` 当前代码，给出“残差策略一次输出多步动作，并 open-loop 一次执行整段 `step_chunk`”的实现方案。

目标是：

- 支持 chunk 级策略输出（例如 `C=10`，动作维 `d=14`，actor 输出 `C*d=140` 维）；
- 环境支持 chunk 执行（`step_chunk`）；
- critic 按宏动作/SMDP 语义学习，避免时序目标不一致；
- 尽量复用现有训练框架（SAC + replay + async）并保持向后兼容。

---

## 1. 为什么要用 SMDP 语义

当策略每次决策输出的是 `C` 步动作 chunk，而环境一次执行 `C` 步时，决策频率从“每步一次”变成了“每 `C` 步一次”。

因此一个 transition 不再是单步 `(s_t, a_t, r_t, s_{t+1})`，而是宏动作 transition：

- `s_t`
- `u_t = a_{t:t+C-1}`（chunk 宏动作）
- `R_t = \sum_{i=0}^{k_t-1} \gamma^i r_{t+i}`（chunk 内累计折扣奖励）
- `s_{t+k_t}`
- `k_t`（chunk 实际执行步数，可能 `< C`，比如提前 done）

对应 Bellman 目标：

\[
Q(s_t, u_t) = R_t + \gamma^{k_t}(1-d_t) V(s_{t+k_t})
\]

这就是本文所说“critic 要按宏动作/SMDP 学”。

如果你执行 chunk 但仍用单步目标，会出现：

- chunk 后半段奖励被漏掉；
- bootstrap 折扣因子错误（`\gamma` vs `\gamma^{k_t}`）；
- value 学习和真实执行机制错位。

---

## 2. 对照 RLT 的关键实现点

RLT（`https://www.pi.website/download/rlt.pdf`）在 chunk 上的核心做法是：

1. VLA 本身输出长 chunk（文中 `H=50`，50Hz 下约 1s），执行时只执行前缀再重规划；
2. RL actor 也输出 chunk（文中常用 `C=10`）；
3. critic 直接建模 chunk 价值 `Q(x, a_{1:C})`，TD 目标用 chunk 累计奖励 + `\gamma^C` bootstrap；
4. replay 存 chunk transition；
5. 可对 chunk 起点做子采样（文中 stride=2）提高样本利用率。

你当前只关心 chunk steps，不关心“参考 VLA chunk 做局部改进”，这是可以的；该部分可选。

### 2.1 RLT 原文摘录（用于实现对齐）

下面是与实现直接相关的原文短句（RLT）：

- `"both policies and critics operate over action chunks"`（Sec. III）
- `"chunk-level C-step estimate of the value"`（Sec. III）
- `"sampled from the replay buffer B"`（Sec. IV, critic TD）
- `"storing intermediate steps into the replay buffer"`（Sec. V）
- `"pick a stride of 2"`（Sec. V）

对应到工程语义就是：

- actor/critic 都是 chunk 动作空间；
- critic 目标是 C-step/`gamma^C` 风格（提前 done 时用实际 `k`）；
- replay 训练样本是 chunk transition，而不是只能做单步；
- 可通过 stride 子采样提升每秒数据利用率。

---

## 3. 在当前代码库中的改造边界

当前关键路径：

- 环境封装：
  - `examples/libero/env_wrappers/task_env.py`
  - `examples/libero/env_wrappers/remote_task_env.py`
  - `examples/libero/scripts/libero_env_server.py`
- 训练主循环：`examples/libero/scripts/train_residual_sac.py`
- 评估循环：`examples/libero/scripts/eval_residual_fast.py`
- Agent/Replay 构造：
  - `examples/libero/utils/config_utils.py`
  - `serl_launcher/serl_launcher/data/replay_buffer.py`
  - `serl_launcher/serl_launcher/agents/continuous/sac.py`

现状是“单步 residual action 存 replay + 单步更新节奏”，要支持 chunk 宏动作，需要改“采样单位”和“target 语义”。

---

## 4. 推荐的数据与训练语义（核心）

### 4.0 论文表述边界（避免误读）

RLT 论文明确给出的是 **chunk/SMDP 训练语义**，而不是固定的 replay 字段命名规范。

- Eq.(3) 给出 chunk TD 目标（chunk 内奖励累积 + `gamma^C` bootstrap）。
- Algorithm 1 给出存储形式 `⟨x_t, a_{t:t+C-1}, ..., x_{t+1}⟩`。其中 `t` 是 chunk 决策索引，因此该 `x_{t+1}` 对应物理时间上的 chunk 下一状态（可理解为 `x_{t+k}`，`k<=C`）。
- 文中还给出 chunk 子采样示例：`<x0,a0:C>`, `<x2,a2:C+2>`（stride=2）。

因此，本文档中的 `obs_t -> obs_{t+k}` 是语义解释（SMDP next state），不是在声称论文给了固定 JSON 字段模板。

### 4.1 Chunk transition 数据结构

建议每条 replay transition 至少包含：

- `observations`: `x_t`
- `actions`: `u_t`（shape `[C*d]` 或 `[C, d]`，推荐存平铺 `[C*d]`）
- `next_observations`: `x_{t+k}`
- `rewards`: `R_t = sum(gamma^i * r_{t+i})`
- `masks`: `(1-done_tk) * gamma^(k_t-1)`
- `dones`: `done_tk`
- `chunk_steps`: `k_t`（调试与统计用）

说明：

- 你现有 SAC 代码是 `target_q = rewards + discount * masks * target_next_q`。
- 若 `masks = (1-done) * gamma^(k-1)`，则等价得到 `R_t + gamma^k * (1-done) * Q_next`。
- 这样可以最小化改 SAC 公式本身。

### 4.2 计数器语义

建议保留两套步数：

- `global_env_step`: 实际环境步数（每执行 1 个底层动作 +1）
- `global_policy_step`: chunk 决策步数（每执行 1 个 chunk +1）

调度器（如 xi scheduler、residual_scale scheduler）要明确挂在哪个时钟上：

- 想随物理步变化：挂 `global_env_step`
- 想随决策步变化：挂 `global_policy_step`

### 4.3 两种落地方式：何时组装 chunk

在工程上有两种实现，都可以实现相同的 SMDP 目标：

1. 预聚合 chunk transition（写入 replay 时组装）
- 插入时直接写 `obs_t, action_chunk, R_t, obs_{t+k}, mask_k`。
- learner 侧最简单，最容易复用当前随机采样 replay + async prefetch 流程。
- 更适合“先最小改动跑通”。

2. 逐步存储 + 采样时滑窗组装（sequence replay）
- buffer 先存 step-level：`obs_t, a_t, r_t, done_t ...`。
- sample 时再组 `a_t...a_{t+k-1}`、`R_t`、`obs_{t+k}`。
- 更灵活（可变 `C`、可变 stride、多种 n-step 实验），但 sampler 复杂度高，且要严格处理 episode 边界。

两者在数学上可以等价；主要区别是“组装发生在数据写入阶段，还是训练采样阶段”。

---

## 5. `step_chunk` 接口设计

### 5.1 本地环境接口

在 `LiberoTaskEnv` 增加：

- `step_chunk(actions: np.ndarray) -> Dict`

其中 `actions` 形状建议支持：

- `[C, d]`
- 或 `[C*d]`（内部 reshape）

返回建议：

- `obs`: 最后一步观测
- `reward_sum`: 不折扣累计奖励（原始和）
- `done`, `truncated`, `info`
- `num_steps`: 实际执行步数 `k_t`
- `rewards`: 每一步 reward 列表（训练端可自行折扣）
- `infos`: 每一步 info（可选，用于诊断）

### 5.2 远程 RPC

`remote_task_env.py` 与 `libero_env_server.py` 同步加 `step_chunk` RPC 方法。

向后兼容要求：

- 保留 `step(action)` 原接口不变；
- 旧训练逻辑不受影响。

---

## 6. 训练循环怎么改（最小可用版本）

以下是最小实现路径（不引入参考 VLA chunk 正则）：

1. 每次决策构造 `x_t`；
2. actor 输出 chunk residual 动作 `u_t`（shape `[C, d_res]`）；
3. 与 base chunk 融合得到最终执行 chunk `a_exec[0:C]`；
4. 调用 `env.step_chunk(a_exec)` 执行整段；
5. 根据 `rewards[0:k]` 计算折扣累计 `R_t`；
6. 构造 `x_{t+k}` 并插入一条 chunk transition；
7. `global_policy_step += 1`，`global_env_step += k`。

注意事项：

- 提前 done 时 `k < C`，后半段动作不再执行；
- bootstrap 必须用 `k` 对应的折扣；
- replay 里动作维度是 chunk 维（`C*d_res`）。

---

## 7. Critic/Actor 网络层面改动

### 7.1 最小改法

不改 SAC 算法结构，只改 action 维度：

- actor 输出维度：`C * d_res`
- critic 动作输入维度：`C * d_res`
- replay action_space 同步改为 `(C * d_res,)`

当前 `serl_launcher` 的 SAC/MLP 接口本身支持任意最后维度动作向量，因此这条路径改动较小。

### 7.2 需要同步校准的超参数

- `target_entropy`（动作维变大后默认值会更负）
- `std_min/std_max`（chunk 输出更高维，采样噪声总量会变化）
- `batch_size` 与 `utd_ratio`（高维动作 + chunk return 方差变化）

---

## 8. 观测构造建议（chunk actor）

当前 `build_residual_step_obs` 主要拼单步 base action。对于 chunk actor，建议显式支持：

- 输入 `base_chunk_prefix`（例如 `[C, d_base]`）
- 可以拼接前 `K` 个 base action（`K<=C`）或整体降维编码后再拼入状态

最小版本可先用：

- 仍只拼第 1 步 base action（可跑通）
- 但长期建议拼 chunk 信息，否则 actor 对后续步欠感知

---

## 9. YAML 设计（向后兼容）

建议新增：

```yaml
residual:
  chunk_horizon: 10           # 现有字段，可复用
  action_dim: 14              # 每步残差动作维

training:
  chunk_policy:
    enabled: false            # 默认 false，保持旧流程
    execute_mode: open_loop   # open_loop / receding_horizon
    decision_stride: 1        # 预留；open_loop 下通常=1
    use_step_chunk: true
    reward_aggregation: discounted_sum
    store_stride: 1           # 可设 2，做 chunk 起点子采样
```

默认 `enabled=false` 时完全走现有单步逻辑。

---

## 10. 评估与日志

评估脚本需与训练一致的 chunk 决策语义，否则指标不可比。

建议日志新增：

- `chunk_len_nominal`
- `chunk_len_executed`
- `chunk_return_discounted`
- `chunk_return_undiscounted`
- `policy_step`
- `env_steps_per_policy_step`

同时保留每底层步日志（可选），用于诊断接触失败位置。

---

## 11. 分阶段落地计划（建议顺序）

### Phase 1（低风险，先跑通）

- 加 `step_chunk`（本地+远程）
- 训练改为 chunk transition（SMDP target 语义）
- actor/critic 动作维改成 `C*d`
- offline 数据先关掉（`offline.enabled=false`）

### Phase 2（提效）

- 增加 chunk 子采样（stride=2）
- async 流程优化（rollout/learn 并行）

### Phase 3（增强）

- 观测中引入 base chunk 信息
- 可选加入“参考 chunk 正则/condition”（若后续需要更稳的探索）

---

## 12. 基于当前代码的改造清单（按文件）

下面按你当前仓库的实际文件给出“必须改/建议改”清单。

### 12.1 必须改（chunk 级残差最小可用）

- `examples/libero/env_wrappers/task_env.py`
  - 新增 `step_chunk(actions)`；
  - 返回 `obs_last/rewards_per_step/done/truncated/info/num_steps`。
- `examples/libero/env_wrappers/remote_task_env.py`
  - 新增 `step_chunk` 客户端 RPC 调用与返回解析。
- `examples/libero/scripts/libero_env_server.py`
  - 新增 RPC method `step_chunk`，转发到 `LiberoTaskEnv.step_chunk`。
- `examples/libero/scripts/train_residual_sac.py`
  - 收集逻辑从“chunk 内逐步 `env.step` + 每步入 buffer”改为“每个决策一次 `env.step_chunk` + 入一条 chunk transition”；
  - replay 插入字段改为 chunk 语义：`actions` 维度 `C*d_res`，`rewards` 为 chunk 折扣和，`masks` 含 `gamma^(k-1)`，并记录 `chunk_steps=k`；
  - `global_policy_step` 改为“每个 chunk 决策 +1”，`global_env_step += k`；
  - schedule（xi/scale）明确按 `global_policy_step` 或 `global_env_step` 选择。
- `examples/libero/utils/config_utils.py`
  - `build_drq_agent(..., action_dim=...)` 传入 chunk 维 `C*d_res`；
  - 保持其它 SAC 架构不变（最小改动）。
- `examples/libero/conf/train_residual_sac*.yaml`
  - 增加/启用 `training.chunk_policy.enabled`、`use_step_chunk`、`store_stride` 等配置；
  - `residual.action_dim` 仍是单步维度，新增 `effective_action_dim = chunk_horizon * residual.action_dim`（可在代码内推导，不必写死）。

### 12.2 建议同步改（保证训练-评估一致）

- `examples/libero/scripts/eval_residual_fast.py`
  - 与训练一致，改成 chunk 决策 + `step_chunk` 执行；
  - 避免“训练 chunk / 评估单步”分布错位。

### 12.3 若保留 offline 管线，还需要改

- `examples/libero/data/offline_residual.py`
  - 离线样本由 step transition 改为 chunk transition（或采样时组装）；
  - `actions` 改为 chunk 维残差。
- `examples/libero/data/offline_bootstrap.py`
  - base bootstrap 也改成 chunk transition 语义，避免在线/离线 target 不一致。

### 12.4 可后置优化（不是最小闭环必需）

- `examples/libero/policy/observation.py`
  - 当前是单步 `base_action` 融合；后续可升级为融合 `base_chunk`（或其编码）。
- `examples/libero/policy/action.py`
  - 增加 chunk 版 compose helper（对 `[C,d]` 批量组合与限幅），减少训练脚本内 reshape/循环代码。

---

## 13. 常见坑位清单

1. 只改 actor 输出，不改 replay target 折扣（错误）
2. 提前 done 时仍按 `C` 折扣（错误，应用实际 `k`）
3. 评估还在单步循环（与训练不一致）
4. 忘了区分 `global_env_step` 和 `global_policy_step`
5. `action_dim` 改成 chunk 后，`target_entropy` 未重设导致温度不稳定

---

## 14. 结论

对于“chunk actor 输出多步 + 一次性 `step_chunk` 执行”的目标，最稳妥做法是：

- 把训练单位从单步 MDP 迁移到 chunk/SMDP；
- replay 存 chunk transition；
- critic 用 chunk return + `gamma^k` bootstrap；
- 先保证语义正确，再做效率优化。

这样可以在最小破坏现有框架的前提下，把 chunk steps 真正做对。
