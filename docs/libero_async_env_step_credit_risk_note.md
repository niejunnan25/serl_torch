# LIBERO 暂缓修复问题台账（Deferred Issues Log）

本文档用于记录 **当前已识别但暂不修复** 的问题，统一说明：

- 当前现状（代码与行为）
- 风险与影响范围
- 规避策略（短期）
- 推荐修复方案（长期）
- 不修复可能造成的后果

后续新增问题，请按本文模板追加新条目。

---

## 文档使用规则

1. 每个问题使用独立 `Issue-XXX` 小节。
2. 必须包含可定位代码路径与关键行号。
3. 必须明确“是否影响当前主实验”。
4. 状态建议使用：`Open` / `Deferred` / `Accepted Risk` / `In Progress` / `Resolved`。

---

## Issue-001: Async Learner 未按 Env-Step 配额更新

- 状态：`Accepted Risk`
- 优先级：`High`
- 首次记录：2026-03-22
- 影响模块：`examples/libero/utils/async_learning.py`、`examples/libero/scripts/train_residual_sac.py`
- 当前目标：吞吐优先与更快收敛（非跨机器严格复现优先）

### 1) 当前现状

异步 learner 目前是“buffer 达标后尽可能持续更新”，而不是“每收集 N 个 env-step 才允许做 M 次 update”。

关键代码：

- `examples/libero/utils/async_learning.py:315-318`
  仅在 `len(online_buffer) >= training_starts` 时允许采样。
- `examples/libero/utils/async_learning.py:328-342`
  后台线程循环执行，采样到 batch 就继续更新。
- `examples/libero/utils/async_learning.py:366-383`
  每次 `update_high_utd` 后仅做 `update_steps += 1`，不和 env-step 建立配额关系。
- `examples/libero/scripts/train_residual_sac.py:298` 与 `:545`
  `async_update_frequency` 传入 async learner；该参数只用于“多久同步一次 actor 权重”，不是 env-step 更新门控。

### 2) 与 probing 的关系

该问题与 probing **无关**。
即使你关闭 probing，只要 `training.async.enabled=true`，异步 learner 仍会出现更新密度随机器速度漂移的问题。

### 3) 风险与影响

- 同一 YAML 在不同机器/负载下，`updates / env_step` 发生漂移。
- 与同步路径（按 env-step 的 `update_every`、`updates_per_step`）语义不等价。
- 训练可复现性与跨实验公平对比变差。
- 在“追求最快收敛”目标下，该问题当前不是阻塞项，但需要监控避免过更新。

### 4) 当前执行策略（吞吐优先）

1. 继续使用异步 learner，不切回同步路径。
2. 将默认 `training.training_starts` 提高到 `1000`，避免过早在小样本上高强度更新。
3. 监控 `learner_update_steps / global_env_step`（或 `train_env_step`）作为核心健康指标。
4. 若该比值长期过高且成功率不涨，优先降低更新强度（例如减小 `utd_ratio` 或增加 `idle_sleep_sec`）。

### 5) 推荐长期修复（Env-Step Credit）

建议引入 `update_credits`（更新额度）：

1. 采集侧按 env-step 计算触发次数 `triggers`（与同步逻辑一致：`update_every` + `training_starts`）。
2. 发放额度：`update_credits += triggers * updates_per_step`。
3. learner 线程仅在 `update_credits > 0` 时做一次 update，更新后 `update_credits -= 1`。
4. `training_starts` 作为前置门控保留：未达标时可累计额度，达标后再消耗。

### 6) 不修复的潜在后果

- 同配置在不同硬件上训练曲线不可直接对比。
- 当采集慢/学习快时可能过拟合旧数据；采集快/学习慢时可能学习滞后。
- chunk-step 场景下该偏差更难直观看出，排障成本更高。

### 7) 何时再提修复优先级

在出现以下任一情况时，将本项从 `Accepted Risk` 升级为 `In Progress`：

1. 需要做跨机器严格对比、ablation 或复现实验。
2. 观测到明显的更新过密/过稀导致训练不稳定。
3. 同一配置在不同机器上的收敛差异超过可接受范围。

---

## Issue-002: Chunk 模式下 Checkpoint Step 语义可能偏移

- 状态：`Accepted Risk`
- 优先级：`Medium`
- 首次记录：2026-03-22
- 影响模块：`examples/libero/scripts/train_residual_sac.py`
- 当前目标：吞吐优先与更快收敛（非 checkpoint 严格对齐优先）

### 1) 当前现状

chunk 模式中，训练更新是在 chunk 结束后按本段 env-step 触发；checkpoint 也是按跨过的 period 命中点落盘。
当一次 chunk 跨越 checkpoint 周期边界时，`checkpoint_<step>.pt` 可能包含边界后触发的更新。

关键代码：

- `examples/libero/scripts/train_residual_sac.py:1334-1341`
  按 chunk 跨度统计本段 env-step 的 update triggers。
- `examples/libero/scripts/train_residual_sac.py:1345-1398`
  先执行本段更新。
- `examples/libero/scripts/train_residual_sac.py:1473-1485`
  再按 period hits 写 checkpoint。

### 2) 风险与影响

- checkpoint 文件名的 step 标签可能“标签正确、状态略偏后”。
- 在严格对齐恢复训练、对齐评估曲线时，可能出现小幅时间语义偏差。

### 3) 当前执行策略（暂不修复）

1. 接受该偏差，优先保证 chunk 流程吞吐和实现简洁。
2. 对关键实验，优先用评估曲线趋势和成功率结论，而不是过度依赖单点 step 的绝对语义。

### 4) 推荐长期修复

1. 将 checkpoint 与 update 执行顺序改为“先命中点快照，再消费本段剩余更新”。
2. 或引入“边界精确快照”逻辑，在 chunk 内部按命中点切分 update 批次。

---

## Issue-003: Step-Level 组 Chunk 采样的阶段边界语义风险

- 状态：`Accepted Risk`
- 优先级：`Medium`
- 首次记录：2026-03-22
- 影响模块：`examples/libero/utils/step_chunk_replay.py`、`examples/libero/scripts/train_residual_sac.py`
- 当前目标：最小改动兼容旧流程

### 1) 当前现状

执行侧已经在 chunk 开始前做了 warmup/random 边界截断，避免单次执行 chunk 跨边界。
但 replay 采样是从 step-level 数据按合法起点组 chunk，合法性判定依赖 episode 连续性与 done，不显式标注训练阶段边界。

关键代码：

- `examples/libero/scripts/train_residual_sac.py:1157-1177`
  执行侧 chunk 会在 warmup/random 边界前截断。
- `examples/libero/utils/step_chunk_replay.py:144-163`
  起点合法性按 episode/transition 连续性判定。

### 2) 风险与影响

- 在阶段切换邻域，采样得到的 chunk 可能混入不同阶段统计分布（语义不如严格 chunk transition 纯粹）。
- 一般不影响“可训练性”，但会降低“严格阶段控制”的可解释性。

### 3) 当前执行策略（暂不修复）

1. 保持 step-level buffer + 采样组 chunk 的最小改动方案。
2. 通过 `sample_stride` 控制 chunk 起点密度，降低边界邻域混样概率。

### 4) 推荐长期修复

1. 在 replay 条目中增加 phase_id / regime 标记，将合法起点约束扩展为“同阶段连续”。
2. 对需要严格实验语义的配置，引入可选 `strict_phase_boundary=true` 采样约束。

---

## 新问题追加模板

复制以下模板新增条目：

```md
## Issue-XXX: <问题标题>

- 状态：`Open|Deferred|Accepted Risk|In Progress|Resolved`
- 优先级：`High|Medium|Low`
- 首次记录：YYYY-MM-DD
- 影响模块：<file paths>

### 1) 当前现状
- <行为 + 代码位置>

### 2) 触发条件
- <在什么配置/模式下触发>

### 3) 风险与影响
- <对正确性/性能/复现性的影响>

### 4) 短期规避策略
1. ...
2. ...

### 5) 推荐长期修复
1. ...
2. ...

### 6) 不修复的潜在后果
- ...
```
