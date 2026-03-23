# LIBERO Chunk-Step Sequence Replay 改造说明

本文说明本次改造在 `serl_torch/examples/libero` 中具体改了什么、如何开启 `chunk-step` 模式，以及当前版本的边界条件。

目标：

- 保留原有步级 residual 训练流程；
- 新增 `chunk-step` 模式：actor 一次输出整段 residual chunk，环境一次 `step_chunk` open-loop 执行；
- replay 采用 step-level 存储，在 `sample()` 时滑窗组装 chunk batch；
- 尽量复用现有 SAC / async learner / eval 流程。

---

## 1. 这次改了什么

### 1.1 环境侧

- `examples/libero/env_wrappers/task_env.py`
  - 新增 `step_chunk(actions)`。
  - 支持输入 `[C, 7]` 或平铺后的 `[C*7]`。
  - 返回：
    - `obs`
    - `observations`
    - `rewards`
    - `dones`
    - `infos`
    - `reward_sum`
    - `num_steps`

- `examples/libero/env_wrappers/remote_task_env.py`
  - 新增远程 `step_chunk` RPC 调用。

- `examples/libero/scripts/libero_env_server.py`
  - 新增 RPC method `step_chunk`，转发到本地环境包装器。

### 1.2 动作侧

- `examples/libero/policy/action.py`
  - 新增 `as_numpy_action_chunk(...)`。
  - 新增 `compose_residual_action_chunk(...)`。
  - 原有单步 `compose_residual_action(...)` 保留不变。

- `examples/libero/policy/__init__.py`
  - 导出新的 chunk helper。

### 1.3 Replay / 训练采样

- `examples/libero/utils/step_chunk_replay.py`
  - 新增 `StepChunkReplayBuffer`。
  - 写入时仍是 step-level：
    - `observations`
    - `actions`
    - `rewards`
    - `dones`
    - `episode_id`
    - `episode_step`
  - 采样时组装 chunk transition：
    - `observations = x_t`
    - `actions = [a_t, ..., a_{t+C-1}]`
    - `rewards = sum(gamma^i r_{t+i})`
    - `next_observations = x_{t+k}`
    - `masks = (1-done) * gamma^(k-1)`
    - `dones`
    - `chunk_steps`
  - 采样性能修复：
    - 不再在每次 `sample()` 时全量扫描 replay。
    - 改为在 `insert()` 时增量维护合法起点索引，`sample()` 直接从合法起点池抽样。

### 1.4 训练 / 评估脚本

- `examples/libero/scripts/train_residual_sac.py`
  - 新增 `chunk_step` 配置分支。
  - `chunk_step.enabled=false` 时，走原来的步级 residual 训练。
  - `chunk_step.enabled=true` 时：
    - agent 输出维度变成 `residual.action_dim * residual.chunk_horizon`
    - rollout 用 `env.step_chunk(...)`
    - replay 改用 `StepChunkReplayBuffer`
    - 每个环境步仍单独写入 step-level 数据
    - learner 采样时自动拿到 chunk batch
  - 语义对齐修复（默认按 `env-step`）：
    - `warmup_base_steps`、`random_steps`、`update_every`、`checkpoint_period` 都按环境步对齐。
    - chunk 模式下会在 warmup/random 边界自动截断执行长度，避免一个 chunk 跨语义边界。

- `examples/libero/scripts/eval_residual_fast.py`
  - 新增 `chunk_step` 配置分支。
  - `chunk_step.enabled=true` 时，评估也改为 chunk actor + `step_chunk` 执行。

- `examples/libero/scripts/async_eval_watch.py`
  - 异步评估现在会把顶层 `chunk_step.*` 配置一并传给 eval 脚本。

### 1.5 配置文件

- `examples/libero/conf/train_residual_sac.yaml`
  - 新增顶层 `chunk_step` 配置段，默认关闭。

- `examples/libero/conf/eval_residual_fast.yaml`
  - 新增顶层 `chunk_step` 配置段，默认关闭。

- `examples/libero/conf/train_residual_sac_chunk_step_sequence.yaml`
  - 新增训练示例配置。

- `examples/libero/conf/eval_residual_fast_chunk_step_sequence.yaml`
  - 新增评估示例配置。

---

## 2. 新增配置怎么理解

新增顶层配置段：

```yaml
chunk_step:
  enabled: false
  sample_stride: 1
  require_full_horizon: false
  pad_action_to_horizon: true
  scheduler_clock: env_step
```

各字段含义：

- `enabled`
  - `false`：保持原来的步级 residual 逻辑。
  - `true`：开启 chunk actor + `step_chunk` + step-sequence replay。

- `sample_stride`
  - sequence replay 采样起点的步长。
  - `1` 表示所有步都可作为 chunk 起点。
  - `2` 表示只采 `t=0,2,4,...` 这类起点。

- `require_full_horizon`
  - `false`：允许 episode 尾部的 `k < C` 样本。
  - `true`：只采完整 `C` 步 chunk。

- `pad_action_to_horizon`
  - 当前实现必须为 `true`。
  - 因为 critic / actor 输入维度固定，尾部不足 `C` 时会把剩余动作位补零。

- `scheduler_clock`
  - `env_step`（默认）：xi / residual scale 调度按真实环境步走。
  - `policy_step`：xi / residual scale 调度按 chunk 决策步走。

---

## 3. 如何开启 chunk-step

最小建议配置：

```yaml
residual:
  chunk_horizon: 10

chunk_step:
  enabled: true
  sample_stride: 2
  require_full_horizon: false
  pad_action_to_horizon: true
  scheduler_clock: env_step

offline:
  enabled: false
```

关键点：

- `residual.action_dim` 仍表示单步 residual 维度；
- agent 的实际输出维度会在代码里自动变成 `action_dim * chunk_horizon`；
- 当前版本要求 `offline.enabled=false`。

---

## 4. 建议直接使用的示例配置

训练：

- `examples/libero/conf/train_residual_sac_chunk_step_sequence.yaml`

评估：

- `examples/libero/conf/eval_residual_fast_chunk_step_sequence.yaml`

如果你用 Hydra 直接跑，通常可以这样：

```bash
python examples/libero/scripts/train_residual_sac.py --config-name train_residual_sac_chunk_step_sequence
```

```bash
python examples/libero/scripts/eval_residual_fast.py --config-name eval_residual_fast_chunk_step_sequence
```

---

## 5. 当前版本的边界条件

### 5.1 已支持

- 原有步级 residual 训练与评估；
- chunk actor 输出整段 residual；
- 环境 open-loop `step_chunk` 执行；
- step-level replay 存储；
- 采样时按 stride 滑窗组 chunk；
- async eval 传递 `chunk_step` 配置。

### 5.2 当前没有接入

- offline residual / bootstrap 到 chunk-step 路径。

也就是说：

- `chunk_step.enabled=true` 时，当前要求 `offline.enabled=false`。

### 5.3 当前仍然是最小观测改法

当前 chunk actor 的输入仍然是：

- 当前观测
- 当前时刻的 base action（即 `base_chunk[0]`）

并没有把整个 `base_chunk` 一起编码进 observation。

这条路径的优点是改动最小、兼容当前 encoder/agent 架构；
如果后续要进一步提升 chunk actor 表达能力，再考虑把 `base_chunk` 一并编码进 observation。

---

## 6. 你后续最可能关心的地方

如果接下来继续迭代，优先级建议是：

1. 跑通在线训练，看 sequence replay 的采样稳定性。
2. 对比 `sample_stride=1` 和 `sample_stride=2`。
3. 再决定是否要把 `base_chunk` 整体并入 observation。
4. 最后再补 offline chunk-step 路径。
