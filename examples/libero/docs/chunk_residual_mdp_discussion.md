# LIBERO Chunk Residual RL 的问题定义讨论

## 1. 当前真正的问题

当前 LIBERO residual RL 已经不是单步 residual，而是 **chunk residual control**：

- 观测输入包含：
  - `images`
  - `robot_proprio`
  - `base_action_chunk`
  - `alpha`
- residual actor 输出：
  - 一个与 `base_action_chunk` 展平后同维度的 residual chunk
- 最终执行：
  - `final_action_chunk = compose(base_action_chunk, residual_chunk)`

真正需要想清楚的是：

**训练时，什么才算一个合法的 decision boundary。**

也就是：

- 只允许 chunk 起点是 decision boundary？
- 还是任意 step 都可以重新作为 decision boundary？

这不是代码细节，而是问题定义本身。

## 2. 当前两种不同语义

### 语义 A：Strict Chunk-Boundary Decision

- 在 chunk 起点算一次 `base_action_chunk_t`
- residual actor 基于 `(obs_t, base_action_chunk_t)` 输出 `residual_chunk_t`
- 然后整段 chunk 执行
- 中间 step 不被视为新的 decision point

这时状态更接近：

```text
x_t = (obs_t, base_action_chunk_t, alpha)
```

但只在 chunk 边界定义。

优点：

- 语义最干净
- 最符合“一次 chunk = 一次决策”
- 不需要为 chunk 内部每一步重新算 `base_action_chunk`

缺点：

- 数据复用低
- replay 不能自然地用 `sample_stride=1`

### 语义 B：Step-Wise Receding-Horizon Decision

- 任意 step 都允许作为新的 decision point
- 所以对每一个 `t`，都定义：

```text
x_t = (obs_t, base_action_chunk_t, alpha)
```

其中 `base_action_chunk_t = f(obs_t)`，由 base policy 在当前 step 的 observation 上重新推出来。

这时，如果 replay 要存 `next_observations`，就必须存：

```text
x_{t+1} = (obs_{t+1}, base_action_chunk_{t+1}, alpha)
```

也就是说，必须在每一步之后都重新计算：

- `next_base_action_chunk`
- `next_residual_obs`

优点：

- 支持 `sample_stride=1`
- 数据复用最高
- 与当前 step-window replay 的滑动窗口采样天然兼容

缺点：

- 训练语义比执行语义更密
- 执行时并没有真的每一步都重新规划一整个 chunk
- 问题定义不如 strict chunk boundary 那么纯

## 3. 这是否违反 MDP

如果 base policy 是固定的，或者在训练中被视为固定映射：

```text
base_action_chunk_t = f(obs_t)
```

那么把它并入状态：

```text
x_t = (obs_t, f(obs_t), alpha)
```

本身 **不违反 MDP**。

原因是：

- `f(obs_t)` 只是当前 observation 的确定性特征
- 不是未来轨迹信息泄漏
- 只要原始 `obs_t` 足以定义环境状态，这种扩展状态仍然是自洽的

真正微妙的点不在 MDP，而在：

**你想学的是 strict chunk-boundary control，还是 step-wise receding-horizon control。**

## 4. 如果把 `sample_stride` 设成 `execute_horizon`，问题会消失吗

不完全是。

### 情况 A：固定 `chunk_horizon`

如果你的真实意图是：

- 只在 chunk 边界采样
- 每一个 chunk 对应一次决策

那么更准确的说法不是：

```text
sample_stride = execute_horizon
```

而是：

```text
sample_stride = chunk_horizon
```

并且 replay 只允许从 chunk 起点采样。

在这个定义下，上面的“每一步都重算 `next_base_action_chunk`”问题基本就不存在了，因为：

- chunk 内部 step 不再被当成新的 decision point
- replay 也不会从 chunk 内部起采

### 情况 B：`execute_horizon` 是动态的

当前实现里，`execute_horizon` 在 episode 末尾或者全局 `max_env_steps` 接近上限时，可能小于 `chunk_horizon`。

这意味着：

- `execute_horizon` 不是一个全局固定值
- 但 `sample_stride` 通常是 replay buffer 的全局固定配置

所以严格来说：

**`sample_stride = execute_horizon` 这个说法在实现上并不稳定。**

因为 `execute_horizon` 会随时变化，而 `sample_stride` 不是逐 chunk 动态变化的。

### 更准确的替代方案

如果你想坚持 strict chunk-boundary 语义，更合理的做法通常是：

1. 只记录 decision start
   - replay 只从显式 decision boundary 采样
2. 或者直接做 direct chunk replay
   - 一个 replay item 就是一条 chunk decision
3. 或者 step replay 里加 decision-start mask
   - sample 时只从合法 decision start 取窗口

这几种都比“让 `sample_stride` 跟着动态 `execute_horizon` 走”更清晰。

## 5. 当前需要做的选择

当前真正需要明确的是：

### 方案 1：保留 `sample_stride=1`

含义：

- 任意 step 都是潜在 decision point
- 必须每一步都重算 `next_base_action_chunk`
- 这是 step-wise receding-horizon training

### 方案 2：回到 chunk 边界采样

含义：

- 只在 chunk 起点做 decision
- replay 只从 chunk 边界采样
- 更适合 strict chunk residual control

## 6. 当前文档结论

如果后续目标是：

- 问题定义最干净
- 训练语义和执行语义尽量一致

那么更推荐：

- strict chunk-boundary decision
- 不使用 `sample_stride=1`
- 不要求为 chunk 内每一步构造新的 `base_action_chunk`

如果后续目标是：

- 最大化数据复用
- 允许任意 step 作为采样起点

那么当前“每一步重算 `next_base_action_chunk`”是自洽且必要的。
