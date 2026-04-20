# AgiBot Residual RL: Chunk Execution And Replan Notes

这份文档记录 `examples/agibot_real` 当前 residual RL 主线里，关于 `step` / `step_chunk`、`base_action_chunk`、`next_observation` 和 `mask` 语义的设计讨论。

相关实现：

- canonical 训练入口：[../scripts/run_residual_training.py](/home/hello/codebase/serl_torch/examples/agibot_real/scripts/run_residual_training.py)
- residual observation helper：[../../../serl_launcher/serl_launcher/residual/observation.py](/home/hello/codebase/serl_torch/serl_launcher/serl_launcher/residual/observation.py)
- policy input helper：[../env/policy_input.py](/home/hello/codebase/serl_torch/examples/agibot_real/env/policy_input.py)
- local robot env：[../env/task_env.py](/home/hello/codebase/serl_torch/examples/agibot_real/env/task_env.py)

## 1. `mask` 的语义

在 replay 里的 `mask` 可以直接理解成：

- `mask = 1`：目标里允许 bootstrap 到 `next_observation`
- `mask = 0`：目标在这里截断，不再看 `next_observation`

对应地，Q-learning / SAC 的目标大致是：

```python
target = reward + discount * mask * V(next_observation)
```

所以：

- `mask = 1` 会继续使用 `next_observation` 的价值
- `mask = 0` 会把当前 transition 当作边界

### AgiBot 当前选择

对 AgiBot 真实机器人，`truncated` 更适合被当成训练边界，而不是 benchmark 式的 “只是 time-limit”。

原因是当前 `truncated` 主要对应这些场景：

- 人工 reset
- 超时
- controller 中断
- 人为判定本回合结束

因此当前主线采用：

```python
done_flag = bool(done or truncated)
mask = 0.0 if done_flag else 1.0
```

也就是：

- `done=True`：截断
- `truncated=True`：也截断

这与旧的 `run_actor_residual.py` 语义一致。

## 2. 为什么 `next_observation` 里必须保存 `next_obs` 对应的 `base_action_chunk`

当前 residual observation 的定义不是只有原始观测，还包括：

- `robot_proprio`
- `base_action`
- `base_action_chunk`
- `alpha`
- 图像键

也就是说，训练里真正的状态更像：

```text
s_t = (raw_obs_t, base_chunk_t)
```

那么下一状态就必须是：

```text
s_{t+1} = (raw_obs_{t+1}, base_chunk_{t+1})
```

这里的 `base_chunk_{t+1}` 必须来自：

```text
base_policy(raw_obs_{t+1})
```

而不能简单把上一步的 chunk 向后平移：

```text
shift(base_chunk_t)
```

### 原因

因为当前 base policy 是闭环策略，不是严格的 open-loop 计划器。  
在真实机器人里，下面这些变化都会让：

```text
base_policy(next_obs) != shift(base_policy(obs))
```

- 执行动作后的真实偏差
- 视觉变化
- 接触与物体运动
- gripper 状态变化
- controller pause/reset/fail
- 人工干预

如果把 `next_observation.base_action_chunk` 近似成平移旧 chunk，那么训练里保存的就不是 actor 在线真实会看到的下一状态，而是一个“伪造的 next state”。

这会带来两个后果：

1. 训练-执行不一致
2. Bellman target 对应的状态空间被偷换

## 3. `next-state infer` 指的是什么

这里的 `next-state infer` 指的是：

1. 执行某一步动作后拿到 `next_obs`
2. 用 `next_obs` 再跑一次 base policy
3. 得到新的 `base_action_chunk`
4. 用它构造 `next_observation`

对应当前实现中的代码形态：

```python
next_base_policy_input = build_agibot_policy_input(next_obs, task_prompt)
next_base_actions, _ = policy_client.infer(next_base_policy_input)
next_residual_obs = build_chunk_residual_obs(
    obs=next_obs,
    base_actions=next_base_actions,
    ...
)
```

注意：

- 这里重新推理的是 **base policy**
- 不是 residual actor

它的目的主要是构造训练里的 `next_observation`。

## 4. `step` 与 `step_chunk` 的真正区别

### 模式 A：逐步 `step`

每一步都：

1. 从当前状态构造 residual observation
2. 拿 base chunk
3. 拿 residual chunk
4. 执行一步或执行 chunk 的第一步
5. 进入下一状态

如果是“严格的一步一重规划”，那意味着：

- base policy 每步刷新
- residual actor 也每步刷新
- 每次都只执行新 chunk 的第一步
- 旧 chunk 剩余部分直接丢弃

这是真正最闭环的语义，但停顿最频繁。

### 模式 B：整段 `step_chunk`

在 chunk 开头：

1. 跑 base policy
2. 跑 residual actor
3. 得到完整 `final_actions`
4. 一次性执行完整 chunk
5. chunk 结束后再刷新

它的特点是：

- 控制链路停顿最少
- 但 chunk 内使用的决策最 stale

### 当前 canonical AgiBot 主线

当前主线更接近 LIBERO：

- residual actor：按 chunk 刷新
- env：逐步执行 `env.step(action)`
- base policy：每步都重新 infer，用来构造下一时刻的 residual observation

这比“严格的一步一重规划”更便宜，也比“整段 chunk 完全不开环刷新”更稳。

## 5. 使用 `step_chunk()` 时，训练里还要不要处理中间步

要。

即使我们把执行路径改成：

```python
chunk_result = env.step_chunk(final_actions)
```

训练侧仍然要按中间步逐条构造 transition：

```text
s_t -> a_t -> s_{t+1}
s_{t+1} -> a_{t+1} -> s_{t+2}
...
```

也就是说：

- `step_chunk()` 只减少 env/controller 的调用次数
- 它不自动消除 replay 里对中间步 transition 的需求

## 6. 一个 chunk 执行后，是否还要对每个中间步重跑 base policy

如果目标是不改变当前 residual RL 的状态定义，那么答案通常是：要。

假设 chunk 长度是 `30`：

- residual actor 在 chunk 开头只采样 `1` 次 residual chunk
- env 可以一次 `step_chunk(final_actions)`
- 但为了构造训练里的 `next_observation`
- 仍然通常要对 30 个中间 `next_obs` 分别跑 base policy

所以：

- residual actor chunk 推理次数：可以是 `1`
- base policy 的中间步推理次数：通常仍接近 `30`

这就是为什么 `step_chunk()` 优化的主要是 env/controller 往返，而不一定是 base policy 推理次数。

## 7. 什么时候可以近似地用旧 chunk 平移

### 最稳的条件

最稳的工程做法是：

- 执行时可以暂时用旧 chunk 左移来加速
- 但最终写进 replay 的 `next_observation.base_action_chunk`
- 仍然回填为 `next_obs` 真正对应的 base policy 输出

这种做法不会改变训练语义，只是把推理时机向后挪。

### 直接把平移 chunk 写进训练数据的前提

只有当下面这个近似足够成立时，风险才可控：

```text
base_policy(next_obs) ≈ shift(base_policy(obs))
```

也就是 base policy 在真实 rollout 分布上，近似满足平移等变。

这通常要求：

- 相邻状态变化很小
- base policy 本身比较稳定
- 不处在接触/分叉/人为中断区域
- 复用窗口足够短

### 对 AgiBot 的工程建议

不建议直接整段 `30/50` 步都只靠平移旧 chunk。

更现实的方式是：

- `chunk_horizon = H`
- `execution_horizon = K`
- 每次只执行前 `K` 步，再刷新一次新的 base chunk / residual chunk

其中：

- `K = 1` 接近逐步闭环
- `K = H` 接近整段 chunk 开环

这类 `K-step refresh` 往往是更适合真实机器人的折中方案。

## 8. 当前结论

对当前 `examples/agibot_real` 主线，比较稳的判断是：

1. `truncated` 应当与 `done` 一样，作为 bootstrap 边界
2. `next_observation` 必须对应 `next_obs` 重新计算出的 base chunk
3. `step_chunk()` 可以优化执行路径，但不能自动省掉训练里对中间步 transition 的处理
4. 如果后续要做加速，优先考虑 `K-step refresh`，而不是整段 chunk 完全不开环刷新
