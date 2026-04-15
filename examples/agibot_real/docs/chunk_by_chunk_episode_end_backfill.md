# AgiBot Residual RL: Chunk-By-Chunk Execution With Episode-End Transition Backfill

这份文档记录一个面向 `examples/agibot_real` 的简化实现方案：

- actor 每次只在 chunk 开头推理一次
- 机器人连续执行完整 `action_chunk`
- episode 运行过程中不立即组装 replay transition
- episode 结束后，再统一回填整条 rollout 的 step-level transition

这份方案的目标不是追求最闭环的控制，而是优先解决当前执行链路里的明显停顿，让真实机器人先稳定、连续地跑起来。

相关实现入口：

- actor 主循环：[../scripts/run_residual_training.py](/Users/niejunnan.25/Documents/codebase/serl_torch/examples/agibot_real/scripts/run_residual_training.py)
- 真实机器人 env：[../env/task_env.py](/Users/niejunnan.25/Documents/codebase/serl_torch/examples/agibot_real/env/task_env.py)
- base policy adapter：[../env/base_policy.py](/Users/niejunnan.25/Documents/codebase/serl_torch/examples/agibot_real/env/base_policy.py)
- residual observation helper：[../residual_observation.py](/Users/niejunnan.25/Documents/codebase/serl_torch/examples/agibot_real/residual_observation.py)

## 1. 当前实现的问题

当前 canonical actor 主线的关键路径是：

1. 在当前 `obs_t` 上推一个完整 `base_action_chunk`
2. residual actor 也推一个完整 `residual_action_chunk`
3. 合成 `final_actions`
4. 进入 `for action in final_actions`
5. 每一步都调用一次 `env.step(action)`
6. 每一步执行后，再立刻用 `next_obs` 重跑一次 base policy，构造 `next_residual_obs`

这会带来两个问题：

### 1.1 执行层不是按 chunk 连续执行

虽然策略侧是按 chunk 生成动作，但 env 侧仍然按单步执行。  
这意味着机器人动作之间会反复被 actor 侧逻辑打断。

### 1.2 chunk 内每一步都插入推理开销

每一步执行后，actor 还要立刻做：

- `base_policy.infer(next_obs)`
- `build_chunk_residual_obs(next_obs, next_base_actions, ...)`

这段计算本身不改变当前 chunk 剩余动作的执行，只是为了构造 replay 里的下一状态。  
所以它放在动作热路径里，代价很高，收益很小。

## 2. 目标方案

目标方案是一个更干净的 `chunk-by-chunk` 执行模式：

1. 在 chunk 开头，基于当前 `obs_t` 推理一次
2. 得到完整 `final_actions_t`
3. 用 `env.step_chunk(final_actions_t)` 连续执行整段动作
4. chunk 执行完成后，只缓存原始 rollout 数据
5. 下一段 chunk 再基于最后一个 `obs` 继续推理
6. 整个 episode 结束后，再统一回填 step-level transition

这个方案的关键思想是：

- 决策单位是 chunk
- 执行单位也是 chunk
- 训练写入单位仍然是 step
- 但 step-level replay 组装从在线热路径移到 episode 尾部

## 3. 这套方案到底在回填什么

需要先澄清一点：

如果 episode 一共执行了 `T` 步动作，那么真实机器人并不是“只拿到了 chunk 起点的 obs”。  
实际上，执行过程中会拿到完整的原始观测序列：

```text
o_0, o_1, o_2, ..., o_T
```

其中：

- `o_0` 是 reset 后初始观测
- `o_{t+1}` 是执行动作 `a_t` 后的真实观测

真正缺的不是原始 `obs`，而是这些 `obs` 上对应的：

- `base_action_chunk_t = base_policy(o_t)`
- `residual_obs_t = build_chunk_residual_obs(o_t, base_action_chunk_t, ...)`

所以所谓“episode 结束后回填”，更准确地说是：

- episode 期间先缓存原始 rollout
- episode 结束后，再为每个 `o_t` 补出 `base_action_chunk_t`
- 然后统一构造 replay 需要的 `residual_obs_t` 和 `next_residual_obs_t`

## 4. 用 3 个 chunk、总共 45 步举例

假设：

- `chunk_horizon = 15`
- episode 一共执行了 3 个 chunk
- 总步数是 45

那么在线运行时，chunk 边界上的决策点大致是：

```text
o_0 -> 推 chunk_0 -> 执行 a_0 ... a_14
o_15 -> 推 chunk_1 -> 执行 a_15 ... a_29
o_30 -> 推 chunk_2 -> 执行 a_30 ... a_44
```

此时在线已知的内容是：

- `o_0, o_15, o_30` 的 base policy 输出一定有
- `o_1 ... o_45` 的原始观测其实也都有
- 只是 `o_1 ... o_14, o_16 ... o_29, o_31 ... o_44` 的 base policy 输出没有在热路径里算出来

所以 episode 结束后需要做的，就是把这些中间状态的 base policy 输出补齐。

## 5. 运行时的在线逻辑

在线运行时建议只做下面这些事情。

### 5.1 reset

1. `obs = env.reset(...)`
2. 初始化一个 episode buffer
3. 先记录 `o_0`

### 5.2 chunk 开头推理

在当前 `obs_t` 上：

1. `base_actions_t, base_info_t = base_policy.infer(obs_t, prompt)`
2. `residual_obs_t = build_chunk_residual_obs(obs_t, base_actions_t, ...)`
3. `residual_action_t = agent.sample_action(residual_obs_t, ...)`
4. `final_actions_t = compose_chunk(base_actions_t, residual_action_t)`

### 5.3 chunk 执行

直接调用：

```python
chunk_result = env.step_chunk(final_actions_t)
```

这一步的目标是让 controller 按 `hz` 连续执行这一段动作，不再在 chunk 内插入 actor 侧推理。

### 5.4 在线只缓存原始 rollout

`chunk_result` 返回后，把这段执行结果记进 episode buffer，包括：

- chunk 起点步号
- chunk 起点的 `obs_t`
- chunk 起点的 `base_actions_t`
- chunk 起点的 `residual_obs_t`
- 计划执行的 `final_actions_t`
- 实际执行步数 `num_steps`
- 真实执行后的 `observations`
- 每步 `rewards`
- 每步 `dones`
- 最终 `truncated`
- 每步 `infos`

然后：

- 如果 chunk 中途终止，就结束 episode
- 否则把 `obs` 更新为 chunk 最后一个观测，进入下一段 chunk

### 5.5 在线阶段不做的事情

在线热路径里不再做：

- 对 chunk 中间每个 `next_obs` 重跑 `base_policy.infer`
- 逐步构造 replay transition
- 每步把 transition 插入 `data_store`

## 6. Episode Buffer 需要缓存什么

建议显式做一个 episode-level buffer。最小字段可以是：

```python
episode_buffer = {
    "initial_obs": o_0,
    "steps": [
        {
            "action": a_t,
            "next_obs": o_{t+1},
            "reward": r_t,
            "done": done_t,
            "truncated": truncated_t,
            "info": info_t,
        },
        ...
    ],
    "chunk_records": [
        {
            "start_step": 0,
            "start_obs": o_0,
            "base_actions": base_chunk_0,
            "residual_obs": residual_obs_0,
            "planned_actions": final_chunk_0,
            "executed_steps": 15,
        },
        ...
    ],
}
```

如果只从“episode 结束后回填 transition”这个目标出发，其实最关键的是：

- `o_0`
- 每一步真实执行的 `a_t`
- 每一步执行后的 `o_{t+1}`
- 每一步的 `reward / done / truncated / info`

`chunk_records` 不是绝对必须，但建议保留，原因是：

- 方便调试“第几个 chunk 发生了终止”
- 方便核对计划动作和实际执行前缀
- 方便后续统计 chunk 级吞吐和停顿

## 7. Episode 结束后的回填算法

episode 结束后，再做 replay 组装。

### 7.1 先恢复完整的状态序列

从 episode buffer 中恢复：

```text
o_0, o_1, o_2, ..., o_T
```

其中：

- `o_0 = initial_obs`
- `o_{t+1} = steps[t]["next_obs"]`

### 7.2 对每个状态补 base policy 输出

对于每个 `o_t`，都计算：

```python
base_chunk_t, _ = base_policy.infer(o_t, prompt=task_prompt)
residual_obs_t = build_chunk_residual_obs(
    obs=o_t,
    base_actions=base_chunk_t,
    image_keys=image_keys,
    residual_alpha=residual_alpha,
)
```

如果继续沿用当前 residual RL 的状态定义，那么原则上需要给所有 `o_t` 都补这个结果。

### 7.3 再按 step 构造 replay transition

对于每个 step `t in [0, T-1]`，构造：

```python
transition_t = {
    "observations": residual_obs_t,
    "actions": a_t,
    "next_observations": residual_obs_{t+1},
    "rewards": r_t,
    "dones": bool(done_t or truncated_t),
    "masks": 0.0 if (done_t or truncated_t) else 1.0,
}
```

然后统一写入 `data_store`。

## 8. `step_chunk()` 的终止语义怎么接

当前 `env.step_chunk()` 返回的是：

- `observations`
- `rewards`
- `dones`
- `done`
- `truncated`
- `infos`
- `num_steps`

对 episode 回填方案来说，这已经够用。

处理原则是：

- 如果一个 chunk 全部执行完，那么这一段有 `chunk_horizon` 条 step 记录
- 如果中途被 `success/fail/reset/timeout` 终止，那么只保留实际执行的前缀
- 未执行的剩余动作直接丢弃，不进入 replay

step-level 的 terminal 判断可按下面方式恢复：

- 对非最后一步：`done=False, truncated=False`
- 对最后一步：用 `chunk_result` 里最后一步的 `done/truncated/info`

如果后续想把这层语义写得更稳，可以考虑让 `step_chunk()` 显式返回每一步的 `truncateds`，但第一版不一定需要。

## 9. 这套方案的优点

### 9.1 动作连续性最好

在线热路径只做：

- chunk 开头推理一次
- 连续执行一整段 chunk

这比当前“每步执行后再推理一次”的方式更接近真实想要的连续控制。

### 9.2 actor 逻辑更干净

当前 actor 同时做了三件事：

- 生成 chunk
- 单步执行
- 单步回填 replay

拆成“在线执行”和“episode 尾部补账”以后，职责会清楚很多。

### 9.3 方便先跑通系统

这套方案不要求你先实现：

- per-step 闭环重规划
- 中间步 batch infer
- 更复杂的异步流水线

它更适合当前阶段先把真实机器人执行链路理顺。

## 10. 这套方案的代价

### 10.1 chunk 内是开环的

如果 `chunk_horizon=15`，在 `20Hz` 下就是大约 `0.75s` 的开环执行。  
中途状态漂移时，不能立刻修正。

### 10.2 episode 结束时会有一次集中补账延迟

由于当前 [AgiBotBasePolicy.infer()](/Users/niejunnan.25/Documents/codebase/serl_torch/examples/agibot_real/env/base_policy.py#L81) 是单帧接口，episode 结束后需要按步补推理。  
episode 越长，这段尾部停顿越明显。

### 10.3 learner 看到数据会更晚

当前如果每步就写 replay，learner 可以更早拿到数据。  
改成 episode 结束后统一写入后，数据可见性会按 episode 粒度延后。

### 10.4 需要处理 episode 中途异常

如果在 episode 结束前进程崩了，而原始 rollout 只放在内存里，这一整段数据可能丢失。

## 11. 我对第一版实现的建议

### 11.1 先把 `chunk_horizon` 控制在保守范围

建议起步：

- `chunk_horizon = 10` 或 `15`

不建议第一版就用特别长的 chunk。

### 11.2 先做“episode 结束后统一回填”

这个版本最简单，也最符合当前目标。  
不要一开始就引入：

- chunk 结束后边执行边补账
- background thread 异步补推理
- actor/learner 更复杂的流水线

### 11.3 建议先把原始 rollout 落盘

第一版最好在 episode 结束回填前，先把原始 episode buffer 以 `jsonl`、`npz` 或等价格式落盘。

原因很现实：

- 便于离线排查
- 便于重跑 backfill
- 防止回填过程中出错导致整段 episode 丢失

### 11.4 不改变当前终止语义

这一版不建议顺手改动 controller 终止逻辑。  
继续沿用当前定义即可：

- `success -> reward=1, done=True`
- `fail -> reward=0, done=True`
- `reset -> reward=0, truncated=True`
- `timeout -> reward=0, truncated=True`

## 12. 对代码改动边界的建议

### 12.1 主要改动点在 actor

主要重构点会在：

- [run_residual_training.py](/Users/niejunnan.25/Documents/codebase/serl_torch/examples/agibot_real/scripts/run_residual_training.py)

要做的核心改动是：

- 去掉 `for action in final_actions: env.step(action)`
- 改成一次 `env.step_chunk(final_actions)`
- 新增 episode buffer
- episode 结束后统一 backfill transition

### 12.2 env 基本可以少改

当前 [task_env.py](/Users/niejunnan.25/Documents/codebase/serl_torch/examples/agibot_real/env/task_env.py) 已经有 `step_chunk()`，第一版不一定需要大改 env。

真正需要确认的是：

- `step_chunk()` 返回的 `observations/rewards/dones/infos` 是否足够回填
- 中途终止时是否只返回已执行前缀

从当前实现看，这条路基本已经具备。

### 12.3 `base_policy` 和 `residual_observation` 暂时不用动

因为这份方案没有改变状态定义，只是把：

- `base_policy.infer`
- `build_chunk_residual_obs`

从“每步执行后立刻调用”改成“episode 尾部统一调用”。

## 13. 一个推荐的第一版伪代码

```python
obs = env.reset(...)
episode_buffer = new_episode_buffer(obs)

while not episode_done:
    base_actions, _ = base_policy.infer(obs, prompt=task_prompt)
    residual_obs = build_chunk_residual_obs(
        obs=obs,
        base_actions=base_actions,
        image_keys=image_keys,
        residual_alpha=residual_alpha,
    )
    residual_actions = agent.sample_action(residual_obs, deterministic=False)
    final_actions = residual_action_spec.compose_chunk(
        base_action_chunk=base_actions,
        residual_action=residual_actions,
    )

    chunk_result = env.step_chunk(final_actions)
    append_chunk_to_episode_buffer(
        episode_buffer,
        start_obs=obs,
        base_actions=base_actions,
        residual_obs=residual_obs,
        final_actions=final_actions,
        chunk_result=chunk_result,
    )

    obs = chunk_result["obs"]
    episode_done = bool(chunk_result["done"] or chunk_result["truncated"])

raw_rollout_path = dump_episode_buffer(episode_buffer)

transitions = backfill_episode_transitions(
    episode_buffer=episode_buffer,
    base_policy=base_policy,
    build_residual_obs_fn=build_chunk_residual_obs,
    task_prompt=task_prompt,
)
for transition in transitions:
    data_store.insert(transition)
```

## 14. 结论

这份方案本质上是在做一件事：

把 replay bookkeeping 从机器人动作热路径里拿出去。

如果当前阶段的优先级是：

- 让真实机器人连续执行更顺
- 简化 actor 热路径
- 先把 chunk 执行语义做干净

那么 `step_chunk + episode 结束后回填 transition` 是一个合理的第一版方案。

它不是最终最优方案，但它很适合作为当前 `agibot_real` 的下一步工程化落点。
