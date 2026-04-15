# AgiBot Residual RL Dataflow And Transition Refactor Plan

本文总结当前 `examples/agibot_real/scripts/run_residual_training.py` 主线的数据流、transition 生成链路，以及针对后续重构的建议。重点覆盖：

- 当前 actor 到 learner 的数据流向
- 为什么 `chunk_replay.py` 这个名字不够准确
- 当前“每 15 步回填一次 transition”的真实语义
- 为什么可以引入 `TransitionAssembler`
- actor 为什么当前仍保留 replay 提交控制权
- 推荐的数据结构、模块边界和伪代码

## 1. 当前主线的真实训练语义

当前主线不是“actor 直接产出 chunk transition，然后 learner 直接训练 chunk transition”。

当前真实语义是：

1. actor 按 `chunk_horizon` 执行动作块
2. chunk 执行完成后，actor 再把这一段真实执行结果回填成 step transition
3. step transition 通过 agentlace 发给 learner
4. learner 侧把 step transition 存进 step-window replay
5. learner 采样时，再把连续的 step 拼成一个 chunk window 样本训练

也就是说：

- 执行单位是 `chunk`
- 存储单位是 `step`
- 训练采样单位是 `chunk window`

当前关键文件：

- [run_residual_training.py](/Users/niejunnan.25/Documents/codebase/serl_torch/examples/agibot_real/scripts/run_residual_training.py)
- [chunk_window_replay.py](/Users/niejunnan.25/Documents/codebase/serl_torch/serl_launcher/serl_launcher/residual/chunk_window_replay.py)
- [step_window_replay_buffer.py](/Users/niejunnan.25/Documents/codebase/serl_torch/serl_launcher/serl_launcher/data/step_window_replay_buffer.py)
- [data_store.py](/Users/niejunnan.25/Documents/codebase/serl_torch/serl_launcher/serl_launcher/data/data_store.py)

## 2. 当前 actor 侧的数据流

### 2.1 actor 初始化阶段

actor 会创建：

- env 和 base policy
- residual agent
- 一个 `QueuedDataStore`
- 一个挂在 `QueuedDataStore` 上的 `TrainerClient`

对应位置：

- [run_residual_training.py](/Users/niejunnan.25/Documents/codebase/serl_torch/examples/agibot_real/scripts/run_residual_training.py#L82)
- [run_residual_training.py](/Users/niejunnan.25/Documents/codebase/serl_torch/examples/agibot_real/scripts/run_residual_training.py#L107)

这里要特别区分两条不同的 actor -> learner 通道：

- transition 数据通道：
  `QueuedDataStore` -> `TrainerClient` -> learner replay
- episode 统计通道：
  `client.request("send-stats", episode_stats)` -> learner `stats_callback`

另外还有一条 learner -> actor 的网络广播通道：

- `server.publish_network(...)` -> `client.recv_network_callback(...)`

### 2.2 actor 每轮 chunk 的执行流程

当前一个 chunk 的执行流程是：

1. 取当前 `obs`
2. 跑一次 `base_policy.infer(obs, prompt=task_prompt)`
3. 组 `residual_obs`
4. residual agent 采样 residual action chunk
5. 合成 `final_actions`
6. 调 `env.step_chunk(action_chunk)` 连续执行这一段
7. 收到：
   - `post-step observations`
   - `rewards`
   - `dones`
   - `infos`
   - `chunk done/truncated`
8. 再对每个 `post-step obs` 逐个做 `base_policy.infer(...)`
9. 把这段 chunk 回填成 step transition
10. 再逐条 `data_store.insert(transition)`

关键代码：

- chunk 起点决策：[run_residual_training.py](/Users/niejunnan.25/Documents/codebase/serl_torch/examples/agibot_real/scripts/run_residual_training.py#L210)
- chunk 执行：[run_residual_training.py](/Users/niejunnan.25/Documents/codebase/serl_torch/examples/agibot_real/scripts/run_residual_training.py#L240)
- backfill：[run_residual_training.py](/Users/niejunnan.25/Documents/codebase/serl_torch/examples/agibot_real/scripts/run_residual_training.py#L154)
- step transition 组装：[run_residual_training.py](/Users/niejunnan.25/Documents/codebase/serl_torch/examples/agibot_real/scripts/run_residual_training.py#L303)

## 3. 当前“每 15 步回填一次 transition”到底是什么意思

如果当前配置是：

- `chunk_horizon = 15`

那么当前逻辑不是：

- 每执行一步立刻构造并写一条 transition

而是：

1. chunk 起点拿到一个 `obs_0`
2. 推出 `15` 个动作
3. 连续执行 `15` 步
4. 得到真实的 `obs_1 ... obs_15`
5. 对 `obs_1 ... obs_15` 各自再跑一次 `base_policy.infer(...)`
6. 从而构造：
   - `transition_0 = (residual_obs_0, action_0, residual_obs_1)`
   - `transition_1 = (residual_obs_1, action_1, residual_obs_2)`
   - ...
   - `transition_14 = (residual_obs_14, action_14, residual_obs_15)`
7. 再把这 15 条 step transition 插入 replay

所以“每 15 步回填一次”本质上是：

- 先执行
- 后记账
- 记账按 step
- 执行按 chunk

这也是当前真机吞吐的关键瓶颈之一，因为每个 chunk 后还要做 `15` 次串行 `base_policy.infer(...)`。

## 4. 当前 learner 侧的数据流

### 4.1 learner 接收 transition

learner 会创建 replay：

- [create_chunk_replay_buffer()](/Users/niejunnan.25/Documents/codebase/serl_torch/serl_launcher/serl_launcher/residual/chunk_window_replay.py#L17)

但这个函数名带有误导性。它实际创建的是：

- [MemoryEfficientStepWindowReplayBufferDataStore](/Users/niejunnan.25/Documents/codebase/serl_torch/serl_launcher/serl_launcher/data/data_store.py#L194)

然后通过：

- [register_data_store("actor_env", replay_buffer)](/Users/niejunnan.25/Documents/codebase/serl_torch/examples/agibot_real/scripts/run_residual_training.py#L561)

把 learner 本地 replay 注册成 actor 这条 agentlace 数据流的接收端。

actor 侧对应的是：

- `TrainerClient("actor_env", ...)`

所以当前 transition 数据流是：

```text
actor data_store.insert(step_transition)
  -> QueuedDataStore
  -> TrainerClient("actor_env")
  -> TrainerServer.register_data_store("actor_env", replay_buffer)
  -> learner replay_buffer.insert(step_transition)
```

### 4.2 replay 内部存什么

replay 里存的是 step transition，不是 macro chunk。

`StepWindowReplayBuffer.insert()` 每次存一条 step 记录，包括：

- `observations`
- `next_observations`
- `actions`
- `rewards`
- `masks`
- `dones`
- `episode_id`
- `episode_step`

对应代码：

- [step_window_replay_buffer.py](/Users/niejunnan.25/Documents/codebase/serl_torch/serl_launcher/serl_launcher/data/step_window_replay_buffer.py#L181)

### 4.3 learner 怎么从 step replay 变成 chunk 训练样本

当 learner `sample()` 时，它不是取单步，而是：

1. 找一个可作为 window 起点的 `start_step_id`
2. 从该位置往后收集连续 step
3. 如果碰到 `dones=True` 或凑满 `window_size`，就截断
4. 构造 chunk-like 样本：
   - `observations = obs_t`
   - `actions = [a_t, a_{t+1}, ...]`
   - `action_mask`
   - `next_observations = next_obs_last`
   - `rewards = discounted_reward_sum`
   - `masks = gamma^(k-1) * last_mask`
   - `dones = boundary`

对应代码：

- [step_window_replay_buffer.py](/Users/niejunnan.25/Documents/codebase/serl_torch/serl_launcher/serl_launcher/data/step_window_replay_buffer.py#L201)

也就是说：

- replay 原子存储单位是 step
- learner 看到的训练样本是 window

## 5. 为什么 `chunk_replay.py` 这个名字不够准

当前 [chunk_window_replay.py](/Users/niejunnan.25/Documents/codebase/serl_torch/serl_launcher/serl_launcher/residual/chunk_window_replay.py) 做的事情不是：

- “actor 直接产出一个 chunk transition，再存进 chunk replay”

而是：

- “step stream 被 learner 侧重新拼成 chunk window”

原来的 `chunk_replay.py` 这个名字会让人误以为：

- 每条 replay 样本就是一个 actor 直接发来的 chunk

但主线不是这样。

### 推荐命名

更准确的命名建议：

- `step_window_replay.py`
- `chunk_window_replay.py`
- `residual_window_replay.py`

如果只选一个，推荐：

- **`chunk_window_replay.py`**

因为它最准确地表达了当前语义：

- 底层是 step stream
- 训练时采样的是 chunk window

## 6. 当前主线里为什么需要 `pending_last_transition`

当前主线有一个真机 / controller 特有边界：

- 下一段 chunk 的第一个动作还没执行
- 但 controller 已经收到了 `success/fail/reset/timeout/infra_abort`

这时 env 会返回 synthetic terminal：

- `controller_action_executed=False`

这意味着：

- 不能凭空生成一条新的 step transition
- 真正该被标成 terminal 的，是“上一条真实执行过的 step”

所以当前主线引入了：

- `pending_last_transition`

逻辑是：

- 非终止 chunk 的最后一条 transition 不立即插入 replay
- 先挂起
- 下一轮如果至少成功执行了 1 步，说明上一条确实是非 terminal，再 flush
- 下一轮如果在第一个动作执行前就 terminal，则把上一条 pending transition 改写成 terminal，再插入 replay

对应代码：

- [_flush_pending_last_transition()](/Users/niejunnan.25/Documents/codebase/serl_torch/examples/agibot_real/scripts/run_residual_training.py#L187)
- [_finalize_pending_last_transition()](/Users/niejunnan.25/Documents/codebase/serl_torch/examples/agibot_real/scripts/run_residual_training.py#L194)

## 7. 为什么可以引入 `TransitionAssembler`

当前 [run_residual_training.py](/Users/niejunnan.25/Documents/codebase/serl_torch/examples/agibot_real/scripts/run_residual_training.py) 的 actor 主循环同时承担了 4 类职责：

- 真机执行
- post-step obs backfill
- terminal / pending 边界修补
- replay transition 组装与写入

这会让主循环很难继续维护。

实际上应该把这些事情拆开：

- actor 负责 rollout control plane
- transition 组装负责 data plane

这就是引入 `TransitionAssembler` 的原因。

### `TransitionAssembler` 应负责的事情

建议职责收敛到这 6 类：

1. 对每个 `post-step obs` 回填 `base_action_chunk`
2. 构造 `next_residual_obs`
3. 按 step 产出 transition
4. 维护 `pending_last_transition`
5. 合并 synthetic terminal 到上一条真实 step
6. 可选地重算 / 补充 dense reward

它不应该负责：

- 真机控制
- learner 通信
- wandb
- checkpoint

## 8. 推荐的新数据流设计

推荐拆成 5 层。

### 8.1 ActorExecution

负责：

- 取 `obs`
- 跑 `base_policy + residual_policy`
- 得到 `action_chunk`
- 调 `env.step_chunk(action_chunk)`
- 产出一个 `RawChunkRecord`

### 8.2 TransitionAssembler

负责：

- 接收 `RawChunkRecord`
- backfill `next_residual_obs`
- 构造 step transitions
- 修补 terminal boundary
- 维护 `pending_last_transition`
- 可选地计算 dense reward

### 8.3 ReplaySink

负责：

- `insert_all(transitions)`

### 8.4 StatsReporter

负责：

- episode stats
- rollout payload
- learner `send-stats`

### 8.5 Learner

保持不变：

- 继续从 chunk window replay 采样训练

整体数据流：

```text
obs
 -> ActorExecution
 -> RawChunkRecord
 -> TransitionAssembler
 -> step transitions
 -> ReplaySink
 -> learner replay
 -> chunk window sample
 -> learner update
 -> network publish
 -> actor recv callback
```

## 9. 推荐的数据结构

### 9.1 `RawChunkRecord`

```python
@dataclass
class RawChunkRecord:
    episode_id: int
    episode_step_start: int

    task_prompt: str

    obs_before_chunk: dict[str, Any]
    residual_obs_before_chunk: dict[str, np.ndarray]

    action_chunk: np.ndarray
    executed_steps: int

    post_step_observations: list[dict[str, Any]]
    rewards: list[float]
    dones: list[bool]
    infos: list[dict[str, Any]]

    final_obs: dict[str, Any]
    chunk_done: bool
    chunk_truncated: bool
    chunk_reward_sum: float
    chunk_info: dict[str, Any]
```

这个结构表达的是：

- 一个 chunk 被怎么计划
- 真正执行了多少步
- 执行后的原始观测、奖励、done、info
- 这一段最终是继续还是结束

### 9.2 `AssemblerState`

```python
@dataclass
class AssemblerState:
    pending_last_transition: dict[str, Any] | None = None
    prefetched_base_actions: np.ndarray | None = None
    prefetched_residual_obs: dict[str, np.ndarray] | None = None
```

这个结构显式承载当前主线已经存在的跨 chunk 状态：

- `pending_last_transition`
- `prefetched`

### 9.3 `AssemblyResult`

```python
@dataclass
class AssemblyResult:
    transitions_to_insert: list[dict[str, Any]]
    next_state: AssemblerState

    next_obs_for_actor: dict[str, Any]
    episode_done: bool

    env_steps_delta: int
    episode_steps_delta: int
    episode_return_delta: float
    episode_success: bool

    last_info: dict[str, Any]
```

这个输出让 actor 主循环只需要：

- insert transitions
- 更新计数器
- 决定 `break / continue`

## 10. `RewardBuilder` 扩展点

如果后续要做稠密奖励，不建议把 reward 逻辑直接写死在 actor 主循环里。

更稳的做法是在 `TransitionAssembler` 内部留一个 reward hook：

```python
class RewardBuilder(Protocol):
    def compute(
        self,
        *,
        obs_t: dict[str, Any],
        action_t: np.ndarray,
        obs_tp1: dict[str, Any],
        env_reward: float,
        env_done: bool,
        env_truncated: bool,
        info_t: dict[str, Any],
    ) -> float: ...
```

这样可以有：

- `PassThroughRewardBuilder`
- `DenseRewardBuilder`
- `SparseOnlyRewardBuilder`

要求是：

- dense reward 尽量只依赖 `(obs_t, action_t, obs_tp1, info_t)`
- 不依赖未来 chunk 的信息

## 11. 推荐的 `TransitionAssembler` 伪代码

```python
class TransitionAssembler:
    def __init__(
        self,
        *,
        base_policy,
        image_keys: tuple[str, ...],
        residual_alpha: float,
        reward_builder=None,
    ):
        self.base_policy = base_policy
        self.image_keys = image_keys
        self.residual_alpha = residual_alpha
        self.reward_builder = reward_builder or PassThroughRewardBuilder()

    def _build_next_residual_obs_batch(
        self,
        *,
        post_step_observations: list[dict[str, Any]],
        task_prompt: str,
    ) -> tuple[list[np.ndarray], list[dict[str, np.ndarray]]]:
        base_chunks = []
        residual_obs_batch = []

        for obs in post_step_observations:
            base_actions, _ = self.base_policy.infer(obs, prompt=task_prompt)
            residual_obs = build_chunk_residual_obs(
                obs=obs,
                base_actions=base_actions,
                image_keys=self.image_keys,
                residual_alpha=self.residual_alpha,
            )
            base_chunks.append(base_actions)
            residual_obs_batch.append(residual_obs)

        return base_chunks, residual_obs_batch

    def process_chunk(
        self,
        *,
        raw: RawChunkRecord,
        state: AssemblerState,
    ) -> AssemblyResult:
        transitions_to_insert = []
        episode_done = False
        episode_success = False

        if raw.executed_steps <= 0:
            if not (raw.chunk_done or raw.chunk_truncated):
                raise RuntimeError(
                    "No action executed but chunk was not terminal/truncated."
                )

            if state.pending_last_transition is not None:
                state.pending_last_transition["rewards"] = (
                    float(state.pending_last_transition["rewards"])
                    + float(raw.chunk_reward_sum)
                )
                state.pending_last_transition["masks"] = 0.0
                state.pending_last_transition["dones"] = bool(
                    raw.chunk_done or raw.chunk_truncated
                )
                transitions_to_insert.append(state.pending_last_transition)
                state.pending_last_transition = None

            return AssemblyResult(
                transitions_to_insert=transitions_to_insert,
                next_state=AssemblerState(),
                next_obs_for_actor=raw.final_obs,
                episode_done=True,
                env_steps_delta=0,
                episode_steps_delta=0,
                episode_return_delta=float(raw.chunk_reward_sum),
                episode_success=bool(raw.chunk_info.get("success", False)),
                last_info=dict(raw.chunk_info),
            )

        executed_actions = raw.action_chunk[: raw.executed_steps]
        executed_post_obs = raw.post_step_observations[: raw.executed_steps]
        executed_rewards = raw.rewards[: raw.executed_steps]
        executed_dones = raw.dones[: raw.executed_steps]
        executed_infos = raw.infos[: raw.executed_steps]

        backfilled_base_chunks, backfilled_residual_obs = (
            self._build_next_residual_obs_batch(
                post_step_observations=executed_post_obs,
                task_prompt=raw.task_prompt,
            )
        )

        current_residual_obs = raw.residual_obs_before_chunk
        chunk_transitions = []

        for i in range(raw.executed_steps):
            step_done = bool(executed_dones[i])
            step_truncated = bool(
                i == raw.executed_steps - 1 and raw.chunk_truncated
            )
            boundary = bool(step_done or step_truncated)

            obs_t = current_residual_obs
            obs_tp1 = backfilled_residual_obs[i]
            action_t = np.asarray(executed_actions[i], dtype=np.float32).reshape(-1)
            info_t = dict(executed_infos[i])

            reward_t = self.reward_builder.compute(
                obs_t=obs_t,
                action_t=action_t,
                obs_tp1=obs_tp1,
                env_reward=float(executed_rewards[i]),
                env_done=step_done,
                env_truncated=step_truncated,
                info_t=info_t,
            )

            transition = {
                "episode_id": int(raw.episode_id),
                "episode_step": int(raw.episode_step_start + i),
                "observations": obs_t,
                "actions": action_t,
                "next_observations": obs_tp1,
                "rewards": float(reward_t),
                "masks": float(0.0 if boundary else 1.0),
                "dones": bool(boundary),
            }
            chunk_transitions.append(transition)
            current_residual_obs = obs_tp1
            episode_success = episode_success or bool(info_t.get("success", False))

        if state.pending_last_transition is not None:
            transitions_to_insert.append(state.pending_last_transition)
            state.pending_last_transition = None

        if raw.chunk_done or raw.chunk_truncated:
            transitions_to_insert.extend(chunk_transitions)
            next_state = AssemblerState()
            episode_done = True
        else:
            transitions_to_insert.extend(chunk_transitions[:-1])
            next_state = AssemblerState(
                pending_last_transition=chunk_transitions[-1],
                prefetched_base_actions=backfilled_base_chunks[-1],
                prefetched_residual_obs=backfilled_residual_obs[-1],
            )

        return AssemblyResult(
            transitions_to_insert=transitions_to_insert,
            next_state=next_state,
            next_obs_for_actor=raw.final_obs,
            episode_done=episode_done,
            env_steps_delta=int(raw.executed_steps),
            episode_steps_delta=int(raw.executed_steps),
            episode_return_delta=float(sum(executed_rewards)),
            episode_success=bool(episode_success),
            last_info=dict(raw.chunk_info if episode_done else executed_infos[-1]),
        )
```

## 12. actor 为什么当前仍然保留 replay 提交控制权

一个容易误解的点是：

- 引入 `TransitionAssembler` 以后，是否应该让 actor 完全不再碰 transition insert

答案是：

- **长期可以**
- **当前最小重构版本不建议一步到位**

### 原因

当前 actor 仍然是 rollout 主控者，它必须知道：

- 什么时候 `env.step_chunk(...)`
- 什么时候 episode 结束
- 什么时候 `env.reset()`
- 当前 `env_steps / episode_steps / episode_return / episode_success`
- 什么时候向 learner 发 episode stats

这些属于 rollout control plane。

所以当前最稳的边界是：

- actor 不再负责“理解 transition 细节”
- 但 actor 仍然保留 replay 提交的最终控制权

也就是：

- assembler 决定“应该写哪些 transition”
- actor 决定“现在正式提交这些 transition”

### 最小重构后的 actor 主循环伪代码

```python
state = AssemblerState()
obs = env.reset()

while training:
    if state.prefetched_base_actions is None:
        base_actions, _ = base_policy.infer(obs, prompt=task_prompt)
        residual_obs = build_chunk_residual_obs(...)
    else:
        base_actions = state.prefetched_base_actions
        residual_obs = state.prefetched_residual_obs

    residual_actions = agent.sample_action(residual_obs, deterministic=False)
    action_chunk = residual_action_spec.compose_chunk(
        base_action_chunk=base_actions,
        residual_action=residual_actions,
    )

    chunk_result = env.step_chunk(action_chunk)

    raw = RawChunkRecord(...)
    assembled = assembler.process_chunk(raw=raw, state=state)

    for transition in assembled.transitions_to_insert:
        data_store.insert(transition)

    state = assembled.next_state
    obs = assembled.next_obs_for_actor

    env_steps += assembled.env_steps_delta
    episode_steps += assembled.episode_steps_delta
    episode_return += assembled.episode_return_delta
    episode_success = episode_success or assembled.episode_success
    last_info = assembled.last_info

    if assembled.episode_done:
        break
```

这一步的好处是：

- 先把职责理顺
- 不同时引入异步复杂度
- 不需要改 agentlace 主干

## 13. 后续可以再走的两步

### 13.1 第二阶段：加入 `ReplaySink`

当前 actor 还在显式写：

- `for transition in ...: data_store.insert(...)`

后续可以再封一层：

- `ReplaySink.insert_all(transitions)`

这样 actor 主循环会更干净。

### 13.2 第三阶段：完全异步化

如果后续目标是进一步提速，可以再把：

- `TransitionAssembler`
- `ReplaySink`

一起移到后台 worker。

那时主线程只负责：

- 推动作
- 执行真机

结构变成：

```text
actor thread
  -> RawChunkRecord queue
  -> transition worker
  -> replay sink
  -> learner replay
```

但这一步会引入新的复杂度：

- chunk 执行顺序与 replay 插入顺序一致性
- `pending_last_transition` 的状态所有权
- actor 什么时候知道 replay 已成功提交
- 崩溃时 raw chunk 是否丢失

所以推荐顺序是：

1. 先纯重构，不改行为
2. 再加 `ReplaySink`
3. 最后再考虑异步 worker

## 14. 完全异步 `TransitionWorker` 方案

上面把完全异步化当成第三阶段，不代表这个方向不好。相反，如果目标是：

- 进一步提升真机吞吐
- 把 chunk 执行热路径和 backfill 热路径彻底拆开
- 为后续 dense reward / batch infer / data QA 留好扩展点

那么 **`TransitionWorker` 方案是合理且值得做的**。

需要强调的是：

- 它不是“概念不对”
- 它只是比“actor 仍保留提交控制权”的方案更复杂

如果当前目标已经从“先理顺逻辑”转向“进一步优化执行效率”，那这个方案完全可以作为正式目标架构。

### 14.1 推荐结构

推荐拆成两个线程或两个独立执行单元：

- `ActorExecutionThread`
- `TransitionWorker`

其中：

- actor 只负责 rollout control plane
- worker 负责 data plane

数据流如下：

```text
ActorExecutionThread
  -> RawChunkRecord queue
  -> TransitionWorker
  -> TransitionAssembler
  -> ReplaySink
  -> learner replay
```

也可以再展开成：

```text
obs
 -> ActorExecutionThread
 -> RawChunkRecord
 -> raw_chunk_queue
 -> TransitionWorker
 -> TransitionAssembler
 -> transitions_to_insert
 -> ReplaySink(data_store.insert)
 -> learner replay
```

### 14.2 这种方案的主要收益

#### A. 真机执行热路径更干净

当前最贵的后处理是：

- 对每个 `post-step obs` 做 `base_policy.infer`
- 构造 `next_residual_obs`
- 修 terminal boundary
- 逐条写 replay

如果这些还都在 actor 主线程里做，那么真机执行和数据加工是串行的。

异步化以后，actor 主线程只做：

- 推动作
- 执行 chunk
- 产出 raw chunk

这会明显降低 actor 线程上的 CPU / RPC / Python 组装负担。

#### B. `TransitionAssembler` 可以独立演化

一旦变成 worker 内部模块，你可以更自由地加：

- dense reward
- reward QA
- info 清洗
- terminal 修补
- batch infer
- raw rollout 落盘

而不污染 actor 主循环。

#### C. 更容易做 batch infer

当前 backfill 的核心代价是：

- 一个 chunk 后面串行做 `K` 次 `base_policy.infer`

如果这些逻辑在 worker 里，后续很自然就可以升级成：

- 对 `post-step obs` 做 batch infer

这条路比继续把所有逻辑塞在 actor 主线程里要顺得多。

#### D. 更容易加失败恢复

如果 worker 之前先把 `RawChunkRecord` 做持久化或轻量日志，那么即使后续 backfill 失败，也不会丢掉已经真实执行过的 chunk。

这对真机非常有价值。

### 14.3 这种方案新增的复杂度

它的复杂度也是真实存在的，主要在 5 个地方。

#### A. 顺序一致性

必须保证：

- raw chunk 的处理顺序
- replay insert 顺序

严格和真实执行顺序一致。

尤其当前还有：

- `pending_last_transition`
- synthetic terminal 合并

这些都要求 worker 看到的 chunk 序列严格有序。

#### B. `pending_last_transition` 的状态所有权

一旦异步化，`pending_last_transition` 就不应该再由 actor 持有，而应该明确收口到 worker 内部。

否则会出现：

- actor 有一份 pending
- worker 也有一份 pending

这会直接把边界语义搞乱。

#### C. actor 的计数器到底看什么时点

现在 actor 的 `env_steps / episode_steps / episode_return / episode_success` 基本是边执行边更新。

异步化后要明确：

- 这些计数器是看“已执行”
- 还是看“已提交 replay”

推荐做法是分开：

- `executed_env_steps`
- `committed_env_steps`

其中：

- rollout 控制看 `executed_env_steps`
- learner warmup / replay 可见度看 `committed_env_steps`

不要再混成一个变量。

#### D. episode stats 何时发送

如果 actor 在 episode 结束时立刻发：

- `send-stats`

但 worker 还没把最后几个 transition 真正提交进 replay，那 learner 看到的：

- `env_steps`
- `replay_size`

就会短时间不同步。

这个不是致命问题，但必须设计清楚。

最稳的做法是：

- actor 发 rollout stats
- worker 单独维护 committed replay stats
- learner 只把 replay warmup 绑定到 replay 自身，不依赖 episode stats 推断数据一定已到位

#### E. 崩溃恢复

如果 actor 把 raw chunk 放进内存队列后程序崩了，那么：

- 已执行的真机数据
- 还没来得及处理进 replay

仍然可能丢。

所以如果异步 worker 是正式生产方案，推荐至少补一层：

- raw chunk journal

不一定要很重，但至少要有：

- append-only JSONL / pickle / npz / msgpack

保证已执行数据可以事后重放给 worker。

### 14.4 推荐的最小异步边界

如果要上 `TransitionWorker`，推荐最小边界是：

- actor 仍然做 chunk 执行
- worker 独占：
  - `TransitionAssembler`
  - `pending_last_transition`
  - `ReplaySink`

不要让 actor 和 worker 共享 transition 级状态。

也就是说，worker 的输入就是：

- `RawChunkRecord`

worker 的输出就是：

- “已经 committed 到 replay”

### 14.5 推荐的数据结构补充

为了让异步边界更清晰，建议再加一个 `ChunkCommitResult`：

```python
@dataclass
class ChunkCommitResult:
    chunk_id: int
    episode_id: int

    committed_transitions: int
    committed_env_steps: int

    episode_done: bool
    episode_return_delta: float
    episode_success: bool

    last_info: dict[str, Any]
```

这样 actor 可以选择：

- 不等待 commit，纯异步推进
- 或者只在 episode 边界等待 commit 确认

### 14.6 推荐的 `TransitionWorker` 伪代码

```python
class TransitionWorker:
    def __init__(
        self,
        *,
        assembler: TransitionAssembler,
        replay_sink: ReplaySink,
        raw_chunk_queue,
        commit_queue=None,
    ):
        self.assembler = assembler
        self.replay_sink = replay_sink
        self.raw_chunk_queue = raw_chunk_queue
        self.commit_queue = commit_queue
        self.state = AssemblerState()

    def run_forever(self):
        while True:
            raw = self.raw_chunk_queue.get()
            if raw is None:
                break

            assembled = self.assembler.process_chunk(
                raw=raw,
                state=self.state,
            )

            self.replay_sink.insert_all(assembled.transitions_to_insert)
            self.state = assembled.next_state

            if self.commit_queue is not None:
                self.commit_queue.put(
                    ChunkCommitResult(
                        chunk_id=raw.chunk_id,
                        episode_id=raw.episode_id,
                        committed_transitions=len(assembled.transitions_to_insert),
                        committed_env_steps=assembled.env_steps_delta,
                        episode_done=assembled.episode_done,
                        episode_return_delta=assembled.episode_return_delta,
                        episode_success=assembled.episode_success,
                        last_info=assembled.last_info,
                    )
                )
```

actor 主线程则可以变成：

```python
while training:
    base_actions, residual_obs = planner.plan(obs, worker_state_hint)
    action_chunk = sample_chunk(...)
    chunk_result = env.step_chunk(action_chunk)

    raw = RawChunkRecord(...)
    raw_chunk_queue.put(raw)

    # 最简单版本：actor 先只更新执行计数，不等待 replay commit
    executed_env_steps += raw.executed_steps
    obs = raw.final_obs

    if raw.chunk_done or raw.chunk_truncated:
        break
```

### 14.7 推荐的落地顺序

如果明确要走异步 worker，我建议还是分成两步，而不是直接一步到位。

#### 第一步：同步版 `TransitionAssembler`

先把当前主线整理成：

- `RawChunkRecord`
- `TransitionAssembler`
- `ReplaySink`

但 actor 仍然同步调用它们。

这样可以先验证：

- assembler 行为和主线完全一致
- terminal boundary、pending、reward、prefetch 都没回归

#### 第二步：再把 assembler + sink 挪到 worker

这样异步化时，变更点只剩：

- 增加 queue
- 增加 worker loop
- 重新定义 counters / stats 的所有权

风险更小。

### 14.8 当前推荐结论

完全异步 `TransitionWorker` 方案不是“备选中的次优解”，而是：

- **更面向性能和长期演进的正式架构方向**

之所以之前没有先推它，只是因为：

- 它比“actor 先保留 replay 提交控制权”的方案更复杂

如果当前优先级已经转成：

- 提升执行吞吐
- 隔离 backfill 代价
- 为 dense reward / batch infer 做准备

那么完全可以把 `TransitionWorker` 方案作为下一阶段主目标。

## 15. 推荐的第一阶段文件拆分

如果要落地这个设计，推荐先拆这几个文件：

- `examples/agibot_real/training/raw_chunk.py`
  - `RawChunkRecord`
  - `AssemblerState`
  - `AssemblyResult`
- `examples/agibot_real/training/reward_builder.py`
  - `RewardBuilder`
  - `PassThroughRewardBuilder`
  - 未来的 dense reward builder
- `examples/agibot_real/training/transition_assembler.py`
  - `TransitionAssembler`
- `serl_launcher/serl_launcher/residual/chunk_window_replay.py`
  - 从原来的 `chunk_replay.py` 重命名而来

## 16. 当前推荐结论

当前最合理的方向是：

- 把 `chunk_replay.py` 改名成 `chunk_window_replay.py`
- 引入 `TransitionAssembler`
- 第一阶段只做纯重构，不改行为
- actor 继续保留 replay 提交控制权
- 后续再视需要引入 `ReplaySink` 和异步 worker

这条路线的优点是：

- 命名准确
- 职责清晰
- 便于后续加 dense reward
- 便于后续做性能优化
- 不需要一次性重写 actor / learner 协议
