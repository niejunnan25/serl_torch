# LIBERO Async Eval 按 Episode 触发的已知风险

## 1. 当前触发语义

当前 LIBERO 训练期 async eval 的触发方式是：

- actor 在每个训练 episode 结束后，上报 `rollout.episode_id`
- learner 维护最新的已完成 episode 计数
- 当 `latest_completed_episode_id` 跨过 `training.async_eval.every_episodes` 的整数倍时
- learner 保存一份 checkpoint，并把 async eval 请求写进队列
- async eval worker 异步消费这份队列

这里要特别区分两件事：

- **评估执行** 是异步的
- **评估触发与排队** 仍然发生在 learner 主流程里

所以，异步 worker 并不能消除“触发时机”和“checkpoint 时机”之间的偏差。

## 2. 已知风险

如果 actor 比 learner 快，learner 可能会在一次 `_maybe_queue_async_eval()` 调用里，补发多个 episode milestone。

例如：

- `training.async_eval.every_episodes = 20`
- `last_queued_async_eval_episode = 0`
- actor 已经把 `latest_completed_episode_id` 推进到了 `45`
- learner 当前的 `update_steps` 仍然停留在同一个值

这时 learner 可能会连续补发：

- `episode = 20`
- `episode = 40`

但这两条请求对应的 checkpoint，可能都是同一个 `update_steps` 下保存出来的同一份权重。

也就是说，**不同的 episode milestone，可能评估的是同一份 learner snapshot**。

## 3. 什么时候最容易发生

这个风险通常不是 steady-state 训练里的主问题，但在下面几种情况下更容易出现：

- replay warmup 阶段：actor 已经在积累 episode，但 learner 还没正式进入主训练循环
- offline pretrain 阶段：learner 在做纯 offline 更新，而 actor 的 episode 计数已经继续前进
- learner update 很重：例如 `critic_actor_ratio`、`utd_ratio` 较高，或者 GPU 较忙

## 4. 影响

这个问题通常 **不会把训练跑坏**，也不会破坏 async eval worker 的执行正确性。

它的主要影响是：

- 浪费 eval 资源
- 让 `episode` 轴上的实验解释变脏
- 可能出现“`episode=20` 和 `episode=40` 实际评估的是同一份 checkpoint”这种情况

所以它更像一个 **语义风险 / 实验解释风险**，不是当前的阻塞性功能 bug。

## 5. 当前结论

当前代码接受这个 tradeoff，原因是：

- 训练主结构仍然保持干净
- checkpoint 继续由 learner 统一持有和保存
- async eval 仍然是异步执行，不会阻塞主训练

因此，这个问题目前被记录为 **已知风险**，但不是必须立刻修复的阻塞项。

## 6. 后续可能的改进方向

如果后面要进一步收紧 episode 语义，可以考虑两类方案：

- 保守方案：在 replay warmup / offline pretrain 期间也定期调用 `_maybe_queue_async_eval()`，尽量减少 milestone 积压
- 严格方案：让 async eval 对应到更明确的 learner snapshot 边界，而不是在 catch-up 时对多个 episode milestone 复用同一个当前 checkpoint
