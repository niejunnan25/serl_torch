# AgiBot Copy Split-Queue Manual Checklist

## Startup

- 用 `examples/agibot_real/scripts/run_residual_training_copy.py` 默认配置启动，确认默认 `config_name=train_residual_copy`。
- learner 和 actor 的日志里确认 `transport_mode=split_queue`。
- 端口确认：
  - control: `runtime.trainer_port`
  - broadcast: `runtime.broadcast_port`
  - data: `runtime.trainer_transport.data_port`

## Actor / Learner Health

- actor 定期日志或 `actor_timers.jsonl` 中确认存在：
  - `accepted_update_id`
  - `committed_update_id`
  - `transport_backlog`
  - `data_queue_depth`
- learner 的 `learner_timers.jsonl` 中确认同样字段存在。
- `transport_backlog` 在稳态下不应持续单调增长。

## Episode Boundary

- 默认 `wait_committed_on_episode_end=false`：
  - episode 结束后 actor 不应等待 replay 完全 commit 再进入 reset。
  - reset 期间 learner 仍可继续 drain queue。
- 默认 `wait_committed_on_shutdown=true`：
  - 正常退出前应把最后一批 pending 数据 commit 完。

## Real-Robot Reset Overlap

- 在 episode 结束后观察：
  - reset 是否先开始
  - learner `committed_update_id` 是否在 reset 期间继续增长
- 如果 reset 明显比之前更快，但 `transport_backlog` 不爆涨，说明重叠生效。

## Backpressure

- 如果 `transport_backlog` 长时间 > 0 且持续扩大：
  - 先看 `data_queue_depth` 是否接近上限
  - 再看 `backfill_policy.max_pending_chunks` 是否过大
  - 再看 learner 侧 replay / update 是否跟不上
- 不建议第一反应直接拉高 timeout；先确认 backlog 来源。

## Failure Signs

- actor 反复 `update()` 失败
- `accepted_update_id` 长时间不增长
- `committed_update_id` 长时间落后于 `accepted_update_id`
- `data_queue_depth` 长时间贴近上限
- episode 结束后 reset 仍明显被 replay drain 阻塞

## First Tuning Knobs

- `runtime.trainer_transport.data_queue_capacity`
- `runtime.trainer_transport.data_socket_hwm`
- `runtime.trainer_transport.control_timeout_ms`
- `runtime.trainer_transport.commit_poll_ms`
- `backfill_policy.max_pending_chunks`
