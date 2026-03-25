# LIBERO 异步评估逻辑说明（训练端 + Watcher）

本文档说明当前 `examples/libero` 里的异步评估实现，重点回答：

1. 评估请求如何触发、如何执行；
2. 请求是否会一直排队、排到什么时候。

截至代码版本（2026-03-25）对应核心文件：

- `examples/libero/scripts/train_residual_sac.py`
- `examples/libero/utils/async_eval.py`
- `examples/libero/scripts/async_eval_watch.py`

---

## 1. 总体架构

异步评估由两个进程协作：

- 训练进程（producer）：按训练 episode 触发，写入评估请求队列。
- watcher 进程（consumer）：轮询队列，串行拉起评估脚本并写结果。

训练启动时会自动拉起 watcher（若 `training.async_eval.enabled=true`）：

- 队列文件：`<run_dir>/async_eval_queue.jsonl`
- 结果文件：`<run_dir>/async_eval_results.jsonl`
- watcher 日志：`<run_dir>/async_eval_watch.log`
- 单次评估目录：`<run_dir>/async_eval/episode_XXXXXX_step_XXXXXXX/`

---

## 2. 触发与入队（训练进程）

触发条件在训练 episode 结束后检查：

- `train_episode_id % training.async_eval.every_episodes == 0`

满足后会先保存当前 step 对应 checkpoint，再向队列追加一条 JSONL 记录，字段包括：

- `eval_index`（自增触发序号）
- `train_episode_id`
- `train_env_step`
- `checkpoint_step`（当前用 env-step）
- `checkpoint_path`

注意：

- warmup 阶段不会触发异步评估；
- 目前是 “episode 触发 eval”，不是 “step 触发 eval”。

---

## 3. 出队与执行（watcher）

watcher 主循环逻辑：

1. 读 `async_eval_queue.jsonl` 全量行；
2. 跳过已完成 `eval_index`（由 `async_eval_results.jsonl` 重建）；
3. 对新请求做 checkpoint 稳定性检查（`file_stable_sec`）；
4. 放入 `pending_requests`；
5. 按 `queue_policy` 取请求启动评估（一次只跑一个）：
   - `all`：按 `eval_index` 从小到大全跑；
   - `latest`：只保留最新请求，旧请求在内存 pending 中被丢弃。

每次评估结束后，会把 `ok/failed/aborted` 结果追加到 `async_eval_results.jsonl`，训练进程再把新结果同步到 TensorBoard。

---

## 4. “会不会一直排队？”的准确结论

### 4.1 训练进行中

会排队，而且在以下条件下可能持续增长：

- `queue_policy=all`（你当前这组配置就是这个）；
- 训练触发速度快于评估完成速度（常见：每次 eval 50 episodes，耗时较长）。

此时 `eval_q`（tqdm 里看到的字段）会上升，它表示“已触发 - 已写入结果”的差值，不是成功数量。

### 4.2 什么时候停止排队

排队停止有两层含义：

1. **停止新增请求**：训练结束后不再 enqueue。
2. **停止消费队列**：训练 `finally` 会主动 `terminate` watcher，不会等待把剩余 pending 全部跑完。

所以当前实现下：

- 不是“训练停了 watcher 还会把队列清空跑完”；
- 而是“训练停了就停 watcher”，可能留下未消费请求在 `async_eval_queue.jsonl` 中。

---

## 5. 当前实现的几个关键点

- 评估种子使用 `training.async_eval.seed`（固定值，不轮换）。
- `queue_policy=latest` 可显著抑制积压，但会牺牲中间 checkpoint 的评估覆盖。
- 若请求对应 checkpoint 在真正启动评估前已不存在，watcher 会告警并跳过本次启动尝试（该请求仍在队列文件里）。
- watcher 只做执行与记录；触发节奏由训练进程决定（watcher 不负责“每 N episode 调度”）。

---

## 6. 对你当前实验配置的直接解读

对于 `ablation_stepchunk_vs_step_task6_xi_warmup_eval50` 下的配置：

- `every_episodes: 50`
- `episodes: 50`
- `queue_policy: all`

意味着：

- 每 50 个训练 episode 触发一次异步评估；
- 每次评估跑 50 个 episode；
- 所有触发点都会排队等待执行（若评估慢，会明显积压）。

