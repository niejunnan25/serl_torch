# 本周四项核心优化说明

这份文档把本周最关键、也最适合直接写进周报的四项优化拆开说明：

1. learner 侧：`bf16` + actor update 阶段冻结 critic 参数梯度
2. replay / transport 侧：`batch_insert()` + `split_queue`
3. OpenPI batch infer 侧：`infer_many()` + LIBERO chunk 训练线
4. actor 热路径：把大量 post-hoc 组装从主控制路径上移开

这四项工作不是彼此独立的小修小补，而是围绕同一条训练链路在不同位置做减压：

- learner 更新更快
- replay 写入更快
- actor 到 learner 的数据提交更顺
- policy backfill 更批量化
- actor 热路径更干净

所以它们共同指向的目标，不是“把某个模块 benchmark 跑快一点”，而是让 residual RL 主线从“能跑”进一步变成“更连续、更高吞吐、更容易长期跑稳”。

## 1. Learner 侧：`bf16` + actor update 阶段冻结 critic 参数梯度

相关文档：

- [2026-04-16-learner-gradient-update-speed-optimization.md](./2026-04-16-learner-gradient-update-speed-optimization.md)
- [2026-04-16-learner-speed-ablation-bf16-and-critic-freeze.md](./2026-04-16-learner-speed-ablation-bf16-and-critic-freeze.md)

相关实现：

- [serl_launcher/serl_launcher/agents/continuous/sac.py](/home/hello/codebase/serl_torch/serl_launcher/serl_launcher/agents/continuous/sac.py:400)
- [test/benchmark_learner_update_speed.py](/home/hello/codebase/serl_torch/test/benchmark_learner_update_speed.py:1)

### 1.1 原来的问题是什么

这轮优化之前，learner 的瓶颈已经比较明确。一次外层 `outer update` 并不是只做一次反向传播，而是一个很重的组合动作。

在 AgiBot 默认配置下：

- `critic_actor_ratio = 4`
- `utd_ratio = 2`

所以一次 learner 外层更新，实际包含：

- 3 次独立 `update_critics(...)`
- 1 次 `update_high_utd(...)`

而 `update_high_utd(...)` 内部又会：

- 再做 2 次 critic update
- 再做 1 次 actor update
- 再做 1 次 temperature update

也就是说，一次 `outer update` 背后实际上是：

- 5 次 critic 梯度更新
- 1 次 actor 梯度更新
- 1 次 temperature 梯度更新
- 多次 replay sample
- 多次图像增强和 CPU -> GPU tensor 搬运

这意味着 learner 天生就是这条链路里最容易掉队的环节之一。如果 learner 更新吞吐上不去，就会出现：

- actor 继续往 replay 塞数据
- learner update 长期追不上
- 训练逐渐从“在线联动”退化成“actor 在前面跑、learner 在后面补”

### 1.2 这次具体改了什么

这次没有先改算法，而是先做两项低侵入优化。

第一项是打开 mixed precision。

代码上，`SACAgent` 新增了统一的 mixed precision 入口：

- `_mixed_precision_config(...)`
- `_autocast_context(...)`

并在 critic / actor / temperature 更新路径上使用 `torch.autocast(...)`。

这件事的重点不是“随便开个 AMP”，而是把它纳入当前 Torch 版 SAC agent 的统一运行时语义里，让配置层可以显式打开：

- `training.mixed_precision.enabled=true`
- `training.mixed_precision.dtype=bfloat16`

第二项是修掉 actor update 阶段的无效 critic 参数梯度。

原来的 actor loss 计算需要 critic 提供 `Q(s, pi(s))`，因此会经过 critic forward；但 actor optimizer 并不会 step critic 参数。也就是说：

- critic 参数梯度会被算出来
- 但这些梯度不会被真正使用

这轮修复的关键点在：

- 不能直接用 `torch.no_grad()` 包 critic forward
- 因为 actor 仍然需要 `dQ/da`

最终实现方式是：

- 用 `_temporarily_freeze_params(...)`
- 在 actor update 阶段临时对 critic 参数做 `requires_grad_(False)`
- 保留 action 到 Q 的梯度路径
- 避免为 critic 参数构建和回传无用梯度

这个修复的性质很重要。它不是改目标函数，而是去掉原来白算的反传。

### 1.3 为什么会有效

`bf16` 的收益主要来自两方面：

- 降低前向 / 反向计算成本
- 降低激活和梯度的显存占用

这对当前这种视觉输入较重、critic 更新次数又多的 learner 很重要。

冻结 critic 参数梯度的收益则更聚焦：

- 只影响 actor update 阶段
- 不改变前面 critic-only update 的逻辑
- 直接减少 `update_high_utd(...)` 里 actor 分支的无效反传开销

所以这两项优化叠加后，改善的是 learner 最重、也最频繁的那部分计算。

### 1.4 实测结果说明了什么

文档中的消融结果是：

- `outer update: 2.8806s -> 2.1040s`
- 总提升约 `26.96%`
- `peak mem: 17052.6MB -> 9310.1MB`
- 显存峰值下降约 `45.2%`

如果拆开看：

- 先开 `bf16`，已经能明显降低 critic-only 阶段耗时
- 再修 actor update 白算 critic grad，主要继续压低 `high_utd` 阶段成本

这组数字的意义不只是“update 更快了”，而是 learner 更有机会接近 actor 的数据产生速度，减少长期落后的风险。

换句话说，这轮优化解决的是：

- 不是简单让 benchmark 更好看
- 而是让在线训练里 learner 不那么容易成为稳定瓶颈

## 2. Replay / Transport 侧：`batch_insert()` + `split_queue`

相关文档：

- [2026-04-17-agentlace-serl-plan-c-design.md](./2026-04-17-agentlace-serl-plan-c-design.md)
- [2026-04-17-plan-c-vectorized-rollout-implementation-report.md](./2026-04-17-plan-c-vectorized-rollout-implementation-report.md)

相关实现：

- [serl_launcher/serl_launcher/data/batch_ops.py](/home/hello/codebase/serl_torch/serl_launcher/serl_launcher/data/batch_ops.py:1)
- [serl_launcher/serl_launcher/data/data_store.py](/home/hello/codebase/serl_torch/serl_launcher/serl_launcher/data/data_store.py:104)
- [serl_launcher/serl_launcher/common/trainer_transport.py](/home/hello/codebase/serl_torch/serl_launcher/serl_launcher/common/trainer_transport.py:87)
- [examples/agibot_real/scripts/run_residual_training.py](/home/hello/codebase/serl_torch/examples/agibot_real/scripts/run_residual_training.py:137)
- [examples/libero/scripts/train_residual_chunk.py](/home/hello/codebase/serl_torch/examples/libero/scripts/train_residual_chunk.py:471)

### 2.1 原来的问题是什么

原来的 trainer data path 有两个明显问题。

第一，数据写入是“逻辑上成批，实际上一条条写”。

当前 residual 训练线虽然常常一次 chunk 就会产出一批 step transition，但写入 replay 时，如果底层还是按单条 insert 处理，就会重复付出很多成本：

- Python 层循环
- ring buffer 多次小写入
- episode / step 元信息重复处理
- 多次锁竞争

第二，control 和 data 共用一条 trainer 通道。

这会导致：

- `update()`、`send-stats`、`get_last_update_id` 挤在同一条 `req/rep`
- 大 payload 的 datastore 更新会连带阻塞小控制请求
- learner 一边 sample/train，一边同步插 replay，会形成更明显的锁竞争

所以旧问题并不是单点慢，而是：

- replay 写入本身有低效
- transport 结构也让这些低效更容易放大

### 2.2 这次具体改了什么

这轮改动分成两层。

第一层是 replay `batch_insert()`。

为了让 batch 写入变成真正的一次性操作，先加了统一的 packed batch helper：

- `pack_transition_batch(...)`
- `packed_batch_slice(...)`
- `ring_write_batch(...)`

然后在几类 replay store 上补了 `batch_insert()`：

- `ReplayBufferDataStore`
- `StepWindowReplayBufferDataStore`
- `MemoryEfficientStepWindowReplayBufferDataStore`

其中最关键的是：

- `MemoryEfficientStepWindowReplayBufferDataStore`

因为 copy 训练线当前真正走的就是这条生产路径。

这意味着在当前主线里，batch 写入不是停留在 toy replay 上，而是已经落到实际 residual RL 的关键 replay 上。

第二层是 repo-local `trainer_transport`。

这轮没有继续把所有 trainer data/control 都塞回外部 `agentlace` 的同步 callback，而是自己在仓库里实现了 transport wrapper：

- `legacy_reqrep`
- `split_queue`

`split_queue` 模式下，控制面和数据面被物理拆开：

- control 继续走 `req/rep`
- data 单独走 `push/pull`
- learner 收到 data 后先放入 bounded queue
- replay 真正 commit 由 worker 线程异步做

同时协议语义也不再只看一个 update id，而是显式区分：

- `accepted_update_id`
- `committed_update_id`

还补了 transport 侧 observability：

- `transport_backlog`
- `data_queue_depth`

### 2.3 为什么会有效

`batch_insert()` 的收益很直接：

- 把“很多次单条小写入”变成“一次 batch 写入”
- 降低 Python 循环、锁开销和 ring buffer 切分成本
- 对 step-window replay 这种天然批量化的数据尤其有效

`split_queue` 的收益则主要来自解耦：

- 小控制请求不再被大数据包直接堵住
- replay 写入不再在 control callback 里同步完成
- actor 可以更早知道 learner 已接收数据
- learner 可以在自己的节奏里 drain queue

所以这轮提升不只是某个函数更快，而是 trainer ingest 整体变得更像一条真正的流水线。

### 2.4 实测结果说明了什么

当前文档里记录的两组核心结果是：

第一组，生产 replay store 的 `batch_insert()` 本身：

- `single_mean_s = 0.0147736858`
- `batch_mean_s = 0.0023672415`
- `speedup_mean = 6.2408866`

也就是：

- 当前生产路径上的 `MemoryEfficientStepWindowReplayBufferDataStore`
- `batch_insert()` 相比单条 insert 约有 `6.24x` 提升

第二组，更接近真实训练负载的 transport under load：

- legacy 总成本约 `0.2053s`
- split_queue 总成本约 `0.01557s`
- 总成本加速约 `13.19x`

这组结果很重要，因为它说明本周的系统优化不是“轻量 smoke benchmark 漂亮”，而是在更接近真实训练负载时依然成立。

在周报里，这一条完全可以被写成：

- 本周最硬的一组系统优化结果之一，是 trainer ingest 和 replay 写入链路的重构与量化提速

## 3. OpenPI Batch Infer：`infer_many()` + LIBERO chunk 训练线

相关文档：

- [examples/libero/docs/openpi_batch_infer_chunk_report_2026_04_17.md](/home/hello/codebase/serl_torch/examples/libero/docs/openpi_batch_infer_chunk_report_2026_04_17.md:1)

相关实现：

- [serl_launcher/serl_launcher/policy/openpi/request_builder.py](/home/hello/codebase/serl_torch/serl_launcher/serl_launcher/policy/openpi/request_builder.py:67)
- [serl_launcher/serl_launcher/policy/openpi/client.py](/home/hello/codebase/serl_torch/serl_launcher/serl_launcher/policy/openpi/client.py:159)
- [examples/libero/scripts/train_residual_chunk.py](/home/hello/codebase/serl_torch/examples/libero/scripts/train_residual_chunk.py:1)

### 3.1 原来的问题是什么

在这轮改动前，JoyRA 路径已经有 chunk 级 batch backfill 的能力，但 OpenPI 路径还没有真正补齐。

这导致 OpenPI 路径上的 backfill 更接近：

- 每个 post-step observation 一次串行 infer

也就是说，chunk 执行完之后，如果要为 chunk 内每个 step 回填下一状态所需的 base action chunk，就得重复做很多次单样本请求。

这会带来两类浪费：

- 多次 websocket / RPC 往返
- 模型侧无法真正利用 batch 推理

### 3.2 这次具体改了什么

这轮改动的关键不是把主 actor 决策改成 batch，而是只把 backfill 路径改成 batch。

这样做的好处是：

- 不改变当前 actor 主决策语义
- 只优化 chunk 执行后那一段本来就适合成批处理的回填路径

在本仓库里，对应改动包括：

- `build_openpi_batch_request(...)`
- `OpenPIPolicyClient.infer_many(...)`

在训练线里，对应的是：

- `chunk` 的 backfill 路径在检测到 client 支持 `infer_many(...)` 后
- 会把一个 chunk 的多个 `PolicyInput` 打成一个 batch 发出去

文档里还记录了外部 `openpi-modified` 的配套改动：

- server 支持 batch wire protocol
- `Policy.infer_many(...)`
- batch request / response 的数据格式

这意味着当前 OpenPI 路径已经不是“伪 batch”，而是真正把一个 chunk 的 backfill 合并成了一次模型调用。

### 3.3 为什么会有效

这条优化本质上是在把重复的 per-step 推理，压缩成 per-chunk 的 batched backfill。

收益来源主要有三部分：

- 减少请求次数
- 减少网络往返
- 让模型真正跑在 batch 模式下

因为 backfill 天然就是“同一段 chunk 执行完后，对一串 post-step obs 做同类型推理”，所以这是最适合做 batch 的地方。

### 3.4 实测结果说明了什么

文档里给出的模型侧 benchmark 是：

- `serial_median_s = 2.8405`
- `batch_median_s = 1.2160`
- `speedup_vs_serial = 2.336x`

这说明模型侧真 batch 已经生效。

再看端到端 actor 吞吐：

- `chunk = 5.803 step/s`
- 原始 `train_residual_step.py = 5.105 step/s`
- 提升约 `13.7%`

端到端没有吃满 `2.336x` 很正常，因为现在新的主瓶颈已经不是 OpenPI 本身，而是：

- trainer socket timeout / backpressure
- learner drain
- remote env RPC
- episode boundary 上的异步收尾

这也是这轮优化很有价值的一点：

- 它不只是把某个模块提速
- 还帮你把系统真正的下一层瓶颈暴露出来了

## 4. Actor 热路径：`build_decision_obs` 基本被清空

相关文档：

- [examples/libero/docs/openpi_batch_infer_chunk_report_2026_04_17.md](/home/hello/codebase/serl_torch/examples/libero/docs/openpi_batch_infer_chunk_report_2026_04_17.md:145)
- [2026-04-16-agibot-transition-dataflow-and-refactor-plan.md](./2026-04-16-agibot-transition-dataflow-and-refactor-plan.md)

相关实现：

- [examples/libero/scripts/train_residual_chunk.py](/home/hello/codebase/serl_torch/examples/libero/scripts/train_residual_chunk.py:620)
- [test/benchmark_libero_rollout_30hz.py](/home/hello/codebase/serl_torch/test/benchmark_libero_rollout_30hz.py:15)

### 4.1 原来的问题是什么

原始 step-wise 训练线里，`build_decision_obs` 不只是一个轻量拼装动作，而是 actor 热路径里相当重的一段工作。

原因在于原来的路径更接近：

- 每执行一步
- 就要尽快为下一步构造新的 residual decision 输入

这意味着 actor 热路径上会混进很多“为了训练记账而不是为了当前控制决策本身”的工作，例如：

- 对下一状态再跑 base policy
- 组装 `next_residual_obs`
- 逐步形成 replay transition 所需的中间量

这类工作放在控制热路径里，会直接拉长每轮 chunk 之间的空转时间。

### 4.2 这次具体改了什么

`chunk` 路径做的不是小修，而是重新安排了工作分布：

- chunk 决策时，只做当前真正需要的决策工作
- chunk 执行后，先缓存原始 rollout
- post-hoc transition assembly 放到更适合的回填路径里
- backfill 如果能 batch 就 batch，如果能异步就异步

也就是说，这轮优化的本质是：

- 不是减少训练需要的信息
- 而是把“训练记账”从“控制热路径”搬到“后处理路径”

### 4.3 为什么会有效

actor 热路径最怕的不是计算量大本身，而是把不影响当前控制的计算也塞进来了。

当 `build_decision_obs` 里掺杂了大量：

- post-step 回填
- next residual obs 组装
- 逐步 replay 准备

actor 就会在 chunk 与 chunk 之间花很多时间做“此刻并不影响机器人继续动”的事情。

`chunk` 的价值就在于把这些工作拆开：

- 当前控制继续尽量快
- 训练数据的完整性通过 post-hoc assembly 保证

### 4.4 实测结果说明了什么

文档里给出的关键数字非常直观：

原始训练线：

- `build_decision_obs ~= 0.1077s`
- `total ~= 0.8509s`

`chunk` 训练线：

- `build_decision_obs ~= 0.000078s`
- `total ~= 0.5025s`

这里最关键的不是 `total` 降了多少，而是：

- `build_decision_obs` 这段原本显著存在于 actor 热路径里的开销
- 已经被压到接近可以忽略

这说明这轮架构调整是有效的：

- 大量原本阻塞当前控制的 post-hoc 组装工作
- 已经被真正挪出了主控制路径

这也是为什么你可以把这一条写得比较明确：

- actor 热路径已经明显被清空
- 当前系统开始从“控制和记账绑在一起”走向“控制优先、训练后处理分离”

## 5. 这四条优化放在一起看，真正提升了什么

如果把这四点单独看，它们分别优化的是：

- learner update
- replay write
- transport ingest
- policy backfill
- actor 热路径

但如果把它们放在一起看，真正改善的是整条 residual RL 在线训练链路的协同关系。

更具体地说，本周这组优化的实际价值是：

- learner 没那么容易长期落后
- replay 写入更接近真正的 batch pipeline
- actor 发数据和 learner 接数据之间不再那么互相阻塞
- OpenPI 路径不再因为缺少 batch backfill 而明显拖后腿
- actor 热路径更专注于“当前控制决策”，而不是夹带大量训练后处理

因此，这周的重点并不是“把某个模块 benchmark 跑快了”，而是：

- 让当前主线从一条耦合较重、容易在多个位置互相拖累的训练链路
- 开始变成一条边界更清楚、吞吐更高、瓶颈更可观察的系统

这也是为什么这些工作很适合在周报里被归纳为：

- 训练性能优化
- 数据链路优化
- actor / learner 解耦
- 仓库主线整理后的结构性提效
