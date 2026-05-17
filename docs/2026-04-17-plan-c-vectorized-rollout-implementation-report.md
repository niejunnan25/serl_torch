# 2026-04-17 Plan C + Vectorized Replay 实施报告

## 1. 这轮实际落地了什么

这轮不是只停留在 benchmark 原型，而是把两类优化真正接进了 `serl_torch` 当前训练链路：

1. replay `batch_insert()`
2. repo-local trainer transport `split_queue`

对应代码如下。

### 1.1 Replay batch insert

- packed batch helper:
  - [serl_launcher/serl_launcher/data/batch_ops.py](/home/hello/codebase/serl_torch/serl_launcher/serl_launcher/data/batch_ops.py:1)
- datastore batch insert:
  - [serl_launcher/serl_launcher/data/data_store.py](/home/hello/codebase/serl_torch/serl_launcher/serl_launcher/data/data_store.py:1)

当前状态：

- `ReplayBufferDataStore`: 已接真正 batch-aware 写入
- `StepWindowReplayBufferDataStore`: 已接真正 batch-aware 写入
- `MemoryEfficientStepWindowReplayBufferDataStore`: 已接真正 batch-aware 写入
- `MemoryEfficientReplayBufferDataStore`: 已支持 `batch_insert()`，但仍是“单锁下顺序 insert”，不是同等级 fully vectorized 写入

注意：

- 当前 copy 训练线实际用的是 `MemoryEfficientStepWindowReplayBufferDataStore`，不是 `MemoryEfficientReplayBufferDataStore`
- 因此最关键的生产路径已经吃到了真正的 replay batch insert 收益

### 1.2 Repo-local Plan C transport

- transport 实现：
  - [serl_launcher/serl_launcher/common/trainer_transport.py](/home/hello/codebase/serl_torch/serl_launcher/serl_launcher/common/trainer_transport.py:1)

当前实现包含：

- `legacy_reqrep` wrapper
- `split_queue` transport
- control/data 双通道
- learner bounded queue
- replay commit worker
- `accepted_update_id`
- `committed_update_id`
- actor 侧 `last_sent_id` / pending resend 语义
- `get_transport_status()`

### 1.3 Config 和 copy 训练线接入

- typed config:
  - [examples/libero/config.py](/home/hello/codebase/serl_torch/examples/libero/config.py:1)
  - [examples/agibot_real/config.py](/home/hello/codebase/serl_torch/examples/agibot_real/config.py:1)
- chunk/mainline yaml:
  - [examples/libero/configs/train_residual_chunk.yaml](/home/hello/codebase/serl_torch/examples/libero/configs/train_residual_chunk.yaml:1)
  - [examples/agibot_real/configs/train_residual.yaml](/home/hello/codebase/serl_torch/examples/agibot_real/configs/train_residual.yaml:1)
- canonical train yaml 也显式声明了 `runtime.trainer_transport`
- 实验脚本默认入口已切换：
  - LIBERO `chunk`: [examples/libero/scripts/train_residual_chunk.py](/home/hello/codebase/serl_torch/examples/libero/scripts/train_residual_chunk.py:1400)
  - AgiBot mainline: [examples/agibot_real/scripts/run_residual_training.py](/home/hello/codebase/serl_torch/examples/agibot_real/scripts/run_residual_training.py:948)

### 1.4 训练脚本中的 transport observability

当前实验训练线会把这些 transport 指标带进 summary / timer jsonl：

- `transport_mode`
- `accepted_update_id`
- `committed_update_id`
- `transport_backlog`
- `data_queue_depth`

对应脚本：

- [examples/libero/scripts/train_residual_chunk.py](/home/hello/codebase/serl_torch/examples/libero/scripts/train_residual_chunk.py:1)
- [examples/agibot_real/scripts/run_residual_training.py](/home/hello/codebase/serl_torch/examples/agibot_real/scripts/run_residual_training.py:1)

## 2. 为什么“原型里提升很多”，而早先 smoke benchmark 看起来不明显

这轮 review 后，结论很明确：

不是当前实现没提升，而是之前看的 benchmark 口径不对。

主要原因有四个。

### 2.1 之前的 smoke benchmark 太轻

[test/benchmark_trainer_transport_real.py](/home/hello/codebase/serl_torch/test/benchmark_trainer_transport_real.py:1) 最早那版 smoke 用的是：

- state-only observation
- 没有 sampler 并发
- 没有 `send-stats` 干扰
- batch 也比较小

在这种负载下，legacy 本来就很快，`split_queue` 很难拉开差距。

### 2.2 当前 legacy 已经不是原始 baseline

当前代码里，legacy 路径已经自动吃到了 replay `batch_insert()` 的收益。

所以现在真实在比的是：

- `legacy req/rep + vectorized replay insert`
vs
- `split_queue + vectorized replay insert`

而不是最初原型里那种：

- `legacy req/rep + per-item insert`
vs
- `Plan C + vectorized replay insert`

因此真实生产对比，不会完整复现最初 synthetic benchmark 里的全部收益。

### 2.3 `commit` 指标不能直接横比

在 legacy 里，replay commit 大多已经包含在 `update()` 内。

在 `split_queue` 里，commit 被异步拆出来了，因此：

- legacy 的 `commit≈0`
- split_queue 会看到单独的非零 `commit`

这不表示 split_queue 更慢，而是语义不同。

正确比较方式应当看：

- `update + stats + commit` 总成本

### 2.4 端到端一定受 Amdahl 限制

即便 trainer ingest 这段提速很大，也不意味着整条训练链路同倍数提速。

真实训练还受这些部分限制：

- env `step_chunk`
- policy / backfill
- reset
- learner 自身 update
- checkpoint / logging

所以局部 5x-10x，并不等于整条训练链路 5x-10x。

## 3. 当前生产实现的实际 benchmark

这轮我额外跑了两类更接近真实实现的 benchmark。

### 3.1 真实 replay store 插入 benchmark

结果文件：

- [test/results/benchmark_current_prod_transport_review_2026_04_17.json](/home/hello/codebase/serl_torch/test/results/benchmark_current_prod_transport_review_2026_04_17.json:1)

其中 `store_only` 部分直接测了当前 copy 训练线实际会用到的 replay store：

- `single_mean_s = 0.0147736858`
- `batch_mean_s = 0.0023672415`
- `speedup_mean = 6.2408866`

解释：

- 这里测的不是 toy replay，而是 `MemoryEfficientStepWindowReplayBufferDataStore`
- 这说明当前生产实现里的 replay `batch_insert()` 本身已经有约 `6.24x` 的提升

### 3.2 当前生产 transport 在训练负载下的 benchmark

同一个结果文件里，`transport_under_load` 部分测的是：

- actor 发 `datastore`
- learner 并发 sampler
- 每轮额外发一次 `send-stats`

结果：

#### legacy req/rep

- `update_mean_s = 0.2045387397`
- `stats_mean_s = 0.0007800892`
- `commit_mean_s = 0.0000029614`

#### split_queue

- `update_mean_s = 0.0118154648`
- `stats_mean_s = 0.0004687418`
- `commit_mean_s = 0.0032865438`

如果按一轮总成本近似计算：

- legacy: `0.2045387397 + 0.0007800892 + 0.0000029614 ≈ 0.2053217903`
- split_queue: `0.0118154648 + 0.0004687418 + 0.0032865438 ≈ 0.0155707504`

对应总成本加速约：

- `0.2053217903 / 0.0155707504 ≈ 13.19x`

也就是说，在当前更接近真实训练负载的口径下：

- 当前生产实现的 trainer ingest 相关成本大约有 `13.2x` 的改进

### 3.3 之前 smoke benchmark 为什么容易误导

我还保留了一份更轻量的 smoke：

- [test/results/benchmark_trainer_transport_real_smoke.json](/home/hello/codebase/serl_torch/test/results/benchmark_trainer_transport_real_smoke.json:1)

这份结果里：

- `legacy_reqrep.update_mean_s ≈ 0.00733`
- `split_queue.update_mean_s ≈ 0.00701`

看上去提升不大。

但这是因为这份 benchmark：

- replay 很轻
- 没有 sampler 并发
- 没有复杂 observation
- 没有 `send-stats` 干扰

它只能作为 transport smoke，不能作为真实收益评估依据。

## 4. 与最初 synthetic benchmark 的关系

最初原型 benchmark 文件：

- [test/benchmark_trainer_datastore_variants.py](/home/hello/codebase/serl_torch/test/benchmark_trainer_datastore_variants.py:1)
- [test/results/benchmark_trainer_datastore_variants_report_2026_04_17.md](/home/hello/codebase/serl_torch/test/results/benchmark_trainer_datastore_variants_report_2026_04_17.md:1)

那份原型的关键结论是：

- vectorized replay insert 有价值
- control/data 拆通道的收益最大

这个结论在当前生产实现里仍然成立。

但要注意：

- 原型里的数字，不能直接当作当前生产代码的最终收益
- 当前生产代码的实际收益，必须看当前生产实现的 benchmark

也就是说：

- 原型结论负责指方向
- 当前实现 benchmark 负责给上线前的真实量化

## 5. 这轮跑过的验证

### 5.1 编译检查

已通过 `py_compile`：

- [serl_launcher/serl_launcher/data/batch_ops.py](/home/hello/codebase/serl_torch/serl_launcher/serl_launcher/data/batch_ops.py:1)
- [serl_launcher/serl_launcher/data/data_store.py](/home/hello/codebase/serl_torch/serl_launcher/serl_launcher/data/data_store.py:1)
- [serl_launcher/serl_launcher/common/trainer_transport.py](/home/hello/codebase/serl_torch/serl_launcher/serl_launcher/common/trainer_transport.py:1)
- [examples/libero/config.py](/home/hello/codebase/serl_torch/examples/libero/config.py:1)
- [examples/agibot_real/config.py](/home/hello/codebase/serl_torch/examples/agibot_real/config.py:1)
- [examples/libero/scripts/train_residual_chunk.py](/home/hello/codebase/serl_torch/examples/libero/scripts/train_residual_chunk.py:1)
- [examples/agibot_real/scripts/run_residual_training.py](/home/hello/codebase/serl_torch/examples/agibot_real/scripts/run_residual_training.py:1)

### 5.2 Targeted tests

新增并手动执行通过：

- replay batch insert:
  - [serl_launcher/tests/data/test_data_store_batch_insert.py](/home/hello/codebase/serl_torch/serl_launcher/tests/data/test_data_store_batch_insert.py:1)
- transport 语义:
  - [serl_launcher/tests/common/test_trainer_transport.py](/home/hello/codebase/serl_torch/serl_launcher/tests/common/test_trainer_transport.py:1)
- config parse:
  - [test/test_trainer_transport_config_parse.py](/home/hello/codebase/serl_torch/test/test_trainer_transport_config_parse.py:1)

### 5.3 额外修正

在真实 benchmark 过程中，还修掉了一个实际问题：

- legacy wrapper 停止时没有显式清理 `agentlace` 的底层 req/rep socket/context
- 这会导致同一进程顺序跑多轮 benchmark 时出现 hang

修正位置：

- [trainer_transport.py](/home/hello/codebase/serl_torch/serl_launcher/serl_launcher/common/trainer_transport.py:1)

## 6. 当前结论

当前实现的结论应当写成下面这版，而不是“提升不明显”：

1. 当前生产实现已经真正接入了 replay `batch_insert()` 和 `split_queue`
2. 当前 copy 训练线最关键的 replay store 路径，已经吃到了真实 vectorized insert 收益
3. 在更接近真实训练负载的 benchmark 下：
   - replay insert 本身约 `6.24x`
   - transport 相关一轮 ingest 总成本约 `13.2x`
4. 之前看起来提升不大的 smoke benchmark，只适合作为连通性测试，不适合作为收益判断依据

## 7. 当前仍然没有覆盖的部分

这轮还没有完成下面两类验证：

1. 完整 LIBERO remote env + policy server 的 e2e 1000-step 对比
2. AgiBot 真机 e2e 验证

因此当前最准确的表述是：

- 生产代码已接通
- targeted tests 已通过
- 当前生产实现 benchmark 已有明显收益
- 但完整外部依赖链路的端到端吞吐，还需要单独做最终验收
