# LIBERO 训练链路已实现优化说明

## 文档目的
这份文档只记录已经在 `serl_torch/examples/libero` 中落地的训练链路优化，不再停留在“建议”层面。

每一项都尽量回答两个问题：
1. 当前原始实现里存在什么问题。
2. 我通过什么方法修复了这个问题。

为了方便后续排查，本文件也会附上主要代码落点、开关配置、使用方式和注意事项。

## 文档范围
本文覆盖的已实现优化包括：
- Env RPC 长连接复用与自动重连
- Actor-Learner 异步解耦
- 同一帧 observation 重复构造的缓存优化
- Replay 采样、`to_torch`、H2D 的预取流水线
- 更细粒度的 profiling 与 profiling 日志

本文不覆盖尚未实现的优化项，例如：
- env server 直接返回 224x224 预处理图像
- RPC payload schema 进一步瘦身
- 更激进的 replay 存储压缩方案

## 变更总览

### 1. Env RPC 每次请求都重新建连

#### 当前存在的问题
原始实现里，`RemoteLiberoTaskEnv._rpc()` 每次调用 `reset/step` 都会：
- 新建一个 `HTTPConnection`
- 发送一次请求
- 立即关闭连接

这会带来两个问题：
- 每个环境步都有额外的建连和断连开销
- env step latency 抖动更大，尤其在高频 `step` 时更明显

#### 我通过什么方法修复了这个问题
我把 RPC 改成了“客户端缓存连接 + 服务端 keep-alive”的方式：
- 在 `RemoteLiberoTaskEnv` 内缓存 `HTTPConnection`
- 新增连接确保与重连逻辑
- 遇到 transport 级错误时自动断开并重连一次
- 服务端切到 HTTP/1.1，并允许 keep-alive

#### 代码落点
- `examples/libero/env_wrappers/remote_task_env.py`
- `examples/libero/scripts/libero_env_server.py`

#### 修复后的行为
- 同一个 env 实例会持续复用连接
- 连接中断后会自动重试一次
- 不再为每次 `reset/step` 都重新握手建连

#### 注意事项
- 这个优化要重启 env server 才会完全生效
- 如果你在跑老的 `serve_env.sh` 启动出来的 server，建议重启一次

---

### 2. Actor 和 learner 完全同步串行

#### 当前存在的问题
原始训练主循环里，采样和学习在同一条临界路径上：

`env/OpenPI/obs构造/采样动作/replay写入/update`

这样会导致：
- actor 必须等待 learner update 完成后才能继续采样
- 单步 wall-clock 直接受到 `env step + preprocess + update` 之和限制
- 当 `utd_ratio` 上升时，在线吞吐会明显下降

#### 我通过什么方法修复了这个问题
我参考 `examples/RoboTwin` 的实现，把 LIBERO 训练脚本接成了线程版 async actor-learner：
- 主线程继续做 actor，负责环境采样和 replay insert
- learner 在后台线程里持续 sample batch 并执行 `update_high_utd(...)`
- actor 和 learner 各自维护一份 agent
- learner 每隔 `training.async.update_frequency` 次更新，把参数同步给 actor
- 保留当前同步模式，作为 fallback

#### 代码落点
- `examples/libero/scripts/train_residual_sac.py`
- `examples/libero/conf/train_residual_sac.yaml`

#### 修复后的行为
- `training.async.enabled=false` 时，仍然走旧的同步路径
- `training.async.enabled=true` 时，主线程不再被 learner update 阻塞
- checkpoint 在异步模式下从 learner 侧保存

#### 关键配置
```yaml
training:
  async:
    enabled: false
    update_frequency: 10
    idle_sleep_sec: 0.002
```

#### 注意事项
- 如果 `training.phases` 里存在 `train=false` 的 phase，脚本会自动禁用 async，避免破坏 phase 语义
- 异步模式下，learner update 的推进速度不再严格等于 actor 的环境步数

---

### 3. 同一帧 observation 在训练中被重复构造

#### 当前存在的问题
原始实现里，同一帧 `obs_raw` 往往会被重复处理多次：
- 给 OpenPI 编码一次
- 给 residual policy 构造 `obs_input` 再做一次
- 为 `next_obs_input` 立即再做一次
- 下一步开始时，同一帧又常常重复构造一次

重复发生的内容包括：
- 图像 `rotate / resize / pad`
- proprio state 拼接
- `base_action` 和 state 的融合

这个问题不仅存在于在线训练，也会出现在：
- offline preload
- bootstrap
- eval

#### 我通过什么方法修复了这个问题
我新增了一套轻量缓存 `LiberoObservationCache`，把 observation 处理拆成三层缓存：
- 图像缓存：缓存预处理后的 `image` 和 `wrist_image`
- state 缓存：缓存原始 state 和 normalizer 后的 state
- step obs 缓存：缓存 `(obs, base_action, image_keys, normalizer)` 对应的最终 `obs_input`

这套缓存同时支持两类 key：
- 在线路径：默认用 `id(obs)`，直接复用同一个 `obs_raw` 对象
- 离线路径：显式传入 `cache_key`，跨 dict 实例复用同一帧结果

#### 代码落点
- `examples/libero/policy/observation.py`
- `examples/libero/policy/openpi_client.py`
- `examples/libero/policy/__init__.py`
- `examples/libero/scripts/train_residual_sac.py`
- `examples/libero/scripts/eval_residual_fast.py`

#### 修复后的行为
- OpenPI 编码和 residual `obs_input` 构造共用同一套缓存
- offline preload、bootstrap、eval 也共用同一套缓存逻辑
- 每个 episode 开始时会清空一次 cache，避免长期堆积无效 key

#### 为什么这样修是安全的
- 图像预处理是纯函数，输入相同就可以复用
- state 构造是纯 observation 变换，也可以复用
- 只有最终 `obs_input` 会把 `base_action` 纳入 key，因此不会错误复用不同动作对应的 state

---

### 4. Replay 采样、`to_torch`、H2D 一直在主路径里同步执行

#### 当前存在的问题
原始训练循环里，batch 更新前的准备工作是同步做的：
- 从 replay 采样 numpy batch
- 递归转成 torch tensor
- 把 batch 搬到 GPU

这部分开销在以下场景会反复叠加：
- `update_every=1`
- `batch_size=128`
- `utd_ratio>=2`
- learner 线程持续高频更新

即使 actor-learner 已经异步化，如果 learner 内部还是“现采现转现搬”，仍然会被固定 CPU/H2D 成本拖住。

#### 我通过什么方法修复了这个问题
我新增了 `_MixedBatchPrefetcher`，把 batch 准备变成后台预取流水线：
- 后台线程提前执行 mixed replay sampling
- 递归执行 `to_torch`
- 对 CPU tensor 执行 `pin_memory`
- 如果目标设备是 CUDA，则用独立 CUDA stream 提前搬到 device
- 主线程或 learner 线程取 batch 时，只需要等待 event 就绪

#### 代码落点
- `examples/libero/scripts/train_residual_sac.py`

#### 修复后的行为
- 异步 learner 模式会自动在 learner 内部使用 prefetcher
- 同步 learner 模式也可以使用轻量 prefetcher
- replay prefetch 支持 online/offline mixed batch，不破坏现有 replay 语义

#### 关键配置
```yaml
training:
  replay_prefetch:
    enabled: true
    queue_size: 2
    pin_memory: true
    to_device: true
```

#### 补充说明
- `queue_size=2` 是一个保守默认值，避免预取队列太深造成无意义的显存/内存占用
- 如果机器没有 CUDA，`to_device` 逻辑会自动退化
- 预取队列深度会额外打到 TensorBoard 和 profiling 日志中，方便观察是否出现“取 batch 跟不上”或“队列一直满”的情况

---

### 5. 缺少更细的 profiling，容易误判真正瓶颈

#### 当前存在的问题
此前训练日志里直接能看到的主要是：
- step 结果
- OpenPI 推理耗时
- episode 结果

但真正关键的训练阶段并没有统一埋点，例如：
- `env.reset`
- `env.step`
- `build_residual_step_obs`
- `agent.sample_actions`
- `agent.update_high_utd`
- `replay sample / to_torch / pin_memory / H2D`
- `checkpoint save`

结果就是：
- 很容易把优化精力花在“看起来慢”的环节上
- 很难直接判断最慢的是 env、预处理、采样、更新还是 checkpoint

#### 我通过什么方法修复了这个问题
我新增了 `_RuntimeProfiler`，作为线程安全的滚动窗口 profiler：
- 支持 duration 类指标和 scalar 类指标
- 保留窗口统计和累计统计
- 周期性输出 JSONL profiling 日志
- 同步输出到 TensorBoard
- 最终把 profiling 快照写进 `summary.json`

另外，我还把 checkpoint profiling 单独做成了包装：
- 记录保存耗时
- 记录当前 checkpoint 文件大小
- 记录 checkpoint 目录总大小
- 记录本次保存带来的目录增量大小

#### 当前 profiling 覆盖的指标
duration 类：
- `env_reset`
- `env_step`
- `build_residual_step_obs`
- `agent_sample_actions`
- `agent_update_high_utd`
- `replay_sample`
- `replay_to_torch`
- `replay_pin_memory`
- `replay_h2d`
- `replay_prepare`
- `checkpoint_save`

value 类：
- `checkpoint_size_mb`
- `checkpoint_dir_size_mb`
- `checkpoint_dir_delta_mb`

#### 代码落点
- `examples/libero/scripts/train_residual_sac.py`

#### 日志输出形式
1. JSONL 文件  
   默认文件名：
   - `profiling_logs.jsonl`

2. TensorBoard  
   路径前缀：
   - `profiling/<metric>/mean_ms`
   - `profiling/<metric>/p95_ms`
   - `profiling/<metric>/max_ms`

3. `summary.json`  
   最终会把 profiling 快照写进 summary 的 `profiling` 字段

#### 关键配置
```yaml
training:
  profiling:
    enabled: false
    window_size: 2048
    log_period_steps: 500
    log_file: profiling_logs.jsonl
```

#### 推荐使用方式
如果想先定位瓶颈，再决定下一步优化，建议先打开 profiling：

```bash
bash tools/train.sh training.profiling.enabled=true
```

如果想更密集地观察短时间内的抖动，可以把日志周期调小：

```bash
bash tools/train.sh \
  training.profiling.enabled=true \
  training.profiling.log_period_steps=200
```

---

## 配置总览

目前与这些优化直接相关的配置主要有：

```yaml
training:
  async:
    enabled: false
    update_frequency: 10
    idle_sleep_sec: 0.002

  replay_prefetch:
    enabled: true
    queue_size: 2
    pin_memory: true
    to_device: true

  profiling:
    enabled: false
    window_size: 2048
    log_period_steps: 500
    log_file: profiling_logs.jsonl
```

## 代码文件汇总
这几次优化主要涉及以下文件：
- `examples/libero/env_wrappers/remote_task_env.py`
- `examples/libero/scripts/libero_env_server.py`
- `examples/libero/policy/observation.py`
- `examples/libero/policy/openpi_client.py`
- `examples/libero/policy/__init__.py`
- `examples/libero/scripts/train_residual_sac.py`
- `examples/libero/scripts/eval_residual_fast.py`
- `examples/libero/conf/train_residual_sac.yaml`

## 验证情况
针对这些改动，已经做过的验证包括：
- `py_compile` 语法检查
- Env RPC 长连接/断线重连的最小 fake server 自测
- async learner 的 smoke test
- observation cache + replay prefetch 的 smoke test
- profiling 的 smoke test

说明：
- 上述验证主要用于确认代码路径和基础行为正确
- 还没有把所有改动在完整 LIBERO 长时间训练上做系统性的 A/B benchmark

## 当前建议的使用顺序
如果后续要继续做训练提速，建议按下面顺序推进：
1. 先打开 `training.profiling.enabled=true`，收一轮真实训练日志。
2. 观察 `env_step`、`build_residual_step_obs`、`agent_update_high_utd`、`replay_h2d`、`checkpoint_save` 谁最慢。
3. 如果 learner 明显拖后腿，优先开 async actor-learner。
4. 如果 replay 准备时间明显高，继续调 `replay_prefetch.queue_size` 和 `pin_memory/to_device`。
5. 如果 env step 仍然占主导，再继续往 env server 侧做 schema 压缩或服务端图像预处理。

## 结论
这几次已经落地的修改，核心思路不是单点暴力提速，而是把训练主链上的几个典型固定成本逐步拆掉：
- 去掉 env RPC 的重复建连成本
- 去掉 actor 被 learner 完全阻塞的串行结构
- 去掉 observation 的重复构造
- 去掉 replay sample 和 H2D 的现取现搬
- 补上足够细的 profiling，让后续优化有数据支撑

换句话说，当前代码相比原始实现，已经不再是“所有环节都同步串在一起、又看不见哪一段最慢”的状态，而是开始具备了可观测、可复用、可继续迭代优化的基础。
