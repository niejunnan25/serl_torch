# LIBERO 残差 RL 训练链路优化笔记

## 适用范围
这份文档聚焦于 `serl_torch/examples/libero` 当前训练链路的运行效率，目标是在不改变任务语义的前提下，尽量缩短 wall-clock 训练时间。

文档包含三部分内容：
- 当前训练流程是如何串起来的；
- 当前代码里已经能看到的主要瓶颈在哪里；
- 除了已有优化项之外，还可以从哪些地方继续提速。

## 从 `serve_env.sh` 开始的当前训练流程
`tools/serve_env.sh` 本身只是一个启动脚本，它做的事情很少：
1. 切到 `examples/libero` 根目录。
2. 尝试激活 LIBERO 对应 conda 环境。
3. 选择 Python 3。
4. 最终执行 `scripts/libero_env_server.py`。

真正的训练时序如下：
1. `tools/run_train.sh` 启动 LIBERO env server。
2. `tools/run_train.sh` 启动 OpenPI server。
3. `tools/train.sh` 启动 `scripts/train_residual_sac.py`。
4. 训练进程通过 `RemoteLiberoTaskEnv` 用 HTTP RPC 调 `reset/step`。
5. 在 replan 点，通过 `OpenPIChunkClient` 用 websocket 请求 base policy action chunk。
6. 每个环境步里，训练进程会做：
   - 构造残差策略输入观测；
   - 采样 residual action；
   - 合成最终动作；
   - 调远程 env `step`；
   - 写 replay；
   - 满足条件时立刻做 `agent.update_high_utd(...)`；
   - 写 step 日志、TensorBoard、可选 checkpoint。
7. 如果开启 async eval，还会再启动一个 watcher，并周期性拉起评测进程。

当前关键文件：
- `examples/libero/tools/serve_env.sh`
- `examples/libero/scripts/libero_env_server.py`
- `examples/libero/env_wrappers/remote_task_env.py`
- `examples/libero/env_wrappers/task_env.py`
- `examples/libero/policy/openpi_client.py`
- `examples/libero/policy/observation.py`
- `examples/libero/scripts/train_residual_sac.py`
- `examples/libero/scripts/async_eval_watch.py`
- `examples/libero/tools/run_train.sh`

## 当前瓶颈判断

### 1. `serve_env.sh` 不是瓶颈本体，瓶颈在它后面的串行主链路
`serve_env.sh` 只是把环境服务拉起来。真正影响吞吐的是训练主循环里的这条串行路径：

`env RPC -> 图像预处理 -> OpenPI replan -> residual policy sample -> env step -> replay insert/sample -> learner update -> logging/checkpoint`

只要其中任意一环慢，整个单 actor 训练吞吐都会被锁住。

### 2. 从现有运行日志看，OpenPI 有开销，但不像是唯一主瓶颈
已有日志显示：
- 某次 `pld_matrix/task2/m06_boot50_w100_xi05` 运行，从 `2026-03-20 00:53` 到 `2026-03-20 13:12`，一共推进了大约 `4.83 万` 个环境步，整体只有约 `1.14 step/s`。
- 同一份 `step_logs.jsonl` 中，OpenPI replan 的 `infer_e2e_ms` 平均约 `77.72 ms`，`p95` 约 `100.98 ms`。
- 在 `chunk_horizon=5` 下，OpenPI 平摊到每个环境步只有约 `15.54 ms/step`。

这说明：
- OpenPI websocket 往返确实有成本；
- 但如果总吞吐只有 `~1 step/s`，那么更大的时间很可能还花在 env step、重复预处理、learner 更新、replay 采样搬运、日志和 checkpoint 上。

### 3. 启动阶段已经有明显 wall-clock 开销
另一条完整运行日志显示：
- `offline preload` 从 `16:33:08` 到 `16:33:44`，约 `36 秒`；
- `Cal-QL critic pretrain` 从 `16:33:44` 到 `16:46:56`，约 `13 分 12 秒`；
- 也就是说，训练刚启动到真正进入 online phase 之前，就已经消耗了接近 `14 分钟`。

如果关注的是“总训练时间”而不是“纯在线阶段时间”，那么离线加载和 critic warm start 也是当前非常实在的瓶颈。

### 4. 在线 replay 的图像内存占用非常重
当前训练脚本使用的是普通 `ReplayBuffer`，会同时存：
- `observations`
- `next_observations`

而默认观测里又包含两路 `224x224x3` 图像。按默认配置粗算：
- online replay `capacity=250000` 时，仅图像部分的 `obs + next_obs` 原始存储量约 `140 GB`；
- offline replay `capacity=50000` 时，仅图像部分也在 `28 GB` 量级。

这会带来几个问题：
- 内存压力大；
- 采样时 CPU cache 命中差；
- numpy 到 torch 到 GPU 的搬运成本上升。

### 5. 当前默认把 OpenPI 和 trainer 放在同一张 GPU 上
`tools/run_train.sh` 里，OpenPI 服务和训练进程都使用同一个 `GPU_ID`：
- `serve_openpi.sh` 里通过 `CUDA_VISIBLE_DEVICES="$GPU_ID"` 启动 OpenPI；
- 训练本身也通过 `CUDA_VISIBLE_DEVICES="$GPU_ID"` 启动。

这意味着当前默认是：
- JAX/XLA 的 OpenPI 服务；
- PyTorch 的 residual learner；

一起争用同一张 GPU。哪怕 OpenPI 单次推理不长，这种资源争用也会放大抖动，尤其在 `update_every=1`、`utd_ratio>=2`、chunk 又不大的时候更明显。

### 6. 当前缺少更细的阶段性耗时指标
目前日志里直接记录得比较完整的是：
- OpenPI 推理时间；
- step/episode 结果；

但对以下耗时没有直接埋点：
- env reset latency
- env step RPC latency
- `build_residual_step_obs(...)` 耗时
- replay sample + host-to-device copy 耗时
- `agent.update_high_utd(...)` 耗时
- checkpoint save 耗时

这会让“定位主瓶颈”变慢。现在能做的是通过代码路径和已有日志推断，但更稳妥的做法是直接补 profiling 指标。

## 已有优化项

### 1. Env RPC 每次请求都重新建连
当前行为：
- `RemoteLiberoTaskEnv._rpc()` 每次都会新建 `HTTPConnection`，请求后立刻关闭。

影响：
- 每个 env `reset/step` 都有额外建连/断连开销；
- step loop 延迟抖动变大。

优化方向：
- 复用长连接，并在失败时自动重连一次。

实现建议：
1. 在 `RemoteLiberoTaskEnv` 中缓存 connection 对象。
2. 增加 `_ensure_conn()` 和 `_reconnect()`。
3. 使用 keep-alive，请求结束后不要立即关闭。
4. 遇到 broken pipe / timeout 时重连并重试一次。

### 2. RPC payload 偏重
当前行为：
- 每次 `reset/step` 都返回完整 observation dict，并用 pickle 序列化。

影响：
- 序列化和内存拷贝成本高；
- 即便是 localhost，也会带来额外 CPU 负担。

优化方向：
- 缩小 observation schema，先减 payload，再考虑换序列化协议。

实现建议：
1. 只返回训练真正需要的图像和 proprio 字段。
2. 保留必要 metadata，去掉冗余字段。
3. 先做 schema 压缩；若仍然偏重，再考虑 `msgpack` 或轻量压缩。

### 3. 图像预处理反复在 client 侧执行
当前行为：
- `extract_residual_images(...)` 每次都会做 rotate / resize / pad。

影响：
- actor loop 中有稳定的 CPU 开销；
- 每步至少要处理两路图像。

优化方向：
- 让 env server 直接返回训练所需尺寸和布局的图像。

实现建议：
1. 在 env server 端增加“直接输出 224x224 预处理图像”的模式。
2. 保留开关，方便 A/B 比较。
3. client 收到预处理图像后直接跳过 PIL 路径。

### 4. Actor 和 learner 仍然是同步串行
当前行为：
- 环境采样和梯度更新在同一个临界路径里。

影响：
- 总吞吐被 `env step + preprocess + OpenPI + update` 的总和限制。

优化方向：
- 做 actor-learner 解耦。

实现建议：
1. actor 进程只负责采样并写入队列 / replay。
2. learner 进程持续消费 batch 更新。
3. 周期性同步 policy 权重给 actor。
4. 保留当前同步模式作为 fallback。

说明：
- 这是工程量最大的方向，但通常也是 wall-clock 提速空间最大的方向。

### 5. OpenPI 推理频率仍有优化空间
当前行为：
- 每个 replan 点都要走一次 websocket 推理。

影响：
- 若 `chunk_horizon` 偏小，往返频率高。

优化方向：
- 在任务允许的前提下，适当增大 `chunk_horizon`。

实现建议：
1. 保守调大 `residual.chunk_horizon`，重新评估成功率。
2. 若 chunk 稳定，可考虑下一个 chunk 的预取或缓存。

### 6. Async eval 可能和训练争抢资源
当前行为：
- async eval 可以复用训练中的 OpenPI 端口。

影响：
- 训练和评测会争用推理服务和 GPU 时间。

优化方向：
- 评测隔离资源，或降低评测压力。

实现建议：
1. OpenPI 最好为 eval 单独部署。
2. 调整 `training.async_eval.every_steps`。
3. 在只关心训练吞吐时，优先用 `queue_policy=latest`。
4. 评测 env 端口与训练 env 端口保持隔离。

### 7. 高频日志会放大 I/O 开销
当前行为：
- step jsonl 和 HTTP access log 都比较频繁。

影响：
- 文件写入、flush、锁竞争都会吃时间。

优化方向：
- 降低高频日志负担，只保留必要指标。

实现建议：
1. 降低 env server 每请求 INFO 日志。
2. step log 改成可选精简字段模式。
3. 适当增大指标聚合周期。

## 继续补充的可加速点

### 8. OpenPI 和训练默认同 GPU，建议优先拆开
当前行为：
- `run_train.sh` 默认让 OpenPI server 和训练进程共用同一张 GPU。

影响：
- JAX/XLA 与 PyTorch 共用同卡，会带来显存和算力争用；
- 推理和训练互相打断，抖动会被放大。

优化方向：
- 把 OpenPI 服务和 trainer 放到不同 GPU；如果卡不够，至少限制 OpenPI 显存比例并单独压测。

实现建议：
1. 给 `run_train.sh` 增加单独的 `--openpi-gpu-id`。
2. 允许训练和 OpenPI 分别设置 `CUDA_VISIBLE_DEVICES`。
3. 若必须同卡，调小 `XLA_PYTHON_CLIENT_MEM_FRACTION`，并确认 trainer 没有频繁 OOM/reclaim。
4. 若机器资源允许，OpenPI 直接部署到另一台节点上。

### 9. 同一帧 observation 在训练中被重复构造
当前行为：
- 当前 step 里会构造一次 `obs_input`；
- 为了生成 `next_observations`，又会立刻构造一次 `next_obs_input`；
- 下一步开始时，这个 `next_obs_input` 对应的同一帧通常又会再构造一次。

影响：
- 同一帧图像的 resize/pad、state 拼接、base_action 拼接会重复做。
- 在训练、bootstrap、offline preload、eval 中都会重复发生。

优化方向：
- 在 trainer 侧先做轻量缓存，再决定是否下沉到 env server。

实现建议：
1. 把“图像预处理”和“state/base_action 融合”拆开。
2. 缓存 `(obs_raw, base_action)` 对应的 `obs_input`。
3. 在 chunk 内直接复用已经算好的 `next_obs_input`。
4. offline preload / bootstrap / eval 也共用同一套缓存逻辑。

### 10. Replay 采样和 host-to-device 搬运可以继续优化
当前行为：
- 训练主循环每次都同步执行：
  - numpy replay `sample()`
  - 递归转 torch
  - 搬到 GPU
- 虽然 replay 已经提供了 `get_iterator()` 预取接口，但训练主循环目前没有使用。

影响：
- `update_every=1`、`batch_size=128`、`utd_ratio>=2` 时，这部分固定成本会持续叠加。

优化方向：
- 做 batch 预取、pin memory、异步 host-to-device copy。

实现建议：
1. 用 replay 的 iterator 提前准备 batch。
2. 对图像 batch 使用 pinned memory。
3. 让采样和 GPU 更新形成轻量流水线。
4. 如果后续做 actor-learner 分离，这部分直接放到 learner 进程内做。

### 11. 建议切换到更省内存的图像 replay
当前行为：
- 当前使用普通 `ReplayBuffer`，完整保存 `obs` 和 `next_obs`。

影响：
- 图像数据重复存储严重；
- 采样成本和内存压力都偏高。

优化方向：
- 使用已有的 `MemoryEfficientReplayBuffer`，至少先对像素观测去重。

实现建议：
1. 在训练脚本中把图像 key 接到 `MemoryEfficientReplayBuffer`。
2. 确认 `obs_stack_horizon=1` 的情况下行为一致。
3. 先对 online replay 切换，再评估是否把 offline buffer 也做成内存高效版本。

### 12. 离线数据启动成本还可以继续压
当前行为：
- 若离线 PKL 中没有 `base_chunks`，训练启动时会重新调用 OpenPI；
- `Cal-QL critic pretrain` 目前是训练启动前的串行步骤。

影响：
- “开始训练”之前就会先消耗大量 wall-clock 时间。

优化方向：
- 把能离线做的都提前离线做，把能延后的都延后。

实现建议：
1. 把 `convert_offline.sh --openpi` 从“可选”提升为“推荐默认”。
2. 尽量确保 offline PKL 里已经带 `base_chunks`。
3. 对 `training.calql_pretrain.steps` 做任务级网格压缩，避免固定 2000 步一刀切。
4. 如果主要关心尽快进入 online RL，可考虑减少 critic pretrain 步数，或改成后台/分段 warm start。

### 13. Checkpoint 目前是同步写盘，且容易积累很大目录
当前行为：
- checkpoint 在训练主循环里同步 `torch.save(...)`；
- `keep_checkpoints=0` 时不会自动清理旧 checkpoint。

影响：
- 保存 checkpoint 时会打断主循环；
- 大量写盘会拖慢训练；
- 已有运行中，`100k` env steps 的 checkpoint 目录接近 `10 GB`。

优化方向：
- 减少同步写盘阻塞，限制 checkpoint 数量。

实现建议：
1. 改成后台线程/子进程异步保存 checkpoint。
2. 如果只做 async eval，保留最近 `k` 个和评测触发点附近的 checkpoint 即可。
3. 适当增大 `checkpoint_period`。
4. 如果只需要恢复 latest，可加一个 rolling checkpoint。

### 14. Reset 阶段的 `num_steps_wait` 也值得压测
当前行为：
- 每次 reset 后，都会额外执行 `num_steps_wait` 次 dummy step，默认是 `10`。

影响：
- 对短 episode 来说是额外固定成本；
- 若 episode 长度集中在 `250~520` 步，这部分通常有 `2%~4%` 的额外开销。

优化方向：
- 对不同任务单独验证 `num_steps_wait` 的必要值，而不是固定用 10。

实现建议：
1. 测试 `10 -> 5 -> 3 -> 0` 对成功率和稳定性的影响。
2. 如果只是为了等物理稳定，可考虑改成 reset 后一次性 settle，而不是固定多步 dummy action。

### 15. 建议先补更细的 profiling，再做针对性提速
当前行为：
- 目前缺少 env / preprocess / update / checkpoint 的直接耗时指标。

影响：
- 容易把优化精力放在“看起来慢”的环节，而不是“真正最慢”的环节。

优化方向：
- 先补阶段性耗时埋点，再做 A/B。

实现建议：
1. 记录 `env.reset` / `env.step` 的 mean / p95。
2. 记录 `build_residual_step_obs` 的 mean / p95。
3. 记录 `agent.sample_actions` 和 `agent.update_high_utd` 的 mean / p95。
4. 记录 replay sample + to_torch + H2D copy 耗时。
5. 记录 checkpoint save 耗时和大小。

## 建议优先级

### Phase A：低风险、见效快
1. 关闭或降低 env server 的每请求 INFO 日志。
2. step logger 改成批量 flush，或增加精简字段模式。
3. 保持 env RPC 长连接，并加失败重连。
4. 把 OpenPI server 和 trainer 拆到不同 GPU。
5. 训练端增加 `obs_input` / 图像预处理缓存。
6. async eval 使用独立 env 端口，并尽量独立 OpenPI 服务。
7. 先补 env/preprocess/update/checkpoint 的耗时埋点。

### Phase B：中等风险、收益明显
1. 压缩 env RPC payload。
2. 引入 `MemoryEfficientReplayBuffer`。
3. 给 replay sample 加预取和 pinned memory。
4. 使用 env server 直接输出预处理后的 `224x224` 图像。
5. 把 `convert_offline.sh --openpi` 变成推荐默认流程。
6. 重新压测 `num_steps_wait`、`checkpoint_period`、`calql_pretrain.steps`。

### Phase C：结构性改造
1. Actor-learner 解耦。
2. 让 logging / checkpoint 都异步化。
3. 如果需要更大吞吐，扩展为多 actor + 单 learner。
4. 保留当前同步版本作为回滚路径。

## 验证清单
每做完一个优化阶段，都建议记录以下指标做 A/B：
1. 训练启动到“第一个 online episode 开始”的 wall-clock 时间。
2. 每秒环境步数（env steps/s）。
3. 每秒策略步数（policy steps/s）。
4. OpenPI infer latency 的 mean / p95。
5. env RPC latency 的 mean / p95。
6. `build_residual_step_obs` 耗时的 mean / p95。
7. replay sample + to_torch + H2D copy 耗时。
8. learner `update_high_utd` 耗时。
9. checkpoint save 耗时与 checkpoint 目录增长速度。
10. 训练成功率相对于 wall-clock 的提升曲线。
11. async eval 的排队延迟和完成滞后。

建议：
- 固定随机种子；
- 使用同一任务配置；
- 一次只改一个方向，避免多个优化混在一起难以归因。

## 简短结论
当前链路里，最值得优先怀疑的并不是 `serve_env.sh` 本身，而是：
- 在线主循环的串行同步结构；
- 重复 observation 预处理；
- learner 更新与 replay 搬运；
- OpenPI 与 trainer 同卡争用；
- 高频日志与同步 checkpoint；
- 以及启动阶段的离线 warm start。

如果只想先拿一轮低风险提速，最建议优先做的是：
1. OpenPI 和 trainer 分 GPU。
2. env RPC 长连接。
3. step 日志批量 flush / 精简字段。
4. trainer 侧缓存 `next_obs_input` 和图像预处理结果。
5. 引入更细的耗时埋点，确认 env / preprocess / update 三者谁最慢。
