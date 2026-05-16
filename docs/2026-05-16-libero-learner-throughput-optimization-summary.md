# LIBERO learner 吞吐优化总结

日期：2026-05-16

当前分支：

```text
refactor/agibot-transition-dataflow
```

相关提交：

```text
c61cfab Optimize residual replay sampling
d4870e4 Optimize visual encoder and target updates
```

## 结论

这一路优化后，当前 learner 的瓶颈已经从“采样和系统开销明显拖慢模型”转成“模型更新本身占主导”。用 `spatial4_chunk_alpha0p1_unfiltered_offline_noent_std0p5` 这条配置模拟完整 learner 外层循环时，当前代码在 GPU5 上测到：

```text
3.99 updates/s
0.250 s / update
```

同一个输出目录里，之前真实训练后期稳定段约为：

```text
2.08 updates/s
0.481 s / update
```

按这个口径比较，当前模拟 learner 比之前稳定日志快约 `92%`，单次完整 learner update 耗时下降约 `48%`。

需要注意：这次模拟没有启动 actor/env、wandb、async eval 和 checkpoint；online/offline replay 都从同一份 prepared offline 数据加载，大小是 7479 windows，而历史真实训练后期 online replay 是 250000。所以下面的数字适合判断 learner 更新链路本身，不等同于完整分布式训练全系统吞吐。

## 当前端到端 learner 模拟

脚本：

```text
test/benchmark_libero_real_replay_update.py
```

命令：

```bash
source /vla/miniconda3/etc/profile.d/conda.sh
conda activate serl_torch

CUDA_VISIBLE_DEVICES=5 python test/benchmark_libero_real_replay_update.py \
  --config-name spatial_4_0514_runtime/spatial4_chunk_alpha0p1_unfiltered_offline_noent_std0p5_ports53500 \
  --mode all \
  --updates 30 \
  --warmup 8 \
  --capacity 10000 \
  --offline-capacity 10000 \
  --json-output test/results/libero_real_replay_update_current_e2e_20260516.json
```

测试条件：

| 项目 | 值 |
|---|---:|
| GPU | NVIDIA H20，`CUDA_VISIBLE_DEVICES=5` |
| batch size | 128 |
| online/offline ratio | 64 / 64 |
| critic_actor_ratio | 2 |
| utd_ratio | 4 |
| 图像 | 两路 224x224 RGB |
| encoder | HF ResNet18，frozen backbone |
| mixed precision | bf16 |
| torch.compile | enabled，actor_critic，inductor，fullgraph |
| fuse_views | auto |
| online replay | 7479 windows |
| offline replay | 7479 windows |

结果：

| 模式 | updates/s | update 总耗时 | 采样等待 | critic-only 更新 | high-UTD 更新 |
|---|---:|---:|---:|---:|---:|
| stage1 兼容路径 | 3.26 | 0.307s | 0.032s | 0.071s | 0.205s |
| stage2 packed + device concat，串行 | 2.71 | 0.369s | 0.083s | 0.063s | 0.224s |
| stage2 packed + device concat + prefetch | 3.99 | 0.250s | 0.026s | 0.068s | 0.156s |

这里最重要的是 stage2 串行和 stage2 prefetch 的对比，因为真实训练脚本现在走的是 prefetch 形态：

| 对比 | 提升 |
|---|---:|
| updates/s：2.71 -> 3.99 | +47.5% |
| update 总耗时：0.369s -> 0.250s | -32.2% |
| 采样等待：0.083s -> 0.026s | -68.7% |

## 对比之前真实训练日志

历史输出目录：

```text
examples/libero/outputs/spatial_4_0514/spatial4_chunk_alpha0p1_unfiltered_offline_noent_std0p5
```

统计文件：

```text
learner/learner_timers.jsonl
```

后 200 条 heartbeat 的稳定统计：

| 指标 | 之前真实训练稳定段 | 当前模拟 learner | 变化 |
|---|---:|---:|---:|
| updates/s | 2.079 | 3.993 | +92.1% |
| update 总耗时 | 0.481s | 0.250s | -47.9% |
| sample_replay_buffer | 0.066s | 0.026s | -60.7% |
| train_critics | 0.110s | 0.068s | -38.2% |
| train | 0.295s | 0.156s | -47.0% |

历史日志全段平均是 `1.64 updates/s`，但里面混有早期编译、replay 变小、GPU 繁忙、checkpoint/eval 等阶段；更公平的参考是后 100 或后 200 条稳定 heartbeat，约 `2.08 updates/s`。

## 已落地优化

### 1. critic ensemble 共享视觉编码

位置：

```text
serl_launcher/serl_launcher/networks/actor_critic_nets.py
```

之前两个 Q head 会各自调用 critic，每个 Q head 都可能重新跑一遍视觉 encoder。现在如果 ensemble 内的 critic 共享同一个 encoder，就先编码 observation 一次，然后把 `obs_enc` 喂给每个 Q head。

这个优化不改变 Q head 的参数，也不改变 loss；它只是避免重复做相同的视觉编码。

效果：这是模型更新时间下降的主要来源之一。真实历史稳定段里 `train_critics≈0.110s`、`train≈0.295s`；当前模拟里分别是 `0.068s` 和 `0.156s`。

### 2. 去掉每步指标 `.cpu()` 同步

位置：

```text
serl_launcher/serl_launcher/agents/continuous/sac.py
```

之前 loss info 里很多字段会在每个 update 内执行：

```python
float(tensor.detach().cpu())
```

这会强迫 CPU 等 GPU。现在改成保留 detached tensor，到日志序列化时再转成 Python 数值。

效果：它不是最大瓶颈，但能减少 high-UTD 下的同步点。收益更像稳定性和调度开销改善，不会像 encoder 共享或 prefetch 那样单项巨大。

### 3. step-window replay 采样向量化

位置：

```text
serl_launcher/serl_launcher/data/step_window_replay_buffer.py
serl_launcher/serl_launcher/data/data_store.py
serl_launcher/serl_launcher/residual/chunk_window_replay.py
```

这部分把 step-window sample 从大量 Python 小循环，改成批量索引、批量窗口 metadata、批量 transition 构造。还支持 `pack_obs_and_next_obs`，减少 observation/next_observation 图像重复搬运和拼接。

效果：历史训练后期 `sample_replay_buffer≈0.066s`；当前 prefetch 模拟里外层循环实际等待采样只剩 `0.026s`。注意这里的 `0.026s` 是“等待时间”，不是后台线程真实采样总时间；真实采样被训练计算隐藏掉了。

### 4. online/offline mixed batch device concat

位置：

```text
serl_launcher/serl_launcher/residual/chunk_window_replay.py
```

现在 `sample_mixed_training_batch(..., device=agent.device, prefer_device_concat=True)` 可以让 online/offline 子 batch 更早转到 GPU，再做 device-side concat，避免 CPU 侧大树拼接之后再整批搬运。

效果：在纯串行 stage2 benchmark 中，采样和拼接仍然要花约 `0.083s/update`；配合 prefetch 后，外层循环只等待约 `0.026s/update`。

### 5. replay batch 预取

位置：

```text
serl_launcher/serl_launcher/residual/chunk_window_replay.py
examples/libero/scripts/train_residual_chunk.py
```

现在 learner 在训练当前 batch 时，后台线程提前准备下一批 replay batch。算法语义不变：batch 仍然来自同一个 replay 分布，只是采样和训练从串行变成部分重叠。

当前端到端模拟中，这是最明确的大收益：

| 模式 | updates/s | update 总耗时 |
|---|---:|---:|
| stage2 串行 | 2.71 | 0.369s |
| stage2 prefetch | 3.99 | 0.250s |

也就是 `+47.5% updates/s`。

### 6. ResNet mean/std buffer 与 fused-view fast path

位置：

```text
serl_launcher/serl_launcher/vision/resnet_v1.py
serl_launcher/serl_launcher/common/encoding.py
examples/libero/config.py
examples/agibot_real/config.py
```

ResNet 的 ImageNet mean/std 现在注册成 buffer，不再每次 forward 创建 tensor。两路图像在安全条件满足时可以 fuse 成一个 `B*V` batch，共享 frozen ResNet backbone 一次前向，再拆回各自的 pooling/head。

安全条件：

- 至少两个 image key；
- encoder 是 `ResNetEncoder`；
- `freeze_backbone=true`；
- 多个 view 共享同一个 backbone；
- 图像 shape/device/dtype 一致。

`encoder.fuse_views=auto` 时，条件满足就启用；不满足自动回退。`true` 会强制启用，不满足时报错；`false` 用旧路径。

单独图像路径 benchmark：

| 图像路径 | mean ms / step | 相对 baseline |
|---|---:|---:|
| baseline_current | 29.82 | 基线 |
| mean/std buffer | 29.55 | +0.9% |
| NCHW layout | 29.99 | -0.6% |
| fused backbone | 29.42 | +1.3% |
| fused + compile | 18.93 | +36.5% |

迁移进真实 production wrapper 后，因为当前 actor/critic 本来已经 `torch.compile(fullgraph=true)`，新增收益变小：

| production wrapper | mean ms / step |
|---|---:|
| `fuse_views=false` + compile | 22.29 |
| `fuse_views=auto` + compile | 22.05 |

所以它是一个安全的结构性优化，但不是当前端到端收益的主来源。

### 7. target network soft update 跳过 frozen 参数

位置：

```text
serl_launcher/serl_launcher/common/common.py
```

之前 target update 会对 frozen ResNet18 backbone 参数也做：

```python
target = (1 - tau) * target + tau * source
```

现在 `TorchRLTrainState.target_update` 会跳过 source 和 target 都 `requires_grad=False` 的参数；trainable 参数仍然照常更新。普通 `soft_update` 保持旧语义。

target update microbenchmark：

| 策略 | mean us | 相对当前旧 loop |
|---|---:|---:|
| 全量参数 loop | 1707 | 基线 |
| 跳过 frozen loop | 983 | +42.4% |
| 跳过 frozen + foreach | 557 | +67.4% |

这个优化对完整 update 的占比不大，但语义很干净。

### 8. high-UTD 编译实验没有继续推进

我们专门测了 high-UTD 的三档编译策略：

| 策略 | mean ms / pattern | 相对当前模块级 compile |
|---|---:|---:|
| eager Python | 115.23 | 慢 69.4% |
| 当前模块级 compile | 68.01 | 基线 |
| loss 级 compile | 67.52 | +0.7% |
| step 级 compile | 66.52 | +2.2% |
| CUDA graph | 未通过 | 不可用 |

结论是：当前正式训练已经在 actor/critic 模块级启用 `torch.compile`，继续把 high-UTD Python loop 往外包一层，短期新增收益很小；CUDA graph 当前兼容性不好，暂时不适合作为主线。

## ResNet10 vs ResNet18 的定位

JAX ResNet10 环境修好之后，我们做了公平 encoder benchmark：

| 路径 | frozen 前向 | frozen 前向 + head 反向 | 全量前向 + 反向 |
|---|---:|---:|---:|
| JAX ResNet10 | 16.14 ms | 17.52 ms | 46.82 ms |
| Torch ResNet18 未编译 | 31.43 ms | 32.11 ms | 98.38 ms |
| Torch ResNet18 编译后 | 21.52 ms | 22.20 ms | 68.90 ms |

ResNet18 的确更慢，但编译后和 ResNet10 的差距不是数量级差距。SERL/JAX 更快还来自 JIT + `lax.scan` 把 high-UTD 更新融合成更粗的图。

## 当前判断

现在最有价值的收益已经来自三件事：

1. critic ensemble 共享 encoder；
2. replay 采样和 batch 构造向量化；
3. replay batch prefetch 把采样隐藏到 GPU 训练后面。

剩余还能做的方向主要是：

- 更深的 CPU/GPU overlap：pinned memory + 独立 CUDA stream + event；
- 减少 high-UTD Python optimizer 调度，但这比想象中收益小，风险更高；
- 更换轻量 encoder 或对齐 ResNet10，但这会改变模型能力；
- 进一步减少非日志步的 GPU 指标 reduction。

如果目标是追 SERL/JAX 后期约 `4 updates/s`，当前模拟 learner 已经达到 `3.99 updates/s`。如果目标是在完整真实训练中稳定达到这个数，还需要把 actor/env、checkpoint、async eval、真实 250k online replay 下的系统抖动也一起压住。
