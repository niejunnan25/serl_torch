# Learner Speed Ablation: bf16 And Actor-Update Critic Freeze

## 文档信息

- 文档类型：learner 梯度更新速度消融实验记录
- 记录时间：北京时间 2026-04-16
- 实验目标：
  验证两个低侵入优化对 learner 梯度更新速度的实际提升。
- 关联优化分析文档：
  `docs/2026-04-16-learner-gradient-update-speed-optimization.md`
- 关联 benchmark 脚本：
  `test/benchmark_learner_update_speed.py`
- 关联代码修复：
  `serl_launcher/serl_launcher/agents/continuous/sac.py`

## 1. 本轮验证的优化点

本轮只验证两个优化点：

1. 开启 bf16 mixed precision
2. 修 actor update 阶段“白算 critic 参数梯度”的问题

这两个优化点的共同特点是：

- 不改变 replay 数据结构
- 不改变 actor / learner 通信协议
- 不改变 `critic_actor_ratio` 和 `utd_ratio`
- 不减少相机数量、图像分辨率、batch size 或模型层数
- 理论上对训练语义影响很小

其中第 2 点的核心是：actor loss 需要经过 critic 计算 `Q(s, pi(s))`，并且需要 `dQ/da` 传回 actor；但 actor optimizer 不会 step critic 参数。所以 actor update 时不应该给 critic 参数计算梯度。

修复方式是：

```text
actor update 阶段临时 requires_grad_(False) 冻结 critic 参数
保留 action tensor 到 Q 的梯度路径
actor backward 后恢复 critic 参数原始 requires_grad 状态
```

注意：这里不能用 `torch.no_grad()` 包住 critic forward。那样会连 `dQ/da` 也切断，actor 就拿不到正确的策略梯度。

## 2. Benchmark 方法

新增脚本：

```text
test/benchmark_learner_update_speed.py
```

这个脚本做的事情：

1. 加载真实 AgiBot learner 配置：
   `examples/agibot_real/configs/train_residual.yaml`
2. 构建真实 DRQ learner 架构：
   3 路 `224x224` 图像、ResNet-18、proprio、chunk residual action
3. 构建真实 `MemoryEfficientStepWindowReplayBufferDataStore`
4. 用 fake transition 填充 replay，模拟 step-window replay 采样
5. 按 learner 主循环的实际更新模式跑：
   `(critic_actor_ratio - 1) * update_critics + update_high_utd`
6. 分别测试 3 个 scenario

本次 benchmark 的关键配置：

```text
device:              NVIDIA H20
torch:               2.3.0+cu121
batch_size:          128
critic_actor_ratio:  4
utd_ratio:           2
fake_steps:          700
episode_length:      200
warmup:              1
iterations:          3
resnet pretrained:   true, local mirror loaded
```

运行命令：

```bash
conda run -n serl_torch env CUDA_VISIBLE_DEVICES=3 \
  python test/benchmark_learner_update_speed.py \
  --warmup 1 \
  --iterations 3
```

语法检查命令：

```bash
conda run -n serl_torch env PYTHONPATH=$PWD:$PWD/serl_launcher \
  python -m py_compile \
  serl_launcher/serl_launcher/agents/continuous/sac.py \
  test/benchmark_learner_update_speed.py
```

## 3. 实验场景

### 3.1 `baseline_fp32_legacy`

含义：

- fp32
- mixed precision 关闭
- 使用修复前 actor update 逻辑

实现方式：

- benchmark 脚本里用 legacy monkeypatch 模拟修复前的 `SACAgent.update(...)`
- 这样即使代码已经修复，也能继续复现实验基线

### 3.2 `bf16_legacy`

含义：

- bf16 mixed precision 开启
- 仍使用修复前 actor update 逻辑

这个场景单独隔离 bf16 的收益。

### 3.3 `bf16_freeze_critic_actor_update`

含义：

- bf16 mixed precision 开启
- 使用当前修复后的 actor update 逻辑
- actor update 阶段冻结 critic 参数，避免白算 critic 参数梯度

这个场景验证“bf16 + 修复 critic 白算梯度”的叠加收益。

## 4. 实验结果

### 4.1 总体结果

| scenario | outer mean (s) | updates/s | sample total (s) | critic-only total (s) | high-utd (s) | peak mem (MB) | speedup vs baseline |
|---|---:|---:|---:|---:|---:|---:|---:|
| baseline_fp32_legacy | 2.8806 | 0.3472 | 0.3213 | 1.6052 | 0.9539 | 17052.6 | 0.00% |
| bf16_legacy | 2.2837 | 0.4379 | 0.3479 | 1.1907 | 0.7449 | 9339.6 | 20.72% |
| bf16_freeze_critic_actor_update | 2.1040 | 0.4753 | 0.3207 | 1.1935 | 0.5896 | 9310.1 | 26.96% |

### 4.2 直接结论

开启 bf16 后：

```text
outer update: 2.8806s -> 2.2837s
提升:         20.72%
updates/s:    0.3472 -> 0.4379
显存峰值:     17052.6MB -> 9339.6MB
显存下降:     约 45.2%
```

在 bf16 基础上修 actor update 白算 critic grad 后：

```text
outer update: 2.2837s -> 2.1040s
相对 bf16:    约 7.87% 额外提升
相对 baseline: 26.96% 总提升
updates/s:    0.4379 -> 0.4753
```

`high_utd` 阶段变化最明显：

```text
bf16 legacy high_utd: 0.7449s
bf16 fixed high_utd:  0.5896s
提升:                 约 20.85%
```

这符合预期：修复点只影响 actor update 阶段，而 actor update 位于 `update_high_utd(...)` 里，不影响前面单独的 `update_critics(...)`。

### 4.3 组件级解读

`sample total` 在三组之间大约 `0.32s - 0.35s`，差异主要是测量噪声和 CPU replay 采样波动。它不是这轮优化的主要收益来源。

`critic-only total` 从 fp32 到 bf16 明显降低：

```text
1.6052s -> 1.1907s
```

说明 bf16 对 ResNet critic 更新阶段收益很明显。

`bf16_legacy` 和 `bf16_freeze_critic_actor_update` 的 `critic-only total` 基本一致：

```text
1.1907s vs 1.1935s
```

说明 actor update 冻结 critic 参数不会影响单独 critic-only update 的计时，这也符合预期。

`high_utd` 从 bf16 legacy 到修复后明显降低：

```text
0.7449s -> 0.5896s
```

这说明原来的 actor update 确实在给 critic 参数做无用反传。修复后仍然保留 `dQ/da`，但不再计算 critic parameter grad。

## 5. 为什么这两个优化值得优先合入

### 5.1 bf16

bf16 是最小侵入的配置级优化：

- H20 支持 bf16
- 代码里已经有 mixed precision 配置路径
- 对速度和显存都有明显收益
- 比 fp16 更稳，通常不需要 gradient scaler

建议默认实验命令里先打开：

```bash
training.mixed_precision.enabled=true \
training.mixed_precision.dtype=bfloat16
```

需要观察：

- `critic_loss`
- `actor_loss`
- `temperature`
- `entropy`
- `predicted_q_gap`
- 成功率曲线

### 5.2 actor update 冻结 critic 参数

这是代码层面的真实浪费修复：

- actor optimizer 不会更新 critic 参数
- 原逻辑会计算 critic 参数梯度，但这些梯度不会被使用
- 修复后保留 actor 所需的 action gradient
- 微基准显示对 `high_utd` 阶段有明显收益

这项优化不应该改变 SAC 的目标函数，只是避免无用梯度计算。

## 6. 当前 benchmark 的局限

这份 benchmark 主要用于比较相对速度，不应被理解为完整训练吞吐。

局限包括：

- fake 数据是随机生成的，不评估学习质量
- 没有启动 agentlace actor / learner 网络通信
- 没有 W&B logging
- 没有 checkpoint 保存
- 没有真实 actor 插入 replay 时的锁竞争
- 没有评估真实 robot / LIBERO 环境端到端吞吐

不过它覆盖了 learner 梯度更新最重的核心路径：

```text
StepWindowReplay sample
chunk batch reshape
CPU -> GPU tensor conversion
DrQ augmentation
critic update
high_utd update
actor update
temperature update
```

所以它适合作为后续优化的第一层消融工具。

## 7. 后续优化建议和消融顺序

后续建议按“低风险 -> 高收益但改变训练行为 -> 工程管线优化”的顺序继续做。

### 7.1 下一步优先：actor-only network broadcast

当前 learner publish 网络时复用完整 checkpoint payload：

```text
params + target_params + optimizer
```

但 actor 采样动作只需要 actor 参数。

之前测到完整 payload 量级：

```text
初始化后完整 payload: 约 720MB
一次 update 后完整 payload: 约 843MB
actor params:          约 144MB
```

建议新增：

```text
snapshot_actor_network_payload(...)
```

只发：

```text
step
params.actor
```

预期收益：

- 降低 periodic publish 卡顿
- 降低 actor 接收和 load 网络的开销
- 对训练主 update 的平均时间影响可能不如 bf16 明显，但会降低长尾 jitter

建议 benchmark：

- 单独测 `snapshot_agent_checkpoint_payload(...)`
- 单独测 `snapshot_actor_network_payload(...)`
- 如果可以，测 `server.publish_network(...)` 阻塞耗时

### 7.2 大收益但改变训练行为：freeze ResNet backbone

之前探索性微基准显示：

```text
update_high_utd default:         约 0.963s
update_high_utd freeze backbone: 约 0.553s
freeze + bf16:                  约 0.505s
```

这可能是最大单项加速之一，但它改变训练行为。

建议消融：

```text
bf16 + fixed actor update
bf16 + fixed actor update + freeze_backbone=true
```

需要同时看：

- update speed
- actor 行为是否变差
- success rate
- Q loss 是否稳定
- 是否更容易过拟合 head / pooling / MLP

如果真机相机位固定、预训练 ResNet 特征足够，冻结 backbone 可能非常划算。

### 7.3 结构性降计算：调 `critic_actor_ratio` 和 `utd_ratio`

当前默认一次 outer update 实际包含：

```text
5 次 critic update
1 次 actor update
1 次 temperature update
```

这是由：

```text
critic_actor_ratio=4
utd_ratio=2
```

共同造成的。

建议逐步消融：

```text
critic_actor_ratio=3, utd_ratio=2
critic_actor_ratio=2, utd_ratio=2
critic_actor_ratio=2, utd_ratio=1
```

这类优化会直接减少反向传播次数，速度收益会很大，但会改变算法训练比例。

建议指标：

- `outer_mean_s`
- `updates/s`
- `critic_updates/s`
- `critic_loss`
- `predicted_q_gap`
- success rate
- replay lag，也就是 `env_steps - online_update_steps`

### 7.4 数据增强优化：vectorized random crop

当前 `batched_random_crop(...)` 是 Python loop 逐样本 crop。之前探索性微基准显示，完全关掉 augmentation 可让 `update_high_utd` 约提升 `8%`。

不建议直接关掉 DrQ augmentation，但建议把 crop 向量化：

```text
一次 pad 整个 batch
一次生成 batch offsets
用 advanced indexing / gather 取 crop
```

预期收益：

- 小于完全去掉 augmentation 的收益
- 但不会改变 DrQ 训练语义
- 可以减少 Python loop 和 kernel launch 碎片

### 7.5 Replay prefetch

当前 learner 更新是同步顺序：

```text
sample batch
train
sample batch
train
```

本轮数据里一次 outer update 的 `sample total` 约 `0.32s`。可以通过后台线程预取 batch 来隐藏一部分 CPU 采样开销。

建议实现：

```text
ReplayPrefetcher
  后台线程 sample_mixed_training_batch
  主线程 consume ready batch
```

需要注意：

- replay buffer lock 竞争
- agentlace insert 同时写入 replay
- offline/online mixed batch 一致性
- prefetch queue 不宜太深

### 7.6 H2D 和 memory layout 优化

可选方向：

```text
pin_memory + non_blocking transfer
channels_last for image tensors and ResNet
torch.backends.cudnn.benchmark = True
torch.compile for actor/critic modules
```

这些需要更谨慎测试，因为收益和 PyTorch / cuDNN / 输入 shape 相关。

当前 benchmark 输出里有 cuDNN execution plan warning，后续可以单独验证：

```text
channels_last 是否减少 warning
cudnn.benchmark 是否稳定加速
torch.compile 是否值得额外 compile 成本
```

## 8. 推荐后续实验矩阵

建议先继续用 `test/benchmark_learner_update_speed.py` 扩展 scenario：

```text
baseline_fp32_legacy
bf16_legacy
bf16_freeze_critic_actor_update
bf16_freeze_critic_actor_update_freeze_backbone
bf16_freeze_critic_actor_update_ratio_3_2
bf16_freeze_critic_actor_update_ratio_2_2
bf16_freeze_critic_actor_update_ratio_2_1
```

然后再补 actor 网络同步 microbench：

```text
full checkpoint payload snapshot
actor-only payload snapshot
full publish_network
actor-only publish_network
actor apply full payload
actor apply actor-only payload
```

最后再补数据管线 microbench：

```text
current random crop
vectorized random crop
sync replay sample
prefetch replay sample
normal H2D
pinned non_blocking H2D
```

## 9. 当前结论

本轮两个优化已经有明确收益：

```text
baseline fp32 legacy:               2.8806s / update
bf16 legacy:                        2.2837s / update
bf16 + actor-update critic freeze:  2.1040s / update
```

总提升：

```text
2.8806s -> 2.1040s
约 26.96% speedup
updates/s: 0.3472 -> 0.4753
```

显存也明显下降：

```text
17052.6MB -> 9310.1MB
约 45.4% reduction
```

因此建议：

1. 后续 learner 默认实验先打开 bf16
2. 保留 actor update 冻结 critic 参数的代码修复
3. 下一轮优先做 actor-only broadcast 和 freeze backbone 消融
4. 再做 `critic_actor_ratio / utd_ratio` 的结构性速度-效果 tradeoff 实验

## 10. 第二轮消融：actor-only broadcast 和 freeze backbone

### 10.1 本轮新增验证项

本轮按新的优先级继续验证两个点：

1. actor-only network broadcast
2. `encoder.resnet.freeze_backbone=true`

明确暂时不做：

```text
critic_actor_ratio / utd_ratio 调参
```

原因是这类调参会改变算法更新比例，当前阶段先优先处理低风险工程优化和 backbone 训练开销。

### 10.2 代码改动

#### actor-only broadcast

新增 helper：

```text
serl_launcher/serl_launcher/common/checkpoint_codec.py
  snapshot_actor_network_payload(...)
```

这个 payload 只包含：

```text
step
params.actor
```

不再包含：

```text
critic params
target critic params
optimizer state
```

正式 learner publish 路径已替换：

```text
examples/agibot_real/scripts/run_residual_training.py
examples/libero/scripts/train_residual_step.py
```

保留 full checkpoint 的路径不变：

- checkpoint save 仍然使用 `snapshot_agent_checkpoint_payload(...)`
- LIBERO async eval checkpoint 仍然使用 full checkpoint payload

这样 actor 参数同步变轻，但训练恢复和 async eval 不受影响。

#### freeze backbone

`freeze_backbone` 已经是现有配置项：

```yaml
encoder:
  resnet:
    freeze_backbone: true
```

本轮没有改默认配置，只在 benchmark 里新增一个 scenario 做消融：

```text
bf16_freeze_critic_actor_update_freeze_backbone
```

### 10.3 Benchmark 命令

```bash
conda run -n serl_torch env CUDA_VISIBLE_DEVICES=3 \
  python test/benchmark_learner_update_speed.py \
  --warmup 1 \
  --iterations 3 \
  --payload-iterations 3
```

关键配置：

```text
device:              NVIDIA H20
torch:               2.3.0+cu121
batch_size:          128
critic_actor_ratio:  4
utd_ratio:           2
fake_steps:          700
episode_length:      200
resnet pretrained:   true, local mirror loaded
```

### 10.4 梯度更新速度结果

| scenario | outer mean (s) | updates/s | sample total (s) | critic-only total (s) | high-utd (s) | peak mem (MB) | speedup vs baseline |
|---|---:|---:|---:|---:|---:|---:|---:|
| baseline_fp32_legacy | 2.9077 | 0.3439 | 0.3538 | 1.5963 | 0.9574 | 17052.6 | 0.00% |
| bf16_legacy | 2.1943 | 0.4557 | 0.2689 | 1.1749 | 0.7503 | 9339.6 | 24.53% |
| bf16_freeze_critic_actor_update | 2.0080 | 0.4980 | 0.2550 | 1.1701 | 0.5828 | 9310.1 | 30.94% |
| bf16_freeze_critic_actor_update_freeze_backbone | 1.6467 | 0.6073 | 0.2550 | 0.8963 | 0.4953 | 1174.0 | 43.37% |

相对上一轮最优配置：

```text
bf16 + actor-update critic freeze:
  2.0080s / update

再加 freeze_backbone:
  1.6467s / update

额外提升:
  约 18.0%
```

相对最原始 baseline：

```text
2.9077s -> 1.6467s
约 43.37% speedup
updates/s: 0.3439 -> 0.6073
```

显存峰值也大幅下降：

```text
bf16 + critic freeze:             9310.1MB
bf16 + critic freeze + backbone:  1174.0MB
```

这个显存数字说明 ResNet backbone 反向图和 optimizer state 是当前内存压力的重要来源。

### 10.5 actor-only payload 结果

| payload benchmark | full | actor-only | improvement |
|---|---:|---:|---:|
| tensor payload MB | 733.3 | 143.9 | 80.38% smaller |
| snapshot mean s | 0.3459 | 0.0395 | 88.57% faster |
| actor apply mean s | 0.1462 | 0.0300 | 79.52% faster |

这里的 full payload 是训练后 snapshot，已经包含 optimizer state；actor-only payload 只包含 actor module state dict。

直接结论：

- learner 端构造 broadcast payload 的时间从 `0.3459s` 降到 `0.0395s`
- actor 端 apply payload 的时间从 `0.1462s` 降到 `0.0300s`
- payload tensor 体积减少约 `80.38%`

这项优化不一定显著改变每次 gradient update 的纯计算时间，但会显著降低周期性 publish / actor sync 的阻塞和长尾 jitter。

### 10.6 第二轮结论

到目前为止，如果按“原始 fp32 legacy”作为 baseline，已验证优化叠加效果是：

```text
baseline fp32 legacy:
  2.9077s / update

bf16:
  2.1943s / update
  24.53% speedup

bf16 + actor update critic freeze:
  2.0080s / update
  30.94% speedup

bf16 + actor update critic freeze + freeze backbone:
  1.6467s / update
  43.37% speedup
```

actor-only broadcast 的同步侧收益是：

```text
payload:  733.3MB -> 143.9MB
snapshot: 0.3459s -> 0.0395s
apply:    0.1462s -> 0.0300s
```

建议下一步：

1. 保留 actor-only broadcast 代码优化
2. 把 `freeze_backbone=true` 作为训练 ablation，而不是立刻默认打开
3. 如果 freeze backbone 的成功率不降，优先考虑把它作为真机快速训练默认项
4. 后续再做 vectorized random crop 和 replay prefetch
5. `critic_actor_ratio / utd_ratio` 暂时不动

## 11. 第三轮消融：vectorized random crop 和 replay prefetch

### 11.1 本轮新增验证项

本轮继续验证两个点：

1. `batched_random_crop(...)` 向量化
2. replay batch prefetch

仍然暂时不做：

```text
critic_actor_ratio / utd_ratio 调参
真实 actor / learner run 验证
```

### 11.2 代码改动

#### vectorized random crop

修改位置：

```text
serl_launcher/serl_launcher/vision/data_augmentations.py
```

原实现是：

```text
flatten batch
for each image:
  random_crop(single image)
stack crops
```

新实现是：

```text
flatten batch
一次性 pad N 张图
一次性生成 N 个 y/x crop offset
用 batch advanced indexing 取 crop
reshape 回原 shape
```

保持的语义：

- 输入仍然是 batch dims + HWC image dims
- 每张图仍然独立随机 crop offset
- 输出 shape / dtype / device 不变
- `num_batch_dims=2` 的 AgiBot / LIBERO 图像 batch 兼容

#### replay prefetch

修改位置：

```text
test/benchmark_learner_update_speed.py
```

这轮先只在 benchmark 里加入 prefetch 验证，没有直接接入正式训练脚本。原因是正式训练里 replay 还会和 agentlace insert、offline replay mixing、shutdown 生命周期交互，建议先用 benchmark 确认收益，再决定是否加配置开关接入 learner。

benchmark 中的 prefetch 逻辑：

```text
后台线程持续 sample_mixed_training_batch(...)
Queue(maxsize=2)
主线程训练时从 queue 取 batch
```

注意：prefetch 场景表里的 `sample total` 表示主线程等待 prefetched batch 的时间，不是后台线程实际采样 CPU 时间。

### 11.3 Benchmark 命令

```bash
conda run -n serl_torch env CUDA_VISIBLE_DEVICES=3 \
  python test/benchmark_learner_update_speed.py \
  --warmup 1 \
  --iterations 3 \
  --payload-iterations 3 \
  --crop-iterations 10
```

关键配置：

```text
device:              NVIDIA H20
torch:               2.3.0+cu121
batch_size:          128
critic_actor_ratio:  4
utd_ratio:           2
fake_steps:          700
episode_length:      200
resnet pretrained:   true, local mirror loaded
prefetch queue:      2
```

### 11.4 梯度更新速度结果

这一轮所有场景都运行在 vectorized crop 代码版本上。因此它和第二轮的 absolute number 不完全等价；crop 的单点收益见下一节 microbenchmark。

| scenario | outer mean (s) | updates/s | sample total (s) | critic-only total (s) | high-utd (s) | peak mem (MB) | speedup vs baseline |
|---|---:|---:|---:|---:|---:|---:|---:|
| baseline_fp32_legacy | 2.5970 | 0.3851 | 0.3353 | 1.3820 | 0.8796 | 17054.0 | 0.00% |
| bf16_legacy | 1.9634 | 0.5093 | 0.3305 | 0.9674 | 0.6654 | 9340.8 | 24.40% |
| bf16_freeze_critic_actor_update | 1.7997 | 0.5557 | 0.3188 | 0.9666 | 0.5141 | 9313.0 | 30.70% |
| bf16_freeze_critic_actor_update_freeze_backbone | 1.4432 | 0.6929 | 0.3151 | 0.6973 | 0.4306 | 1174.2 | 44.43% |
| bf16_freeze_critic_actor_update_freeze_backbone_prefetch | 1.1978 | 0.8348 | 0.0027 | 0.7364 | 0.4587 | 1174.2 | 53.88% |

对比 freeze backbone 后是否加 prefetch：

```text
no prefetch:
  1.4432s / update
  sample wait 0.3151s
  updates/s 0.6929

prefetch queue=2:
  1.1978s / update
  sample wait 0.0027s
  updates/s 0.8348

额外提升:
  约 17.0%
```

这说明 replay sampling 在模型计算被 bf16 / freeze grad / freeze backbone 压低后，已经变成可观的可隐藏开销。prefetch 可以把主线程 sample wait 基本隐藏掉。

### 11.5 Random crop microbenchmark

输入 shape：

```text
[128, 1, 224, 224, 3]
```

结果：

| random crop benchmark | legacy loop | vectorized | improvement |
|---|---:|---:|---:|
| mean s | 0.016546 | 0.000617 | 96.27% faster |

解释：

- 旧版逐样本 loop 有大量 Python 调用、小 kernel launch 和 CUDA scalar sync
- 新版一次性 pad 和 indexing，基本把 crop 开销压到了很低
- 单点 crop 很快，但整轮 learner 里模型 forward/backward 仍然是主耗时，所以 outer update 的整体提升不会等于 96%

### 11.6 Payload benchmark 同轮结果

本轮 payload benchmark 仍然保持 actor-only 优势：

| payload benchmark | full | actor-only | improvement |
|---|---:|---:|---:|
| tensor payload MB | 733.3 | 143.9 | 80.38% smaller |
| snapshot mean s | 1.9214 | 0.4783 | 75.11% faster |
| actor apply mean s | 0.2551 | 0.0444 | 82.59% faster |

这轮 snapshot 绝对时间比上一轮更高，可能受系统负载和 CPU 内存状态影响；但 actor-only 仍然显著更小、更快。

### 11.7 第三轮结论

当前 benchmark 里，如果不动 `critic_actor_ratio / utd_ratio`，组合优化后的 fake learner update 速度已经从本轮 baseline：

```text
baseline fp32 legacy:
  2.5970s / update
  0.3851 updates/s
```

提升到：

```text
bf16 + actor-update critic freeze + freeze_backbone + replay prefetch:
  1.1978s / update
  0.8348 updates/s
```

总提升：

```text
约 53.88% speedup
updates/s 提升约 2.17x
```

本轮两项新增优化的判断：

1. vectorized random crop 建议直接保留
   - 不改变 DrQ crop 语义
   - 单点 crop 加速非常明显
   - 代码复杂度可控

2. replay prefetch 建议下一步接入正式 learner 时加配置开关
   - benchmark 中收益明显
   - 但正式训练涉及 replay 写入、offline replay、异常处理和 shutdown
   - 建议用 `training.replay_prefetch.enabled` 和 `training.replay_prefetch.queue_size` 控制

3. `critic_actor_ratio / utd_ratio` 继续暂时不动
   - 当前已经有明显工程收益
   - 训练比例调参应留到学习质量验证阶段

## 12. 第四轮消融：big-batch sample、StepWindow fast sample、actor/temp batch 裁剪

### 12.1 本轮目标

本轮继续保持 benchmark-only，不接入正式 learner 训练流程。

验证三个点：

1. outer update 一次性 big-batch sample
2. benchmark-only StepWindow replay fast sample
3. actor / temperature update 只传必要 batch 字段

仍然暂时不做：

```text
正式接入 training.replay_prefetch.enabled
GPU transfer prefetch / pinned memory
critic_actor_ratio / utd_ratio 调参
真实 actor / learner run 验证
```

### 12.2 代码改动范围

修改位置：

```text
test/benchmark_learner_update_speed.py
```

这轮没有改正式训练脚本，也没有替换生产 `MemoryEfficientStepWindowReplayBuffer.sample(...)`。

新增 benchmark-only 路径：

```text
big-batch sample:
  每个 outer update 一次 sample critic_actor_ratio * batch_size
  再 split 成 critic_actor_ratio 个 batch

fast StepWindow sample:
  用 monkeypatch 替换当前 replay_buffer.sample
  一次性构造 [B, window_size] step index matrix
  批量 gather actions / rewards / masks / observations / next_observations
  保留原 sample 等价性检查

actor/temp batch trim:
  update_high_utd 的 actor+temperature 阶段只传 observations + action_mask
```

fast sample 在 benchmark 启动时会用固定 start ids 对比 legacy sample 和 fast sample，检查 nested batch 的 key、shape、dtype 和数值。

### 12.3 Benchmark 命令

```bash
conda run -n serl_torch env CUDA_VISIBLE_DEVICES=3 \
  PYTHONPATH=$PWD:$PWD/serl_launcher \
  python test/benchmark_learner_update_speed.py \
  --warmup 1 \
  --iterations 3 \
  --payload-iterations 3 \
  --crop-iterations 10 \
  --sample-iterations 10 \
  --json-output /tmp/learner_update_speed_benchmark_round4b.json
```

关键配置：

```text
device:              NVIDIA H20
torch:               2.3.0+cu121
batch_size:          128
critic_actor_ratio:  4
utd_ratio:           2
fake_steps:          700
episode_length:      200
resnet pretrained:   true, local mirror loaded
```

### 12.4 梯度更新速度结果

| scenario | outer mean (s) | updates/s | sample total (s) | critic-only total (s) | high-utd (s) | peak mem (MB) | speedup vs baseline |
|---|---:|---:|---:|---:|---:|---:|---:|
| baseline_fp32_legacy | 2.5288 | 0.3954 | 0.2679 | 1.3806 | 0.8802 | 17054.0 | 0.00% |
| bf16_legacy | 1.9645 | 0.5090 | 0.3279 | 0.9707 | 0.6658 | 9340.8 | 22.32% |
| bf16_freeze_critic_actor_update | 1.8068 | 0.5535 | 0.3177 | 0.9706 | 0.5183 | 9313.0 | 28.55% |
| bf16_freeze_critic_actor_update_freeze_backbone | 1.4685 | 0.6810 | 0.3370 | 0.6983 | 0.4330 | 1174.2 | 41.93% |
| bf16_freeze_critic_actor_update_freeze_backbone_fast_sample | 1.3465 | 0.7427 | 0.2171 | 0.6984 | 0.4308 | 1174.2 | 46.75% |
| bf16_freeze_critic_actor_update_freeze_backbone_big_batch | 1.3957 | 0.7165 | 0.2672 | 0.6966 | 0.4318 | 1174.2 | 44.81% |
| bf16_freeze_critic_actor_update_freeze_backbone_big_batch_fast_sample | 1.3131 | 0.7616 | 0.1835 | 0.6982 | 0.4313 | 1174.2 | 48.08% |
| bf16_freeze_critic_actor_update_freeze_backbone_big_batch_fast_sample_trim_actor_batch | 1.3186 | 0.7584 | 0.1870 | 0.6995 | 0.4321 | 1174.2 | 47.86% |
| bf16_freeze_critic_actor_update_freeze_backbone_prefetch | 1.1359 | 0.8804 | 0.0008 | 0.7035 | 0.4315 | 1174.2 | 55.08% |

以 `bf16 + critic freeze + freeze_backbone` 作为本轮工程优化基线：

| variant | outer mean (s) | sample total (s) | extra speedup vs freeze_backbone |
|---|---:|---:|---:|
| freeze_backbone baseline | 1.4685 | 0.3370 | 0.00% |
| fast sample only | 1.3465 | 0.2171 | 8.30% |
| big-batch only | 1.3957 | 0.2672 | 4.96% |
| big-batch + fast sample | 1.3131 | 0.1835 | 10.58% |
| big-batch + fast sample + actor/temp trim | 1.3186 | 0.1870 | 10.20% |
| replay prefetch | 1.1359 | 0.0008 | 22.65% |

### 12.5 Replay sample microbenchmark

| replay sample benchmark | mean s | improvement |
|---|---:|---:|
| legacy single batch | 0.042219 | - |
| fast single batch | 0.033847 | 19.83% faster |
| legacy 4x separate samples | 0.207228 | - |
| legacy big batch + split | 0.288123 | 39.04% slower |
| fast big batch + split | 0.189248 | 8.68% faster |

解释：

1. fast StepWindow sample 是明确正收益
2. legacy big-batch sample 在纯 sample microbenchmark 里反而变慢
3. big-batch 在完整 learner 里仍有小幅收益，可能来自减少采样调用和同步边界，但它不是最稳的单点优化
4. big-batch 和 fast sample 叠加后，sample total 从 `0.3370s` 降到 `0.1835s`

### 12.6 Random crop 和 payload 同轮结果

| random crop benchmark | legacy loop | vectorized | improvement |
|---|---:|---:|---:|
| mean s | 0.011729 | 0.000610 | 94.80% faster |

| payload benchmark | full | actor-only | improvement |
|---|---:|---:|---:|
| tensor payload MB | 733.3 | 143.9 | 80.38% smaller |
| snapshot mean s | 0.4373 | 0.0870 | 80.11% faster |
| actor apply mean s | 0.1601 | 0.0334 | 79.16% faster |

### 12.7 第四轮结论

本轮最值得继续推进的是 StepWindow replay sample 向量化。

```text
freeze_backbone baseline:
  1.4685s / update
  sample total 0.3370s

fast sample only:
  1.3465s / update
  sample total 0.2171s
  extra speedup 8.30%

big-batch + fast sample:
  1.3131s / update
  sample total 0.1835s
  extra speedup 10.58%
```

判断：

1. StepWindow fast sample 建议作为下一轮重点
   - 不改变 batch 内容语义
   - 已加入 benchmark 等价性检查
   - 能直接减少 CPU 采样开销，不只是隐藏采样

2. big-batch sample 可以保留为候选，但不建议单独优先生产化
   - 完整 learner 里有约 `4.96%` 额外收益
   - 纯 sample microbenchmark 里 legacy big-batch 反而更慢
   - 更适合作为 fast sample 后的组合优化

3. actor/temp batch 裁剪暂时不建议继续投入
   - `1.3131s -> 1.3186s`，没有正收益
   - 当前 DrQ high_utd 已经先把 full batch 搬到 GPU 给 critic 使用
   - 裁剪只减少 actor/temp 阶段的二次 dict traversal，收益太小

4. replay prefetch 仍然是 benchmark 里最快路径
   - `1.1359s / update`
   - 但它是隐藏 sample wait，不减少后台 CPU sample 工作
   - 正式接入仍然建议等 StepWindow sample 向量化评估后再做

下一步优先级建议：

```text
1. 把 StepWindow fast sample 从 benchmark-only 整理成可测试的正式实现候选
2. 给 fast sample 增加更多边界测试：terminal window、short window、wrap-around、multiple image keys
3. 再评估 big-batch + fast sample 是否值得接入正式 learner
4. 最后再考虑 replay prefetch 和 GPU transfer prefetch
```

## 13. 第五轮消融：torch.compile

### 13.1 本轮目标

本轮继续保持 benchmark-only，验证 `torch.compile` 是否能进一步降低 learner gradient update 时间。

本轮只在 benchmark 脚本里加 compile 场景：

```text
test/benchmark_learner_update_speed.py
```

没有修改正式 learner 训练脚本。

### 13.2 compile 策略

没有直接 compile 整个 update/loss。原因是完整 update 里包含：

```text
dict batch traversal
DrQ random crop
optimizer step
target update
policy sampling
Python 控制流
```

这些都容易导致 graph break 或编译成本过高。

本轮采用 module-level compile：

```text
compile_critic:
  torch.compile(agent.state.modules["critic"])
  torch.compile(agent.state.target_modules["critic"])

compile_actor_critic:
  compile critic
  compile target critic
  compile actor
```

使用参数：

```text
backend:   inductor
mode:      default
fullgraph: false
dynamic:   false
```

注意：`actor` 返回自定义 `DiagGaussianDistribution`，理论上比 critic 更容易产生 graph break；因此本轮分别测 `compile_critic` 和 `compile_actor_critic`。

### 13.3 Benchmark 命令

`compile_critic`：

```bash
conda run -n serl_torch env CUDA_VISIBLE_DEVICES=3 \
  PYTHONPATH=$PWD:$PWD/serl_launcher \
  python test/benchmark_learner_update_speed.py \
  --include-compile \
  --scenario-filter freeze_backbone_compile_critic \
  --torch-compile-mode default \
  --warmup 1 \
  --iterations 3 \
  --payload-iterations 1 \
  --crop-iterations 1 \
  --sample-iterations 1 \
  --json-output /tmp/learner_update_speed_benchmark_compile_critic_round5.json
```

`compile_actor_critic`：

```bash
conda run -n serl_torch env CUDA_VISIBLE_DEVICES=3 \
  PYTHONPATH=$PWD:$PWD/serl_launcher \
  python test/benchmark_learner_update_speed.py \
  --include-compile \
  --scenario-filter freeze_backbone_compile_actor_critic \
  --torch-compile-mode default \
  --warmup 1 \
  --iterations 3 \
  --payload-iterations 1 \
  --crop-iterations 1 \
  --sample-iterations 1 \
  --json-output /tmp/learner_update_speed_benchmark_compile_actor_critic_round5.json
```

`fast_sample + compile_actor_critic`：

```bash
conda run -n serl_torch env CUDA_VISIBLE_DEVICES=3 \
  PYTHONPATH=$PWD:$PWD/serl_launcher \
  python test/benchmark_learner_update_speed.py \
  --include-compile \
  --scenario-filter fast_sample_compile_actor_critic \
  --torch-compile-mode default \
  --warmup 1 \
  --iterations 3 \
  --payload-iterations 1 \
  --crop-iterations 1 \
  --sample-iterations 1 \
  --json-output /tmp/learner_update_speed_benchmark_compile_fastsample_actor_critic_round5.json
```

### 13.4 冷启动编译成本

`torch.compile` 首次编译成本很高。

一个 `warmup=0, iterations=1` 的 `compile_critic` smoke 中：

```text
outer mean: 88.7385s
critic-only total: 38.0554s
high-utd: 50.2986s
```

这不是 steady-state 性能，而是把首次 inductor 编译时间算进了 update。真实训练如果要用 compile，需要：

```text
1. 显式 warmup compile
2. 不把首轮 compile 时间计入 learner throughput
3. 只用于长时间训练，不适合很短的 smoke run
```

### 13.5 Steady-state 结果

下面用第四轮的 `bf16 + critic freeze + freeze_backbone` 作为比较基线：

```text
freeze_backbone baseline:
  outer mean 1.4685s
  sample total 0.3370s
  critic-only total 0.6983s
  high-utd 0.4330s
```

| variant | outer mean (s) | updates/s | sample total (s) | critic-only total (s) | high-utd (s) | extra speedup vs freeze_backbone |
|---|---:|---:|---:|---:|---:|---:|
| freeze_backbone baseline | 1.4685 | 0.6810 | 0.3370 | 0.6983 | 0.4330 | 0.00% |
| compile_critic | 1.1501 | 0.8695 | 0.2854 | 0.5269 | 0.3377 | 21.68% |
| compile_actor_critic | 1.0883 | 0.9188 | 0.3147 | 0.4825 | 0.2910 | 25.89% |
| fast_sample + compile_actor_critic | 0.9617 | 1.0399 | 0.1909 | 0.4807 | 0.2899 | 34.51% |
| big_batch + fast_sample + compile_actor_critic | 1.1396 | 0.8775 | 0.3657 | 0.4813 | 0.2925 | 22.39% |
| replay prefetch | 1.1359 | 0.8804 | 0.0008 | 0.7035 | 0.4315 | 22.65% |

相对本轮 fp32 legacy baseline `2.5288s/update`：

| variant | outer mean (s) | speedup vs fp32 legacy |
|---|---:|---:|
| compile_critic | 1.1501 | 54.52% |
| compile_actor_critic | 1.0883 | 56.96% |
| fast_sample + compile_actor_critic | 0.9617 | 61.97% |
| replay prefetch | 1.1359 | 55.08% |

### 13.6 观察

1. `compile_critic` 是明确正收益
   - critic-only total 从 `0.6983s` 降到 `0.5269s`
   - high-utd 从 `0.4330s` 降到 `0.3377s`
   - outer update 额外提升 `21.68%`

2. `compile_actor_critic` 比只 compile critic 更好
   - outer update `1.1501s -> 1.0883s`
   - 额外约 `5.37%`
   - actor 返回 distribution object，但这轮实际没有阻止收益

3. `fast_sample + compile_actor_critic` 是当前 benchmark-only 最快非 prefetch 路径
   - outer update `0.9617s`
   - updates/s `1.0399`
   - 比 freeze_backbone baseline 额外提升 `34.51%`
   - 比 replay prefetch 的 `1.1359s` 还快约 `15.34%`

4. `big_batch + fast_sample + compile_actor_critic` 不建议继续优先投入
   - sample total 变成 `0.3657s`
   - outer update 退回 `1.1396s`
   - 和前面结论一致：big-batch 不是稳健单点，容易因为大 batch 构造成本抵消收益

5. compile 会产生 cudnn plan warning
   - benchmark 中出现 `CUDNN_BACKEND_EXECUTION_PLAN_DESCRIPTOR` warning
   - 训练仍能跑完
   - 后续正式接入前需要确认 warning 不影响长期稳定性

### 13.7 第五轮结论

`torch.compile` 值得继续探索，但建议只作为可配置实验路径，不要默认打开。

推荐下一步：

```text
1. 保留 benchmark-only torch.compile 场景
2. 优先考虑 compile_actor_critic，而不是 compile 整个 update
3. 和 StepWindow fast sample 组合验证
4. 真实训练前必须显式 warmup，避免首轮 80s+ 编译时间污染训练 throughput
5. 正式接入时加配置：
   training.torch_compile.enabled
   training.torch_compile.target = critic | actor_critic
   training.torch_compile.mode = default | reduce-overhead
```

当前最佳 benchmark-only 组合：

```text
bf16
+ actor-update critic freeze
+ freeze_backbone
+ StepWindow fast sample
+ torch.compile(actor + critic + target critic)

outer mean:
  0.9617s / update

updates/s:
  1.0399

相对 fp32 legacy:
  约 61.97% speedup
```

## 14. 第六轮消融：torch.compile 组合矩阵、配置 sweep 和 cold-start trade-off

### 14.1 本轮目标

这一轮继续沿着 `torch.compile` 方向细化，重点回答三个问题：

1. `compile_critic` 和 `compile_actor_critic` 在不同 variant 上分别能带来多少收益？
2. 为什么 `big-batch + compile` 不如 `fast_sample + compile`？
3. `torch.compile` 的冷启动编译成本多大，真实训练里多久回本？

另外补做了：

```text
prefetch + compile
fast_sample + prefetch + compile
compile config sweep:
  backend
  mode
  fullgraph
  dynamic
```

### 14.2 当前脚本支持的 compile 维度

基于 benchmark 脚本新增：

```text
--include-compile
--scenario-filter
--torch-compile-backend
--torch-compile-mode
--torch-compile-fullgraph
--torch-compile-dynamic
```

默认 compile 参数为：

```text
backend=inductor
mode=default
fullgraph=false
dynamic=false
```

### 14.3 非 compile 对照矩阵

为了让 compile 和 non-compile 对比尽量在同一版脚本中闭环，这一轮重新测了当前脚本下的 non-compile 对照：

| variant | outer mean (s) | updates/s | sample total (s) | critic-only total (s) | high-utd (s) |
|---|---:|---:|---:|---:|---:|
| freeze_backbone | 1.3436 | 0.7443 | 0.2146 | 0.6985 | 0.4304 |
| fast_sample | 1.3110 | 0.7628 | 0.1858 | 0.6956 | 0.4295 |
| big_batch | 1.4091 | 0.7097 | 0.2842 | 0.6957 | 0.4291 |
| big_batch + fast_sample | 1.3653 | 0.7324 | 0.2410 | 0.6960 | 0.4283 |
| prefetch | 1.1798 | 0.8476 | 0.0013 | 0.7295 | 0.4490 |
| fast_sample + prefetch | 1.1248 | 0.8890 | 0.0005 | 0.6961 | 0.4283 |

### 14.4 compile target × variant 矩阵

steady-state 配置：

```text
warmup=1
iterations=3
backend=inductor
mode=default
fullgraph=false
dynamic=false
```

| variant | no compile | compile_critic | compile_critic delta | compile_actor_critic | compile_actor_critic delta |
|---|---:|---:|---:|---:|---:|
| freeze_backbone | 1.3436 | 1.1640 | 13.37% | 1.0498 | 21.87% |
| fast_sample | 1.3110 | 1.0627 | 18.94% | 0.9733 | 25.76% |
| big_batch + fast_sample | 1.3653 | 1.2004 | 12.08% | 1.0989 | 19.52% |
| prefetch | 1.1798 | 0.8694 | 26.31% | 0.7871 | 33.29% |
| fast_sample + prefetch | 1.1248 | - | - | 0.7874 | 30.00% |

`compile_actor_critic` 在每个可比 variant 上都优于 `compile_critic`：

| variant | compile_actor_critic 比 compile_critic 更快 |
|---|---:|
| freeze_backbone | 9.81% |
| fast_sample | 8.42% |
| big_batch + fast_sample | 8.46% |
| prefetch | 9.47% |

结论：

1. `compile_actor_critic` 稳定优于 `compile_critic`
2. 这个结论在 `fast_sample`、`prefetch`、`big_batch + fast_sample` 下都成立
3. 当前最强的 steady-state 路径已经不是单纯 prefetch，而是 `prefetch + compile_actor_critic`

### 14.5 为什么 big-batch + compile 不好

这是这轮最重要的一个负例。

对比：

```text
fast_sample + compile_actor_critic:
  outer mean   0.9733s
  sample total 0.2005s
  critic total 0.4813s
  high_utd     0.2914s

big_batch + fast_sample + compile_actor_critic:
  outer mean   1.0989s
  sample total 0.3260s
  critic total 0.4812s
  high_utd     0.2916s
```

差异：

```text
outer update: 变慢 12.90%
sample total: 增大 62.61%
critic/high_utd: 基本不变
```

这说明 `big-batch + compile` 的退化几乎完全来自 sample path，而不是模型计算。

也就是说：

```text
compile 优化的是 forward/backward
big-batch 恶化的是 replay batch 构造成本
```

所以它们叠加后，sample 端把 compile 的收益吃掉了。

这也进一步支持前面的判断：

```text
StepWindow fast sample + compile_actor_critic
比
big-batch + compile_actor_critic
更值得优先做
```

### 14.6 fast_sample 和 prefetch 的关系

这一轮专门补了：

```text
fast_sample + prefetch
fast_sample + prefetch + compile_actor_critic
```

non-compile：

| variant | outer mean (s) | sample total (s) |
|---|---:|---:|
| prefetch | 1.1798 | 0.0013 |
| fast_sample + prefetch | 1.1248 | 0.0005 |

这里 `fast_sample + prefetch` 相对 `prefetch` 仍有约 `4.66%` 提升，说明在 non-compile 场景下，虽然 prefetch 已经隐藏了大量等待，但更快的采样仍能略微改善 wall-clock。

compile_actor_critic：

| variant | outer mean (s) | sample total (s) |
|---|---:|---:|
| prefetch + compile_actor_critic | 0.7871 | 0.0006 |
| fast_sample + prefetch + compile_actor_critic | 0.7874 | 0.0005 |

这里两者几乎一样：

```text
outer delta ≈ 0.04%
```

解释：

1. prefetch 已经把主线程 sample wait 几乎完全隐藏
2. compile 把计算进一步压低后，主线程看到的 sample wait 仍接近 0
3. 这时 fast sample 的主要价值就不再是 wall-clock，而更像是减少后台 CPU 采样负载、留出更多系统余量

因此：

```text
fast_sample
在 no-prefetch 场景下直接改善 wall-clock

在 prefetch + compile 场景下主要改善 CPU headroom，
但对主线程 outer mean 影响已经很小
```

### 14.7 compile config sweep

聚焦 `fast_sample + compile_actor_critic`，测试不同 compile 配置：

| backend | mode | fullgraph | dynamic | outer mean (s) | updates/s | peak mem (MB) | 相对 default |
|---|---|---:|---:|---:|---:|---:|---:|
| inductor | default | false | false | 0.9733 | 1.0275 | 1044.3 | baseline |
| inductor | default | true | false | 0.9535 | 1.0488 | 1042.9 | 2.04% faster |
| inductor | default | false | true | 1.1354 | 0.8808 | 1522.3 | 16.66% slower |
| inductor | reduce-overhead | false | false | 1.6351 | 0.6116 | 1029.2 | 68.00% slower |
| aot_eager | default | false | false | 1.4244 | 0.7021 | 1137.1 | 46.35% slower |

观察：

1. `backend=inductor` 明显优于 `aot_eager`
2. `mode=default` 明显优于 `reduce-overhead`
3. `dynamic=true` 不适合这里
   - 更慢
   - 显存从 `~1044MB` 涨到 `~1522MB`
4. `fullgraph=true` 有小幅正收益
   - 对 `fast_sample + compile_actor_critic`，额外约 `2.04%`

再看当前最强组合 `prefetch + compile_actor_critic`：

| config | outer mean (s) | updates/s |
|---|---:|---:|
| default / fullgraph=false | 0.7871 | 1.2705 |
| default / fullgraph=true | 0.7822 | 1.2784 |

这里 `fullgraph=true` 也仍然是小幅正收益：

```text
额外约 0.62%
```

所以目前 compile 配置的推荐值变成：

```text
backend=inductor
mode=default
fullgraph=true
dynamic=false
```

### 14.8 cold-start compile 成本和回本点

为了评估真实训练 trade-off，这一轮额外测了 cold first-step：

```text
warmup=0
iterations=1
```

结果非常明确：compile 冷启动很贵。

| variant | cold first-step (s) | steady-state (s) | matching non-compile (s) | 每步节省 (s) | break-even updates |
|---|---:|---:|---:|---:|---:|
| freeze_backbone + compile_critic | 88.7 | 1.1640 | 1.3436 | 0.1796 | 487.6 |
| freeze_backbone + compile_actor_critic | 122.1 | 1.0498 | 1.3436 | 0.2938 | 412.1 |
| fast_sample + compile_actor_critic | 122.8 | 0.9733 | 1.3110 | 0.3377 | 360.7 |
| prefetch + compile_actor_critic | 122.3 | 0.7871 | 1.1798 | 0.3927 | 309.3 |
| prefetch + compile_actor_critic + fullgraph=true | 123.6 | 0.7822 | 1.1798 | 0.3976 | 308.9 |

这里 break-even 的含义是：

```text
需要大约这么多 learner outer updates，
编译冷启动的额外时间才会被后续每步节省抵消
```

结论：

1. 如果真实训练只跑几百个 update，compile 不划算
2. 如果训练会稳定跑几千、几万步，compile 很值得
3. `prefetch + compile_actor_critic` 的回本点最低，约 `309` 步
4. 真实系统里最好把 compile warmup 放在 training_starts 前后，避免污染吞吐观测

### 14.9 当前结论

当前 benchmark-only 最优组合已经更新为：

```text
bf16
+ actor-update critic freeze
+ freeze_backbone
+ replay prefetch
+ torch.compile(actor + critic + target critic)
+ backend=inductor
+ mode=default
+ fullgraph=true
+ dynamic=false

outer mean:
  0.7822s / update

updates/s:
  1.2784
```

相对当前 non-compile `freeze_backbone` 基线：

```text
1.3436s -> 0.7822s
约 41.78% speedup
```

相对当前 non-compile `prefetch`：

```text
1.1798s -> 0.7822s
约 33.70% speedup
```

本轮优先级判断：

1. `compile_actor_critic` 明确优于 `compile_critic`
2. `fast_sample + compile_actor_critic` 是最好的 no-prefetch 路径
3. `prefetch + compile_actor_critic (+ fullgraph=true)` 是当前最好的 overall benchmark-only 路径
4. `big-batch + compile` 仍然不推荐
5. compile 正式接入前必须显式处理 cold-start，不能直接默认打开

## 15. 正式 LIBERO 脚本验证：bf16 on/off

前面第 1~14 节主要是 benchmark-only 消融。这里额外补一轮：

```text
examples/libero/scripts/train_residual_step.py
```

的正式 learner 路径验证，确认：

1. 正式脚本已经吃到当前代码里的 `critic freeze`
2. 正式脚本已经走 `snapshot_actor_network_payload(...)`
3. 正式脚本在打开 `training.mixed_precision.enabled=true` 后，真实 update 确实能变快

### 15.1 为什么这轮不用 actor/env/policy 全链路

为了只测 learner update，本轮没有把 remote env server、policy server、actor 一起拉起来，而是直接使用正式脚本的：

```text
runtime.role=learner
+ offline.enabled=true
+ offline.pretrain_steps=N
```

路径。

这样仍然走的是正式 `train_residual_step.py` 的 learner 代码，不是 test benchmark 脚本，但可以避免 actor / env / RPC 噪声干扰。

本轮使用：

- task: `libero_10_task_8`
- prepared offline replay: `data/residual/offline_data/libero_10_task_8/openpi_chunk5_alpha0p1`
- replay preload: `offline.load_max_transitions=256`
- `critic_actor_ratio=4`
- `utd_ratio=1`
- `freeze_backbone=true`

### 15.2 运行命令

fp32 对照：

```bash
source /vla/miniconda3/etc/profile.d/conda.sh
conda activate /vla/miniconda3/envs/serl_torch

export PYTHONPATH=/home/hello/codebase:/home/hello/codebase/serl_torch:/home/hello/codebase/serl_torch/serl_launcher:$PYTHONPATH
export CUDA_VISIBLE_DEVICES=0

python examples/libero/scripts/train_residual_step.py \
  runtime.role=learner \
  runtime.trainer_port=59988 \
  runtime.broadcast_port=59989 \
  task.task_id=8 \
  offline.enabled=true \
  offline.load_max_transitions=256 \
  offline.pretrain_steps=2 \
  training.max_update_steps=2 \
  training.log_period=1 \
  training.checkpoint.every_steps=0 \
  training.mixed_precision.enabled=false \
  wandb.debug=true
```

bf16 对照：

```bash
source /vla/miniconda3/etc/profile.d/conda.sh
conda activate /vla/miniconda3/envs/serl_torch

export PYTHONPATH=/home/hello/codebase:/home/hello/codebase/serl_torch:/home/hello/codebase/serl_torch/serl_launcher:$PYTHONPATH
export CUDA_VISIBLE_DEVICES=0

python examples/libero/scripts/train_residual_step.py \
  runtime.role=learner \
  runtime.trainer_port=60088 \
  runtime.broadcast_port=60089 \
  task.task_id=8 \
  offline.enabled=true \
  offline.load_max_transitions=256 \
  offline.pretrain_steps=2 \
  training.max_update_steps=2 \
  training.log_period=1 \
  training.checkpoint.every_steps=0 \
  training.mixed_precision.enabled=true \
  wandb.debug=true
```

### 15.3 结果

先看 2-step：

| variant | offline preload | pretrain total | avg sec / update | updates/s |
|---|---:|---:|---:|---:|
| fp32 | 12.57s | 91.12s | 45.56s | 0.0219 |
| bf16 | 24.72s | 58.03s | 29.01s | 0.0345 |

对应：

```text
45.56s -> 29.01s
约 36.32% 更快

0.0219 -> 0.0345 updates/s
约 57.03% 吞吐提升
```

再补 3-step：

| variant | offline preload | pretrain total | avg sec / update | updates/s |
|---|---:|---:|---:|---:|
| fp32 | 33.38s | 129.07s | 43.02s | 0.0232 |
| bf16 | 38.68s | 42.95s | 14.32s | 0.0698 |

对应：

```text
43.02s -> 14.32s
约 66.72% 更快
```

把这两组一共 `5` 个正式 learner updates 合并看：

| metric | fp32 | bf16 |
|---|---:|---:|
| total update time | 220.19s | 100.98s |
| avg sec / update | 44.04s | 20.20s |
| updates/s | 0.0227 | 0.0495 |

合并结果：

```text
44.04s -> 20.20s
约 54.14% 更快

0.0227 -> 0.0495 updates/s
约 118.05% 吞吐提升
```

### 15.4 如何解读这轮正式验证

这轮正式验证有两个重要 caveat：

1. 机器是共享 H20，跑实验时其他 GPU job 也在同时运行，绝对耗时有抖动
2. `offline preload` 是 CPU / replay insert 成本，不是 learner update 本身；它和 bf16 没有稳定正相关

所以这轮更应该看：

- `offline pretrain total / avg sec per update`

而不是只看整个作业 wall time。

这轮的保守结论是：

1. 在正式 `train_residual_step.py` learner 路径里，打开 bf16 的收益是明确正的
2. 共享机器上按这轮数据看，收益区间大致在 `36% ~ 67%`
3. 如果把 2-step 和 3-step 合并，当前更稳妥的工作数字可以先记成：

```text
约 54% step-time 改善
约 2.18x updates/s
```

### 15.5 当前建议

到这一步可以认为：

1. `examples/libero/configs/*.yaml` 打开 bf16 是值得的
2. LIBERO 正式脚本已经实际吃到：
   - `critic freeze`
   - `actor-only broadcast`
   - `vectorized crop`
   - `bf16`
3. 后续如果继续做正式路径验证，下一优先级更适合放在：
   - replay prefetch 正式接入
   - StepWindow fast sample 正式化
   - `torch.compile(actor_critic)` 加配置开关和 warmup
