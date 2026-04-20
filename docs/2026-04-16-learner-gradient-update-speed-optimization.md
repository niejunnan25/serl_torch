# Learner Gradient Update Speed Optimization Notes

## 文档信息

- 文档类型：learner 梯度更新速度瓶颈分析与优化建议
- 记录时间：北京时间 2026-04-16
- 主要参考 run：
  `examples/agibot_real/output/2026-04-15_17-06-53`
- 主要代码路径：
  `examples/agibot_real/scripts/run_residual_training.py`
- 可复用范围：
  `examples/libero/scripts/run_residual_training.py` 和当前实验用的
  `examples/libero/scripts/run_residual_training_optimized.py` 与 AgiBot learner 结构高度相似，本文大多数 learner 更新优化也适用。

## 0. 先说明这次日志的边界

用户给出的目录：

```text
examples/agibot_real/output/2026-04-15_17-06-53
```

这个目录里的 `summary.json` 和 `.hydra/config.yaml` 显示该 run 是 `runtime.role=actor`，不是 `learner`。目录里也没有 `learner_timers.jsonl`，只有：

- `actor_timers.jsonl`
- `episode_logs.jsonl`
- `run_residual_training.log`
- `summary.json`

所以这个 run 不能直接给出 learner 每个训练阶段的实测耗时。但它仍然有价值，因为它给出了同一套训练配置下 actor 侧的数据吞吐和关键配置。

从该 run 的 actor timer 粗略看：

- `sample_actions` 平均约 `0.016s`
- `step_env` 平均约 `2.124s`
- `build_decision_obs` 平均约 `2.043s`
- `total` 平均约 `4.181s`
- `total` 中位数约 `2.938s`

当前 `chunk_horizon=15`，所以 actor 正常运行时约是每个 chunk 2.9 秒量级，即大约 `5 env steps/s`。这给 learner 一个很直观的压力参考：如果 learner 每个 env step 至少想做接近 1 次 online update，那 learner 端更新吞吐需要明显高于当前默认配置下的实际能力。

## 1. 当前 learner 一次 update 到底做了什么

AgiBot learner 的核心更新入口在：

```text
examples/agibot_real/scripts/run_residual_training.py
```

关键函数是 `_run_training_update(...)`。

当前配置里和速度最相关的字段是：

```yaml
obs:
  image_keys: [image_rgb_0, image_rgb_1, image_rgb_2]
residual:
  chunk_horizon: 15
encoder:
  type: resnet
  shared: true
  use_proprio: true
  resnet:
    model_name: microsoft/resnet-18
    pretrained: true
    freeze_backbone: false
sac:
  critic_ensemble_size: 2
  critic_subsample_size: 2
  utd_ratio: 2
replay:
  batch_size: 128
training:
  critic_actor_ratio: 4
  mixed_precision:
    enabled: false
    dtype: bfloat16
```

当前 `_run_training_update(...)` 的结构是：

```python
for _ in range(max(0, critic_actor_ratio - 1)):
    batch = sample_mixed_training_batch(...)
    agent, _ = agent.update_critics(batch)

batch = sample_mixed_training_batch(...)
agent, update_info = agent.update_high_utd(batch, utd_ratio=cfg.sac.utd_ratio)
```

在默认配置下：

- `critic_actor_ratio=4`
- `utd_ratio=2`

所以一次外层 `update_steps += 1` 实际包含：

- 3 次单独的 `update_critics(...)`
- 1 次 `update_high_utd(...)`

而 `update_high_utd(...)` 在 `serl_launcher/serl_launcher/agents/continuous/sac.py` 里又会：

- 把 batch 按 `utd_ratio` 切成 2 个 minibatch
- 对每个 minibatch 做 1 次 critic update
- 最后用完整 batch 做 1 次 actor + temperature update

也就是说，默认一次 learner 外层 update 实际是：

```text
5 次 critic 梯度更新
1 次 actor 梯度更新
1 次 temperature 梯度更新
4 次 replay sample
多次 obs / next_obs 图像增强
```

这点很重要。日志里的 `update_steps` 不是“单次反向传播次数”，而是“一个包含多次 critic 更新的外层训练迭代计数”。如果只看 `updates_per_sec`，会低估每个 update 背后的实际计算量。

## 2. 本地微基准观察

由于给定 run 没有 learner timer，我用当前代码结构在同机 H20 GPU 上做了一个小型微基准。这个基准使用同一套 AgiBot learner 网络结构、3 路 `224x224` 图像、ResNet-18、`batch_size=128`、`critic_actor_ratio=4`、`utd_ratio=2`。

这个微基准不是严格生产 profile，但足够判断瓶颈量级。

### 2.1 默认配置

```text
sample_mixed_training_batch mean: 0.054s / batch
update_critics mean:             0.537s
update_high_utd mean:            0.961s
outer update mean:               2.789s
```

结论：

- replay sample 不是第一瓶颈，约占 outer update 的 `7% - 8%`
- 主要时间花在模型前向、反向、optimizer step 上
- 默认 outer update 约 `0.36 updates/s`

如果 actor 正常可产生约 `5 env steps/s`，而 learner 逻辑又要求 `online_update_steps < env_steps`，那么当前 learner 很容易变成追赶状态，而不是轻松跟上。

### 2.2 开 bf16 mixed precision

H20 支持 bf16，但当前配置默认关闭：

```yaml
training:
  mixed_precision:
    enabled: false
```

微基准观察：

```text
outer update: 2.789s -> 2.339s
显存峰值:     17.1GB -> 9.3GB
```

收益：

- 约 `16%` 速度提升
- 显存峰值下降约 `45%`

这是当前最值得优先打开的低侵入优化。

### 2.3 actor loss 阶段冻结 critic 参数

当前 actor loss 需要通过 critic 计算 `Q(s, pi(s))`，但 actor update 不应该更新 critic 参数。

现在代码路径是：

```text
SACAgent.update(...)
  if "actor" in networks_to_update:
      self.state.zero_grad(["actor"])
      actor_loss, actor_info = self.policy_loss_fn(batch)
      actor_loss.backward()
      self.state.optimizer_step("actor")
```

问题是：`policy_loss_fn(...)` 内部会调用 critic forward。因为 critic 参数默认仍然 `requires_grad=True`，所以 actor backward 阶段会给 critic 参数也算梯度。之后只 step actor optimizer，critic 的这部分梯度不会被使用。

注意：这里不能简单用 `torch.no_grad()` 包住 critic forward，因为 actor loss 仍然需要 `dQ/da` 传回 policy action。正确做法是临时把 critic 参数设成 `requires_grad_(False)`，但保留对 action 的梯度路径。

微基准观察：

```text
update_high_utd: 0.968s -> 0.769s
```

收益约 `20.5%`。这属于语义基本不变、收益很实在的代码优化。

### 2.4 冻结 ResNet backbone

当前配置：

```yaml
encoder:
  resnet:
    freeze_backbone: false
```

如果冻结 backbone，微基准观察：

```text
update_high_utd default:          0.963s
update_high_utd freeze backbone:  0.553s
update_high_utd freeze + bf16:    0.505s
```

这是最大单项加速之一。但它改变训练行为：视觉 backbone 不再随在线数据更新，只训练 pooling/head/MLP 等后续模块。

适合场景：

- 真机相机位比较固定
- 预训练 ResNet 特征已经够用
- 当前主要目标是让 learner 跟上 actor
- 可以接受用一点视觉适应能力换更新吞吐

不适合场景：

- 视觉分布和预训练差异很大
- 希望在线微调视觉特征
- 当前样本效率比吞吐更重要

### 2.5 去掉 DrQ random crop

当前 DrQ 图像增强在：

```text
serl_launcher/serl_launcher/agents/continuous/drq.py
serl_launcher/serl_launcher/vision/data_augmentations.py
```

`batched_random_crop(...)` 当前是 Python for loop 逐样本 crop：

```python
crops = []
for i in range(flat.shape[0]):
    crops.append(random_crop(flat[i], ...))
out = torch.stack(crops, dim=0)
```

微基准观察：

```text
update_high_utd: 0.966s -> 0.888s
```

完全去掉增强约 `8%` 提升。这不建议作为默认方案，因为会改变 DrQ 正则化效果。但它说明 crop 增强有可优化空间，尤其是可以做向量化 crop，而不是逐样本 Python 循环。

## 3. 当前主要瓶颈排序

按当前证据，瓶颈优先级大致是：

1. 模型计算量过大
2. bf16 没打开
3. actor update 里白算 critic 参数梯度
4. ResNet backbone 可训练带来的巨大反向开销
5. learner -> actor 广播 payload 过大
6. random crop 图像增强不是向量化实现
7. replay sample 和 CPU->GPU 搬运有优化空间，但不是第一瓶颈
8. checkpoint、W&B、JSONL logging 是周期性开销，不是常规 step 主瓶颈

## 4. 可以立刻做的低风险优化

### 4.1 开启 bf16 mixed precision

建议先尝试：

```bash
bash examples/agibot_real/tools/run_learner.sh \
  runtime.role=learner \
  training.mixed_precision.enabled=true \
  training.mixed_precision.dtype=bfloat16
```

或 LIBERO 对应入口加同样 override。

预期收益：

- 单次 outer update 约 `10% - 20%` 加速
- 显存明显下降

风险：

- bf16 数值通常比 fp16 稳，H20 支持 bf16
- 仍建议观察 `critic_loss`、`actor_loss`、`temperature`、`q_predicted_gap` 是否异常

### 4.2 actor update 阶段冻结 critic 参数

建议在 `SACAgent.update(...)` 的 actor update 段临时冻结 critic 参数：

```python
critic_params = list(self.state.modules["critic"].parameters())
prev_requires_grad = [p.requires_grad for p in critic_params]
for p in critic_params:
    p.requires_grad_(False)
try:
    self.state.zero_grad(["actor"])
    with self._autocast_context():
        actor_loss, actor_info = self.policy_loss_fn(batch)
    actor_loss.backward()
    self.state.optimizer_step("actor")
finally:
    for p, requires_grad in zip(critic_params, prev_requires_grad):
        p.requires_grad_(requires_grad)
```

关键点：

- 不要用 `torch.no_grad()` 包住 critic forward
- 只冻结 critic 参数梯度
- 保留 `Q` 对 action 的梯度，这样 actor 仍然能通过 critic 得到策略梯度

预期收益：

- actor update 阶段约 `20%` 加速
- outer update 总收益取决于 `critic_actor_ratio` 和 `utd_ratio`

风险：

- 语义上与 SAC 常规做法一致
- 需要加一个小单测或 smoke test，确保 actor 参数仍有梯度、critic 参数在 actor-only update 不产生新梯度

### 4.3 learner -> actor 广播只发 actor 参数

当前 `server.publish_network(...)` 使用：

```python
snapshot_agent_checkpoint_payload(agent, step=int(update_steps))
```

而 `snapshot_agent_checkpoint_payload(...)` 会克隆：

- `params`
- `target_params`
- `optimizer`

这对 checkpoint 保存是合理的，但对 actor 网络广播过重。actor 采样动作只需要 actor 参数，通常不需要：

- critic 参数
- target critic 参数
- optimizer state
- temperature optimizer state

实测 payload 大小量级：

```text
初始化后完整 payload: 约 720MB
一次 update 后完整 payload: 约 843MB
actor params: 约 144MB
critic params: 约 288MB
target critic params: 约 288MB
critic optimizer state: 约 120MB
```

建议新增一个轻量 payload：

```python
def snapshot_actor_network_payload(agent, *, step: int) -> dict[str, Any]:
    return {
        "step": int(step),
        "params": {
            "actor": clone_to_cpu(agent.state.modules["actor"].state_dict()),
        },
    }
```

当前 `apply_checkpoint_payload_to_agent(..., load_optimizers=False)` 已经是按 payload 里存在的模块逐个加载，所以 actor-only payload 从结构上是可兼容的。

预期收益：

- 降低周期性 publish 的 CPU clone 时间
- 降低网络传输和 actor 接收 load 时间
- 减少训练过程中因广播造成的 jitter

风险：

- 如果 actor 端某些逻辑未来依赖 temperature 或 target critic，需要额外确认
- 当前 actor 的 `sample_action(...)` 只走 actor module，这个优化是合理的

### 4.4 调低 `critic_actor_ratio` 或 `utd_ratio`

如果目标是“learner 更新吞吐更快”，最直接的结构性旋钮是：

```yaml
training:
  critic_actor_ratio: 4
sac:
  utd_ratio: 2
```

默认一次 outer update 是 5 次 critic update。如果改成：

```yaml
training.critic_actor_ratio=2
sac.utd_ratio=1
```

一次 outer update 就会接近：

```text
2 次 critic update
1 次 actor update
1 次 temperature update
```

这会显著提升 `updates_per_sec`，但也会改变算法的 critic/actor 更新比例。

建议实验顺序：

```text
保守实验：critic_actor_ratio=3, utd_ratio=2
中等实验：critic_actor_ratio=2, utd_ratio=2
激进实验：critic_actor_ratio=2, utd_ratio=1
```

观察指标：

- `updates_per_sec`
- `critic_loss`
- `actor_loss`
- `predicted_q_gap`
- 成功率曲线
- actor 端策略刷新后的行为稳定性

## 5. 中等风险但收益很大的优化

### 5.1 冻结 ResNet backbone

建议作为单独实验开关，而不是直接改默认：

```bash
encoder.resnet.freeze_backbone=true
```

预期收益：

- `update_high_utd` 可能接近 `40% - 50%` 加速
- 显存和 backward 开销明显下降

建议实验方式：

- 跑一组只开 `bf16`
- 跑一组 `bf16 + freeze_backbone`
- 对比成功率、Q 值稳定性、loss 曲线

如果冻结 backbone 后性能没有明显下降，这可能是当前真机训练最划算的速度优化。

### 5.2 分阶段冻结 / 解冻 backbone

比永久冻结更稳的策略：

```text
前 N 个 update 冻结 backbone
replay 规模足够后再低学习率解冻
或每 K 次 update 才允许 backbone 更新一次
```

这样可以先提高早期吞吐，又保留后期视觉适应能力。

实现上需要更细粒度的 optimizer param group 或 requires_grad 调度。

### 5.3 降低视觉输入成本

当前是 3 路 `224x224` 图像。可尝试：

```text
只用关键相机：3 路 -> 2 路或 1 路
降低分辨率：224 -> 160 或 128
换更轻 encoder：resnet18 -> small encoder
```

这类优化会改变 observation 表达能力，风险高于 bf16 和冻结无用梯度，但通常收益很大。

推荐先不要一次性改多项，避免不知道到底是哪项影响了训练。

## 6. 数据和增强管线优化

### 6.1 向量化 `batched_random_crop`

当前 `batched_random_crop(...)` 是逐样本 Python loop。可以改成一次性 pad，然后用 batch index + 随机 offset 做 gather / advanced indexing。

目标：

- 避免 `B * image_keys * obs_next` 次 Python 函数调用
- 避免每个样本单独创建 generator
- 减少 GPU kernel launch 碎片

当前收益估计：

- 完全去掉 augmentation 可带来约 `8%` 的 `update_high_utd` 加速
- 向量化 crop 不会拿满这 8%，但应该能回收其中一部分

风险：

- 需要确保输出与原始 crop 语义一致
- 需要确保 batch 维度兼容 `num_batch_dims=2`

### 6.2 replay sample 预取

当前 `_run_training_update(...)` 是同步顺序：

```text
sample batch
GPU update
sample batch
GPU update
...
```

可以增加 learner-side prefetch queue：

```text
后台线程提前 sample + reshape
主线程拿 ready batch 做 update
```

这个优化主要隐藏 CPU replay sample 时间。因为 replay sample 目前约 `0.05s/batch`，一次 outer update 采样 4 次，理论上最多隐藏约 `0.2s` 量级。

风险：

- replay buffer 当前有 lock，需要确认 prefetch 线程和 agentlace insert 的锁竞争
- offline/online mixed replay 也要一起处理
- 预取太深会让 batch 稍微更 stale，但 off-policy 通常可接受

### 6.3 pinned memory + non_blocking H2D

当前 `_to_torch(...)` 直接把 numpy batch 转 tensor 并 `.to(device)`。

可以考虑：

- CPU batch 放 pinned memory
- `.to(device, non_blocking=True)`
- 和 CUDA stream / prefetch queue 搭配

但这需要更系统的数据管线改造。就当前证据看，它不是第一优先级。

## 7. 同步、日志和 checkpoint 优化

### 7.1 区分 checkpoint payload 和 network broadcast payload

现在 checkpoint 保存和网络广播都复用完整 checkpoint payload，这是不理想的。

建议拆成两个概念：

```text
checkpoint payload:
  用于恢复训练，包含 actor/critic/target/optimizer

actor broadcast payload:
  用于 actor 采样，只包含 actor 参数和 step
```

这样既不影响 checkpoint 恢复能力，又能显著降低 actor 同步成本。

### 7.2 减少 publish 频率或异步 publish

当前 learner 每 `steps_per_update` 个 update publish 一次：

```python
if update_steps % steps_per_update == 0:
    server.publish_network(...)
```

配置里 `steps_per_update=30` 同时也被 actor 用于 `client.update()` 频率。这个字段现在语义有点复用：

- actor 侧：多少 env steps 拉一次网络
- learner 侧：多少 learner updates 广播一次网络

优化方向：

- 拆成 `actor_sync_env_period` 和 `learner_publish_update_period`
- 允许 learner publish 更低频，但 actor 可以更频繁 poll 最新轻量 payload
- 或者让 publish clone 在后台线程做，避免阻塞 learner 更新主循环

### 7.3 checkpoint 保存不要和训练主循环抢时间

当前 `checkpoint_every=2500`，频率不高，但完整 checkpoint 大小可能到 GB 级别。保存时会 `torch.save(...)`，可能造成明显 pause。

建议：

- checkpoint 保存走后台线程或后台进程
- 或先 actor-only broadcast，完整 checkpoint 只保留低频保存
- 训练日志里单独记录 checkpoint 保存耗时

## 8. 需要补的 profiler / timer

当前 learner timer 只有：

- `sample_replay_buffer`
- `train_critics`
- `train`

这还不够定位细节。建议补更细粒度 timer：

```text
sample_online_replay
sample_offline_replay
concat_mixed_batch
reshape_chunk_batch
to_torch
augment_observations
critic_forward_backward
critic_optimizer_step
target_update
actor_forward_backward
temperature_update
snapshot_actor_payload
snapshot_checkpoint_payload
publish_network
checkpoint_save
wandb_log
jsonl_log
```

对 CUDA 计算阶段，建议用 `torch.cuda.Event` 或在 timing 前后显式 `torch.cuda.synchronize()`，否则普通 `time.time()` 可能只测到 kernel launch 时间，而不是真实 GPU 完成时间。

建议新增一行更明确的 learner heartbeat：

```text
learner heartbeat:
  updates_per_sec=...
  critic_updates_per_sec=...
  env_lag=env_steps-online_update_steps
  replay_sample_ms=...
  train_ms=...
  publish_ms=...
```

这样以后就不用靠猜了。

## 9. 推荐实施路线

### Phase 1：低风险、立刻收益

建议先做：

```text
1. 打开 bf16 mixed precision
2. actor update 阶段冻结 critic 参数
3. network publish 改为 actor-only payload
4. 增加 publish / checkpoint / augmentation / to_torch 细粒度 timer
```

预期：

- 不显著改变算法语义
- 能马上降低单次 update latency
- 能把后续瓶颈看得更清楚

### Phase 2：结构性降计算

建议做独立 ablation：

```text
1. freeze_backbone=true
2. critic_actor_ratio 从 4 降到 3 或 2
3. utd_ratio 从 2 降到 1
```

每次只改一个主要变量，避免混淆结果。

### Phase 3：数据管线优化

建议在 Phase 1 和 Phase 2 后再做：

```text
1. vectorized batched_random_crop
2. replay sample prefetch queue
3. pinned memory + non_blocking transfer
```

这些优化更偏工程实现，收益通常不如直接减少模型反向计算，但能进一步抹平 latency。

### Phase 4：更激进的模型优化

如果仍然跟不上 actor，再考虑：

```text
1. 减少相机路数
2. 降低图像分辨率
3. 换 small encoder
4. torch.compile / channels_last / cudnn benchmark
5. 多 GPU learner 或分离 encoder/critic 计算
```

这些都有更明显的训练行为风险，建议放在后面。

## 10. 最推荐的前三个改动

如果只做三件事，我建议按这个顺序：

```text
1. training.mixed_precision.enabled=true
2. actor loss 阶段冻结 critic 参数梯度
3. learner publish actor-only payload
```

原因：

- 第 1 个几乎是配置级收益，H20 很适合 bf16
- 第 2 个修掉了真实存在的无用反传
- 第 3 个修掉了 actor 同步路径上的超大 payload

如果可以接受算法行为变化，再加：

```text
4. encoder.resnet.freeze_backbone=true
5. critic_actor_ratio / utd_ratio ablation
```

这两项可能是吞吐提升最大的，但需要用成功率和 loss 曲线确认不会牺牲太多训练质量。
