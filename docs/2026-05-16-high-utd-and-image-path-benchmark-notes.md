# High-UTD 编译策略与图像路径基准记录

日期：2026-05-16

实验分支：

```text
bench/high-utd-image-path-benchmarks
```

## 当前 ResNet18 是否已经编译

当前 `spatial_4_0514_runtime/spatial4_scripts_2_alpha0p1_unfiltered_offline_noent_std0p5_ports53500` 这条训练配置里，Torch ResNet18 所在的 actor/critic 路径是会被 `torch.compile` 编译的。

配置来源：

```yaml
training:
  mixed_precision:
    enabled: true
    dtype: bfloat16
  torch_compile:
    enabled: true
    target: actor_critic
    backend: inductor
    mode: default
    fullgraph: true
    dynamic: false
```

训练脚本会调用：

```python
agent = apply_torch_compile(
    agent,
    compile_cfg=cfg.training.torch_compile,
)
```

`apply_torch_compile` 会编译：

- `agent.state.modules["critic"]`
- `agent.state.target_modules["critic"]`
- `agent.state.modules["actor"]`

所以当前生产路径不是未编译 ResNet18；它是“模块级编译后的 ResNet18 + Python high-UTD 外循环”。

## High-UTD 三档是什么意思

这三档是递进关系，核心区别是：把多少训练步骤从 Python 边界里拿出来，交给 PyTorch 编译器或 CUDA runtime 一次性处理。

不过实际测试时，它们是三种互斥策略，不是简单叠加开关。也就是说，我们分别跑：

- 当前基线
- 第一档
- 第二档
- 第三档

然后比较哪个更值得做。

### 当前基线：模块级编译

当前生产路径大致是：

```python
critic = torch.compile(critic)
target_critic = torch.compile(target_critic)
actor = torch.compile(actor)

for i in range(utd_ratio):
    loss = critic_loss(minibatch_i)
    loss.backward()
    optimizer.step()
    target_update()
```

也就是说，模型模块本身被编译了，但 high-UTD 的循环、反向传播、优化器 step、target update 仍然由 Python 一步一步调度。

### 第一档：编译 critic loss

第一档把更大的 loss 前向函数编译起来：

```python
compiled_loss = torch.compile(critic_loss)

for i in range(utd_ratio):
    loss = compiled_loss(minibatch_i)
    loss.backward()
    optimizer.step()
    target_update()
```

它比当前模块级编译多吞掉了一些东西，比如 target 计算、loss reduction、若干张量操作。但 backward 和 optimizer 仍然在 Python 外面。

这档风险最低，语义最接近当前代码。

### 第二档：编译单个 critic train step

第二档尝试把单个 critic 更新步也编译起来：

```python
compiled_train_step = torch.compile(train_step)

for i in range(utd_ratio):
    compiled_train_step(minibatch_i)
```

其中 `train_step` 包括：

```python
optimizer.zero_grad()
loss = critic_loss(minibatch)
loss.backward()
optimizer.step()
target_update()
```

这档更接近 JAX 的单步训练函数，但 PyTorch 对 optimizer、反向图、参数原地更新的编译支持没有 JAX 那么自然，所以风险更高。

### 第三档：CUDA graph 捕获整个 high-UTD pattern

第三档尝试把整个 high-UTD critic pattern 捕获成一个 CUDA graph：

```python
with torch.cuda.graph(graph):
    for i in range(utd_ratio):
        train_step(minibatch_i)

graph.replay()
```

这最接近“一次提交整段训练更新”，理论上能最大幅度减少 Python 调度开销。

但它限制也最硬：

- batch shape 必须固定；
- 不能有不受控的动态分配；
- optimizer 必须支持 capture；
- 随机数、target update、Lazy 参数创建都要提前处理；
- 某些 PyTorch/HF/Inductor/CUDNN 组合可能不能捕获。

## High-UTD fake benchmark

脚本：

```text
test/benchmark_high_utd_compile_strategies.py
```

结果文件：

```text
test/results/high_utd_compile_strategies_20260516.json
test/results/high_utd_cuda_graph_smoke_20260516.json
```

测试条件：

- GPU：NVIDIA H20，`CUDA_VISIBLE_DEVICES=5`
- batch size：128
- UTD：4
- 图像：两路 224x224 RGB
- action dim：35
- backbone：ResNet18，frozen
- mixed precision：bf16
- 权重：随机初始化，只比较计算路径
- 一个 timed unit：一次完整 high-UTD critic pattern，也就是 4 次 critic minibatch update

结果：

| 策略 | mean ms / pattern | pattern/s | 相对当前基线 |
|---|---:|---:|---:|
| eager Python，无编译 | 115.23 | 8.68 | 慢 69.4% |
| 当前基线，模块级编译 | 68.01 | 14.70 | 基线 |
| 第一档，loss 级编译 | 67.52 | 14.81 | 快 0.7% |
| 第二档，step 级编译 | 66.52 | 15.03 | 快 2.2% |
| 第三档，CUDA graph | 未通过 | 未通过 | 当前不可用 |

第三档在小图 smoke test 下失败，错误是 CUDA graph capture 期间底层 CUDA 操作失败，随后 PyTorch caching allocator 报 capture 状态异常。因此这条路线目前不适合直接作为主线优化。

### High-UTD 结论

当前正式训练已经启用 actor/critic 模块级 `torch.compile`，所以“再往上包一层 loss 或 train step”带来的新增收益很小。

这个 fake benchmark 里：

- 从 eager 到当前模块级编译，收益很大；
- 从当前模块级编译到第一档，只有约 0.7%；
- 从当前模块级编译到第二档，只有约 2.2%；
- 第三档暂时不兼容。

所以短期不建议优先重构 high-UTD 编译路径。它理论上重要，但在当前 PyTorch 2.3、HF ResNet18、bf16、模块级 compile 已开启的条件下，新增收益没有想象中大。

## 图像增强与布局四层优化

脚本：

```text
test/benchmark_image_path_layout_strategies.py
```

结果文件：

```text
test/results/image_path_layout_strategies_20260516.json
```

测试条件：

- GPU：NVIDIA H20，`CUDA_VISIBLE_DEVICES=5`
- batch size：128
- 图像：两路 224x224 RGB
- crop padding：4
- backbone：ResNet18，frozen
- pooling/head：spatial learned embeddings + 256 bottleneck
- mixed precision：bf16
- 权重：随机初始化
- 一个 timed unit：随机裁剪 + encoder 前向 + trainable pooling/head 反向

四层是累积关系：

1. `layer1_buffered_norm`：mean/std 变成 buffer，不再每次 forward 创建 tensor。
2. `layer2_nchw_layout`：crop 后保持 NCHW，encoder 直接吃 NCHW，减少 NHWC/NCHW 来回转换。
3. `layer3_fused_backbone`：两路 view 合并成 `B*V`，共享 backbone 一次前向，view-specific pooling/head 仍然分开。
4. `layer4_compiled_fused`：在第三层基础上 compile fused image path。

结果：

| 图像路径 | mean ms / step | step/s | 相对 baseline |
|---|---:|---:|---:|
| baseline_current | 29.82 | 33.54 | 基线 |
| layer1_buffered_norm | 29.55 | 33.84 | 快 0.9% |
| layer2_nchw_layout | 29.99 | 33.35 | 慢 0.6% |
| layer3_fused_backbone | 29.42 | 33.99 | 快 1.3% |
| layer4_compiled_fused | 18.93 | 52.83 | 快 36.5% |

### 图像路径结论

单独做 mean/std buffer、NCHW layout、两路 view fused backbone，在未编译条件下收益都很小，基本在噪声附近。

真正明显的是第四层：fused image path 再配合 `torch.compile`。这说明优化点不是单个 mean/std tensor 或一次 permute，而是需要给 Inductor 一个更大的、形状固定的图，让它把 crop、normalize、backbone 调用边界、pooling/head 这一段一起优化。

注意：上面的 `baseline_current` 是未编译图像路径。后续把 fused-view fast path 迁移进真实 `EncodingWrapper` 后，又单独测了 production wrapper 的 fullgraph compile 路径：

| 生产 wrapper 路径 | mean ms / step | step/s |
|---|---:|---:|
| `fuse_views=false` + compile | 22.29 | 44.86 |
| `fuse_views=auto` + compile | 22.05 | 45.35 |

真实 ResNet18 数值检查也通过：同一组权重、同一组输入，旧 per-view 路径和 fused 路径在 `train=false` 下输出 shape 都是 `(16, 512)`，最大绝对误差约 `1.3e-3`，平均绝对误差约 `1.9e-4`。这个误差来自 cuDNN 对不同 batch size 可能选择不同卷积实现，属于浮点实现差异，不是语义差异。

这说明当前正式训练的模块级 `torch.compile(fullgraph=true)` 已经能把旧 per-view wrapper 优化掉相当一部分开销；fused-view fast path 在生产编译路径上的新增收益约 1%。因此它是一个安全的结构性改进和回退机制，但不能指望它单独带来未编译 benchmark 里的 36% 端到端收益。

## 建议

短期优先级建议调整为：

1. 暂时不要优先做 high-UTD CUDA graph。当前兼容性差，而且第一档/第二档新增收益小。
2. fused-view fast path 已经可以安全迁移到生产代码，但当前 production wrapper compile 下新增收益偏小。
3. 迁移时不要先改 replay 存储格式，可以只在训练 update 内部把两路图像临时整理成 `B*V` 的 backbone 输入。
4. 保留 view-specific pooling/head 参数，避免改变模型语义。
5. 如果正式迁移，必须加等价性测试：同一组输入下，旧 per-view backbone 路径和新 fused backbone 路径输出 shape 一致，并检查数值差异在合理范围内。

最值得继续做的下一步是：用正式 agent fake batch 跑端到端 update benchmark，确认 fused-view fast path 在完整 actor/critic 更新里是否还有间接收益；如果仍然只有 1% 左右，就应把优化重点转回 replay/device overlap 或模型结构对齐。

## Target Update Frozen Skip

脚本：

```text
test/benchmark_target_update_frozen_skip.py
```

结果文件：

```text
test/results/target_update_frozen_skip_20260516.json
test/results/target_update_frozen_skip_compile_modules_20260516.json
```

测试对象是接近真实 LIBERO residual learner 的 critic：

- 两路 ResNet18 frozen backbone；
- proprio projection；
- 两个 Q head ensemble；
- action dim 35；
- 只计时 target critic soft update，不计 forward/backward。

参数统计：

| 参数类型 | tensor 数 | element 数 |
|---|---:|---:|
| 全部参数 | 96 | 14,256,258 |
| frozen 参数 | 60 | 11,176,512 |
| trainable 参数 | 36 | 3,079,746 |

未编译 module 包装下：

| target update 策略 | mean us | updates/s | 相对当前 |
|---|---:|---:|---:|
| 当前全量参数 loop | 1707.07 | 585.80 | 基线 |
| 跳过 frozen 参数 loop | 982.56 | 1017.75 | 快 42.4% |
| 跳过 frozen + foreach | 557.29 | 1794.40 | 快 67.4% |

`torch.compile` module 包装下：

| target update 策略 | mean us | updates/s | 相对当前 |
|---|---:|---:|---:|
| 当前全量参数 loop | 1839.17 | 543.72 | 基线 |
| 跳过 frozen 参数 loop | 1018.94 | 981.41 | 快 44.6% |
| 跳过 frozen + foreach | 569.64 | 1755.50 | 快 69.0% |

结论：target update 现在确实在白白扫 frozen ResNet18 backbone。只跳过双方都 frozen 的参数已经能把 target update 自身压掉约 40% 多；再用 foreach 批量更新 trainable 参数，可以压掉约 67%-69%。这个优化语义安全，值得优先实现。
