# Residual RL 当前问题与隐患记录

本文档记录当前分支 `feat/libero-obs-only-rawstate` 上，残差强化学习流程里已经实现的功能、仍然存在的语义风险、以及一些容易被误解但并非当前 blocker 的点。

适用配置：

- 完整训练 YAML：
  `examples/libero/conf/ablation_stepchunk_vs_step_task6_alpha_warmup_utd_sweep/train_residual_sac_ablation_stepchunk_alpha50_warmup100ep_utd4_onlineprefill_obs_only_rawstate_epsgating.yaml`

当前这份完整配置已经包含：

- `residual.observation.state_mode: raw`
- `residual.epsilon_gating.enabled: true`
- `normalization.enabled: false`

## 1. `state_mode=raw` 的真实语义

### 当前实现是什么

`state_mode=raw` 当前表示：

- 不再把 `base_action`
- 不再把 `base_action_chunk`
- 不再把 `alpha`

拼到 `observations["state"]` 里。

对应代码：

- `examples/libero/runtime/obs_adapter.py`
- `examples/libero/utils/step_chunk_replay.py`

也就是说，当前 `raw` 更准确的意思是：

- `state = state_core`

而不是：

- `state = 完全独立于 normalizer 的原样状态`

### 为什么这会有歧义

`state_core` 的构造仍然走：

- `build_libero_state(..., normalizer=normalizer)`

对应：

- `examples/libero/runtime/obs_adapter.py`

只要 `normalizer` 不为空，`build_libero_state()` 就会调用：

- `normalizer.normalize_state(state)`

因此：

- 当 `normalization.enabled=false` 时，当前运行拿到的确实是未归一化 proprio。
- 但当以后重新打开 `normalization.enabled=true` 时，`state_mode=raw` 仍会经过 `normalize_state`。

### 当前是否是 blocker

不是当前 blocker。

因为你当前明确要求：

- 直接关闭归一化

所以在当前这条实验线上，`raw` 运行时语义是对的。

### 后续隐患

这个点最大的风险不是“现在跑错了”，而是：

- 后面别人看到 `state_mode=raw`
- 很容易误以为它是“绝对 raw state”
- 但实际上它仍受 `normalization.enabled` 影响

如果后续要把这个语义彻底做干净，应该拆成：

- `state_mode`: 控制是否拼 `base/base_chunk/alpha`
- `state_normalization`: 单独控制 state 是否归一化

## 2. 当前实现的是“第二版”，不是 paper-style 的 actor-only A/B

### 当前实现是什么

当前分支实现的是你后来明确指定的“第二版”：

- actor 输入：`image + raw proprio`
- critic 输入：`image + raw proprio + final_action`

也就是说，actor 和 critic 现在共用同一个 `observations["state"]` 语义，只是这个 `state` 已经从 fused state 换成了 raw proprio。

相关代码：

- `serl_launcher/serl_launcher/common/encoding.py:81`
- `serl_launcher/serl_launcher/networks/actor_critic_nets.py:308`
- `serl_launcher/serl_launcher/agents/continuous/sac.py:215`

### 不是哪一种

它不是更早讨论过的那种：

- actor 用 obs-only
- critic 仍保留 fused-state

也就是不是“只改 actor、critic 先不动”的 paper-style A/B。

### 当前是否需要把它当成问题

不需要。

对当前目标来说，这不是 bug，也不是缺失功能。它只是一个范围说明：

- 我们当前实现的是你明确拍板的第二版
- 不是第一版

如果后面你不想做“actor-only obs-only，critic 保持 fused-state”，那这一点就不应继续当成待办。

## 3. `shared_encoder=true` 与 `actor stop_gradient=True` 到底意味着什么

### 当前数据流

当前 actor / critic 的主干可以概括为：

1. 图像输入
   - `image`, `wrist_image`
   - 进 ResNet encoder
   - 得到 image latent

2. state 输入
   - `observations["state"]`
   - 进 `LazyLinear + LayerNorm + tanh`
   - 得到 proprio latent

3. 拼接
   - `image latent + proprio latent`

4. actor
   - 拼接特征进 policy MLP
   - 输出 residual action 分布

5. critic
   - 拼接特征进 critic
   - 再和 `final_action` 结合
   - 输出 `Q(s, a)`

相关代码：

- `serl_launcher/serl_launcher/common/encoding.py:62`
- `serl_launcher/serl_launcher/common/encoding.py:81`
- `serl_launcher/serl_launcher/networks/actor_critic_nets.py:303`
- `serl_launcher/serl_launcher/agents/continuous/sac.py:574`

### `shared_encoder=true` 的影响

当 `shared_encoder=true` 时：

- actor 和 critic 共用同一个 `EncodingWrapper`
- 也就是共用同一套图像 encoder 和 proprio 投影层参数

对应：

- `serl_launcher/serl_launcher/agents/continuous/drq.py:154`
- `serl_launcher/serl_launcher/agents/continuous/drq.py:162`

这意味着当前结构并不是“完全解耦的 actor / critic 编码器”。

### `actor stop_gradient=True` 的影响

Policy 前向里，actor 调用 encoder 时用了：

- `stop_gradient=True`

对应：

- `serl_launcher/serl_launcher/networks/actor_critic_nets.py:308`

而 `EncodingWrapper` 会在这种情况下：

- 对 image latent `detach()`
- 对 proprio latent `detach()`

对应：

- `serl_launcher/serl_launcher/common/encoding.py:74`
- `serl_launcher/serl_launcher/common/encoding.py:93`

因此当前真实效果是：

- actor 使用 encoder 产出的特征
- 但 actor 的梯度不会回传去更新 encoder
- encoder 参数主要由 critic 这一路来更新

### 这会造成什么影响

从“数据输入输出流”的角度看：

- actor 和 critic 看的是同一套编码特征空间
- 但这套特征空间的学习方向，主要受 critic 驱动

这不是 bug，也不违背当前第二版目标，但它确实意味着：

- 当前实验不是“actor/critic 完全独立建模”的结构试验
- 更像是“共享表征 + critic 驱动表征 + actor 在该表征上出 residual”

如果以后要做更纯粹的结构消融，可能还需要进一步比较：

- `shared_encoder=true`
- `shared_encoder=false`

## 4. 论文 / 官方实现到底是不是 “actor obs-only, critic sum(base,residual)”

### 结论

是的，论文正文和 Appendix E 都明确这么写。

OpenReview 论文 PDF 在 Sec. 4.3 写道：

- residual policy 的输入可选 `observation alone` 或 `observation + base action`
- 他们的实验显示 `observation alone` 通常更好
- critic 的 action 输入可选：
  - residual only
  - concat(base, residual)
  - sum(base, residual)
- 他们的实验显示 `sum(base, residual)` 最好

可直接查看：

- OpenReview PDF: https://openreview.net/pdf?id=e5jGTEiJMT
- arXiv 摘要页: https://arxiv.org/abs/2412.13630

在 OpenReview PDF 中可定位到：

- Sec. 4.3 `Design Choices & Implementation Details`
- Appendix E.1 `Input of Residual Policy`
- Appendix E.2 `Input of Critic`

其中明确写到：

- “using only observation typically produces better results”
- “using the sum of both actions yields the best performance”

### 官方代码是不是也这么实现

官方仓库 `third_party/policy_decorator` 里，命令行参数直接暴露了这两个设计轴：

- `--actor-input {obs, obs_base_action}`
- `--critic-input {res, sum, concat}`

对应：

- `third_party/policy_decorator/online/pi_dec_bet_maniskill2.py:84`
- `third_party/policy_decorator/online/pi_dec_bet_maniskill2.py:85`

并且默认值就是：

- `actor-input = obs`
- `critic-input = sum`

同文件训练逻辑里也明确体现为：

- actor 输入可只用 observation
- critic 输入默认用 `base_action + scaled_residual`

对应：

- `third_party/policy_decorator/online/pi_dec_bet_maniskill2.py:509`
- `third_party/policy_decorator/online/pi_dec_bet_maniskill2.py:513`
- `third_party/policy_decorator/online/pi_dec_bet_maniskill2.py:526`
- `third_party/policy_decorator/online/pi_dec_bet_maniskill2.py:528`

### 这对当前分支的含义

这说明我们当前的“第二版”并不是在复刻论文的那条 actor-only A/B。

当前分支做的是：

- actor 和 critic 都改成吃 raw proprio
- critic 仍然用 final action 评估

所以：

- 当前实现是受论文启发
- 但不是论文那条最干净的结构消融复现

## 5. `warmup=100 but do not write replay` 的意义

### 当前代码实际做了什么

当前 warmup 不是“纯跑 100 个 episode 不留痕迹”，而是：

- 用 base-only 动作执行 warmup
- 构造 transition
- 写入 online replay buffer

对应：

- `examples/libero/scripts/train_residual_sac.py:1177`
- `examples/libero/scripts/train_residual_sac.py:1187`
- `examples/libero/scripts/train_residual_sac.py:1311`
- `examples/libero/scripts/train_residual_sac.py:1323`

而且 warmup 阶段记录的是：

- `alpha = 0.0`
- `alpha_obs = 0.0`
- `a_res_policy = 0`
- `a_final = a_base`

### 如果“不进 replay”，那 warmup 还有什么意义

在你当前这套系统里，base policy 是无状态推理模型，warmup 又不更新网络参数，因此：

- 如果 warmup 不写 replay
- 也不触发任何学习

那么它几乎没有算法上的收益。

更准确地说，它只剩下下面这些作用：

- 诊断用：
  - 分离“仅仅跑了一段 base-only 前缀”与“base-only 数据污染 replay”这两个因素
- 工程用：
  - 提前把 env / server / OpenPI 通路跑热
  - 排查启动阶段的问题

但它不会给 agent 带来真正的学习信号。

### 结论

所以：

- `warmup but do not write replay`

在当前系统里更像一个诊断开关，不是推荐的训练策略。

如果目的只是训练 residual policy，本身没有太强意义。

## 6. “alpha 本身的重新设计还没动” 到底是什么意思

### 当前已经动了什么

当前分支已经接入了两件事：

1. `obs-only/rawstate`
2. `epsilon-gating`

也就是：

- 改了“网络看什么”
- 改了“行为策略何时启用 residual”

### 当前还没动什么

`alpha` 本身的设计仍然沿用旧逻辑，主要体现在：

1. 主配置仍然是
   - `residual.alpha: 0.5`
   - 见当前完整 YAML：`line 44`

2. `alpha_scheduler` 仍然关闭
   - `training.alpha_scheduler.enabled: false`
   - 见当前完整 YAML：`line 217`

3. offline residual 标签的投影方式没改
   - residual 仍按 `limits * alpha * expert_reference_scale` 归一化 / 裁剪

4. critic 侧 residual -> final action 的变换逻辑没改
   - 仍然是 `base_action + residual * limits * alpha`

对应：

- `serl_launcher/serl_launcher/agents/continuous/sac.py:215`
- `serl_launcher/serl_launcher/agents/continuous/sac.py:581`

### 这为什么算“还没动”

因为你前面一直在讨论的核心问题之一，其实就是：

- `alpha=0.5` 会不会太大
- `alpha` 是否应该重新选值
- 是否应该和 gating 解耦
- 是否要做新的调度或新的探索设计

而当前这条 merged 分支并没有回答这些问题。它只是把：

- `rawstate`
- `epsilon-gating`

先接上了。

换句话说：

- 现在我们改变了 residual policy 的输入语义
- 也改变了 residual 的启用概率
- 但 residual 的幅度上限 `alpha` 仍然沿用旧设定

### 当前隐患

这意味着：

- 如果后面训练仍然不稳
- 不能自动说明 `rawstate` 或 `gating` 没用

因为 `alpha=0.5` 本身仍然可能过大。

## 7. 当前建议如何理解这些点

如果从“当前功能开发是否完成”来看：

- `state_mode=raw`：已可用
- `epsilon-gating`：已可用
- 完整训练流程：已 smoke test 跑通

如果从“后续是否还有概念性风险”来看，最需要记住的是：

1. `raw` 仍受 `normalization.enabled` 影响
2. 当前实现是你指定的第二版，不是论文的 actor-only A/B
3. `shared_encoder + actor stop_gradient` 意味着表征仍主要由 critic 驱动
4. `warmup without replay` 只适合作诊断，不是强训练策略
5. `alpha` 问题目前并没有被这次功能开发解决
