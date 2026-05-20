# KeyRL Stage1 信用分配与 OOD 讨论

日期：2026-05-20

## 背景

当前 KeyRL 第一版采用固定窗口 gate：

```text
stage1: [30, 75)
stage2: [110, 160)
```

在线 rollout 中，只有 active stage 会启用 residual policy 并写入 replay；base-only 阶段不调用 residual policy，也不写 residual replay。由于使用 active-only replay，stage 边界会切断 value bootstrap，避免 learner 在 next step 已经 inactive 的状态上继续假设 residual actor 会生效。

这带来了一个重要问题：stage window 不只是动作生效窗口，同时也是 residual learner 能看到的训练信号窗口。

## 当前观察

初步实验里出现了一个反直觉现象：

```text
single_stage [30,75) 成功率下降；
two_stage [30,75)+[110,160) 成功率更高，收敛更快。
```

从 stage1-only 的指标看：

```text
rollout/cumulative_success_rate: 约 0.65 -> 0.38
rollout/recent_success_rate_20: 后期长期掉到 0.0-0.3 区间

residual/mean_abs: 约 0.52 -> 0.75+
residual/max_abs: 长期接近 1.0
residual/saturation_rate: 约 0.05 -> 0.2-0.4
action/mean_abs_delta_from_base: 约 0.10 -> 0.14-0.15

learner/q_predicted_mean: 接近 0
learner/q_target_mean: 接近 0
learner/q_actor_predicted_mean: 接近 0
```

这说明 stage1-only 不是完全没有变化，而是 residual 越学越强、越来越接近饱和，但 critic 没有给 actor 明确的正价值信号。

## Base Rollout 轨迹分析

对 `/Users/n/Downloads/10000-records` 的 base policy rollout 做过一次分析：

```text
总轨迹数: 50
成功轨迹数: 33
失败轨迹数: 17
base success rate: 66%
```

成功轨迹的成功 step，按 policy step 计：

```text
mean = 130.0
median = 126
q10 = 116.2
q75 = 136
q90 = 140.4
min = 112
max = 190
```

按当前窗口统计，成功 reward 落在：

```text
stage1 [30,75):       0 / 33
gap [75,110):         0 / 33
stage2 [110,160):    32 / 33
after stage2:         1 / 33
```

一个成功样例是：

```text
/Users/n/Downloads/10000-records/episode_2_True.pkl
/Users/n/Downloads/10000-records/episode_2_True.mp4
```

该轨迹信息：

```text
task_id = 4
task = pick up the black bowl in the top drawer of the wooden cabinet and place it on the plate
policy action length = 113
first positive gripper change ~= policy step 38
first negative gripper change ~= policy step 60
success reward = policy step 112
```

这条轨迹说明 stage1 确实覆盖了接近、接触、夹取、拿出抽屉等敏感阶段；但最终成功 reward 明确发生在 stage2 开始后。

## 用户视角

用户提出了几个重要判断：

1. 从回放视频看，stage1 和 stage2 的动作差异不一定非常大。stage1 也是接触敏感阶段，不应简单认为它不重要。
2. stage1-only 成功率下降可能不仅仅是因为没有学会，也可能是因为 RL 让抓取动作更快或更激进，提前把系统带入 base VLA 不熟悉的后续状态分布。
3. 如果 stage1 RL 真的更快完成抓取，那么 residual 关闭后，base VLA 接管时可能处在 OOD 状态，从而导致后续搬运或放置失败。
4. 但也存在另一个可能：stage1 根本没有学会抓取，因为 sparse final reward 和 active-only replay 让奖励信号传不过来。如果是这样，就谈不上“更快抓取导致 OOD”。
5. 如果能加入密集或中间奖励，尤其是用视觉判别器判断阶段是否完成，stage1 可能会变得可学习。

## 当前判断

我的判断是：stage1 失败至少有两个可能机制。

### 假设 A：stage1 没拿到有效奖励信号

stage1-only replay 只保留 `[30,75)` 的 transition。由于任务 reward 基本是 sparse final success：

```text
成功前 reward = 0
成功时 reward = 1
```

而成功通常发生在 policy step 116-140 附近。active-only replay 又在 stage 边界断开 bootstrap，所以 learner 看到的是：

```text
stage1 内 reward 基本全是 0
stage1 末尾不向后 bootstrap
critic 学到 Q ~= 0
actor 缺少有效价值梯度
```

在这种情况下，stage1 residual 可能只是被无效梯度、entropy 变化或数值噪声推向更大、更饱和的动作，而不是学会了更好的抓取。

因此，`Q ~= 0` 不能说明“已经抓到了”。它更准确地说明：在当前 replay/reward 定义下，stage1 action 没有被连接到最终成功 reward。

### 假设 B：stage1 学到局部更快抓取，但造成后续 OOD

用户提出的 OOD 假设也合理。即使 stage1 学到了一些局部有用的行为，它也可能改变后续状态分布：

```text
抓取得更早；
抓取得更激进；
物体位置与专家/base 成功轨迹不同；
stage1 结束时 base VLA 接管，但状态已经偏离它熟悉的分布。
```

这种情况下，stage1 局部 milestone 可能更好，但 final success 下降。这里的问题不是“stage1 没用”，而是 stage1 的局部优化目标没有约束下游 VLA 可接管性。

不过，以当前证据看，假设 A 更强，因为已有指标显示：

```text
Q target/predicted 贴近 0；
residual 越来越饱和；
成功率下降；
还没有证据证明 stage1 milestone 变好。
```

所以在证明 stage1 milestone 确实改善之前，不应直接把失败归因于“更快抓取导致 OOD”。

## 关键区分

这次讨论里最重要的区分是：

```text
物理关键性 != RL 可学习性
```

stage1 在物理任务上很关键，因为抓取失败会导致后续必然失败。但在当前训练定义下，它远离 final reward，且 active-only replay 断开了后续信用分配，所以它的 RL 可学习性很低。

stage2 也很关键，并且更接近 final success reward。因此 stage2 在 sparse reward 下更容易学。

这解释了为什么视频上看两个阶段都像关键接触阶段，但训练表现却差异很大。

## 关于视觉判别器奖励

用户提出可以训练一个环境判别器，例如 ResNet，根据当前图像判断阶段是否完成：

```text
stage1 completed: bowl 已被成功夹起 / 已离开抽屉
stage2 completed: bowl 已放到 plate 上 / 接近最终成功状态
```

如果判别器可靠，可以给 one-shot milestone reward：

```text
第一次达到 stage1 milestone: +r_stage1
第一次达到 stage2 milestone: +r_stage2
final task success: +r_success
```

这种方法本质上是 visual milestone reward，而不是普通 dense reward。它正好解决 stage1 Q 信号缺失的问题：stage1 replay 内会出现局部正奖励，critic 可以区分哪些动作真的让任务进入了更好的中间状态。

需要注意：

1. reward 应该是 one-shot，而不是每一步重复给。
2. 判别器最好有 hysteresis，例如 high threshold 触发、触发后锁定，避免 score 抖动重复给奖励。
3. stage1 reward 不宜过大，否则可能诱导 policy 只追求“拿起/碰出碗”，而不关心后续是否可完成。
4. 如果能用 sim privileged state 自动标注训练数据，再训练视觉判别器，会比纯人工或时间段弱标注可靠。
5. 单帧图像可能不够，必要时可考虑多视角、wrist 图像、最近几帧或 proprio/gripper state。

一个保守的 reward scale 可以是：

```text
stage1 milestone: +0.5
stage2 milestone: +0.5 或 +1.0
final success: +1.0
```

## 下一步验证

为了区分上面两个机制，建议先做诊断，而不是马上扩大训练矩阵。

### 1. 统计 stage1 milestone

对 base、stage1-only、two-stage checkpoint 分别统计：

```text
step 75 前是否夹住碗；
step 75 前碗是否离开抽屉；
first grasp / first lift / drawer-exit step 是否提前；
stage1 结束时物体位置是否更偏离成功 base 轨迹。
```

判读：

```text
stage1 milestone 没提高，final success 下降：
  更支持“没学会抓取 / 奖励信号传不过来”。

stage1 milestone 提高，但 final success 下降：
  更支持“局部更快抓取导致下游 OOD”。

stage1 milestone 更早但状态偏离专家成功轨迹：
  说明需要约束 stage1 结束状态，而不只是奖励抓取完成。
```

### 2. 补 stage2-only 对照

当前对比是：

```text
stage1-only: [30,75)
two-stage: [30,75) + [110,160)
```

还需要：

```text
stage2-only: [110,160)
```

如果 stage2-only 接近或超过 two-stage，则说明当前主要收益来自 reward-proximal 的 stage2，stage1 可以先不启用。

如果 two-stage 明显好于 stage2-only，则说明 stage1 虽然单独学不动，但可能通过改变进入 stage2 的状态分布产生协同。

### 3. 再考虑 visual milestone reward

在 sparse reward 结果清楚后，再做：

```text
stage1-only + visual milestone reward
stage1 + stage2 + visual milestone reward
```

这能验证一个更强的结论：

```text
stage1 不是不关键，而是在 sparse final reward + active-only replay 下不可学习；
加入阶段性视觉 milestone reward 后，stage1 residual 才可能稳定优化。
```

## 当前结论

stage1-only 下降不应简单解释为“stage1 不重要”。更合理的结论是：

```text
stage1 物理上关键，但当前 sparse reward + active-only replay + boundary cutoff 的定义，
让它缺少有效信用分配。
```

同时，用户提出的 OOD 风险是后续必须验证的第二层问题：

```text
即使 stage1 学到更快抓取，也可能把下游 base VLA 带到不熟悉的状态分布。
```

因此，后续分析应先回答：

```text
stage1 residual 到底有没有提高局部 milestone？
```

只有这个问题成立后，才进一步讨论：

```text
局部 milestone 提高是否导致下游 OOD？
```
