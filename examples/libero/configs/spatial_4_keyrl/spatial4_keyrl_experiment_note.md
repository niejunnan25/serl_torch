## LIBERO Spatial Task 4 KeyRL 实验说明

### 目的

这组实验用于验证第一版 KeyRL / key-action residual RL：

```text
base policy 负责大部分轨迹；
residual policy 只在预设关键阶段启用；
非关键阶段不调用 residual policy，也不写 residual replay。
```

核心问题是：只在关键交互窗口训练 residual RL，是否比全程介入更稳定、更有效。

### KeyRL 设置

当前只做固定窗口验证，不做 learned key detector。

单阶段：

```text
[30, 75)
```

双阶段：

```text
[30, 75) + [110, 160)
```

大致理解：

```text
[30,75): 早期接近、对齐、抓取附近
[110,160): 后期搬运、放置、释放附近
```

运行语义：

```text
不在窗口内：
  执行 base action
  不调用 residual policy
  不写 online replay

在窗口内：
  final_action = base_action + alpha * residual_action * action_limit
  transition 写入 residual replay
```

已有 full prepared offline replay 可以复用；加载时会按 active window 过滤，不需要为每个窗口重新生成离线数据。

### 实验矩阵

```text
alpha:   0.1 / 0.2 / 0.5
std_max: 0.5 / 1.0
stage:   single_stage / two_stage
```

总计：

```text
3 * 2 * 2 = 12 组
```

变量含义：

```text
alpha:
  residual 修正幅度。越大越能修正，也越可能破坏 base policy。

std_max:
  residual policy 探索强度上限。越大探索越强，也可能更不稳定。

stage:
  single_stage 验证只修早期抓取窗口是否足够；
  two_stage 验证加上后期放置/释放窗口是否更好。
```

### 配置列表

| 配置 | 状态 | 说明 |
| --- | --- | --- |
| `spatial4_keyrl_alpha0p1_std0p5_single_stage_30_75.yaml` | 待运行 | alpha=0.1, std=0.5, 单阶段 |
| `spatial4_keyrl_alpha0p1_std0p5_two_stage_30_75_110_160.yaml` | 待运行 | alpha=0.1, std=0.5, 双阶段 |
| `spatial4_keyrl_alpha0p1_std1p0_single_stage_30_75.yaml` | 待运行 | alpha=0.1, std=1.0, 单阶段 |
| `spatial4_keyrl_alpha0p1_std1p0_two_stage_30_75_110_160.yaml` | 待运行 | alpha=0.1, std=1.0, 双阶段 |
| `spatial4_keyrl_alpha0p2_std0p5_single_stage_30_75.yaml` | 待运行 | alpha=0.2, std=0.5, 单阶段 |
| `spatial4_keyrl_alpha0p2_std0p5_two_stage_30_75_110_160.yaml` | 待运行 | alpha=0.2, std=0.5, 双阶段 |
| `spatial4_keyrl_alpha0p2_std1p0_single_stage_30_75.yaml` | 已运行 | alpha=0.2, std=1.0, 单阶段 |
| `spatial4_keyrl_alpha0p2_std1p0_two_stage_30_75_110_160.yaml` | 已运行 | alpha=0.2, std=1.0, 双阶段 |
| `spatial4_keyrl_alpha0p5_std0p5_single_stage_30_75.yaml` | 待运行 | alpha=0.5, std=0.5, 单阶段 |
| `spatial4_keyrl_alpha0p5_std0p5_two_stage_30_75_110_160.yaml` | 待运行 | alpha=0.5, std=0.5, 双阶段 |
| `spatial4_keyrl_alpha0p5_std1p0_single_stage_30_75.yaml` | 已运行 | alpha=0.5, std=1.0, 单阶段 |
| `spatial4_keyrl_alpha0p5_std1p0_two_stage_30_75_110_160.yaml` | 已运行 | alpha=0.5, std=1.0, 双阶段 |

备注：前 4 组已运行实验最初用的是旧文件名，没有 `std1p0` 后缀；它们实际继承的是 `std_max=1.0`，等价于上表中的四个 `std1p0` 配置。

### 输出目录

每组实验输出到：

```text
/vla/users/niejunnan/codebase/serl_torch/examples/libero/outputs/spatial_4_keyrl/<config_stem>
```

W&B 的 `exp_name` 也使用相同的 config stem。

### 主要观察点

优先看：

```text
eval success rate
收敛速度
最终成功率
训练是否稳定
single_stage vs two_stage 差异
std0p5 vs std1p0 差异
alpha0p1 / alpha0p2 / alpha0p5 差异
```

重点回答：

```text
1. two_stage 是否明显优于 single_stage？
2. single_stage 下降是否说明早期抓取窗口不适合强 residual 介入？
3. std_max=0.5 是否比 std_max=1.0 更稳定？
4. alpha=0.1 是否能减少 residual 对 base policy 的破坏？
5. alpha=0.5 是否带来更强修正，还是主要带来不稳定？
```

### 当前假设

目前早期观察是：

```text
single_stage 成功率可能下降；
two_stage 收敛更快、成功率更高。
```

一个可能解释是：

```text
早期抓取阶段对扰动敏感，base policy 本来已经能做得不错；
后期放置/释放阶段更接近任务成功信号，可能更适合 residual RL 修正。
```

后续如果这个趋势持续，说明 KeyRL 的关键不只是“少开 RL”，而是要把 RL 开在真正影响成功的阶段。
