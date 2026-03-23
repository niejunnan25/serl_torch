# LIBERO Task 6 Matrix 实验结论（仅 2026-03-21）

Date: 2026-03-23

本文档只总结 `2026-03-21` 这一天的 8 组 `matrix` 实验结论，并且和 `2026-03-19` 那份结论文档保持同样原则：

- 每给出一个结论，都提供详细依据；
- 依据优先来自实际 run 目录中的配置、日志和评估结果；
- 明确区分“可以直接下结论的部分”和“因为评估链路中断而暂时不能下结论的部分”。

## 1. 分析范围

本次只覆盖以下 8 组 run 在 `2026-03-21` 的训练结果：

- `train_residual_sac_task6_matrix_xi005_grip1`
- `train_residual_sac_task6_matrix_xi005_grip2`
- `train_residual_sac_task6_matrix_xi0075_grip1`
- `train_residual_sac_task6_matrix_xi0075_grip2`
- `train_residual_sac_task6_matrix_xi010_grip1`
- `train_residual_sac_task6_matrix_xi010_grip2`
- `train_residual_sac_task6_matrix_xi0125_grip1`
- `train_residual_sac_task6_matrix_xi0125_grip2`

实验变量仍然只有两维：

- `xi ∈ {0.05, 0.075, 0.10, 0.125}`
- `grip1 / grip2 = residual.gripper_delta_limit 1.0 / 2.0`

## 2. 证据来源

本结论使用的证据主要来自下面几类文件：

1. 配置文件  
   `.../.hydra/config.yaml`

2. 异步评估结果  
   `.../async_eval_results.jsonl`

3. 训练在线 episode 统计  
   `.../episode_logs.jsonl`

4. 当前 step 进度  
   `.../step_logs.jsonl`

5. 异步评估失败日志  
   `.../async_eval/step_*/eval_runner.log`

代表性路径示例：

- 配置：
  `/vla/users/niejunnan/codebase/serl_torch/examples/libero/outputs/libero/train_residual_sac_task6_matrix_xi005_grip1/2026-03-21/03-17-15/.hydra/config.yaml`
- 异步评估：
  `/vla/users/niejunnan/codebase/serl_torch/examples/libero/outputs/libero/train_residual_sac_task6_matrix_xi005_grip1/2026-03-21/03-17-15/async_eval_results.jsonl`
- 评估失败日志：
  `/vla/users/niejunnan/codebase/serl_torch/examples/libero/outputs/libero/train_residual_sac_task6_matrix_xi005_grip1/2026-03-21/03-17-15/async_eval/step_0160000/eval_runner.log`

## 3. 先给一张当前状态总表

| 实验 | 当前训练步数 | 当前在线 SR | 当前 recent SR | 最佳有效 eval | 最后一次有效 eval | 首次 eval 失败 step | eval 失败次数 | 最大 checkpoint | `summary.json` |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `xi=0.05, grip=1` | `179688` | `0.589` | `0.900` | `0.86 @ 155k` | `0.86 @ 155k` | `160k` | `4` | `175k` | 不存在 |
| `xi=0.05, grip=2` | `187063` | `0.600` | `1.000` | `0.98 @ 155k` | `0.94 @ 165k` | `170k` | `4` | `185k` | 不存在 |
| `xi=0.075, grip=1` | `191641` | `0.496` | `0.750` | `0.84 @ 5k` | `0.78 @ 170k` | `175k` | `4` | `190k` | 不存在 |
| `xi=0.075, grip=2` | `174242` | `0.466` | `0.800` | `0.96 @ 155k` | `0.96 @ 155k` | `160k` | `3` | `170k` | 不存在 |
| `xi=0.10, grip=1` | `264554` | `0.671` | `0.850` | `0.92 @ 155k` | `0.72 @ 175k` | `180k` | `17` | `260k` | 不存在 |
| `xi=0.10, grip=2` | `186105` | `0.295` | `0.750` | `0.74 @ 155k` | `0.50 @ 170k` | `175k` | `3` | `185k` | 不存在 |
| `xi=0.125, grip=1` | `160503` | `0.125` | `0.250` | `0.58 @ 140k` | `0.58 @ 140k` | `145k` | `4` | `160k` | 不存在 |
| `xi=0.125, grip=2` | `269308` | `0.359` | `0.700` | `0.66 @ 135k` | `0.24 @ 180k` | `185k` | `17` | `265k` | 不存在 |

## 4. 配置层面的先决条件

在解读 `2026-03-21` 的结果之前，必须先明确三件事：

1. 这批 run 目标训练预算是 `300000` env steps，而不是 `100000`。
2. 这批 run 的异步评估固定 seed 是 `7`，而 `2026-03-19` 是 `3407`。
3. 这批 run 开启了 `replay_prefetch`。

代表性配置证据：

- `training.max_online_env_steps=300000`
- `training.async_eval.fixed_seed=7`
- `training.replay_prefetch.enabled=true`
- `offline.ratio=0.5`
- `offline.bootstrap_base.enabled=false`
- `training.warmup_base_episodes=0`
- `training.training_starts=200`

见：

`/vla/users/niejunnan/codebase/serl_torch/examples/libero/outputs/libero/train_residual_sac_task6_matrix_xi005_grip1/2026-03-21/03-17-15/.hydra/config.yaml`

这意味着：

- `2026-03-21` 的异步评估分数不能和 `2026-03-19` 直接横向比较绝对大小；
- 但同一天内部的 8 组相互比较仍然是有效的。

## 5. 结论与依据

## 结论 1：`2026-03-21` 这批 run 不是“训练完整结束”的状态，而是“中途停止 + 没有最终 summary”的状态

### 依据

1. 8 组 run 的目标预算都是 `300000` env steps，但当前实际都停在 `160k` 到 `269k` 之间，没有一组跑满：
   - 最低是 `xi=0.125, grip=1`，当前约 `160503`
   - 最高是 `xi=0.125, grip=2`，当前约 `269308`

2. 8 组 run 的目录里都没有 `summary.json`：
   - 总表中 `summary.json` 一列全部为“不存在”

3. 用更严格的进程检查方式看，当前没有真正的训练脚本在跑：
   - `ps -ef | rg 'examples/libero/scripts/train_residual_sac.py' | rg -v 'rg|pgrep|bash -c'`
   - 返回为空

### 解释

所以对这批 run 的合理描述不是“训练完成后的最终结果”，而是：

- 当前目录里保留下来的中途状态；
- 需要从 `episode_logs.jsonl`、`step_logs.jsonl` 和 `async_eval_results.jsonl` 去恢复它们的阶段性表现。

## 结论 2：在 `2026-03-21` 这一天内部比较时，`xi=0.05, grip=2` 是最强、也最稳的一组

### 依据

1. 它的最佳有效异步评估是全场最高：
   - `0.98 @ 155k`

2. 它的最后一次有效异步评估仍然很高：
   - `0.94 @ 165k`

3. 它的在线训练指标也处于第一梯队：
   - 当前在线 running success rate = `0.600`
   - 当前 recent success rate = `1.000`

4. 它不像某些组那样只有一个瞬时峰值，而是从 `135k` 后持续保持高位：
   - `135k:0.94`
   - `140k:0.90`
   - `145k:0.92`
   - `150k:0.90`
   - `155k:0.98`
   - `160k:0.96`
   - `165k:0.94`

### 解释

这组最像“真正已经学成”的配置：

- 高分不是偶然单点；
- online 和 eval 两侧都强；
- 后段在 eval 链路失效之前没有出现明显退化。

## 结论 3：`xi=0.10, grip=1` 是在线表现最强的一组，但由于后段失去可靠评估，它目前是“最有潜力、也最需要补评估确认”的组

### 依据

1. 它当前在线训练指标是 8 组里最高的：
   - 当前 step = `264554`
   - 当前在线 running success rate = `0.671`
   - 当前 recent success rate = `0.850`

2. 它在有效 eval 期内也确实很强：
   - `best eval = 0.92 @ 155k`
   - `160k=0.90`
   - `165k=0.86`

3. 但它的最后一次有效 eval 已经出现明显回落：
   - `170k=0.70`
   - `175k=0.72`
   - 随后从 `180k` 开始连续评估失败

4. 它是 8 组中后段“盲飞”最长的一组之一：
   - 首次失败 step = `180k`
   - 失败次数 = `17`
   - 但训练还继续写到了 `264k+`

### 解释

这组的关键问题不是“看起来弱”，而是：

- 它在线很强；
- 有效 eval 到 `155k~165k` 时也很强；
- 但 `180k+` 没有可靠评估了。

因此对它最准确的判断不是“已经超过 `xi=0.05, grip=2`”，而是：

- 很可能是最值得继续追的高潜力组；
- 但必须补统一评估才能确认后半段是否继续提升。

## 结论 4：`xi=0.075, grip=2` 是另一组稳定强组，且比同 `xi` 的 `grip=1` 更清晰

### 依据

1. 它的最佳有效 eval 很高：
   - `0.96 @ 155k`

2. 它的最后一次有效 eval 仍然保持在峰值：
   - `last ok = 0.96 @ 155k`

3. 它后半段爬升过程比较清楚：
   - `130k:0.38`
   - `135k:0.72`
   - `140k:0.64`
   - `145k:0.72`
   - `150k:0.82`
   - `155k:0.96`

4. 它和同 `xi` 的 `grip=1` 相比，更像稳定增长而不是“早期就跳出高点”：
   - `xi=0.075, grip=1` 的最佳值是 `0.84`，但第一次就出现在 `5k`
   - `xi=0.075, grip=2` 的高分集中出现在中后段，更符合正常收敛直觉

### 解释

如果把 `2026-03-21` 里“非最低 `xi` 但已经表现很好”的组挑出来，`xi=0.075, grip=2` 是最稳妥的一组。

## 结论 5：`xi=0.05, grip=1` 也明显学起来了，但仍然整体弱于 `xi=0.05, grip=2`

### 依据

1. 它当前在线训练表现已经不弱：
   - 当前在线 running success rate = `0.589`
   - 当前 recent success rate = `0.900`

2. 它的有效 eval 也在后期明显提升：
   - `150k=0.76`
   - `155k=0.86`

3. 但和同 `xi` 的 `grip=2` 相比，仍然全面落后：
   - `best eval: 0.86 vs 0.98`
   - `last ok eval: 0.86 vs 0.94`
   - `online running SR: 0.589 vs 0.600`
   - `recent SR: 0.900 vs 1.000`

### 解释

这组不是失败组，而是“已经不错，但不是这一档里最优”的组。

## 结论 6：`xi=0.10, grip=2` 和 `xi=0.125, grip=2` 都说明较大动作幅度下训练更容易变得不稳定

### 依据

1. `xi=0.10, grip=2`
   - 当前在线 running SR 只有 `0.295`
   - 最佳 eval `0.74 @ 155k`
   - 但最后一次有效 eval 已掉到 `0.50 @ 170k`

2. `xi=0.125, grip=2`
   - 当前在线 running SR = `0.359`
   - 最佳 eval `0.66 @ 135k`
   - 但最后一次有效 eval 只剩 `0.24 @ 180k`

3. `xi=0.125, grip=2` 的后半段有效 eval 波动尤其大：
   - `150k:0.66`
   - `155k:0.02`
   - `160k:0.04`
   - `165k:0.10`
   - `170k:0.06`
   - `175k:0.48`
   - `180k:0.24`

### 解释

这两组都不是完全没学，但都体现出一个共同问题：

- 较大的 residual 动作幅度下，策略容易出现“能打出高点，但很难稳定保持”的情况。

## 结论 7：`xi=0.125, grip=1` 仍然是最弱组之一，不应作为后续重点方向

### 依据

1. 它当前在线表现是 8 组里最低：
   - running success rate = `0.125`
   - recent success rate = `0.250`

2. 它第一次有效 eval 失败来得也最早之一：
   - 首次失败 step = `145k`

3. 它唯一能拿得出手的有效高点也只有一个：
   - `0.58 @ 140k`
   - 此前从 `105k` 到 `135k` 大多接近 `0`

### 解释

这组更像“非常偶发地打出一个高点”，而不是稳定学成。

## 结论 8：`2026-03-21` 的后半段分析必须和“异步评估链路失效”这个事实绑定在一起看，否则会误判策略本身

### 依据

1. 8 组 run 都在中后段开始出现连续 eval 失败：
   - 最早从 `145k` 开始
   - 最晚从 `185k` 开始

2. 失败不是单一原因，而是多种配置/接口不兼容：

| 实验 | 首次失败 step | 失败模式 | 代表日志 |
| --- | --- | --- | --- |
| `xi=0.05, grip=1` | `160k` | `action_dim` | `/vla/users/niejunnan/codebase/serl_torch/examples/libero/outputs/libero/train_residual_sac_task6_matrix_xi005_grip1/2026-03-21/03-17-15/async_eval/step_0160000/eval_runner.log` |
| `xi=0.05, grip=2` | `170k` | `arm_delta_limit` | `/vla/users/niejunnan/codebase/serl_torch/examples/libero/outputs/libero/train_residual_sac_task6_matrix_xi005_grip2/2026-03-21/03-16-16/async_eval/step_0170000/eval_runner.log` |
| `xi=0.075, grip=1` | `175k` | `arm_delta_limit` | `/vla/users/niejunnan/codebase/serl_torch/examples/libero/outputs/libero/train_residual_sac_task6_matrix_xi0075_grip1/2026-03-21/03-17-09/async_eval/step_0175000/eval_runner.log` |
| `xi=0.075, grip=2` | `160k` | `arm_delta_limit` | `/vla/users/niejunnan/codebase/serl_torch/examples/libero/outputs/libero/train_residual_sac_task6_matrix_xi0075_grip2/2026-03-21/03-16-20/async_eval/step_0160000/eval_runner.log` |
| `xi=0.10, grip=1` | `180k` | `eval.residual_scale` | `/vla/users/niejunnan/codebase/serl_torch/examples/libero/outputs/libero/train_residual_sac_task6_matrix_xi010_grip1/2026-03-21/03-16-25/async_eval/step_0180000/eval_runner.log` |
| `xi=0.10, grip=2` | `175k` | `arm_delta_limit` | `/vla/users/niejunnan/codebase/serl_torch/examples/libero/outputs/libero/train_residual_sac_task6_matrix_xi010_grip2/2026-03-21/03-16-40/async_eval/step_0175000/eval_runner.log` |
| `xi=0.125, grip=1` | `145k` | `action_dim` | `/vla/users/niejunnan/codebase/serl_torch/examples/libero/outputs/libero/train_residual_sac_task6_matrix_xi0125_grip1/2026-03-21/03-17-04/async_eval/step_0145000/eval_runner.log` |
| `xi=0.125, grip=2` | `185k` | `arm_delta_limit` | `/vla/users/niejunnan/codebase/serl_torch/examples/libero/outputs/libero/train_residual_sac_task6_matrix_xi0125_grip2/2026-03-21/03-16-57/async_eval/step_0185000/eval_runner.log` |

3. 训练在线日志在 eval 失败之后仍然继续写入：
   - 例如 `xi=0.10, grip=1` 在 `180k` 后还继续写到 `264k+`

### 解释

这说明：

- 后半段“没有 eval”并不等于“策略已经坏了”；
- 也不等于“策略一定还在变好”；
- 只能说明 eval 链路失效了，后半段进入了无可靠验证的盲区。

## 结论 9：如果现在要给 `2026-03-21` 这批 run 做阶段性优先级排序，推荐顺序是 `xi=0.05, grip=2` > `xi=0.10, grip=1` > `xi=0.075, grip=2` > `xi=0.05, grip=1`

### 依据

1. `xi=0.05, grip=2`
   - 当前最稳、best/last eval 都最高

2. `xi=0.10, grip=1`
   - 当前在线最强
   - 有效 eval 期内也强
   - 但需要补评估确认

3. `xi=0.075, grip=2`
   - 有效 eval 很高，且爬升过程清楚

4. `xi=0.05, grip=1`
   - 已经明显学起来，但始终弱于同 `xi` 的 `grip=2`

### 解释

这个排序回答的是“后续最值得继续补评估/保留的配置优先级”，不是“所有问题都已经有最终结论”。

## 6. 建议如何在后续文档里表述这批结果

建议把 `2026-03-21` 这批结果写成下面这种口径：

1. `2026-03-21` 的 matrix 实验在更长训练预算 `300k` 下继续推进，大多数 run 的在线表现较 `2026-03-19` 有明显提升。
2. 由于异步评估 seed 从 `3407` 改为 `7`，这一天的 eval 分数不能直接与 `2026-03-19` 横向比较绝对值。
3. `xi=0.05, grip=2` 是当前最稳且最强的组；`xi=0.10, grip=1` 是高潜力组，但需要补统一评估确认 `180k+` 后的真实表现。
4. `xi=0.075, grip=2` 是一组值得保留的稳定强组。
5. `xi=0.125, grip=1` 仍然偏弱；`xi=0.125, grip=2` 虽然有所提升，但波动很大。
6. `145k~185k` 后的异步评估链路失效是这批 run 的核心分析限制，必须在最终结论里明确标注。

## 7. 推荐后续优先补评估的 checkpoint

如果后续要补一轮统一、可比的评估，优先建议从以下 checkpoint 开始：

- `xi=0.05, grip=2 @ 155000`
- `xi=0.05, grip=2 @ 165000`
- `xi=0.10, grip=1 @ 155000`
- `xi=0.10, grip=1 @ 175000`
- `xi=0.10, grip=1 @ 260000`
- `xi=0.075, grip=2 @ 155000`
- `xi=0.05, grip=1 @ 155000`
- `xi=0.125, grip=2 @ 135000`

这些点分别覆盖了：

- 当前最可靠的强组；
- 高潜力但后段盲飞的组；
- 中高 `xi` 下最值得核验的代表 checkpoint。
