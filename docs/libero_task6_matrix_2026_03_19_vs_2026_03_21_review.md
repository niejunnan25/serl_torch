# LIBERO Task 6 Matrix 训练对比总结（2026-03-19 vs 2026-03-21）

Date: 2026-03-23

## 1. 这份文档在回答什么

这份文档不再把 `2026-03-19` 和 `2026-03-21` 分开顺序叙述，而是直接回答三个更关键的问题：

1. 两天的 8 组 `matrix` 实验，哪些训练条件是相同的，哪些地方真的改了。
2. 哪些指标可以直接横向比较，哪些不能直接比。
3. 这 8 组实验在两天里的变化趋势是什么，当前最稳、最强、最值得补评估的是哪些。

## 2. 实验范围

实验根目录：

`/vla/users/niejunnan/codebase/serl_torch/examples/libero/outputs/libero`

本次比较的 8 组实验：

- `train_residual_sac_task6_matrix_xi005_grip1`
- `train_residual_sac_task6_matrix_xi005_grip2`
- `train_residual_sac_task6_matrix_xi0075_grip1`
- `train_residual_sac_task6_matrix_xi0075_grip2`
- `train_residual_sac_task6_matrix_xi010_grip1`
- `train_residual_sac_task6_matrix_xi010_grip2`
- `train_residual_sac_task6_matrix_xi0125_grip1`
- `train_residual_sac_task6_matrix_xi0125_grip2`

矩阵含义：

- `xi` 对应 `residual.xi`，取值为 `0.05 / 0.075 / 0.10 / 0.125`
- `grip1 / grip2` 对应 `residual.gripper_delta_limit=1.0 / 2.0`

Run 对应关系：

| 实验 | 2026-03-19 | 2026-03-21 |
| --- | --- | --- |
| `xi=0.05, grip=1` | `16-31-40` | `03-17-15` |
| `xi=0.05, grip=2` | `16-32-06` | `03-16-16` |
| `xi=0.075, grip=1` | `16-31-45` | `03-17-09` |
| `xi=0.075, grip=2` | `16-32-21` | `03-16-20` |
| `xi=0.10, grip=1` | `16-31-50` | `03-16-25` |
| `xi=0.10, grip=2` | `16-32-29` | `03-16-40` |
| `xi=0.125, grip=1` | `16-31-59` | `03-17-04` |
| `xi=0.125, grip=2` | `16-32-42` | `03-16-57` |

## 3. 两天到底改了什么

### 3.1 没变的部分

对照代表性配置：

- `2026-03-19`：
  `/vla/users/niejunnan/codebase/serl_torch/examples/libero/outputs/libero/train_residual_sac_task6_matrix_xi005_grip1/2026-03-19/16-31-40/.hydra/config.yaml`
- `2026-03-21`：
  `/vla/users/niejunnan/codebase/serl_torch/examples/libero/outputs/libero/train_residual_sac_task6_matrix_xi005_grip1/2026-03-21/03-17-15/.hydra/config.yaml`

两天共享的核心训练设定是一致的：

- 同一个任务：`libero_10 task_id=6`
- 同一个训练 seed：`task.fixed_env_seed=7`
- 同一个 residual 维度：`action_dim=7`
- 同一个 replay 设定：`batch_size=128`，`capacity=250000`
- 同一个 offline 混合方式：`offline.enabled=true`，`offline.ratio=0.5`，`offline.symmetric_replay=true`
- 同一个 offline 数据源：`data/offline/libero_10_task_6`
- 同一个 critic warm start：`training.calql_pretrain.enabled=true`，`steps=2000`
- 同一个启动门槛：`training_starts=200`

关于 bootstrap、expert 数据、在线预热，这两天也没有差异：

- `offline.bootstrap_base.enabled=false`，所以没有 bootstrap base 成功轨迹
- offline 数据一直都在用，可以视作这批实验共享的 expert/offline 数据底座
- `warmup_base_episodes=0`、`warmup_base_steps=0`、`random_steps=0`，所以没有单独的在线 warmup

### 3.2 变了的部分

真正跨日期改变的关键项只有下面这些：

| 配置项 | 2026-03-19 | 2026-03-21 | 影响 |
| --- | --- | --- | --- |
| `training.max_online_env_steps` | `100000` | `300000` | 3 月 21 日允许训练更久 |
| `training.async_eval.fixed_seed` | `3407` | `7` | 两天的异步评估分数不再是同一评估分布 |
| `training.replay_prefetch.enabled` | 不存在 | `true` | 3 月 21 日新增 replay prefetch |
| `training.async.*` | 不存在 | 新增但 `enabled=false` | 配置块新增，不影响本批 run |
| `training.profiling.*` | 不存在 | 新增但 `enabled=false` | 配置块新增，不影响本批 run |

## 4. 这两天哪些能比，哪些不能直接比

这是读结果时最重要的前提。

可以直接比较的：

- 两天的配置搜索方向可以直接比较，因为变量仍然只有 `xi` 和 `gripper_delta_limit`
- 两天的在线训练表现可以比较趋势，尤其是 `episode_logs.jsonl` 里的 `running_success_rate` 和 `recent_success_rate`
- 同一天内部的异步评估曲线可以比较，因为同一天内的 eval seed 一致

不能直接比较的：

- `2026-03-19` 和 `2026-03-21` 的异步评估绝对分数不能直接横向比高低，因为 `async_eval.fixed_seed` 从 `3407` 变成了 `7`
- 因此不能简单说“3 月 21 的 eval 更高，所以策略一定更强”，这里面混入了评估分布变化

这意味着：

- `2026-03-19` 更适合作为完整基线，因为 8 组都正常结束到 `100k`
- `2026-03-21` 更适合作为“长训练预算下，哪些配置继续变强、哪些有潜力”的趋势观察

## 5. 横向对照表

表中：

- `3/19 末次 eval` 指 `95k` 的最后一次有效异步评估
- `3/21 末次有效 eval` 指该 run 在异步评估开始系统性失败前，最后一次成功写入的评估
- `在线 SR` 来自 `episode_logs.jsonl` 的最后一条记录

| 实验 | 3/19 最佳 eval | 3/19 末次 eval | 3/19 在线 SR | 3/21 最佳 eval | 3/21 末次有效 eval | 3/21 在线 SR | 3/21 当前步数 | 在线 SR 增量 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `xi=0.05, grip=1` | `0.56 @ 75k` | `0.28 @ 95k` | `0.350` | `0.86 @ 155k` | `0.86 @ 155k` | `0.589` | `179563` | `+0.240` |
| `xi=0.05, grip=2` | `0.74 @ 90k` | `0.72 @ 95k` | `0.457` | `0.98 @ 155k` | `0.94 @ 165k` | `0.600` | `186909` | `+0.143` |
| `xi=0.075, grip=1` | `0.62 @ 80k` | `0.30 @ 95k` | `0.295` | `0.84 @ 5k` | `0.78 @ 170k` | `0.496` | `191628` | `+0.200` |
| `xi=0.075, grip=2` | `0.40 @ 65k` | `0.22 @ 95k` | `0.275` | `0.96 @ 155k` | `0.96 @ 155k` | `0.466` | `173958` | `+0.191` |
| `xi=0.10, grip=1` | `0.40 @ 95k` | `0.40 @ 95k` | `0.244` | `0.92 @ 155k` | `0.72 @ 175k` | `0.672` | `265120` | `+0.428` |
| `xi=0.10, grip=2` | `0.50 @ 85k` | `0.12 @ 95k` | `0.174` | `0.74 @ 155k` | `0.50 @ 170k` | `0.295` | `186047` | `+0.121` |
| `xi=0.125, grip=1` | `0.48 @ 10k` | `0.12 @ 95k` | `0.137` | `0.58 @ 140k` | `0.58 @ 140k` | `0.125` | `160196` | `-0.013` |
| `xi=0.125, grip=2` | `0.06 @ 10k` | `0.00 @ 95k` | `0.109` | `0.66 @ 135k` | `0.24 @ 180k` | `0.360` | `269762` | `+0.251` |

## 6. 对比结论

下面的每条结论都配了明确依据，便于直接写进汇报。

### 结论 1：3 月 21 日整体训练得更深，在线表现普遍强于 3 月 19 日

依据：

- 8 组里有 7 组的最终在线 `running_success_rate` 高于 3 月 19 日
- 提升最大的 3 组分别是：
  - `xi=0.10, grip=1`：`0.244 -> 0.672`，增量 `+0.428`
  - `xi=0.125, grip=2`：`0.109 -> 0.360`，增量 `+0.251`
  - `xi=0.05, grip=1`：`0.350 -> 0.589`，增量 `+0.240`
- 只有 `xi=0.125, grip=1` 没有提升：`0.137 -> 0.125`
- 这和配置变化一致：3 月 21 日把 `max_online_env_steps` 从 `100k` 拉长到了 `300k`，因此更像是在看“继续训练之后是否还能涨”

解释：

- 这条结论主要建立在在线指标上，而不是跨日期的 eval 绝对分数上
- 因为 eval seed 在两天不一样，在线指标反而是更稳妥的跨日期趋势信号

### 结论 2：跨两天都最稳的一组仍然是 `xi=0.05, grip=2`

依据：

- 3 月 19 日，它是 8 组里最强的一组：
  - 最佳 eval `0.74 @ 90k`
  - 末次 eval `0.72 @ 95k`
  - 在线 SR `0.457`
- 3 月 21 日，它仍然是最稳的一组之一：
  - 最佳 eval `0.98 @ 155k`
  - 末次有效 eval `0.94 @ 165k`
  - 在线 SR `0.600`
  - `recent_success_rate=1.00`
- 它不像很多 run 一样只是在中途短暂冲高，而是在两个日期里都保持了高位结果

解释：

- 如果只看“完整基线”，3 月 19 日的首选就是它
- 如果看“长训练以后仍然稳定”，3 月 21 日它仍然没有掉队
- 所以这组是当前最稳、最不依赖额外解释的一组

### 结论 3：3 月 21 日最值得重点补评估的是 `xi=0.10, grip=1`

依据：

- 这组在 3 月 19 日其实并不突出：
  - 最佳 eval `0.40 @ 95k`
  - 在线 SR `0.244`
- 但到了 3 月 21 日，它的训练状态变化最大：
  - 在线 SR 提升到 `0.672`，是全场最高
  - 当前步数达到 `265120`，也是跑得最久的一组之一
  - 最佳 eval 到了 `0.92 @ 155k`
  - 即使最后一次有效 eval `175k`，也还有 `0.72`
- 这说明它在长训练预算下，确实有“后劲”

限制：

- 这组从 `180k` 开始异步评估失败，代表日志：
  `/vla/users/niejunnan/codebase/serl_torch/examples/libero/outputs/libero/train_residual_sac_task6_matrix_xi010_grip1/2026-03-21/03-16-25/async_eval/step_0180000/eval_runner.log`
- 错误是 `Could not override 'eval.residual_scale'`
- 因此 `180k+` 之后到底继续变好，还是已经过峰值，目前没有可靠 eval 证据

解释：

- 这组不是当前“最稳”的冠军，但它是当前“最值得追加统一评估确认”的候选

### 结论 4：3 月 19 日里很多 run 都是中期最好，说明 `100k` 不是天然最佳停点

依据：

- `xi=0.05, grip=1`：`0.56 @ 75k -> 0.28 @ 95k`
- `xi=0.075, grip=1`：`0.62 @ 80k -> 0.30 @ 95k`
- `xi=0.10, grip=2`：`0.50 @ 85k -> 0.12 @ 95k`
- `xi=0.125, grip=1`：`0.48 @ 10k -> 0.12 @ 95k`

解释：

- 这里的下降不是小抖动，而是持续性回落
- 所以 `100k` 可以作为统一训练预算，但不适合作为天然的模型选择点
- 这也是为什么 3 月 21 日把训练预算拉长之后，有些配置会表现出更强的后劲

进一步解释：

- 但也不能因此反过来说“训练越久一定越好”
- 更准确的说法是：不同配置的最佳 checkpoint 出现时机不同，`100k` 只是统一预算，不是统一最优点

### 结论 5：高 `xi` 在两天里都更危险，尤其 `xi=0.125`

依据：

- `xi=0.125, grip=1` 在两天里都偏弱：
  - 3 月 19 日在线 SR `0.137`，末次 eval `0.12`
  - 3 月 21 日在线 SR `0.125`，最佳 eval 虽有 `0.58 @ 140k`，但之后很快就没有更多有效 eval
- `xi=0.125, grip=2` 在 3 月 21 日在线上有改善：
  - 在线 SR `0.109 -> 0.360`
  - 最佳 eval `0.66 @ 135k`
- 但它的后期稳定性仍然不好：
  - 末次有效 eval 只剩 `0.24 @ 180k`
- 这说明高 `xi` 的确可能在更长训练里把在线表现拉起来，但策略稳定性和泛化质量仍然更脆

解释：

- 如果现在要做收敛区间判断，`xi>=0.125` 仍然应被视为高风险区间

### 结论 6：3 月 21 日的主要问题不是“没学到”，而是“后半段评估链路失效”

依据：

- 8 组 run 都没有跑满配置中的 `300k`
- 8 组 run 都没有生成 `summary.json`
- 当前也没有活跃训练进程：
  `ps -ef | rg 'examples/libero/scripts/train_residual_sac.py' | rg -v 'rg|bash -c'`
- 异步评估从 `145k` 到 `185k` 之间开始系统性失败，首个失败 step 如下：
  - `xi=0.05, grip=1`：`160k`
  - `xi=0.05, grip=2`：`170k`
  - `xi=0.075, grip=1`：`175k`
  - `xi=0.075, grip=2`：`160k`
  - `xi=0.10, grip=1`：`180k`
  - `xi=0.10, grip=2`：`175k`
  - `xi=0.125, grip=1`：`145k`
  - `xi=0.125, grip=2`：`185k`
- 失败模式主要有三类：
  - `unexpected keyword argument 'action_dim'`
  - `Could not override 'residual.arm_delta_limit'`
  - `Could not override 'eval.residual_scale'`

代表日志：

- `/vla/users/niejunnan/codebase/serl_torch/examples/libero/outputs/libero/train_residual_sac_task6_matrix_xi005_grip1/2026-03-21/03-17-15/async_eval/step_0160000/eval_runner.log`
- `/vla/users/niejunnan/codebase/serl_torch/examples/libero/outputs/libero/train_residual_sac_task6_matrix_xi005_grip2/2026-03-21/03-16-16/async_eval/step_0170000/eval_runner.log`
- `/vla/users/niejunnan/codebase/serl_torch/examples/libero/outputs/libero/train_residual_sac_task6_matrix_xi010_grip1/2026-03-21/03-16-25/async_eval/step_0180000/eval_runner.log`

解释：

- 所以 3 月 21 日后半段不能简单理解为“没有结果”
- 更准确的说法是：训练继续推进了一段，但评估链路坏了，导致关键 checkpoint 缺少可信的统一验证

## 7. 如果要写汇报，建议这样概括

可以把两天结果概括成下面三句话：

1. `2026-03-19` 给出了完整的 `100k` 基线，说明在同一套评估 seed `3407` 下，最稳的配置是 `xi=0.05, grip=2`，而且很多 run 在 `100k` 之前就已达到峰值。
2. `2026-03-21` 把训练预算提高到 `300k` 后，绝大多数配置的在线训练表现继续提升，其中 `xi=0.10, grip=1` 的提升幅度最大，说明部分配置存在明显的长训练后劲。
3. 但 `2026-03-21` 的异步评估 seed 已变成 `7`，且 `145k-185k` 后评估链路失效，因此这一天更适合用来判断“谁值得继续补评估”，而不适合直接替代 `2026-03-19` 成为最终定版结论。

## 8. 当前最合理的阶段性判断

如果现在就要给阶段性结论，我建议这样写：

- 最稳的配置仍然是 `xi=0.05, grip=2`
- 3 月 21 日最值得追加统一评估的是 `xi=0.10, grip=1`
- `xi=0.075, grip=2` 和 `xi=0.05, grip=1` 也值得保留
- `xi=0.125, grip=1` 依然偏弱
- `xi=0.125, grip=2` 虽然在线提升明显，但稳定性还不够，不能仅凭在线指标下最终结论

## 9. 建议补评估的 checkpoint

如果后续要用统一 eval seed 和统一 eval 脚本补评估，优先级建议如下：

- `xi=0.05, grip=2 @ 90k`
- `xi=0.05, grip=2 @ 155k`
- `xi=0.10, grip=1 @ 155k`
- `xi=0.10, grip=1 @ 175k`
- `xi=0.075, grip=2 @ 155k`
- `xi=0.05, grip=1 @ 155k`

这样补完之后，3 月 19 日和 3 月 21 日的结论才真正能收敛到同一条判断链上。
