# 2026-04-23 LIBERO Residual Training Entrypoints

## Current Entrypoints

当前 LIBERO residual 训练只保留三个入口：

| Entrypoint | Launcher mode | 用途 |
| --- | --- | --- |
| `examples/libero/scripts/train_residual_step.py` | `step` | 逐 step rollout 的 reference/debug 入口 |
| `examples/libero/scripts/train_residual_chunk.py` | `chunk` | 日常主线；actor 执行 chunk，并在 actor 侧完成 transition assembly |
| `examples/libero/scripts/train_residual_processor.py` | `processor` | split/pipeline 主线；actor 发送 raw chunk，processor 负责 assembly 和提交 |

统一启动方式：

```bash
bash examples/libero/tools/launch_residual_training.sh \
  --mode chunk \
  --config-file examples/libero/configs/spatial_4_0514_runtime/spatial4_chunk_alpha0p1_unfiltered_offline_noent_std1p0_ports53100.yaml \
  --policy-gpu 0
```

`processor` 模式会额外启动 processor 进程，并要求配置中启用 dedicated backfill policy。launcher 会检查 `--mode` 与配置名是否一致，避免 `--mode step` 配到 `processor` 配置这类混合拓扑。

## Shared Training Semantics

三个入口的 residual RL 语义保持一致：

1. frozen base policy 预测 `chunk_horizon` 长度的 `base_action_chunk`
2. 当前观测被整理成 residual observation
3. residual agent 输出 residual action chunk
4. `ResidualActionSpec` 将 base action 与 residual action 合成为最终动作
5. replay 中存储的是按 step 展开的 transition
6. learner 从 step replay 中按 chunk window 采样训练 batch

因此，三者的区别不在 SAC/DRQ 更新公式，而在 rollout dataflow、transition assembly 位置、transport 和进程边界。

## Mode Differences

| Mode | Runtime roles | Env execution | Transition assembly | 推荐场景 |
| --- | --- | --- | --- | --- |
| `step` | actor + learner | actor 逐步 `env.step(...)` | actor 每步即时构造 | reference baseline、debug、语义对照 |
| `chunk` | actor + learner | actor 使用 `env.step_chunk(...)` | actor 本地 post-hoc assembly | 默认日常训练入口 |
| `processor` | actor + processor + learner | actor 使用 `env.step_chunk(...)` | processor pipeline assembly | 需要拆分 actor/processor/learner 边界时 |

## Practical Recommendation

- 默认跑实验：使用 `train_residual_chunk.py` / `--mode chunk`
- 需要最直接的行为对照：使用 `train_residual_step.py` / `--mode step`
- 需要 processor-side pipeline 或 raw rollout recycle：使用 `train_residual_processor.py` / `--mode processor`

配置文件名同样使用 mode 名称，例如 `spatial4_chunk_...` 和 `train_residual_task4_exp1_processor...`；`--mode` 必须与这个名称一致。
