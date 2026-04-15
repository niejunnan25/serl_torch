# AgiBot Chunk-Boundary Training: Minimal Startup

这份说明只覆盖最小可跑通路径：

- `chunk-boundary` actor
- `chunk-boundary` learner
- `OpenPI` policy server
- 真机 `robot-service`

对应脚本：

- [run_residual_training_chunk_boundary.py](/Users/niejunnan.25/Documents/codebase/serl_torch/examples/agibot_real/scripts/run_residual_training_chunk_boundary.py)
- [train_residual_chunk_boundary.yaml](/Users/niejunnan.25/Documents/codebase/serl_torch/examples/agibot_real/configs/train_residual_chunk_boundary.yaml)

## 1. 前置条件

默认假设：

- 你在机器上已经能正常跑 `examples/agibot_real`
- `OpenPI` policy server 已准备好
- robot runtime 依赖已经准备好
- 当前使用默认端口：
  - policy: `30001`
  - trainer: `5488`
  - broadcast: `5489`

当前默认 chunk-boundary 配置关键值：

- `residual.chunk_horizon=15`
- `training.training_starts=64`
- `training.steps_per_update=1`

## 2. 启动 policy server

如果你当前就是用 `OpenPI`，最小要求是把策略服务先起在 `30001`。

如果你已经有自己的启动方式，保持下面两项一致即可：

- `policy.type=openpi`
- `policy.port=30001`

## 3. 启动 robot-service

在 `examples/agibot_real` 目录下执行：

```bash
cd /Users/niejunnan.25/Documents/codebase/serl_torch/examples/agibot_real
bash tools/start_robot_service.sh
```

如果你平时已经有自己的 robot runtime 启动方式，也可以继续沿用，只要 actor 所在 shell 后面能正常：

- `source robot/service/env.sh`
- 访问真机 SDK / DDS

## 4. 启动 learner

新脚本已经默认指向 `train_residual_chunk_boundary.yaml`，所以最小命令是：

```bash
cd /Users/niejunnan.25/Documents/codebase/serl_torch/examples/agibot_real
python scripts/run_residual_training_chunk_boundary.py runtime.role=learner
```

如果你想覆盖几个最常用参数：

```bash
python scripts/run_residual_training_chunk_boundary.py \
  runtime.role=learner \
  wandb.project=agibot_real \
  training.max_update_steps=50000
```

## 5. 启动 actor

actor 仍然需要先加载 robot runtime：

```bash
cd /Users/niejunnan.25/Documents/codebase/serl_torch/examples/agibot_real
source robot/service/env.sh
python scripts/run_residual_training_chunk_boundary.py runtime.role=actor
```

如果你想显式覆盖 policy 地址或 task prompt：

```bash
source robot/service/env.sh
python scripts/run_residual_training_chunk_boundary.py \
  runtime.role=actor \
  policy.type=openpi \
  policy.host=127.0.0.1 \
  policy.port=30001 \
  task.name=agibot_real_default \
  task.prompt='Pick up the object with the right hand and place it at the target location.'
```

## 6. 运行时工作流

当前仍然是 controller-only 工作流：

- `g`: ready / resume
- `p`: pause
- `r`: reset
- `s`: success
- `f`: fail
- `h`: help

chunk-boundary 版本的行为是：

- actor 在 chunk 边界推一次 chunk action
- 用 `env.step_chunk(...)` 连续执行整个 chunk
- 一个 chunk 结束后才写一条 replay transition

## 7. 最小检查点

如果系统正常，通常会看到这些现象：

- learner 先进入 replay warmup，等到 `training.training_starts=64` 个 chunk transition
- actor 每次只在 chunk 边界停顿一下，中间 15 步连续执行
- actor 日志里会看到 `chunk_transitions`
- learner 日志里 `replay_size` 现在表示 chunk transition 数，不是 env step 数

## 8. 最常见的两个问题

### learner 一直不开始训练

先看：

- actor 是否真的连上了 learner
- actor 是否真的在插入 replay
- 你是不是把 `training.training_starts` 设得太大

chunk-boundary 配置下，`training.training_starts` 数的是 chunk transition，不是 step。

### actor 一直卡住不执行

先看：

- 是否已经 `source robot/service/env.sh`
- `robot-service` 是否真的启动成功
- terminal 里是否按了 `g`

如果 controller 没进入 `RUNNING`，actor 会阻塞在 chunk 执行等待上。
