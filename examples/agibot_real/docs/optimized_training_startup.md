# AgiBot Optimized Training Startup

这份文档只回答一件事：如果你要直接用
[../configs/train_residual_optimized.yaml](../configs/train_residual_optimized.yaml)
配合
[../scripts/run_residual_training_copy.py](../scripts/run_residual_training_copy.py)
启动训练，应该怎么跑。

这份 optimized 配置相对 canonical
[../configs/train_residual.yaml](../configs/train_residual.yaml)
的核心变化是：

- learner 打开了 `training.torch_compile.enabled=true`
- actor 仍然走原来的控制路径，`torch_compile` 不作用在 actor
- wandb 名字和输出目录会单独写到 `*_optimized`

如果你要做正式长跑训练，推荐 actor 和 learner 都统一使用：

```text
--config-name train_residual_optimized
```

并且统一走 copy 入口：

```text
python scripts/run_residual_training_copy.py ...
```

这条 copy 线的语义是：

```text
chunk execute -> post-hoc step transition assembly -> step-window replay
```

也就是说：

- actor 仍然按 chunk 执行动作
- replay 里仍然存 per-step transition
- learner 合同没有改成 direct chunk replay

## 先说明一个限制

当前：

- [../tools/run_actor.sh](../tools/run_actor.sh)
- [../tools/run_learner.sh](../tools/run_learner.sh)

默认还是调用 canonical
[../scripts/run_residual_training.py](../scripts/run_residual_training.py)。

所以如果你要用 copy 入口，**现在不要直接用 wrapper**，而是直接运行：

```bash
python scripts/run_residual_training_copy.py ...
```

## JoyRA 默认命令

这份 optimized yaml 默认还是：

- `policy.type=joyra`
- `policy.port=9001`

所以如果你本来就跑 JoyRA，最小启动顺序是：

1. 启动 JoyRA server
2. 启动 robot-service
3. 启动 learner
4. 启动 actor

### 1. 启动 JoyRA server

```bash
cd /vla/users/niejunnan/codebase/serl_torch/examples/agibot_real
JOYRA_ROOT=/path/to/JoyRA \
JOYRA_CKPT_PATH=/path/to/checkpoints/steps_xxx.pt \
bash tools/serve_joyra.sh --port 9001
```

### 2. 启动 robot-service

```bash
cd /vla/users/niejunnan/codebase/serl_torch/examples/agibot_real
bash tools/start_robot_service.sh
```

### 3. 启动 learner

```bash
cd /vla/users/niejunnan/codebase/serl_torch/examples/agibot_real
python scripts/run_residual_training_copy.py \
  --config-name train_residual_optimized \
  runtime.role=learner
```

如果你还想补 learner 侧常见 override，也是在后面继续追加：

```bash
training.max_update_steps=300000 training.checkpoint.dir=checkpoints
```

### 4. 启动 actor

actor 终端先准备真机环境：

```bash
cd /vla/users/niejunnan/codebase/serl_torch/examples/agibot_real
source robot/service/env.sh
```

然后启动：

```bash
python scripts/run_residual_training_copy.py \
  --config-name train_residual_optimized \
  runtime.role=actor
```

## OpenPI 命令

如果你想用 OpenPI 跑 optimized 配置，需要覆盖掉 yaml 里的 JoyRA 默认值。

### 1. 启动 OpenPI server

```bash
cd /vla/users/niejunnan/codebase/serl_torch/examples/agibot_real
OPENPI_ROOT=/path/to/openpi \
POLICY_DIR=/path/to/policy/checkpoint \
bash tools/serve_openpi.sh --port 30001
```

### 2. 启动 robot-service

```bash
cd /vla/users/niejunnan/codebase/serl_torch/examples/agibot_real
bash tools/start_robot_service.sh
```

### 3. 启动 learner

```bash
cd /vla/users/niejunnan/codebase/serl_torch/examples/agibot_real
python scripts/run_residual_training_copy.py \
  --config-name train_residual_optimized \
  runtime.role=learner \
  policy.type=openpi \
  policy.host=127.0.0.1 \
  policy.port=30001
```

### 4. 启动 actor

```bash
cd /vla/users/niejunnan/codebase/serl_torch/examples/agibot_real
source robot/service/env.sh
python scripts/run_residual_training_copy.py \
  --config-name train_residual_optimized \
  runtime.role=actor \
  policy.type=openpi \
  policy.host=127.0.0.1 \
  policy.port=30001
```

如果还要额外覆盖任务，可以继续在 actor 命令后面追加：

```bash
task.name=agibot_real_default task.prompt='Pick up the object with the right hand and place it at the target location.'
```

## Copy + Optimized + 异步分离写数据

如果你要的是：

- copy 入口：
  [../scripts/run_residual_training_copy.py](../scripts/run_residual_training_copy.py)
- optimized yaml：
  [../configs/train_residual_optimized.yaml](../configs/train_residual_optimized.yaml)
- async backfill：
  `backfill_policy.enabled=true`

那么需要再额外起一台 dedicated backfill policy server。

这里的语义是：

- 主 policy server 负责 actor 当前 chunk 的控制决策
- backfill policy server 负责 chunk 执行后的 post-hoc residual obs backfill
- replay 仍然由主线程按顺序 commit，不是后台乱序直写

注意两点：

- `policy.type` 和 backfill server 的 backend 必须一致
- `policy.port` 和 `backfill_policy.port` 不要相同

### JoyRA: async dedicated backfill

这是和当前 optimized yaml 默认值最一致的一组命令。

终端 1，主 JoyRA policy：

```bash
cd /vla/users/niejunnan/codebase/serl_torch/examples/agibot_real
JOYRA_ROOT=/path/to/JoyRA \
JOYRA_CKPT_PATH=/path/to/checkpoints/steps_xxx.pt \
bash tools/serve_joyra.sh --port 9001
```

终端 2，backfill JoyRA policy：

```bash
cd /vla/users/niejunnan/codebase/serl_torch/examples/agibot_real
JOYRA_ROOT=/path/to/JoyRA \
JOYRA_CKPT_PATH=/path/to/checkpoints/steps_xxx.pt \
bash tools/serve_joyra.sh --port 9011
```

终端 3，robot-service：

```bash
cd /vla/users/niejunnan/codebase/serl_torch/examples/agibot_real
bash tools/start_robot_service.sh
```

终端 4，learner：

```bash
cd /vla/users/niejunnan/codebase/serl_torch/examples/agibot_real
python scripts/run_residual_training_copy.py \
  --config-name train_residual_optimized \
  runtime.role=learner
```

终端 5，actor：

```bash
cd /vla/users/niejunnan/codebase/serl_torch/examples/agibot_real
source robot/service/env.sh
python scripts/run_residual_training_copy.py \
  --config-name train_residual_optimized \
  runtime.role=actor \
  policy.type=joyra \
  policy.host=127.0.0.1 \
  policy.port=9001 \
  backfill_policy.enabled=true \
  backfill_policy.host=127.0.0.1 \
  backfill_policy.port=9011 \
  backfill_policy.max_pending_chunks=2
```

### OpenPI: async dedicated backfill

如果你要用 OpenPI，也是一模一样的结构，只是换成两台 OpenPI server。

终端 1，主 OpenPI policy：

```bash
cd /vla/users/niejunnan/codebase/serl_torch/examples/agibot_real
OPENPI_ROOT=/path/to/openpi \
POLICY_DIR=/path/to/policy/checkpoint \
bash tools/serve_openpi.sh --port 30001
```

终端 2，backfill OpenPI policy：

```bash
cd /vla/users/niejunnan/codebase/serl_torch/examples/agibot_real
OPENPI_ROOT=/path/to/openpi \
POLICY_DIR=/path/to/policy/checkpoint \
bash tools/serve_openpi.sh --port 30011
```

终端 3，robot-service：

```bash
cd /vla/users/niejunnan/codebase/serl_torch/examples/agibot_real
bash tools/start_robot_service.sh
```

终端 4，learner：

```bash
cd /vla/users/niejunnan/codebase/serl_torch/examples/agibot_real
python scripts/run_residual_training_copy.py \
  --config-name train_residual_optimized \
  runtime.role=learner \
  policy.type=openpi \
  policy.host=127.0.0.1 \
  policy.port=30001
```

终端 5，actor：

```bash
cd /vla/users/niejunnan/codebase/serl_torch/examples/agibot_real
source robot/service/env.sh
python scripts/run_residual_training_copy.py \
  --config-name train_residual_optimized \
  runtime.role=actor \
  policy.type=openpi \
  policy.host=127.0.0.1 \
  policy.port=30001 \
  backfill_policy.enabled=true \
  backfill_policy.host=127.0.0.1 \
  backfill_policy.port=30011 \
  backfill_policy.max_pending_chunks=2
```

## 一组最常用的完整命令

如果你只想复制一组最常用命令，默认 JoyRA 就用这一组。

终端 1，policy server：

```bash
cd /vla/users/niejunnan/codebase/serl_torch/examples/agibot_real
JOYRA_ROOT=/path/to/JoyRA \
JOYRA_CKPT_PATH=/path/to/checkpoints/steps_xxx.pt \
bash tools/serve_joyra.sh --port 9001
```

终端 2，robot-service：

```bash
cd /vla/users/niejunnan/codebase/serl_torch/examples/agibot_real
bash tools/start_robot_service.sh
```

终端 3，learner：

```bash
cd /vla/users/niejunnan/codebase/serl_torch/examples/agibot_real
python scripts/run_residual_training_copy.py \
  --config-name train_residual_optimized \
  runtime.role=learner
```

终端 4，actor：

```bash
cd /vla/users/niejunnan/codebase/serl_torch/examples/agibot_real
source robot/service/env.sh
python scripts/run_residual_training_copy.py \
  --config-name train_residual_optimized \
  runtime.role=actor
```

## 备注

- 这份文档现在假设你明确要用
  [../scripts/run_residual_training_copy.py](../scripts/run_residual_training_copy.py)。
- `tools/run_actor.sh` 和 `tools/run_learner.sh` 默认仍然调用
  [../scripts/run_residual_training.py](../scripts/run_residual_training.py)，
  还没有切到 copy 入口。
- optimized 配置的主要收益在 learner，不在 actor。
- 如果你只是做短时间调试，canonical
  [../configs/train_residual.yaml](../configs/train_residual.yaml)
  往往更稳。
