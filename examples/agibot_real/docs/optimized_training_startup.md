# AgiBot Optimized Training Startup

这份文档只回答一件事：如果你要直接用
[../configs/train_residual_optimized.yaml](../configs/train_residual_optimized.yaml)
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
bash tools/run_learner.sh --config-name train_residual_optimized
```

等价直跑：

```bash
cd /vla/users/niejunnan/codebase/serl_torch/examples/agibot_real
python scripts/run_residual_training.py --config-name train_residual_optimized runtime.role=learner
```

### 4. 启动 actor

actor 终端先准备真机环境：

```bash
cd /vla/users/niejunnan/codebase/serl_torch/examples/agibot_real
source robot/service/env.sh
```

然后启动：

```bash
bash tools/run_actor.sh --config-name train_residual_optimized
```

等价直跑：

```bash
python scripts/run_residual_training.py --config-name train_residual_optimized runtime.role=actor
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
bash tools/run_learner.sh \
  --config-name train_residual_optimized \
  policy.type=openpi \
  policy.host=127.0.0.1 \
  policy.port=30001
```

### 4. 启动 actor

```bash
cd /vla/users/niejunnan/codebase/serl_torch/examples/agibot_real
source robot/service/env.sh
bash tools/run_actor.sh \
  --config-name train_residual_optimized \
  policy.type=openpi \
  policy.host=127.0.0.1 \
  policy.port=30001
```

如果还要额外覆盖任务，可以继续在 actor 命令后面追加：

```bash
task.name=agibot_real_default task.prompt='Pick up the object with the right hand and place it at the target location.'
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
bash tools/run_learner.sh --config-name train_residual_optimized
```

终端 4，actor：

```bash
cd /vla/users/niejunnan/codebase/serl_torch/examples/agibot_real
source robot/service/env.sh
bash tools/run_actor.sh --config-name train_residual_optimized
```

## 备注

- `tools/run_actor.sh` 和 `tools/run_learner.sh` 默认仍然调用
  [../scripts/run_residual_training.py](../scripts/run_residual_training.py)，只是通过
  `--config-name train_residual_optimized` 切到 optimized yaml。
- optimized 配置的主要收益在 learner，不在 actor。
- 如果你只是做短时间调试，canonical
  [../configs/train_residual.yaml](../configs/train_residual.yaml)
  往往更稳。
