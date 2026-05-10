# AgiBot `train_residual_copy` Startup

当前使用的配置文件：

- [examples/agibot_real/configs/train_residual_copy.yaml](/home/hello/codebase/serl_torch/examples/agibot_real/configs/train_residual_copy.yaml)

当前这份 yaml 已经固定为：

- `policy.port=9001`
- `backfill_policy.enabled=true`
- `backfill_policy.port=9011`
- `residual.chunk_horizon=15`

所以当前推荐拓扑是：

1. 主 JoyRA server
2. backfill JoyRA server
3. robot-service
4. learner
5. actor

## 端口

- 主 JoyRA: `9001`
- backfill JoyRA: `9011`
- learner trainer port: `5488`
- learner broadcast port: `5489`
- split-queue data port: `5490`

## 终端 1: 主 JoyRA server

```bash
cd /home/hello/codebase/JoyRA
conda activate joyra
export CUDA_VISIBLE_DEVICES=0
export PYTHONPATH=/home/hello/codebase/JoyRA:$PYTHONPATH
python deployment/real_infer/server.py \
  --host 0.0.0.0 \
  --port 9001 \
  --ckpt-path /path/to/checkpoints/steps_xxx.pt
```

## 终端 2: backfill JoyRA server

```bash
cd /home/hello/codebase/JoyRA
conda activate joyra
export CUDA_VISIBLE_DEVICES=1
export PYTHONPATH=/home/hello/codebase/JoyRA:$PYTHONPATH
python deployment/real_infer/server.py \
  --host 0.0.0.0 \
  --port 9011 \
  --ckpt-path /path/to/checkpoints/steps_xxx.pt
```

如果只有一张 GPU，也可以先继续用同一张卡起两个进程，只是主控制推理和 backfill 推理不会获得理想的硬件隔离。

## 终端 3: robot-service

```bash
cd /home/hello/codebase/serl_torch/examples/agibot_real
bash tools/start_robot_service.sh
```

## 终端 4: learner

```bash
cd /home/hello/codebase/serl_torch
conda activate robot
export PYTHONPATH=/home/hello/codebase:/home/hello/codebase/serl_torch:/home/hello/codebase/serl_torch/serl_launcher:$PYTHONPATH
python examples/agibot_real/scripts/run_residual_training_copy.py \
  runtime.role=learner
```

## 终端 5: actor

先准备 robot runtime：

```bash
cd /home/hello/codebase/serl_torch/examples/agibot_real
source robot/service/env.sh
```

然后启动 actor：

```bash
cd /home/hello/codebase/serl_torch
conda activate robot
export PYTHONPATH=/home/hello/codebase:/home/hello/codebase/serl_torch:/home/hello/codebase/serl_torch/serl_launcher:$PYTHONPATH
python examples/agibot_real/scripts/run_residual_training_copy.py \
  runtime.role=actor
```

## 机器人复位

有两种常用方式。

### 方式 1: actor 里按 `r`

- 在 actor 的 terminal controller 里按 `r`
- 当前回合会被标成 `truncated`
- 外层训练循环会调用当前配置里的 `reset_hook`
- 当前这份配置默认就是 `reset_to_task_initial_pose`

### 方式 2: 单独执行一次复位脚本

```bash
cd /home/hello/codebase/serl_torch
conda activate robot
export PYTHONPATH=/home/hello/codebase:/home/hello/codebase/serl_torch:/home/hello/codebase/serl_torch/serl_launcher:$PYTHONPATH
python examples/agibot_real/scripts/reset_robot.py --task-name office_setting
```

如果要直接按默认任务复位，也可以不传参数：

```bash
cd /home/hello/codebase/serl_torch
conda activate robot
export PYTHONPATH=/home/hello/codebase:/home/hello/codebase/serl_torch:/home/hello/codebase/serl_torch/serl_launcher:$PYTHONPATH
python examples/agibot_real/scripts/reset_robot.py
```

注意：

- 这个脚本只会把机器人姿态复位到任务初始位，不会重置场景里的物体状态。
- 当前 `train_residual_copy.yaml` 的任务名是 `office_setting`，所以手动复位时最合适的命令是 `--task-name office_setting`。

## 备注

- 因为 `backfill_policy.port=9011` 已经写进 yaml，所以现在默认就是双 JoyRA server 模式。
- 如果只起一个 JoyRA server，actor 在 backfill 阶段会尝试连接 `9011` 并失败。
- 因为 `residual.chunk_horizon=15` 已经写进 yaml，所以 actor 和 learner 不需要再额外传这个 override。
