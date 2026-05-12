# AgiBot `train_residual_copy.yaml` Startup

这份文档只回答一件事：如何用当前推荐配置
[../configs/train_residual_copy.yaml](../configs/train_residual_copy.yaml)
拉起完整训练。

对应训练入口：

- [../scripts/run_residual_training_copy.py](../scripts/run_residual_training_copy.py)

当前默认脚本入口就是 `train_residual_copy`，所以不需要再额外传 `--config-name train_residual_copy`。

## 推荐拓扑

如果你要当前更快、更稳的真机训练链路，推荐用双 JoyRA server：

1. 主 policy server
2. dedicated backfill server
3. robot-service
4. learner
5. actor

这样主控制推理和 backfill 推理不共用一个端口，真机 chunk 卡顿会更少。

这里的语义是：

- 主 actor chunk 决策仍然是单样本推理
- backfill 仍然是 `1 个 chunk -> 1 次 batched infer_many`
- 只有把主 policy server 和 backfill server 分成两个独立 JoyRA 进程时，主控制推理和 backfill 推理才是系统级并行
- 最好再把两个 server 放到不同 GPU 上

一句话说：

- 按下面的双 server 方式启动，就是当前推荐的最佳部署
- 但“最佳”成立的前提是你真的起了两个独立 JoyRA server，且最好分到不同 GPU

默认端口建议：

- 主 policy server: `9001`
- backfill server: `9011`
- trainer port: `5488`
- broadcast port: `5489`
- split-queue data port: `5490`

## 终端 1: 主 JoyRA policy server

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

## 终端 2: dedicated backfill JoyRA server

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

## 终端 3: robot-service

```bash
cd /home/hello/codebase/serl_torch/examples/agibot_real
bash tools/start_robot_service.sh
```

## 终端 4: learner

```bash
cd /home/hello/codebase/serl_torch
conda activate serl_torch
export PYTHONPATH=/home/hello/codebase:/home/hello/codebase/serl_torch:/home/hello/codebase/serl_torch/serl_launcher:$PYTHONPATH
RUN_GROUP=office_setting_2026-05-13_001
python examples/agibot_real/scripts/run_residual_training_copy.py \
  runtime.role=learner \
  launch.run_group=${RUN_GROUP}
```

## 终端 5: actor

先准备 robot runtime：

```bash
cd /home/hello/codebase/serl_torch/examples/agibot_real
source robot/service/env.sh
```

然后回到 repo root 启动 actor：

```bash
cd /home/hello/codebase/serl_torch
conda activate serl_torch
export PYTHONPATH=/home/hello/codebase:/home/hello/codebase/serl_torch:/home/hello/codebase/serl_torch/serl_launcher:$PYTHONPATH
RUN_GROUP=office_setting_2026-05-13_001
python examples/agibot_real/scripts/run_residual_training_copy.py \
  runtime.role=actor \
  backfill_policy.port=9011 \
  launch.run_group=${RUN_GROUP}
```

这里的 `backfill_policy.port=9011` 很关键。默认 yaml 里 backfill 还是跟主 policy 共用 `9001`，如果你想吃到 dedicated backfill server 的收益，就要显式改到 `9011`。

如果你只有一张 GPU，也可以把两个 server 都起起来，但那样只是进程和端口解耦，不是最理想的硬件并行。

## 输出目录组织

`run_residual_training_copy.py` 的 actor 和 learner 是两个独立 Hydra 进程。
如果不显式传同一个 `launch.run_group`，它们会各自按启动时刻生成目录。

当前配置推荐使用：

```text
outputs/agibot_real_copy/train_residual_copy/<run_group>/learner
outputs/agibot_real_copy/train_residual_copy/<run_group>/actor
```

这样做有两个好处：

- `learner/` 下保留 checkpoint、learner summary 和 learner 日志
- `actor/` 下保留 rollout 视频、episode log、actor summary 和 actor 日志

不要把 actor 和 learner 直接指到完全相同的 `hydra.run.dir`。
那样会让 `.hydra/`、`summary.json` 和主日志文件互相覆盖。

## 最小可运行版本

如果你现在只想先跑通，不追求最优部署，可以只起一个 JoyRA server：

### 单 server policy

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

### learner

```bash
cd /home/hello/codebase/serl_torch
conda activate serl_torch
export PYTHONPATH=/home/hello/codebase:/home/hello/codebase/serl_torch:/home/hello/codebase/serl_torch/serl_launcher:$PYTHONPATH
RUN_GROUP=office_setting_2026-05-13_001
python examples/agibot_real/scripts/run_residual_training_copy.py \
  runtime.role=learner \
  launch.run_group=${RUN_GROUP}
```

### actor

```bash
cd /home/hello/codebase/serl_torch/examples/agibot_real
source robot/service/env.sh
cd /home/hello/codebase/serl_torch
conda activate serl_torch
export PYTHONPATH=/home/hello/codebase:/home/hello/codebase/serl_torch:/home/hello/codebase/serl_torch/serl_launcher:$PYTHONPATH
RUN_GROUP=office_setting_2026-05-13_001
python examples/agibot_real/scripts/run_residual_training_copy.py \
  runtime.role=actor \
  launch.run_group=${RUN_GROUP}
```

这个最小版本会直接使用 `train_residual_copy.yaml` 里的默认值：

- `policy.port=9001`
- `backfill_policy.enabled=true`
- `backfill_policy.port=${policy.port}`
- `runtime.trainer_transport.mode=split_queue`

也就是：

- trainer 走新的 `split_queue`
- backfill 开着
- 但主控制和 backfill 共用同一个 JoyRA server

## 运行时控制

actor 起来后，terminal controller 默认按键是：

- `g`: ready
- `p`: pause
- `r`: reset
- `s`: success
- `f`: fail
- `h`: help

## 当前这份配置的关键默认值

[../configs/train_residual_copy.yaml](../configs/train_residual_copy.yaml) 当前关键值是：

- `runtime.trainer_transport.mode=split_queue`
- `runtime.trainer_transport.control_timeout_ms=2000`
- `backfill_policy.enabled=true`
- `backfill_policy.max_pending_chunks=4`
- `residual.chunk_horizon=30`
- `task.hz=30`
- `training.training_starts=1000`
- `training.steps_per_update=30`
- `training.critic_actor_ratio=4`
- `sac.utd_ratio=2`

如果你没有特殊实验需求，先直接按上面的双 server 启动方式跑。
