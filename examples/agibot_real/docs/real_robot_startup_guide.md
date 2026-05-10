# AgiBot Real Robot Startup Guide

这份文档面向当前 `examples/agibot_real` 的 canonical 真机主线，目标是把真机 bring-up、训练、评估和停机流程写清楚。

当前主线以这些文件为准：

- 配置：[../configs/train_residual.yaml](../configs/train_residual.yaml)
- 训练入口：[../scripts/run_residual_training.py](../scripts/run_residual_training.py)
- actor wrapper：[../tools/run_actor.sh](../tools/run_actor.sh)
- learner wrapper：[../tools/run_learner.sh](../tools/run_learner.sh)
- robot-service wrapper：[../tools/start_robot_service.sh](../tools/start_robot_service.sh)
- eval 入口：[../scripts/evaluate_checkpoint.py](../scripts/evaluate_checkpoint.py)

如果你想直接用 optimized 配置启动训练，请看：

- [optimized_training_startup.md](optimized_training_startup.md)

## 1. 当前主线的运行语义

当前默认配置不是逐步推理、逐步执行，而是：

- `controller.enabled=true`
- `task.hz=20`
- `residual.chunk_horizon=15`

也就是：

1. 在 chunk 起点推一次 `base_policy + residual_policy`
2. 一次得到 15 个 `final_actions`
3. 真机连续执行这 15 步
4. chunk 执行完后，再回填这 15 步对应的 step transition
5. 然后再进入下一轮 chunk

这意味着默认是约 `15 / 20 = 0.75s` 的开环 chunk 执行。

## 2. 你需要准备什么

至少要有下面这些环境或资源：

- `serl_torch` Python 环境
- `robot` 真机 runtime 依赖
- `openpi` 或 `joyra` base-policy 服务
- repo-local AgiBot runtime 文件
- 机器人上电且网络可达

当前最常见的环境拆分是：

- `serl_torch`
  跑 learner、actor、本仓库 Python 代码
- `openpi` 或 `openpi-modified`
  跑 OpenPI server
- `joyra`
  跑 JoyRA server

## 3. 一次性准备

### 3.1 安装 `serl_torch`

```bash
cd /home/hello/codebase/serl_torch
conda activate serl_torch
pip install -r serl_launcher/requirements.txt
pip install -e ./serl_launcher
```

如果你用 OpenPI client，还需要能导入 OpenPI client 包。

### 3.2 准备 repo-local robot runtime

如果 `examples/agibot_real/vendor/` 下的 runtime 还没准备好，先执行：

```bash
cd /home/hello/codebase/serl_torch/examples/agibot_real
bash tools/prepare_robot_runtime.sh --from-dir /path/to/forwarder
```

也支持：

```bash
bash tools/prepare_robot_runtime.sh --from-tar /path/to/forwarder_x86_v1.7.0.tar.gz
```

或者：

```bash
export AGIBOT_FORWARDER_URL=https://...
bash tools/prepare_robot_runtime.sh
```

如果你计划不依赖 ROS：

```bash
export AGIBOT_NO_ROS=1
```

## 4. 真机启动前检查

上真机前先确认下面这些事情：

- 机器人本体已上电
- `10.42.0.*` 网段可用
- actor 所在机器能访问机器人
- base-policy server 对应端口可用
- 当前 shell 对应的 conda 环境正确
- 你准备启动 actor 的终端是一个真实 TTY

最后一点很重要。当前 controller 默认是 terminal interface，如果 actor 终端没有 TTY，键盘接口会不可用，程序会卡在 `WAIT_READY`，看起来像“没反应”。

## 5. 推荐启动顺序

当前推荐顺序是：

1. 启动 base-policy server
2. 启动 robot-service
3. 启动 learner
4. 在机器人终端启动 actor
5. 用 `g/p/r/s/f` 控制 episode

## 6. 启动 base-policy server

### 6.1 OpenPI

```bash
cd /home/hello/codebase/serl_torch/examples/agibot_real
OPENPI_ROOT=/path/to/openpi \
POLICY_DIR=/path/to/policy/checkpoint \
bash tools/serve_openpi.sh --port 30001
```

常见环境变量：

- `OPENPI_ROOT`
- `POLICY_DIR`
- `DEFAULT_POLICY_DIR`
- `OPENPI_CONDA_ENV`
- `OPENPI_CONDA_PREFIX`

如果训练和评估都用默认配置，actor / learner 里通常不需要再改 `policy.port`，默认就是 `30001`。

### 6.2 JoyRA

```bash
cd /home/hello/codebase/serl_torch/examples/agibot_real
JOYRA_ROOT=/path/to/JoyRA \
JOYRA_CKPT_PATH=/path/to/checkpoints/steps_xxx.pt \
bash tools/serve_joyra.sh --port 9001
```

如果切 JoyRA，训练脚本最常见的 override 是：

```bash
policy.type=joyra policy.port=9001
```

## 7. 启动 robot-service

标准方式：

```bash
cd /home/hello/codebase/serl_torch/examples/agibot_real
bash tools/start_robot_service.sh
```

这个 wrapper 会：

- 激活 `serl_torch` 环境
- `source robot/service/env.sh`
- 使用 repo-local `robot/service/conf/copilot.pbtxt`
- 调用 [../scripts/start_robot_service.py](../scripts/start_robot_service.py)

如果你只想看帮助：

```bash
bash tools/start_robot_service.sh --help
```

## 8. 启动 learner

最小命令：

```bash
cd /home/hello/codebase/serl_torch/examples/agibot_real
bash tools/run_learner.sh
```

等价直跑：

```bash
cd /home/hello/codebase/serl_torch/examples/agibot_real
python scripts/run_residual_training.py runtime.role=learner
```

常见 override：

```bash
bash tools/run_learner.sh \
  training.max_update_steps=300000 \
  training.checkpoint.dir=checkpoints \
  wandb.project=agibot_real
```

如果启用 prepared offline replay：

```bash
bash tools/run_learner.sh \
  offline.enabled=true \
  offline.prepared_path=/path/to/prepared/offline \
  offline.pretrain_steps=1000 \
  offline.ratio=0.5
```

## 9. 启动 actor

actor 终端必须先加载真机 runtime 环境：

```bash
cd /home/hello/codebase/serl_torch/examples/agibot_real
source robot/service/env.sh
```

然后再启动 actor：

```bash
bash tools/run_actor.sh
```

等价直跑：

```bash
python scripts/run_residual_training.py runtime.role=actor
```

常见 override：

```bash
bash tools/run_actor.sh \
  policy.type=openpi \
  policy.host=127.0.0.1 \
  policy.port=30001 \
  task.name=agibot_real_default \
  task.prompt='Pick up the object with the right hand and place it at the target location.'
```

如果切 JoyRA：

```bash
bash tools/run_actor.sh policy.type=joyra policy.port=9001
```

## 10. 运行时工作流

当前 controller 默认开启，terminal 按键是：

- `g`
  ready / resume
- `p`
  pause
- `r`
  reset
- `s`
  success
- `f`
  fail
- `h`
  help

一个典型 episode 的时序是：

1. `env.reset()`
2. reset hook 把机器人回到 task initial pose
3. controller 进入 `WAIT_READY`
4. 你按 `g`
5. actor 推一个 15-step chunk
6. 机器人连续执行这个 chunk
7. chunk 结束后脚本回填 step transition
8. 下一轮继续推下一个 chunk
9. 你按 `s`、`f` 或 `r` 结束当前 episode

## 11. 成功 / 失败 / 超时 / reset 语义

当前 canonical train/eval 是 controller-only episode 边界。

也就是：

- `s` => `reward=1.0`, `done=true`, `truncated=false`
- `f` => `reward=0.0`, `done=true`, `truncated=false`
- `r` => `reward=0.0`, `done=false`, `truncated=true`
- episode step limit => `reward=0.0`, `done=false`, `truncated=true`

当前 `task.success_hook` 字段仍然保留在配置里，但 canonical 主线已经不再用它做 success 判定。

## 12. reset 逻辑

当前默认 reset 不是复杂的 task-specific choreography，而是：

1. 调用 `task.reset_hook`
2. 默认 hook 是 `reset_to_task_initial_pose`
3. 机械臂回到任务初始位姿
4. 重新读一帧 obs
5. controller 重新进入 `WAIT_READY`

也就是说，当前 reset 语义是：

- 回到 task initial pose
- 等操作者确认
- 再开始下一回合

如果你按 `r`，当前回合会被标成 `truncated`，然后外层循环重新触发 reset。

## 13. 当前配置里最关键的字段

当前默认训练配置是 [../configs/train_residual.yaml](../configs/train_residual.yaml)，最关键的字段包括：

- `policy.type=openpi`
- `policy.host=127.0.0.1`
- `policy.port=30001`
- `runtime.trainer_port=5488`
- `runtime.broadcast_port=5489`
- `task.hz=20.0`
- `task.max_episode_steps=150`
- `controller.enabled=true`
- `residual.alpha=0.2`
- `residual.chunk_horizon=15`

如果 actor 和 learner 的这些字段不一致，最容易出问题：

- `runtime.trainer_host`
- `runtime.trainer_port`
- `runtime.broadcast_port`
- `policy.type`
- `policy.host`
- `policy.port`
- `residual.chunk_horizon`
- `obs.image_keys`
- `obs.vector_obs_keys`

## 14. 输出目录和日志

默认 Hydra 输出根目录是：

```text
output
```

训练 run 默认落在：

```text
output/train_residual/<timestamp>/
```

常见文件包括：

- `summary.json`
- `episode_logs.jsonl`
- `actor_timers.jsonl`
- `learner_timers.jsonl`
- `checkpoints/`
- `wandb/`

如果是 standalone eval，通常会看到：

- `summary.json`
- `episode_logs.jsonl`

## 15. 最常见的问题

### 15.1 actor 看起来卡住了

先检查两件事：

- 你有没有在 actor 同一个终端先 `source robot/service/env.sh`
- 这个终端是不是有 TTY，controller 能不能读到键盘输入

其次看是不是停在 `WAIT_READY`，这时要按 `g`。

### 15.2 robot-service 起不来

先确认：

- `tools/prepare_robot_runtime.sh` 是否跑过
- `robot/service/env.sh` 是否存在
- `robot/service/conf/copilot.pbtxt` 是否存在

### 15.3 policy server 连不上

先核对：

- `policy.type`
- `policy.host`
- `policy.port`
- 训练脚本和 server 实际端口是否一致

### 15.4 learner 没在更新

当前主线 learner 仍然更偏 episode-gated，而不是严格 chunk-boundary online update。长 episode 期间 learner 可能看起来比较安静，这不是 actor 没在执行。

## 16. 推荐停机顺序

推荐顺序：

1. 先结束 actor
2. 再结束 learner
3. 再停 robot-service
4. 最后停 base-policy server

不要在 actor 还在跑的时候先把 robot-service 或 policy server 杀掉。

## 17. standalone eval

当前不支持和 actor 并发的 `async eval`，推荐单独跑 checkpoint eval。

最小命令：

```bash
cd /home/hello/codebase/serl_torch/examples/agibot_real
source robot/service/env.sh
python scripts/evaluate_checkpoint.py \
  eval.checkpoint_path=/path/to/checkpoints \
  eval.checkpoint_step=10000
```

也可以直接传单个 checkpoint 文件：

```bash
python scripts/evaluate_checkpoint.py \
  eval.checkpoint_path=/path/to/checkpoints/checkpoint_10000
```

## 18. 当前主线和实验脚本的区别

你现在真正准备上真机的主线是：

- [../scripts/run_residual_training.py](../scripts/run_residual_training.py)

实验性 chunk-boundary 脚本是：

- [../scripts/run_residual_training_chunk_boundary.py](../scripts/run_residual_training_chunk_boundary.py)

当前真机 README 和 wrapper 命令，默认都以主线脚本为准，不以实验脚本为准。
