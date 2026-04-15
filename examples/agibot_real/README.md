# AgiBot Real Residual RL

`examples/agibot_real/` 当前保留一条 canonical residual-RL 训练主线，并补齐了：

- 临时参考版 offline data prepare / load
- 单独的 checkpoint eval 入口
- 和 `examples/libero` 对齐的 rollout payload / JSONL artifact / learner logging contract

如果你想先看仓库整体结构，请回到：

- [../../README.md](../../README.md)

## 当前主线是什么

当前 canonical 主线是：

- config:
  [configs/train_residual.yaml](configs/train_residual.yaml)
- entrypoint:
  [scripts/run_residual_training.py](scripts/run_residual_training.py)
- actor wrapper:
  [tools/run_actor.sh](tools/run_actor.sh)
- learner wrapper:
  [tools/run_learner.sh](tools/run_learner.sh)

训练拓扑和 `examples/libero` 一样：

- 一个 typed config parser：
  [config.py](config.py)
- 一个 actor / learner 共用入口，通过 `runtime.role=actor|learner` 切角色
- example 自己管理 observation、policy input、controller、robot runtime

旧的拆分训练链路和旧 eval 入口已经移除。当前 canonical 入口包括：

- 训练：
  [scripts/run_residual_training.py](scripts/run_residual_training.py)
- standalone offline prepare：
  [scripts/prepare_offline_data.py](scripts/prepare_offline_data.py)
- standalone checkpoint eval：
  [scripts/evaluate_checkpoint.py](scripts/evaluate_checkpoint.py)

这里仍然不支持 `async eval`，也不支持和 actor rollout 并发评估。真机推荐流程仍然是先训练/rollout，再单独加载 checkpoint 做 eval。

## 这个 example 的边界

当前实现是 AgiBot 真机 residual RL，约束比较明确：

- `env.backend` 只支持 `local`
- 不引入 `remote env`
- `env.action_dim` 必须是 `14`
- `task.control_mode` 必须是 `camera_position`
- base-policy 图像输入是：
  - `head`
  - `left wrist`
  - `right wrist`
- OpenPI 用 canonical 14D `state/pose`
- JoyRA 额外可以消费 18D `pose + head + waist`
- 不管 base policy backend 是 OpenPI 还是 JoyRA，进入 residual RL 前都会统一成同一个 14D dual-arm action chunk

当前 residual learner 训练时使用的 observation 字段是：

- `robot_proprio`
- `base_action`
- `base_action_chunk`
- `alpha`
- `image_rgb_0`
- `image_rgb_1`
- `image_rgb_2`

## 目录结构

- [configs/train_residual.yaml](configs/train_residual.yaml)
  当前 canonical 训练配置
- [config.py](config.py)
  typed config 定义与解析
- [scripts/run_residual_training.py](scripts/run_residual_training.py)
  actor / learner 共用训练入口
- [scripts/prepare_offline_data.py](scripts/prepare_offline_data.py)
  临时参考版 offline prepare 入口
- [scripts/evaluate_checkpoint.py](scripts/evaluate_checkpoint.py)
  单独 checkpoint eval 入口
- [eval_runner.py](eval_runner.py)
  standalone eval runner
- [scripts/start_robot_service.py](scripts/start_robot_service.py)
  repo-local robot-service Python launcher
- [tools/run_actor.sh](tools/run_actor.sh)
  actor shell wrapper
- [tools/run_learner.sh](tools/run_learner.sh)
  learner shell wrapper
- [tools/start_robot_service.sh](tools/start_robot_service.sh)
  robot-service shell wrapper
- [tools/prepare_robot_runtime.sh](tools/prepare_robot_runtime.sh)
  准备 repo-local forwarder / SDK runtime
- [tools/serve_openpi.sh](tools/serve_openpi.sh)
  OpenPI server wrapper
- [tools/serve_joyra.sh](tools/serve_joyra.sh)
  JoyRA server wrapper
- [env/base_policy.py](env/base_policy.py)
  example-local base policy adapter
- [env/controller.py](env/controller.py)
  人工 gating / success / fail / reset 控制器
- [env/task_env.py](env/task_env.py)
  真机环境主体
- [env/observation.py](env/observation.py)
  observation 解析和 policy input 所需 state / image 逻辑
- [residual_observation.py](residual_observation.py)
  residual observation schema
- `robot/service/`
  repo-local robot-service 运行时文件
- `vendor/a2d_sdk/`
  vendored SDK / forwarder 资产

## 依赖和运行环境

最常见的环境拆分是：

- `serl_torch`
  learner、actor 脚本、本仓库代码
- `robot`
  真实机器人 runtime
- `openpi` 或 `openpi-modified`
  OpenPI server
- `joyra`
  JoyRA server

最小安装通常至少包括：

```bash
cd /vla/users/niejunnan/codebase/serl_torch
conda activate serl_torch
pip install -r serl_launcher/requirements.txt
pip install -e ./serl_launcher
```

`agentlace` 需要手工安装。

如果你使用 `policy.type=openpi`，训练环境还需要能导入：

```bash
pip install -e /path/to/openpi/packages/openpi-client
```

如果不是 editable install，可以补：

```bash
export PYTHONPATH=/vla/users/niejunnan/codebase:/vla/users/niejunnan/codebase/serl_torch/serl_launcher:$PYTHONPATH
```

## Actor 和 robot-service 的 shell 环境

actor 终端需要先加载 repo-local robot runtime 环境：

```bash
cd /vla/users/niejunnan/codebase/serl_torch/examples/agibot_real
source robot/service/env.sh
```

这一步会设置真机 runtime 用到的环境变量，例如：

- `LOCATOR_IP`
- `AORTA_DISCOVERY_URI`
- `PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION`
- ROS / DDS 相关变量

`tools/run_actor.sh` 不会自动 `source robot/service/env.sh`，所以请在启动 actor 的同一个终端里先执行这一步。

`tools/start_robot_service.sh` 会自己 `source robot/service/env.sh`，所以如果你走这个 wrapper，通常不需要在启动 robot-service 前再手工做一遍。只有你直接运行 [scripts/start_robot_service.py](scripts/start_robot_service.py) 或其他底层命令时，才需要自己先准备这套环境变量。

当前 learner 已经尽量和 robot SDK import path 解耦，通常不需要 `source robot/service/env.sh`。如果你为了方便想从同一个准备好的 shell 启 learner，也可以。

## 准备 repo-local robot runtime

如果 forwarder / vendored runtime 还没准备好，先跑：

```bash
cd /vla/users/niejunnan/codebase/serl_torch/examples/agibot_real
bash tools/prepare_robot_runtime.sh --from-dir /path/to/forwarder
```

也支持：

- `--from-tar /path/to/forwarder_x86_v1.7.0.tar.gz`
- `--from-url https://...`

或者用环境变量：

- `AGIBOT_FORWARDER_DIR`
- `AGIBOT_FORWARDER_TAR`
- `AGIBOT_FORWARDER_URL`

如果你计划用 `--no-ros` 模式跑 robot-service，可以设：

```bash
export AGIBOT_NO_ROS=1
```

## 当前默认配置

canonical 配置是：

- [configs/train_residual.yaml](configs/train_residual.yaml)

当前默认关键参数：

- `policy.type=openpi`
- `policy.host=127.0.0.1`
- `policy.port=30001`
- `policy.id=pi05_agibot`
- `env.backend=local`
- `env.action_dim=14`
- `task.control_mode=camera_position`
- `task.max_episode_steps=150`
- `controller.enabled=true`
- `residual.alpha=0.2`
- `residual.chunk_horizon=50`
- `training.training_starts=1000`
- `training.steps_per_update=30`
- `training.critic_actor_ratio=4`
- `training.max_env_steps=300000`
- `training.max_update_steps=300000`
- `training.max_episodes=2000`
- `offline.enabled=false`
- `logging.episode_log_file=episode_logs.jsonl`

当前配置解析还有几个显式约束：

- `obs.stack_horizon` 目前必须是 `1`
- `encoder.use_proprio=true` 时，`obs.vector_obs_keys` 不能为空
- `env.action_dim` 目前强制为 `14`

## 推荐启动顺序

当前推荐顺序：

1. 启动 base-policy server
2. 准备 robot runtime
3. 启动 robot-service
4. 启动 learner
5. 在机器人终端启动 actor

## 1. 启动 base-policy server

### OpenPI

```bash
cd /vla/users/niejunnan/codebase/serl_torch/examples/agibot_real
OPENPI_ROOT=/path/to/openpi \
POLICY_DIR=/path/to/policy/checkpoint \
bash tools/serve_openpi.sh --port 30001
```

常见可覆盖环境变量：

- `OPENPI_ROOT`
- `POLICY_DIR`
- `DEFAULT_POLICY_DIR`
- `OPENPI_CONDA_ENV`
- `OPENPI_CONDA_PREFIX`

### JoyRA

```bash
cd /vla/users/niejunnan/codebase/serl_torch/examples/agibot_real
JOYRA_ROOT=/path/to/JoyRA \
JOYRA_CKPT_PATH=/path/to/checkpoints/steps_xxx.pt \
bash tools/serve_joyra.sh --port 9001
```

常见可覆盖环境变量：

- `JOYRA_ROOT`
- `JOYRA_CKPT_PATH`
- `JOYRA_SERVER_PY`
- `JOYRA_CONDA_ENV`
- `JOYRA_CONDA_PREFIX`

切 JoyRA 训练时，最常见的 actor / learner override 是：

```bash
policy.type=joyra policy.port=9001
```

## 2. 启动 robot-service

先确保 runtime 已经准备好，再执行：

```bash
cd /vla/users/niejunnan/codebase/serl_torch/examples/agibot_real
bash tools/start_robot_service.sh
```

这个 wrapper 会：

- 激活你指定的 conda env
- `source robot/service/env.sh`
- 使用 repo-local `robot/service/conf/copilot.pbtxt`
- 调用 [scripts/start_robot_service.py](scripts/start_robot_service.py)

如果你已经准备好 shell 环境，也可以看 help：

```bash
bash tools/start_robot_service.sh --help
```

## 3. 启动 learner

最常见的命令：

```bash
cd /vla/users/niejunnan/codebase/serl_torch/examples/agibot_real
bash tools/run_learner.sh
```

等价的直跑方式：

```bash
python scripts/run_residual_training.py runtime.role=learner
```

常见 overrides：

```bash
bash tools/run_learner.sh \
  training.max_update_steps=300000 \
  training.checkpoint.dir=checkpoints \
  wandb.project=agibot_real
```

如果要启用 prepared offline replay：

```bash
bash tools/run_learner.sh \
  offline.enabled=true \
  offline.prepared_path=/path/to/prepared/offline \
  offline.pretrain_steps=1000 \
  offline.ratio=0.5
```

## 4. 启动 actor

先在同一个终端里加载 robot runtime：

```bash
cd /vla/users/niejunnan/codebase/serl_torch/examples/agibot_real
source robot/service/env.sh
bash tools/run_actor.sh
```

等价的直跑方式：

```bash
source robot/service/env.sh
python scripts/run_residual_training.py runtime.role=actor
```

常见 overrides：

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

## controller 模式

当前默认是：

- `controller.enabled=true`

也就是 actor 会进入人工 gating 的真机工作流。
当前 canonical train/eval 配置已经强制要求 `controller.enabled=true`，关闭 controller 会在配置解析阶段直接报错。

默认终端按键：

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

当前 reset / success 逻辑可以通过 config 里的 hook 字段覆盖：

- `task.reset_hook`
- `task.success_hook`

其中当前 canonical train/eval 流程里：

- `task.reset_hook` 继续生效
- `task.success_hook` 字段保留，但已经停用，不再参与 success / done / reward 判定

成功语义已经统一收敛成 controller-only：

- controller `success` => `reward=1.0`, `done=true`, `truncated=false`, `success=true`
- controller `fail` => `reward=0.0`, `done=true`, `truncated=false`, `success=false`
- controller `reset` => `reward=0.0`, `done=false`, `truncated=true`, `success=false`
- episode step limit timeout => `reward=0.0`, `done=false`, `truncated=true`, `success=false`

## actor / learner 需要对齐的配置

至少下面这些字段需要一致：

- `runtime.trainer_host`
- `runtime.trainer_port`
- `runtime.broadcast_port`
- `policy.type`
- `policy.host`
- `policy.port`
- `env.action_dim`
- `residual.chunk_horizon`
- `obs.image_keys`
- `obs.vector_obs_keys`

## 输出目录

默认 Hydra 输出目录来自配置：

- `launch.output_root=outputs/agibot_real`

默认训练 run 会写到：

```text
outputs/agibot_real/train_residual/<timestamp>/
```

典型内容包括：

- `summary.json`
- `checkpoints/`
- `episode_logs.jsonl`
- `actor_timers.jsonl`
- `learner_timers.jsonl`
- `wandb/`

如果是 standalone eval，还会写：

- `episode_logs.jsonl`
- `summary.json`

如果是 offline prepare，还会写：

- `summary.json`
- prepared offline directory
- prepared `manifest.json`

## 当前实现上的边界

这条主线现在主要围绕下面这些文件展开：

- [config.py](config.py)
- [scripts/run_residual_training.py](scripts/run_residual_training.py)
- [env/base_policy.py](env/base_policy.py)
- [env/controller.py](env/controller.py)
- [env/task_env.py](env/task_env.py)
- [env/observation.py](env/observation.py)
- [residual_observation.py](residual_observation.py)

example-local base policy adapter 做了 backend 归一化：

- OpenPI 原生走 canonical 14D action space
- JoyRA 可以消费 18D state、返回更宽 action chunk
- `agibot_real` 会在进入 residual 观测构造和 action compose 之前，把不同 backend 统一到同一个 canonical 14D dual-arm action chunk

这是当前新 AgiBot 工作应该沿着扩展的实现形状。

## Offline Data

当前已经补了一个临时参考版 offline pipeline：

- prepare:
  [scripts/prepare_offline_data.py](scripts/prepare_offline_data.py)
- loader / manifest:
  [offline_data.py](offline_data.py)

但要明确一点：

- 这套 raw dataset 输入格式只是临时参考版，主要为了先把 config、prepared artifact、learner mixing 和 pretrain 流程接通
- 当前实现参考了 `examples/libero` 的 prepare 形状，并不代表最终 AgiBot 真机数据来源方案
- 后续一旦明确真实 AgiBot residual offline 数据来源，需要再改一轮 [offline_data.py](offline_data.py)

最小 prepare 方式示例：

```bash
python scripts/prepare_offline_data.py \
  offline.enabled=true \
  offline.prepare.raw_dataset_path=/path/to/reference_raw_dataset
```

prepare 完成后，再把生成的 `offline.prepared_path` 给 learner。

## Standalone Eval

真机不支持和 actor rollout 并发的 `async eval`，所以当前只提供 standalone checkpoint eval：

```bash
source robot/service/env.sh
python scripts/evaluate_checkpoint.py \
  eval.checkpoint_path=/path/to/checkpoints \
  eval.checkpoint_step=10000
```

也可以直接传单个 checkpoint 文件：

```bash
python scripts/evaluate_checkpoint.py \
  eval.checkpoint_path=/path/to/checkpoint_10000.pt
```

## 当前没有文档化的内容

如果你看到旧文档里还有旧 eval / 旧训练入口，请以当前目录树和这份 README 为准。
