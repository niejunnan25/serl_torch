# AgiBot Real Residual RL

`examples/agibot_real/` 当前推荐的真机 residual-RL copy 主线是：

- 配置: [configs/train_residual_copy.yaml](configs/train_residual_copy.yaml)
- 训练入口: [scripts/run_residual_training_copy.py](scripts/run_residual_training_copy.py)

这份 README 只围绕这条主线来写。
如果你现在在看 `run_residual_training.py`、旧 wrapper、或者更早的 optimized 文档，请把它们当成历史参考或 legacy baseline，而不是当前推荐流程。

如果你想先看仓库整体结构，请回到 [../../README.md](../../README.md)。

## 当前主线

当前推荐流程是：

- actor / learner 仍然共用一个入口，通过 `runtime.role=actor|learner` 切角色
- actor 执行 `step_chunk`
- 训练数据仍然按 per-step transition 写入 replay，不是直接存 chunk
- base policy 当前推荐用 JoyRA
- copy 训练线支持 chunk 级 batched backfill
- 默认 copy 配置已经启用 `async_commit` transport 和 async backfill
- 如果再切到 dedicated backfill server，可以进一步降低控制时延竞争

这条线和旧版本最大的区别是：

- backfill 不再默认逐 observation 串行推理
- JoyRA 路径下，一个 chunk 的 post-step observations 可以合成一次 batched infer
- 在默认 async backfill 模式下，episode 结束后可以先做物理 reset，再收尾上一局 replay commit

## 新流程怎么工作

先给一个高层图：

1. actor 从主 policy server 拿当前 chunk 的 base action chunk
2. residual policy 产生 residual action
3. 两者组合后执行 `env.step_chunk(...)`
4. `step_chunk` 返回 chunk 内每一步的 obs / reward / done / info
5. backfill 路径根据这些 post-step observations 构造下一步 residual observations
6. 组装 per-step transitions，按顺序写入 replay
7. learner 按已 commit 的 env steps 触发更新

### 默认异步 backfill

这是当前默认 [train_residual_copy.yaml](configs/train_residual_copy.yaml) 的行为，因为：

- `policy.type=joyra`
- `runtime.trainer_transport.mode=async_commit`
- `backfill_policy.enabled=true`
- `backfill_policy.port=${policy.port}`

此时会启动 async backfill coordinator，而且 copy 训练线会走 JoyRA chunk 级 batch infer：

- 主 actor 仍然只向主 policy server 发单样本 chunk 决策请求
- 每个 chunk 的 backfill 仍然是一次 batched JoyRA 请求
- 默认情况下，这些 batch backfill 请求也打到主 policy server
- staged reset 已经会生效，因为 async coordinator 已经存在

这里要特别区分：

- `batched backfill`
  指一个 chunk 内的多个 `post_step_observations` 会被打包成一次 `infer_many(...)` 请求
- `async backfill coordinator`
  指 backfill 这条链路在 actor 主线程之外异步推进
- `parallel inference`
  指主控制推理和 backfill 推理是否真的落到两个独立 server / GPU 上并行执行

当前默认配置只有前两者，不自动拥有第三者。
原因是 JoyRA server 虽然支持 batch 请求，但单个 server 进程里的推理调用仍然是同步执行的；如果主控制和 backfill 共用同一个 JoyRA server，它们本质上还是在争同一个推理执行点。

所以要区分两件事：

- `batched backfill`
- `async backfill coordinator`
- `dedicated backfill server`

前两者在当前默认 copy 配置里已经有了。
第三者只有在你把 `backfill_policy.port` 指到单独的 JoyRA server 时才会启用。

如果你显式把 `backfill_policy.enabled=false`，copy 训练线才会退回同步 batched backfill fallback：

- batched backfill 还在
- 但 backfill 会回到 actor 主线程
- staged reset 也不会触发

### Dedicated Backfill Server

当你保持 `backfill_policy.enabled=true`，并把 `backfill_policy.port` 指到单独的 JoyRA server 端口时，copy 训练线会变成：

- 主 actor 仍然只向主 policy server 发单样本 chunk 决策请求
- transition assembly 走单独的 `AsyncTransitionAssemblyCoordinator`
- 每个 chunk 只提交 1 个 backfill job
- 非终止 chunk:
  回填 `H-1` 个 `post_step_observations`，最后一个 tail 由下一次 decision obs handoff
- 终止 / 截断 chunk:
  回填全部 `H` 个 `post_step_observations`
- replay commit 按 chunk seq 严格顺序进行

这里的并行语义也要说清楚：

- 对单个 chunk 的 backfill 来说，当前是 `1 个 chunk -> 1 次 batched infer_many RPC`
- 对 actor 进程来说，当前只会有 1 条后台 transition assembly 流，因为 `AsyncTransitionAssemblyCoordinator` 的执行器是 `max_workers=1`
- 对系统来说，只有当你把主 policy server 和 backfill server 分成两个独立 JoyRA 进程时，主控制推理和 backfill 推理才是“系统级并行”

如果这两个 JoyRA server 再分别放到不同 GPU 上，例如：

- 主 policy: `--gpu-id 0`
- backfill policy: `--gpu-id 1`

那才是当前最理想的 dedicated backfill 部署。

如果你只是把两个 server 起在不同端口，但仍然绑在同一张 GPU 上，也会比单 server 更解耦一些，但 GPU 计算资源仍然会互相争抢，收益通常不如分 GPU 明显。

### Episode 边界的 staged reset

当前 copy 主线已经支持 staged reset：

- episode 结束后，如果启用了 async backfill，会先执行 `prepare_episode_reset()`
- 这一步只做物理 reset / reset hook，不提前打开下一局 controller
- 然后继续 drain 上一局的 `commit_replay`
- 到下一局真正开始时，再执行 `start_episode_after_reset()`
- 这时才抓最新 obs，并真正 `start_episode()`

这样做解决了两个真机问题：

- 不会再出现“上一局还在 commit，下一局 controller 已经被提前激活，`s/f/r` 误打到下一局”的问题
- 不会复用几秒前缓存下来的旧 reset obs，下一局首帧会重新抓最新观测

## 关键文件

- [configs/train_residual_copy.yaml](configs/train_residual_copy.yaml)
  当前推荐 copy 训练配置
- [configs/train_residual_optimized.yaml](configs/train_residual_optimized.yaml)
  legacy `optimized` 配置，保留作 `sync_commit` 对照
- [config.py](config.py)
  typed config 定义与解析
- [scripts/run_residual_training_copy.py](scripts/run_residual_training_copy.py)
  当前推荐训练入口
- [transition_assembly.py](transition_assembly.py)
  chunk -> step transitions 的后处理与 batched backfill 逻辑
- [env/task_env.py](env/task_env.py)
  真机环境主体，包含 staged reset
- [env/controller.py](env/controller.py)
  人工 gating / success / fail / reset 控制器
- [env/base_policy.py](env/base_policy.py)
  AgiBot example-local base policy adapter
- [residual_observation.py](residual_observation.py)
  residual observation schema
- [docs/optimized_training_startup.md](docs/optimized_training_startup.md)
  针对旧 optimized 启动方式和 copy 训练线的补充说明
- [docs/real_robot_startup_guide.md](docs/real_robot_startup_guide.md)
  真机 bring-up / 训练 / 评估通用说明

下面这些文件目前不是这份 README 的主线：

- [scripts/run_residual_training.py](scripts/run_residual_training.py)
- [tools/run_actor.sh](tools/run_actor.sh)
- [tools/run_learner.sh](tools/run_learner.sh)

原因很简单：

- `run_actor.sh` / `run_learner.sh` 仍然指向 `run_residual_training.py`
- 这份 README 以 `run_residual_training_copy.py` 为准

所以如果你按本文档启动，请直接运行 Python 入口，不要默认走旧 wrapper。

## 环境准备

最常见的环境拆分是：

- `serl_torch`
  actor / learner / 本仓库代码
- `robot`
  真机 runtime
- `joyra`
  JoyRA policy server

最小安装通常至少包括：

```bash
cd /home/hello/codebase/serl_torch
conda activate serl_torch
pip install -r serl_launcher/requirements.txt
pip install -e ./serl_launcher
```

如果不是 editable install，通常还需要：

```bash
export PYTHONPATH=/home/hello/codebase:/home/hello/codebase/serl_torch/serl_launcher:$PYTHONPATH
```

## Robot 运行时

actor 所在终端需要先加载 repo-local robot runtime：

```bash
cd /home/hello/codebase/serl_torch/examples/agibot_real
source robot/service/env.sh
```

如果 forwarder / vendored runtime 还没准备好，先执行：

```bash
cd /home/hello/codebase/serl_torch/examples/agibot_real
bash tools/prepare_robot_runtime.sh --from-dir /path/to/forwarder
```

如果你要启动 repo-local robot-service，可以再参考：

- [tools/start_robot_service.sh](tools/start_robot_service.sh)
- [scripts/start_robot_service.py](scripts/start_robot_service.py)

## JoyRA Server

当前推荐直接复用 JoyRA 仓库里的：

- `JoyRA/deployment/real_infer/server.py`

这个 server 现在既能处理主 actor 的单样本请求，也能处理 backfill 的 batch 请求。
如果你启用 async dedicated backfill，推荐启动两个独立实例：

- 主 policy server: 例如 `9001`
- backfill server: 例如 `9011`

示例：

```bash
cd /home/hello/codebase/serl_torch/examples/agibot_real
bash tools/serve_joyra.sh --joyra-root /path/to/JoyRA --ckpt-path /path/to/checkpoint.pt --port 9001
```

再开一个终端：

```bash
cd /home/hello/codebase/serl_torch/examples/agibot_real
bash tools/serve_joyra.sh --joyra-root /path/to/JoyRA --ckpt-path /path/to/checkpoint.pt --port 9011
```

如果你先想走默认单 server 配置，只启动主 server 即可。

## 推荐启动顺序

### 开箱默认流程

这条流程最容易先跑通：

1. 启动 JoyRA 主 policy server
2. 准备 robot runtime
3. 启动 robot-service
4. 启动 learner
5. 在机器人终端启动 actor

此时配置保持：

```yaml
backfill_policy:
  enabled: true
  port: ${policy.port}
```

特点是：

- copy 训练线已经有 chunk 级 batched backfill
- async coordinator 和 staged reset 已经生效
- 但主 actor 请求和 backfill 请求仍然共享同一个 JoyRA server

### 推荐双 Server 真机流程

如果你的目标是降低 chunk 卡顿和 episode 边界等待，推荐：

1. 启动 JoyRA 主 policy server
2. 启动 JoyRA backfill server
3. 准备 robot runtime
4. 启动 robot-service
5. 启动 learner
6. 在机器人终端启动 actor，并把 `backfill_policy.port` 指到 dedicated backfill server

## 启动命令

### Learner

```bash
cd /home/hello/codebase/serl_torch
conda activate serl_torch
export PYTHONPATH=/home/hello/codebase:/home/hello/codebase/serl_torch/serl_launcher:$PYTHONPATH
python examples/agibot_real/scripts/run_residual_training_copy.py \
  runtime.role=learner
```

### Actor

先在 actor 终端准备 robot 环境：

```bash
cd /home/hello/codebase/serl_torch/examples/agibot_real
source robot/service/env.sh
```

然后回到 repo root 启动：

```bash
cd /home/hello/codebase/serl_torch
conda activate serl_torch
export PYTHONPATH=/home/hello/codebase:/home/hello/codebase/serl_torch/serl_launcher:$PYTHONPATH
python examples/agibot_real/scripts/run_residual_training_copy.py \
  runtime.role=actor
```

### Actor with Dedicated Backfill

```bash
cd /home/hello/codebase/serl_torch
conda activate serl_torch
export PYTHONPATH=/home/hello/codebase:/home/hello/codebase/serl_torch/serl_launcher:$PYTHONPATH
python examples/agibot_real/scripts/run_residual_training_copy.py \
  runtime.role=actor \
  backfill_policy.enabled=true \
  backfill_policy.host=127.0.0.1 \
  backfill_policy.port=9011 \
  backfill_policy.max_pending_chunks=4
```

## 默认配置和推荐配置

当前 [train_residual_copy.yaml](configs/train_residual_copy.yaml) 的默认值是：

- `policy.type=joyra`
- `policy.port=9001`
- `runtime.trainer_transport.mode=async_commit`
- `runtime.trainer_transport.data_port=5490`
- `backfill_policy.enabled=true`
- `backfill_policy.port=${policy.port}`
- `backfill_policy.max_pending_chunks=4`
- `residual.chunk_horizon=30`
- `task.hz=30`
- `controller.terminal_grace_sec=0.15`
- `training.training_starts=1000`
- `training.steps_per_update=30`
- `training.critic_actor_ratio=4`
- `sac.utd_ratio=2`

怎么理解这份默认配置：

- `mode=async_commit`
  代表 actor 和 learner 之间的数据面/控制面已经拆开，常规 `update()` 只追求 accepted，shutdown 时再等 committed
- `enabled=true` 且 `port=${policy.port}`
  代表默认先走“异步 backfill on 主 server”，所以 staged reset 已经会生效
- `max_pending_chunks=4`
  这是当前推荐的 backlog / stale guard 起点，更适合真机 dedicated backfill
- `chunk_horizon=30` 且 `hz=30`
  一个 chunk 大约对应 1 秒控制时间
- `terminal_grace_sec=0.15`
  现在已经接进 controller 运行时，会在 terminal 事件后短暂忽略 `s/f/r`，避免 episode 边界粘键

如果你想回退到旧的 trainer 通信方式，可以继续使用 [train_residual_optimized.yaml](configs/train_residual_optimized.yaml)。它保留的是：

- `runtime.trainer_transport.mode=sync_commit`
- `backfill_policy.port=${policy.port}`
- `backfill_policy.max_pending_chunks=10`

如果你现在是在真机上追求更流畅的交互，我更推荐从下面这个组合开始：

```yaml
backfill_policy:
  enabled: true
  host: 127.0.0.1
  port: 9011
  max_pending_chunks: 4
  mode: thread
```

原因是：

- 你已经有 JoyRA chunk 级 batch backfill
- dedicated backfill server 能把控制时延和 replay 组装时延解耦
- `max_pending_chunks=4` 比 `10` 更适合在线真机 RL 的 replay 新鲜度

如果你只是想做最小化 debug，也可以临时把 `enabled=false` 关掉 async backfill，退回同步 batched fallback。

## 当前实现的几个约束

- `env.backend` 目前只支持 `local`
- `task.control_mode` 必须是 `camera_position`
- `env.action_dim` 必须是 `14`
- `obs.stack_horizon` 目前必须是 `1`
- `backfill_policy.mode` 当前只支持 `thread`

在 AgiBot 这条线里：

- OpenPI / JoyRA 最终都会被 canonicalize 成 AgiBot 的 14D action chunk
- JoyRA server 真实返回的是 raw 18D chunk，再由 AgiBot 侧裁到 canonical 14D

## 当前 observation / residual schema

当前 residual learner 主要使用这些字段：

- `robot_proprio`
- `base_action`
- `base_action_chunk`
- `alpha`
- `image_rgb_0`
- `image_rgb_1`
- `image_rgb_2`

当前 base policy 图像输入是：

- 头部相机
- 左腕相机
- 右腕相机

## 一些容易混淆的点

### 为什么这里不再显式传 `--config-name`

[run_residual_training_copy.py](scripts/run_residual_training_copy.py) 现在默认就是：

- `config_name="train_residual_copy"`

所以按本文档这条主线启动时，不需要再额外传 `--config-name`。只有当你想回退到旧的 `train_residual_optimized.yaml` 或其它实验 yaml 时，才需要显式 override。

### 为什么 README 不再推荐 `tools/run_actor.sh`

因为当前 `tools/run_actor.sh` / `tools/run_learner.sh` 还是指向旧的：

- [scripts/run_residual_training.py](scripts/run_residual_training.py)

而这份 README 以：

- [scripts/run_residual_training_copy.py](scripts/run_residual_training_copy.py)

为准。

### `backfill_policy.enabled=true` 是不是就必须再起一个 backfill server

不是。

默认 `backfill_policy.port=${policy.port}` 就是“单 JoyRA server + async backfill coordinator”。
只有当你想把主 actor 决策请求和 backfill 请求彻底解耦时，才需要再起第二个 JoyRA server，并把 `backfill_policy.port` 指过去。

## 相关文档

- [docs/optimized_training_startup.md](docs/optimized_training_startup.md)
- [docs/real_robot_startup_guide.md](docs/real_robot_startup_guide.md)
- [docs/chunk_execution_and_replan_notes.md](docs/chunk_execution_and_replan_notes.md)
- [docs/chunk_by_chunk_episode_end_backfill.md](docs/chunk_by_chunk_episode_end_backfill.md)
