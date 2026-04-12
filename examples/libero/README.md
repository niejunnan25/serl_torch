# LIBERO Example

这份 README 只描述当前 `examples/libero/` 目录里真实存在、还能对上的主流程。

如果你想先看仓库整体结构，请回到：

- [../../README.md](../../README.md)

## 这个目录现在负责什么

`examples/libero/` 当前主要负责四件事：

1. LIBERO 环境适配，包括本地 env 和 remote env client/server。
2. residual RL 训练入口。
3. residual observation schema。
4. LIBERO 相关的补充设计文档和实验输出目录。

当前这条主线已经比较收敛，目录里真正的主入口只有两个：

- [scripts/run_residual_training.py](scripts/run_residual_training.py)
- [scripts/serve_env.py](scripts/serve_env.py)

也就是说，这里现在没有单独的：

- `run_actor.py`
- `run_learner.py`
- `serve_openpi.sh`
- 离线数据转换脚本
- 独立评测脚本

如果你在旧文档里看到这些名字，请以当前目录结构为准。

## 目录结构

- [configs/train_residual.yaml](configs/train_residual.yaml)
  当前 residual 训练的 canonical Hydra 配置
- [scripts/run_residual_training.py](scripts/run_residual_training.py)
  训练入口，通过 `runtime.role=actor|learner` 切 actor / learner
- [scripts/serve_env.py](scripts/serve_env.py)
  LIBERO HTTP RPC env server
- [tools/serve_env.sh](tools/serve_env.sh)
  启动 env server 的 shell 包装
- [config.py](config.py)
  typed config 定义与解析
- [residual_observation.py](residual_observation.py)
  residual observation 构造和 observation space 定义
- [env/](env/)
  LIBERO path setup、本地 env、remote env、观测解析
- [docs/](docs/)
  补充设计说明和流程图
- [outputs/](outputs/)
  历史实验输出

## 当前训练链路

当前实现是一个 `base chunk policy + residual DRQ-SAC` 的异步 actor / learner 流程：

1. actor 连接 policy backend，当前支持：
   - `policy.type=openpi`
   - `policy.type=joyra`
2. actor 从 base policy 拿到一个 `chunk_horizon` 长度的 `base_action_chunk`
3. actor 用当前 observation 构造 residual observation：
   - 图像
   - `robot_proprio`
   - `base_action`
   - `base_action_chunk`
   - `alpha`
4. residual agent 输出 residual chunk
5. `ResidualActionSpec` 把 base chunk 和 residual chunk 组合成最终动作
6. actor 与环境交互，并把 transition 发给 learner
7. learner 从 replay 采样，更新 DRQ-SAC，再通过 agentlace 广播最新参数给 actor

当前实现里，actor 在每一步环境交互后都会重新计算 `next_base_action_chunk`，所以它更接近：

- step-wise receding-horizon residual training

如果你要看这个问题定义的讨论，可以看：

- [docs/chunk_residual_mdp_discussion.md](docs/chunk_residual_mdp_discussion.md)

## 依赖和运行环境

最常见的环境拆分是：

- `serl_torch`
  actor / learner
- `libero`
  env server
- 你自己的 policy server 环境
  OpenPI 或 JoyRA 服务

最小安装通常至少要有：

```bash
cd /vla/users/niejunnan/codebase/serl_torch
conda activate serl_torch
pip install -e serl_launcher
```

如果你使用 `policy.type=openpi`，actor 进程还需要能导入：

```bash
pip install -e /path/to/openpi/packages/openpi-client
```

`train_residual.yaml` 当前还默认把 ResNet-18 路径写成了本机绝对路径：

```yaml
encoder:
  resnet:
    model_name: /vla/users/niejunnan/codebase/serl_torch/pretrained_models/microsoft--resnet-18
```

如果你的机器路径不同，记得在命令行里覆盖：

```bash
encoder.resnet.model_name=/your/path/to/microsoft--resnet-18
```

## LIBERO 路径是怎么解析的

[env/setup.py](env/setup.py) 会自动处理 LIBERO 相关路径。

### `libero_root`

优先级：

1. 命令行显式传 `libero_root=...`
2. 仓库内默认候选：
   - `<repo>/third_party/LIBERO`
   - 其他 repo candidate

要求这个目录必须是完整的 LIBERO checkout，至少包含：

- `libero/libero/bddl_files`
- `libero/libero/init_files`
- `libero/libero/assets/scenes/libero_tabletop_base_style.xml`

### `libero_datasets_root`

优先级：

1. 命令行显式传 `libero_datasets_root=...`
2. 环境变量 `LIBERO_DATASETS_ROOT`
3. repo 邻近的默认候选目录

### `libero_config_dir`

优先级：

1. 命令行显式传 `libero_config_dir=...`
2. 环境变量 `LIBERO_CONFIG_PATH`
3. 默认缓存目录：
   - `$XDG_CACHE_HOME/serl_torch/libero_config`
   - 如果没设 `XDG_CACHE_HOME`，则是 `~/.cache/serl_torch/libero_config`

运行时会自动生成 `config.yaml` 并设置 `LIBERO_CONFIG_PATH`，所以通常不需要手工准备 LIBERO 配置文件。

## 当前默认配置

当前 canonical 配置是：

- [configs/train_residual.yaml](configs/train_residual.yaml)

默认关键参数包括：

- `task.suite_name=libero_10`
- `task.task_id=8`
- `runtime.role=actor`
- `policy.type=openpi`
- `policy.host=localhost`
- `policy.port=30001`
- `env.backend=remote`
- `env.remote.host=127.0.0.1`
- `env.remote.port=30000`
- `env.action_dim=7`
- `residual.alpha=0.35`
- `residual.chunk_horizon=5`
- `training.training_starts=1000`
- `training.max_env_steps=300000`
- `training.max_update_steps=300000`

另外当前配置解析还有两个显式约束：

- `obs.stack_horizon` 目前必须是 `1`
- `encoder.use_proprio=true` 时，`obs.vector_obs_keys` 不能为空

## 推荐启动顺序

当前推荐至少开 4 个终端：

1. LIBERO env server
2. base policy server
3. learner
4. actor

### 1. 启动 LIBERO env server

从 repo root：

```bash
cd /vla/users/niejunnan/codebase/serl_torch
bash examples/libero/tools/serve_env.sh \
  --host 127.0.0.1 \
  --port 30000
```

`tools/serve_env.sh` 会：

- 切到 `examples/libero/`
- 尝试加载 `/vla/miniconda3/etc/profile.d/conda.sh`
- 默认激活 `/vla/users/niejunnan/envs/libero`
- 最终执行 `scripts/serve_env.py`

如果你的环境不同，可以覆盖：

- `LIBERO_CONDA_ENV`
- `LIBERO_CONDA_PREFIX`
- `LIBERO_PYTHON_BIN`

例如：

```bash
LIBERO_CONDA_ENV=libero bash examples/libero/tools/serve_env.sh --port 30000
```

### 2. 启动 base policy server

这一步当前不在 `examples/libero/` 目录内提供脚本。

你需要自己准备一个和当前 client 协议兼容的服务，并让它监听：

- `policy.host`
- `policy.port`

默认配置下是：

- `localhost:30001`

当前代码支持两种 backend：

- `policy.type=openpi`
  由 [serl_launcher/serl_launcher/policy/openpi/client.py](../../serl_launcher/serl_launcher/policy/openpi/client.py) 连接
- `policy.type=joyra`
  由 [serl_launcher/serl_launcher/policy/joyra/client.py](../../serl_launcher/serl_launcher/policy/joyra/client.py) 连接

这个 example 本身不负责启动它们。

### 3. 启动 learner

```bash
cd /vla/users/niejunnan/codebase/serl_torch/examples/libero
conda run -n serl_torch python scripts/run_residual_training.py \
  runtime.role=learner \
  libero_root=/vla/users/niejunnan/codebase/serl_torch/third_party/LIBERO \
  libero_datasets_root=/vla/users/niejunnan/datasets \
  encoder.resnet.model_name=/vla/users/niejunnan/codebase/serl_torch/pretrained_models/microsoft--resnet-18 \
  hydra.run.dir=/vla/users/niejunnan/codebase/serl_torch/examples/libero/outputs/libero/train_residual/learner
```

### 4. 启动 actor

```bash
cd /vla/users/niejunnan/codebase/serl_torch/examples/libero
conda run -n serl_torch python scripts/run_residual_training.py \
  runtime.role=actor \
  libero_root=/vla/users/niejunnan/codebase/serl_torch/third_party/LIBERO \
  libero_datasets_root=/vla/users/niejunnan/datasets \
  encoder.resnet.model_name=/vla/users/niejunnan/codebase/serl_torch/pretrained_models/microsoft--resnet-18 \
  hydra.run.dir=/vla/users/niejunnan/codebase/serl_torch/examples/libero/outputs/libero/train_residual/actor
```

### 5. actor / learner 必须对齐的配置

至少这些字段必须一致：

- `runtime.trainer_host`
- `runtime.trainer_port`
- `runtime.broadcast_port`
- `policy.host`
- `policy.port`
- `env.backend`
- 如果是 remote env，还包括：
  - `env.remote.host`
  - `env.remote.port`

如果这些参数没有对齐，最常见的现象是：

- actor 连接不上 learner
- actor 拿不到最新参数
- actor 连不上 env server
- actor 连不上 policy server

## `env.backend=local` 和 `env.backend=remote` 的区别

### `env.backend=remote`

这是当前默认模式。

优点：

- actor / learner 的 Python 环境不需要直接跑 LIBERO / MuJoCo / OpenGL
- 环境相关依赖集中在 env server 那个进程里

要求：

- 先启动 `scripts/serve_env.py`
- `env.remote.host` / `env.remote.port` 配对正确

### `env.backend=local`

这种模式不走 HTTP RPC，actor 进程直接创建 `LiberoTaskEnv`。

优点：

- 不需要单独起 env server

代价：

- actor 所在环境必须能直接 import LIBERO
- actor 所在环境也必须具备 MuJoCo / 渲染相关依赖

如果你切到 local mode，通常需要至少覆盖：

```bash
env.backend=local
```

同时可以去掉 `env.remote.*` 的实际依赖。

## 输出和日志

训练脚本退出前会在每个 run dir 下写：

- `summary.json`

learner 通常还会写：

- `checkpoints/`
- `wandb/`

如果你没有显式传 `hydra.run.dir`，输出目录会遵循 `train_residual.yaml` 里的默认规则：

```yaml
hydra:
  run:
    dir: ${launch.output_root}/${hydra:job.config_name}/${now:%Y-%m-%d_%H-%M-%S}
```

当前 `launch.output_root` 默认是：

- `outputs/libero`

为了方便区分 actor / learner，实际使用时更推荐手工指定不同的 `hydra.run.dir`。

## 观测和动作语义

当前 residual observation 的构造逻辑在：

- [residual_observation.py](residual_observation.py)
- [env/observation.py](env/observation.py)
- [env/policy_input.py](env/policy_input.py)

当前默认观测约定：

- 图像会被处理成 `224 x 224`
- `robot_proprio` 维度是 8
- 支持的 image key 映射包括：
  - `image -> image_rgb_0`
  - `wrist_image -> image_rgb_1`
  - `image_rgb_0/1/2`

当前 `train_residual.yaml` 里默认是：

```yaml
obs:
  image_keys: [image, wrist_image]
  vector_obs_keys: [robot_proprio, base_action_chunk, alpha]
```

动作方面要注意：

- `env.action_dim` 必须和运行时 LIBERO env 的 action space 维度一致
- 当前默认是 7
- 如果不一致，`LiberoTaskEnv` 会直接报错

## 相关文档

当前还值得看的补充文档有：

- [docs/chunk_residual_mdp_discussion.md](docs/chunk_residual_mdp_discussion.md)
- [docs/libero_current_framework.md](docs/libero_current_framework.md)

## 常见问题

### 1. 找不到 LIBERO checkout

优先检查：

- `libero_root=...` 是否传对
- `third_party/LIBERO` 是否真的是完整 checkout

### 2. 找不到 LIBERO datasets

优先检查：

- `libero_datasets_root=...`
- `LIBERO_DATASETS_ROOT`

### 3. 找不到 LIBERO config

优先检查：

- `libero_config_dir=...`
- `LIBERO_CONFIG_PATH`
- `~/.cache/serl_torch/libero_config/config.yaml`

### 4. actor 能连 learner，但环境起不来

优先检查：

- `env.backend` 是否为 `remote`
- `tools/serve_env.sh` 是否已经启动
- `env.remote.host` / `env.remote.port` 是否一致

### 5. actor 卡在 policy 推理

优先检查：

- `policy.type` 是否和实际服务一致
- `policy.host` / `policy.port` 是否正确
- `openpi-client` 或 JoyRA 依赖是否已安装

### 6. 训练一启动就报 ResNet 权重路径错误

优先检查：

- `encoder.resnet.model_name` 是否仍然指向别人机器上的绝对路径

### 7. `obs.stack_horizon` 改成了大于 1 之后报错

这是当前实现的显式限制，不是 Hydra 配置写法问题。LIBERO residual DRQ 目前只支持：

- `obs.stack_horizon=1`

### 8. learner 一直在等 replay buffer

当前 learner 会在 `len(replay_buffer) >= training.training_starts` 之前一直等待。

默认值是：

- `training.training_starts=1000`

如果 actor 没在正常采样，learner 会持续打印 filling replay buffer 日志。
