# LIBERO Example

这份 README 只描述当前 `examples/libero/` 目录里真实存在、还能对上的主流程。

如果你想先看仓库整体结构，请回到：

- [../../README.md](../../README.md)

## 这个目录现在负责什么

`examples/libero/` 当前主要负责：

- LIBERO 环境适配
- local env 和 remote env server / client
- residual RL 训练入口
- 独立 residual checkpoint 评估入口
- 训练期 async eval
- residual observation schema 和 typed config

当前真正的主入口有四个：

- [scripts/run_residual_training.py](scripts/run_residual_training.py)
- [scripts/serve_env.py](scripts/serve_env.py)
- [scripts/evaluate_checkpoint.py](scripts/evaluate_checkpoint.py)
- [tools/serve_env.sh](tools/serve_env.sh)

`scripts/process_eval_queue.py` 是 async eval worker，通常不需要手工启动；learner 在 `training.async_eval.enabled=true` 时会自动拉起它。

## 目录结构

- [configs/train_residual.yaml](configs/train_residual.yaml)
  当前 canonical 训练配置
- [configs/eval_residual.yaml](configs/eval_residual.yaml)
  当前 canonical 评估配置
- [config.py](config.py)
  typed config 定义与解析
- [scripts/run_residual_training.py](scripts/run_residual_training.py)
  actor / learner 共用训练入口，通过 `runtime.role=actor|learner` 切角色
- [scripts/serve_env.py](scripts/serve_env.py)
  LIBERO HTTP RPC env server
- [scripts/evaluate_checkpoint.py](scripts/evaluate_checkpoint.py)
  checkpoint eval 入口
- [scripts/process_eval_queue.py](scripts/process_eval_queue.py)
  async eval worker
- [eval_runner.py](eval_runner.py)
  评估主循环
- [async_eval.py](async_eval.py)
  训练期 async eval runtime
- [env/](env/)
  本地 env、remote env、观测解析、LIBERO 路径 bootstrap
- [residual_observation.py](residual_observation.py)
  residual observation 构造和 observation space
- [tools/serve_env.sh](tools/serve_env.sh)
  env server shell wrapper
- `outputs/`
  历史运行产物，不是当前实现说明

## 当前训练链路

当前实现是一条 `base chunk policy + residual DRQ-SAC` 的异步 actor / learner 流程：

1. actor 从 base policy backend 拉一个 `chunk_horizon` 长度的 `base_action_chunk`
2. actor 用当前 observation 构造 residual observation：
   - 图像
   - `robot_proprio`
   - `base_action`
   - `base_action_chunk`
   - `alpha`
3. residual agent 输出 residual action chunk
4. `ResidualActionSpec` 把 base chunk 和 residual chunk 组合成最终动作
5. actor 与环境交互，把 transition 发给 learner
6. learner 从 replay 采样，更新 DRQ-SAC，再通过 agentlace 广播参数给 actor

当前 actor 在每次环境 step 后都会重新计算下一次 decision 的 `base_action_chunk`，所以它是：

- step-wise receding-horizon residual training

相关讨论可以看：

- [docs/chunk_residual_mdp_discussion.md](docs/chunk_residual_mdp_discussion.md)

## 支持的 backend

### policy backend

当前代码支持两种 chunk policy backend：

- `policy.type=openpi`
- `policy.type=joyra`

对应 client 在：

- [../../serl_launcher/serl_launcher/policy/openpi/client.py](../../serl_launcher/serl_launcher/policy/openpi/client.py)
- [../../serl_launcher/serl_launcher/policy/joyra/client.py](../../serl_launcher/serl_launcher/policy/joyra/client.py)

### env backend

当前代码支持两种环境模式：

- `env.backend=local`
  actor 直接在本进程创建 `LiberoTaskEnv`
- `env.backend=remote`
  actor 通过 HTTP RPC 连接 env server

当前默认主线仍然是 `env.backend=remote`。

## 依赖和安装

最常见的环境拆分是：

- `serl_torch`
  actor / learner / eval
- `libero`
  env server
- 你自己的 policy server 环境
  OpenPI 或 JoyRA

最小安装通常至少包括：

```bash
cd /vla/users/niejunnan/codebase/serl_torch
conda activate serl_torch
pip install -r serl_launcher/requirements.txt
pip install -e ./serl_launcher
```

`agentlace` 需要手工安装。

如果你使用 `policy.type=openpi`，还需要：

```bash
pip install -e /path/to/openpi/packages/openpi-client
```

如果不是 editable install，可以补：

```bash
export PYTHONPATH=/vla/users/niejunnan/codebase:/vla/users/niejunnan/codebase/serl_torch/serl_launcher:$PYTHONPATH
```

## LIBERO 路径怎么解析

[env/setup.py](env/setup.py) 会自动处理 LIBERO 相关路径。

### `libero_root`

优先级：

1. 命令行传 `libero_root=...`
2. 仓库内默认候选：
   - `<repo>/third_party/LIBERO`
   - 邻近 repo candidate

要求这个目录必须是完整的 LIBERO checkout，至少包含：

- `libero/libero/bddl_files`
- `libero/libero/init_files`
- `libero/libero/assets/scenes/libero_tabletop_base_style.xml`

### `libero_config_dir`

优先级：

1. 命令行传 `libero_config_dir=...`
2. 环境变量 `LIBERO_CONFIG_PATH`
3. 默认缓存目录：
   - `$XDG_CACHE_HOME/serl_torch/libero_config`
   - 或 `~/.cache/serl_torch/libero_config`

运行时会自动写 `config.yaml` 到这个目录，并设置 `LIBERO_CONFIG_PATH`。

### `libero_datasets_root`

优先级：

1. 命令行传 `libero_datasets_root=...`
2. 环境变量 `LIBERO_DATASETS_ROOT`
3. 仓库邻近的默认候选目录

## 当前默认配置

canonical 训练配置是：

- [configs/train_residual.yaml](configs/train_residual.yaml)

当前默认关键参数：

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
- `training.steps_per_update=30`
- `training.critic_actor_ratio=4`
- `training.max_env_steps=300000`
- `training.max_update_steps=300000`

当前配置解析还有两个显式约束：

- `obs.stack_horizon` 目前必须是 `1`
- `encoder.use_proprio=true` 时，`obs.vector_obs_keys` 不能为空

## 推荐启动顺序

最常见的主线至少会开 4 个终端：

1. LIBERO env server
2. base policy server
3. learner
4. actor

如果开训练期 async eval，再额外起一个独立的 eval env server。

## 1. 启动 LIBERO env server

从 repo root：

```bash
cd /vla/users/niejunnan/codebase/serl_torch
bash examples/libero/tools/serve_env.sh \
  --host 127.0.0.1 \
  --port 30000
```

`tools/serve_env.sh` 会：

- 切到 `examples/libero/`
- 尝试初始化 conda
- 优先使用你显式指定的 Python / env
- 默认尝试激活 `/vla/users/niejunnan/envs/libero`
- 最终执行 `scripts/serve_env.py`

常见可覆盖环境变量：

- `LIBERO_CONDA_ENV`
- `LIBERO_CONDA_PREFIX`
- `LIBERO_PYTHON_BIN`

例如：

```bash
LIBERO_CONDA_ENV=libero bash examples/libero/tools/serve_env.sh --port 30000
```

如果你启用 async eval，还需要单独起一个 eval env server，例如：

```bash
LIBERO_CONDA_ENV=libero bash examples/libero/tools/serve_env.sh --port 30010
```

这个端口必须和 `training.async_eval.env.remote.port` 对齐，并且不能和训练 actor 的 `env.remote.port` 相同。

## 2. 启动 base policy server

这一步当前不由 `examples/libero/` 提供启动脚本。

你需要自己准备一个兼容当前 client 协议的服务，并让它监听：

- `policy.host`
- `policy.port`

默认配置下是：

- `localhost:30001`

切到 JoyRA 时，最常见的是：

```bash
policy.type=joyra policy.port=9001
```

## 3. 启动 learner

```bash
cd /vla/users/niejunnan/codebase/serl_torch/examples/libero
conda run -n serl_torch python scripts/run_residual_training.py \
  runtime.role=learner \
  libero_root=/vla/users/niejunnan/codebase/serl_torch/third_party/LIBERO \
  libero_datasets_root=/vla/users/niejunnan/datasets \
  encoder.resnet.model_name=/vla/users/niejunnan/codebase/serl_torch/pretrained_models/microsoft--resnet-18
```

如果你希望结果写到固定目录，可以加：

```bash
hydra.run.dir=/abs/path/to/run_dir
```

## 4. 启动 actor

```bash
cd /vla/users/niejunnan/codebase/serl_torch/examples/libero
conda run -n serl_torch python scripts/run_residual_training.py \
  runtime.role=actor \
  libero_root=/vla/users/niejunnan/codebase/serl_torch/third_party/LIBERO \
  libero_datasets_root=/vla/users/niejunnan/datasets \
  encoder.resnet.model_name=/vla/users/niejunnan/codebase/serl_torch/pretrained_models/microsoft--resnet-18
```

## 5. actor / learner 必须对齐的配置

至少下面这些字段需要一致：

- `runtime.trainer_host`
- `runtime.trainer_port`
- `runtime.broadcast_port`
- `policy.type`
- `policy.host`
- `policy.port`
- `residual.chunk_horizon`
- `env.action_dim`

如果用 remote env，还需要对齐：

- `env.remote.host`
- `env.remote.port`

## 6. 启用训练期 async eval

当前 async eval 由 learner 自动拉起 worker。

最小要求：

- `training.async_eval.enabled=true`
- 单独起一个 dedicated eval env server
- `training.async_eval.env.backend=remote`
- `training.async_eval.env.remote.host/port` 不得和训练 env 相同

最常见的启动方式：

```bash
cd /vla/users/niejunnan/codebase/serl_torch/examples/libero
conda run -n serl_torch python scripts/run_residual_training.py \
  runtime.role=learner \
  training.async_eval.enabled=true \
  training.async_eval.env.remote.host=127.0.0.1 \
  training.async_eval.env.remote.port=30010
```

async eval 相关产物默认会写在当前 Hydra run dir 下，例如：

- `async_eval_queue.jsonl`
- `async_eval_results.jsonl`
- `async_eval_worker.log`
- `async_eval_checkpoints/`
- `async_eval_runs/`

## 7. 跑 checkpoint eval

评估和训练 actor 一样，仍然依赖两个外部服务先启动好：

- LIBERO env server
- base policy server

当前 canonical eval 配置是：

- [configs/eval_residual.yaml](configs/eval_residual.yaml)

最常见的评估命令：

```bash
cd /vla/users/niejunnan/codebase/serl_torch/examples/libero
conda run -n serl_torch python scripts/evaluate_checkpoint.py \
  eval.checkpoint_path=/abs/path/to/checkpoints \
  eval.episodes=20 \
  eval.deterministic=true \
  libero_root=/vla/users/niejunnan/codebase/serl_torch/third_party/LIBERO \
  libero_datasets_root=/vla/users/niejunnan/datasets \
  encoder.resnet.model_name=/vla/users/niejunnan/codebase/serl_torch/pretrained_models/microsoft--resnet-18
```

`eval.checkpoint_path` 支持两种输入：

- 单个 checkpoint 文件，例如 `checkpoint_25000.pt`
- checkpoint 目录，此时会自动选最新的 `checkpoint_*.pt`

如果你想指定目录里的某个 step，可以再传：

```bash
eval.checkpoint_step=25000
```

如果把 `eval.checkpoint_path=null`，脚本会退化成 base-policy-only eval，并把 residual action 置零。

## 常见 overrides

切 JoyRA：

```bash
policy.type=joyra policy.port=9001
```

改任务：

```bash
task.suite_name=libero_10 task.task_id=0
```

改 remote env 地址：

```bash
env.remote.host=127.0.0.1 env.remote.port=30000
```

改 ResNet 本地路径：

```bash
encoder.resnet.model_name=/abs/path/to/microsoft--resnet-18
```

## 输出目录

默认 Hydra 输出目录来自配置：

- `launch.output_root=outputs/libero`

训练默认写到：

```text
outputs/libero/train_residual/<timestamp>/
```

评估默认写到：

```text
outputs/libero/eval_residual/<suite>_task_<id>/<timestamp>/
```

典型内容包括：

- `summary.json`
- `episode_logs.jsonl`
- `checkpoints/`
- `wandb/`
- async eval 相关 JSONL / worker log / eval run 子目录

## 实现上的当前边界

当前这条主线已经不再依赖很多旧入口。新的 LIBERO 工作最好围绕下面这些文件展开：

- [config.py](config.py)
- [scripts/run_residual_training.py](scripts/run_residual_training.py)
- [scripts/evaluate_checkpoint.py](scripts/evaluate_checkpoint.py)
- [env/](env/)
- [residual_observation.py](residual_observation.py)
- `serl_launcher/serl_launcher/policy/*`
- `serl_launcher/serl_launcher/residual/*`

如果你在旧笔记里看到已经不存在的脚本名，请以当前目录树和这份 README 为准。
