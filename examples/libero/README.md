# LIBERO Example

这份 README 只描述当前 `examples/libero/` 目录里真实存在、还能对上的主流程。

如果你想先看仓库整体结构，请回到：

- [../../README.md](../../README.md)

## 这个目录现在负责什么

`examples/libero/` 当前主要负责：

- LIBERO 环境适配
- local env 和 remote env server / client
- residual offline data 准备与加载
- residual RL 训练入口
- 独立 residual checkpoint 评估入口
- 训练期 eval
- residual observation schema 和 typed config

当前最常用的入口有六个：

- [scripts/run_residual_training_1_baseline.py](scripts/run_residual_training_1_baseline.py)
- [scripts/run_residual_training_2_chunk_local.py](scripts/run_residual_training_2_chunk_local.py)
- [scripts/run_residual_offline_prepare.py](scripts/run_residual_offline_prepare.py)
- [scripts/serve_env.py](scripts/serve_env.py)
- [scripts/run_residual_eval.py](scripts/run_residual_eval.py)
- [tools/serve_env.sh](tools/serve_env.sh)

`runtime/async_eval_worker.py` 是训练期 eval worker，通常不需要手工启动；learner 在 `training.async_eval.enabled=true` 时会自动拉起它。

## 训练脚本演化顺序

`run_residual_training_1/2/3/4/5_*.py` 这组名字里的数字表示的是演化顺序，不表示“当前推荐程度”。

- [scripts/run_residual_training_1_baseline.py](scripts/run_residual_training_1_baseline.py)
  reference baseline；逐 step rollout，最适合拿来对照原始语义
- [scripts/run_residual_training_2_chunk_local.py](scripts/run_residual_training_2_chunk_local.py)
  当前最稳的优化线；chunk execute，但 transition 仍在 actor 本地后处理
- [scripts/run_residual_training_3_split_proto.py](scripts/run_residual_training_3_split_proto.py)
  第一代 split 原型；开始拆 `actor / processor / learner`
- [scripts/run_residual_training_4_split_refined.py](scripts/run_residual_training_4_split_refined.py)
  split 架构的过渡整理版；比 proto 更统一，但还不是最终 runtime 形态
- [scripts/run_residual_training_5_split_pipeline.py](scripts/run_residual_training_5_split_pipeline.py)
  最新的 split / pipeline 演化版本；角色边界最清楚，但仍然更偏架构演进线，而不是默认起跑线

如果你只是想跑当前最稳定的训练入口，优先看：

- baseline 对照：`1_baseline`
- 日常优化实验：`2_chunk_local`

如果你是在看 split 架构如何一步步演进，再按 `3 -> 4 -> 5` 去读。

这里有个刻意保留的边界：

- `scripts/` 放可直接 `python` 运行的厚入口
- `env/` 放 LIBERO-specific adapter，包括和 LIBERO task / dataset / observation 强绑定的 offline helper
- `runtime/` 只放训练 / 评估运行期编排相关 helper，不放 prepared dataset 规则

## 目录结构

- [configs/train_residual.yaml](configs/train_residual.yaml)
  当前 canonical 训练配置
- [configs/eval_residual.yaml](configs/eval_residual.yaml)
  当前 canonical 评估配置
- [config.py](config.py)
  typed config 定义与解析
- [scripts/run_residual_training_1_baseline.py](scripts/run_residual_training_1_baseline.py)
  reference baseline 训练入口；actor / learner 共用，通过 `runtime.role=actor|learner` 切角色
- [scripts/run_residual_training_2_chunk_local.py](scripts/run_residual_training_2_chunk_local.py)
  当前最稳的优化训练入口；actor 支持 `env.step_chunk(...)`、post-hoc assembly、async backfill 和 `async_commit` transport
- [scripts/run_residual_training_3_split_proto.py](scripts/run_residual_training_3_split_proto.py)
  split 架构第一代原型；显式引入 `actor / processor / learner`
- [scripts/run_residual_training_4_split_refined.py](scripts/run_residual_training_4_split_refined.py)
  split 架构过渡整理版；协议和 session 使用更统一
- [scripts/run_residual_training_5_split_pipeline.py](scripts/run_residual_training_5_split_pipeline.py)
  当前最新的 split / pipeline 版本；更多协议与 transport 细节已经下沉到 `runtime/`
- [scripts/run_residual_offline_prepare.py](scripts/run_residual_offline_prepare.py)
  离线数据准备入口，默认也读取 `configs/train_residual.yaml`
- [scripts/serve_env.py](scripts/serve_env.py)
  LIBERO HTTP RPC env server
- [env/offline_data.py](env/offline_data.py)
  LIBERO-specific offline helper；包含 task/dataset resolve、demo 转换和 prepared replay 加载入口
- [scripts/run_residual_eval.py](scripts/run_residual_eval.py)
  checkpoint eval 入口
- [runtime/async_eval_worker.py](runtime/async_eval_worker.py)
  训练期 eval worker；由 learner 自动拉起
- [runtime/async_eval_runtime.py](runtime/async_eval_runtime.py)
  训练期 async eval runtime glue
- [runtime/transition_assembly.py](runtime/transition_assembly.py)
  optimized actor 的 post-hoc chunk transition assembly
- [env/](env/)
  本地 env、remote env、观测解析、LIBERO 路径 bootstrap
- [runtime/](runtime/)
  example-local runtime support；放训练/评估编排 helper 和内部 worker entrypoint
- [../../serl_launcher/serl_launcher/residual/observation.py](../../serl_launcher/serl_launcher/residual/observation.py)
  公共 residual observation schema
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

### `run_residual_training_2_chunk_local.py` 这条优化线是什么

`run_residual_training_2_chunk_local.py` 和 canonical 的
[scripts/run_residual_training_1_baseline.py](scripts/run_residual_training_1_baseline.py) 最大的区别在 actor 数据流：

- canonical 版本：
  `step -> 立刻补 next_residual_obs -> 立刻写 step transition`
- optimized 版本：
  `step_chunk -> 收集 raw chunk -> post-hoc 组装 step transition -> 写 replay`

所以 optimized 版本里：

- env 执行单元是 chunk
- replay 存储单元仍然是 step transition
- learner 仍然从 step replay 采样 chunk window

也就是说，optimized 版本不是 direct chunk replay；它只是把 transition 的组装时机，从 inline step-wise assembly 改成了 post-hoc chunk assembly。

当前 optimized 版本支持两种模式：

- 默认同步模式：
  不传 `backfill_policy.*`，actor 在 `step_chunk(...)` 后同步完成整段 assembly
- async backfill 模式：
  传 `++backfill_policy.enabled=true` 后，actor 只负责控制推进，chunk 内整段 backfill 和 replay assembly 由后台线程完成；如果再配一个 dedicated backfill policy server，可以把这条后处理路径从主 decision policy 服务上分流出去

```text
chunk n 最后一步 next_observations
==
chunk n+1 第一步 observations
```

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
cd /home/hello/codebase/serl_torch
conda activate serl_torch
pip install -r serl_launcher/requirements.txt
pip install -e ./serl_launcher
pip install -e .
```

`agentlace` 默认会随 `serl_launcher/requirements.txt` / `serl_launcher/setup.py` 从 `niejunnan25/agentlace` 的 `885f5fc` 安装。

这里的 `pip install -e .` 是 repo 根的 editable install。它负责让：

- `import serl_torch.examples.libero...`
- `import serl_torch.examples.agibot_real...`

这类绝对导入稳定成立。做完这一步之后，直接运行：

- `python examples/libero/scripts/run_residual_training_1_baseline.py`
- `python examples/libero/scripts/run_residual_training_2_chunk_local.py`
- `python examples/libero/scripts/run_residual_training_3_split_proto.py`

通常不需要再额外补仓库级 `PYTHONPATH`。

如果你使用 `policy.type=openpi`，还需要：

```bash
pip install -e ./third_party/openpi-client
```

这只安装 vendored 的 client 包。  
如果你还要启动 OpenPI policy server，仍然需要完整的 OpenPI 仓库，并设置 `OPENPI_ROOT`。

如果根 repo 和 `serl_launcher/` 这两个 editable install 都没做，可以补：

```bash
export PYTHONPATH=/home/hello/codebase:/home/hello/codebase/serl_torch/serl_launcher:$PYTHONPATH
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

当前 `scripts/run_residual_training_1_baseline.py` 和 `scripts/run_residual_offline_prepare.py` 都默认读取这份配置；也就是说，prepare / train 共用同一套 `task`、`policy`、`obs`、`residual` 默认值。

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
- `residual.alpha=0.1`
- `residual.chunk_horizon=5`
- `offline.prepared_path=data/residual/offline_data/libero_10_task_8/openpi_chunk5_alpha0p1`
- `offline.prepare.output_root=data/residual/offline_data`
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
3. 可选：offline data prepare
4. learner
5. actor

如果开训练期 async eval，再额外起一个独立的 eval env server。

## 1. 启动 LIBERO env server

从 repo root：

```bash
cd /home/hello/codebase/serl_torch
bash examples/libero/tools/serve_env.sh \
  --host 127.0.0.1 \
  --port 30000
```

`tools/serve_env.sh` 会：

- 切到 `examples/libero/`
- 尝试初始化 conda
- 优先使用你显式指定的 Python / env
- 默认尝试激活 `/vla/users/niejunnan/envs/libero`
- 最终执行 LIBERO env server 入口

如果你更偏好手动激活 conda 环境并直接用 Python 启动，更推荐从 repo root 运行：

```bash
conda activate libero
cd /path/to/serl_torch
python examples/libero/scripts/serve_env.py --host 127.0.0.1 --port 30000
```

这样只依赖 repo root，不需要再 `cd examples/libero`。如果你不想依赖当前工作目录，也可以直接运行脚本绝对路径：

```bash
conda activate libero
python /path/to/serl_torch/examples/libero/scripts/serve_env.py --host 127.0.0.1 --port 30000
```

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

如果你使用 `policy.type=openpi`，当前目录已经提供了两个常用 wrapper：

- [tools/serve_openpi_policy.sh](tools/serve_openpi_policy.sh)
  默认启动 `pi0_libero`
- [tools/serve_openpi_10000_policy.sh](tools/serve_openpi_10000_policy.sh)
  默认启动 `pi0_10000`

最常见的启动方式：

```bash
cd /home/hello/codebase/serl_torch
bash examples/libero/tools/serve_openpi_policy.sh \
  --gpu-id 0 \
  --port 30001
```

如果你更想用 `pi0_10000`，可以换成：

```bash
cd /home/hello/codebase/serl_torch
bash examples/libero/tools/serve_openpi_10000_policy.sh \
  --gpu-id 0 \
  --port 30001
```

当然，你也可以自己准备一个兼容当前 client 协议的服务，并让它监听：

- `policy.host`
- `policy.port`

默认配置下是：

- `localhost:30001`

切到 JoyRA 时，最常见的是：

```bash
policy.type=joyra policy.port=9001
```

## 3. 准备 offline data（可选）

如果你想使用 residual offline data，当前 canonical 入口是：

- [scripts/run_residual_offline_prepare.py](scripts/run_residual_offline_prepare.py)

它默认也读取：

- [configs/train_residual.yaml](configs/train_residual.yaml)

也就是说，如果你不额外 override，prepare 会直接使用当前训练默认配置里的：

- `task.suite_name`
- `task.task_id`
- `policy.host`
- `policy.port`
- `residual.alpha`
- `residual.chunk_horizon`
- `offline.prepare.output_root`

最常见的准备方式：

```bash
cd /home/hello/codebase/serl_torch
conda run -n serl_torch python examples/libero/scripts/run_residual_offline_prepare.py \
  offline.enabled=true \
  libero_root=/home/hello/codebase/serl_torch/third_party/LIBERO \
  libero_datasets_root=/vla/users/niejunnan/datasets
```

这一步依赖：

- LIBERO demo 数据可被 `libero_datasets_root` 找到
- base policy server 已经在 `policy.host:policy.port` 上启动

默认配置下，prepared 数据会生成到：

```text
data/residual/offline_data/libero_10_task_8/openpi_chunk5_alpha0p1
```

prepare 完成后，脚本会在终端打印下一步 learner 命令。

## 4. 启动 learner

```bash
cd /home/hello/codebase/serl_torch
conda run -n serl_torch python examples/libero/scripts/run_residual_training_1_baseline.py \
  runtime.role=learner \
  libero_root=/home/hello/codebase/serl_torch/third_party/LIBERO \
  libero_datasets_root=/vla/users/niejunnan/datasets \
  encoder.resnet.model_name=/home/hello/codebase/serl_torch/pretrained_models/microsoft--resnet-18
```

如果你要加载 prepared offline data，最常见的 learner 命令是：

```bash
cd /home/hello/codebase/serl_torch
conda run -n serl_torch python examples/libero/scripts/run_residual_training_1_baseline.py \
  runtime.role=learner \
  offline.enabled=true \
  offline.prepared_path=data/residual/offline_data/libero_10_task_8/openpi_chunk5_alpha0p1 \
  libero_root=/home/hello/codebase/serl_torch/third_party/LIBERO \
  libero_datasets_root=/vla/users/niejunnan/datasets \
  encoder.resnet.model_name=/home/hello/codebase/serl_torch/pretrained_models/microsoft--resnet-18
```

如果你希望结果写到固定目录，可以加：

```bash
hydra.run.dir=/abs/path/to/run_dir
```

## 5. 启动 actor

```bash
cd /home/hello/codebase/serl_torch
conda run -n serl_torch python examples/libero/scripts/run_residual_training_1_baseline.py \
  runtime.role=actor \
  libero_root=/home/hello/codebase/serl_torch/third_party/LIBERO \
  libero_datasets_root=/vla/users/niejunnan/datasets \
  encoder.resnet.model_name=/home/hello/codebase/serl_torch/pretrained_models/microsoft--resnet-18
```

### 5.1 启动 optimized 实验线

当前推荐把 [scripts/run_residual_training_2_chunk_local.py](scripts/run_residual_training_2_chunk_local.py) 作为日常优化线；默认 reference baseline 仍然是 [scripts/run_residual_training_1_baseline.py](scripts/run_residual_training_1_baseline.py)。

`3_split_proto / 4_split_refined / 5_split_pipeline` 属于 split 架构演进链。它们不是按“推荐程度”排出来的，而是按演化顺序排出来的；如果你只是要稳定跑实验，不必从 `5` 开始。

`optimized` 默认读取 [configs/train_residual_optimized.yaml](configs/train_residual_optimized.yaml)，其中：

- `runtime.trainer_transport.mode=async_commit`
- `runtime.trainer_transport.data_port=5690`
- `wait_committed_on_episode_end=false`
- `wait_committed_on_shutdown=true`

最常见的仍然是两种用法：

- 同步 optimized：
  使用 `env.step_chunk(...)`，但 actor 仍在本进程同步完成 post-hoc assembly
- async dedicated backfill：
  actor 主线程只负责控制推进，后台线程负责 assembly，并额外起一个 second policy server 专门服务 backfill

#### 最小同步 optimized 启动方式

learner：

```bash
cd /home/hello/codebase/serl_torch
conda run -n serl_torch python examples/libero/scripts/run_residual_training_2_chunk_local.py \
  runtime.role=learner \
  task.suite_name=libero_spatial \
  task.task_id=4 \
  policy.port=30101 \
  env.remote.port=30100 \
  training.async_eval.enabled=false \
  libero_root=/home/hello/codebase/serl_torch/third_party/LIBERO \
  libero_datasets_root=/vla/users/niejunnan/datasets
```

actor：

```bash
cd /home/hello/codebase/serl_torch
conda run -n serl_torch python examples/libero/scripts/run_residual_training_2_chunk_local.py \
  runtime.role=actor \
  task.suite_name=libero_spatial \
  task.task_id=4 \
  policy.port=30101 \
  env.remote.port=30100 \
  training.async_eval.enabled=false \
  libero_root=/home/hello/codebase/serl_torch/third_party/LIBERO \
  libero_datasets_root=/vla/users/niejunnan/datasets
```

这时 optimized actor 的语义是：

```text
chunk execute -> synchronous post-hoc step transition assembly -> step-window replay
```

#### async dedicated backfill 启动方式

除了主 decision policy server 外，再额外起一个 backfill policy server。最简单的做法是让两个 server 使用同一个 checkpoint，只是监听不同端口。

主 decision policy server：

```bash
cd /home/hello/codebase/serl_torch
bash examples/libero/tools/serve_openpi_10000_policy.sh \
  --gpu-id 6 \
  --port 30101
```

backfill policy server：

```bash
cd /home/hello/codebase/serl_torch
bash examples/libero/tools/serve_openpi_10000_policy.sh \
  --gpu-id 7 \
  --port 30102
```

learner 仍然不需要额外改动，继续只连主训练端口：

```bash
cd /home/hello/codebase/serl_torch
conda run -n serl_torch python examples/libero/scripts/run_residual_training_2_chunk_local.py \
  runtime.role=learner \
  task.suite_name=libero_spatial \
  task.task_id=4 \
  policy.port=30101 \
  env.remote.port=30100 \
  training.async_eval.enabled=false \
  libero_root=/home/hello/codebase/serl_torch/third_party/LIBERO \
  libero_datasets_root=/vla/users/niejunnan/datasets
```

actor 额外打开 `backfill_policy`：

```bash
cd /home/hello/codebase/serl_torch
conda run -n serl_torch python examples/libero/scripts/run_residual_training_2_chunk_local.py \
  runtime.role=actor \
  task.suite_name=libero_spatial \
  task.task_id=4 \
  policy.port=30101 \
  env.remote.port=30100 \
  training.async_eval.enabled=false \
  libero_root=/home/hello/codebase/serl_torch/third_party/LIBERO \
  libero_datasets_root=/vla/users/niejunnan/datasets \
  ++backfill_policy.enabled=true \
  ++backfill_policy.host=127.0.0.1 \
  ++backfill_policy.port=30102 \
  ++backfill_policy.max_pending_chunks=2
```

这时 optimized actor 的语义是：

```text
chunk execute
-> async full-chunk backfill
-> ordered step transition commit
-> step-window replay
```

使用这条模式时有两个实践约束：

- `policy.port` 和 `backfill_policy.port` 最好指向两个不同的服务，否则后台 assembly 会和主 decision 路径抢同一个推理服务
- `backfill_policy` 服务最好和主 decision 服务使用同一份 checkpoint；否则 replay 里的 residual observation 分布会和 actor 实际控制分布偏离

## 6. actor / learner 必须对齐的配置

至少下面这些字段需要一致：

- `runtime.trainer_host`
- `runtime.trainer_port`
- `runtime.broadcast_port`
- `runtime.trainer_transport.mode`
- `runtime.trainer_transport.data_port`
- `policy.type`
- `policy.host`
- `policy.port`
- `residual.chunk_horizon`
- `env.action_dim`

如果用 remote env，还需要对齐：

- `env.remote.host`
- `env.remote.port`

## 7. 启用训练期 eval

当前训练期 eval 由 learner 自动拉起 worker，内部仍然通过 async worker 实现。

已知的 episode 触发语义风险说明见：

- [docs/async_eval_episode_trigger_risk.md](docs/async_eval_episode_trigger_risk.md)

最小要求：

- `training.async_eval.enabled=true`
- 单独起一个 dedicated eval env server
- `training.async_eval.env.backend=remote`
- `training.async_eval.env.remote.host/port` 不得和训练 env 相同

最常见的启动方式：

```bash
cd /home/hello/codebase/serl_torch
conda run -n serl_torch python examples/libero/scripts/run_residual_training_1_baseline.py \
  runtime.role=learner \
  training.async_eval.enabled=true \
  training.async_eval.env.remote.host=127.0.0.1 \
  training.async_eval.env.remote.port=30010
```

训练期 eval 相关产物默认会写在当前 Hydra run dir 下，例如：

- `async_eval_queue.jsonl`
- `async_eval_results.jsonl`
- `async_eval_worker.log`
- `async_eval_checkpoints/`
- `async_eval_runs/`

### `libero_spatial task4` 完整启动示例

如果你要直接运行：

- `configs/train_residual_libero_spatial_task4.yaml`

可以按下面这 5 个终端分别启动。这个例子使用：

- `train env`: `127.0.0.1:30100`
- `policy`: `127.0.0.1:30101`
- `eval env`: `127.0.0.1:30110`
- `learner`: `GPU 5`
- `actor + policy`: `GPU 6`

1. `pi0_10000` policy server

```bash
source /vla/miniconda3/etc/profile.d/conda.sh
conda activate openpi-modified
export CUDA_VISIBLE_DEVICES=6
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.4
export PYTHONPATH=/vla/users/niejunnan/codebase/openpi/src:${PYTHONPATH}
cd /vla/users/niejunnan/codebase/openpi
uv run scripts/serve_policy.py \
  --port 30101 \
  policy:checkpoint \
  --policy.config=pi0_libero_baseline_10_bs32_150000 \
  --policy.dir=/vla/users/niejunnan/assets/openpi-assets/serl_torch_ckpt/pi0_10000
```

2. 训练 env server

```bash
source /vla/miniconda3/etc/profile.d/conda.sh
conda activate /vla/users/niejunnan/envs/libero
cd /home/hello/codebase/serl_torch
python examples/libero/scripts/serve_env.py --host 127.0.0.1 --port 30100
```

3. eval env server

```bash
source /vla/miniconda3/etc/profile.d/conda.sh
conda activate /vla/users/niejunnan/envs/libero
cd /home/hello/codebase/serl_torch
python examples/libero/scripts/serve_env.py --host 127.0.0.1 --port 30110
```

4. learner

```bash
source /vla/miniconda3/etc/profile.d/conda.sh
conda activate serl_torch
export CUDA_VISIBLE_DEVICES=5
cd /home/hello/codebase/serl_torch
python examples/libero/scripts/run_residual_training_1_baseline.py \
  --config-name train_residual_libero_spatial_task4 \
  runtime.role=learner \
  policy.port=30101 \
  env.remote.port=30100 \
  training.async_eval.env.remote.port=30110 \
  libero_root=/home/hello/codebase/serl_torch/third_party/LIBERO \
  libero_datasets_root=/vla/users/niejunnan/datasets
```

5. actor

```bash
source /vla/miniconda3/etc/profile.d/conda.sh
conda activate serl_torch
export CUDA_VISIBLE_DEVICES=6
cd /home/hello/codebase/serl_torch
python examples/libero/scripts/run_residual_training_1_baseline.py \
  --config-name train_residual_libero_spatial_task4 \
  runtime.role=actor \
  policy.port=30101 \
  env.remote.port=30100 \
  training.async_eval.env.remote.port=30110 \
  libero_root=/home/hello/codebase/serl_torch/third_party/LIBERO \
  libero_datasets_root=/vla/users/niejunnan/datasets
```

## 8. 跑 checkpoint eval

评估和训练 actor 一样，仍然依赖两个外部服务先启动好：

- LIBERO env server
- base policy server

当前 canonical eval 配置是：

- [configs/eval_residual.yaml](configs/eval_residual.yaml)

最常见的评估命令：

```bash
cd /home/hello/codebase/serl_torch
conda run -n serl_torch python examples/libero/scripts/run_residual_eval.py \
  eval.checkpoint_path=/abs/path/to/checkpoints \
  eval.episodes=20 \
  eval.deterministic=true \
  libero_root=/home/hello/codebase/serl_torch/third_party/LIBERO \
  libero_datasets_root=/vla/users/niejunnan/datasets \
  encoder.resnet.model_name=/home/hello/codebase/serl_torch/pretrained_models/microsoft--resnet-18
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
- [scripts/run_residual_training_1_baseline.py](scripts/run_residual_training_1_baseline.py)
- [scripts/run_residual_eval.py](scripts/run_residual_eval.py)
- [env/](env/)
- [runtime/](runtime/)
- [../../serl_launcher/serl_launcher/residual/observation.py](../../serl_launcher/serl_launcher/residual/observation.py)
- `serl_launcher/serl_launcher/policy/*`
- `serl_launcher/serl_launcher/residual/*`

其中：

- `env/offline_data.py` 继续保留在 `env/`，因为它承载的是 LIBERO-specific offline 规则
- `runtime/` 现在只放 `async eval` 和 `optimized actor transition assembly` 这类运行编排模块

如果你在旧笔记里看到已经不存在的脚本名，请以当前目录树和这份 README 为准。
