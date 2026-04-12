## LIBERO Example

这份 README 只讲当前 `examples/libero/` 这条主链。

如果你想看仓库整体分层，请回到：

- [README.md](../README.md)

### 这个目录负责什么

`examples/libero/` 现在主要负责：

- LIBERO 环境适配
- remote env server
- residual 训练入口
- residual 评测入口
- offline / online 数据准备
- 实验配置和补充文档

共享训练逻辑已经尽量下沉到：

- `serl_launcher/training/`
- `serl_launcher/agents/`
- `serl_launcher/residual/`

所以这里主要保留：

- 环境层
- example 自身的 residual 语义
- 用户入口
- 具体实验配置

### 目录结构

- `conf/`
  当前主流程的 Hydra 基础配置
- `configs/`
  历史实验 yaml 和实验矩阵
- `env_wrappers/`
  本地 env、remote env、LIBERO setup
- `runtime/`
  观测构造和 policy 输入适配
- `scripts/train/`
  训练入口
- `scripts/eval/`
  评测入口
- `scripts/data/`
  数据准备入口
- `scripts/services/`
  环境服务入口
- `tools/`
  仍保留的服务 / 数据 / 评测 shell 包装
- `docs/`
  LIBERO 补充设计说明

### 路径假设

下面的示例命令都按当前机器的真实路径直接写：

- repo root:
  `/vla/users/niejunnan/codebase/serl_torch`
- LIBERO root:
  `/vla/users/niejunnan/codebase/serl_torch/third_party/LIBERO`
- LIBERO datasets:
  `/vla/users/niejunnan/datasets`
- OpenPI checkpoint:
  `/vla/users/niejunnan/openpi-assets/checkpoints/pi05_libero`

如果你的机器路径不同，直接替换命令里的绝对路径即可。

### 运行环境

训练 / 评测 / 数据准备默认使用 `serl_torch`：

```bash
cd /vla/users/niejunnan/codebase/serl_torch
conda activate serl_torch
pip install -e serl_launcher
pip install -e /vla/users/niejunnan/codebase/openpi/packages/openpi-client
```

常用环境：

- `serl_torch`
  actor / learner / eval / 数据准备
- `libero`
  env server
- `openpi-modified`
  OpenPI 服务

### 当前主流程

现在训练主流程只有一个：

- [run_actor_residual.py](scripts/train/run_actor_residual.py)

这个脚本通过：

- `reference_style.role=learner`
- `reference_style.role=actor`

来分别启动 learner 和 actor。

当前 canonical 配置是：

- [train_reference_style_residual.yaml](conf/train_reference_style_residual.yaml)

旧的这些入口已经删除：

- `scripts/train/run_actor.py`
- `scripts/train/run_learner.py`
- `scripts/train/launch_async_train.py`
- 对应的旧 shell wrapper

### 当前怎么跑

#### 1. 起 env server

```bash
cd /vla/users/niejunnan/codebase/serl_torch
bash examples/libero/tools/serve_env.sh \
  --host 127.0.0.1 \
  --port 30000
```

#### 2. 起 OpenPI

```bash
cd /vla/users/niejunnan/codebase/serl_torch
POLICY_CONFIG=pi05_libero \
POLICY_DIR=/vla/users/niejunnan/openpi-assets/checkpoints/pi05_libero \
bash examples/libero/tools/serve_openpi.sh \
  --port 30001 \
  --gpu-id 0
```

#### 3. 起 learner

```bash
cd /vla/users/niejunnan/codebase/serl_torch/examples/libero
conda run -n serl_torch python scripts/train/run_actor_residual.py \
  reference_style.role=learner \
  hydra.run.dir=/vla/users/niejunnan/codebase/serl_torch/examples/libero/outputs/reference_style/task8/learner \
  libero_root=/vla/users/niejunnan/codebase/serl_torch/third_party/LIBERO \
  libero_datasets_root=/vla/users/niejunnan/datasets
```

#### 4. 起 actor

```bash
cd /vla/users/niejunnan/codebase/serl_torch/examples/libero
conda run -n serl_torch python scripts/train/run_actor_residual.py \
  reference_style.role=actor \
  hydra.run.dir=/vla/users/niejunnan/codebase/serl_torch/examples/libero/outputs/reference_style/task8/actor \
  libero_root=/vla/users/niejunnan/codebase/serl_torch/third_party/LIBERO \
  libero_datasets_root=/vla/users/niejunnan/datasets
```

#### 5. 这几组配置要保持一致

actor 和 learner 必须对齐：

- `reference_style.trainer_port`
- `reference_style.broadcast_port`
- `env.remote.host`
- `env.remote.port`
- `openpi.host`
- `openpi.port`

如果你改了其中一侧，另一侧也要一起改。

### Hydra 配置怎么切

脚本默认吃的是：

- `conf/train_reference_style_residual.yaml`

如果你要换到别的实验 yaml，直接像普通 Hydra 一样传：

```bash
--config-path /abs/path/to/config/dir --config-name your_config_name
```

或者继续用默认配置，再叠加 overrides。

`configs/exp11`、`configs/exp12` 这类目录现在更适合作为：

- 历史实验记录
- 实验矩阵
- 端口 / 资源分配参考

而不是新的 canonical 主流程入口。

### 输出和日志怎么看

最常看的产物：

- learner run dir 下的：
  - `summary.json`
  - `checkpoints/`
  - `wandb/`
- actor run dir 下的：
  - `summary.json`

如果你希望 actor / learner 落在同一个实验根目录，最简单的方式就是手动指定：

- `.../your_run/learner`
- `.../your_run/actor`

### 评测怎么跑

评测入口仍然是：

- [evaluate_checkpoint.py](scripts/eval/evaluate_checkpoint.py)

最小示例：

```bash
cd /vla/users/niejunnan/codebase/serl_torch
conda run -n serl_torch python examples/libero/scripts/eval/evaluate_checkpoint.py \
  policy.type=openpi \
  policy.id=pi05_libero \
  env.remote.host=127.0.0.1 \
  env.remote.port=30000 \
  openpi.host=127.0.0.1 \
  openpi.port=30001 \
  eval.checkpoint_path=/abs/path/to/checkpoint.pt \
  libero_root=/vla/users/niejunnan/codebase/serl_torch/third_party/LIBERO \
  libero_datasets_root=/vla/users/niejunnan/datasets
```

### 数据准备入口

旧的离线转换、online prefill 收集和 HDF5 统计脚本已经移除。

当前这个目录只保留在线训练、评测和 env service 主流程入口。

### LIBERO runtime config 是怎么处理的

当前 wrapper 会在运行时自动：

1. 解析 `libero_root`
2. 解析 `libero_datasets_root`
3. 生成给上游 LIBERO 使用的 `config.yaml`
4. 设置 `LIBERO_CONFIG_PATH`

默认生成目录不会写回 repo，而是写到：

- `$XDG_CACHE_HOME/serl_torch/libero_config`
- 如果没有设置 `XDG_CACHE_HOME`，则回落到 `~/.cache/serl_torch/libero_config`

所以通常不需要手工准备 `config.yaml`。

### 常见问题

#### 1. 找不到 LIBERO config

通常看这两个路径：

- `$XDG_CACHE_HOME/serl_torch/libero_config`
- `~/.cache/serl_torch/libero_config`

#### 2. 换了 datasets 路径怎么办

不用手改 `config.yaml`。

直接把命令里的：

- `libero_datasets_root=/vla/users/niejunnan/datasets`

替换成你的新路径即可。

#### 3. learner OOM

常见原因：

- 同一张卡上残留旧的 OpenPI 服务
- env / OpenPI / learner / actor 全都堆在同一张卡上

建议训练前先清理旧服务，再起新一轮。

#### 4. agentlace 偶发 timeout warning

如果看到：

- `Failed to send message ... potential timeout`

通常是 actor 到 learner 的控制面 RPC 偶发超时，不一定表示训练挂掉。先看：

- `env_steps`
- `update_steps`
- episode 是否还在继续增长

### 当前验证状态

当前这条 LIBERO 主链已经至少验证过：

- offline data preparation
- online prefill collection
- async train smoke
- checkpoint eval smoke
