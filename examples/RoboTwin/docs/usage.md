# RoboTwin 残差 SAC 使用说明

本文档说明如何在 Docker 容器 `robotwin_njn_new` 中启动训练与评估。

---

## 快速开始

| 操作 | 前置服务 | 命令 |
|------|----------|------|
| **训练** | OpenPI (9000) + 环境服务 (9200) | `bash tools/train.sh` |
| **评估** | OpenPI (9000) | `bash tools/eval.sh eval.checkpoint_path=/path/to/checkpoint.pt` |
| **启动 OpenPI** | - | `bash tools/serve_openpi.sh <任务名> [--port 9000]` |
| **启动环境服务** | - | `bash tools/serve_env.sh`（需 `robotwin2` 环境） |

所有命令均在 `examples/RoboTwin` 目录执行。训练/评估用 `serl_torch` 环境；环境服务用 `robotwin2` 环境。

---

## 一、环境概览

### 1.1 Docker 与 Conda 环境

运行环境为 Docker 容器 `robotwin_njn_new`，内置以下 conda 环境：

| 环境名 | 用途 |
|--------|------|
| `base` | 默认环境 |
| `openpi` | OpenPI（VLA 基策略）服务端 |
| `robotwin2` | RoboTwin 仿真环境 RPC 服务端 |
| `serl_torch` | 残差 SAC 训练与评估 |

### 1.2 服务与端口

| 服务 | 默认端口 | 说明 |
|------|----------|------|
| **OpenPI** | 9000（首个任务） | VLA 基策略，由 `batch_serve_policy.sh` 启动 |
| **RoboTwin 环境服务** | 9100 或 9200 | 仿真环境 RPC，由 `robotwin_env_server.py` 启动 |

### 1.3 关键路径

- RoboTwin 根目录：`/vla/users/niejunnan/codebase/RoboTwin`
- serl_torch 示例根目录：`/vla/users/niejunnan/codebase/serl_torch/examples/RoboTwin`
- OpenPI 脚本：`/vla/users/niejunnan/test/openpi/tools/batch_serve_policy.sh`

---

## 二、启动训练

### 2.1 前置条件

训练需要两个服务同时运行：

1. **OpenPI 服务**（提供基策略 base action chunk）
2. **RoboTwin 环境服务**（提供仿真环境）

### 2.2 步骤一：进入 Docker 容器

```bash
docker exec -it robotwin_njn_new /bin/bash
```

### 2.3 步骤二：启动 OpenPI 服务

在**第一个终端**中，任选其一：

**方式 A：使用 RoboTwin 自带脚本（推荐，单任务）**

```bash
cd /vla/users/niejunnan/codebase/serl_torch/examples/RoboTwin
bash tools/serve_openpi.sh place_a2b_left              # 默认端口 9000
bash tools/serve_openpi.sh place_a2b_left --port 9000 # 指定端口
bash tools/serve_openpi.sh adjust_bottle --port 9001 --gpu_id 1
```

**方式 B：使用 OpenPI 批量脚本（多任务）**

```bash
conda activate openpi
bash /vla/users/niejunnan/test/openpi/tools/batch_serve_policy.sh place_a2b_left
# 或指定端口：bash .../batch_serve_policy.sh --port 9000 9001 place_a2b_left adjust_bottle
```

多任务时端口依次为 9000、9001、9002...，需在训练配置中指定对应端口。

### 2.4 步骤三：启动 RoboTwin 环境服务

在**第二个终端**中：

```bash
docker exec -it robotwin_njn_new /bin/bash
conda activate robotwin2

cd /vla/users/niejunnan/codebase/serl_torch/examples/RoboTwin
bash tools/serve_env.sh                    # 默认端口 9200
# 或指定端口：bash tools/serve_env.sh --port 9100
```

> 注意：`train.sh` 默认连接端口 **9200**，若使用其他端口，需在启动训练时覆盖：`env.remote.port=9100`

### 2.5 步骤四：启动训练

在**第三个终端**中：

```bash
docker exec -it robotwin_njn_new /bin/bash
conda activate serl_torch

cd /vla/users/niejunnan/codebase/serl_torch/examples/RoboTwin
bash tools/train.sh
```

### 2.6 训练脚本参数覆盖

`train.sh` 支持通过 Hydra 覆盖配置，例如：

```bash
# 更换任务
bash tools/train.sh task.name=adjust_bottle

# 覆盖多个参数
bash tools/train.sh seed=42 training.max_online_env_steps=100000

# 若环境服务使用 9100 端口
bash tools/train.sh env.remote.port=9100
```

### 2.7 训练输出

- 日志与 checkpoint 目录：`outputs/train_residual_sac/<日期>/<时间>/`
- Checkpoint 文件：`checkpoints/checkpoint_<step>.pt`

---

## 三、启动评估

### 3.1 前置条件

评估使用**本地环境**（不依赖 RoboTwin 环境服务），只需：

1. **OpenPI 服务**（与训练相同方式启动）

### 3.2 一键评估

`eval.sh` 已配置默认 OpenPI 连接（localhost:9000）与 RoboTwin 根目录，只需指定 checkpoint 即可运行：

```bash
# 1. 进入容器并激活环境
docker exec -it robotwin_njn_new /bin/bash
conda activate serl_torch
cd /vla/users/niejunnan/codebase/serl_torch/examples/RoboTwin

# 2. 确保 OpenPI 已启动（另一终端运行 batch_serve_policy.sh）

# 3. 一键评估（checkpoint 路径支持相对路径，相对于 examples/RoboTwin）
bash tools/eval.sh eval.checkpoint_path=outputs/train_residual_sac/2026-03-06/08-17-17/checkpoints/checkpoint_500.pt
```

### 3.3 评估参数覆盖

```bash
# 评估更多 episode
bash tools/eval.sh eval.checkpoint_path=/path/to/checkpoint.pt eval.episodes=50

# 更换任务（需与 OpenPI 启动的任务一致，端口对应）
bash tools/eval.sh eval.checkpoint_path=/path/to/checkpoint.pt task.name=place_a2b_left openpi.port=9000

# 仅评估 base policy（残差全零，不加载 checkpoint）
bash tools/eval.sh eval.checkpoint_path=null
```

### 3.4 评估输出

- 输出目录：`outputs/eval_residual_fast/<日期>/<时间>/`
- 汇总结果：`summary.json`

---

## 四、端口与任务对应关系

使用 `batch_serve_policy.sh` 启动多任务时：

| 任务顺序 | 端口 |
|----------|------|
| 第 1 个任务 | 9000 |
| 第 2 个任务 | 9001 |
| 第 3 个任务 | 9002 |
| ... | ... |

训练/评估配置中的 `openpi.port` 需与对应任务的端口一致。单任务时默认 9000。

---

## 五、常见问题

### Q: 训练时报错连接环境服务失败

确认 `robotwin_env_server.py` 已启动，且端口与 `train.sh` 中 `env.remote.port` 一致（默认 9200）。

### Q: 训练/评估时报错连接 OpenPI 失败

确认 `batch_serve_policy.sh` 已启动，且 `openpi.port` 与任务端口一致。

### Q: 评估时任务名与训练不一致

在 `eval.sh` 中通过 `task.name=xxx` 覆盖，并确保 OpenPI 已启动该任务。

### Q: 如何查看训练曲线

```bash
tensorboard --logdir outputs/train_residual_sac
```
