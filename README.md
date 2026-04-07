# serl_torch

`serl_torch` 是当前这套 residual RL / VLA 实验代码仓库，核心围绕 PyTorch 版训练组件、LIBERO 场景、OpenPI 基座策略服务，以及一批已经整理好的实验配置与启动脚本。

这份 README 以“当前仓库怎么用”为目标，不追求论文式完整背景，更偏向一个能直接上手的入口文档。你如果现在主要在跑 LIBERO，建议先看下面的 `examples/libero` 部分；如果你正在复现 `exp10`，可以直接跳到本文的 `exp10 示例`。

## 仓库包含什么

- `serl_launcher/`: 通用的 PyTorch RL 组件与网络、replay buffer、agent 实现
- `examples/libero/`: LIBERO residual RL 主工作区，包含训练、评测、异步 actor/learner、离线数据转换、实验配置
- `examples/RoboTwin/`: RoboTwin 版本的 residual RL 管线
- `pretrained_models/`: 本地 ResNet 权重目录
- `tools/`: 仓库级辅助脚本，例如下载 ResNet
- `docs/`: 实验分析、异步评测与实现说明
- `third_party/`: 本地依赖或参考仓库

## 当前仓库的本地假设

这个仓库目前明显带有本地工作区约定，第一次使用前最好先知道这几点：

- 很多脚本和 YAML 默认使用 `/vla/users/niejunnan/...` 路径
- LIBERO 训练相关脚本默认会尝试激活 `serl_torch` 环境
- LIBERO 环境服务默认会尝试激活 `LIBERO_CONDA_ENV`，或者 `/vla/users/niejunnan/envs/libero`
- OpenPI 服务脚本最终会切到 `openpi-modified` 环境
- OpenPI 环境里需要能直接使用 `uv`
- OpenPI 仓库默认位置是 `/vla/users/niejunnan/codebase/openpi`
- OpenPI checkpoint 默认位置是 `/vla/users/niejunnan/openpi-assets/checkpoints/`
- LIBERO dataset 会优先从 `LIBERO_DATASETS_ROOT`、`../datasets`、`../../datasets` 等位置自动找

如果你的机器路径布局不同，通常有两种改法：

1. 通过环境变量覆盖，例如 `SERL_CONDA_ENV`、`LIBERO_CONDA_ENV`、`OPENPI_ROOT`、`LIBERO_DATASETS_ROOT`
2. 直接在 YAML 里覆写路径字段，例如 `openpi_root=...`、`libero_datasets_root=...`

## 建议的最小安装

这不是一个“单环境就能覆盖所有组件”的仓库，比较常见的是拆成训练环境、LIBERO 环境、OpenPI 环境三套。若你先只想把训练侧跑通，可以先准备：

```bash
cd /vla/users/niejunnan/codebase/serl_torch
python -m pip install -r serl_launcher/requirements.txt
python -m pip install -e serl_launcher
```

如果本地还没有 ResNet 权重，可以下载：

```bash
cd /vla/users/niejunnan/codebase/serl_torch
python tools/download_resnet.py --models microsoft/resnet-18
```

常用环境变量：

```bash
export SERL_CONDA_ENV=serl_torch
export LIBERO_CONDA_ENV=libero
export OPENPI_CONDA_ENV=openpi-modified
```

如果你已经按当前工作区默认环境配置好了，这些变量通常也可以不手动设置。

## 从仓库根目录启动 LIBERO

最基础的 LIBERO 用法是三步：起环境服务、起 OpenPI、再训练或评测。

### 启动环境服务

```bash
cd /vla/users/niejunnan/codebase/serl_torch
bash examples/libero/tools/serve_env.sh
```

默认监听 `127.0.0.1:30000`。

### 启动 OpenPI 服务

```bash
cd /vla/users/niejunnan/codebase/serl_torch
bash examples/libero/tools/serve_openpi.sh
```

默认监听 `30001`，并优先使用本地 `pi0_libero` checkpoint。

### 启动训练

```bash
cd /vla/users/niejunnan/codebase/serl_torch
bash examples/libero/tools/train.sh \
  task.suite_name=libero_10 \
  task.task_id=0 \
  offline.enabled=true \
  normalization.enabled=true
```

### 启动评测

```bash
cd /vla/users/niejunnan/codebase/serl_torch
bash examples/libero/tools/eval.sh \
  task.task_id=0 \
  eval.checkpoint_path=/abs/path/to/checkpoint_0005000.pt
```

## exp10 示例

如果你当前的重点是：

```text
examples/libero/configs/exp10
```

那最常见的就是 `libero_10 task_id=8` 的异步 residual RL 训练。这个目录已经把端口、GPU、输出路径、pi0/pi05 checkpoint 选择都封装进去了。

### 方式 1：直接用封装脚本

推荐先用这一种。

```bash
cd /vla/users/niejunnan/codebase/serl_torch
bash examples/libero/configs/exp10/COMMANDS.sh launch
```

等价的绝对路径写法是：

```bash
bash /vla/users/niejunnan/codebase/serl_torch/examples/libero/configs/exp10/COMMANDS.sh launch
```

当前 `launch` 简写默认等价于：

```bash
bash examples/libero/configs/exp10/COMMANDS.sh null launch
```

也就是默认启动 `null` 这组配置。它会自动拉起这 5 个角色：

1. `env`
2. `async_eval_env`
3. `openpi`
4. `learner`
5. `actor`

常见变体：

```bash
bash examples/libero/configs/exp10/COMMANDS.sh null_utdeff1p launch
bash examples/libero/configs/exp10/COMMANDS.sh null_unfreeze launch
bash examples/libero/configs/exp10/COMMANDS.sh null_pi05 launch
```

如果你要顺序跑完整个 `exp10` 的 8 组配置，可以看：

```bash
bash examples/libero/configs/exp10/FULL_COMMANDS.sh
```

### 方式 2：原始方法，手动开 5 个终端

如果你想更细地控制每个进程，下面给的是和 `null` 这组 run 对齐的原始启动方法。

先在任意一个终端准备路径：

```bash
cd /vla/users/niejunnan/codebase/serl_torch
export ROOT=$(pwd)
export CFG=examples/libero/configs/exp10/chunk/libero_10_task_8_chunk_async_null.yaml
export RUN_ROOT=$ROOT/examples/libero/outputs/exp10/manual/null
export BOOTSTRAP=$RUN_ROOT/agentlace_bootstrap.pkl
mkdir -p "$RUN_ROOT" "$RUN_ROOT/learner" "$RUN_ROOT/actor"
```

终端 1，训练环境服务：

```bash
cd /vla/users/niejunnan/codebase/serl_torch
bash examples/libero/tools/serve_env.sh \
  --host 127.0.0.1 \
  --port 36790
```

终端 2，异步评测环境服务：

```bash
cd /vla/users/niejunnan/codebase/serl_torch
bash examples/libero/tools/serve_env.sh \
  --host 127.0.0.1 \
  --port 36792
```

终端 3，OpenPI 服务：

```bash
cd /vla/users/niejunnan/codebase/serl_torch
bash examples/libero/tools/serve_openpi.sh \
  --port 36791 \
  --gpu-id 0
```

终端 4，Learner：

```bash
cd /vla/users/niejunnan/codebase/serl_torch
bash examples/libero/tools/run_learner.sh \
  "$CFG" \
  --bootstrap "$BOOTSTRAP" \
  --gpu_id 1 \
  hydra.run.dir="$RUN_ROOT/learner"
```

终端 5，Actor：

```bash
cd /vla/users/niejunnan/codebase/serl_torch
bash examples/libero/tools/run_actor.sh \
  "$CFG" \
  --bootstrap "$BOOTSTRAP" \
  --gpu_id 0 \
  hydra.run.dir="$RUN_ROOT/actor"
```

建议启动顺序就是上面这 5 步。`learner` 先起来后等待 bootstrap 文件是正常的，这个文件会由 `actor` 初始化。

如果你要手动启动 `pi05` 版本，主要区别在 `openpi` 终端：

```bash
cd /vla/users/niejunnan/codebase/serl_torch
POLICY_CONFIG=pi05_libero \
POLICY_DIR=/vla/users/niejunnan/openpi-assets/checkpoints/pi05_libero \
bash examples/libero/tools/serve_openpi.sh \
  --port 40011 \
  --gpu-id 0
```

## 常用文档入口

- `examples/libero/README.md`: LIBERO 子目录说明
- `examples/libero/configs/exp10/README.md`: `exp10` 目录专门说明
- `examples/RoboTwin/README.md`: RoboTwin 子目录说明
- `docs/`: 仓库级实验分析与实现说明

## 一句话总结

如果你现在只想尽快开始跑：

1. 先 `cd /vla/users/niejunnan/codebase/serl_torch`
2. 再跑 `bash examples/libero/configs/exp10/COMMANDS.sh launch`
3. 如果要手动控进程，就按上面的 5 终端方式启动
