# serl_torch

`serl_torch` 是当前这套 residual RL / VLA 实验代码仓库。现在这条分支的主线用法，已经收敛到：

- `serl_launcher/` 放公共训练、数据、runtime、policy backend 逻辑
- `examples/libero/` 放 LIBERO 环境相关的 schema、runtime adapter、env wrapper 和实验配置
- `examples/libero/configs/exp11/` 作为当前推荐的端到端训练入口

如果你现在要真正跑通一组实验，建议直接按下面的 `exp11 + pi05` 流程走。

## 仓库结构

- `serl_launcher/`: 通用 residual 数据管线、runtime、policy backend、replay、agent、训练辅助
- `examples/libero/`: LIBERO 任务适配、环境服务、数据物化、训练/评测脚本、实验配置
- `examples/RoboTwin/`: RoboTwin 示例
- `pretrained_models/`: 本地视觉 backbone 权重
- `docs/`: 额外说明
- `third_party/`: 本地依赖或参考仓库

## 当前默认假设

这套代码当前默认假设本机存在下面这些路径：

- `LIBERO` submodule: `/vla/users/niejunnan/codebase/serl_torch/third_party/LIBERO`
- LIBERO datasets: `/vla/users/niejunnan/datasets`
- OpenPI repo: `/vla/users/niejunnan/codebase/openpi`
- OpenPI `pi05` checkpoint: `/vla/users/niejunnan/openpi-assets/checkpoints/pi05_libero`

如果你的机器路径不同，就在命令行显式覆盖：

- `libero_root=...`  # only needed if you want to override the bundled `third_party/LIBERO`
- `libero_datasets_root=...`
- `OPENPI_ROOT=...`
- `POLICY_DIR=...`

## 环境准备

训练环境：

```bash
cd /vla/users/niejunnan/codebase/serl_torch
conda activate serl_torch
pip install -e /vla/users/niejunnan/codebase/serl_torch/serl_launcher
pip install -e /vla/users/niejunnan/codebase/openpi/packages/openpi-client
```

如果没有本地 ResNet-18 权重：

```bash
cd /vla/users/niejunnan/codebase/serl_torch
python tools/download_resnet.py --models microsoft/resnet-18
```

运行时一般会用到三套环境：

- `serl_torch`: actor / learner / 数据物化
- `libero`: 环境服务
- `openpi-modified`: OpenPI 服务

## 当前推荐主线：`exp11` 的 `pi05` 训练

这条主线依赖两类统一格式数据：

- offline residual training PKL
- online prefill residual training PKL

训练配置示例：

- [libero_10_task_8_chunk_async_null_utdeff1p_unfreeze_pi05.yaml](/vla/users/niejunnan/codebase/serl_torch/examples/libero/configs/exp11/chunk/libero_10_task_8_chunk_async_null_utdeff1p_unfreeze_pi05.yaml)

### 1. 生成 offline 数据

先启动一个专门的 OpenPI 服务：

```bash
cd /vla/users/niejunnan/codebase/serl_torch
POLICY_CONFIG=pi05_libero \
POLICY_DIR=/vla/users/niejunnan/openpi-assets/checkpoints/pi05_libero \
bash examples/libero/tools/serve_openpi.sh \
  --port 40111 \
  --gpu-id 2
```

然后在另一个终端物化 offline 数据：

```bash
cd /vla/users/niejunnan/codebase/serl_torch/examples/libero
CONVERT_CONDA_ENV=serl_torch \
bash tools/convert_offline.sh \
  --suite_name libero_10 \
  --task_id 8 \
  --chunk_horizon 5 \
  --residual_alpha 0.10 \
  --libero_datasets_root /vla/users/niejunnan/datasets \
  --openpi_host 127.0.0.1 \
  --openpi_port 40111 \
  --output_dir data/residual_training/offline_pi05_alpha01
```

生成结果会落到：

- `examples/libero/data/residual_training/offline_pi05_alpha01/libero_10_task_8`

### 2. 生成 online warmup / prefill 数据

先起一个 LIBERO 环境服务：

```bash
cd /vla/users/niejunnan/codebase/serl_torch
bash examples/libero/tools/serve_env.sh \
  --host 127.0.0.1 \
  --port 40110
```

然后在另一个终端收集在线 warmup 数据：

```bash
cd /vla/users/niejunnan/codebase/serl_torch/examples/libero
PREFILL_CONDA_ENV=serl_torch \
bash tools/collect_online_prefill.sh \
  /vla/users/niejunnan/codebase/serl_torch/examples/libero/configs/exp11/chunk/libero_10_task_8_chunk_async_null_pi05.yaml \
  --episodes 100 \
  --output_dir data/residual_training/online_pi05 \
  libero_datasets_root=/vla/users/niejunnan/datasets \
  openpi.host=127.0.0.1 \
  openpi.port=40111 \
  env.remote.host=127.0.0.1 \
  env.remote.port=40110 \
  training.async_eval.enabled=false
```

生成结果会落到：

- `examples/libero/data/residual_training/online_pi05/libero_10_task_8/stepchunk/manifest.json`

### 3. 启动端到端训练

当前最推荐的方式是直接用 Hydra launcher，而不是再套一层历史实验脚本。这样你可以显式指定 GPU 和路径覆盖。

```bash
cd /vla/users/niejunnan/codebase/serl_torch
POLICY_CONFIG=pi05_libero \
POLICY_DIR=/vla/users/niejunnan/openpi-assets/checkpoints/pi05_libero \
bash examples/libero/tools/launch_async_train.sh \
  /vla/users/niejunnan/codebase/serl_torch/examples/libero/configs/exp11/chunk/libero_10_task_8_chunk_async_null_utdeff1p_unfreeze_pi05.yaml \
  launch.actor_gpu=1 \
  launch.learner_gpu=3 \
  launch.openpi_gpu=4 \
  libero_datasets_root=/vla/users/niejunnan/datasets
```

这条命令会自动拉起 5 个角色：

1. training env server
2. async eval env server
3. OpenPI server
4. standalone learner
5. actor

### 4. 训练日志怎么看

每次运行都会生成一个 Hydra run 目录，例如：

- `examples/libero/outputs/exp11/libero_10_task_8_chunk_async_null_utdeff1p_unfreeze_pi05/<timestamp>/`

常看的文件：

- `launch_async_train.log`
- `support/env_server.log`
- `support/async_eval_env_server.log`
- `support/openpi_server.log`
- `support/learner.log`
- `support/actor.log`

训练启动成功后，你应该能在 `learner.log` 里看到：

- offline preload 完成
- online prefill 加载完成
- `Starting standalone agentlace learner`

在 `actor.log` 里看到：

- `Connected actor to external agentlace learner`
- `Start phase=residual_rl`

## 端到端训练里的组件关系

当前 `exp11` 这条链路里，主要组件是：

- `launch_async_train.py`: 一次性拉起整组训练服务
- `libero_env_server.py`: 提供远程 LIBERO 环境
- `OpenPIPolicyClient`: actor 调用的 base policy client
- `train_residual_sac.py`: actor 主进程，负责 rollout
- `run_learner.py`: learner 主进程，负责 offline preload、online replay、参数更新、checkpoint
- `agentlace_bootstrap.pkl`: actor 初始化后写出的 bootstrap，上游给 learner 启动

数据流是：

- env -> actor -> OpenPI -> actor -> env
- actor -> learner replay
- learner -> actor 参数广播

## 额外说明

- `examples/libero/configs/exp10/` 现在主要作为历史实验配置参考，不是这条分支的推荐主线
- `examples/libero/data/` 现在只用于放真实数据，不再放 Python 代码
- 当前 `OpenPI` 已经作为 policy backend 抽到了 `serl_launcher/policy/openpi/`

## 最短总结

如果你今天只想把 `pi05` 的一组训练真正跑起来，就按下面顺序：

1. `pip install -e serl_launcher`
2. `pip install -e /vla/users/niejunnan/codebase/openpi/packages/openpi-client`
3. 生成 `offline_pi05_alpha01`
4. 生成 `online_pi05`
5. 用 `launch_async_train.sh + exp11 pi05 yaml` 启动整组训练
