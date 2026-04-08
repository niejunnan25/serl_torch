# serl_torch

`serl_torch` 是当前这套 residual RL / VLA 实验仓库。

这份根目录 README 只回答三类问题：

1. 这个仓库现在怎么分层；
2. 代码应该从哪里开始看；
3. 如果你要真正跑实验，应该去看哪个 example 的 README。

如果你当前要跑 LIBERO，请直接看：

- [examples/libero/README.md](examples/libero/README.md)

## 仓库定位

当前主线是一套围绕：

- chunk base policy
- residual RL
- actor / learner 异步训练
- 多环境 example 适配

展开的实验代码。

目前已经重点整理过两条边界：

- `serl_launcher/` 负责公共训练与算法基础设施
- `examples/<env>/` 负责环境适配、脚本入口、实验配置

## 当前代码分层

推荐按下面这套分层理解：

- `serl_launcher/training/`
  通用训练基础设施，例如 checkpoint、profiling、async runtime、telemetry
- `serl_launcher/residual/algorithms/`
  residual 算法接口与实现
- `serl_launcher/residual/train/`
  residual actor / learner 编排
- `serl_launcher/residual/data/`
  residual 训练数据 schema、materialize、loader
- `serl_launcher/policy/`
  base chunk policy backend
- `serl_launcher/agents/`
  RL agent 本体和 builder
- `examples/libero/`
  LIBERO 环境适配、环境服务、数据准备、训练/评测脚本、实验配置

## 仓库目录

- `serl_launcher/`
  公共训练、policy backend、agent、residual 编排
- `examples/libero/`
  当前最完整、最常用的 example
- `examples/RoboTwin/`
  另一个 example
- `docs/`
  设计说明、重构记录、分析文档
- `third_party/`
  本地依赖或参考仓库
- `pretrained_models/`
  本地模型权重

## 当前推荐入口

如果你要理解这套仓库现在的主线，建议按下面顺序看：

1. [examples/libero/README.md](examples/libero/README.md)
2. [serl_launcher/serl_launcher/training/](serl_launcher/serl_launcher/training/)
3. [serl_launcher/serl_launcher/residual/algorithms/](serl_launcher/serl_launcher/residual/algorithms/)
4. [serl_launcher/serl_launcher/residual/train/](serl_launcher/serl_launcher/residual/train/)
5. [examples/libero/runtime/](examples/libero/runtime/)

## 安装

最常见的训练环境是 `serl_torch`：

```bash
cd /vla/users/niejunnan/codebase/serl_torch
conda activate serl_torch
pip install -e serl_launcher
pip install -e /vla/users/niejunnan/codebase/openpi/packages/openpi-client
```

如果没有本地 ResNet-18 权重：

```bash
cd /vla/users/niejunnan/codebase/serl_torch
python tools/download_resnet.py --models microsoft/resnet-18
```

当前常用到三套环境：

- `serl_torch`
  actor / learner / eval / 数据准备
- `libero`
  LIBERO 环境服务
- `openpi-modified`
  OpenPI 服务

## 当前默认路径假设

这套代码通常会默认使用下面这些本机路径：

- LIBERO submodule:
  `/vla/users/niejunnan/codebase/serl_torch/third_party/LIBERO`
- LIBERO datasets:
  `/vla/users/niejunnan/datasets`
- OpenPI repo:
  `/vla/users/niejunnan/codebase/openpi`
- OpenPI checkpoint:
  `/vla/users/niejunnan/openpi-assets/checkpoints/pi05_libero`

如果你的机器不同，请在命令行或环境变量里显式覆盖，例如：

- `libero_root=...`
- `libero_datasets_root=...`
- `OPENPI_ROOT=...`
- `POLICY_DIR=...`

## Example README 的分工

根目录 README 不放很长的 task 级运行命令。

当前建议按下面方式分工：

- 根目录 README
  负责讲仓库结构、安装、分层、入口导航
- `examples/libero/README.md`
  负责讲 LIBERO 具体怎么跑，包括：
  - 环境变量
  - 本地 config/cache 逻辑
  - scripts / tools 说明
  - offline / online 数据准备
  - async train
  - eval smoke
  - 日志和常见问题

## 当前状态

当前已经验证过的主链路包括：

- offline data preparation
- online prefill collection
- async train smoke
- checkpoint eval smoke

对应的具体命令和配置，统一放在：

- [examples/libero/README.md](examples/libero/README.md)
