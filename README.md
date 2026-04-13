# serl_torch

`serl_torch` 是当前这套 residual RL / VLA 实验仓库。

这份根目录 README 现在主要回答四类问题：

1. 这个仓库现在怎么分层；
2. 代码应该从哪里开始看；
3. 如果你要真正跑实验，应该去看哪个 example 的 README。
4. 少量已经验证过的关键运行摘要应该怎么看。

如果你当前要跑 LIBERO，请直接看：

- [examples/libero/README.md](examples/libero/README.md)

如果你当前要跑 AgiBot 真机 residual RL，请直接看：

- [examples/agibot_real/README.md](examples/agibot_real/README.md)

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
- `examples/agibot_real/`
  AgiBot 真机环境适配、controller runtime、训练/评测脚本、服务脚本

## 仓库目录

- `serl_launcher/`
  公共训练、policy backend、agent、residual 编排
- `examples/libero/`
  当前最完整、最常用的 example
- `examples/agibot_real/`
  当前 AgiBot 真机 residual RL example
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
2. [examples/agibot_real/README.md](examples/agibot_real/README.md)
3. [serl_launcher/serl_launcher/training/](serl_launcher/serl_launcher/training/)
4. [serl_launcher/serl_launcher/residual/algorithms/](serl_launcher/serl_launcher/residual/algorithms/)
5. [serl_launcher/serl_launcher/residual/train/](serl_launcher/serl_launcher/residual/train/)
6. [examples/libero/runtime/](examples/libero/runtime/)

## 安装

最常见的训练环境是 `serl_torch`：

```bash
cd /vla/users/niejunnan/codebase/serl_torch
conda activate serl_torch
pip install -e /workspace/serl_torch/serl_launcher
pip install -e /workspace/openpi/packages/openpi-client
```

如果没有本地 ResNet-18 权重：

```bash
cd /vla/users/niejunnan/codebase/serl_torch
python tools/download_resnet.py --models microsoft/resnet-18
```

当前常用到四套环境：

- `serl_torch`
  actor / learner / eval / 数据准备
- `robot`
  当前 AgiBot 真机本地训练时常用的合并环境
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

根目录 README 原则上不放很长的 task 级运行命令，但会补少量已经验证过、并且反复容易卡住的关键运行摘要。

当前建议按下面方式分工：

- 根目录 README
  负责讲仓库结构、安装、分层、入口导航，以及少量已经验证过的跨 example 运行摘要
- `examples/libero/README.md`
  负责讲 LIBERO 具体怎么跑，包括：
  - 环境变量
  - 本地 config/cache 逻辑
  - scripts / tools 说明
  - offline / online 数据准备
  - async train
  - eval smoke
  - 日志和常见问题
- `examples/agibot_real/README.md`
  负责讲 AgiBot 真机的 controller 模式、env backend、service/tooling、训练命令和更多真机注意事项

## AgiBot 真机当前主线

`examples/agibot_real` 现在只保留一条 canonical 训练主线：

- config: `examples/agibot_real/configs/train_residual.yaml`
- entrypoint: `examples/agibot_real/scripts/run_residual_training.py`
- wrappers:
  - `examples/agibot_real/tools/run_actor.sh`
  - `examples/agibot_real/tools/run_learner.sh`

当前默认假设：

- `env.backend=local`
- actor / learner 分进程
- controller 由 actor 前台终端接管
- 默认基座策略服务是 OpenPI，地址 `127.0.0.1:30001`

推荐启动顺序：

1. 启动 base-policy server
2. 启动 `robot-service`
3. 启动 learner
4. 启动 actor

controller 默认按键：

- `g`: ready / resume
- `p`: pause
- `r`: reset
- `s`: success
- `f`: fail
- `h`: help

旧的 Agentlace bootstrap 训练链和旧 eval 入口已经从 `examples/agibot_real` 移除。新的 canonical eval 入口还没有补上，所以当前这个 example 只文档化训练主线。

更完整的 AgiBot 真机说明，请直接看：

- [examples/agibot_real/README.md](examples/agibot_real/README.md)

## 当前状态

当前已经验证过的主链路包括：

- offline data preparation
- online prefill collection
- async train smoke
- AgiBot real local actor / learner / controller train startup

对应的具体命令和配置，统一放在：

- [examples/libero/README.md](examples/libero/README.md)
- [examples/agibot_real/README.md](examples/agibot_real/README.md)
