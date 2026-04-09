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
  负责讲 AgiBot 真机的 controller 模式、env backend、service/tooling、训练/评测命令和更多真机注意事项

## AgiBot 真机跑通流程摘要

下面这段是当前仓库在 `2026-04-09` 这次排障中实际验证过的本地真机训练流程摘要。

目标不是替代 [examples/agibot_real/README.md](examples/agibot_real/README.md)，而是把最容易卡住的启动顺序、运行现象和排障点集中写清楚。

### 适用场景

这份流程对应的是当前推荐路径：

- `examples/agibot_real`
- `env.backend=local`
- actor / learner 分进程
- controller 由 actor 终端接管
- OpenPI 通过 `localhost:9000` 提供 chunk policy

当前验证时使用的关键路径是：

- AgiBot SDK: `/workspace/tangyili/a2d_sdk`
- SERL repo: `/workspace/residual_rl/serl_torch`
- 本地 ResNet-18 镜像:
  `/workspace/residual_rl/serl_torch/pretrained_models/microsoft--resnet-18`
- 训练 config:
  `examples/agibot_real/conf/train_office_setting_mouse_pi05.yaml`

### 先理解这条链路里谁负责什么

- `robot-service`
  负责 AgiBot SDK / 机器人侧服务初始化
- OpenPI server
  负责 base chunk policy 推理
- learner
  负责 RL learner、replay 消费和参数更新
- actor
  负责真实机器人 env、controller、base policy + residual policy 组合动作、在线数据采集

actor 和 learner 之间通过同一个 `agentlace_bootstrap.pkl` 握手。

### 一个很重要的前提

`tools/run_actor.sh` / `tools/run_learner.sh` 会帮你切 conda env，但它们不会自动执行：

```bash
source /workspace/tangyili/a2d_sdk/env.sh
```

如果你的 AgiBot 本地环境依赖这个 `env.sh` 导出的动态库、DDS、protobuf 运行时变量，那就必须在你启动 actor/learner 的 shell 里先手工 `source` 它。

这是这条链路里最容易忽略的前置条件之一。

### 推荐启动顺序

建议至少用 3 到 4 个终端：

1. `robot-service`
2. OpenPI
3. learner
4. actor

如果你的 OpenPI 已经在别的 docker/container 里稳定跑着，并且监听的是配置里的 `localhost:9000`，那这一项可以直接复用。

### 终端 A：启动 AgiBot SDK 服务

```bash
cd /workspace/tangyili/a2d_sdk
source env.sh
source /root/miniconda3/etc/profile.d/conda.sh
conda activate robot
robot-service -s -c ./conf/copilot.pbtxt
```

说明：

- `robot-service` 在当前环境里是可以正常启动的
- 它启动前会有一段等待时间，看起来像“停住”一会儿是正常现象
- 如果它没有起来，actor 后面的本地机器人接口大概率也不会正常

### 终端 B：启动 OpenPI

如果你已经有现成的 OpenPI 服务监听 `localhost:9000`，可以直接跳过这一步。

如果没有，就按 `examples/agibot_real/README.md` 里的 OpenPI 服务方式单独启动。关键点只有一个：

- `openpi.port` 必须和训练 config 里一致

当前这次验证里，actor 读取到的是：

```yaml
openpi:
  host: localhost
  port: 9000
```

### 终端 C：启动 learner

```bash
cd /workspace/tangyili/a2d_sdk
source env.sh
source /root/miniconda3/etc/profile.d/conda.sh
conda activate robot

export SERL_CONDA_ENV=robot
export SERL_RESNET_MODEL=/workspace/residual_rl/serl_torch/pretrained_models/microsoft--resnet-18

cd /workspace/residual_rl/serl_torch/examples/agibot_real

bash tools/run_learner.sh \
  conf/train_office_setting_mouse_pi05.yaml \
  --bootstrap /workspace/residual_rl/serl_torch/examples/agibot_real/outputs/agibot_real/office_setting_mouse_pi05/agentlace_bootstrap.pkl \
  hydra.run.dir=/workspace/residual_rl/serl_torch/examples/agibot_real/outputs/agibot_real/office_setting_mouse_pi05/learner
```

说明：

- learner 和 actor 必须使用同一个 `--bootstrap` 文件路径
- learner 先起是正常的，它会先打印 `Waiting for actor bootstrap`
- 只有当 actor 成功把 bootstrap 文件写出来以后，learner 才会继续往下走

### 终端 D：启动 actor

```bash
cd /workspace/tangyili/a2d_sdk
source env.sh
source /root/miniconda3/etc/profile.d/conda.sh
conda activate robot

export SERL_CONDA_ENV=robot
export SERL_RESNET_MODEL=/workspace/residual_rl/serl_torch/pretrained_models/microsoft--resnet-18

cd /workspace/residual_rl/serl_torch/examples/agibot_real

bash tools/run_actor.sh \
  conf/train_office_setting_mouse_pi05.yaml \
  --bootstrap /workspace/residual_rl/serl_torch/examples/agibot_real/outputs/agibot_real/office_setting_mouse_pi05/agentlace_bootstrap.pkl \
  hydra.run.dir=/workspace/residual_rl/serl_torch/examples/agibot_real/outputs/agibot_real/office_setting_mouse_pi05/actor
```

说明：

- actor 终端必须保持前台，因为它不仅在跑 env，还负责 controller 键盘输入
- 当前代码已经支持离线加载本地 `microsoft/resnet-18` 镜像
- 如果仓库根目录下存在 `pretrained_models/microsoft--resnet-18`，当前代码会自动把 `microsoft/resnet-18` 解析到这个本地目录
- 即便如此，真机离线环境里仍然建议显式导出 `SERL_RESNET_MODEL`，这样日志更清晰，排障也更直接

### 正常日志应该长什么样

一条已经跑通到训练主循环的日志，大致会经历下面这些阶段：

1. learner 打印 `Waiting for actor bootstrap`
2. actor 打印 config、AgiBot task、Residual algorithm
3. actor 打印 `Agentlace bootstrap saved ...`
4. learner 打印 `Loaded bootstrap payload ...`
5. learner 打印 `Starting standalone agentlace learner at 127.0.0.1:5488`
6. actor 打印 `Connected actor to external agentlace learner at 127.0.0.1:5488`
7. actor 打印 `Initialized DrQ agent, replay buffer, and offline pipeline`
8. actor 打印 `Start controller phase=...`

如果已经走到第 8 步，说明：

- actor / learner 握手成功
- 本地 ResNet 权重加载成功
- OpenPI 客户端初始化已经过了
- 训练循环已经正式进入 controller episode

### 为什么看起来“卡住了”

这是当前 AgiBot controller 模式最容易误判的一点。

在 `controller.enabled=true` 的情况下，episode `reset` 完成以后，controller 会先进入 `WAIT_READY` 状态，而不会立刻开始推理和发动作。

也就是说：

- 看到 `Start controller phase=...`
- 看到 `reset_to_task_initial_pose: ...`
- 但还没继续打印推理相关输出

不代表程序卡死；它通常是在等操作员确认。

### 什么时候真正开始推理

要在 actor 终端里按：

```text
g
```

注意：

- 直接按 `g`
- 不需要回车
- 按键必须发生在 actor 所在的那个前台终端

一旦按下 `g`，controller 会从 `WAIT_READY` 切到 `RUNNING`，之后 actor 才会真正调用 OpenPI 的 `infer_chunk()`，开始下发动作。

### Controller 按键

当前默认键位是：

- `g`: ready / resume
- `p`: pause
- `r`: reset / truncate 当前 episode
- `s`: 标记成功
- `f`: 标记失败
- `h`: 重新打印帮助

这几个键只对“拥有 env 的终端”生效：

- `env.backend=local`
  actor 终端生效
- `env.backend=remote`
  `tools/serve_env.sh` 终端生效

### 当前已经验证过的几个关键修复

这条 AgiBot 本地训练链路在这次排障里实际修过几类问题，当前仓库代码已经包含这些修复：

- actor 启动链里避免过早导入 TensorBoard / TensorFlow，修掉了 DDS 初始化前的段错误风险
- `microsoft/resnet-18` 支持自动解析到仓库内的本地镜像目录
- actor controller runtime 的 `profiling_last_flush_step` 状态字段已补齐，不会在第一轮 episode 直接崩
- OpenPI 客户端签名兼容做过向下兼容处理
- AgiBot retargeter 某些 `(3, 1)` / `(3,)` 形状问题已修

### 最常见的卡点和判断方法

#### 1. learner 一直等 bootstrap

现象：

- learner 卡在 `Waiting for actor bootstrap`

说明：

- learner 本身没坏
- actor 还没有把 bootstrap 文件写出来
- 要优先查 actor 是否已经在更早阶段崩了

#### 2. actor 一启动就崩，连 task 日志都没有

优先检查：

- 当前 shell 是否先执行了 `/workspace/tangyili/a2d_sdk/env.sh`
- 当前 conda env 是否真的是可运行 AgiBot 的 `robot` 或等价 merged env
- 是否在使用最新代码

#### 3. actor 报 HuggingFace / `microsoft/resnet-18` 下载错误

说明：

- 当前环境没外网，或者不允许直连 HuggingFace

处理方式：

- 确认本地目录存在：
  `/workspace/residual_rl/serl_torch/pretrained_models/microsoft--resnet-18`
- 建议显式导出：
  `SERL_RESNET_MODEL=/workspace/residual_rl/serl_torch/pretrained_models/microsoft--resnet-18`

#### 4. actor 看起来不动，但已经打印 `Start controller phase=...`

优先判断：

- 这通常不是卡死
- 大概率只是还没有按 `g`

#### 5. 没法按键

如果你看到类似：

```text
stdin is not a TTY
```

说明：

- 这个 actor 不是跑在一个真正可交互的前台终端里
- 例如用了非交互 `docker exec`，或者后台跑了脚本

这时候 controller 按键线程不会生效，必须改成真实 TTY 终端运行。

#### 6. learner 报 `Address already in use`，端口是 `5488`

说明：

- 旧 learner 还在占用 trainer port

处理方式：

- 先停掉旧 learner
- 再重新启动新的 learner / actor

#### 7. CUDA driver 版本过旧 warning

这条 warning 当前不是 bootstrap / controller 主流程的直接阻塞项，但说明：

- 当前 PyTorch 和本机 NVIDIA driver 版本不完全匹配
- 后续如果你要稳定使用 GPU，还是应该单独处理驱动和 PyTorch 组合

### 当前建议的最小操作心智模型

把这条链路简单理解成下面四件事最不容易乱：

1. 先把 `robot-service` 和 OpenPI 准备好
2. learner 先起，等 actor 写 bootstrap
3. actor 连上 learner 以后，会在 reset 后等待人工 ready
4. 在 actor 终端按 `g`，推理和动作才真正开始

如果你已经看到：

- `Connected actor to external agentlace learner`
- `Initialized DrQ agent...`
- `Start controller phase=...`

那下一步最应该做的事情通常不是继续怀疑代码，而是先在 actor 终端按一次 `g`。

## 当前状态

当前已经验证过的主链路包括：

- offline data preparation
- online prefill collection
- async train smoke
- checkpoint eval smoke
- AgiBot real local actor / learner / controller train startup

对应的具体命令和配置，统一放在：

- [examples/libero/README.md](examples/libero/README.md)
- [examples/agibot_real/README.md](examples/agibot_real/README.md)
