# serl_torch

`serl_torch` 是当前这套 residual RL / VLA 实验仓库。

当前主线已经收敛到两条 example：

- [examples/libero/README.md](examples/libero/README.md)
  LIBERO residual RL，支持本地 env 和 remote env server，带独立 checkpoint eval
- [examples/agibot_real/README.md](examples/agibot_real/README.md)
  AgiBot 真机 residual RL，当前只文档化 canonical 训练主线

如果你现在要真正跑实验，建议直接从对应 example 的 README 开始。

## 当前仓库在做什么

当前主线围绕下面几件事展开：

- chunk base policy
- residual action composition
- actor / learner 异步训练
- 多环境 example 适配

代码边界现在比较清楚：

- `serl_launcher/`
  当前仍在复用的公共库代码，包括 agent、policy client、checkpoint、RPC、timer、replay buffer、vision encoder
- `examples/libero/`
  LIBERO 环境适配、训练脚本、eval 脚本、env server、typed config
- `examples/agibot_real/`
  AgiBot 真机环境适配、controller runtime、robot-service 启动脚本、训练脚本、typed config

## 目录导航

- `serl_launcher/`
  当前主线复用的公共库
- `examples/libero/`
  当前最完整的仿真 example
- `examples/agibot_real/`
  当前真机 example
- `examples/RoboTwin/`
  另一个 example，不是当前文档主线
- `third_party/`
  vendored 依赖，例如 `third_party/LIBERO`
- `docs/`
  设计说明、重构笔记、分析文档
- `pretrained_models/`
  本地缓存的视觉 backbone 权重
- `scripts/`
  仓库级工具脚本，例如 ResNet 下载脚本

## 推荐阅读顺序

如果你要快速理解现在这套代码，推荐顺序是：

1. [examples/libero/README.md](examples/libero/README.md)
2. [examples/agibot_real/README.md](examples/agibot_real/README.md)
3. `serl_launcher/serl_launcher/policy/`
4. `serl_launcher/serl_launcher/residual/`
5. `serl_launcher/serl_launcher/data/`
6. `serl_launcher/serl_launcher/agents/continuous/`

## 安装

最常见的 Python 环境是 `serl_torch`。

```bash
cd /vla/users/niejunnan/codebase/serl_torch
conda activate serl_torch
pip install -r serl_launcher/requirements.txt
pip install -e ./serl_launcher
```

`agentlace` 目前没有写进 `serl_launcher/setup.py`，需要手工安装。仓库里的注释给出的参考做法是：

```bash
git clone https://github.com/youliangtan/agentlace.git
cd agentlace
git checkout cf2c337c5e3694cdbfc14831b239bd657bc4894d
pip install -e .
```

如果你使用 `policy.type=openpi`，还需要让训练环境能导入 `openpi-client`：

```bash
pip install -e /path/to/openpi/packages/openpi-client
```

如果你不是用 editable install，也可以手工补 `PYTHONPATH`：

```bash
export PYTHONPATH=/vla/users/niejunnan/codebase:/vla/users/niejunnan/codebase/serl_torch/serl_launcher:$PYTHONPATH
```

## 视觉权重

如果本机没有缓存 HuggingFace ResNet 权重，可以先下载：

```bash
cd /vla/users/niejunnan/codebase/serl_torch
python scripts/download_resnet.py --models microsoft/resnet-18
```

下载完成后，把配置里的 `encoder.resnet.model_name` 指到本地目录，例如：

```text
pretrained_models/microsoft--resnet-18
```

## 常见环境拆分

当前常见到的环境拆分大概是：

- `serl_torch`
  actor / learner / eval / 通用脚本
- `libero`
  LIBERO env server
- `robot`
  AgiBot 真机本地 runtime
- `openpi` 或你们自己的 OpenPI 环境
  base policy server
- `joyra`
  JoyRA server

## 当前默认路径假设

这套代码通常会默认使用下面这些本机路径假设：

- LIBERO checkout:
  `/vla/users/niejunnan/codebase/serl_torch/third_party/LIBERO`
- LIBERO datasets:
  `/vla/users/niejunnan/datasets`
- OpenPI repo:
  `/vla/users/niejunnan/codebase/openpi`
- 本地 ResNet 缓存：
  `/vla/users/niejunnan/codebase/serl_torch/pretrained_models`

如果你的机器路径不同，请在命令行或环境变量里显式覆盖，例如：

- `libero_root=...`
- `libero_datasets_root=...`
- `libero_config_dir=...`
- `encoder.resnet.model_name=...`
- `OPENPI_ROOT=...`
- `POLICY_DIR=...`

## 两条当前主线

### LIBERO

当前 canonical LIBERO 主线包括：

- config:
  `examples/libero/configs/train_residual.yaml`
- actor / learner entrypoint:
  `examples/libero/scripts/run_residual_training.py`
- env server:
  `examples/libero/scripts/serve_env.py`
- checkpoint eval:
  `examples/libero/scripts/evaluate_checkpoint.py`

这条线支持：

- `policy.type=openpi`
- `policy.type=joyra`
- `env.backend=local`
- `env.backend=remote`
- 训练期 async eval

更完整的启动方式见：

- [examples/libero/README.md](examples/libero/README.md)

### AgiBot Real

当前 canonical AgiBot 真机主线包括：

- config:
  `examples/agibot_real/configs/train_residual.yaml`
- actor / learner entrypoint:
  `examples/agibot_real/scripts/run_residual_training.py`
- actor wrapper:
  `examples/agibot_real/tools/run_actor.sh`
- learner wrapper:
  `examples/agibot_real/tools/run_learner.sh`
- robot-service wrapper:
  `examples/agibot_real/tools/start_robot_service.sh`

这条线当前约束比较明确：

- `env.backend=local`
- `env.action_dim=14`
- `task.control_mode=camera_position`
- 当前只文档化训练主线
- canonical eval 入口还没有补齐

更完整的启动方式见：

- [examples/agibot_real/README.md](examples/agibot_real/README.md)

## 当前哪些目录是主线，哪些不是

建议把下面这些目录当成当前主线：

- `serl_launcher/serl_launcher/`
- `examples/libero/`
- `examples/agibot_real/`

下面这些目录更适合作为补充参考，而不是当前文档主线：

- `examples/RoboTwin/`
- `reference/`
- `third_party/`
- `examples/libero/outputs/`

尤其是 `outputs/` 目录，里面主要是历史运行产物，不应该当成当前实现说明。

## README 分工

- 根目录 `README.md`
  负责讲仓库定位、安装、分层、入口导航
- `examples/libero/README.md`
  负责讲 LIBERO 训练、env server、async eval、checkpoint eval
- `examples/agibot_real/README.md`
  负责讲 AgiBot 真机 runtime、robot-service、actor / learner 启动和 controller 工作流

## 当前状态

当前已经收敛并仍在维护的，是一套 typed-config 驱动的 residual RL 主线：

- 训练脚本不再按环境各自散落很多旧入口
- policy backend 统一走 `serl_launcher.policy.*`
- residual action 统一走 `serl_launcher.residual.typed_action`
- replay / checkpoint / timer / JSONL 等基础设施集中在 `serl_launcher/`

如果你想确认某条命令现在还能不能跑，请优先看 example 目录里的 README 和实际存在的脚本名，而不是旧文档或历史输出目录。
