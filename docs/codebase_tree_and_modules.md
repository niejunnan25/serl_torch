# serl_torch 目录树与模块组件说明（实仓扫描版）

- 扫描时间：2026-03-20
- 仓库根目录：`/vla/users/niejunnan/codebase/serl_torch`
- 说明：以下目录树是“用于工程理解的压缩版”，默认省略了 `__pycache__`、运行产物中的大量中间文件；会重点标出训练/评估/服务相关组件。

## 1. 顶层目录树（压缩版）

```text
serl_torch/
├── .github/
│   └── workflows/
│       └── pre-commit.yaml
├── .libero/
│   └── config.yaml
├── docs/
│   ├── MIGRATION_JAX_TO_TORCH_INSTRUCTION.md
│   ├── PLD_IMPLEMENTATION_REVIEW.md
│   ├── codebase_guide_robotwin.md
│   ├── sim_quick_start*.md
│   ├── real_franka*.md
│   └── images/
├── examples/
│   ├── RoboTwin/
│   │   ├── conf/
│   │   ├── data/
│   │   ├── docs/
│   │   ├── env_wrappers/
│   │   ├── policy/
│   │   ├── scripts/
│   │   ├── tools/
│   │   └── outputs/
│   ├── libero/
│   │   ├── conf/
│   │   ├── data/
│   │   ├── docs/
│   │   ├── env_wrappers/
│   │   ├── policy/
│   │   ├── scripts/
│   │   ├── tools/
│   │   └── outputs/
│   └── agibot/
├── franka_sim/
│   ├── franka_sim/
│   │   ├── controllers/
│   │   ├── envs/
│   │   ├── test/
│   │   └── mujoco_gym_env.py
│   ├── requirements.txt
│   └── setup.py
├── policy/
│   └── openpi/
├── pretrained_models/
│   ├── microsoft--resnet-18/
│   └── microsoft--resnet-50/
├── scripts/
│   ├── async_drq.py
│   ├── async_sac_state.py
│   ├── bc_policy.py
│   ├── train_reward_classifier.py
│   └── test_classifier.py
├── serl_launcher/
│   ├── serl_launcher/
│   │   ├── agents/
│   │   ├── common/
│   │   ├── data/
│   │   ├── networks/
│   │   ├── residual/
│   │   ├── utils/
│   │   ├── vision/
│   │   └── wrappers/
│   ├── requirements.txt
│   └── setup.py
├── serl_robot_infra/
│   ├── franka_env/
│   ├── robot_servers/
│   └── setup.py
└── tools/
    ├── download.py
    ├── download_resnet.py
    └── test_resnet_loading.py
```

## 2. 顶层模块说明

### 2.1 `serl_launcher/`（算法核心库）

作用：SERL 的通用算法与训练基础设施（PyTorch 版）。

核心子模块：

- `agents/continuous/`
  - `sac.py`：SAC agent。
  - `drq.py`：DrQ agent（继承/扩展 SAC）。
  - `bc.py`：行为克隆 agent。
  - `vice.py`：VICE agent。
- `data/`
  - `replay_buffer.py`：通用 replay buffer。
  - `memory_efficient_replay_buffer.py`：内存高效 replay。
  - `data_store.py`：与异步通信层结合的数据存储（可被 actor/learner 共享）。
  - `dataset.py`：样本采样与数据集抽象。
- `networks/`
  - `actor_critic_nets.py`：policy/Q/value 等网络定义。
  - `mlp.py`：MLP/ResMLP。
  - `reward_classifier.py`：奖励分类器网络与加载逻辑。
  - `lagrange.py`：拉格朗日乘子模块。
- `vision/`
  - `resnet_v1.py`、`mobilenet.py`、`small_encoders.py`：视觉编码器。
  - `data_augmentations.py`：图像增强。
- `wrappers/`
  - `serl_obs_wrappers.py`：SERL 观测格式封装。
  - `chunking.py`：时序 chunk 观测/动作封装。
  - `video_recorder.py`：录制视频。
- `utils/`
  - `launcher.py`：agent/replay/wandb/trainer 工厂。
  - `checkpoint_utils.py`：checkpoint 保存/加载。
  - `train_utils.py`：batch 拼接等训练工具。
  - `timer_utils.py`：计时统计。
- `common/`
  - 训练状态、评估流程、优化器和编码层共用逻辑。

当前仓库现状（重要）：

- `serl_launcher/serl_launcher/residual/` 目录当前仅见 `__pycache__/*.pyc`，未扫描到 `.py` 源码文件。

---

### 2.2 `examples/libero/`（你当前重点使用的训练管线）

作用：LIBERO + OpenPI + residual RL 的完整工程入口（包含训练、异步评估、环境 RPC、数据转换）。

关键组件：

- `conf/`（55 个 YAML）
  - `train_residual_sac.yaml`：主训练模板。
  - `train_pld_*.yaml`：PLD 对齐实验配置。
  - `train_residual_sac_task6_matrix_*.yaml`：矩阵实验配置。
  - `eval_residual_fast.yaml`：评估配置。
- `scripts/`
  - `train_residual_sac.py`：训练主入口（含 offline mixing、warmup、CQL/CalQL 风格预热、async eval watcher 对接）。
  - `eval_residual_fast.py`：快速评估入口。
  - `libero_env_server.py`：环境 HTTP RPC 服务端（`/rpc`）。
  - `convert_hdf5_to_offline.py`：HDF5 转 offline PKL。
  - `compute_hdf5_stats.py`：归一化统计计算。
  - `async_eval_watch.py`：训练过程中异步评估触发器。
- `env_wrappers/`
  - `task_env.py`：本地 LIBERO env 封装。
  - `remote_task_env.py`：远程 RPC env client。
  - `setup.py`：`libero_root/openpi_root/datasets_root` 路径解析。
- `policy/`
  - `openpi_client.py`：调用 OpenPI 服务。
  - `observation.py`：观测提取/缓存与预处理。
  - `action.py`：base+residual 动作组合、受控维度与限幅逻辑。
- `data/`
  - `hdf5_utils.py`：任务规格和数据路径解析。
  - `normalizer.py`：状态/动作归一化读取与应用。
- `utils/`
  - `config_utils.py`：从 YAML 构建 agent/超参。
  - `logger.py`：JSONL 日志。
  - `paths.py`：仓库路径解析与导入辅助。
- `tools/`（生产启动脚本）
  - `run_train.sh`：一键拉起训练 env server / openpi / train / async eval。
  - `serve_env.sh`、`serve_openpi.sh`：服务单独启动。
  - `train.sh`、`eval.sh`：直接训练/评估。
  - `sync_shared_cache.sh`：跨机器共享 cache 同步。

---

### 2.3 `examples/RoboTwin/`（另一套 residual RL 实战样例）

作用：RoboTwin 任务的 Stage-1/2 residual RL 训练与评估流程。

结构与 `examples/libero` 高度同构：

- `conf/`：训练/评估配置。
- `scripts/train_residual_sac.py`：训练主入口。
- `scripts/eval_residual_fast.py`：评估入口。
- `scripts/robotwin_env_server.py`：环境 RPC 服务。
- `scripts/convert_lerobot_to_offline.py`：数据转换。
- `scripts/aggregate_eval_ci.py`：多次评估统计聚合。
- `env_wrappers/`、`policy/`、`data/`、`utils/`：对应功能与 libero 类似。

---

### 2.4 `serl_robot_infra/`（真实机器人基础设施）

作用：真实 Franka 机器人交互层（ROS + Flask + Gym 客户端）。

- `robot_servers/`
  - `franka_server.py`：机器人控制服务主入口。
  - `franka_gripper_server.py` / `robotiq_gripper_server.py`：夹爪服务。
  - `gripper_server.py`：夹爪抽象基类。
- `franka_env/`
  - `envs/franka_env.py`：真实机器人 gym env。
  - `envs/relative_env.py`、`envs/wrappers.py`：坐标系与干预/奖励 wrapper。
  - `envs/{peg_env,pcb_env,cable_env,bin_relocation_env}`：任务级环境。
  - `camera/`：相机采集接口（RealSense/Video）。
  - `spacemouse/`：专家干预输入设备接口。

---

### 2.5 `franka_sim/`（MuJoCo 仿真环境）

作用：Franka 仿真环境（快速本地算法验证）。

- `franka_sim/envs/panda_pick_gym_env.py`：主要仿真任务环境。
- `franka_sim/controllers/opspace.py`：操作空间控制器。
- `franka_sim/mujoco_gym_env.py`：MuJoCo gym 基类封装。
- `franka_sim/test/`：渲染和人工交互测试脚本。

---

### 2.6 `scripts/`（仓库级通用训练脚本）

作用：独立于具体 example 的通用脚本，偏“历史/基础示例”。

- `async_drq.py`：异步 actor/learner 训练样例（视觉）。
- `async_sac_state.py`：异步 SAC 状态观测样例。
- `bc_policy.py`：行为克隆训练/评估。
- `train_reward_classifier.py`、`test_classifier.py`：奖励分类器训练与测试。

---

### 2.7 `tools/`（仓库级下载/检查工具）

- `download_resnet.py`：下载并校验 HuggingFace ResNet 权重。
- `download.py`：简版下载脚本（含镜像配置）。
- `test_resnet_loading.py`：本地权重加载测试。

---

### 2.8 `policy/openpi/`

- 当前仅见 `__pycache__/*.pyc`，未扫描到 `.py` 源码文件。
- 若后续要长期维护，建议补齐源码或迁移到 `examples/*/policy` 的可读实现。

---

### 2.9 `pretrained_models/`

作用：本地缓存的视觉 backbone 权重。

- `microsoft--resnet-18/`、`microsoft--resnet-50/`
  - 包含 `config.json`、`pytorch_model.bin`、`model.safetensors` 等。

---

### 2.10 `.libero/config.yaml`

作用：LIBERO 数据与资源路径统一配置。

当前指向：

- `benchmark_root` / `bddl_files` / `init_states` / `assets`：LIBERO 代码库路径。
- `datasets`：`/vla/users/niejunnan/datasets`。

## 3. 训练/评估主链路（按你当前常用）

以 `examples/libero` 为例：

1. `tools/run_train.sh` 读取 YAML，解析端口与模式。
2. 启动（或复用）环境服务：`scripts/libero_env_server.py`。
3. 启动（或复用）OpenPI 服务：`tools/serve_openpi.sh -> uv run scripts/serve_policy.py`。
4. 训练进程：`scripts/train_residual_sac.py`。
5. 若配置启用异步评估：`scripts/async_eval_watch.py` 触发 `scripts/eval_residual_fast.py`。
6. 产出写入 `examples/libero/outputs/libero/<run_name>/<date>/<time>/...`。

## 4. 目录体量与管理建议

- `examples/*/outputs/` 会快速膨胀（训练日志、ckpt、异步评估日志、tb）。
- `examples/libero/conf/` 目前配置文件数量较多（55 个），建议继续按“实验矩阵前缀”命名并分组管理。
- `examples/agibot/` 当前为空目录，可作为后续新任务接入入口。

## 5. 本次扫描中发现的注意点

1. `serl_launcher/serl_launcher/residual/` 仅有字节码缓存，没有源码。
2. `policy/openpi/` 同样仅见字节码缓存，没有源码。
3. `examples` 下包含大量历史输出，理解代码时建议过滤 `outputs/` 与 `__pycache__/`。

