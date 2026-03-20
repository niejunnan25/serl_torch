# serl_torch 当前代码结构与模块说明

本文基于当前仓库清理后的状态整理。当前仓库已经移除了旧的 `serl_robot_infra/`、旧 README 和一批历史文档，因此现在的代码主线更聚焦在：

- `serl_launcher/`：通用 RL 训练核心
- `examples/libero/`：当前最完整的 OpenPI + residual RL 训练管线
- `examples/RoboTwin/`：另一套 benchmark/task 的 residual RL 实现
- `franka_sim/`：通用 Panda/Franka MuJoCo 仿真环境

如果后续要接入真实机器人，例如 AgiBot，建议在当前结构上新增一个新的 example 包，而不是继续把代码塞进 `examples/libero/` 或 `examples/RoboTwin/`。

## 1. 当前顶层目录结构

```text
serl_torch/
├── .github/                  CI / workflow 配置
├── .libero/                  本地 LIBERO 配置目录
├── docs/                     顶层文档
├── examples/
│   ├── libero/               LIBERO residual RL 实现
│   └── RoboTwin/             RoboTwin residual RL 实现
├── franka_sim/               MuJoCo Panda/Franka 仿真包
├── pretrained_models/        本地缓存的视觉 backbone 权重
├── scripts/                  较早期的通用训练脚本 / 实验脚本
├── serl_launcher/            可复用训练核心库
└── tools/                    顶层工具脚本
```

## 2. 各模块作用

### 2.1 `serl_launcher/`

这是整个仓库最底层、最通用的训练核心。它不关心 LIBERO、RoboTwin、AgiBot 这些具体任务，只提供“怎么训练 agent”的公共能力。

主要子模块如下：

- `serl_launcher/agents/continuous/`
  - `bc.py`：行为克隆
  - `drq.py`：DrQ agent
  - `sac.py`：SAC agent
  - `vice.py`：奖励分类器相关 agent
- `serl_launcher/data/`
  - `replay_buffer.py`：基础 replay buffer
  - `memory_efficient_replay_buffer.py`：更省内存的 replay
  - `dataset.py` / `data_store.py`：数据读取、缓存与流转
- `serl_launcher/networks/`
  - actor/critic MLP、分类器、Lagrange 等网络模块
- `serl_launcher/vision/`
  - ResNet、MobileNet、数据增强、spatial pooling 等视觉编码组件
- `serl_launcher/common/`
  - 编码、评估、optimizer、typing、wandb 等通用逻辑
- `serl_launcher/utils/`
  - checkpoint、训练辅助、timer、launcher 等工具
- `serl_launcher/wrappers/`
  - observation wrapper、chunking、norm、video recorder 等环境包装器

这一层是“算法底座”。如果以后接 AgiBot，通常不需要大改这层，最多是：

- 新增一个更贴合真实机器人输入的 wrapper
- 或在 `agents/continuous/sac.py` 里做少量算法增强

### 2.2 `examples/libero/`

这是当前仓库里最完整、最工程化的一条 residual RL 训练线，已经包含：

- OpenPI base policy 推理
- LIBERO 环境 RPC 服务
- residual SAC/DrQ 训练
- offline demo 转换
- normalization stats
- profiling、异步 actor-learner、replay prefetch、obs cache 等优化

它的子目录职责最清晰，后续接新环境时最值得参考。

#### `examples/libero/conf/`

Hydra 配置目录。

作用：

- 定义训练超参数
- 定义 task、offline、profiling、async learner 等行为
- 管理不同实验变体

如果你做 `examples/agibot/`，这里几乎一定要复制一套。

#### `examples/libero/data/`

数据相关组件。

作用：

- `normalizer.py`：状态和动作归一化
- `hdf5_utils.py`：HDF5 数据读取/转换
- `stats/`：每个任务对应的统计量 JSON

如果 AgiBot 也要支持离线数据混合训练，必须补这一层。

#### `examples/libero/env_wrappers/`

环境适配层，是“任务/机器人 -> 统一训练接口”的关键边界。

主要文件：

- `task_env.py`
  - 本地 LIBERO 环境封装
  - 暴露统一的 `reset/step/close` 接口
- `remote_task_env.py`
  - 训练进程侧的 RPC 客户端
  - 负责通过 HTTP 调用 env server
- `setup.py`
  - 解析 LIBERO 路径
  - 设置 Python path
  - 写 `.libero/config.yaml`

这一层的意义是：把外部环境系统包装成 trainer 能理解的统一接口。

#### `examples/libero/policy/`

策略输入输出的适配层。

主要文件：

- `observation.py`
  - 图像预处理
  - 状态拼接
  - residual observation 构造
  - observation cache
- `action.py`
  - residual action compose
  - action index 选择
  - residual bound / xi 相关逻辑
- `openpi_client.py`
  - 把当前观测编码成 OpenPI 需要的输入格式
  - 调用 OpenPI server 获得 base action chunk

这一层决定“你的环境观测长什么样”和“策略真正吃什么张量”。

#### `examples/libero/scripts/`

Python 主入口。

主要文件：

- `train_residual_sac.py`
  - 主训练入口
  - 当前最核心的训练脚本
- `eval_residual_fast.py`
  - 快速评估入口
- `libero_env_server.py`
  - 环境 RPC server
- `compute_hdf5_stats.py`
  - 计算 normalization 统计量
- `convert_hdf5_to_offline.py`
  - 把 demo 转为 offline replay 格式
- `async_eval_watch.py`
  - 监控异步评估输出

这层可以理解成“把所有模块串起来”的 orchestration 层。

#### `examples/libero/tools/`

Shell 启动脚本层。

作用：

- `serve_env.sh`：启动 env server
- `serve_openpi.sh`：启动 OpenPI server
- `train.sh` / `run_train.sh`：启动训练
- `eval.sh`：启动评估
- `compute_stats.sh` / `convert_offline.sh`：数据预处理

这层主要方便批量跑实验和固定运行方式。

#### `examples/libero/utils/`

一些辅助工具：

- `config_utils.py`：从 Hydra 配置构造 agent、解析图像键、控制维度等
- `logger.py`：JSONL 日志
- `paths.py`：仓库路径解析
- `constants.py`：维度等常量

### 2.3 `examples/RoboTwin/`

这一套和 `examples/libero/` 的结构非常像，也是：

- `conf/`
- `data/`
- `env_wrappers/`
- `policy/`
- `scripts/`
- `tools/`
- `utils/`

它的价值主要有两点：

- 证明当前 residual RL 框架已经可以迁移到另一个任务域
- 提供了另一份“怎么接不同环境系统”的参考实现

其中：

- `env_wrappers/setup.py` 展示了如何对接一个外部大项目的配置与 Python path
- `scripts/robotwin_env_server.py` 展示了一个更复杂的 env server 启动流程

如果以后接 AgiBot，`examples/RoboTwin/` 在“对接外部项目”这件事上，比 `examples/libero/` 更像模板。

### 2.4 `franka_sim/`

这是一个通用的 Panda/Franka MuJoCo 仿真包，不直接等于真实机器人，但对以下场景有帮助：

- 快速验证动作接口
- 做视觉/控制最小闭环测试
- 验证 `serl_launcher` 训练脚本

如果未来 AgiBot 接入前，你想先构建一个“接口完全一致的仿真版本”，这里可以作为参考。

### 2.5 `scripts/`

顶层 `scripts/` 更像历史保留下来的通用脚本和早期训练脚本，比如：

- `async_drq.py`
- `async_sac_state.py`
- `bc_policy.py`
- `train_reward_classifier.py`

它们更多体现的是通用算法或早期范式，不是当前主线入口。当前若要继续扩展 residual RL，优先参考 `examples/libero/` 和 `examples/RoboTwin/`。

### 2.6 `tools/`

顶层 `tools/` 目前偏向模型下载与测试，例如：

- `download.py`
- `download_resnet.py`
- `test_resnet_loading.py`

主要作用是准备视觉 backbone 权重，不承担训练主流程。

### 2.7 `pretrained_models/`

本地缓存的 HuggingFace / ResNet 模型目录。属于运行时资源目录，不是核心代码。

## 3. 当前代码主线是怎么跑起来的

以 `examples/libero/` 为例，当前主线可以概括为：

1. 启动 env server
2. 启动 OpenPI server
3. 训练脚本通过 `RemoteLiberoTaskEnv` 调 env server
4. 训练脚本通过 `OpenPIChunkClient` 调 OpenPI server
5. 当前观测经 `policy/observation.py` 转成 residual policy 输入
6. residual policy 预测增量动作
7. `policy/action.py` 把 `base_action + residual_action` 融合成最终动作
8. transition 写入 replay
9. `serl_launcher` 中的 SAC/DrQ agent 负责 update

也就是说：

- 环境侧差异主要收敛在 `env_wrappers/`
- 观测/动作差异主要收敛在 `policy/`
- 训练算法大部分复用 `serl_launcher/`

这也是为什么新增一个新环境时，最好新建一个独立 example 包，而不是直接改动底层算法库。

## 4. 如果要添加真实机器人环境，例如 AgiBot，推荐怎么做

## 4.1 推荐目录组织

建议新增：

```text
examples/agibot/
├── conf/
├── data/
├── docs/
├── env_wrappers/
├── policy/
├── scripts/
├── tools/
└── utils/
```

如果 AgiBot 的驱动、相机、标定、安全逻辑很多，建议再单独新建一层，例如：

```text
agibot_robot_infra/
```

或者：

```text
examples/agibot/runtime/
```

把所有“只和真机硬件打交道”的代码放进去，而不是塞进 trainer 脚本。

## 4.2 最少需要补哪些组件

### A. 本地环境适配器

建议新增：

- `examples/agibot/env_wrappers/task_env.py`

职责：

- 封装 AgiBot SDK/控制接口
- 暴露统一接口：
  - `reset(seed, episode_id, episode_info=None)`
  - `expert_precheck(seed, episode_id)`
  - `step(action)`
  - `close(clear_cache=False)`
- 暴露统一属性：
  - `current_instruction`
  - `step_limit`
  - `take_action_cnt`

这一层是必须的，因为当前训练脚本默认就是按这个接口调用环境。

### B. 远程环境客户端

建议新增：

- `examples/agibot/env_wrappers/remote_task_env.py`

职责：

- 在训练进程里通过 HTTP RPC 调用真实机器人环境服务
- 保持接口与本地 `task_env.py` 尽量一致

这层是为了让：

- 训练进程可以跑在 `serl_torch` 环境
- 真机控制和硬件依赖跑在另一个更稳定的 runtime 环境

### C. 环境 RPC 服务端

建议新增：

- `examples/agibot/scripts/agibot_env_server.py`

职责：

- 启动 AgiBot 环境服务
- 把 `reset/step/expert_precheck/close` 暴露成 RPC
- 管理机器人实例生命周期

当前 LIBERO 和 RoboTwin 都是这个架构。真实机器人更应该这样做，因为硬件依赖通常更重、更脆弱。

### D. 环境启动/路径配置

建议新增：

- `examples/agibot/env_wrappers/setup.py`

职责：

- 解析 AgiBot SDK 路径
- 配置 `sys.path`
- 读取机器人配置、相机配置、标定配置
- 做启动前检查

如果 AgiBot 外部工程比较大，这一层非常必要。

### E. 观测适配层

建议新增：

- `examples/agibot/policy/observation.py`

职责：

- 将 AgiBot 原始观测转换为 residual RL / OpenPI 需要的统一格式
- 处理图像 resize / rotate / pad
- 组装 state 向量
- 加 observation cache

如果你想尽量复用现在 LIBERO 的 observation 代码，最简单的做法是让 AgiBot env 直接产出这些兼容 key：

- 前视图像：
  - `image` 或 `front_rgb`
- 腕部图像：
  - `wrist_image` 或 `hand_rgb`
- 末端位置：
  - `ee_pos`
- 末端姿态：
  - `ee_ori` 或 quaternion
- gripper：
  - `gripper_states`

这样很多逻辑都能直接复用，而不用大改 trainer。

### F. 动作适配层

建议新增：

- `examples/agibot/policy/action.py`

职责：

- 定义 base action 和 residual action 的组合方式
- 定义 residual 控制哪些维度
- 定义动作幅度限制和 gripper clipping

当前 LIBERO 默认动作是 7 维：

```text
[dx, dy, dz, d_rx, d_ry, d_rz, gripper]
```

如果 AgiBot 也能对齐成这个布局，复用成本最低。

如果 AgiBot 不是这个动作定义，就需要：

- 修改常量
- 修改 action compose
- 修改训练配置里的 `residual.action_dim` / `residual.action_indices`

### G. OpenPI 输入适配

建议新增：

- `examples/agibot/policy/openpi_client.py`

职责：

- 定义如何把 AgiBot observation 编码给 OpenPI
- 如果 prompt、图像 key、state 维度不同，在这里统一处理

如果你用的还是同一个 OpenPI 接口格式，可以直接复用现有实现，再只改 observation adapter。

### H. 训练与评估入口

建议新增：

- `examples/agibot/scripts/train_residual_sac.py`
- `examples/agibot/scripts/eval_residual_fast.py`

最务实的做法不是从零写，而是从 `examples/libero/scripts/train_residual_sac.py` 复制一份开始改。

因为你当前已经在 LIBERO 版本里补了：

- async actor-learner
- replay prefetch
- observation cache
- profiling
- async checkpoint writer
- OpenPI prefetch

这些都已经是有价值的工程能力，直接继承最省事。

### I. 配置文件

建议新增：

- `examples/agibot/conf/train_residual_sac.yaml`
- `examples/agibot/conf/eval_residual_fast.yaml`

职责：

- 指定 AgiBot 的动作维度
- 指定图像 key
- 指定 env server 地址
- 指定机器人安全参数
- 指定 offline 数据路径
- 指定 profiling、checkpoint、async learner 等开关

### J. 启动脚本

建议新增：

- `examples/agibot/tools/serve_env.sh`
- `examples/agibot/tools/serve_openpi.sh`
- `examples/agibot/tools/train.sh`
- `examples/agibot/tools/eval.sh`

这一步不是必须，但对稳定运行非常重要，尤其真机常常需要固定环境变量、CUDA 设备、SDK 路径。

## 4.3 真实机器人特有、当前仓库里已经没有的组件

由于旧的 `serl_robot_infra/` 已经被删除，现在仓库里没有通用的真机控制基础设施。要接 AgiBot，除了上面那些 residual RL 组件，你还需要补一层“机器人 runtime”。

最少应包含：

- 机械臂控制驱动
  - 发送笛卡尔增量或关节动作
  - 查询当前状态
- 夹爪驱动
  - 开合控制
  - 当前开合量反馈
- 相机采集
  - 头部相机
  - 腕部相机
  - 帧时间戳
- 标定与坐标变换
  - 相机到机器人坐标
  - tool center point 定义
  - 末端姿态表示转换
- 安全模块
  - 限速
  - 限位
  - 急停
  - watchdog
  - reset/recover
- 观测同步
  - 相机帧、机器人状态、gripper 状态对齐
- 数据采集
  - 录演示
  - 保存 offline 数据
  - 保存统计量
- 人工接管/teleop
  - 用于收集示教和做故障恢复

这部分建议单独放在：

- `agibot_robot_infra/`

或者：

- `examples/agibot/runtime/`

不要和 trainer 主逻辑耦合在一起。

## 4.4 最推荐的接入顺序

建议按下面顺序做，这样风险最小：

1. 先定义统一 observation/action 契约
   - 明确图像 key、state key、action 维度
2. 先写 `task_env.py`
   - 直接本地调 AgiBot SDK，确认 `reset/step` 能跑通
3. 再写 `agibot_env_server.py` 和 `remote_task_env.py`
   - 把环境变成远程可调用
4. 复用 `examples/libero/policy/observation.py` / `action.py`
   - 不够用时再 fork
5. 从 `examples/libero/scripts/train_residual_sac.py` 复制训练入口
   - 先跑通最小在线训练
6. 再补 offline 数据转换、normalizer、评估脚本
7. 最后再补 profiling、安全策略、恢复逻辑

## 4.5 一个最小可复用接口建议

如果你想让 AgiBot 尽量少改当前 trainer，建议把环境输出统一成下面这种格式：

```python
obs = {
    "front_rgb": np.ndarray,      # H x W x 3, uint8
    "hand_rgb": np.ndarray,       # H x W x 3, uint8
    "ee_pos": np.ndarray,         # (3,)
    "ee_ori": np.ndarray,         # (3,) axis-angle，或提供 quaternion
    "gripper_states": np.ndarray, # (1,) or (2,)
}
```

动作尽量统一成：

```python
action = np.ndarray(shape=(7,), dtype=np.float32)
# [dx, dy, dz, d_rx, d_ry, d_rz, gripper]
```

这样你可以最大化复用：

- `examples/libero/policy/observation.py`
- `examples/libero/policy/action.py`
- `examples/libero/policy/openpi_client.py`
- `examples/libero/scripts/train_residual_sac.py`

## 5. 总结

当前仓库已经不再包含旧的 Franka 真机基础设施，主结构可以理解为：

- `serl_launcher/` 负责算法底座
- `examples/libero/` 负责当前最完整的 residual RL 工程实现
- `examples/RoboTwin/` 提供另一套任务接入模板
- `franka_sim/` 提供仿真环境能力

如果要接入 AgiBot，最推荐的方式是：

1. 新建 `examples/agibot/`
2. 复用 `serl_launcher/` 的算法层
3. 复用 `examples/libero/` 的训练主线
4. 单独新增 AgiBot 的 `env_wrappers/`、`policy/`、`scripts/`、`tools/`
5. 单独补一层真机 runtime，用于控制、相机、标定与安全

从工程角度看，真正必须新写的不是 SAC 本身，而是下面这几块：

- 环境适配
- 观测适配
- 动作适配
- 真机 runtime
- 配置与启动脚本

只要这几层边界设计得干净，后面无论接 AgiBot、Franka、UR、双臂系统，训练主干都可以持续复用。
