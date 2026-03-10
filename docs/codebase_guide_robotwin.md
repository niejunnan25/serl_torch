# SERL 代码仓库解读与 RobotWin 仿真接入指南

本文档面向“把 `serl` 当作主代码仓库二次开发”的场景，重点回答两个问题：

1. 这个仓库现在的结构和训练链路是什么。
2. 如果要新增 **RobotWin 仿真环境**，应该怎么接入、怎么组织工程。

## 1. 仓库结构总览

仓库根目录：`/Users/niejunnan.25/Documents/codebase/serl`

核心目录：

- `serl_launcher/`：算法与训练框架核心（JAX + Agentlace）
- `franka_sim/`：MuJoCo 仿真环境（Panda pick）
- `serl_robot_infra/`：真实机器人基础设施（ROS + Flask + gym env client）
- `examples/`：训练入口脚本（actor/learner、demo、reward classifier）
- `docs/`：使用文档（sim 与 real）

建议把仓库理解为“**训练框架层 + 环境层 + 任务脚本层**”：

- 训练框架层：`serl_launcher`
- 环境层：`franka_sim`（sim）和 `serl_robot_infra/franka_env`（real）
- 任务脚本层：`examples/*`

## 2. 当前训练链路（以 async 训练为例）

`examples/async_*/*.py` 一般都遵循同一个模式：

1. 构建环境并叠 wrapper
2. 构建 agent（DrQ/SAC/BC/VICE）
3. 启动异步 actor/learner
4. actor 采样并通过 Agentlace 发送数据
5. learner 更新参数并发布最新网络

典型文件：

- `examples/async_drq_sim/async_drq_sim.py`
- `examples/async_peg_insert_drq/async_drq_randomized.py`

关键模块位置：

- agent 工厂：`serl_launcher/serl_launcher/utils/launcher.py`
- 算法实现：`serl_launcher/serl_launcher/agents/continuous/*.py`
- replay buffer：`serl_launcher/serl_launcher/data/*.py`
- 通用 wrapper：`serl_launcher/serl_launcher/wrappers/*.py`

## 3. 扩展新仿真环境的最小接口要求

要被 SERL 训练脚本稳定使用，新的仿真环境建议满足：

1. `gym`/`gymnasium` 风格 API：`reset()`, `step()`, `action_space`, `observation_space`
2. `observation_space` 尽量是如下结构：
   - `{"state": Dict(...), "images": Dict(...)} 或至少 {"state": ...}`
3. `action_space` 为连续 `Box`
4. `step` 返回奖励与终止信号语义稳定

> 当前 `serl` 主干仍以 `gym` 写法为主（不是 `gymnasium`），新增模块建议先与现有脚本保持一致，避免一次性改太多层。

## 4. 如何新增 RobotWin 仿真（推荐工程方案）

建议采用“**新增独立仿真包 + 新增 example 任务目录**”的做法，不直接侵入 `franka_sim`。

### 4.1 新增目录结构

在仓库根目录新增：

```text
robotwin_sim/
  setup.py
  requirements.txt
  robotwin_sim/
    __init__.py
    envs/
      __init__.py
      robotwin_task_gym_env.py
    adapters/
      robotwin_to_serl_obs.py
```

作用：

- `robotwin_sim/envs/*`：封装 RobotWin 原生环境
- `robotwin_sim/adapters/*`：观测/动作转换，和 SERL 训练输入对齐

### 4.2 注册环境 ID

在 `robotwin_sim/robotwin_sim/__init__.py` 中注册 gym env：

```python
from gym.envs.registration import register

register(
    id="RobotWinTask-v0",
    entry_point="robotwin_sim.envs:RobotWinTaskGymEnv",
    max_episode_steps=200,
)

register(
    id="RobotWinTaskVision-v0",
    entry_point="robotwin_sim.envs:RobotWinTaskGymEnv",
    max_episode_steps=200,
    kwargs={"image_obs": True},
)
```

### 4.3 实现 RobotWin 环境封装类

在 `robotwin_sim/envs/robotwin_task_gym_env.py` 里做两件事：

1. 把 RobotWin 原始 obs 转成 SERL 友好字典
2. 把动作范围统一为 `[-1, 1]`（在 env 内部做缩放）

建议输出格式：

```python
obs = {
    "state": {
        "robot/joint_pos": ...,
        "robot/joint_vel": ...,
        "tcp_pos": ...,
        "tcp_vel": ...,
    },
    "images": {
        "front": ...,
        "wrist": ...,
    }
}
```

### 4.4 新增训练入口（example）

新增目录：

```text
examples/async_robotwin_drq/
  async_drq_robotwin.py
  run_actor.sh
  run_learner.sh
  tmux_launch.sh
```

实现策略：

- 初版直接复制 `examples/async_drq_sim/async_drq_sim.py`
- 仅替换环境名和必要 wrapper
- 先跑通单任务，再泛化到多个 RobotWin task

### 4.5 wrapper 组合建议

如果 RobotWin 的 obs 不完全兼容 SERL，可以使用：

1. 自定义 adapter wrapper（先把 obs 结构规范化）
2. `SERLObsWrapper`
3. `ChunkingWrapper`
4. `RecordEpisodeStatistics`

示例：

```python
env = gym.make("RobotWinTaskVision-v0")
env = RobotWinToSERLObsWrapper(env)   # 你新增的适配层
env = SERLObsWrapper(env)
env = ChunkingWrapper(env, obs_horizon=1, act_exec_horizon=None)
env = RecordEpisodeStatistics(env)
```

## 5. 你需要改动的文件清单（最小可运行版本）

### 必做

- 新增 `robotwin_sim/*` 全套
- 新增 `examples/async_robotwin_drq/*`
- 在根 README 或 docs 增加 RobotWin 快速开始

### 可选（建议）

- 在 `serl_launcher/serl_launcher/utils/launcher.py` 增加 `make_robotwin_drq_agent(...)`（若需要不同网络超参）
- 在 `serl_launcher/serl_launcher/wrappers/` 增加复用型 `robotwin_obs_wrapper.py`

## 6. 构建与运行建议

### 6.1 安装顺序

```bash
cd serl/serl_launcher
pip install -e .
pip install -r requirements.txt

cd ../../robotwin_sim
pip install -e .
pip install -r requirements.txt
```

### 6.2 冒烟测试

先做最小测试，确认环境注册与 step/reset 正常：

```bash
python - <<'PY'
import gym
import robotwin_sim

env = gym.make("RobotWinTaskVision-v0")
obs = env.reset()
print(type(obs))
a = env.action_space.sample()
out = env.step(a)
print('step ok, len=', len(out))
PY
```

### 6.3 启动训练

先从 sim 单机开始（不接真机链路）：

```bash
cd examples/async_robotwin_drq
bash run_learner.sh
bash run_actor.sh
```

## 7. 工程组织建议（避免后期难维护）

1. **不要把 RobotWin 逻辑散落进 `serl_launcher` 算法层**。
2. RobotWin 相关代码集中在 `robotwin_sim/` 与 `examples/async_robotwin_*`。
3. 任务配置尽量参数化（camera key、reward、action scale），避免复制脚本。
4. 先保证一个 task 稳定收敛，再扩展 task family。

## 8. 常见坑与规避

1. **obs 结构不统一**：先写 adapter，把 key/shape 固定。
2. **动作尺度不一致**：统一外部 `[-1, 1]`，内部再映射。
3. **done/terminated 语义混乱**：统一在 env 层处理。
4. **图像 dtype/shape 不一致**：强制 `uint8` + 固定分辨率。
5. **训练脚本复制过多**：尽快抽 config，减少重复。

## 9. 推荐落地顺序（两周版本）

1. 第 1-2 天：完成 `robotwin_sim` 最小 env 封装 + 注册。
2. 第 3-4 天：复制 `async_drq_sim` 改成 `async_robotwin_drq`，跑通 1k steps。
3. 第 5-7 天：补齐日志、checkpoint、eval。
4. 第 8-10 天：调 reward 与动作尺度，跑出第一条有效学习曲线。
5. 第 11-14 天：扩展第二个 task，验证抽象是否合理。

---

如果你要继续，我可以在下一步直接帮你生成一个 **可运行的 `robotwin_sim` 模板目录和骨架代码**（含 `setup.py`、env class、example 脚本）。
