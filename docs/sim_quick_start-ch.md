# 在仿真中快速开始使用 SERL

这是一个用于 SERL 训练的最小 Mujoco 仿真环境。环境由一台 Panda 机械臂和一个立方体组成，目标是将立方体抬升到目标位置。该环境基于 `franka_sim` 与 `gym` 接口实现。

![](./images/franka_sim.png)

## 安装

**安装 Franka Sim 库**
```bash
    cd franka_sim
    pip install -e .
    pip install -r requirements.txt
```

可通过 `python franka_sim/franka_sim/test/test_gym_env_human.py` 测试 `franka_sim` 是否正常运行。

开始前请确保基于 `franka_sim` 的仿真环境工作正常。

*注意：如果你做离屏渲染，请将 `MUJOCO_GL` 设为 `egl`。
可执行 `export MUJOCO_GL=egl`，并记得在脚本中将渲染参数设为 False。
如果出现 `Cannot initialize a EGL device display due to GLIBCXX not found` 错误，可尝试执行 `conda install -c conda-forge libstdcxx-ng`（[参考](https://stackoverflow.com/a/74132234)）*

可选安装 `tmux`：`sudo apt install tmux`

## 1. 基于状态观测的训练示例

**✨ 一行命令启动（需要 `tmux`）✨**
```bash
bash examples/async_sac_state_sim/tmux_launch.sh
```

关闭 tmux 会话：`tmux kill-session -t serl_session`。

### 不使用一行 tmux 启动脚本

你也可以在两个不同终端中分别运行命令。

```bash
cd examples/async_sac_state_sim
```

运行 learner 节点：
```bash
bash run_learner.sh
```

运行带渲染窗口的 actor 节点：
```bash
# 如果运行在不同机器上，请添加 --ip x.x.x.x
bash run_actor.sh
```

你也可以把 learner 与 actor 分别部署在不同机器上。例如 learner 节点在 `ip=x.x.x.x` 的 PC 上运行时，可以在另一台可访问该 IP 的机器上启动 actor，并在 `run_actor.sh` 的命令里添加 `--ip x.x.x.`。

移除 `run_learner.sh` 中的 `--debug` 标志，可将训练统计上传到 `wandb`。

## 2. 基于图像观测的训练示例

**✨ 一行命令启动（需要 `tmux`）✨**

```bash
bash examples/async_drq_sim/tmux_launch.sh
```

### 不使用一行 tmux 启动脚本

你也可以在两个不同终端中分别运行命令。

```bash
cd examples/async_drq_sim

# 如需使用预训练 ResNet 权重，请下载
wget https://github.com/rail-berkeley/serl/releases/download/resnet10/resnet10_params.pkl
```

运行 learner 节点：
```bash
bash run_learner.sh
```

运行带渲染窗口的 actor 节点：
```bash
# 如果运行在不同机器上，请添加 --ip x.x.x.x
bash run_actor.sh
```

## 3. 基于图像观测并使用 20 条示范轨迹的训练示例

**✨ 一行命令启动（需要 `tmux`）✨**
```bash
bash examples/async_drq_sim/tmux_rlpd_launch.sh
```

### 不使用一行 tmux 启动脚本

你也可以在两个不同终端中分别运行命令。

```bash
cd examples/async_drq_sim

# 如需使用预训练 ResNet 权重，请下载
# 目前仅支持手动下载，仓库公开后会支持自动下载
wget https://github.com/rail-berkeley/serl/releases/download/resnet10/resnet10_params.pkl

# 下载 20 条示范轨迹
wget \
https://github.com/rail-berkeley/serl/releases/download/franka_sim_lift_cube_demos/franka_lift_cube_image_20_trajs.pkl
```

运行 learner 节点，并在 `--demo_path` 参数中提供示范轨迹路径。
```bash
bash run_learner.sh --demo_path franka_lift_cube_image_20_trajs.pkl
```

运行带渲染窗口的 actor 节点：
```bash
# 如果运行在不同机器上，请添加 --ip x.x.x.x
bash run_actor.sh
```

## 使用 RLDS logger 保存与加载轨迹

这提供了一种在 SERL 训练中保存与加载轨迹的方法。轨迹以 [Tensorflow RLDS dataset](https://github.com/google-research/rlds) 格式保存和加载。该标准与 [RTX datasets](https://robotics-transformer-x.github.io/) 兼容，因此潜在可用于其他机器人学习任务。

### 安装

这需要额外安装 `oxe_envlogger`：
```bash
git clone git@github.com:rail-berkeley/oxe_envlogger.git
cd oxe_envlogger
pip install -e .
```

### 用法

**保存轨迹**

以上述示例为例，可通过传入 `rlds_logger_path` 参数将 replay buffer 数据保存到指定路径。

```bash
./run_learner.sh --log_rlds_path /path/to/save
```

数据将以如下结构保存：

```bash
 - /path/to/save
    - dataset_info.json
    - features.json
    - serl_rlds_dataset-train.tfrecord-00000
    - serl_rlds_dataset-train.tfrecord-00001
    ....
```

**加载轨迹**

同样在上述示例中，可通过传入 `preload_rlds_path` 参数从指定路径加载 replay buffer 数据。

```bash
./run_learner.sh --preload_rlds_path /path/to/load
```

这与 `examples/async_rlpd_drq_sim/run_learner.sh` 脚本类似，该脚本使用 `--demo_path` 参数加载 `.pkl` 格式离线示范轨迹。


### 故障排查

1. 如果出现内存不足（Out of Memory）错误，可在 `run_learner.sh` 中通过添加 `--batch_size` 参数减小 batch size。例如：`bash run_learner.sh --batch_size 64`。
2. 如果提供的离线 RLDS 数据报错，通常表示该数据与当前 SERL 格式不兼容。你可以在 `examples/async_drq_sim/asyn_drq_sim.py` 中提供自定义数据变换函数 `data_transform(data, metadata) -> data`。
