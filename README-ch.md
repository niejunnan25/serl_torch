# SERL：面向样本高效机器人强化学习的软件套件

![](https://github.com/rail-berkeley/serl/workflows/pre-commit/badge.svg)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Static Badge](https://img.shields.io/badge/Project-Page-a)](https://serl-robot.github.io/)
[![Discord](https://img.shields.io/discord/1302866684612444190?label=Join%20Us%20on%20Discord&logo=discord&color=7289da)](https://discord.gg/G4xPJEhwuC)


![](./docs/images/tasks-banner.gif)

**项目主页：[https://serl-robot.github.io/](https://serl-robot.github.io/)**


**🔴 重要 🔴：本仓库正在弃用。请查看我们的新项目 HIL-SERL：[https://hil-serl.github.io/](https://hil-serl.github.io/)**


SERL 提供了一组库、环境封装器和示例，用于训练机器人操作任务的强化学习策略。下文将介绍如何使用 SERL，并通过示例进行说明。

🎬：[SERL 视频](https://www.youtube.com/watch?v=Um4CjBmHdcw)，以及关于样本高效 RL 的[补充视频](https://www.youtube.com/watch?v=17NrtKHdPDw)。

**目录**
- [SERL：面向样本高效机器人强化学习的软件套件](#serl面向样本高效机器人强化学习的软件套件)
  - [安装](#安装)
  - [概览与代码结构](#概览与代码结构)
  - [在仿真中快速开始使用 SERL](#在仿真中快速开始使用-serl)
  - [在真实机器人上运行 Franka 机械臂](#在真实机器人上运行-franka-机械臂)
  - [贡献](#贡献)
  - [引用](#引用)

## 重要更新
#### 2024 年 6 月 24 日
对于使用 SERL 执行需要控制夹爪的任务（例如抓取物体）的用户，我们强烈建议在夹爪动作变化上加入一个小惩罚项，这将显著提升训练速度。
详情请参考：[PR #65](https://github.com/rail-berkeley/serl/pull/65)。

此外，我们也建议在训练期间除加载离线示范数据外，同时在线进行人工干预。如果你有 Franka 机器人和 SpaceMouse，这甚至可以简单到在训练时触碰 SpaceMouse。

#### 2024 年 4 月 25 日
我们修复了干预动作坐标系中的一个重大问题。请参见发布版本 [v0.1.1](https://github.com/rail-berkeley/serl/releases/tag/v0.1.1)，并将你的代码更新到 main 分支。

## 安装
1. **创建 Conda 环境：**
    运行以下命令创建环境：
    ```bash
    conda create -n serl python=3.10
    ```

2. **按如下方式安装 PyTorch：**
    - CPU：
        ```bash
        pip install --upgrade torch torchvision
        ```

    - NVIDIA GPU（CUDA 12.1 示例）：
        ```bash
        pip install --upgrade torch torchvision --index-url https://download.pytorch.org/whl/cu121
        ```
    - 请根据你的 CUDA/系统版本参考 [PyTorch 官方安装页面](https://pytorch.org/get-started/locally/)。

3. **安装 serl_launcher**
    ```bash
    cd serl_launcher
    pip install -e .
    pip install -r requirements.txt
    ```

## 概览与代码结构

SERL 提供了一套通用库，帮助用户训练机器人操作任务的强化学习策略。RL 实验的主要结构包含 actor 节点和 learner 节点，两者都与机器人 gym 环境交互。两个节点异步运行，数据通过网络由 actor 发送到 learner，通信基于 [agentlace](https://github.com/youliangtan/agentlace)。learner 会周期性地将策略同步给 actor。这种设计为并行训练与推理提供了灵活性。

<p align="center">
  <img src="./docs/images/software_design.png" width="80%"/>
</p>

**代码结构表**

| 代码目录 | 说明 |
| --- | --- |
| [serl_launcher](https://github.com/rail-berkeley/serl/blob/main/serl_launcher) | SERL 主体代码 |
| [serl_launcher.agents](https://github.com/rail-berkeley/serl/blob/main/serl_launcher/serl_launcher/agents/) | Agent 策略（如 DRQ、SAC、BC） |
| [serl_launcher.wrappers](https://github.com/rail-berkeley/serl/blob/main/serl_launcher/serl_launcher/wrappers) | Gym 环境封装器 |
| [serl_launcher.data](https://github.com/rail-berkeley/serl/blob/main/serl_launcher/serl_launcher/data) | Replay Buffer 与数据存储 |
| [serl_launcher.vision](https://github.com/rail-berkeley/serl/blob/main/serl_launcher/serl_launcher/vision) | 视觉相关模型与工具 |
| [franka_sim](./franka_sim) | Franka 的 Mujoco 仿真 gym 环境 |
| [serl_robot_infra](./serl_robot_infra/) | 真实机器人运行所需基础设施 |
| [serl_robot_infra.robot_servers](https://github.com/rail-berkeley/serl/blob/main/serl_robot_infra/robot_servers/) | 通过 ROS 向机器人发送命令的 Flask 服务 |
| [serl_robot_infra.franka_env](https://github.com/rail-berkeley/serl/blob/main/serl_robot_infra/franka_env/) | 真实 Franka 机器人的 Gym 环境 |

## 在仿真中快速开始使用 SERL

我们提供了一个仿真环境，方便你在 Franka 机器人上试用 SERL。

查看：[在仿真中快速开始使用 SERL](/docs/sim_quick_start-ch.md)
 - [基于状态观测的训练示例](/docs/sim_quick_start-ch.md#1-基于状态观测的训练示例)
 - [基于图像观测的训练示例](/docs/sim_quick_start-ch.md#2-基于图像观测的训练示例)
 - [基于图像观测并使用 20 条示范轨迹的训练示例](/docs/sim_quick_start-ch.md#3-基于图像观测并使用-20-条示范轨迹的训练示例)

## 在真实机器人上运行 Franka 机械臂

我们提供了在真实 Franka 机器人上使用 SERL 运行 RL 策略的分步指南。

查看：[在真实机器人上运行 Franka 机械臂](/docs/real_franka-ch.md)
 - [插销插入 📍](/docs/real_franka-ch.md#1-插销插入-)
 - [PCB 组件插入 🖥️](/docs/real_franka-ch.md#2-pcb-组件插入-️)
 - [线缆布线 🔌](/docs/real_franka-ch.md#3-线缆布线-)
 - [物体搬运 🗑️](/docs/real_franka-ch.md#4-物体搬运-️)

## 贡献

欢迎为本仓库做贡献。如果你对代码库有改进，请 fork 后提交 PR。在提交 PR 之前，请先运行 `pre-commit run --all-files`，确保代码格式正确。

## 引用

如果你在研究中使用了这份代码，请引用我们的论文：

```bibtex
@misc{luo2024serl,
      title={SERL: A Software Suite for Sample-Efficient Robotic Reinforcement Learning},
      author={Jianlan Luo and Zheyuan Hu and Charles Xu and You Liang Tan and Jacob Berg and Archit Sharma and Stefan Schaal and Chelsea Finn and Abhishek Gupta and Sergey Levine},
      year={2024},
      eprint={2401.16013},
      archivePrefix={arXiv},
      primaryClass={cs.RO}
}
```
