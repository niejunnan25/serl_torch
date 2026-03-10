# 在真实机器人上运行 Franka 机械臂

我们展示了如何在 4 个不同任务中将 SERL 用于真实机械臂：插销插入（Peg Insertion）、PCB 组件插入（PCB Component Insertion）、线缆布线（Cable Routing）和物体搬运（Object Relocation）。我们还提供了详细步骤来复现插销插入任务，作为整个 SERL 套件的安装与运行测试。

在真实机器人上运行时，需要单独的 gym 环境。我们的示例中将 gym 环境作为机器人服务端的客户端。机器人服务端是一个 Flask 服务，通过 ROS 向机器人发送命令。gym 环境通过 post 请求与机器人服务端通信。

![](./images/robot_infra_interfaces.png)


### `serl_robot_infra` 安装

请按照 `serl_robot_infra` 中的 [README](../serl_robot_infra/README-ch.md) 完成安装和机器人基础操作说明。其中包含基于阻抗控制的 [serl_franka_controllers](https://github.com/rail-berkeley/serl_franka_controllers) 安装说明。

安装完成后，你应该能够运行机器人服务端，并与 gym `franka_env`（硬件）进行交互。

> 注意：以下示例代码不能直接开箱即用，因为你仍需要自定义数据、检查点和机器人环境。我们提供这些代码作为在真实机器人上使用 SERL 的参考。建议按任务顺序逐步学习，从第一个任务（插销插入）到最后一个任务（料箱搬运），再根据你的需求修改代码。

## 1. 插销插入 📍

![](./images/peg.png)

> 示例位置：[examples/async_peg_insert_drq/](../examples/async_peg_insert_drq/)

> 环境和默认配置位于 `serl_robot_infra/franka_env/envs/peg_env/`

> `franka_env.envs.wrappers.SpacemouseIntervention` gym wrapper 提供了使用 spacemouse 对机器人进行干预的能力。它适合用于示范采集、机器人测试，以及验证训练环境是否按预期工作。

插销插入任务非常适合用来开始在真实机器人上运行 SERL。在最简单设置下，策略可在单卡 GPU 上 30 分钟内收敛并达到 100% 成功率，因此非常适合快速排查搭建问题。以下流程假设你有一台 Franka 机械臂、一个 Robotiq Hand-E 夹爪和两台 RealSense D405 相机。

### 操作流程
1. 从 [FMB](https://functional-manipulation-benchmark.github.io/files/index.html) 的 **Single-Object Manipulation Objects** 部分选择并 3D 打印 (1) 一个 **Assembly Object** 和 (1) 对应的 **Assembly Board**。将板固定在工作台，并让夹爪抓住插销。
2. 为 RealSense D405 3D 打印 (2) 个腕部相机支架，并安装到 Robotiq 夹爪的螺纹位。基于 [peg_env/config.py](../serl_robot_infra/franka_env/envs/peg_env/config.py) 创建你自己的配置，并在 `REALSENSE_CAMERAS` 中更新相机序列号。
3. 通过编辑 `Desk > Settings > End-effector > Mechnical Data > Mass` 补偿腕部相机重量。
4. 解锁机器人并在 Desk 中激活 FCI。然后运行以下命令启动 franka_server：
    ```bash
    python serl_robot_infra/robot_servers/franka_server.py --gripper_type=<Robotiq|Franka|None> --robot_ip=<robot_IP> --gripper_ip=<[Optional] Robotiq_gripper_IP>
    ```
    这会启动阻抗控制器与 Flask 服务端，准备接收请求。
5. 该任务的奖励通过判断末端执行器位姿是否匹配固定目标位姿来给出。先通过 `curl -X POST http://127.0.0.1:5000/close_gripper` 抓住目标插销，然后手动将机械臂移动到插销插入板中的姿态。使用 `curl -X POST http://127.0.0.1:5000/getpos_euler` 打印当前位姿，并将测得的末端位姿填入 [peg_env/config.py](../serl_robot_infra/franka_env/envs/peg_env/config.py) 的 `TARGET_POSE`。

    **注意：目标位姿下请确保腕关节居中（远离关节极限），且 z 轴欧拉角为正，以避免不连续性。**

6. 在配置文件中将 `RANDOM_RESET` 设为 `False` 以加速训练。注意，只有当它设为 `True` 时，策略才会泛化到任意板位姿，但建议先在基础任务跑通后再尝试。
7. 使用 spacemouse 录制 20 条示范轨迹。
    ```bash
    cd examples/async_peg_insert_drq
    python record_demo.py
    ```
    轨迹会保存在 `examples/async_peg_insert_drq/peg_insertion_20_trajs_{UUID}.pkl`。
8. 修改 `run_learner.sh` 和 `run_actor.sh` 中的 `demo_path` 与 `checkpoint_path`。随后同时运行 learner 与 actor，使用收集到的 demos 训练 RL agent。
    ```bash
    bash run_learner.sh
    bash run_actor.sh
    ```
9. 如果过程无误，在关闭 `RANDOM_RESET` 时策略应在 30 分钟内达到 100% 成功率；开启 `RANDOM_RESET` 时约 60 分钟。
10. 检查点会自动保存。可在 `run_actor.sh` 中设置 `--eval_checkpoint_step=CHECKPOINT_NUMBER_TO_EVAL` 和 `--eval_n_trajs=N_TIMES_TO_EVAL` 来评估，然后运行：
    ```bash
    bash run_actor.sh
    ```
    若策略在 `RANDOM_RESET` 下训练，测试时移动板位置后也应能完成插销插入。


以插销插入任务为例，我们对环境的封装如下。gym wrappers 的可组合性让我们可以方便地给环境增加或移除功能。（[代码](../examples/async_peg_insert_drq/async_drq_randomized.py)）

```python
env = gym.make('FrankaPegInsert-Vision-v0')  # 创建 gym 环境
env = GripperCloseEnv(env)         # 插销任务始终保持夹爪闭合
env = SpacemouseIntervention(env)  # 使用 spacemouse 干预机器人
env = RelativeFrame(env)           # 将 TCP 绝对参考系转换为相对参考系
env = Quat2EulerWrapper(env)       # 将旋转从四元数转换为欧拉角
env = SERLObsWrapper(env)          # 将观测转换为 SERL 格式
env = ChunkingWrapper(env)         # 对观测做 chunking
env = RecordEpisodeStatistics(env) # 记录 episode 统计信息
```


### 2. PCB 组件插入 🖥️

![](./images/pcb.png)

> 示例位置：[examples/async_pcb_insert_drq/](../examples/async_pcb_insert_drq/)

> 环境和默认配置位于 `serl_robot_infra/franka_env/envs/pcb_env/`

与插销插入类似，本任务的奖励同样通过判断末端执行器位姿是否匹配固定目标位姿来给出。将测得的末端位姿更新到 [peg_env/config.py](../serl_robot_infra/franka_env/envs/peg_env/config.py) 的 `TARGET_POSE`。

这里先用机器人录制示范轨迹，再运行 learner 与 actor 节点。
```bash
# 录制示范轨迹
python record_demo.py

# 运行 learner 与 actor
bash run_learner.sh
bash run_actor.sh
```

还提供了以 BC 作为策略的基线。训练 BC 只需执行：
```bash
python3 examples/bc_policy.py ....TODO_ADD_ARGS.....
```

运行 BC 策略只需执行：
```bash
bash run_bc.sh
```

### 3. 线缆布线 🔌

![](./images/cable.png)

> 示例位置：[examples/async_cable_routing_drq/](../examples/async_cable_routing_drq/)

> 环境和默认配置位于 `serl_robot_infra/franka_env/envs/cable_env/`

在线缆布线任务中，我们提供了一个基于图像的奖励分类器示例。它替代了依赖 `config.py` 中已知 `TARGET_POSE` 的硬编码奖励判断器。该图像奖励分类器基于预训练 ResNet10，再训练成判断线缆是否布线成功的分类器。分类器使用成功与失败示例的示范轨迹训练。

```bash
# 注意：请填入自定义路径以训练奖励分类器
python train_reward_classifier.py \
    --classifier_ckpt_path CHECKPOINT_OUTPUT_DIR \
    --positive_demo_paths PATH_TO_POSITIVE_DEMO1.pkl \
    --positive_demo_paths PATH_TO_POSITIVE_DEMO2.pkl \
    --negative_demo_paths PATH_TO_NEGATIVE_DEMO1.pkl \
```

奖励分类器以 gym wrapper `franka_env.envs.wrapper.BinaryRewardClassifier` 的形式使用。该 wrapper 会对当前观测进行分类，若成功则返回奖励 1，否则返回 0。

然后在 actor 节点的 BC 与 DRQ 策略中使用该奖励分类器，其路径通过 `run_bc.sh` 和 `run_actor.sh` 中的 `--reward_classifier_ckpt_path` 参数传入。


### 4. 物体搬运 🗑️

![](./images/forward.png)

![](./images/backward.png)

> 示例位置：[examples/async_bin_relocation_fwbw_drq/](../examples/async_bin_relocation_fwbw_drq/)

> 环境和默认配置位于 `serl_robot_infra/franka_env/envs/bin_env/`

这个料箱搬运示例展示了前向与后向策略的用法。双任务建模对 RL 任务很有帮助，能协助机器人完成“重置”。在该场景里，机器人需要将物体在两个料箱间搬运：前向策略将物体从右箱移动到左箱，后向策略将物体从左箱移回右箱。

1. 录制示范轨迹

已提供多个工具脚本用于录制示范轨迹（例如 `record_demo.py` 用于 RLPD，`record_transitions.py` 用于训练奖励分类器，`reward_bc_demos.py` 用于 BC 策略）。注意前向与后向轨迹需要分别录制不同示范数据。

2. 奖励分类器

与线缆布线示例类似，需要分别为前向和后向策略训练两个奖励分类器。由于观测同时包含腕部相机和前视相机，我们使用 `FrontCameraWrapper(env)` 仅向奖励分类器提供前视图像。

```bash
# 注意：请填入自定义路径以训练前向与后向策略的奖励分类器
python train_reward_classifier.py \
    --classifier_ckpt_path CHECKPOINT_OUTPUT_DIR \
    --positive_demo_paths PATH_TO_POSITIVE_DEMO1.pkl \
    --positive_demo_paths PATH_TO_POSITIVE_DEMO2.pkl \
    --negative_demo_paths PATH_TO_NEGATIVE_DEMO1.pkl \
```

随后在 actor 节点的 BC 与 DRQ 策略中使用这些奖励分类器，在 `run_actor.sh` 中通过 `--fw_reward_classifier_ckpt_path` 与 `--bw_reward_classifier_ckpt_path` 传入检查点路径。若要与 BC 基线比较，则在 `run_bc.sh` 中通过 `--reward_classifier_ckpt_path` 传入分类器路径。

3. 运行 2 个 learner + 1 个 actor（2 套策略）

最后，两个 learner 节点将分别学习前向与后向策略。actor 节点在 RL 训练过程中会在前向与后向策略之间切换，并使用各自对应的奖励分类器。

```bash
bash run_actor.sh

# 运行 2 个 learner
bash run_fw_learner.sh
bash run_bw_learner.sh
```
