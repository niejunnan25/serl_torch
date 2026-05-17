# AgiBot Real Robot Bring-up

这份 README 只说明一件事：把这个仓库同步到一台新的机器人控制机后，如何安装依赖，并确认代码已经能连上 AgiBot 真机。

完整 residual RL 训练、learner / actor 启动和 backfill 配置见本文后面的标准启动命令；新机器先把下面的连接测试跑通，再启动训练。

## 前提

新的机器需要满足：

- Linux，`x86_64` 或 `aarch64`
- Python `3.10`
- 能接入机器人网络，机器上至少有一个 `10.42.0.*` 地址
- 已经拿到本仓库完整代码
- `examples/agibot_real/vendor/a2d_sdk/wheels/` 下有 AgiBot SDK wheels
- 如果要启动带 ROS forwarder 的 `robot-service`，还需要 forwarder bundle，例如 `forwarder_x86_v1.7.0.tar.gz` 或已解压的 forwarder 目录

连接测试本身不需要启动 JoyRA / OpenPI policy server，也不需要启动 learner。

## 1. 安装 Python 环境

下面假设仓库路径是 `/path/to/serl_torch`，请按实际路径替换。

```bash
cd /path/to/serl_torch

conda create -n serl_torch python=3.10 -y
conda activate serl_torch

python -m pip install --upgrade pip setuptools wheel
pip install -r serl_launcher/requirements.txt
pip install -e ./serl_launcher
pip install -e .
```

AgiBot actor 会用到 URDF / KDL retargeter，至少再装：

```bash
pip install urdf-parser-py
```

`PyKDL` 和 `pykdl_utils` 通常来自机器人镜像或 ROS/KDL 环境。安装后用下面的命令确认：

```bash
python - <<'PY'
from scipy.spatial.transform import Rotation
from urdf_parser_py.urdf import URDF
from pykdl_utils.kdl_kinematics import KDLKinematics
from pykdl_utils.kdl_parser import kdl_tree_from_urdf_model

print("retargeter deps OK")
PY
```

如果这里报 `ModuleNotFoundError: pykdl_utils` 或 `ModuleNotFoundError: PyKDL`，先在这台机器人控制机上补齐对应的 KDL Python 绑定；否则后面 actor 初始化到 `BodyRetargeter` 时会失败。

`ruckig` 只是平滑 reset 的可选依赖，没有它也会退回一次性关节指令：

```bash
pip install ruckig
```

## 2. 检查仓库内 AgiBot SDK

确认 vendored wheels 存在：

```bash
ls examples/agibot_real/vendor/a2d_sdk/wheels
```

至少应看到：

```text
a2d_sdk-1.5.0-py3-none-any.whl
genie_msgs_pb-0.8.0-py3-none-any.whl
cosine_bus-2.0.0-cp310-cp310-linux_x86_64.whl
cosine_bus-2.0.0-cp310-cp310-linux_aarch64.whl
```

然后测试 SDK bootstrap：

```bash
cd /path/to/serl_torch
conda activate serl_torch

python - <<'PY'
from serl_torch.examples.agibot_real.robot.sdk_bootstrap import ensure_repo_local_a2d_sdk

print("AgiBot SDK ready:", ensure_repo_local_a2d_sdk())
PY
```

如果这一步提示 Python 版本不对，请确认当前环境是 Python `3.10`。这里的 `cosine_bus` wheel 是 CPython 3.10 专用的。

## 3. 准备 robot runtime

进入 AgiBot example 目录：

```bash
cd /path/to/serl_torch/examples/agibot_real
conda activate serl_torch
```

如果你有 forwarder tar：

```bash
bash tools/prepare_robot_runtime.sh --from-tar /path/to/forwarder_x86_v1.7.0.tar.gz
```

如果你有已解压的 forwarder 目录：

```bash
bash tools/prepare_robot_runtime.sh --from-dir /path/to/forwarder
```

如果当前机器不需要 ROS forwarder，可以跳过 forwarder：

```bash
AGIBOT_NO_ROS=1 bash tools/prepare_robot_runtime.sh
```

成功时会看到类似：

```text
SDK ready: ...
Forwarder ready: ...
```

或在 `AGIBOT_NO_ROS=1` 时看到跳过 forwarder 的提示。

## 4. 检查机器人网络

每个需要和机器人通信的终端都要先 source 这个环境：

```bash
cd /path/to/serl_torch/examples/agibot_real
source robot/service/env.sh
```

如果网络正确，下面两个变量应该有值：

```bash
echo "$LOCATOR_IP"
echo "$AORTA_DISCOVERY_URI"
```

`LOCATOR_IP` 应该是本机的 `10.42.0.*` 地址，`AORTA_DISCOVERY_URI` 默认是：

```text
http://10.42.0.101:2379
```

如果 `source robot/service/env.sh` 输出：

```text
no ip in 10.42.0.* found, can not communicate with robot
```

先检查网线、网卡 IP、机器人网络和防火墙。代码侧还没到能连接机器人的阶段。

也可以直接看网卡：

```bash
ip -o -4 addr list | grep '10.42.0.'
```

## 5. 启动 robot-service

开一个终端，保持它一直运行：

```bash
cd /path/to/serl_torch/examples/agibot_real
conda activate serl_torch
bash tools/start_robot_service.sh
```

如果这台机器明确不跑 ROS forwarder：

```bash
cd /path/to/serl_torch/examples/agibot_real
conda activate serl_torch
AGIBOT_NO_ROS=1 bash tools/start_robot_service.sh
```

这个 wrapper 会自动：

- 激活 `serl_torch`
- `source robot/service/env.sh`
- 使用 `robot/service/conf/copilot.pbtxt`
- 调用 `scripts/start_robot_service.py`

先不要关掉这个终端。

## 6. 测试是否连上机器人

另开一个终端，运行只读连接测试。这个测试会读取相机、关节和夹爪状态，不会给机器人发送动作。

```bash
cd /path/to/serl_torch
conda activate serl_torch
source examples/agibot_real/robot/service/env.sh

python - <<'PY'
from serl_torch.examples.agibot_real.robot.interface import AgiBotRobotNode

node = AgiBotRobotNode(hz=20.0)
try:
    head = node.get_img_head()
    left = node.get_img_left_wrist()
    right = node.get_img_right_wrist()
    joint = node.get_joint_state()
    head_joints = node.get_head_joint_states()
    waist_joints = node.get_waist_joint_states()
    arm_joints = node.get_arm_joint_states()

    if head is None or left is None or right is None:
        raise RuntimeError("camera image is None")
    if joint is None or len(joint) != 16:
        raise RuntimeError(f"bad joint state: {joint}")
    if len(head_joints) != 2 or len(waist_joints) != 2 or len(arm_joints) != 14:
        raise RuntimeError(
            "bad robot state lengths: "
            f"head={len(head_joints)} waist={len(waist_joints)} arm={len(arm_joints)}"
        )

    print("CONNECTED")
    print("head image:", head.shape)
    print("left wrist image:", left.shape)
    print("right wrist image:", right.shape)
    print("joint state length:", len(joint))
    print("head joints:", head_joints)
    print("waist joints:", waist_joints)
    print("arm joints length:", len(arm_joints))
finally:
    node.shutdown()
PY
```

看到 `CONNECTED` 就说明 Python 环境、repo-local AgiBot SDK、robot-service、网络发现、相机和基础关节状态都已经通了。

常见失败含义：

- `Repo-local AgiBot SDK bootstrap failed`：检查 Python 是否是 `3.10`，以及 `vendor/a2d_sdk/wheels/` 是否完整
- `no ip in 10.42.0.* found`：机器没有连到机器人网段
- `Timed out waiting for AgiBot camera / joint state readiness`：robot-service 没启动、机器人没上电、相机/关节状态没发布，或 DDS/发现服务不可达
- `camera image is None`：相机流没有完整起来

## 7. 可选：测试动作通道

只有在确认机器人周围安全、急停可用、机械臂工作空间清空后，才运行下面的 reset 测试。它会真的移动机器人到任务初始姿态。

```bash
cd /path/to/serl_torch
conda activate serl_torch
source examples/agibot_real/robot/service/env.sh

python examples/agibot_real/scripts/reset_robot.py \
  --task-name office_setting \
  --hz 20
```

看到 `Robot reset completed.` 就说明基础动作通道也能工作。

到这里为止，新机器已经具备运行本仓库 AgiBot 真机代码的最低条件。之后再根据实验需要启动 JoyRA / OpenPI policy server、learner 和 actor。


## 8.准备离线数据：


### 终端 1：生成离线 pkl
```bash
sudo docker exec -it docker--agibot /bin/bash
conda activate robot
python /home/hello/codebase/serl_torch/examples/agibot_real/scripts/run_residual_offline_prepare.py \
  --config-name train_residual \
  task.hz=15.0 \
  policy.type=openpi \
  policy.host=127.0.0.1 \
  policy.port=30001 \
  policy.id=pi05_task_3463_3540_mouse_only_right_hand_camera_position_15hz \
  policy.action_layout=right_arm \
  env.arm_layout=right_arm \
  env.action_dim=7 \
  env.robot_action_dim=14 \
  'residual.action_mask=[true,true,true,true,true,true,true]' \
  'residual.action_limits=[1.0,1.0,1.0,1.0,1.0,1.0,1.0]' \
  residual.chunk_horizon=15 \
  training.steps_per_update=15 \
  offline.prepared_path=null \
  offline.prepare.raw_dataset_path=/home/hello/codebase/niejunnan/datasets/task_3463_3540_mouse_only_right_hand_camera_position_15hz \
  offline.prepare.output_root=/home/hello/codebase/serl_torch/examples/agibot_real/outputs/offline_data \
  offline.prepare.max_episodes=50 \
  offline.prepare.filter_unrepresentable_steps=true
```



### 终端 2：运行 JoyRA 服务端
```bash
bash /home/hello/codebase/serl_torch/examples/agibot_real/tools/serve_joyra.sh \
  --joyra-root /home/hello/codebase/JoyRA \
  --ckpt-path /home/hello/codebase/JoyRA/outputs/pre_ego30w_sq_nw1000_nw-all-fourier_vla_post_sq_3w_office_1/checkpoints/steps_30000_pytorch_model.pt \
  --port 9002
```

### 终端 2： 运行 OpenPi 服务端
```bash
cd /home/hello/codebase/serl_torch/examples/agibot_real

bash tools/serve_openpi.sh \
  --openpi-root /home/hello/codebase/niejunnan/openpi \
  --policy-dir /home/hello/codebase/niejunnan/openpi-assets/pi05_task_3463_3540_mouse_only_right_hand_camera_position_15hz/4000/ \
  --policy-config pi05_task_3463_3540_mouse_only_right_hand_camera_position_15hz \
  --port 30001 \
  --gpu-id 0
```



## 9. 标准启动命令

### 终端 1：启动 JoyRA

```bash
bash /home/hello/codebase/tangyili/code/serl_torch/examples/agibot_real/tools/serve_joyra.sh \
  --joyra-root /home/hello/codebase/JoyRA \
  --ckpt-path /home/hello/codebase/JoyRA/outputs/pre_ego30w_sq_nw1000_nw-all-fourier_vla_post_sq_3w_office_1/checkpoints/steps_30000_pytorch_model.pt \
  --port 8001
```

### 终端 2：启动 learner

```bash
sudo docker exec -it docker--agibot /bin/bash
conda activate robot
cd /home/hello/codebase/serl_torch
export PYTHONPATH=/home/hello/codebase/serl_torch/serl_launcher:$PYTHONPATH
export CUDA_VISIBLE_DEVICES=0

python examples/agibot_real/scripts/run_residual_training.py \
  runtime.role=learner \
  policy.port=8001 \
  backfill_policy.port=8001
```

### 终端 3：启动 actor

```bash
sudo docker exec -it docker--agibot /bin/bash
conda activate robot
cd /home/hello/codebase/tangyili/code/serl_torch/examples/agibot_real
source robot/service/env.sh
cd /home/hello/codebase/tangyili/code/serl_torch
export PYTHONPATH=/home/hello/codebase/tangyili/code/serl_torch/serl_launcher:$PYTHONPATH
export CUDA_VISIBLE_DEVICES=1

python examples/agibot_real/scripts/run_residual_training.py \
  runtime.role=actor \
  policy.port=8001 \
  backfill_policy.port=8001
```

`train_residual.yaml` 默认已经开启 `backfill_policy.enabled=true`。上面的命令把 `policy.port` 和 `backfill_policy.port` 都设为 `8001`，表示主控制推理和 backfill 共用同一个 JoyRA 服务。如果需要减少真机控制路径和 backfill 的推理竞争，可以再启动一个 JoyRA 服务，把 `backfill_policy.port` 改成独立端口。

### 可选：standalone processor 模式

默认 `train_residual.yaml` 仍然是 in-process processor，actor 会在本进程内完成 transition assembly 和 replay commit。需要把真机交互和数据处理拆开时，使用 `train_residual_processor.yaml`，并额外启动 processor 角色。

learner：

```bash
python examples/agibot_real/scripts/run_residual_training.py \
  --config-name train_residual_processor \
  runtime.role=learner \
  policy.port=8001 \
  backfill_policy.port=8001
```

processor：

```bash
python examples/agibot_real/scripts/run_residual_processor.py \
  policy.port=8001 \
  backfill_policy.port=8001
```

actor：

```bash
python examples/agibot_real/scripts/run_residual_training.py \
  --config-name train_residual_processor \
  runtime.role=actor \
  policy.port=8001 \
  backfill_policy.port=8001
```

这个模式下 `processor.mode=standalone`、`processor_batching.enabled=true`、`recycle.enabled=true`。上面的最小命令仍然让主控制推理和 processor backfill 共用 `8001`。如果要隔离推理竞争，需要再启动一个 JoyRA 服务到独立端口，例如 `9011`，然后把三条命令里的 `backfill_policy.port` 改成 `9011`。如果磁盘空间有限，可以额外加 `recycle.enabled=false`。
