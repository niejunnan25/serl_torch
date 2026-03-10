# SERL Robot Infra
![](../docs/images/robot_infra_interfaces.png)

机器人代码结构如下：
包含一个通过 ROS 向机器人发送命令的 Flask 服务端，以及一个通过 post 请求与 Flask 服务端通信的机器人 gym 环境。

- `robot_server`：托管 Flask 服务端，通过 ROS 向机器人发送命令
- `franka_env`：机器人 gym 环境，通过 post 请求与 Flask 服务端通信


### 安装

1. 按照[这里](https://frankaemika.github.io/docs/requirements.html)的说明安装 `libfranka` 和 `franka_ros`。

2. 然后安装 `serl_franka_controllers`：https://github.com/rail-berkeley/serl_franka_controllers

3. 最后安装本包及其依赖。
    ```bash
    conda activate serl
    pip install -e .
    ```

### 使用

**Robot Server**

开始使用机器人前，先给机器人上电（地面控制箱背部的小开关）。在浏览器访问机器人 IP 地址解锁机器人，然后按黑白按钮让机器人进入 FCI 控制模式（蓝灯）。

然后进入 `serl_robot_infra` 并运行 franka server。该步骤需要在 ROS 环境中进行。

```bash
conda activate serl

# 启动 http 服务与 ros 控制器的脚本
python serl_robot_infra/robot_servers/franka_server.py \
    --gripper_type=<Robotiq|Franka|None> \
    --robot_ip=<robot_IP> \
    --gripper_ip=<[Optional] Robotiq_gripper_IP> \
    --reset_joint_target=<[Optional] robot_joints_when_robot_resets>

# 如果你使用 Robotiq 夹爪，在运行 franka_server 后激活夹爪
curl -X POST http://127.0.0.1:5000/activate_gripper
```

这会启动 ROS 节点阻抗控制器和 HTTP 服务。你可以尝试移动末端执行器来测试是否正常：若阻抗控制器正在运行，机械臂应当是顺应的。

HTTP 服务用于 ROS 控制器与 gym 环境之间的通信。可用 HTTP 请求包括：

| Request | 说明 |
| --- | --- |
| startimp | 停止阻抗控制器 |
| stopimp | 启动阻抗控制器 |
| pose | 控制机器人移动到基坐标系下给定末端位姿（xyz+quaternion） |
| getpos | 返回当前基坐标系下末端位姿（xyz+quaternion） |
| getpos_euler | 返回当前基坐标系下末端位姿（xyz+rpy） |
| getvel | 返回当前基坐标系下末端速度 |
| getforce | 返回刚度坐标系下末端估计受力 |
| gettorque | 返回刚度坐标系下末端估计力矩 |
| getq | 返回当前关节位置 |
| getdq | 返回当前关节速度 |
| getjacobian | 返回当前 zero-jacobian |
| getstate | 返回全部机器人状态 |
| jointreset | 执行关节复位 |
| activate_gripper | 激活夹爪（仅 Robotiq）。启动 franka_server 后需执行一次以控制 Robotiq 夹爪。 |
| reset_gripper | 重置夹爪（仅 Robotiq） |
| get_gripper | 返回当前夹爪位置 |
| close_gripper | 完全闭合夹爪 |
| open_gripper | 完全打开夹爪 |
| move_gripper | 将夹爪移动到指定位置 |
| clearerr | 清除错误 |
| update_param | 更新阻抗控制器参数 |

这些命令也可以直接在终端调用。常用命令示例：
```bash
curl -X POST http://127.0.0.1:5000/activate_gripper # 激活夹爪
curl -X POST http://127.0.0.1:5000/close_gripper # 闭合夹爪
curl -X POST http://127.0.0.1:5000/open_gripper # 打开夹爪
curl -X POST http://127.0.0.1:5000/getpos_euler # 打印当前末端位姿
curl -X POST http://127.0.0.1:5000/jointreset # 执行关节复位
curl -X POST http://127.0.0.1:5000/stopimp # 停止阻抗控制器
curl -X POST http://127.0.0.1:5000/startimp # 启动阻抗控制器（**仅在 stopimp 之后执行**）
```

**Robot Env（客户端）**

最后，我们使用 gym env 接口与机器人服务端交互，定义在本仓库的 `franka_env` 中。只需在 `robot_infra` 目录执行 `pip install -e .`，然后在代码中通过 `gym.make("Franka-{ENVIRONMENT NAME}-v0)` 初始化环境。

示例：
```py
import gym
import franka_env
env = gym.make("FrankaEnv-Vision-v0")
```

### 已提供环境

1. peg insertion
2. pcb insertion
3. cable routing
4. bin relocation

请参考 `serl/examples/` 目录中的对应示例。
