# 简介：
本包提供了一个使用 Mujoco 编写的简易 Franka 机械臂与 Robotiq 夹爪仿真器。
其中包含基于状态和基于视觉的 Franka 抬升立方体任务环境。

# 安装：
- 在 `serl` 目录下进入 `franka_sim`。
- 在你的 `serl` conda 环境中执行 `pip install -e .` 安装本包。
- 执行 `pip install -r requirements.txt` 安装仿真依赖。

# 浏览环境
- 运行 `python franka_sim/test/test_gym_env_human.py` 启动显示窗口并可视化任务。

# 致谢：
- 该仿真最初由 [Kevin Zakka](https://kzakka.com/) 构建。
- 在 Kevin 授权下，我们在其基础上采用了 Gymnasium 环境实现。

# 备注：
- 在 CPU 机器上运行时，若出现与 `egl` 相关错误：
```bash
export MUJOCO_GL=egl
conda install -c conda-forge libstdcxx-ng
```
