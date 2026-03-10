# Robotiq 2F-85 描述（MJCF）

需要 MuJoCo 2.2.2 或更高版本。

## 概览

该包包含 [Robotiq 85mm 双指自适应夹爪](https://robotiq.com/products/2f85-140-adaptive-robot-gripper) 的简化机器人描述（MJCF），由 [Robotiq](https://robotiq.com/) 开发。其来源于[公开可用的 URDF 描述](https://github.com/ros-industrial/robotiq/tree/kinetic-devel/robotiq_2f_85_gripper_visualization)。

<p float="left">
  <img src="2f85.png" width="400">
</p>

## URDF → MJCF 推导步骤

1. 在 URDF 的 `<robot>` 子句中加入 `<mujoco> <compiler discardvisual="false"/> </mujoco>`，以保留可视化几何体。
2. 将 URDF 加载到 MuJoCo 中，并保存对应 MJCF。
3. 手动编辑 MJCF，将公共属性提取到 `<default>` 段。
4. 添加 `<exclude>` 子句，防止连杆机构之间发生碰撞。
5. 将碰撞垫拆分为两个垫，以获得更多接触点。
6. 提高碰撞垫摩擦与优先级。
7. 添加 `impratio=10` 以获得更好的无滑动效果。
8. 添加 `scene.xml`，其中包含机器人、带纹理的地面、天空盒和雾效。
9. 在 `scene.xml` 中添加悬挂盒。

## 许可证

该模型基于 [BSD-2-Clause License](LICENSE) 发布。
