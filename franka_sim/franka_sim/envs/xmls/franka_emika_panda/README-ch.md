# Franka Emika Panda 描述（MJCF）

需要 MuJoCo 2.3.3 或更高版本。

## 概览

该包包含 [Franka Emika Panda](https://www.franka.de/) 的简化机器人描述（MJCF），由 [Franka Emika](https://www.franka.de/company) 开发。其来源于[公开可用的 URDF 描述](https://github.com/frankaemika/franka_ros/tree/develop/franka_description)。

<p float="left">
  <img src="panda.png" width="400">
</p>

## URDF → MJCF 推导步骤

1. 使用 [Blender](https://www.blender.org/) 将 DAE [mesh 文件](https://github.com/frankaemika/franka_ros/tree/develop/franka_description/meshes/visual)转换为 OBJ 格式。
2. 使用 [`obj2mjcf`](https://github.com/kevinzakka/obj2mjcf) 处理 `.obj` 文件。
3. 从为 `link0` 创建的子网格中移除完全平坦的 `link0_6`。
4. 使用 [V-HACD](https://github.com/kmammou/v-hacd) 为 `link5` 的 STL 碰撞 [mesh 文件](https://github.com/frankaemika/franka_ros/tree/develop/franka_description/meshes/collision)创建凸分解。
5. 在 [URDF](https://github.com/frankaemika/franka_ros/tree/develop/franka_description/robots) 的 `<robot>` 子句中加入 `<mujoco> <compiler discardvisual="false"/> </mujoco>`，以保留可视化几何体。
6. 将 URDF 加载到 MuJoCo 中，并保存对应 MJCF。
7. 依据 [inertial.yaml](https://github.com/frankaemika/franka_ros/blob/develop/franka_description/robots/common/inertial.yaml) 对齐惯性参数。
8. 在底座添加追踪灯。
9. 手动编辑 MJCF，将公共属性提取到 `<default>` 段。
10. 添加 `<exclude>` 子句，防止 `link7` 与 `link8` 之间碰撞。
11. 手工为指尖设计碰撞几何体。
12. 为机械臂添加位置控制执行器。
13. 添加等式约束，使左手指模仿右手指位置。
14. 添加 tendon 以在两指间均分力，并在该 tendon 上添加位置执行器。
15. 添加 `scene.xml`，其中包含机器人、带纹理的地面、天空盒和雾效。

## 许可证

该模型基于 [Apache-2.0 License](LICENSE) 发布。
