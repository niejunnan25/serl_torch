# PLD 论文与当前实现对比（Stage-1/2）

## 结论

当前 `examples/RoboTwin` 的 Stage-1/2 与论文
**Self-Improving Vision-Language-Action Models with Data Generation via Residual RL**
在核心算法上已经对齐，属于“可复现趋势”的实现。

仍保留的主要差异是：
- Cal-QL 为 CQL-style conservative 近似实现，不是论文源码逐式复刻。
- Stage-3（蒸馏回 VLA）不在本目录实现范围。

## 对比范围

- 论文：
  - https://arxiv.org/abs/2511.00091
  - https://arxiv.org/src/2511.00091
- 代码：
  - `examples/RoboTwin/scripts/train_residual_sac.py`
  - `examples/RoboTwin/scripts/eval_residual_fast.py`
  - `examples/RoboTwin/core/common.py`
  - `serl_launcher/serl_launcher/agents/continuous/sac.py`
  - `examples/RoboTwin/conf/train_residual_sac.yaml`

## 一致点（核心）

1. 残差策略形式一致：`pi_delta(a_delta | s, a_b)`，执行 `a = a_b + a_delta`。
2. 残差是逐步决策，不是 residual chunk 输出。
3. 残差输入一致：`images_t + state_t + base_action_t`。
4. 幅度约束一致：残差先 clip 到 `[-1,1]`，再映射到 `[-xi,xi]`。
5. `xi` 调度语义一致：`training.xi_scheduler` 直接调 `xi_t`。
6. warmup 口径一致：`warmup_base_episodes`（默认 100）。
7. probing 口径一致：`T_base ~ U(0, alpha*T)`，且 probing prefix 不写 replay。
8. 双 buffer + 1:1 混采一致：`offline.symmetric_replay=true`。
9. critic:actor 更新比一致：`utd_ratio=2`。
10. 目标熵、温度、gamma、Polyak、batch、buffer 默认值与论文主设定一致。
11. OTF TD backup 已实现（默认 `otf_num_samples=1`）。
12. 采样-学习异步解耦已实现（`training.async.*`）。
13. 训练预算协议入口已提供（250k steps + 3 seeds + 50 eval episodes）。

## 不一致点（当前保留）

1. Cal-QL 的严格形式
- 当前是 CQL-style conservative critic 预训练近似。
- 对论文“Cal-QL”语义方向一致，但不是逐公式完全同构。

2. Stage-3 缺失
- 当前目录聚焦 Stage-1/2。
- 若目标是完整 PLD 三阶段复现，还需补 Stage-3 SFT pipeline。

## 平台差异说明（非 bug）

- 论文实验常见 7-DoF 机器人动作空间。
- 当前 RoboTwin 以 ALOHA 14D 为基动作空间，并通过 `action_dim/action_indices` 适配残差维度。
- 这属于平台迁移差异，不影响 Stage-1/2 算法结构对齐判断。

