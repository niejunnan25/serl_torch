# RoboTwin Stage1/2 实现审查报告（更新）

日期：2026-03-03

## 审查范围

- 代码：
  - `examples/RoboTwin/scripts/train_residual_sac.py`
  - `examples/RoboTwin/scripts/eval_residual_fast.py`
  - `examples/RoboTwin/env_wrappers/`
  - `examples/RoboTwin/policy/`
  - `serl_launcher/serl_launcher/agents/continuous/sac.py`
  - `serl_launcher/serl_launcher/agents/continuous/drq.py`
- 配置：
  - `examples/RoboTwin/conf/train_residual_sac.yaml`
  - `examples/RoboTwin/conf/eval_residual_fast.yaml`
- 脚本：
  - `examples/RoboTwin/scripts/run_stage12_repro.sh`
  - `examples/RoboTwin/scripts/aggregate_eval_ci.py`

## 已完成的验证

- 语法检查通过：
  - `python -m py_compile`（train/eval/common/sac/drq/aggregate）
  - `bash -n examples/RoboTwin/scripts/run_stage12_repro.sh`
- 依赖探测（当前主机）：
  - 缺失：`gym`, `hydra`, `omegaconf`, `tensorflow`, `wandb`
  - 存在：`torch`, `requests`, `imageio`

## 主要发现（按严重程度）

## [HIGH] 1. 当前主机无法直接启动训练/评估

- 现象
  - 入口脚本会在 import 阶段失败（如 `gym` / `hydra` 不存在）。
- 影响
  - 当前机器无法做端到端运行验证。
- 建议
  - 先补齐最小依赖，再做 smoke run。

## [HIGH] 2. `expert_check` 长期失败时可能“看起来卡住”

- 现象
  - 默认 phase 很大，若 `max_seed_attempts_per_phase` 不设且 expert_check 持续失败，会长时间尝试 seed。
- 影响
  - 训练进度慢，容易误判为死循环。
- 建议
  - 显式设置 `training.max_seed_attempts_per_phase`（如 5000/10000）。

## [HIGH] 3. 训练存在重依赖链（tensorflow/wandb）

- 现象
  - 训练脚本依赖的 `concat_batches` 来源模块顶层引入 `tensorflow/wandb`。
- 影响
  - 即使不启用相关功能，也可能因依赖缺失无法启动。
- 建议
  - 把轻量工具函数拆到无重依赖模块，或改为延迟导入。

## [MEDIUM] 4. `run_stage12_repro.sh` 对 checkpoint 存在性缺少保护

- 现象
  - 脚本直接 `ls checkpoint_*.pt | tail -n 1` 获取最新权重。
- 影响
  - 若训练中断且未产出 checkpoint，评估阶段会直接失败。
- 建议
  - 增加“未找到 checkpoint”分支并输出清晰错误。

## [MEDIUM] 5. `resnet-pretrained` 首次依赖联网下载

- 现象
  - 无公网环境下可能无法初始化预训练编码器。
- 影响
  - 训练/评估启动失败。
- 建议
  - 提供本地权重路径配置，或提前预置权重文件。

## [MEDIUM] 6. Eval checkpoint 加载包含 optimizer state，跨版本兼容性有限

- 现象
  - 评估加载也会尝试恢复优化器状态。
- 影响
  - 不同模型结构/优化器设置下可能报错。
- 建议
  - 为 eval 增加“仅加载网络参数”选项。

## [LOW] 7. `utd_ratio` 对 batch 可整除有隐含约束

- 现象
  - `batch_size % utd_ratio != 0` 会在运行时报错。
- 建议
  - 增加配置前置校验和友好报错。

## [LOW] 8. `openpi.chunk_horizon/replan_every` 当前不直接驱动训练主循环

- 现象
  - 主循环窗口由 `residual.chunk_horizon` 控制。
- 建议
  - 文档继续强调该点，减少误配置。

## 总体结论

- 从静态检查和配置一致性看，当前 Stage-1/2 代码链路闭合，且已覆盖 `xi_scheduler` 与异步采样-学习。
- 但“能否一次跑起来”仍受环境条件限制：
  - 依赖完整性（`gym/hydra/omegaconf/...`）
  - OpenPI 服务可达
  - RoboTwin 环境可正常 reset/step
  - 预训练权重可用（若启用 `resnet-pretrained`）
