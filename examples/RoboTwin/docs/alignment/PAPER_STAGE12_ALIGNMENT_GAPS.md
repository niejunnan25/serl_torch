# RoboTwin 与论文 Stage-1/2 差距清单（归档版）

本文档原本用于“对齐前差距跟踪”。当前仓库已完成大部分条目，现改为归档状态说明。

- 最新实现说明：`PAPER_STAGE12_ALIGNMENT_IMPLEMENTATION.md`
- 最新论文对比：`PAPER_IMPLEMENTATION_COMPARISON.md`

## 状态总览

| 条目 | 当前状态 | 备注 |
|---|---|---|
| Cal-QL critic 预训练 | 部分对齐 | 已有 CQL-style conservative 预训练；非严格论文 Cal-QL 逐式复刻 |
| OTF TD backup | 已对齐 | `sac.otf_num_samples` 已接入 |
| warmup episodes 口径 | 已对齐 | `training.warmup_base_episodes` |
| probing alpha 口径 | 已对齐 | `training/eval.probing_alpha`，支持 `U(0, alpha*T)` |
| probing prefix 不入 replay | 已对齐 | 训练中 probing 仅用于状态初始化 |
| critic:actor=2:1 | 已对齐 | `sac.utd_ratio=2` |
| AdamW + grad clip | 已对齐 | `sac.optimizer.*` |
| 论文默认超参 preset | 已对齐 | 主配置已切到论文默认预算与关键超参 |
| 3-layer MLP 默认 | 已对齐 | policy/critic 默认 `[256,256,256]` |
| 预训练视觉编码器默认 | 已对齐 | `sac.encoder_type=resnet-pretrained` |
| xi 语义参数 | 已对齐 | `residual.xi` 已接入执行和离线反解 |
| xi scheduler 语义 | 已对齐 | `training.xi_scheduler` 直接调 `xi_t` |
| 异步采样-学习 | 已对齐 | `training.async.*` |
| 3 seeds + CI 脚本 | 已对齐 | `scripts/run_stage12_repro.sh` + `aggregate_eval_ci.py` |
| Stage-3 蒸馏闭环 | 未覆盖 | 本目录范围外 |

## 剩余高优先级（若追求更严格 paper-faithful）

1. 将当前 CQL-style critic 预训练替换为严格论文 Cal-QL 目标。
2. 在本仓库内补齐 Stage-3 SFT 训练入口与完整闭环。
