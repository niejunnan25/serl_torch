# `docs/` 目录索引

这份索引用来说明当前 `docs/` 里还保留哪些 repo 级文档。过时的 refactor 施工日志和旧启动命令已经清理掉；真实运行入口请优先看各 example 的 README。

截至 `2026-05-16`，当前 `docs/` 下共有 `13` 篇主题文档（不含本 README）。`docs/refactors/` 目前不再保留历史施工日志。

## 推荐入口

- LIBERO 训练、评测、OpenPI 相关流程：
  [../examples/libero/README.md](../examples/libero/README.md)
- AgiBot 真机 bring-up、训练、评估和停机：
  [../examples/agibot_real/README.md](../examples/agibot_real/README.md)
- 当前 repo 总览和安装：
  [../README.md](../README.md)

## 当前主线文档

### 设计与风险

- [2026-04-13-mainline-review-findings.md](./2026-04-13-mainline-review-findings.md)
  主线代码 review 与风险盘点。
- [2026-04-16-agibot-transition-dataflow-and-refactor-plan.md](./2026-04-16-agibot-transition-dataflow-and-refactor-plan.md)
  AgiBot residual RL transition 数据流、chunk/step 语义和重构方案。
- [2026-04-17-agentlace-serl-plan-c-design.md](./2026-04-17-agentlace-serl-plan-c-design.md)
  trainer transport 的 Plan C 设计。
- [2026-04-17-agibot-copy-split-queue-checklist.md](./2026-04-17-agibot-copy-split-queue-checklist.md)
  AgiBot async commit / queue 相关检查清单。

### LIBERO 训练线

- [2026-04-23-libero-residual-training-entrypoints.md](./2026-04-23-libero-residual-training-entrypoints.md)
  当前 LIBERO residual 训练入口、launcher mode 和推荐使用方式。
- [2026-04-23-libero-spatial-0-to-9-offline-prepare-commands.md](./2026-04-23-libero-spatial-0-to-9-offline-prepare-commands.md)
  `libero_spatial` task 0-9 filtered offline 数据准备命令记录。

### 性能与 benchmark

- [2026-04-16-learner-gradient-update-speed-optimization.md](./2026-04-16-learner-gradient-update-speed-optimization.md)
  learner update 速度瓶颈分析与优化路线。
- [2026-04-16-learner-speed-ablation-bf16-and-critic-freeze.md](./2026-04-16-learner-speed-ablation-bf16-and-critic-freeze.md)
  bf16 与 actor-update critic freeze 的消融实验。
- [2026-04-17-plan-c-vectorized-rollout-implementation-report.md](./2026-04-17-plan-c-vectorized-rollout-implementation-report.md)
  Plan C、`batch_insert()`、`async_commit` 的实施报告和 benchmark。
- [2026-04-17-weekly-optimization-highlights.md](./2026-04-17-weekly-optimization-highlights.md)
  4 月中旬优化结果的阶段汇总。
- [2026-05-16-jax-resnet10-cuda-env-and-encoder-benchmark.md](./2026-05-16-jax-resnet10-cuda-env-and-encoder-benchmark.md)
  JAX ResNet10 CUDA 环境修复与编码器测速。
- [2026-05-16-high-utd-and-image-path-benchmark-notes.md](./2026-05-16-high-utd-and-image-path-benchmark-notes.md)
  High-UTD 编译策略与图像路径 benchmark。
- [2026-05-16-libero-learner-throughput-optimization-summary.md](./2026-05-16-libero-learner-throughput-optimization-summary.md)
  当前 LIBERO learner 吞吐优化总结。

## 维护约定

- 当前可执行流程写进对应 example 的 README。
- repo 级、跨模块、能反复引用的设计和 benchmark 文档放在 `docs/`。
- 旧启动命令、阶段性 refactor 施工日志、只指向不存在路径的历史记录不要继续保留。
- 新增文档尽量使用日期前缀和明确主题名。
