# `docs/` 目录索引

这份索引文档用来回答一件事：

当前仓库里的 `docs/` 到底在记录什么，哪些文档值得优先看，哪些更适合作为历史 refactor 档案来查。

截至 `2026-04-17`，当前 `docs/` 下共有：

- 根目录主题文档 `8` 篇（不含本 README）
- `docs/refactors/` 历史文档 `28` 篇

## 1. `docs/` 目录主要记录什么

当前 `docs/` 更像是一个 **repo 级设计与优化记录区**，而不是日常使用手册。

这里的文档主要分成三类：

- **主线 review / 风险盘点**
  记录当前代码主线还存在哪些语义风险、工程风险或安全风险。
- **设计说明 / 优化分析 / benchmark 复盘**
  记录 residual RL 主线的数据流、性能瓶颈、优化思路、实验结果和实施报告。
- **历史 refactor 过程档案**
  记录某一轮重构是如何分阶段推进的，适合回溯“为什么会变成现在这样”。

如果你要找的是：

- 真机启动、AgiBot 运行方式、优化训练命令：
  请优先看 `examples/agibot_real/docs/`
- LIBERO 训练、评测、OpenPI 相关实验说明：
  请优先看 `examples/libero/docs/`

也就是说：

- `docs/`：跨 example、跨模块、偏设计与演进
- `examples/*/docs`：贴近具体训练线、运行方式和实验记录

## 2. 根目录文档：当前最值得优先看的内容

根目录下这批文档，基本可以理解为当前仓库里最“主线”的设计与优化记录。

### [2026-04-17-weekly-optimization-highlights.md](./2026-04-17-weekly-optimization-highlights.md)

用途：

- 把本周最关键的 4 条优化结果整理成一篇更适合写周报和阶段汇报的说明文档
- 解释“数字背后到底改了什么”

这篇文档会集中展开说明：

- learner 侧 `bf16 + freeze critic grad`
- replay `batch_insert()` + `async_commit`
- OpenPI `infer_many()` + optimized LIBERO residual training
- actor 热路径上的 `build_decision_obs` 去负担

如果你想快速讲清楚“这一周优化做了什么、为什么有效、提升体现在哪里”，建议优先看这篇。

### [2026-04-13-mainline-review-findings.md](./2026-04-13-mainline-review-findings.md)

用途：

- 主线代码 review 记录
- 梳理当前最值得优先关注的风险点

适合在这些场景下阅读：

- 你想知道当前主线还有哪些工程风险没有彻底处理
- 你想写“已识别问题 / 后续改进项”

关注点包括：

- remote env RPC 的 `pickle` 边界风险
- `torch.load` 的不可信权重加载风险
- AgiBot stale observation 风险
- controller 终止竞态
- async eval 语义风险

### [2026-04-16-agibot-transition-dataflow-and-refactor-plan.md](./2026-04-16-agibot-transition-dataflow-and-refactor-plan.md)

用途：

- AgiBot residual RL 数据流和 transition 语义说明
- 解释当前 actor / learner / replay 到底怎么配合

这篇文档的重要性很高，因为它把当前主线的真实训练语义讲清楚了：

- 执行单位是 `chunk`
- 存储单位是 `step`
- 训练采样单位是 `chunk window`

同时它也是 `TransitionAssembler`、`ReplaySink`、`StatsReporter` 这类后续重构的设计起点。

### [2026-04-16-learner-gradient-update-speed-optimization.md](./2026-04-16-learner-gradient-update-speed-optimization.md)

用途：

- learner 更新速度的瓶颈分析
- 低风险与中风险优化项的优先级梳理

适合在这些场景下阅读：

- 你要解释这周为什么在做 learner 优化
- 你要说明当前 learner 慢在哪里
- 你要给后续优化排序

这篇文档把问题拆得比较清楚：

- 当前一个 outer update 实际做了多少次 critic / actor / temperature 更新
- replay sample 是否是真瓶颈
- bf16、冻结 critic grad、冻结 backbone、减少视觉开销分别意味着什么

### [2026-04-16-learner-speed-ablation-bf16-and-critic-freeze.md](./2026-04-16-learner-speed-ablation-bf16-and-critic-freeze.md)

用途：

- 上一篇分析文档的实测消融结果
- 给优化结论提供 benchmark 证据

这篇文档更适合写周报或技术总结，因为里面有比较直接的量化结果，例如：

- `bf16` 的速度提升
- 显存下降幅度
- actor update 阶段冻结 critic 参数梯度的额外收益

如果你只想引用“优化带来了什么提升”，这篇比分析文档更方便。

### [2026-04-17-agentlace-serl-plan-c-design.md](./2026-04-17-agentlace-serl-plan-c-design.md)

用途：

- trainer transport 改造的设计文档
- 解释为什么需要 `Plan C`

核心价值在于把 transport 问题讲成了一个完整协议问题，而不是单纯“换个 socket”：

- control / data 分通道
- replay commit 异步化
- `accepted_update_id` / `committed_update_id` 双语义
- bounded queue 和 backpressure

如果你要解释为什么这周会开始做 `async_commit` 这条线，这篇文档是关键材料。

### [2026-04-17-plan-c-vectorized-rollout-implementation-report.md](./2026-04-17-plan-c-vectorized-rollout-implementation-report.md)

用途：

- `Plan C` 在当前仓库里的实际实施报告
- replay `batch_insert()` 和 `async_commit` 的真实 benchmark 复盘

这篇文档回答的是：

- 这轮优化到底已经实现了什么
- 当前生产路径吃到了哪些收益
- 为什么早期 smoke benchmark 看起来不明显

如果要说“trainer ingest 和 replay 写入这周到底优化了多少”，优先看这篇。

### [2026-04-17-agibot-copy-split-queue-checklist.md](./2026-04-17-agibot-copy-split-queue-checklist.md)

用途：

- AgiBot `copy` 训练线接入 `async_commit` 之后的人工检查清单

这篇不是设计文档，而是偏运维 / 实验检查单。

适合在这些场景下阅读：

- 你要验证 `copy` + `async_commit` 是否健康
- 你要观察 episode boundary、reset overlap、backpressure 是否正常

## 3. `docs/refactors/`：历史重构档案区

`docs/refactors/` 里的 `28` 篇文档，主要是 **重构过程中的阶段记录**，更适合回答：

- 这一轮重构是怎么一步步推进的？
- 当时为什么这样拆模块？
- 某个阶段的 package boundary 是怎么定的？

不建议把这个目录当成“当前主线使用手册”。

这里面最主要有几组文档。

### 3.1 `2026-04-08-libero-thin-train-loop-*`

这一组文档记录的是 LIBERO 训练入口瘦身和 actor / learner runtime 拆分过程。

主要价值：

- 说明早期为什么要把 entrypoint 变薄
- 说明 actor loop、runtime session、support helper 等边界是怎么抽出来的
- 保留了分阶段修正过程

推荐阅读顺序：

1. [2026-04-08-libero-thin-train-loop-summary.md](./refactors/2026-04-08-libero-thin-train-loop-summary.md)
2. [2026-04-08-libero-thin-train-loop-second-stage-summary.md](./refactors/2026-04-08-libero-thin-train-loop-second-stage-summary.md)
3. [2026-04-08-libero-thin-train-loop-continuation-summary.md](./refactors/2026-04-08-libero-thin-train-loop-continuation-summary.md)

`v1` 到 `v16` 更像详细施工日志，适合在你真的要回溯某一阶段细节时再看。

### 3.2 `2026-04-08-serl-launcher-training-phase*`

这一组文档记录的是 `serl_launcher` 包边界和 training / residual 模块迁移过程。

主要价值：

- 回答 training 基础设施为什么被拆到 `serl_launcher.training`
- 回答 residual algorithm 和 runtime 早期是怎么分层的
- 记录 package rename 和 import update 的真实演进

推荐阅读顺序：

1. [2026-04-08-serl-launcher-training-phase1.md](./refactors/2026-04-08-serl-launcher-training-phase1.md)
2. [2026-04-08-serl-launcher-training-phase2.md](./refactors/2026-04-08-serl-launcher-training-phase2.md)
3. [2026-04-08-serl-launcher-training-phase3a.md](./refactors/2026-04-08-serl-launcher-training-phase3a.md)
4. [2026-04-08-serl-launcher-training-phase3b.md](./refactors/2026-04-08-serl-launcher-training-phase3b.md)
5. [2026-04-08-serl-launcher-training-phase3c.md](./refactors/2026-04-08-serl-launcher-training-phase3c.md)

### 3.3 其余几篇更接近“专题说明”

- [2026-04-09-libero-scripts-layout-and-validation-followups.md](./refactors/2026-04-09-libero-scripts-layout-and-validation-followups.md)
  说明 LIBERO 脚本布局和最终入口命名为什么调整成当前样子。
- [2026-04-12-agibot-agentlace-bootstrap-note.md](./refactors/2026-04-12-agibot-agentlace-bootstrap-note.md)
  说明 AgiBot 早期为什么还挂着 `agentlace` 相关 bootstrap。
- [2026-04-12-agibot-chunk-replay-note.md](./refactors/2026-04-12-agibot-chunk-replay-note.md)
  讨论 AgiBot 为什么会采用 step-level storage + chunk-level training sample 这条 replay 路径。

## 4. 推荐阅读路径

如果你只是想快速进入状态，可以按用途读。

### 4.1 写周报 / 写阶段总结

推荐顺序：

1. [2026-04-16-learner-speed-ablation-bf16-and-critic-freeze.md](./2026-04-16-learner-speed-ablation-bf16-and-critic-freeze.md)
2. [2026-04-17-weekly-optimization-highlights.md](./2026-04-17-weekly-optimization-highlights.md)
3. [2026-04-17-plan-c-vectorized-rollout-implementation-report.md](./2026-04-17-plan-c-vectorized-rollout-implementation-report.md)
4. [2026-04-16-agibot-transition-dataflow-and-refactor-plan.md](./2026-04-16-agibot-transition-dataflow-and-refactor-plan.md)
5. [2026-04-13-mainline-review-findings.md](./2026-04-13-mainline-review-findings.md)

这样读的好处是：

- 先看到优化结果
- 再看到优化对象和设计背景
- 最后补上风险与后续项

### 4.2 查“为什么要这么设计”

推荐顺序：

1. [2026-04-16-agibot-transition-dataflow-and-refactor-plan.md](./2026-04-16-agibot-transition-dataflow-and-refactor-plan.md)
2. [2026-04-17-agentlace-serl-plan-c-design.md](./2026-04-17-agentlace-serl-plan-c-design.md)
3. [refactors/2026-04-12-agibot-chunk-replay-note.md](./refactors/2026-04-12-agibot-chunk-replay-note.md)
4. [refactors/2026-04-08-serl-launcher-training-phase1.md](./refactors/2026-04-08-serl-launcher-training-phase1.md)

### 4.3 查“这轮性能优化到底提升了什么”

推荐顺序：

1. [2026-04-16-learner-gradient-update-speed-optimization.md](./2026-04-16-learner-gradient-update-speed-optimization.md)
2. [2026-04-16-learner-speed-ablation-bf16-and-critic-freeze.md](./2026-04-16-learner-speed-ablation-bf16-and-critic-freeze.md)
3. [2026-04-17-weekly-optimization-highlights.md](./2026-04-17-weekly-optimization-highlights.md)
4. [2026-04-17-plan-c-vectorized-rollout-implementation-report.md](./2026-04-17-plan-c-vectorized-rollout-implementation-report.md)

### 4.4 查“仓库是怎么一步步整理成现在这样的”

推荐顺序：

1. [refactors/2026-04-08-libero-thin-train-loop-summary.md](./refactors/2026-04-08-libero-thin-train-loop-summary.md)
2. [refactors/2026-04-08-libero-thin-train-loop-continuation-summary.md](./refactors/2026-04-08-libero-thin-train-loop-continuation-summary.md)
3. [refactors/2026-04-08-serl-launcher-training-phase1.md](./refactors/2026-04-08-serl-launcher-training-phase1.md)
4. [refactors/2026-04-09-libero-scripts-layout-and-validation-followups.md](./refactors/2026-04-09-libero-scripts-layout-and-validation-followups.md)

## 5. 当前对 `docs/` 的理解

就目前内容来看，这个目录已经形成了比较明确的分工：

- `docs/` 根目录：
  当前主线最值得引用的 review、设计和优化文档
- `docs/refactors/`：
  历史重构过程档案
- `examples/agibot_real/docs/` 与 `examples/libero/docs/`：
  面向具体训练线、评测线和运行方式的专题文档

这个结构是可用的，而且比把所有文档都堆在根目录里更清楚。

## 6. 后续维护建议

如果后面继续写文档，建议保持下面这套约定。

- **repo 级、跨模块、能反复引用的文档** 放在 `docs/`
- **阶段性重构施工日志** 放在 `docs/refactors/`
- **AgiBot / LIBERO 专用运行文档与实验记录** 放在各自 `examples/*/docs/`
- 根目录新增文档时，尽量保持：
  - 日期前缀
  - 主题明确
  - 文件名直接反映问题域
- 一轮重构结束后，优先补“summary / implementation report”，不要只留下中间推演稿

如果后续 `docs/` 根目录继续增长，一个自然的下一步会是再细分成：

- `docs/reviews/`
- `docs/optimization/`
- `docs/transport/`

但以当前规模来看，先通过这份 `README` 做索引已经足够。
