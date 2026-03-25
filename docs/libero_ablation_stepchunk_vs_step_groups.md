# LIBERO 三组 Step vs StepChunk 消融实验说明

本文档说明以下三组实验的目的与差异：

- `outputs/libero/ablation_stepchunk_vs_step_task8_xi_warmup`
- `outputs/libero/ablation_stepchunk_vs_step_task6_xi_warmup`
- `outputs/libero/ablation_stepchunk_vs_step_task6_xi_warmup_eval50`

对应配置目录：

- `examples/libero/conf/ablation_stepchunk_vs_step_task8_xi_warmup`
- `examples/libero/conf/ablation_stepchunk_vs_step_task6_xi_warmup`
- `examples/libero/conf/ablation_stepchunk_vs_step_task6_xi_warmup_eval50`

---

## 1. 每组实验内部都在做什么

每个目录下都是同一套 `2 x 2 x 2` 矩阵，共 8 个配置：

- 维度A（残差执行粒度）：
  - `step`（`chunk_step.enabled: false`）
  - `stepchunk`（`chunk_step.enabled: true`）
- 维度B（残差强度）：
  - `xi=0.1`
  - `xi=0.5`
- 维度C（是否 warmup）：
  - `nowarmup`（`training.warmup.episodes: 0`）
  - `warmup100ep`（`training.warmup.episodes: 100`）

所以三组实验的核心都不是“单模型调参”，而是同一张 ablation 网格在不同实验条件下的对照。

---

## 2. 三组实验的目的

### 2.1 `task8_xi_warmup`

目的：在 **task8** 上完成第一轮主消融，回答主问题：

- step vs stepchunk 谁更优？
- xi 大小（0.1/0.5）如何影响收敛与稳定性？
- warmup 100 episode 是否改善训练早期表现？

这是首组主实验，给出最初结论基线。

### 2.2 `task6_xi_warmup`

目的：把同一消融矩阵迁移到 **task6** 做跨任务复核（replication）。

核心是验证：task8 上得到的趋势，在 task6 是否仍然成立。

这组与 task8 组尽量保持同构，仅替换任务语义相关内容（见第3节）。

### 2.3 `task6_xi_warmup_eval50`

目的：仍然是 task6 同一 ablation，但把异步评估触发频率从每20个训练episode降到每50个训练episode，降低评估扰动与排队压力。

这组主要回答“**结论是否对 eval 频率鲁棒**”，而不是改算法本身。

---

## 3. 三组之间的关键区别（实际配置差异）

### 3.1 `task8_xi_warmup` vs `task6_xi_warmup`

主要变化只有任务相关三项：

- `task.task_id`：`8 -> 6`
- `offline.dataset_paths`：`...libero_10_task_8 -> ...libero_10_task_6`
- `hydra.run.dir` 输出根目录：`...task8... -> ...task6...`

其余训练超参、异步评估参数（包含 `every_episodes: 20`）保持一致。

### 3.2 `task6_xi_warmup` vs `task6_xi_warmup_eval50`

功能相关变化只有一项：

- `training.async_eval.every_episodes`：`20 -> 50`

同时输出目录名切换为 `...task6_xi_warmup_eval50...`，用于和前一组结果隔离。

所以 `eval50` 组本质是“评估频率对照组”。

---

## 4. 如何理解三组的关系

建议把它们看成三层递进：

1. `task8_xi_warmup`：先得到主结论；
2. `task6_xi_warmup`：检验主结论的跨任务可迁移性；
3. `task6_xi_warmup_eval50`：检验主结论对评估调度（20 vs 50）的稳健性。

如果某个趋势在三组都一致，可信度最高；如果只在单组成立，需要进一步定位是任务差异还是评估触发差异造成的。
