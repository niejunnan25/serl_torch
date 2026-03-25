# LIBERO 三组消融实验的 Git 提交追溯（2026-03-25）

## 1. 当前仓库快照

- 当前分支：`main`
- 当前 `HEAD`：`c35783bb85f692ed7e3cada63656c6a1a22c6278`
- 对应日志命令：
  - `git rev-parse HEAD`
  - `git reflog --date=iso`

## 2. 追溯方法

本次三个输出目录内（`train_residual_sac.log` / `.hydra/*` / `summary.json`）没有直接落盘 `git sha` 字段。  
因此使用两类证据联合追溯：

1. 运行时间窗口（从输出目录日期时间提取）
2. `git reflog` 中该时间点最近一次 `HEAD` 变更记录

说明：该方法能定位“运行时仓库头指针”，但不能 100% 排除当时存在未提交本地改动。

## 3. 三组实验对应提交

### A. `ablation_stepchunk_vs_step_task8_xi_warmup`

- 目录：`examples/libero/outputs/libero/ablation_stepchunk_vs_step_task8_xi_warmup`
- 运行时间窗口：`2026-03-24 20:36:05` 到 `2026-03-24 20:37:52`（+08:00）
- `reflog` 对应 `HEAD`：`b14e73ee268a2878f183cebed44950182a0f11a3`
- 对应提交信息：`feat(libero): standardize async-eval seed and consolidate ablation configs`
- 结论：该组实验基线代码可按 `b14e73e` 追溯（高置信度）。

### B. `ablation_stepchunk_vs_step_task6_xi_warmup`

- 目录：`examples/libero/outputs/libero/ablation_stepchunk_vs_step_task6_xi_warmup`
- 运行时间窗口：`2026-03-24 21:06:19` 到 `2026-03-24 21:06:30`（+08:00）
- `reflog` 对应 `HEAD`：`2421c3902abb3c6101b9d11abbb1cf981b7e9190`
- 对应提交信息：`exp(libero): enable async training for task8 step/chunk-step ablation configs`
- 补充：`task6` 配置相关提交 `16f70cd` 的时间是 `21:08:40`，晚于该组运行时间，因此这组极可能包含“当时本地未提交配置改动”。
- 结论：代码头指针是 `2421c39`，但配置层面大概率是本地工作区版本（中高置信度）。

### C. `ablation_stepchunk_vs_step_task6_xi_warmup_eval50`

- 目录：`examples/libero/outputs/libero/ablation_stepchunk_vs_step_task6_xi_warmup_eval50`
- 运行时间：`2026-03-25 00:44:37`（+08:00）
- `reflog` 对应 `HEAD`：`c35783bb85f692ed7e3cada63656c6a1a22c6278`
- 对应变更记录：`pull origin main: Fast-forward`
- 补充：该配置目录当前是未跟踪目录（`git status` 显示 `?? examples/libero/conf/ablation_stepchunk_vs_step_task6_xi_warmup_eval50/`），所以实验配置明确来自本地未提交文件。
- 结论：代码头指针为 `c35783b`，配置来自本地未提交目录（高置信度）。

## 4. 关键证据文件

- 训练日志示例：
  - `examples/libero/outputs/libero/ablation_stepchunk_vs_step_task8_xi_warmup/*/*/*/train_residual_sac.log`
  - `examples/libero/outputs/libero/ablation_stepchunk_vs_step_task6_xi_warmup/*/*/*/train_residual_sac.log`
  - `examples/libero/outputs/libero/ablation_stepchunk_vs_step_task6_xi_warmup_eval50/*/*/*/train_residual_sac.log`
- Hydra 记录示例：
  - `.../.hydra/hydra.yaml`（包含 `runtime.output_dir`、`job.config_name`）
- Git 时序证据：
  - `git reflog --date=iso`

## 5. 后续建议（避免再次不可追溯）

建议在训练入口固定落盘一个 `git_provenance.json`（每个 run 目录），至少包含：

- `git_commit`
- `git_branch`
- `git_status_porcelain`（是否 dirty）
- `config_path`
- `launch_cmd`

这样后续每次复现实验都可以直接从输出目录拿到完整来源。
