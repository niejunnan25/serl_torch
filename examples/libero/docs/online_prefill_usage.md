# LIBERO Online Prefill 使用说明

## 1. 目标

`online_prefill` 的目标是复用 warmup 数据，避免多卡并行时每个训练进程都重复执行 base-only warmup。

典型收益场景：
- 同一台机器并行多个实验（例如 8 卡 8 个训练任务）。
- 多个任务 warmup 配置相同（例如都需要 `warmup.episodes=100`）。

## 2. 功能行为

训练脚本支持在 warmup 前先加载预先采集好的在线数据到 online replay buffer。

核心规则：
- 当 `training.warmup.episodes > 0` 且 `training.online_prefill.enabled=true` 时：
  - 先加载 prefill 数据到 online replay。
  - 如果加载数量不足 `warmup.episodes`，剩余部分继续走运行时 warmup。
- 当 `training.warmup.episodes <= 0` 时：
  - 即使 `training.online_prefill.enabled=true`，也不会加载 prefill。
- `step` 与 `stepchunk` 数据严格区分，模式不匹配会被拒绝加载。

## 3. 数据目录约定

建议目录结构：

```text
examples/libero/data/online_prefill/
  libero_10_task_6/
    step/
      manifest.json
      episode_000000.pkl
      episode_000001.pkl
      ...
    stepchunk/
      manifest.json
      episode_000000.pkl
      ...
```

说明：
- 模式由采集配置自动决定：
  - `chunk_step.enabled=false` -> `step`
  - `chunk_step.enabled=true` -> `stepchunk`
- 仓库已默认忽略本地缓存：
  - `.gitignore` 已包含 `examples/libero/data/online_prefill/`
  - 且全局也忽略 `*.pkl`

## 4. 采集 prefill 数据

推荐使用（自动启动/复用 env 与 OpenPI 服务）：

```bash
bash /vla/users/niejunnan/codebase/serl_torch/examples/libero/tools/run_collect_online_prefill.sh \
  /vla/users/niejunnan/codebase/serl_torch/examples/libero/conf/ablation_stepchunk_vs_step_task6_alpha_warmup/train_residual_sac_ablation_step_alpha50_warmup100ep.yaml \
  --gpu_id 0 \
  --episodes 100
```

参数说明：
- 第一个位置参数：训练配置文件（文件名或绝对路径）。
- `--gpu_id`：OpenPI 服务使用的 GPU（默认 `0`）。
- `--episodes`：要采集多少个 prefill episode。默认读取 `training.warmup.episodes`。
- `--output_dir`：可选，默认写入 `examples/libero/data/online_prefill`。
- 其余参数支持 Hydra override，例如：
  - `openpi.port=30011`
  - `env.remote.port=30010`

查看帮助：

```bash
bash /vla/users/niejunnan/codebase/serl_torch/examples/libero/tools/run_collect_online_prefill.sh --help
```

如果你已经手工启动好了服务，也可以使用轻量脚本：

```bash
bash /vla/users/niejunnan/codebase/serl_torch/examples/libero/tools/collect_online_prefill.sh \
  /abs/path/to/train.yaml --episodes 100
```

## 5. 训练配置

在训练 YAML 中添加/设置：

```yaml
training:
  warmup:
    episodes: 100
  online_prefill:
    enabled: true
    dataset_paths:
      - /vla/users/niejunnan/codebase/serl_torch/examples/libero/data/online_prefill/libero_10_task_6/step

推荐做法（避免命令行覆盖导致配置混乱）：
- 复制一份专用配置，例如 `*_onlineprefill.yaml`。
- 仅在该文件里把 `online_prefill.enabled` 设为 `true`，并写死 `dataset_paths`。
- 训练时只传 YAML 路径，不再传 `training.online_prefill.*` 的 CLI override。

示例（本次 task6 stepchunk）：

```bash
bash /vla/users/niejunnan/codebase/serl_torch/examples/libero/tools/run_train.sh \
  /vla/users/niejunnan/codebase/serl_torch/examples/libero/conf/ablation_stepchunk_vs_step_task6_alpha_warmup_utd_sweep/train_residual_sac_ablation_stepchunk_alpha50_warmup100ep_utd4_onlineprefill.yaml \
  --gpu_id 0
```

`dataset_paths` 支持三种形式：
- 目录路径（含 `manifest.json` 或 `episode_*.pkl`）。
- `manifest.json` 路径。
- 单个 `episode_xxx.pkl` 路径。

## 6. 多卡并行推荐流程

以 8 卡并行为例：
1. 先按模式分别采集一次 prefill（`step` 一套，`stepchunk` 一套）。
2. 在 warmup 组配置里开启 `training.online_prefill.enabled=true` 并指向对应模式路径。
3. `nowarmup` 组保持 `warmup.episodes=0`，即使 `enabled=true` 也不会加载，保证对照实验干净。
4. 多个训练进程共享同一份 prefill 文件即可。

## 7. 如何确认加载成功

训练日志会出现类似信息：
- `online prefill: episodes_loaded=.../... files_loaded=.../... inserted=...`
- 如果只覆盖了部分 warmup，会提示还需补齐剩余 warmup episode。

训练 summary 中包含：
- `configured_warmup_episodes`
- `warmup_source`（`online_prefill` / `online_prefill+runtime` / `runtime` / `disabled`）
- `online_prefill_stats`

## 8. 常见问题

1. 报错 `mode does not match training config`
- 原因：`step` 配置用了 `stepchunk` 数据，或反之。
- 处理：按模式分开采集和配置路径。

2. 报错 `dataset_paths is empty`
- 原因：开启了 `enabled=true` 且 `warmup.episodes>0`，但没填路径。
- 处理：设置正确的 `training.online_prefill.dataset_paths`。

3. 训练仍在执行 warmup
- 原因：prefill 加载数量小于 `warmup.episodes`。
- 处理：补采更多 episode，或降低 warmup 配置。

4. `nowarmup` 实验担心被 prefill 污染
- 只要 `warmup.episodes=0`，训练逻辑会自动跳过 prefill 加载。
