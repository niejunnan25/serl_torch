# LIBERO Example

这份 README 负责回答：

1. `examples/libero/` 目录是干什么的；
2. LIBERO 这条链路要怎么真正跑起来；
3. 当前有哪些脚本、工具和推荐配置。

如果你只想快速理解仓库整体结构，请回到：

- [README.md](../README.md)

## 目录定位

`examples/libero/` 是当前最完整的一条 example 训练链路，负责：

- LIBERO 环境适配
- 远程 env server
- residual 训练 / 评测脚本入口
- offline / online 数据准备
- 实验配置

其中：

- 公共训练与算法逻辑在 `serl_launcher/`
- `examples/libero/` 自己主要保留环境层和用户入口

## 目录结构

- `conf/`
  Hydra 基础配置
- `configs/`
  具体实验配置，当前主线是 `exp11`
- `runtime/`
  LIBERO 的 data/runtime bindings、obs adapter、policy adapter
- `env_wrappers/`
  本地 env、remote env、setup 逻辑
- `scripts/`
  按工作流分组后的 Python 入口
- `tools/`
  常用 shell 启动脚本
- `data/`
  真实数据、materialized 数据、stats
- `docs/`
  LIBERO 相关补充说明

## 环境准备

### 训练环境

```bash
cd /vla/users/niejunnan/codebase/serl_torch
conda activate serl_torch
pip install -e serl_launcher
pip install -e /vla/users/niejunnan/codebase/openpi/packages/openpi-client
```

### 常用环境

- `serl_torch`
  actor / learner / eval / 数据准备
- `libero`
  env server
- `openpi-modified`
  OpenPI 服务

## 常用路径变量

推荐先准备这些环境变量：

```bash
export SERL_ROOT=/vla/users/niejunnan/codebase/serl_torch
export LIBERO_ROOT=$SERL_ROOT/third_party/LIBERO
export LIBERO_DATASETS_ROOT=/vla/users/niejunnan/datasets
export POLICY_DIR=/vla/users/niejunnan/openpi-assets/checkpoints/pi05_libero
export VALID_ROOT=$SERL_ROOT/examples/libero/outputs/validation
```

## LIBERO runtime config 是怎么处理的

当前 env wrapper 会自动根据：

- `libero_root`
- `libero_datasets_root`

生成 LIBERO runtime config，并设置 `LIBERO_CONFIG_PATH`。

默认生成目录不再写进仓库，而是写到：

- `$XDG_CACHE_HOME/serl_torch/libero_config`
- 如果没有设置 `XDG_CACHE_HOME`，则回落到 `~/.cache/serl_torch/libero_config`

这意味着：

- 通常不需要手工准备 `config.yaml`
- 也不需要在 repo 里保留 `.local/libero_config`

如果你想显式指定位置，可以：

- 传 `libero_config_dir=...`
- 或设置 `LIBERO_CONFIG_PATH=/your/path`

如果你换了 datasets 路径，也不需要手改这份 config，只要：

- 传 `libero_datasets_root=/new/path`
- 或设置 `LIBERO_DATASETS_ROOT=/new/path`

下次运行时会自动重写。

## 当前推荐配置

当前最常用的几份 `exp11` 配置：

- 主线训练配置：
  [libero_10_task_8_chunk_async_null_utdeff1p_unfreeze_pi05.yaml](configs/exp11/chunk/libero_10_task_8_chunk_async_null_utdeff1p_unfreeze_pi05.yaml)
- task9 smoke 配置：
  [libero_10_task_9_chunk_async_null_utdeff1p_unfreeze_pi05.yaml](configs/exp11/chunk/libero_10_task_9_chunk_async_null_utdeff1p_unfreeze_pi05.yaml)
- task9 20-episode 混合训练 smoke：
  [libero_10_task_9_chunk_async_null_utdeff1p_unfreeze_pi05_mix20.yaml](configs/exp11/chunk/libero_10_task_9_chunk_async_null_utdeff1p_unfreeze_pi05_mix20.yaml)

## `scripts/` 目录说明

当前 `scripts/` 已按工作流整理：

### `scripts/train/`

- [run_actor.py](scripts/train/run_actor.py)
  actor 训练入口
- [run_learner.py](scripts/train/run_learner.py)
  learner 训练入口
- [launch_async_train.py](scripts/train/launch_async_train.py)
  一次性拉起 env / OpenPI / learner / actor

### `scripts/eval/`

- [evaluate_checkpoint.py](scripts/eval/evaluate_checkpoint.py)
  对单个 checkpoint 做评测
- [process_eval_queue.py](scripts/eval/process_eval_queue.py)
  处理训练过程中排入队列的异步评测请求

### `scripts/data/`

- [collect_online_prefill.py](scripts/data/collect_online_prefill.py)
  收集在线 warmup / prefill 数据
- [prepare_offline_demos.py](scripts/data/prepare_offline_demos.py)
  从 LIBERO HDF5 demos 生成离线 residual-training 数据
- [compute_normalization_stats.py](scripts/data/compute_normalization_stats.py)
  计算 normalization 统计量

### `scripts/services/`

- [serve_env.py](scripts/services/serve_env.py)
  远程 LIBERO env server

## `tools/` 入口说明

常用 shell 包装入口：

- [serve_env.sh](tools/serve_env.sh)
- [serve_openpi.sh](tools/serve_openpi.sh)
- [collect_online_prefill.sh](tools/collect_online_prefill.sh)
- [convert_offline.sh](tools/convert_offline.sh)
- [train.sh](tools/train.sh)
- [run_actor.sh](tools/run_actor.sh)
- [run_learner.sh](tools/run_learner.sh)
- [launch_async_train.sh](tools/launch_async_train.sh)
- [eval.sh](tools/eval.sh)

## 最推荐的端到端用法

如果你只是想验证链路是否跑通，最稳的是先跑一轮 task9 smoke：

1. 准备 `20` 条 offline 数据
2. 准备 `20` 条 online prefill 数据
3. 用 `mix20` 配置跑 `20` 个 episode 的 async train
4. 用训练出来的 checkpoint 跑一轮 eval smoke

下面这套就是已经验证过的一组命令。

## 详细示例：task9 smoke / validation

### 1. 准备 offline residual-training 数据

先起一个给 offline demo 转换用的 OpenPI 服务：

```bash
cd $SERL_ROOT
POLICY_CONFIG=pi05_libero \
POLICY_DIR=$POLICY_DIR \
bash examples/libero/tools/serve_openpi.sh \
  --port 40191 \
  --gpu-id 5
```

然后在另一个终端跑 offline data preparation：

```bash
cd $SERL_ROOT/examples/libero
conda run -n serl_torch python scripts/data/prepare_offline_demos.py \
  --suite_name libero_10 \
  --task_id 9 \
  --chunk_horizon 5 \
  --max_episodes 20 \
  --policy_type openpi \
  --policy_id pi05_libero \
  --openpi_host 127.0.0.1 \
  --openpi_port 40191 \
  --residual_alpha 0.10 \
  --libero_root $LIBERO_ROOT \
  --libero_datasets_root $LIBERO_DATASETS_ROOT \
  --output_dir $VALID_ROOT/materialize_offline_task9_20ep_rlnew
```

输出目录示例：

- `examples/libero/outputs/validation/materialize_offline_task9_20ep_rlnew`

### 2. 准备 online warmup / prefill 数据

先起一个 LIBERO env server：

```bash
cd $SERL_ROOT
bash examples/libero/tools/serve_env.sh \
  --host 127.0.0.1 \
  --port 40190
```

然后收集在线 prefill：

```bash
cd $SERL_ROOT/examples/libero
conda run -n serl_torch python scripts/data/collect_online_prefill.py \
  $SERL_ROOT/examples/libero/configs/exp11/chunk/libero_10_task_9_chunk_async_null_utdeff1p_unfreeze_pi05.yaml \
  --episodes 20 \
  --output_dir $VALID_ROOT/materialize_online_task9_20ep_rlnew \
  env.remote.host=127.0.0.1 \
  env.remote.port=40190 \
  openpi.host=127.0.0.1 \
  openpi.port=40191 \
  offline.enabled=false \
  training.online_prefill.enabled=false \
  training.warmup.episodes=20 \
  libero_root=$LIBERO_ROOT \
  libero_datasets_root=$LIBERO_DATASETS_ROOT
```

输出目录示例：

- `examples/libero/outputs/validation/materialize_online_task9_20ep_rlnew`

### 3. 跑一轮 20-episode async train

使用：

- [libero_10_task_9_chunk_async_null_utdeff1p_unfreeze_pi05_mix20.yaml](configs/exp11/chunk/libero_10_task_9_chunk_async_null_utdeff1p_unfreeze_pi05_mix20.yaml)

训练命令：

```bash
cd $SERL_ROOT/examples/libero
POLICY_CONFIG=pi05_libero \
POLICY_DIR=$POLICY_DIR \
bash tools/launch_async_train.sh \
  $SERL_ROOT/examples/libero/configs/exp11/chunk/libero_10_task_9_chunk_async_null_utdeff1p_unfreeze_pi05_mix20.yaml \
  libero_root=$LIBERO_ROOT \
  libero_datasets_root=$LIBERO_DATASETS_ROOT \
  hydra.run.dir=$VALID_ROOT/libero_10_task_9_train_mix20_example
```

如果 offline / online 准备和训练都放在同一张卡上，比如 `GPU5`，训练前要先停掉之前的 OpenPI / env 服务，避免 OOM。

### 4. 跑一轮 checkpoint eval smoke

先起独立的 eval env / OpenPI：

```bash
cd $SERL_ROOT
bash examples/libero/tools/serve_env.sh \
  --host 127.0.0.1 \
  --port 40170
```

```bash
cd $SERL_ROOT
POLICY_CONFIG=pi05_libero \
POLICY_DIR=$POLICY_DIR \
bash examples/libero/tools/serve_openpi.sh \
  --port 40171 \
  --gpu-id 5
```

然后跑 eval：

```bash
cd $SERL_ROOT
conda run -n serl_torch python examples/libero/scripts/eval/evaluate_checkpoint.py \
  task.suite_name=libero_10 \
  task.task_id=9 \
  policy.type=openpi \
  policy.id=pi05_libero \
  env.backend=remote \
  env.action_dim=7 \
  env.remote.host=127.0.0.1 \
  env.remote.port=40170 \
  openpi.host=127.0.0.1 \
  openpi.port=40171 \
  residual.alpha=0.1 \
  residual.image_keys='[image,wrist_image]' \
  residual.observation.state_mode=fused \
  residual.action_mask='[true,true,true,true,true,true,true]' \
  residual.action_limits='[1.0,1.0,1.0,1.0,1.0,1.0,1.0]' \
  residual.clip_gripper=true \
  residual.chunk_horizon=5 \
  chunk_step.enabled=true \
  chunk_step.sample_stride=2 \
  chunk_step.require_full_horizon=false \
  chunk_step.pad_action_to_horizon=true \
  chunk_step.scheduler_clock=env_step \
  sac.image_keys='[image,wrist_image]' \
  sac.obs_stack_horizon=1 \
  sac.resnet.freeze_backbone=false \
  sac.temperature_init=0.01 \
  sac.backup_entropy=true \
  normalization.enabled=false \
  eval.checkpoint_path=$VALID_ROOT/libero_10_task_9_train_mix20_rlnew_r2/actor/checkpoints/checkpoint_1600.pt \
  eval.episodes=1 \
  eval.max_env_steps_per_episode=40 \
  logging.tensorboard=false \
  hydra.run.dir=$VALID_ROOT/libero_10_task_9_eval_smoke_from_mix20 \
  libero_root=$LIBERO_ROOT \
  libero_datasets_root=$LIBERO_DATASETS_ROOT
```

## 训练日志怎么看

每次训练 run 里最常看的文件：

- `launch_async_train.log`
- `support/env_server.log`
- `support/openpi_server.log`
- `support/learner.log`
- `support/actor.log`
- `actor/step_logs.jsonl`
- `actor/episode_logs.jsonl`
- `actor/summary.json`

## 常见问题

### 1. 训练 / eval 找不到 LIBERO config

通常不需要手工建 config 文件。
确认：

- `libero_root`
- `libero_datasets_root`

传得对不对即可。默认 runtime config 会写到：

- `~/.cache/serl_torch/libero_config`

### 2. 换了 datasets 路径怎么办

不用手改 config 文件。
直接：

- 传 `libero_datasets_root=/new/path`
  或
- `export LIBERO_DATASETS_ROOT=/new/path`

下次运行时会自动按新路径重写 runtime config。

### 3. learner OOM

常见原因是：

- 同一张卡上残留旧的 OpenPI 服务
- env / OpenPI / learner / actor 全都堆在一张卡上

建议训练前先清理旧服务，再起新一轮。

### 4. agentlace 偶发 timeout warning

如果看到：

- `Failed to send message ... potential timeout`

通常是 actor 到 learner 的控制面 RPC 偶发超时，不一定表示训练挂掉。先看：

- `train_env_step`
- `learner_update_steps`
- episode 是否仍在继续增长

## 当前状态

当前这条 LIBERO 主链已经验证过：

- offline data preparation
- online prefill collection
- async train smoke
- checkpoint eval smoke
