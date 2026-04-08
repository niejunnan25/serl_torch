## LIBERO Example

这份 README 只讲一件事：

**怎么使用 `examples/libero/` 这条链路。**

如果你想看仓库整体分层，请回到：

- [README.md](../README.md)

### 这个目录负责什么

`examples/libero/` 是当前最完整的一条 example，主要负责：

- LIBERO 环境适配
- remote env server
- residual 训练 / 评测入口
- offline / online 数据准备
- 实验配置和工具脚本

公共训练逻辑已经放进：

- `serl_launcher/training/`
- `serl_launcher/residual/algorithms/`
- `serl_launcher/residual/train/`

所以这里主要保留：

- 环境层
- 用户入口
- 具体实验配置

### 目录结构

- `conf/`
  Hydra 基础配置
- `configs/`
  具体实验配置，当前主线是 `exp11`
- `runtime/`
  LIBERO 的 data/runtime bindings、obs adapter、policy adapter
- `env_wrappers/`
  本地 env、remote env、LIBERO setup 逻辑
- `scripts/`
  按工作流分组后的 Python 入口
- `tools/`
  常用 shell 包装入口
- `data/`
  真实数据、prepared 数据、stats
- `docs/`
  LIBERO 补充设计说明

### 路径假设

下面所有命令都按当前这台机器的真实路径直接写出来，不要求你先 `export` 一组环境变量。

当前假设：

- repo root:
  `/vla/users/niejunnan/codebase/serl_torch`
- LIBERO root:
  `/vla/users/niejunnan/codebase/serl_torch/third_party/LIBERO`
- LIBERO datasets:
  `/vla/users/niejunnan/datasets`
- OpenPI checkpoint:
  `/vla/users/niejunnan/openpi-assets/checkpoints/pi05_libero`

如果你的机器路径不同，**直接把命令里的绝对路径替换掉即可**。

### 运行环境

训练相关命令默认使用 `serl_torch`：

```bash
cd /vla/users/niejunnan/codebase/serl_torch
conda activate serl_torch
pip install -e serl_launcher
pip install -e /vla/users/niejunnan/codebase/openpi/packages/openpi-client
```

常用环境：

- `serl_torch`
  actor / learner / eval / 数据准备
- `libero`
  env server
- `openpi-modified`
  OpenPI 服务

### 当前推荐配置

当前最常用的 `exp11` 配置：

- 主线训练配置：
  [libero_10_task_8_chunk_async_null_utdeff1p_unfreeze_pi05.yaml](configs/exp11/chunk/libero_10_task_8_chunk_async_null_utdeff1p_unfreeze_pi05.yaml)
- 最小 async 训练入口配置：
  [libero_10_task_8_chunk_async_null.yaml](configs/exp11/chunk/libero_10_task_8_chunk_async_null.yaml)
- task9 smoke 配置：
  [libero_10_task_9_chunk_async_null_utdeff1p_unfreeze_pi05.yaml](configs/exp11/chunk/libero_10_task_9_chunk_async_null_utdeff1p_unfreeze_pi05.yaml)
- task9 `mix20` 验证配置：
  [libero_10_task_9_chunk_async_null_utdeff1p_unfreeze_pi05_mix20.yaml](configs/exp11/chunk/libero_10_task_9_chunk_async_null_utdeff1p_unfreeze_pi05_mix20.yaml)

### `scripts/` 入口说明

#### `scripts/train/`

- [run_actor.py](scripts/train/run_actor.py)
  启动 actor
- [run_learner.py](scripts/train/run_learner.py)
  启动 learner
- [launch_async_train.py](scripts/train/launch_async_train.py)
  一次性拉起 env / OpenPI / learner / actor

#### `scripts/eval/`

- [evaluate_checkpoint.py](scripts/eval/evaluate_checkpoint.py)
  对单个 checkpoint 执行评测
- [process_eval_queue.py](scripts/eval/process_eval_queue.py)
  消费训练过程里的异步评测队列

#### `scripts/data/`

- [prepare_offline_demos.py](scripts/data/prepare_offline_demos.py)
  从 LIBERO HDF5 demos 生成离线 residual-training 数据
- [collect_online_prefill.py](scripts/data/collect_online_prefill.py)
  从在线 rollout 收集 warmup / prefill 数据
- [compute_normalization_stats.py](scripts/data/compute_normalization_stats.py)
  计算 normalization 所需的 state/action 统计量

#### `scripts/services/`

- [serve_env.py](scripts/services/serve_env.py)
  启动远程 LIBERO env server

### `tools/` 入口说明

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

### LIBERO runtime config 是怎么处理的

当前 wrapper 会在运行时自动：

1. 解析 `libero_root`
2. 解析 `libero_datasets_root`
3. 生成一份给上游 LIBERO 使用的 `config.yaml`
4. 设置 `LIBERO_CONFIG_PATH`

默认生成目录不会再写进 repo，而是写到：

- `$XDG_CACHE_HOME/serl_torch/libero_config`
- 如果没有设置 `XDG_CACHE_HOME`，则回落到 `~/.cache/serl_torch/libero_config`

所以通常不需要你手工准备 `config.yaml`。

如果你换了 datasets 路径，也不需要手改 config，只要把命令里的：

- `libero_datasets_root=/vla/users/niejunnan/datasets`

或：

- `--libero_datasets_root /vla/users/niejunnan/datasets`

替换成新的路径即可。下次运行时会自动重写。

### 两种推荐用法

#### 用法 1：直接 `bash + yaml 路径`

如果你只是想最快把一轮 async 训练拉起来，用这一种就行。

这个入口会自动拉起：

- env server
- OpenPI
- learner
- actor
- 并把训练产物写到一个 Hydra run 目录

示例：

```bash
cd /vla/users/niejunnan/codebase/serl_torch/examples/libero
POLICY_CONFIG=pi05_libero \
POLICY_DIR=/vla/users/niejunnan/openpi-assets/checkpoints/pi05_libero \
bash tools/launch_async_train.sh \
  /vla/users/niejunnan/codebase/serl_torch/examples/libero/configs/exp11/chunk/libero_10_task_8_chunk_async_null.yaml \
  libero_root=/vla/users/niejunnan/codebase/serl_torch/third_party/LIBERO \
  libero_datasets_root=/vla/users/niejunnan/datasets
```

这条命令如果不显式指定 `hydra.run.dir`，默认会写到：

- `/vla/users/niejunnan/codebase/serl_torch/examples/libero/outputs/exp11/libero_10_task_8_chunk_async_null/<timestamp>/`

这个目录下面最常看的内容有：

- `launch_async_train.log`
- `support/env_server.log`
- `support/openpi_server.log`
- `support/learner.log`
- `support/actor.log`
- `actor/checkpoints/`
- `actor/summary.json`

也就是说，**默认其实已经落在 `outputs/exp11` 下面了**，只是会继续按：

- `配置名`
- `时间戳`

再分一层，避免多次运行互相覆盖。

如果你想把输出目录显式指定出来，可以再加：

```bash
hydra.run.dir=/vla/users/niejunnan/codebase/serl_torch/examples/libero/outputs/exp11/libero_10_task_8_chunk_async_null
```

这时整轮训练会明确写到：

- `/vla/users/niejunnan/codebase/serl_torch/examples/libero/outputs/exp11/libero_10_task_8_chunk_async_null/`

如果你希望仍然写到 `outputs/exp11` 下面，我更推荐这样写：

```bash
cd /vla/users/niejunnan/codebase/serl_torch/examples/libero
POLICY_CONFIG=pi05_libero \
POLICY_DIR=/vla/users/niejunnan/openpi-assets/checkpoints/pi05_libero \
bash tools/launch_async_train.sh \
  /vla/users/niejunnan/codebase/serl_torch/examples/libero/configs/exp11/chunk/libero_10_task_8_chunk_async_null.yaml \
  libero_root=/vla/users/niejunnan/codebase/serl_torch/third_party/LIBERO \
  libero_datasets_root=/vla/users/niejunnan/datasets \
  hydra.run.dir=/vla/users/niejunnan/codebase/serl_torch/examples/libero/outputs/exp11/libero_10_task_8_chunk_async_null
```

这时日志目录会是：

- `/vla/users/niejunnan/codebase/serl_torch/examples/libero/outputs/exp11/libero_10_task_8_chunk_async_null/`

里面的结构通常是：

```text
libero_10_task_8_chunk_async_null/
  launch_async_train.log
  support/
    env_server.log
    openpi_server.log
    learner.log
    actor.log
  actor/
    checkpoints/
    step_logs.jsonl
    episode_logs.jsonl
    summary.json
```

如果你**真的**把：

```bash
hydra.run.dir=/vla/users/niejunnan/codebase/serl_torch/examples/libero/outputs/exp11
```

直接设成顶层目录，那么日志和产物就会直接摊在：

- `/vla/users/niejunnan/codebase/serl_torch/examples/libero/outputs/exp11/`

也就是类似：

```text
outputs/exp11/
  launch_async_train.log
  support/
  actor/
```

这样虽然能跑，但不太推荐，因为：

- 多跑几次会混到同一个目录里
- 不同 run 的日志和 checkpoint 容易互相覆盖
- 后面回看时不容易分辨哪一轮对应哪一份配置

所以更合理的两种方式是：

- 直接不传 `hydra.run.dir`，用默认的
- 显式传一个带运行名的子目录，比如：
  - `/vla/users/niejunnan/codebase/serl_torch/examples/libero/outputs/exp11/libero_10_task_8_chunk_async_null`

如果你之后还会重复跑这个配置，记得手动换一个目录名，避免新一轮结果覆盖旧目录。

这套方式最适合：

- 先确认当前环境能不能起
- 先验证一份 yaml 能不能直接跑
- 日常最省事地起一轮训练

#### 用法 2：逐个终端启动

如果你想更清楚地控制每个组件，或者想单独观察：

- env server
- OpenPI
- learner
- actor

那就用逐终端方式。

如果你只是想确认链路是通的，最稳的是先跑一轮 task9 smoke：

1. 准备 `20` 条 offline 数据
2. 准备 `20` 条 online prefill 数据
3. 用 `mix20` 配置跑一轮 `20` episode 的 async train
4. 对训练得到的 checkpoint 跑一轮 eval smoke

下面这套命令就是已经实际跑通过的一组示例。

### 详细示例：task9 smoke / validation

#### 1. 准备 offline residual-training 数据

先起一个 OpenPI 服务：

```bash
cd /vla/users/niejunnan/codebase/serl_torch
POLICY_CONFIG=pi05_libero \
POLICY_DIR=/vla/users/niejunnan/openpi-assets/checkpoints/pi05_libero \
bash examples/libero/tools/serve_openpi.sh \
  --port 40191 \
  --gpu-id 5
```

然后在另一个终端跑 offline 数据准备：

```bash
cd /vla/users/niejunnan/codebase/serl_torch/examples/libero
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
  --libero_root /vla/users/niejunnan/codebase/serl_torch/third_party/LIBERO \
  --libero_datasets_root /vla/users/niejunnan/datasets \
  --output_dir /vla/users/niejunnan/codebase/serl_torch/examples/libero/outputs/validation/materialize_offline_task9_20ep_rlnew
```

输出目录示例：

- `examples/libero/outputs/validation/materialize_offline_task9_20ep_rlnew`
- 对应绝对路径：
  `/vla/users/niejunnan/codebase/serl_torch/examples/libero/outputs/validation/materialize_offline_task9_20ep_rlnew`

#### 2. 准备 online warmup / prefill 数据

先起一个 LIBERO env server：

```bash
cd /vla/users/niejunnan/codebase/serl_torch
bash examples/libero/tools/serve_env.sh \
  --host 127.0.0.1 \
  --port 40190
```

然后收集在线 prefill：

```bash
cd /vla/users/niejunnan/codebase/serl_torch/examples/libero
conda run -n serl_torch python scripts/data/collect_online_prefill.py \
  /vla/users/niejunnan/codebase/serl_torch/examples/libero/configs/exp11/chunk/libero_10_task_9_chunk_async_null_utdeff1p_unfreeze_pi05.yaml \
  --episodes 20 \
  --output_dir /vla/users/niejunnan/codebase/serl_torch/examples/libero/outputs/validation/materialize_online_task9_20ep_rlnew \
  env.remote.host=127.0.0.1 \
  env.remote.port=40190 \
  openpi.host=127.0.0.1 \
  openpi.port=40191 \
  offline.enabled=false \
  training.online_prefill.enabled=false \
  training.warmup.episodes=20 \
  libero_root=/vla/users/niejunnan/codebase/serl_torch/third_party/LIBERO \
  libero_datasets_root=/vla/users/niejunnan/datasets
```

输出目录示例：

- `examples/libero/outputs/validation/materialize_online_task9_20ep_rlnew`
- 对应绝对路径：
  `/vla/users/niejunnan/codebase/serl_torch/examples/libero/outputs/validation/materialize_online_task9_20ep_rlnew`

#### 3. 跑一轮 20-episode async train

使用配置：

- [libero_10_task_9_chunk_async_null_utdeff1p_unfreeze_pi05_mix20.yaml](configs/exp11/chunk/libero_10_task_9_chunk_async_null_utdeff1p_unfreeze_pi05_mix20.yaml)

训练命令：

```bash
cd /vla/users/niejunnan/codebase/serl_torch/examples/libero
POLICY_CONFIG=pi05_libero \
POLICY_DIR=/vla/users/niejunnan/openpi-assets/checkpoints/pi05_libero \
bash tools/launch_async_train.sh \
  /vla/users/niejunnan/codebase/serl_torch/examples/libero/configs/exp11/chunk/libero_10_task_9_chunk_async_null_utdeff1p_unfreeze_pi05_mix20.yaml \
  libero_root=/vla/users/niejunnan/codebase/serl_torch/third_party/LIBERO \
  libero_datasets_root=/vla/users/niejunnan/datasets \
  hydra.run.dir=/vla/users/niejunnan/codebase/serl_torch/examples/libero/outputs/validation/libero_10_task_9_train_mix20_example
```

这轮训练的输出目录就是：

- `/vla/users/niejunnan/codebase/serl_torch/examples/libero/outputs/validation/libero_10_task_9_train_mix20_example/`

其中最重要的产物通常在：

- `support/`
- `actor/checkpoints/`
- `actor/summary.json`
- `actor/step_logs.jsonl`
- `actor/episode_logs.jsonl`

如果 offline / online 准备和训练都放在同一张卡上，比如 `GPU5`，训练前要先停掉之前残留的 OpenPI / env 服务，避免 OOM。

#### 4. 跑一轮 checkpoint eval smoke

先起独立的 eval env / OpenPI：

```bash
cd /vla/users/niejunnan/codebase/serl_torch
bash examples/libero/tools/serve_env.sh \
  --host 127.0.0.1 \
  --port 40170
```

```bash
cd /vla/users/niejunnan/codebase/serl_torch
POLICY_CONFIG=pi05_libero \
POLICY_DIR=/vla/users/niejunnan/openpi-assets/checkpoints/pi05_libero \
bash examples/libero/tools/serve_openpi.sh \
  --port 40171 \
  --gpu-id 5
```

然后跑 eval：

```bash
cd /vla/users/niejunnan/codebase/serl_torch
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
  eval.checkpoint_path=/vla/users/niejunnan/codebase/serl_torch/examples/libero/outputs/validation/libero_10_task_9_train_mix20_rlnew_r2/actor/checkpoints/checkpoint_1600.pt \
  eval.episodes=1 \
  eval.max_env_steps_per_episode=40 \
  logging.tensorboard=false \
  hydra.run.dir=/vla/users/niejunnan/codebase/serl_torch/examples/libero/outputs/validation/libero_10_task_9_eval_smoke_from_mix20 \
  libero_root=/vla/users/niejunnan/codebase/serl_torch/third_party/LIBERO \
  libero_datasets_root=/vla/users/niejunnan/datasets
```

这轮 eval 的输出目录就是：

- `/vla/users/niejunnan/codebase/serl_torch/examples/libero/outputs/validation/libero_10_task_9_eval_smoke_from_mix20/`

最常看的是：

- `summary.json`
- `step_logs.jsonl`
- `episode_logs.jsonl`

### 日志怎么看

一轮 async train 里最常看的文件：

- `launch_async_train.log`
- `support/env_server.log`
- `support/openpi_server.log`
- `support/learner.log`
- `support/actor.log`
- `actor/step_logs.jsonl`
- `actor/episode_logs.jsonl`
- `actor/summary.json`

### 常见问题

#### 1. 找不到 LIBERO config

通常不是缺文件，而是 `libero_root` 或 `libero_datasets_root` 传错了。

默认 runtime config 会自动写到：

- `~/.cache/serl_torch/libero_config`

#### 2. 换了 datasets 路径怎么办

不用手改 `config.yaml`。

直接把命令里的：

- `libero_datasets_root=/vla/users/niejunnan/datasets`

或者：

- `--libero_datasets_root /vla/users/niejunnan/datasets`

替换成你的新路径即可。

#### 3. learner OOM

常见原因：

- 同一张卡上残留旧的 OpenPI 服务
- env / OpenPI / learner / actor 全都堆在同一张卡上

建议训练前先清理旧服务，再起新一轮。

#### 4. agentlace 偶发 timeout warning

如果看到：

- `Failed to send message ... potential timeout`

通常是 actor 到 learner 的控制面 RPC 偶发超时，不一定表示训练挂掉。先看：

- `train_env_step`
- `learner_update_steps`
- episode 是否仍在继续增长

### 当前验证状态

当前这条 LIBERO 主链已经实际验证过：

- offline data preparation
- online prefill collection
- async train smoke
- checkpoint eval smoke
