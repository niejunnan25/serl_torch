# serl_torch

`serl_torch` 是当前这套 residual RL / VLA 实验代码仓库。当前这条分支里，推荐按下面这套分层来理解代码：

- `serl_launcher/training/`：通用训练基础设施
- `serl_launcher/residual/algorithms/`：residual 算法接口与实现
- `serl_launcher/residual/train/`：residual actor / learner 编排
- `serl_launcher/policy/`：base chunk policy backend
- `examples/libero/`：LIBERO 环境适配、环境服务、数据物化、训练/评测脚本、实验配置

如果你现在要真正跑通一组实验，建议直接按下面的 `exp11 + pi05` 流程走。

## 仓库结构

- `serl_launcher/`: 公共训练、replay、policy backend、agent、residual train 编排
- `examples/libero/`: LIBERO 任务适配、环境服务、数据物化、训练/评测脚本、实验配置
- `examples/RoboTwin/`: RoboTwin 示例
- `pretrained_models/`: 本地视觉 backbone 权重
- `docs/`: 额外说明
- `third_party/`: 本地依赖或参考仓库

## 当前默认假设

这套代码当前默认假设本机存在下面这些路径：

- `LIBERO` submodule: `/vla/users/niejunnan/codebase/serl_torch/third_party/LIBERO`
- LIBERO datasets: `/vla/users/niejunnan/datasets`
- OpenPI repo: `/vla/users/niejunnan/codebase/openpi`
- OpenPI `pi05` checkpoint: `/vla/users/niejunnan/openpi-assets/checkpoints/pi05_libero`

如果你的机器路径不同，就在命令行显式覆盖：

- `libero_root=...`
- `libero_datasets_root=...`
- `OPENPI_ROOT=...`
- `POLICY_DIR=...`

### 关于 LIBERO runtime config

当前端到端训练、eval、online materialize、remote env server 都会通过
[setup.py](/vla/users/niejunnan/codebase/serl_torch/examples/libero/env_wrappers/setup.py)
自动解析 `libero_root` / `libero_datasets_root`，然后生成一个 LIBERO runtime config 目录，并设置 `LIBERO_CONFIG_PATH`。

当前默认生成位置是：

- `$XDG_CACHE_HOME/serl_torch/libero_config/`
- 如果没有设置 `XDG_CACHE_HOME`，则回落到 `~/.cache/serl_torch/libero_config/`

所以：

- **通常不需要手工准备这个配置文件**
- 它不再默认写到仓库里的 `examples/libero/.local/`
- 只有你想把 LIBERO 配置放到别的位置时，才需要显式传 `libero_config_dir=...`

## 环境准备

训练环境：

```bash
cd /vla/users/niejunnan/codebase/serl_torch
conda activate serl_torch
pip install -e /vla/users/niejunnan/codebase/serl_torch/serl_launcher
pip install -e /vla/users/niejunnan/codebase/openpi/packages/openpi-client
```

如果没有本地 ResNet-18 权重：

```bash
cd /vla/users/niejunnan/codebase/serl_torch
python tools/download_resnet.py --models microsoft/resnet-18
```

运行时一般会用到三套环境：

- `serl_torch`: actor / learner / eval / 数据物化
- `libero`: 环境服务
- `openpi-modified`: OpenPI 服务

## 当前推荐配置

当前最常用的几份 `exp11` 配置：

- 主线训练配置：
  [libero_10_task_8_chunk_async_null_utdeff1p_unfreeze_pi05.yaml](/vla/users/niejunnan/codebase/serl_torch/examples/libero/configs/exp11/chunk/libero_10_task_8_chunk_async_null_utdeff1p_unfreeze_pi05.yaml)
- task9 验证配置：
  [libero_10_task_9_chunk_async_null_utdeff1p_unfreeze_pi05.yaml](/vla/users/niejunnan/codebase/serl_torch/examples/libero/configs/exp11/chunk/libero_10_task_9_chunk_async_null_utdeff1p_unfreeze_pi05.yaml)
- task9 20-episode 离线+在线混合训练验证配置：
  [libero_10_task_9_chunk_async_null_utdeff1p_unfreeze_pi05_mix20.yaml](/vla/users/niejunnan/codebase/serl_torch/examples/libero/configs/exp11/chunk/libero_10_task_9_chunk_async_null_utdeff1p_unfreeze_pi05_mix20.yaml)

## 最推荐的端到端用法

长期训练请直接用 `launch_async_train.sh + exp11 yaml`。
如果只是想验证链路是否跑通，最稳的是先跑一轮 task9 smoke：

1. 物化 `20` 条 offline 数据
2. 物化 `20` 条 online prefill 数据
3. 用 `mix20` 配置跑 `20` 个 episode 的 async train
4. 用训练出来的 checkpoint 跑一轮 eval smoke

下面这套命令就是这次实际验证跑通的一套示例。

## 详细示例：task9 的完整 smoke / validation 流程

### 0. 先准备几个路径变量

```bash
export SERL_ROOT=/vla/users/niejunnan/codebase/serl_torch
export LIBERO_ROOT=$SERL_ROOT/third_party/LIBERO
export LIBERO_DATASETS_ROOT=/vla/users/niejunnan/datasets
export POLICY_DIR=/vla/users/niejunnan/openpi-assets/checkpoints/pi05_libero
export VALID_ROOT=$SERL_ROOT/examples/libero/outputs/validation
```

如果你喜欢用 `tmux`，可以新建一个 session：

```bash
tmux new -s rl-new
```

### 1. 物化 offline residual-training 数据

先起一个专门给 offline conversion 用的 OpenPI 服务：

```bash
cd $SERL_ROOT
POLICY_CONFIG=pi05_libero \
POLICY_DIR=$POLICY_DIR \
bash examples/libero/tools/serve_openpi.sh \
  --port 40191 \
  --gpu-id 5
```

然后在另一个终端跑 offline conversion：

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

输出目录会是：

- [materialize_offline_task9_20ep_rlnew](/vla/users/niejunnan/codebase/serl_torch/examples/libero/outputs/validation/materialize_offline_task9_20ep_rlnew)

关键文件：

- [manifest.json](/vla/users/niejunnan/codebase/serl_torch/examples/libero/outputs/validation/materialize_offline_task9_20ep_rlnew/libero_10_task_9/manifest.json)
- `episode_000000.pkl` 到 `episode_000019.pkl`

### 2. 物化 online warmup / prefill 数据

先起一个 LIBERO 环境服务：

```bash
cd $SERL_ROOT
bash examples/libero/tools/serve_env.sh \
  --host 127.0.0.1 \
  --port 40190
```

然后收集在线 warmup / prefill：

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

输出目录会是：

- [materialize_online_task9_20ep_rlnew](/vla/users/niejunnan/codebase/serl_torch/examples/libero/outputs/validation/materialize_online_task9_20ep_rlnew)

关键文件：

- [manifest.json](/vla/users/niejunnan/codebase/serl_torch/examples/libero/outputs/validation/materialize_online_task9_20ep_rlnew/libero_10_task_9/stepchunk/manifest.json)
- `episode_000000.pkl` 到 `episode_000019.pkl`

### 3. 用 20 条 offline + 20 条 online 数据拉起一轮 20-episode 训练

这里直接用：

- [libero_10_task_9_chunk_async_null_utdeff1p_unfreeze_pi05_mix20.yaml](/vla/users/niejunnan/codebase/serl_torch/examples/libero/configs/exp11/chunk/libero_10_task_9_chunk_async_null_utdeff1p_unfreeze_pi05_mix20.yaml)

这份配置已经把：

- offline dataset path
- online prefill manifest path
- 20-episode 训练 budget
- 验证用端口

都收好了。

**注意**

如果你把 offline / online 数据物化和训练都放在同一张卡上，比如 `GPU5`，那在开始训练前要先停掉上一步的 `OpenPI` 服务和环境服务；否则 GPU 上可能同时挂两套服务，导致训练时 learner OOM。

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

这条命令会自动拉起：

1. training env server
2. OpenPI server
3. standalone learner
4. actor

输出目录会类似：

- [libero_10_task_9_train_mix20_rlnew_r2](/vla/users/niejunnan/codebase/serl_torch/examples/libero/outputs/validation/libero_10_task_9_train_mix20_rlnew_r2)

这次实际验证里，训练最后的关键结果是：

- `train_env_step = 1600`
- `train_episode_id = 20`
- `learner_update_steps = 74`

关键文件：

- [summary.json](/vla/users/niejunnan/codebase/serl_torch/examples/libero/outputs/validation/libero_10_task_9_train_mix20_rlnew_r2/actor/summary.json)
- [episode_logs.jsonl](/vla/users/niejunnan/codebase/serl_torch/examples/libero/outputs/validation/libero_10_task_9_train_mix20_rlnew_r2/actor/episode_logs.jsonl)
- [checkpoint_1600.pt](/vla/users/niejunnan/codebase/serl_torch/examples/libero/outputs/validation/libero_10_task_9_train_mix20_rlnew_r2/actor/checkpoints/checkpoint_1600.pt)

### 4. 用训练出来的 checkpoint 跑一轮 eval smoke

先起一套独立的 eval env / OpenPI：

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

每次运行都会生成一个 Hydra run 目录。常看的文件：

- `launch_async_train.log`
- `support/env_server.log`
- `support/openpi_server.log`
- `support/learner.log`
- `support/actor.log`
- `actor/step_logs.jsonl`
- `actor/episode_logs.jsonl`
- `actor/summary.json`

训练启动成功后，你应该能在 `learner.log` 里看到：

- offline preload 完成
- online prefill 加载完成
- `Starting standalone agentlace learner`

在 `actor.log` 里看到：

- `Connected actor to external agentlace learner`
- `Start phase=residual_rl`

## 端到端训练里的组件关系

当前 `exp11` 这条链路里，主要组件是：

- `scripts/train/launch_async_train.py`: 一次性拉起整组训练服务
- `scripts/services/serve_env.py`: 提供远程 LIBERO 环境
- `OpenPIPolicyClient`: actor / eval / materialize 调用的 base policy client
- `scripts/train/run_actor.py`: actor 主进程，负责 rollout
- `scripts/train/run_learner.py`: learner 主进程，负责 offline preload、online replay、参数更新、checkpoint
- `agentlace_bootstrap.pkl`: actor 初始化后写出的 bootstrap，上游给 learner 启动

数据流是：

- env -> actor -> OpenPI -> actor -> env
- actor -> learner replay
- learner -> actor 参数广播

## `examples/libero/scripts` 目录说明

当前脚本按工作流分组：

- `scripts/train/run_actor.py`
  actor 训练入口，负责启动 residual rollout 主循环
- `scripts/train/run_learner.py`
  learner 训练入口，负责 offline preload、online replay、参数更新和 checkpoint
- `scripts/train/launch_async_train.py`
  一次性拉起 env server、OpenPI、learner、actor 的异步训练编排入口
- `scripts/eval/evaluate_checkpoint.py`
  对单个 residual checkpoint 做评测
- `scripts/eval/process_eval_queue.py`
  消费训练过程中排入队列的异步评测请求
- `scripts/data/collect_online_prefill.py`
  收集在线 warmup / prefill 数据
- `scripts/data/prepare_offline_demos.py`
  从 LIBERO HDF5 demos 生成离线 residual-training 数据
- `scripts/data/compute_normalization_stats.py`
  从 HDF5 demos 计算 normalization 统计量
- `scripts/services/serve_env.py`
  启动远程 LIBERO 环境服务

## 额外说明

- `examples/libero/configs/exp10/` 现在主要作为历史实验配置参考，不是这条分支的推荐主线
- `examples/libero/data/` 现在只用于放真实数据，不再放 Python 代码
- 当前 `OpenPI` 已经作为 policy backend 抽到了 `serl_launcher/policy/openpi/`
- 当前端到端链路已经验证过：
  - offline materialize
  - online materialize
  - async train
  - eval smoke

## 最短总结

如果你今天只想把 `pi05` 的一组训练真正跑起来，就按下面顺序：

1. `pip install -e serl_launcher`
2. `pip install -e /vla/users/niejunnan/codebase/openpi/packages/openpi-client`
3. 准备 offline residual-training 数据
4. 准备 online warmup / prefill 数据
5. 用 `launch_async_train.sh + exp11 yaml` 启动训练
6. 用 `scripts/eval/evaluate_checkpoint.py` 做 checkpoint smoke eval
