# LIBERO RLPD

这份 README 只描述当前 `examples/libero/rlpd/` 目录里的 direct-action RLPD 主线。

如果你要看 residual 主线，请回到：

- [../README.md](../README.md)

## 这条线是什么

当前这条 RLPD 线是：

- direct-action DRQ-SAC / RLPD
- actor 用 `env.step(...)`，不是 `env.step_chunk(...)`
- replay 是标准 per-step replay
- offline 数据直接保留 expert raw action
- actor / learner 双进程，通过 `runtime.role=actor|learner` 切角色

它不是 residual 训练，也不是 `run_residual_training_optimized.py` 那条 chunk-exec optimized 线。

## 目录结构

- [run_training.py](run_training.py)
  actor / learner 共用训练入口
- [prepare_offline_data.py](prepare_offline_data.py)
  direct-action offline data 准备入口
- [evaluate_checkpoint.py](evaluate_checkpoint.py)
  checkpoint eval 入口
- [process_eval_queue.py](process_eval_queue.py)
  训练期 async eval worker；通常不需要手工启动
- [config.py](config.py)
  RLPD typed config 定义与解析
- [observation.py](observation.py)
  direct observation 构造
- [offline_data.py](offline_data.py)
  raw demo -> prepared dataset，以及 prepared replay 加载
- [replay.py](replay.py)
  per-step replay 和 online/offline mixing
- [runtime.py](runtime.py)
  rollout / eval runtime helper
- [async_eval.py](async_eval.py)
  训练期 async eval runtime
- [eval_runner.py](eval_runner.py)
  direct-action eval 主循环

## 默认配置

训练配置：

- [../configs/train_rlpd.yaml](../configs/train_rlpd.yaml)
  reference 配置，默认 `sync_commit`
- [../configs/train_rlpd_optimized.yaml](../configs/train_rlpd_optimized.yaml)
  只把 transport 改成 `async_commit`

评估配置：

- [../configs/eval_rlpd.yaml](../configs/eval_rlpd.yaml)

reference 训练配置的关键默认值：

- `replay.batch_size=256`
- `offline.enabled=true`
- `offline.ratio=0.5`
- `offline.pretrain_steps=0`
- `training.training_starts=200`
- `training.random_steps=300`
- `training.critic_actor_ratio=4`
- `sac.utd_ratio=1`
- `runtime.trainer_transport.mode=sync_commit`

## 推荐启动顺序

和 residual 主线不同，当前 direct-action RLPD 不依赖 base policy server。

最常见的主线至少会开 3 个终端：

1. LIBERO env server
2. learner
3. actor

如果你要 prepare offline data，再额外开一个 prepare 终端。  
如果你启用 `training.async_eval.enabled=true`，还需要再起一个独立的 eval env server。

下面这些命令都默认假设：

- repo root 是：`/home/hello/codebase/serl_torch`
- 训练脚本运行在 `serl_torch` 这个 conda 环境
- env server 运行在能导入 LIBERO 的环境里
- 数据集根目录是：`/vla/users/niejunnan/datasets`

## 当前命令

### 0. 启动 LIBERO env server

当前 RLPD 默认 `env.backend=remote`，所以 actor 和 eval 之前要先有 env server。

```bash
source /vla/miniconda3/etc/profile.d/conda.sh
conda activate /vla/users/niejunnan/envs/libero
cd /home/hello/codebase/serl_torch
python examples/libero/scripts/serve_env.py --host 127.0.0.1 --port 30000
```

如果你启用训练期 async eval，还需要一个单独的 eval env server，例如：

```bash
source /vla/miniconda3/etc/profile.d/conda.sh
conda activate /vla/users/niejunnan/envs/libero
cd /home/hello/codebase/serl_torch
python examples/libero/scripts/serve_env.py --host 127.0.0.1 --port 30010
```

### 1. 准备 offline 数据

如果你已经有 prepared dataset，可以跳过这一步。

```bash
source /vla/miniconda3/etc/profile.d/conda.sh
conda activate serl_torch
cd /home/hello/codebase/serl_torch
python examples/libero/rlpd/prepare_offline_data.py \
  --config-name train_rlpd \
  task.suite_name=libero_10 \
  task.task_id=8 \
  offline.enabled=true \
  libero_root=/home/hello/codebase/serl_torch/third_party/LIBERO \
  libero_datasets_root=/vla/users/niejunnan/datasets
```

默认输出目录会长成：

```text
data/rlpd/offline_data/libero_10_task_8/direct_image-wrist_image_proprio
```

### 2. 启动 learner

```bash
source /vla/miniconda3/etc/profile.d/conda.sh
conda activate serl_torch
cd /home/hello/codebase/serl_torch
python examples/libero/rlpd/run_training.py \
  --config-name train_rlpd \
  runtime.role=learner \
  task.suite_name=libero_10 \
  task.task_id=8 \
  offline.enabled=true \
  offline.prepared_path=data/rlpd/offline_data/libero_10_task_8/direct_image-wrist_image_proprio \
  libero_root=/home/hello/codebase/serl_torch/third_party/LIBERO \
  libero_datasets_root=/vla/users/niejunnan/datasets \
  encoder.resnet.model_name=/home/hello/codebase/serl_torch/pretrained_models/microsoft--resnet-18
```

### 3. 启动 actor

```bash
source /vla/miniconda3/etc/profile.d/conda.sh
conda activate serl_torch
cd /home/hello/codebase/serl_torch
python examples/libero/rlpd/run_training.py \
  --config-name train_rlpd \
  runtime.role=actor \
  task.suite_name=libero_10 \
  task.task_id=8 \
  offline.enabled=true \
  offline.prepared_path=data/rlpd/offline_data/libero_10_task_8/direct_image-wrist_image_proprio \
  libero_root=/home/hello/codebase/serl_torch/third_party/LIBERO \
  libero_datasets_root=/vla/users/niejunnan/datasets \
  encoder.resnet.model_name=/home/hello/codebase/serl_torch/pretrained_models/microsoft--resnet-18
```

### 4. 启动 `async_commit` 版本

这条不是另一种算法，只是 transport 更激进。

learner:

```bash
source /vla/miniconda3/etc/profile.d/conda.sh
conda activate serl_torch
cd /home/hello/codebase/serl_torch
python examples/libero/rlpd/run_training.py \
  --config-name train_rlpd_optimized \
  runtime.role=learner \
  task.suite_name=libero_10 \
  task.task_id=8 \
  offline.enabled=true \
  offline.prepared_path=data/rlpd/offline_data/libero_10_task_8/direct_image-wrist_image_proprio \
  libero_root=/home/hello/codebase/serl_torch/third_party/LIBERO \
  libero_datasets_root=/vla/users/niejunnan/datasets \
  encoder.resnet.model_name=/home/hello/codebase/serl_torch/pretrained_models/microsoft--resnet-18
```

actor:

```bash
source /vla/miniconda3/etc/profile.d/conda.sh
conda activate serl_torch
cd /home/hello/codebase/serl_torch
python examples/libero/rlpd/run_training.py \
  --config-name train_rlpd_optimized \
  runtime.role=actor \
  task.suite_name=libero_10 \
  task.task_id=8 \
  offline.enabled=true \
  offline.prepared_path=data/rlpd/offline_data/libero_10_task_8/direct_image-wrist_image_proprio \
  libero_root=/home/hello/codebase/serl_torch/third_party/LIBERO \
  libero_datasets_root=/vla/users/niejunnan/datasets \
  encoder.resnet.model_name=/home/hello/codebase/serl_torch/pretrained_models/microsoft--resnet-18
```

### 5. 手动评估 checkpoint

当前 direct-action eval 默认要求 `eval.checkpoint_path` 必填；只有显式传
`eval.allow_random_policy=true` 才允许无 checkpoint 调试跑。

这里的 checkpoint 不是 `.pkl`，而是训练产出的 `checkpoint_*.pt`；你也可以直接传整个 checkpoint 目录。

```bash
source /vla/miniconda3/etc/profile.d/conda.sh
conda activate serl_torch
cd /home/hello/codebase/serl_torch
python examples/libero/rlpd/evaluate_checkpoint.py \
  --config-name eval_rlpd \
  task.suite_name=libero_10 \
  task.task_id=8 \
  eval.checkpoint_path=/home/hello/codebase/serl_torch/outputs/libero_rlpd/train_rlpd/2026-04-20_12-00-00/checkpoints \
  libero_root=/home/hello/codebase/serl_torch/third_party/LIBERO \
  libero_datasets_root=/vla/users/niejunnan/datasets \
  encoder.resnet.model_name=/home/hello/codebase/serl_torch/pretrained_models/microsoft--resnet-18
```

## `libero_spatial task 4` 示例

这个例子使用：

- `train env`: `127.0.0.1:30100`
- `eval env`: `127.0.0.1:30110`

1. 训练 env server

```bash
source /vla/miniconda3/etc/profile.d/conda.sh
conda activate /vla/users/niejunnan/envs/libero
cd /home/hello/codebase/serl_torch
python examples/libero/scripts/serve_env.py --host 127.0.0.1 --port 30100
```

2. eval env server

```bash
source /vla/miniconda3/etc/profile.d/conda.sh
conda activate /vla/users/niejunnan/envs/libero
cd /home/hello/codebase/serl_torch
python examples/libero/scripts/serve_env.py --host 127.0.0.1 --port 30110
```

3. 准备 offline 数据

```bash
source /vla/miniconda3/etc/profile.d/conda.sh
conda activate serl_torch
cd /home/hello/codebase/serl_torch
python examples/libero/rlpd/prepare_offline_data.py \
  --config-name train_rlpd \
  task.suite_name=libero_spatial \
  task.task_id=4 \
  offline.enabled=true \
  libero_root=/home/hello/codebase/serl_torch/third_party/LIBERO \
  libero_datasets_root=/vla/users/niejunnan/datasets
```

4. learner

```bash
source /vla/miniconda3/etc/profile.d/conda.sh
conda activate serl_torch
cd /home/hello/codebase/serl_torch
python examples/libero/rlpd/run_training.py \
  --config-name train_rlpd \
  runtime.role=learner \
  task.suite_name=libero_spatial \
  task.task_id=4 \
  offline.enabled=true \
  env.remote.port=30100 \
  training.async_eval.env.remote.port=30110 \
  offline.prepared_path=data/rlpd/offline_data/libero_spatial_task_4/direct_image-wrist_image_proprio \
  libero_root=/home/hello/codebase/serl_torch/third_party/LIBERO \
  libero_datasets_root=/vla/users/niejunnan/datasets \
  encoder.resnet.model_name=/home/hello/codebase/serl_torch/pretrained_models/microsoft--resnet-18
```

5. actor

```bash
source /vla/miniconda3/etc/profile.d/conda.sh
conda activate serl_torch
cd /home/hello/codebase/serl_torch
python examples/libero/rlpd/run_training.py \
  --config-name train_rlpd \
  runtime.role=actor \
  task.suite_name=libero_spatial \
  task.task_id=4 \
  offline.enabled=true \
  env.remote.port=30100 \
  training.async_eval.env.remote.port=30110 \
  offline.prepared_path=data/rlpd/offline_data/libero_spatial_task_4/direct_image-wrist_image_proprio \
  libero_root=/home/hello/codebase/serl_torch/third_party/LIBERO \
  libero_datasets_root=/vla/users/niejunnan/datasets \
  encoder.resnet.model_name=/home/hello/codebase/serl_torch/pretrained_models/microsoft--resnet-18
```

6. eval

```bash
source /vla/miniconda3/etc/profile.d/conda.sh
conda activate serl_torch
cd /home/hello/codebase/serl_torch
python examples/libero/rlpd/evaluate_checkpoint.py \
  --config-name eval_rlpd \
  task.suite_name=libero_spatial \
  task.task_id=4 \
  env.remote.port=30110 \
  eval.checkpoint_path=/home/hello/codebase/serl_torch/outputs/libero_rlpd/train_rlpd/2026-04-20_12-00-00/checkpoints \
  libero_root=/home/hello/codebase/serl_torch/third_party/LIBERO \
  libero_datasets_root=/vla/users/niejunnan/datasets \
  encoder.resnet.model_name=/home/hello/codebase/serl_torch/pretrained_models/microsoft--resnet-18
```

## 训练语义

actor 侧：

- 观测只保留图像和 `robot_proprio`
- policy 直接输出 env action
- 前 `training.random_steps` 步用 `[-1, 1]` 均匀随机动作 warmup
- 每一步直接写一条标准 transition 到 replay

learner 侧：

- online replay 和 offline replay 都是 per-step replay
- batch 按 `offline.ratio` 混合采样
- 每个 outer update 做
  - `critic_actor_ratio - 1` 次 `update_critics(...)`
  - 再 1 次 `update_high_utd(batch, utd_ratio)`
- 当前 reference 默认
  - `critic_actor_ratio=4`
  - `utd_ratio=1`

## async eval

如果 `training.async_eval.enabled=true`，learner 会自动拉起
[process_eval_queue.py](process_eval_queue.py)。

通常不需要手工启动它；只有你在单独调试 async eval worker 时，才需要直接运行这个脚本。

## 注意事项

- 上面的命令都按主 README 的习惯写成了：先 `source /vla/miniconda3/etc/profile.d/conda.sh`，再 `conda activate ...`，然后进入 repo root 执行。
- 当前默认 `env.backend=remote`，所以 actor 和 eval 前必须先启动 env server。
- direct-action RLPD 不需要 base policy server；这是它和 residual 主线很大的一个区别。
- `offline.enabled=true` 时，`offline.prepared_path` 必须和当前任务、obs 配置匹配。
- `libero_root` 如果不能被自动发现，就需要显式传；`libero_datasets_root` 和 `encoder.resnet.model_name` 在默认路径不对时也建议显式传。
- 这条线不依赖 base policy，也不读取 residual observation。
- `train_rlpd_optimized.yaml` 只是 transport optimized，不是 chunk-exec optimized RLPD。
- 如果你要做高吞吐 chunk 版本，建议单开一条新入口，不要直接把这条 canonical step-wise RLPD 改坏。
