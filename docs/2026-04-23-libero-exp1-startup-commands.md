# 2026-04-23 LIBERO exp1 启动命令

## 适用范围

这份文档给 `exp1` 下两组 task4 对比实验提供可直接启动的命令：

- `2_chunk optimized`
  配置：
  `examples/libero/configs/exp1/train_residual_task4_exp1_chunk_local.yaml`
- `5_split serl`
  配置：
  `examples/libero/configs/exp1/train_residual_task4_exp1_split_pipeline.yaml`

两组配置当前都已经对齐到：

- `task = libero_spatial / 4`
- `offline.enabled = true`
- `offline.prepared_path = data/residual/offline_data/libero_spatial_task_4/openpi_chunk5_alpha0p1`
- `offline.prepare.filter_unrepresentable_steps = true`
- `training.async_eval.enabled = true`
- `training.async_eval.every_episodes = 50`
- `training.async_eval.episodes = 50`
- `training.max_env_steps = 600000`
- `training.max_update_steps = 600000`

## 重要说明

1. 下面先保留一组“单实验默认端口”示例：
   - 训练 env server: `30000`
   - async eval env server: `30010`
   - decision policy server: `30001`
   - backfill policy server: `30002`
   - trainer: `5688/5689/5690`
   - processor transport: `5700`，仅 `5_split` 需要
2. 如果你要两组同时跑，不能直接复用上面这组默认端口。本文后面已经给了一套并行端口方案。
3. 你指定的 GPU 布局是：
   - `2_chunk optimized`: `GPU 0,1`
   - `5_split serl`: `GPU 1,2`
4. 这意味着两组会共享 `GPU 1`：
   - `2_chunk` 的 learner + backfill 在 `GPU 1`
   - `5_split` 的 actor + decision policy 在 `GPU 1`
5. 这套 `task4` 并行命令能跑通，但不适合作为严格公平的吞吐 benchmark；
   如果你要做严格对比，请串行跑，或者换成不共享 GPU 的拓扑。
6. 所有命令默认从 repo root `/vla/users/niejunnan/codebase/serl_torch` 运行。
7. 下面所有 Python 启动命令都显式使用：
   `source /vla/miniconda3/etc/profile.d/conda.sh && conda activate serl_torch`
8. 下面所有 policy server 命令都显式指定
   `OPENPI_ROOT=/vla/users/niejunnan/codebase/openpi-modified`，
   避免误起旧 `openpi` 仓库。
9. 下面所有训练脚本都使用 Hydra 的显式写法：
   `--config-path ../configs/exp1 --config-name ...`。
10. 如果这台机器有多人共用 WandB，不要依赖全局 `wandb login`。
    建议在你自己的 `tmux` session 里单独执行：
    `export WANDB_API_KEY=<your_key>`
    和
    `export WANDB_ENTITY=<your_username_or_team>`。
    这样这次 session 里启动的 learner 会只上传到你的账号。

## 每个 tmux session 先做一次的 WandB 隔离

下面这段只需要在你自己的 `tmux` session 里执行一次，然后这个 session
里新开的训练窗口都会继承同样的 WandB 凭据：

```bash
export WANDB_API_KEY=<your_wandb_api_key>
export WANDB_ENTITY=<your_wandb_username_or_team>
```

如果你想显式覆盖配置里的 entity，也可以在 Python 命令后面追加：

```bash
+wandb.entity=<your_wandb_username_or_team>
```

## 实验 A: `2_chunk optimized`

为了和 `5_split serl` 的部署方式更一致，这里也使用 dedicated backfill policy server。

### 终端 1: 训练 env server

```bash
cd /vla/users/niejunnan/codebase/serl_torch
bash examples/libero/tools/serve_env.sh \
  --host 127.0.0.1 \
  --port 30000
```

### 终端 2: async eval env server

```bash
cd /vla/users/niejunnan/codebase/serl_torch
bash examples/libero/tools/serve_env.sh \
  --host 127.0.0.1 \
  --port 30010
```

### 终端 3: decision policy server on `GPU 0`

```bash
cd /vla/users/niejunnan/codebase/serl_torch
OPENPI_ROOT=/vla/users/niejunnan/codebase/openpi-modified \
bash examples/libero/tools/serve_openpi_10000_policy.sh \
  --gpu-id 0 \
  --port 30001
```

### 终端 4: backfill policy server on `GPU 1`

```bash
cd /vla/users/niejunnan/codebase/serl_torch
OPENPI_ROOT=/vla/users/niejunnan/codebase/openpi-modified \
bash examples/libero/tools/serve_openpi_10000_policy.sh \
  --gpu-id 1 \
  --port 30002
```

### 终端 5: learner on `GPU 1`

```bash
cd /vla/users/niejunnan/codebase/serl_torch
source /vla/miniconda3/etc/profile.d/conda.sh
conda activate serl_torch
export CUDA_VISIBLE_DEVICES=1
python examples/libero/scripts/run_residual_training_2_chunk_local.py \
  --config-path ../configs/exp1 \
  --config-name train_residual_task4_exp1_chunk_local \
  runtime.role=learner \
  libero_root=/vla/users/niejunnan/codebase/serl_torch/third_party/LIBERO \
  libero_datasets_root=/vla/users/niejunnan/datasets
```

### 终端 6: actor on `GPU 0`

```bash
cd /vla/users/niejunnan/codebase/serl_torch
source /vla/miniconda3/etc/profile.d/conda.sh
conda activate serl_torch
export CUDA_VISIBLE_DEVICES=0
python examples/libero/scripts/run_residual_training_2_chunk_local.py \
  --config-path ../configs/exp1 \
  --config-name train_residual_task4_exp1_chunk_local \
  runtime.role=actor \
  libero_root=/vla/users/niejunnan/codebase/serl_torch/third_party/LIBERO \
  libero_datasets_root=/vla/users/niejunnan/datasets \
  backfill_policy.enabled=true \
  backfill_policy.host=127.0.0.1 \
  backfill_policy.port=30002 \
  backfill_policy.max_pending_chunks=2
```

这组配置的 WandB 名称和输出目录是：

- `wandb.exp_name=libero_spatial_task_4_exp1_chunk_local`
- `launch.output_root=outputs/exp1/libero_spatial_task4/chunk_local`
- `hydra.run.dir=outputs/exp1/libero_spatial_task4/chunk_local/<role>/<timestamp>`

## 实验 B: `5_split serl`

这组实验比 `2_chunk` 多一个 `processor` 角色，并且必须额外起 dedicated backfill policy server。

### 终端 1: 训练 env server

```bash
cd /vla/users/niejunnan/codebase/serl_torch
bash examples/libero/tools/serve_env.sh \
  --host 127.0.0.1 \
  --port 30000
```

### 终端 2: async eval env server

```bash
cd /vla/users/niejunnan/codebase/serl_torch
bash examples/libero/tools/serve_env.sh \
  --host 127.0.0.1 \
  --port 30010
```

### 终端 3: decision policy server on `GPU 1`

```bash
cd /vla/users/niejunnan/codebase/serl_torch
OPENPI_ROOT=/vla/users/niejunnan/codebase/openpi-modified \
bash examples/libero/tools/serve_openpi_10000_policy.sh \
  --gpu-id 1 \
  --port 30001
```

### 终端 4: backfill policy server on `GPU 2`

```bash
cd /vla/users/niejunnan/codebase/serl_torch
OPENPI_ROOT=/vla/users/niejunnan/codebase/openpi-modified \
bash examples/libero/tools/serve_openpi_10000_policy.sh \
  --gpu-id 2 \
  --port 30002
```

### 终端 5: learner on `GPU 2`

```bash
cd /vla/users/niejunnan/codebase/serl_torch
source /vla/miniconda3/etc/profile.d/conda.sh
conda activate serl_torch
export CUDA_VISIBLE_DEVICES=2
python examples/libero/scripts/run_residual_training_5_split_pipeline.py \
  --config-path ../configs/exp1 \
  --config-name train_residual_task4_exp1_split_pipeline \
  runtime.role=learner \
  libero_root=/vla/users/niejunnan/codebase/serl_torch/third_party/LIBERO \
  libero_datasets_root=/vla/users/niejunnan/datasets
```

### 终端 6: processor on CPU

```bash
cd /vla/users/niejunnan/codebase/serl_torch
source /vla/miniconda3/etc/profile.d/conda.sh
conda activate serl_torch
export CUDA_VISIBLE_DEVICES=
python examples/libero/scripts/run_residual_training_5_split_pipeline.py \
  --config-path ../configs/exp1 \
  --config-name train_residual_task4_exp1_split_pipeline \
  runtime.role=processor \
  libero_root=/vla/users/niejunnan/codebase/serl_torch/third_party/LIBERO \
  libero_datasets_root=/vla/users/niejunnan/datasets
```

### 终端 7: actor on `GPU 1`

```bash
cd /vla/users/niejunnan/codebase/serl_torch
source /vla/miniconda3/etc/profile.d/conda.sh
conda activate serl_torch
export CUDA_VISIBLE_DEVICES=1
python examples/libero/scripts/run_residual_training_5_split_pipeline.py \
  --config-path ../configs/exp1 \
  --config-name train_residual_task4_exp1_split_pipeline \
  runtime.role=actor \
  libero_root=/vla/users/niejunnan/codebase/serl_torch/third_party/LIBERO \
  libero_datasets_root=/vla/users/niejunnan/datasets
```

这组配置的 WandB 名称和输出目录是：

- `wandb.exp_name=libero_spatial_task_4_exp1_split_pipeline`
- `launch.output_root=outputs/exp1/libero_spatial_task4/split_pipeline`
- `hydra.run.dir=outputs/exp1/libero_spatial_task4/split_pipeline/<role>/<timestamp>`

## 如果要并行跑两组实验

下面这套命令按你的要求拆成两组并行运行：

- `2_chunk optimized`:
  - `GPU 0`: actor + decision policy
  - `GPU 1`: learner + backfill policy
- `5_split serl`:
  - `GPU 1`: actor + decision policy
  - `GPU 2`: learner + backfill policy
  - `processor`: CPU

### 并行端口规划

`2_chunk optimized` 使用：

- 训练 env server: `30000`
- async eval env server: `30010`
- decision policy server: `30001`
- backfill policy server: `30002`
- trainer: `5688/5689/5690`

`5_split serl` 使用：

- 训练 env server: `30200`
- async eval env server: `30210`
- decision policy server: `30201`
- backfill policy server: `30202`
- trainer: `5798/5799/5800`
- processor transport: `5810`

### A 组并行命令: `2_chunk optimized` on `GPU 0,1`

#### 终端 A1: 训练 env server

```bash
cd /vla/users/niejunnan/codebase/serl_torch
bash examples/libero/tools/serve_env.sh \
  --host 127.0.0.1 \
  --port 30000
```

#### 终端 A2: async eval env server

```bash
cd /vla/users/niejunnan/codebase/serl_torch
bash examples/libero/tools/serve_env.sh \
  --host 127.0.0.1 \
  --port 30010
```

#### 终端 A3: decision policy server on `GPU 0`

```bash
cd /vla/users/niejunnan/codebase/serl_torch
OPENPI_ROOT=/vla/users/niejunnan/codebase/openpi-modified \
bash examples/libero/tools/serve_openpi_10000_policy.sh \
  --gpu-id 0 \
  --port 30001
```

#### 终端 A4: backfill policy server on `GPU 1`

```bash
cd /vla/users/niejunnan/codebase/serl_torch
OPENPI_ROOT=/vla/users/niejunnan/codebase/openpi-modified \
bash examples/libero/tools/serve_openpi_10000_policy.sh \
  --gpu-id 1 \
  --port 30002
```

#### 终端 A5: learner on `GPU 1`

```bash
cd /vla/users/niejunnan/codebase/serl_torch
source /vla/miniconda3/etc/profile.d/conda.sh
conda activate serl_torch
export CUDA_VISIBLE_DEVICES=1
python examples/libero/scripts/run_residual_training_2_chunk_local.py \
  --config-path ../configs/exp1 \
  --config-name train_residual_task4_exp1_chunk_local \
  runtime.role=learner \
  policy.port=30001 \
  env.remote.port=30000 \
  training.async_eval.env.remote.port=30010 \
  runtime.trainer_port=5688 \
  runtime.broadcast_port=5689 \
  runtime.trainer_transport.data_port=5690 \
  libero_root=/vla/users/niejunnan/codebase/serl_torch/third_party/LIBERO \
  libero_datasets_root=/vla/users/niejunnan/datasets
```

#### 终端 A6: actor on `GPU 0`

```bash
cd /vla/users/niejunnan/codebase/serl_torch
source /vla/miniconda3/etc/profile.d/conda.sh
conda activate serl_torch
export CUDA_VISIBLE_DEVICES=0
python examples/libero/scripts/run_residual_training_2_chunk_local.py \
  --config-path ../configs/exp1 \
  --config-name train_residual_task4_exp1_chunk_local \
  runtime.role=actor \
  policy.port=30001 \
  env.remote.port=30000 \
  training.async_eval.env.remote.port=30010 \
  runtime.trainer_port=5688 \
  runtime.broadcast_port=5689 \
  runtime.trainer_transport.data_port=5690 \
  backfill_policy.enabled=true \
  backfill_policy.host=127.0.0.1 \
  backfill_policy.port=30002 \
  backfill_policy.max_pending_chunks=2 \
  libero_root=/vla/users/niejunnan/codebase/serl_torch/third_party/LIBERO \
  libero_datasets_root=/vla/users/niejunnan/datasets
```

### B 组并行命令: `5_split serl` on `GPU 1,2`

#### 终端 B1: 训练 env server

```bash
cd /vla/users/niejunnan/codebase/serl_torch
bash examples/libero/tools/serve_env.sh \
  --host 127.0.0.1 \
  --port 30200
```

#### 终端 B2: async eval env server

```bash
cd /vla/users/niejunnan/codebase/serl_torch
bash examples/libero/tools/serve_env.sh \
  --host 127.0.0.1 \
  --port 30210
```

#### 终端 B3: decision policy server on `GPU 1`

```bash
cd /vla/users/niejunnan/codebase/serl_torch
OPENPI_ROOT=/vla/users/niejunnan/codebase/openpi-modified \
bash examples/libero/tools/serve_openpi_10000_policy.sh \
  --gpu-id 1 \
  --port 30201
```

#### 终端 B4: backfill policy server on `GPU 2`

```bash
cd /vla/users/niejunnan/codebase/serl_torch
OPENPI_ROOT=/vla/users/niejunnan/codebase/openpi-modified \
bash examples/libero/tools/serve_openpi_10000_policy.sh \
  --gpu-id 2 \
  --port 30202
```

#### 终端 B5: learner on `GPU 2`

```bash
cd /vla/users/niejunnan/codebase/serl_torch
source /vla/miniconda3/etc/profile.d/conda.sh
conda activate serl_torch
export CUDA_VISIBLE_DEVICES=2
python examples/libero/scripts/run_residual_training_5_split_pipeline.py \
  --config-path ../configs/exp1 \
  --config-name train_residual_task4_exp1_split_pipeline \
  runtime.role=learner \
  policy.port=30201 \
  backfill_policy.port=30202 \
  env.remote.port=30200 \
  training.async_eval.env.remote.port=30210 \
  runtime.trainer_port=5798 \
  runtime.broadcast_port=5799 \
  runtime.trainer_transport.data_port=5800 \
  processor_transport.port=5810 \
  libero_root=/vla/users/niejunnan/codebase/serl_torch/third_party/LIBERO \
  libero_datasets_root=/vla/users/niejunnan/datasets
```

#### 终端 B6: processor on CPU

```bash
cd /vla/users/niejunnan/codebase/serl_torch
source /vla/miniconda3/etc/profile.d/conda.sh
conda activate serl_torch
export CUDA_VISIBLE_DEVICES=
python examples/libero/scripts/run_residual_training_5_split_pipeline.py \
  --config-path ../configs/exp1 \
  --config-name train_residual_task4_exp1_split_pipeline \
  runtime.role=processor \
  policy.port=30201 \
  backfill_policy.port=30202 \
  env.remote.port=30200 \
  training.async_eval.env.remote.port=30210 \
  runtime.trainer_port=5798 \
  runtime.broadcast_port=5799 \
  runtime.trainer_transport.data_port=5800 \
  processor_transport.port=5810 \
  libero_root=/vla/users/niejunnan/codebase/serl_torch/third_party/LIBERO \
  libero_datasets_root=/vla/users/niejunnan/datasets
```

#### 终端 B7: actor on `GPU 1`

```bash
cd /vla/users/niejunnan/codebase/serl_torch
source /vla/miniconda3/etc/profile.d/conda.sh
conda activate serl_torch
export CUDA_VISIBLE_DEVICES=1
python examples/libero/scripts/run_residual_training_5_split_pipeline.py \
  --config-path ../configs/exp1 \
  --config-name train_residual_task4_exp1_split_pipeline \
  runtime.role=actor \
  policy.port=30201 \
  backfill_policy.port=30202 \
  env.remote.port=30200 \
  training.async_eval.env.remote.port=30210 \
  runtime.trainer_port=5798 \
  runtime.broadcast_port=5799 \
  runtime.trainer_transport.data_port=5800 \
  processor_transport.port=5810 \
  libero_root=/vla/users/niejunnan/codebase/serl_torch/third_party/LIBERO \
  libero_datasets_root=/vla/users/niejunnan/datasets
```
