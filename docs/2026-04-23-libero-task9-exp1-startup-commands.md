# 2026-04-23 LIBERO task9 exp1 启动命令

## 适用范围

这份文档给 `exp1` 下两组 task9 对比实验提供可直接启动的命令：

- `chunk_local`
  配置：
  `examples/libero/configs/exp1/train_residual_task9_exp1_chunk_local.yaml`
- `split_pipeline`
  配置：
  `examples/libero/configs/exp1/train_residual_task9_exp1_split_pipeline.yaml`

两组配置当前都已经对齐到：

- `task = libero_spatial / 9`
- `offline.enabled = true`
- `offline.prepared_path = data/residual/offline_data/libero_spatial_task_9/openpi_chunk5_alpha0p1`
- `offline.prepare.filter_unrepresentable_steps = true`
- `training.async_eval.enabled = true`
- `training.async_eval.every_episodes = 50`
- `training.async_eval.episodes = 50`
- `training.max_env_steps = 600000`
- `training.max_update_steps = 600000`

## 重要说明

1. 这份文档按你当前的实际 GPU 拓扑来写：
   - `chunk_local`: `GPU 4,5`
   - `split_pipeline`: `GPU 6,7`
2. 角色绑定按下面这条规则写：
   - `decision policy` 跟 `actor` 同卡
   - `backfill policy` 跟 `learner` 同卡
3. 这两组现在不共享 GPU。
4. 所有命令默认从 repo root `/vla/users/niejunnan/codebase/serl_torch` 运行。
5. 下面所有 Python 启动命令都显式使用：
   `source /vla/miniconda3/etc/profile.d/conda.sh && conda activate serl_torch`
6. 下面所有 policy server 命令都显式指定
   `OPENPI_ROOT=/vla/users/niejunnan/codebase/openpi-modified`，
   避免误起旧 `openpi` 仓库。
7. 下面所有训练脚本都使用 Hydra 的显式写法：
   `--config-path ../configs/exp1 --config-name ...`。
8. 如果这台机器有多人共用 WandB，不要依赖全局 `wandb login`。
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

## 并行端口规划

`chunk_local` 使用：

- 训练 env server: `30400`
- async eval env server: `30410`
- decision policy server: `30401`
- backfill policy server: `30402`
- trainer: `5888/5889/5890`

`split_pipeline` 使用：

- 训练 env server: `30600`
- async eval env server: `30610`
- decision policy server: `30601`
- backfill policy server: `30602`
- trainer: `5998/5999/6000`
- processor transport: `6010`

## 实验 A: `task9 chunk_local` on `GPU 4,5`

### 终端 A1: 训练 env server

```bash
cd /vla/users/niejunnan/codebase/serl_torch
bash examples/libero/tools/serve_env.sh \
  --host 127.0.0.1 \
  --port 30400
```

### 终端 A2: async eval env server

```bash
cd /vla/users/niejunnan/codebase/serl_torch
bash examples/libero/tools/serve_env.sh \
  --host 127.0.0.1 \
  --port 30410
```

### 终端 A3: decision policy server on `GPU 4`

```bash
cd /vla/users/niejunnan/codebase/serl_torch
OPENPI_ROOT=/vla/users/niejunnan/codebase/openpi-modified \
bash examples/libero/tools/serve_openpi_10000_policy.sh \
  --gpu-id 4 \
  --port 30401
```

### 终端 A4: backfill policy server on `GPU 5`

```bash
cd /vla/users/niejunnan/codebase/serl_torch
OPENPI_ROOT=/vla/users/niejunnan/codebase/openpi-modified \
bash examples/libero/tools/serve_openpi_10000_policy.sh \
  --gpu-id 5 \
  --port 30402
```

### 终端 A5: learner on `GPU 5`

```bash
cd /vla/users/niejunnan/codebase/serl_torch
source /vla/miniconda3/etc/profile.d/conda.sh
conda activate serl_torch
export CUDA_VISIBLE_DEVICES=5
python examples/libero/scripts/run_residual_training_2_chunk_local.py \
  --config-path ../configs/exp1 \
  --config-name train_residual_task9_exp1_chunk_local \
  runtime.role=learner \
  policy.port=30401 \
  env.remote.port=30400 \
  training.async_eval.env.remote.port=30410 \
  runtime.trainer_port=5888 \
  runtime.broadcast_port=5889 \
  runtime.trainer_transport.data_port=5890 \
  libero_root=/vla/users/niejunnan/codebase/serl_torch/third_party/LIBERO \
  libero_datasets_root=/vla/users/niejunnan/datasets
```

### 终端 A6: actor on `GPU 4`

```bash
cd /vla/users/niejunnan/codebase/serl_torch
source /vla/miniconda3/etc/profile.d/conda.sh
conda activate serl_torch
export CUDA_VISIBLE_DEVICES=4
python examples/libero/scripts/run_residual_training_2_chunk_local.py \
  --config-path ../configs/exp1 \
  --config-name train_residual_task9_exp1_chunk_local \
  runtime.role=actor \
  policy.port=30401 \
  env.remote.port=30400 \
  training.async_eval.env.remote.port=30410 \
  runtime.trainer_port=5888 \
  runtime.broadcast_port=5889 \
  runtime.trainer_transport.data_port=5890 \
  backfill_policy.enabled=true \
  backfill_policy.host=127.0.0.1 \
  backfill_policy.port=30402 \
  backfill_policy.max_pending_chunks=2 \
  libero_root=/vla/users/niejunnan/codebase/serl_torch/third_party/LIBERO \
  libero_datasets_root=/vla/users/niejunnan/datasets
```

这组配置的 WandB 组织方式是：

- `project=libero`
- `group=libero_spatial_task9_exp1`
- `exp_name=libero_spatial_task_9_exp1_chunk_local`
- `launch.output_root=outputs/exp1/libero_spatial_task9/chunk_local`
- `hydra.run.dir=outputs/exp1/libero_spatial_task9/chunk_local/<role>/<timestamp>`

## 实验 B: `task9 split_pipeline` on `GPU 6,7`

### 终端 B1: 训练 env server

```bash
cd /vla/users/niejunnan/codebase/serl_torch
bash examples/libero/tools/serve_env.sh \
  --host 127.0.0.1 \
  --port 30600
```

### 终端 B2: async eval env server

```bash
cd /vla/users/niejunnan/codebase/serl_torch
bash examples/libero/tools/serve_env.sh \
  --host 127.0.0.1 \
  --port 30610
```

### 终端 B3: decision policy server on `GPU 6`

```bash
cd /vla/users/niejunnan/codebase/serl_torch
OPENPI_ROOT=/vla/users/niejunnan/codebase/openpi-modified \
bash examples/libero/tools/serve_openpi_10000_policy.sh \
  --gpu-id 6 \
  --port 30601
```

### 终端 B4: backfill policy server on `GPU 7`

```bash
cd /vla/users/niejunnan/codebase/serl_torch
OPENPI_ROOT=/vla/users/niejunnan/codebase/openpi-modified \
bash examples/libero/tools/serve_openpi_10000_policy.sh \
  --gpu-id 7 \
  --port 30602
```

### 终端 B5: learner on `GPU 7`

```bash
cd /vla/users/niejunnan/codebase/serl_torch
source /vla/miniconda3/etc/profile.d/conda.sh
conda activate serl_torch
export CUDA_VISIBLE_DEVICES=7
python examples/libero/scripts/run_residual_training_5_split_pipeline.py \
  --config-path ../configs/exp1 \
  --config-name train_residual_task9_exp1_split_pipeline \
  runtime.role=learner \
  policy.port=30601 \
  backfill_policy.port=30602 \
  env.remote.port=30600 \
  training.async_eval.env.remote.port=30610 \
  runtime.trainer_port=5998 \
  runtime.broadcast_port=5999 \
  runtime.trainer_transport.data_port=6000 \
  processor_transport.port=6010 \
  libero_root=/vla/users/niejunnan/codebase/serl_torch/third_party/LIBERO \
  libero_datasets_root=/vla/users/niejunnan/datasets
```

### 终端 B6: processor on CPU

```bash
cd /vla/users/niejunnan/codebase/serl_torch
source /vla/miniconda3/etc/profile.d/conda.sh
conda activate serl_torch
export CUDA_VISIBLE_DEVICES=
python examples/libero/scripts/run_residual_training_5_split_pipeline.py \
  --config-path ../configs/exp1 \
  --config-name train_residual_task9_exp1_split_pipeline \
  runtime.role=processor \
  policy.port=30601 \
  backfill_policy.port=30602 \
  env.remote.port=30600 \
  training.async_eval.env.remote.port=30610 \
  runtime.trainer_port=5998 \
  runtime.broadcast_port=5999 \
  runtime.trainer_transport.data_port=6000 \
  processor_transport.port=6010 \
  libero_root=/vla/users/niejunnan/codebase/serl_torch/third_party/LIBERO \
  libero_datasets_root=/vla/users/niejunnan/datasets
```

### 终端 B7: actor on `GPU 6`

```bash
cd /vla/users/niejunnan/codebase/serl_torch
source /vla/miniconda3/etc/profile.d/conda.sh
conda activate serl_torch
export CUDA_VISIBLE_DEVICES=6
python examples/libero/scripts/run_residual_training_5_split_pipeline.py \
  --config-path ../configs/exp1 \
  --config-name train_residual_task9_exp1_split_pipeline \
  runtime.role=actor \
  policy.port=30601 \
  backfill_policy.port=30602 \
  env.remote.port=30600 \
  training.async_eval.env.remote.port=30610 \
  runtime.trainer_port=5998 \
  runtime.broadcast_port=5999 \
  runtime.trainer_transport.data_port=6000 \
  processor_transport.port=6010 \
  libero_root=/vla/users/niejunnan/codebase/serl_torch/third_party/LIBERO \
  libero_datasets_root=/vla/users/niejunnan/datasets
```

这组配置的 WandB 组织方式是：

- `project=libero`
- `group=libero_spatial_task9_exp1`
- `exp_name=libero_spatial_task_9_exp1_split_pipeline`
- `launch.output_root=outputs/exp1/libero_spatial_task9/split_pipeline`
- `hydra.run.dir=outputs/exp1/libero_spatial_task9/split_pipeline/<role>/<timestamp>`
