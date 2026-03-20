# LIBERO Residual RL (OpenPI + DrQ-SAC)

This folder now provides a fuller LIBERO residual RL pipeline in `serl_torch`:

- OpenPI base-policy serving
- LIBERO env RPC serving
- online residual RL training / evaluation
- HDF5 demo stats computation
- HDF5 demo conversion to offline episode PKLs
- offline replay mixing and base-policy bootstrap

The detailed runbook is in [`docs/usage.md`](./docs/usage.md).

## Default Ports

- LIBERO env server: `30000`
- OpenPI server: `30001`

## Quick Start

### 1. Start the LIBERO env server

```bash
cd /vla/users/niejunnan/codebase/serl_torch/examples/libero
bash tools/serve_env.sh
```

### 2. Start the OpenPI server

```bash
cd /vla/users/niejunnan/codebase/serl_torch/examples/libero
bash tools/serve_openpi.sh
```

By default this now serves the local `pi0_libero` checkpoint at:

```text
/vla/users/niejunnan/openpi-assets/checkpoints/pi0_libero
```

Or specify a checkpoint explicitly:

```bash
cd /vla/users/niejunnan/codebase/serl_torch/examples/libero
bash tools/serve_openpi.sh \
  --policy-config pi0_libero \
  --policy-dir /abs/path/to/checkpoint
```

### 3. Compute normalization stats from HDF5

```bash
cd /vla/users/niejunnan/codebase/serl_torch/examples/libero
bash tools/compute_stats.sh --suite_name libero_10 --all
```

### 4. Convert HDF5 demos to offline PKLs

```bash
cd /vla/users/niejunnan/codebase/serl_torch/examples/libero
bash tools/convert_offline.sh --suite_name libero_10 --all
```

Optional: precompute `base_chunks` with OpenPI during conversion:

```bash
cd /vla/users/niejunnan/codebase/serl_torch/examples/libero
bash tools/convert_offline.sh \
  --openpi \
  --suite_name libero_10 \
  --all
```

### 5. Train with mixed online + offline replay

```bash
cd /vla/users/niejunnan/codebase/serl_torch/examples/libero
bash tools/train.sh \
  task.suite_name=libero_10 \
  task.task_id=0 \
  offline.enabled=true \
  offline.dataset_paths='[data/offline/libero_10_task_0]' \
  normalization.enabled=true
```

### 6. Evaluate

```bash
cd /vla/users/niejunnan/codebase/serl_torch/examples/libero
bash tools/eval.sh \
  task.task_id=0 \
  eval.checkpoint_path=/abs/path/to/checkpoint_0005000.pt
```

## Notes

- `libero_datasets_root` is now configurable and auto-detects `/vla/users/niejunnan/datasets` in this workspace.
- The expected raw demo layout is `<datasets_root>/libero_10/<task_name>_demo.hdf5`.
- Offline training is opt-in via `offline.enabled=true`.
- Bootstrap is opt-in via `offline.bootstrap_base.enabled=true`.
- The default training config now runs a Cal-QL-style critic pretrain stage before online residual RL when offline data is available.
- Normalization stats are saved under `data/stats/<suite>_task_<id>.json`.




```bash
bash /vla/users/niejunnan/codebase/serl_torch/examples/libero/tools/run_train.sh train_residual_sac_xi025_mix50_calql_on_utd2_start200.yaml --gpu_id 0
bash /vla/users/niejunnan/codebase/serl_torch/examples/libero/tools/run_train.sh train_residual_sac_xi035_mix50_calql_on_utd2_start200.yaml --gpu_id 1
bash /vla/users/niejunnan/codebase/serl_torch/examples/libero/tools/run_train.sh train_residual_sac_xi050_mix50_calql_on_utd2_start200.yaml --gpu_id 2
bash /vla/users/niejunnan/codebase/serl_torch/examples/libero/tools/run_train.sh train_residual_sac_xi075_mix50_calql_on_utd2_start200.yaml --gpu_id 3
bash /vla/users/niejunnan/codebase/serl_torch/examples/libero/tools/run_train.sh train_residual_sac_xi025_mix25_calql_off_utd4_start1000.yaml --gpu_id 4
bash /vla/users/niejunnan/codebase/serl_torch/examples/libero/tools/run_train.sh train_residual_sac_xi035_mix25_calql_off_utd4_start1000.yaml --gpu_id 5
bash /vla/users/niejunnan/codebase/serl_torch/examples/libero/tools/run_train.sh train_residual_sac_xi050_mix25_calql_off_utd4_start1000.yaml --gpu_id 6
bash /vla/users/niejunnan/codebase/serl_torch/examples/libero/tools/run_train.sh train_residual_sac_xi075_mix25_calql_off_utd4_start1000.yaml --gpu_id 7


```
