# LIBERO Residual RL (OpenPI + DrQ-SAC)

This folder now provides a fuller LIBERO residual RL pipeline in `serl_torch`:

- OpenPI base-policy serving
- LIBERO env RPC serving
- online residual RL training / evaluation
- HDF5 demo stats computation
- HDF5 demo conversion to offline episode PKLs
- offline replay mixing from expert PKLs

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

### 4. Materialize offline residual-training PKLs

```bash
cd /vla/users/niejunnan/codebase/serl_torch/examples/libero
bash tools/convert_offline.sh \
  --suite_name libero_10 \
  --all \
  --residual_alpha 0.35 \
  --output_dir data/residual_training/offline_alpha035
```

This step now always requires a running OpenPI server, because `base_chunks` and
offline projected actions are materialized into the dataset itself.

If you switch any of the following, you should regenerate the offline training PKLs:

- `residual.alpha`
- the OpenPI/base policy checkpoint
- residual projection settings such as `action_indices`, `action_limits`, or `expert_reference_scale`

For online warmup / prefill data, you should also recollect or rematerialize the
episodes into the unified `libero_residual_training` format.

Example online warmup collection:

```bash
cd /vla/users/niejunnan/codebase/serl_torch/examples/libero
bash tools/collect_online_prefill.sh \
  train_residual_sac.yaml \
  --episodes 100 \
  --output_dir data/residual_training/online
```

Example for a separate `pi05` materialization root:

```bash
cd /vla/users/niejunnan/codebase/serl_torch/examples/libero
bash tools/convert_offline.sh \
  --suite_name libero_10 \
  --task_id 8 \
  --residual_alpha 0.10 \
  --output_dir data/residual_training/offline_pi05_alpha01
```

If the online warmup episodes are collected with `pi05`, store them in a separate root too:

```bash
cd /vla/users/niejunnan/codebase/serl_torch/examples/libero
bash tools/collect_online_prefill.sh \
  train_residual_sac.yaml \
  --episodes 100 \
  --output_dir data/residual_training/online_pi05
```

### 5. Train with mixed online + offline replay

```bash
cd /vla/users/niejunnan/codebase/serl_torch/examples/libero
bash tools/train.sh \
  task.suite_name=libero_10 \
  task.task_id=0 \
  offline.enabled=true \
  offline.dataset_paths='[data/residual_training/offline_alpha035/libero_10_task_0]' \
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
- When offline training is enabled, `offline.dataset_paths` must point to materialized expert PKLs.
- The default training config now runs a Cal-QL-style critic pretrain stage before online residual RL when offline data is available.
- Normalization stats are saved under `data/stats/<suite>_task_<id>.json`.




```bash
bash /vla/users/niejunnan/codebase/serl_torch/examples/libero/tools/run_train.sh train_residual_sac.yaml --gpu_id 0
bash /vla/users/niejunnan/codebase/serl_torch/examples/libero/tools/run_train.sh train_pld_residual_sac.yaml --gpu_id 1
bash /vla/users/niejunnan/codebase/serl_torch/examples/libero/tools/run_train.sh train_residual_sac_chunk_step_sequence.yaml --gpu_id 2


```
