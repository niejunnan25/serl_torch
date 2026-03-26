# LIBERO Usage Guide

## Overview

This pipeline uses three runtimes:

1. OpenPI server
2. LIBERO env server
3. `serl_torch` train / eval process

Default ports:

- `30000`: LIBERO env server
- `30001`: OpenPI server

## Data Layout

The expected raw HDF5 dataset layout is:

```text
<datasets_root>/
  libero_10/
    KITCHEN_SCENE3_turn_on_the_stove_and_put_the_moka_pot_on_it_demo.hdf5
    ...
```

In this workspace, the auto-resolver will pick:

```text
/vla/users/niejunnan/datasets
```

unless you override `libero_datasets_root`.

## Service Startup

### LIBERO env server

```bash
cd /vla/users/niejunnan/codebase/serl_torch/examples/libero
bash tools/serve_env.sh
```

Override port if needed:

```bash
bash tools/serve_env.sh --port 30010
```

### OpenPI server

Basic startup:

```bash
cd /vla/users/niejunnan/codebase/serl_torch/examples/libero
bash tools/serve_openpi.sh
```

This now defaults to serving the local `pi0_libero` checkpoint:

```text
/vla/users/niejunnan/openpi-assets/checkpoints/pi0_libero
```

Explicit checkpoint mode:

```bash
bash tools/serve_openpi.sh \
  --policy-config pi0_libero \
  --policy-dir /abs/path/to/checkpoint \
  --port 30001
```

Environment options:

- `OPENPI_ROOT`: override the OpenPI repo path
- `OPENPI_CONDA_ENV` / `OPENPI_CONDA_PREFIX`: optional conda activation before serving
- `--default-policy`: bypass the local checkpoint and use OpenPI's built-in default `LIBERO` policy

## HDF5 Stats

Compute stats for all tasks:

```bash
cd /vla/users/niejunnan/codebase/serl_torch/examples/libero
bash tools/compute_stats.sh --suite_name libero_10 --all
```

Compute stats for one task:

```bash
bash tools/compute_stats.sh --suite_name libero_10 --task_id 0
```

Output:

```text
data/stats/libero_10_task_0.json
```

The state vector is built as:

```text
ee_pos(3) || ee_ori(3) || gripper_states(2)
```

and the action vector is the raw 7D LIBERO action.

## HDF5 -> Offline PKL Conversion

Convert all tasks:

```bash
cd /vla/users/niejunnan/codebase/serl_torch/examples/libero
bash tools/convert_offline.sh --suite_name libero_10 --all
```

Convert one task:

```bash
bash tools/convert_offline.sh --suite_name libero_10 --task_id 0
```

Optional: precompute OpenPI base chunks during conversion:

```bash
bash tools/convert_offline.sh \
  --openpi \
  --suite_name libero_10 \
  --task_id 0
```

Output layout:

```text
data/offline/
  libero_10_task_0/
    episode_000000.pkl
    ...
    manifest.json
```

The PKL payload is a compact episode format:

- RGB frames are stored once per episode
- proprio / action / reward / done arrays are stored once per episode
- optional `base_chunks` are stored per chunk start if precomputed

## Training

### Online warmup prefill

See `online_prefill_usage.md` for how to pre-collect warmup episodes and load them into online replay (`tools/run_collect_online_prefill.sh`):

```text
examples/libero/docs/online_prefill_usage.md
```

### Online only

```bash
cd /vla/users/niejunnan/codebase/serl_torch/examples/libero
bash tools/train.sh \
  task.suite_name=libero_10 \
  task.task_id=0
```

### Online + offline mixed replay

```bash
cd /vla/users/niejunnan/codebase/serl_torch/examples/libero
bash tools/train.sh \
  task.suite_name=libero_10 \
  task.task_id=0 \
  offline.enabled=true \
  offline.dataset_paths='[data/offline/libero_10_task_0]' \
  offline.ratio=0.5 \
  normalization.enabled=true
```

### Enable bootstrap

Bootstrap means collecting successful base-policy rollouts first and inserting zero-residual transitions into the offline buffer.

```bash
cd /vla/users/niejunnan/codebase/serl_torch/examples/libero
bash tools/train.sh \
  task.suite_name=libero_10 \
  task.task_id=0 \
  offline.enabled=true \
  offline.dataset_paths='[data/offline/libero_10_task_0]' \
  offline.bootstrap_base.enabled=true \
  offline.bootstrap_base.success_episodes=20
```

Important config fields:

- `offline.enabled`: enable offline replay loading
- `offline.dataset_paths`: list of offline episode dirs or PKL files
- `offline.ratio`: fraction of each batch sampled from offline replay
- `offline.symmetric_replay`: if true, force 1:1 online/offline batches
- `offline.bootstrap_base.enabled`: enable base-policy bootstrap collection
- `training.calql_pretrain`: optional critic-only warm start on offline replay before online training
- `normalization.enabled`: load `data/stats/<task_key>.json`

## Evaluation

```bash
cd /vla/users/niejunnan/codebase/serl_torch/examples/libero
bash tools/eval.sh \
  task.task_id=0 \
  eval.checkpoint_path=/abs/path/to/checkpoint_0005000.pt
```

## Common Overrides

Use a custom datasets root:

```bash
bash tools/train.sh libero_datasets_root=/custom/datasets ...
```

Use a different port pair:

```bash
bash tools/serve_env.sh --port 30010
bash tools/serve_openpi.sh --port 30011
bash tools/train.sh env.remote.port=30010 openpi.port=30011
```

## Notes on Seeding

The current `serl_torch` LIBERO wrapper still resets using the per-episode `seed` passed by training / evaluation. That means the sampling distribution is controlled by the trainer, not by a fixed env seed like `openpi/examples/libero/main_10.py`.

In practice:

- `main_10.py`: closer to fixed-seed benchmark playback
- `serl_torch/examples/libero`: better suited for RL training loops that want episode-wise seed control

## Recommended End-to-End Order

1. Start `tools/serve_env.sh`
2. Start `tools/serve_openpi.sh`
3. Run `tools/compute_stats.sh`
4. Run `tools/convert_offline.sh`
5. Run `tools/train.sh`
6. Run `tools/eval.sh`
