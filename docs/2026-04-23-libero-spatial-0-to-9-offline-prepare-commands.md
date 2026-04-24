# LIBERO Spatial Task 0-9 Offline Prepare Commands

This note records the exact commands used on 2026-04-23 to regenerate filtered residual offline datasets for `libero_spatial_task_{0..9}`.

## Output Root

All prepared datasets are written under:

```bash
/vla/users/niejunnan/codebase/serl_torch/examples/libero/data/residual/offline_data
```

Each task lands at:

```bash
/vla/users/niejunnan/codebase/serl_torch/examples/libero/data/residual/offline_data/libero_spatial_task_<TASK_ID>/openpi_chunk5_alpha0p1
```

## Policy Servers

Run these in separate shells or tmux windows:

```bash
cd /vla/users/niejunnan/codebase/openpi-modified
source .venv/bin/activate
CUDA_VISIBLE_DEVICES=1 uv run scripts/serve_policy.py \
  --port 31101 \
  policy:checkpoint \
  --policy.config=pi0_libero_baseline_10_bs32_150000 \
  --policy.dir=/vla/users/niejunnan/assets/openpi-assets/serl_torch_ckpt/pi0_10000
```

```bash
cd /vla/users/niejunnan/codebase/openpi-modified
source .venv/bin/activate
CUDA_VISIBLE_DEVICES=4 uv run scripts/serve_policy.py \
  --port 31401 \
  policy:checkpoint \
  --policy.config=pi0_libero_baseline_10_bs32_150000 \
  --policy.dir=/vla/users/niejunnan/assets/openpi-assets/serl_torch_ckpt/pi0_10000
```

```bash
cd /vla/users/niejunnan/codebase/openpi-modified
source .venv/bin/activate
CUDA_VISIBLE_DEVICES=6 uv run scripts/serve_policy.py \
  --port 31601 \
  policy:checkpoint \
  --policy.config=pi0_libero_baseline_10_bs32_150000 \
  --policy.dir=/vla/users/niejunnan/assets/openpi-assets/serl_torch_ckpt/pi0_10000
```

```bash
cd /vla/users/niejunnan/codebase/openpi-modified
source .venv/bin/activate
CUDA_VISIBLE_DEVICES=7 uv run scripts/serve_policy.py \
  --port 31701 \
  policy:checkpoint \
  --policy.config=pi0_libero_baseline_10_bs32_150000 \
  --policy.dir=/vla/users/niejunnan/assets/openpi-assets/serl_torch_ckpt/pi0_10000
```

## Prepare Workers

Run these in `serl_torch` after activating the `serl_torch` conda env.

GPU 1 / port 31101:

```bash
cd /vla/users/niejunnan/codebase/serl_torch
source /vla/miniconda3/etc/profile.d/conda.sh
conda activate serl_torch
for task_id in 0 4 8; do
  python examples/libero/scripts/run_residual_offline_prepare.py \
    offline.enabled=true \
    task.suite_name=libero_spatial \
    task.task_id=$task_id \
    policy.host=127.0.0.1 \
    policy.port=31101 \
    offline.prepare.output_root=/vla/users/niejunnan/codebase/serl_torch/examples/libero/data/residual/offline_data \
    offline.prepare.filter_unrepresentable_steps=true \
    libero_root=/vla/users/niejunnan/codebase/serl_torch/third_party/LIBERO \
    libero_datasets_root=/vla/users/niejunnan/datasets
done
```

GPU 4 / port 31401:

```bash
cd /vla/users/niejunnan/codebase/serl_torch
source /vla/miniconda3/etc/profile.d/conda.sh
conda activate serl_torch
for task_id in 1 5 9; do
  python examples/libero/scripts/run_residual_offline_prepare.py \
    offline.enabled=true \
    task.suite_name=libero_spatial \
    task.task_id=$task_id \
    policy.host=127.0.0.1 \
    policy.port=31401 \
    offline.prepare.output_root=/vla/users/niejunnan/codebase/serl_torch/examples/libero/data/residual/offline_data \
    offline.prepare.filter_unrepresentable_steps=true \
    libero_root=/vla/users/niejunnan/codebase/serl_torch/third_party/LIBERO \
    libero_datasets_root=/vla/users/niejunnan/datasets
done
```

GPU 6 / port 31601:

```bash
cd /vla/users/niejunnan/codebase/serl_torch
source /vla/miniconda3/etc/profile.d/conda.sh
conda activate serl_torch
for task_id in 2 6; do
  python examples/libero/scripts/run_residual_offline_prepare.py \
    offline.enabled=true \
    task.suite_name=libero_spatial \
    task.task_id=$task_id \
    policy.host=127.0.0.1 \
    policy.port=31601 \
    offline.prepare.output_root=/vla/users/niejunnan/codebase/serl_torch/examples/libero/data/residual/offline_data \
    offline.prepare.filter_unrepresentable_steps=true \
    libero_root=/vla/users/niejunnan/codebase/serl_torch/third_party/LIBERO \
    libero_datasets_root=/vla/users/niejunnan/datasets
done
```

GPU 7 / port 31701:

```bash
cd /vla/users/niejunnan/codebase/serl_torch
source /vla/miniconda3/etc/profile.d/conda.sh
conda activate serl_torch
for task_id in 3 7; do
  python examples/libero/scripts/run_residual_offline_prepare.py \
    offline.enabled=true \
    task.suite_name=libero_spatial \
    task.task_id=$task_id \
    policy.host=127.0.0.1 \
    policy.port=31701 \
    offline.prepare.output_root=/vla/users/niejunnan/codebase/serl_torch/examples/libero/data/residual/offline_data \
    offline.prepare.filter_unrepresentable_steps=true \
    libero_root=/vla/users/niejunnan/codebase/serl_torch/third_party/LIBERO \
    libero_datasets_root=/vla/users/niejunnan/datasets
done
```

## Verification

To verify that all filtered manifests exist:

```bash
python - <<'PY'
from pathlib import Path
root = Path('/vla/users/niejunnan/codebase/serl_torch/examples/libero/data/residual/offline_data')
for task_id in range(10):
    manifest = root / f'libero_spatial_task_{task_id}' / 'openpi_chunk5_alpha0p1' / 'manifest.json'
    print(task_id, manifest.exists(), manifest)
PY
```
