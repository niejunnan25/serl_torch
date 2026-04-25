# spatial_4 launch_residual_training commands

These commands launch every config in:

`/vla/users/niejunnan/codebase/serl_torch/examples/libero/configs/spatial_4`

GPU assignment rule:

- Each config uses two GPUs.
- The learner process uses the first GPU in the pair.
- Actor, train env, eval env, policy, and scripts_5 backfill policy all use the second GPU in the pair.
- Pairs rotate as `0,1`, `2,3`, `4,5`, `6,7`, then repeat.

If an output directory already exists and is non-empty, add either `--clean-output-dir` or `--reuse-output-dir` before running.

## scripts_2

### spatial4_scripts_2_alpha0p1_filtered_offline

GPUs: learner `0`, actor/env/policy `1`

```bash
source /vla/miniconda3/etc/profile.d/conda.sh
conda activate serl_torch
cd /vla/users/niejunnan/codebase/serl_torch
bash examples/libero/tools/launch_residual_training.sh \
  --script-id 2 \
  --config-file /vla/users/niejunnan/codebase/serl_torch/examples/libero/configs/spatial_4/spatial4_scripts_2_alpha0p1_filtered_offline.yaml \
  --output-root /vla/users/niejunnan/codebase/serl_torch/examples/libero/outputs/spatial_4/spatial4_scripts_2_alpha0p1_filtered_offline \
  --learner-gpu 0 \
  --actor-gpu 1 \
  --env-gpu 1 \
  --eval-env-gpu 1 \
  --policy-gpu 1 \
  --with-eval-env \
  --wait-timeout-sec 300 \
  --libero-root /vla/users/niejunnan/codebase/serl_torch/third_party/LIBERO \
  --libero-datasets-root /vla/users/niejunnan/datasets \
  --openpi-root /vla/users/niejunnan/codebase/openpi-modified \
  --policy-config pi0_libero_baseline_10_bs32_150000 \
  --policy-dir /vla/users/niejunnan/assets/openpi-assets/serl_torch_ckpt/pi0_10000
```

### spatial4_scripts_2_alpha0p1_online_only

GPUs: learner `2`, actor/env/policy `3`

```bash
source /vla/miniconda3/etc/profile.d/conda.sh
conda activate serl_torch
cd /vla/users/niejunnan/codebase/serl_torch
bash examples/libero/tools/launch_residual_training.sh \
  --script-id 2 \
  --config-file /vla/users/niejunnan/codebase/serl_torch/examples/libero/configs/spatial_4/spatial4_scripts_2_alpha0p1_online_only.yaml \
  --output-root /vla/users/niejunnan/codebase/serl_torch/examples/libero/outputs/spatial_4/spatial4_scripts_2_alpha0p1_online_only \
  --learner-gpu 2 \
  --actor-gpu 3 \
  --env-gpu 3 \
  --eval-env-gpu 3 \
  --policy-gpu 3 \
  --with-eval-env \
  --wait-timeout-sec 300 \
  --libero-root /vla/users/niejunnan/codebase/serl_torch/third_party/LIBERO \
  --libero-datasets-root /vla/users/niejunnan/datasets \
  --openpi-root /vla/users/niejunnan/codebase/openpi-modified \
  --policy-config pi0_libero_baseline_10_bs32_150000 \
  --policy-dir /vla/users/niejunnan/assets/openpi-assets/serl_torch_ckpt/pi0_10000
```

### spatial4_scripts_2_alpha0p1_unfiltered_offline

GPUs: learner `4`, actor/env/policy `5`

```bash
source /vla/miniconda3/etc/profile.d/conda.sh
conda activate serl_torch
cd /vla/users/niejunnan/codebase/serl_torch
bash examples/libero/tools/launch_residual_training.sh \
  --script-id 2 \
  --config-file /vla/users/niejunnan/codebase/serl_torch/examples/libero/configs/spatial_4/spatial4_scripts_2_alpha0p1_unfiltered_offline.yaml \
  --output-root /vla/users/niejunnan/codebase/serl_torch/examples/libero/outputs/spatial_4/spatial4_scripts_2_alpha0p1_unfiltered_offline \
  --learner-gpu 4 \
  --actor-gpu 5 \
  --env-gpu 5 \
  --eval-env-gpu 5 \
  --policy-gpu 5 \
  --with-eval-env \
  --wait-timeout-sec 300 \
  --libero-root /vla/users/niejunnan/codebase/serl_torch/third_party/LIBERO \
  --libero-datasets-root /vla/users/niejunnan/datasets \
  --openpi-root /vla/users/niejunnan/codebase/openpi-modified \
  --policy-config pi0_libero_baseline_10_bs32_150000 \
  --policy-dir /vla/users/niejunnan/assets/openpi-assets/serl_torch_ckpt/pi0_10000
```

### spatial4_scripts_2_alpha0p2_filtered_offline

GPUs: learner `6`, actor/env/policy `7`

```bash
source /vla/miniconda3/etc/profile.d/conda.sh
conda activate serl_torch
cd /vla/users/niejunnan/codebase/serl_torch
bash examples/libero/tools/launch_residual_training.sh \
  --script-id 2 \
  --config-file /vla/users/niejunnan/codebase/serl_torch/examples/libero/configs/spatial_4/spatial4_scripts_2_alpha0p2_filtered_offline.yaml \
  --output-root /vla/users/niejunnan/codebase/serl_torch/examples/libero/outputs/spatial_4/spatial4_scripts_2_alpha0p2_filtered_offline \
  --learner-gpu 6 \
  --actor-gpu 7 \
  --env-gpu 7 \
  --eval-env-gpu 7 \
  --policy-gpu 7 \
  --with-eval-env \
  --wait-timeout-sec 300 \
  --libero-root /vla/users/niejunnan/codebase/serl_torch/third_party/LIBERO \
  --libero-datasets-root /vla/users/niejunnan/datasets \
  --openpi-root /vla/users/niejunnan/codebase/openpi-modified \
  --policy-config pi0_libero_baseline_10_bs32_150000 \
  --policy-dir /vla/users/niejunnan/assets/openpi-assets/serl_torch_ckpt/pi0_10000
```

### spatial4_scripts_2_alpha0p2_online_only

GPUs: learner `0`, actor/env/policy `1`

```bash
source /vla/miniconda3/etc/profile.d/conda.sh
conda activate serl_torch
cd /vla/users/niejunnan/codebase/serl_torch
bash examples/libero/tools/launch_residual_training.sh \
  --script-id 2 \
  --config-file /vla/users/niejunnan/codebase/serl_torch/examples/libero/configs/spatial_4/spatial4_scripts_2_alpha0p2_online_only.yaml \
  --output-root /vla/users/niejunnan/codebase/serl_torch/examples/libero/outputs/spatial_4/spatial4_scripts_2_alpha0p2_online_only \
  --learner-gpu 0 \
  --actor-gpu 1 \
  --env-gpu 1 \
  --eval-env-gpu 1 \
  --policy-gpu 1 \
  --with-eval-env \
  --clean-output-dir \
  --wait-timeout-sec 300 \
  --libero-root /vla/users/niejunnan/codebase/serl_torch/third_party/LIBERO \
  --libero-datasets-root /vla/users/niejunnan/datasets \
  --openpi-root /vla/users/niejunnan/codebase/openpi-modified \
  --policy-config pi0_libero_baseline_10_bs32_150000 \
  --policy-dir /vla/users/niejunnan/assets/openpi-assets/serl_torch_ckpt/pi0_10000
```

### spatial4_scripts_2_alpha0p2_unfiltered_offline

GPUs: learner `2`, actor/env/policy `3`

```bash
source /vla/miniconda3/etc/profile.d/conda.sh
conda activate serl_torch
cd /vla/users/niejunnan/codebase/serl_torch
bash examples/libero/tools/launch_residual_training.sh \
  --script-id 2 \
  --config-file /vla/users/niejunnan/codebase/serl_torch/examples/libero/configs/spatial_4/spatial4_scripts_2_alpha0p2_unfiltered_offline.yaml \
  --output-root /vla/users/niejunnan/codebase/serl_torch/examples/libero/outputs/spatial_4/spatial4_scripts_2_alpha0p2_unfiltered_offline \
  --learner-gpu 2 \
  --actor-gpu 3 \
  --env-gpu 3 \
  --eval-env-gpu 3 \
  --policy-gpu 3 \
  --with-eval-env \
  --clean-output-dir \
  --wait-timeout-sec 300 \
  --libero-root /vla/users/niejunnan/codebase/serl_torch/third_party/LIBERO \
  --libero-datasets-root /vla/users/niejunnan/datasets \
  --openpi-root /vla/users/niejunnan/codebase/openpi-modified \
  --policy-config pi0_libero_baseline_10_bs32_150000 \
  --policy-dir /vla/users/niejunnan/assets/openpi-assets/serl_torch_ckpt/pi0_10000
```

## scripts_5

### spatial4_scripts_5_alpha0p1_filtered_offline

GPUs: learner `4`, actor/env/policy/backfill `5`

```bash
source /vla/miniconda3/etc/profile.d/conda.sh
conda activate serl_torch
cd /vla/users/niejunnan/codebase/serl_torch
bash examples/libero/tools/launch_residual_training.sh \
  --script-id 5 \
  --config-file /vla/users/niejunnan/codebase/serl_torch/examples/libero/configs/spatial_4/spatial4_scripts_5_alpha0p1_filtered_offline.yaml \
  --output-root /vla/users/niejunnan/codebase/serl_torch/examples/libero/outputs/spatial_4/spatial4_scripts_5_alpha0p1_filtered_offline \
  --learner-gpu 4 \
  --actor-gpu 5 \
  --env-gpu 5 \
  --eval-env-gpu 5 \
  --policy-gpu 5 \
  --backfill-gpu 5 \
  --with-eval-env \
  --clean-output-dir \
  --wait-timeout-sec 300 \
  --libero-root /vla/users/niejunnan/codebase/serl_torch/third_party/LIBERO \
  --libero-datasets-root /vla/users/niejunnan/datasets \
  --openpi-root /vla/users/niejunnan/codebase/openpi-modified \
  --policy-config pi0_libero_baseline_10_bs32_150000 \
  --policy-dir /vla/users/niejunnan/assets/openpi-assets/serl_torch_ckpt/pi0_10000
```

### spatial4_scripts_5_alpha0p1_online_only

GPUs: learner `6`, actor/env/policy/backfill `7`

```bash
source /vla/miniconda3/etc/profile.d/conda.sh
conda activate serl_torch
cd /vla/users/niejunnan/codebase/serl_torch
bash examples/libero/tools/launch_residual_training.sh \
  --script-id 5 \
  --config-file /vla/users/niejunnan/codebase/serl_torch/examples/libero/configs/spatial_4/spatial4_scripts_5_alpha0p1_online_only.yaml \
  --output-root /vla/users/niejunnan/codebase/serl_torch/examples/libero/outputs/spatial_4/spatial4_scripts_5_alpha0p1_online_only \
  --learner-gpu 6 \
  --actor-gpu 7 \
  --env-gpu 7 \
  --eval-env-gpu 7 \
  --policy-gpu 7 \
  --backfill-gpu 7 \
  --with-eval-env \
  --clean-output-dir \
  --wait-timeout-sec 300 \
  --libero-root /vla/users/niejunnan/codebase/serl_torch/third_party/LIBERO \
  --libero-datasets-root /vla/users/niejunnan/datasets \
  --openpi-root /vla/users/niejunnan/codebase/openpi-modified \
  --policy-config pi0_libero_baseline_10_bs32_150000 \
  --policy-dir /vla/users/niejunnan/assets/openpi-assets/serl_torch_ckpt/pi0_10000
```

### spatial4_scripts_5_alpha0p1_unfiltered_offline

GPUs: learner `0`, actor/env/policy/backfill `1`

```bash
source /vla/miniconda3/etc/profile.d/conda.sh
conda activate serl_torch
cd /vla/users/niejunnan/codebase/serl_torch
bash examples/libero/tools/launch_residual_training.sh \
  --script-id 5 \
  --config-file /vla/users/niejunnan/codebase/serl_torch/examples/libero/configs/spatial_4/spatial4_scripts_5_alpha0p1_unfiltered_offline.yaml \
  --output-root /vla/users/niejunnan/codebase/serl_torch/examples/libero/outputs/spatial_4/spatial4_scripts_5_alpha0p1_unfiltered_offline \
  --learner-gpu 0 \
  --actor-gpu 1 \
  --env-gpu 1 \
  --eval-env-gpu 1 \
  --policy-gpu 1 \
  --backfill-gpu 1 \
  --with-eval-env \
  --clean-output-dir \
  --wait-timeout-sec 300 \
  --libero-root /vla/users/niejunnan/codebase/serl_torch/third_party/LIBERO \
  --libero-datasets-root /vla/users/niejunnan/datasets \
  --openpi-root /vla/users/niejunnan/codebase/openpi-modified \
  --policy-config pi0_libero_baseline_10_bs32_150000 \
  --policy-dir /vla/users/niejunnan/assets/openpi-assets/serl_torch_ckpt/pi0_10000
```

### spatial4_scripts_5_alpha0p2_filtered_offline

GPUs: learner `2`, actor/env/policy/backfill `3`

```bash
source /vla/miniconda3/etc/profile.d/conda.sh
conda activate serl_torch
cd /vla/users/niejunnan/codebase/serl_torch
bash examples/libero/tools/launch_residual_training.sh \
  --script-id 5 \
  --config-file /vla/users/niejunnan/codebase/serl_torch/examples/libero/configs/spatial_4/spatial4_scripts_5_alpha0p2_filtered_offline.yaml \
  --output-root /vla/users/niejunnan/codebase/serl_torch/examples/libero/outputs/spatial_4/spatial4_scripts_5_alpha0p2_filtered_offline \
  --learner-gpu 2 \
  --actor-gpu 3 \
  --env-gpu 3 \
  --eval-env-gpu 3 \
  --policy-gpu 3 \
  --backfill-gpu 3 \
  --with-eval-env \
  --clean-output-dir \
  --wait-timeout-sec 300 \
  --libero-root /vla/users/niejunnan/codebase/serl_torch/third_party/LIBERO \
  --libero-datasets-root /vla/users/niejunnan/datasets \
  --openpi-root /vla/users/niejunnan/codebase/openpi-modified \
  --policy-config pi0_libero_baseline_10_bs32_150000 \
  --policy-dir /vla/users/niejunnan/assets/openpi-assets/serl_torch_ckpt/pi0_10000
```

### spatial4_scripts_5_alpha0p2_online_only

GPUs: learner `4`, actor/env/policy/backfill `5`

```bash
source /vla/miniconda3/etc/profile.d/conda.sh
conda activate serl_torch
cd /vla/users/niejunnan/codebase/serl_torch
bash examples/libero/tools/launch_residual_training.sh \
  --script-id 5 \
  --config-file /vla/users/niejunnan/codebase/serl_torch/examples/libero/configs/spatial_4/spatial4_scripts_5_alpha0p2_online_only.yaml \
  --output-root /vla/users/niejunnan/codebase/serl_torch/examples/libero/outputs/spatial_4/spatial4_scripts_5_alpha0p2_online_only \
  --learner-gpu 4 \
  --actor-gpu 5 \
  --env-gpu 5 \
  --eval-env-gpu 5 \
  --policy-gpu 5 \
  --backfill-gpu 5 \
  --with-eval-env \
  --clean-output-dir \
  --wait-timeout-sec 300 \
  --libero-root /vla/users/niejunnan/codebase/serl_torch/third_party/LIBERO \
  --libero-datasets-root /vla/users/niejunnan/datasets \
  --openpi-root /vla/users/niejunnan/codebase/openpi-modified \
  --policy-config pi0_libero_baseline_10_bs32_150000 \
  --policy-dir /vla/users/niejunnan/assets/openpi-assets/serl_torch_ckpt/pi0_10000
```

### spatial4_scripts_5_alpha0p2_unfiltered_offline

GPUs: learner `6`, actor/env/policy/backfill `7`

```bash
source /vla/miniconda3/etc/profile.d/conda.sh
conda activate serl_torch
cd /vla/users/niejunnan/codebase/serl_torch
bash examples/libero/tools/launch_residual_training.sh \
  --script-id 5 \
  --config-file /vla/users/niejunnan/codebase/serl_torch/examples/libero/configs/spatial_4/spatial4_scripts_5_alpha0p2_unfiltered_offline.yaml \
  --output-root /vla/users/niejunnan/codebase/serl_torch/examples/libero/outputs/spatial_4/spatial4_scripts_5_alpha0p2_unfiltered_offline \
  --learner-gpu 6 \
  --actor-gpu 7 \
  --env-gpu 7 \
  --eval-env-gpu 7 \
  --policy-gpu 7 \
  --backfill-gpu 7 \
  --with-eval-env \
  --clean-output-dir \
  --wait-timeout-sec 300 \
  --libero-root /vla/users/niejunnan/codebase/serl_torch/third_party/LIBERO \
  --libero-datasets-root /vla/users/niejunnan/datasets \
  --openpi-root /vla/users/niejunnan/codebase/openpi-modified \
  --policy-config pi0_libero_baseline_10_bs32_150000 \
  --policy-dir /vla/users/niejunnan/assets/openpi-assets/serl_torch_ckpt/pi0_10000
```
