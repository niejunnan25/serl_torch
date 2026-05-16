# 个人训练备忘

下面内容只作为开发备忘，**不作为修改代码行为的依据**。

```bash
CODE_ROOT=/vla/users/niejunnan/codebase/serl_torch
LIBERO_ROOT=/vla/users/niejunnan/codebase/serl_torch/third_party/LIBERO
LIBERO_DATASETS_ROOT=/vla/users/niejunnan/datasets
OPENPI_ROOT=/vla/users/niejunnan/codebase/openpi-modified
POLICY_CONFIG=pi0_libero_baseline_10_bs32_150000
POLICY_DIR=/vla/users/niejunnan/assets/openpi-assets/serl_torch_ckpt/pi0_10000
```

进入仓库：

```bash
source /vla/miniconda3/etc/profile.d/conda.sh
conda activate serl_torch
cd /vla/users/niejunnan/codebase/serl_torch
```

新环境安装：

```bash
pip install -e ./serl_launcher
pip install -e .
```

## 启动训练

模板：

```bash
cd /vla/users/niejunnan/codebase/serl_torch

bash examples/libero/tools/launch_residual_training.sh \
  --script-id 2 \
  --config-file examples/libero/configs/spatial_4_0514_runtime/spatial4_scripts_2_alpha0p1_unfiltered_offline_noent_std1p0_ports53100.yaml \
  --output-root /vla/users/niejunnan/codebase/serl_torch/examples/libero/outputs/spatial_4_0514/spatial4_scripts_2_alpha0p1_unfiltered_offline_noent_std1p0 \
  --learner-gpu 1 \
  --actor-gpu 0 \
  --env-gpu 0 \
  --eval-env-gpu 0 \
  --policy-gpu 0 \
  --with-eval-env \
  --clean-output-dir \
  --wait-timeout-sec 600 \
  --libero-root /vla/users/niejunnan/codebase/serl_torch/third_party/LIBERO \
  --libero-datasets-root /vla/users/niejunnan/datasets \
  --openpi-root /vla/users/niejunnan/codebase/openpi-modified \
  --policy-config pi0_libero_baseline_10_bs32_150000 \
  --policy-dir /vla/users/niejunnan/assets/openpi-assets/serl_torch_ckpt/pi0_10000
```

换机器只替换：

```text
--libero-root
--libero-datasets-root
--openpi-root
--policy-config
--policy-dir
```

换实验只替换：

```text
--config-file
--output-root
--learner-gpu
--actor-gpu
--env-gpu
--eval-env-gpu
--policy-gpu
```

重跑保留 `--clean-output-dir`，续跑删掉这个参数并换新的 `--output-root`。

## 常用配置

```
examples/libero/configs/spatial_4_0514_runtime/
examples/libero/configs/long_3/
```

### Spatial Task 4（打开柜子拿黑碗放盘子上）

```text
examples/libero/configs/spatial_4_0514_runtime/spatial4_scripts_2_alpha0p1_unfiltered_offline_noent_std0p5_ports53500.yaml
examples/libero/configs/spatial_4_0514_runtime/spatial4_scripts_2_alpha0p1_unfiltered_offline_noent_std1p0_ports53100.yaml
examples/libero/configs/spatial_4_0514_runtime/spatial4_scripts_2_alpha0p1_unfiltered_offline_noent_std5p0_ports53600.yaml
examples/libero/configs/spatial_4_0514_runtime/spatial4_scripts_2_alpha0p2_unfiltered_offline_noent_std0p5_ports53200.yaml
examples/libero/configs/spatial_4_0514_runtime/spatial4_scripts_2_alpha0p2_unfiltered_offline_noent_std1p0_ports53300.yaml
examples/libero/configs/spatial_4_0514_runtime/spatial4_scripts_2_alpha0p2_unfiltered_offline_noent_std5p0_ports53400.yaml
examples/libero/configs/spatial_4_0514_runtime/spatial4_scripts_2_alpha0p5_unfiltered_offline_noent_std0p5_ports53700.yaml
examples/libero/configs/spatial_4_0514_runtime/spatial4_scripts_2_alpha0p5_unfiltered_offline_noent_std1p0_ports53800.yaml
examples/libero/configs/spatial_4_0514_runtime/spatial4_scripts_2_alpha0p5_unfiltered_offline_noent_std5p0_ports53900.yaml
```

### 动作限幅消融

```text
examples/libero/configs/spatial_4_0514_runtime/spatial4_scripts_2_alpha0p1_unfiltered_offline_noent_std1p0_limits_xyz1_rot0p4_grip0p3_ports54100.yaml
examples/libero/configs/spatial_4_0514_runtime/spatial4_scripts_2_alpha0p2_unfiltered_offline_noent_std1p0_limits_xyz1_rot0p4_grip0p3_ports54200.yaml
examples/libero/configs/spatial_4_0514_runtime/spatial4_scripts_2_alpha0p3_unfiltered_offline_noent_std1p0_limits_xyz1_rot0p4_grip0p3_ports54300.yaml
```

### Long Task 3

```text
examples/libero/configs/long_3/long3_scripts_2_alpha0p1_unfiltered_offline_noent_std0p5_ports53500.yaml
examples/libero/configs/long_3/long3_scripts_2_alpha0p1_unfiltered_offline_noent_std1p0_ports53100.yaml
examples/libero/configs/long_3/long3_scripts_2_alpha0p1_unfiltered_offline_noent_std5p0_ports53600.yaml
examples/libero/configs/long_3/long3_scripts_2_alpha0p2_unfiltered_offline_noent_std0p5_ports53200.yaml
examples/libero/configs/long_3/long3_scripts_2_alpha0p2_unfiltered_offline_noent_std1p0_ports53300.yaml
examples/libero/configs/long_3/long3_scripts_2_alpha0p2_unfiltered_offline_noent_std5p0_ports53400.yaml
examples/libero/configs/long_3/long3_scripts_2_alpha0p5_unfiltered_offline_noent_std0p5_ports53700.yaml
examples/libero/configs/long_3/long3_scripts_2_alpha0p5_unfiltered_offline_noent_std1p0_ports53800.yaml
examples/libero/configs/long_3/long3_scripts_2_alpha0p5_unfiltered_offline_noent_std5p0_ports53900.yaml
```

## 准备离线数据

启动 Policy Server：

```bash
cd /vla/users/niejunnan/codebase/serl_torch

LOG_DIR=/vla/users/niejunnan/codebase/serl_torch/examples/libero/outputs/offline_prepare/policy
mkdir -p "${LOG_DIR}"

bash examples/libero/tools/serve_openpi_10000_policy.sh \
  --port 55101 \
  --gpu-id 6 \
  2>&1 | tee "${LOG_DIR}/policy_gpu6_port55101.log"
```

另一个终端跑 prepare：

```bash
source /vla/miniconda3/etc/profile.d/conda.sh
conda activate serl_torch
cd /vla/users/niejunnan/codebase/serl_torch

LOG_ROOT=/vla/users/niejunnan/codebase/serl_torch/examples/libero/outputs/offline_prepare/alpha0p1
mkdir -p "${LOG_ROOT}"

python examples/libero/scripts/run_residual_offline_prepare.py \
  --config-name spatial_4_0514_runtime/spatial4_scripts_2_alpha0p1_unfiltered_offline_noent_std1p0_ports53100 \
  policy.host=127.0.0.1 \
  policy.port=55101 \
  offline.prepared_path=null \
  libero_root=/vla/users/niejunnan/codebase/serl_torch/third_party/LIBERO \
  libero_datasets_root=/vla/users/niejunnan/datasets \
  hydra.run.dir="${LOG_ROOT}" \
  2>&1 | tee "${LOG_ROOT}/prepare.log"
```

换 alpha：

```text
--config-name
LOG_ROOT
policy.port
```

换 prepared replay 输出位置：

```text
offline.prepare.output_root=/abs/path/to/offline_data_root
```

## 查看日志

```bash
RUN_ROOT=/vla/users/niejunnan/codebase/serl_torch/examples/libero/outputs/spatial_4_0514/spatial4_scripts_2_alpha0p1_unfiltered_offline_noent_std1p0

# key log files:
# ${RUN_ROOT}/services/train_env.log
# ${RUN_ROOT}/services/eval_env.log
# ${RUN_ROOT}/services/policy.log
# ${RUN_ROOT}/learner/launcher.log
# ${RUN_ROOT}/learner/run_residual_training_2_chunk_local.log
# ${RUN_ROOT}/learner/learner_timers.jsonl
# ${RUN_ROOT}/learner/async_eval_results.jsonl
# ${RUN_ROOT}/actor/launcher.log
# ${RUN_ROOT}/actor/run_residual_training_2_chunk_local.log
# ${RUN_ROOT}/actor/actor_timers.jsonl
# ${RUN_ROOT}/actor/episode_logs.jsonl
```

常用监控：

```bash
# learner
tail -f "${RUN_ROOT}/learner/run_residual_training_2_chunk_local.log"

# actor
tail -f "${RUN_ROOT}/actor/run_residual_training_2_chunk_local.log"

# 成功率
tail -f "${RUN_ROOT}/actor/episode_logs.jsonl"

# 异步评估
tail -f "${RUN_ROOT}/learner/async_eval_results.jsonl"
```

## 停止训练

```bash
cd /vla/users/niejunnan/codebase/serl_torch

bash examples/libero/tools/stop_launched_training.sh \
  --output-root /vla/users/niejunnan/codebase/serl_torch/examples/libero/outputs/spatial_4_0514/spatial4_scripts_2_alpha0p1_unfiltered_offline_noent_std1p0
```

## 单独评估 checkpoint

先启动 env server 和 policy server，再：

```bash
source /vla/miniconda3/etc/profile.d/conda.sh
conda activate serl_torch
cd /vla/users/niejunnan/codebase/serl_torch

python examples/libero/scripts/run_residual_eval.py \
  eval.checkpoint_path=/abs/path/to/checkpoint_or_checkpoint_dir \
  eval.episodes=50 \
  eval.deterministic=true \
  task.suite_name=libero_spatial \
  task.task_id=4 \
  policy.host=127.0.0.1 \
  policy.port=41001 \
  env.remote.host=127.0.0.1 \
  env.remote.port=41000 \
  libero_root=/vla/users/niejunnan/codebase/serl_torch/third_party/LIBERO \
  libero_datasets_root=/vla/users/niejunnan/datasets
```

只评估 base policy：

```text
eval.checkpoint_path=null
```

## SwanLab 登录

```bash
conda activate serl_torch
swanlab login --relogin
```
