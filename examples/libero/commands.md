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
  --mode chunk \
  --config-file examples/libero/configs/spatial_4_0514_runtime/spatial4_chunk_alpha0p1_unfiltered_offline_noent_std1p0_ports53100.yaml \
  --output-root /vla/users/niejunnan/codebase/serl_torch/examples/libero/outputs/spatial_4_0514/spatial4_chunk_alpha0p1_unfiltered_offline_noent_std1p0 \
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
examples/libero/configs/spatial_4_0514_runtime/spatial4_chunk_alpha0p1_unfiltered_offline_noent_std0p5_ports53500.yaml
examples/libero/configs/spatial_4_0514_runtime/spatial4_chunk_alpha0p1_unfiltered_offline_noent_std1p0_ports53100.yaml
examples/libero/configs/spatial_4_0514_runtime/spatial4_chunk_alpha0p1_unfiltered_offline_noent_std5p0_ports53600.yaml
examples/libero/configs/spatial_4_0514_runtime/spatial4_chunk_alpha0p2_unfiltered_offline_noent_std0p5_ports53200.yaml
examples/libero/configs/spatial_4_0514_runtime/spatial4_chunk_alpha0p2_unfiltered_offline_noent_std1p0_ports53300.yaml
examples/libero/configs/spatial_4_0514_runtime/spatial4_chunk_alpha0p2_unfiltered_offline_noent_std5p0_ports53400.yaml
examples/libero/configs/spatial_4_0514_runtime/spatial4_chunk_alpha0p5_unfiltered_offline_noent_std0p5_ports53700.yaml
examples/libero/configs/spatial_4_0514_runtime/spatial4_chunk_alpha0p5_unfiltered_offline_noent_std1p0_ports53800.yaml
examples/libero/configs/spatial_4_0514_runtime/spatial4_chunk_alpha0p5_unfiltered_offline_noent_std5p0_ports53900.yaml
```

### Spatial Task 4 追加运行配置

```text
examples/libero/configs/spatial_4_0514_runtime/spatial4_chunk_alpha0p5_unfiltered_offline_noent_std0p5_ports54000.yaml
examples/libero/configs/spatial_4_0514_runtime/spatial4_chunk_alpha0p1_unfiltered_offline_noent_std0p5_optupd_ports60000.yaml
examples/libero/configs/spatial_4_0514_runtime/spatial4_chunk_alpha0p1_unfiltered_offline_noent_std1p0_optupd_ports60100.yaml
examples/libero/configs/spatial_4_0514_runtime/spatial4_chunk_alpha0p1_unfiltered_offline_noent_std5p0_optupd_ports60200.yaml
examples/libero/configs/spatial_4_0514_runtime/spatial4_chunk_alpha0p2_unfiltered_offline_noent_std0p5_optupd_ports60300.yaml
examples/libero/configs/spatial_4_0514_runtime/spatial4_chunk_alpha0p2_unfiltered_offline_noent_std1p0_optupd_ports60400.yaml
examples/libero/configs/spatial_4_0514_runtime/spatial4_chunk_alpha0p2_unfiltered_offline_noent_std5p0_optupd_ports60500.yaml
```

### 动作限幅消融

```text
examples/libero/configs/spatial_4_0514_runtime/spatial4_chunk_alpha0p1_unfiltered_offline_noent_std1p0_limits_xyz1_rot0p4_grip0p3_ports54100.yaml
examples/libero/configs/spatial_4_0514_runtime/spatial4_chunk_alpha0p2_unfiltered_offline_noent_std1p0_limits_xyz1_rot0p4_grip0p3_ports54200.yaml
examples/libero/configs/spatial_4_0514_runtime/spatial4_chunk_alpha0p3_unfiltered_offline_noent_std1p0_limits_xyz1_rot0p4_grip0p3_ports54300.yaml
examples/libero/configs/spatial_4_0514_runtime/spatial4_chunk_alpha0p5_unfiltered_offline_noent_std1p0_limits_xyz1_rot0p4_grip0p3_ports54400.yaml
examples/libero/configs/spatial_4_0514_runtime/spatial4_chunk_alpha0p5_unfiltered_offline_noent_std1p0_limits_xyz1_rot0p4_grip0p3_ports54500.yaml
```

### Long Task 3

```text
examples/libero/configs/long_3/long3_chunk_alpha0p1_unfiltered_offline_noent_std0p5_ports53500.yaml
examples/libero/configs/long_3/long3_chunk_alpha0p1_unfiltered_offline_noent_std1p0_ports53100.yaml
examples/libero/configs/long_3/long3_chunk_alpha0p1_unfiltered_offline_noent_std5p0_ports53600.yaml
examples/libero/configs/long_3/long3_chunk_alpha0p2_unfiltered_offline_noent_std0p5_ports53200.yaml
examples/libero/configs/long_3/long3_chunk_alpha0p2_unfiltered_offline_noent_std1p0_ports53300.yaml
examples/libero/configs/long_3/long3_chunk_alpha0p2_unfiltered_offline_noent_std5p0_ports53400.yaml
examples/libero/configs/long_3/long3_chunk_alpha0p5_unfiltered_offline_noent_std0p5_ports53700.yaml
examples/libero/configs/long_3/long3_chunk_alpha0p5_unfiltered_offline_noent_std1p0_ports53800.yaml
examples/libero/configs/long_3/long3_chunk_alpha0p5_unfiltered_offline_noent_std5p0_ports53900.yaml
```

### Long Task 3 动作限幅消融

```text
examples/libero/configs/long_3/long3_chunk_alpha0p1_unfiltered_offline_noent_std1p0_limits_xyz1_rot0p4_grip0p3_ports56100.yaml
examples/libero/configs/long_3/long3_chunk_alpha0p2_unfiltered_offline_noent_std1p0_limits_xyz1_rot0p4_grip0p3_ports56200.yaml
examples/libero/configs/long_3/long3_chunk_alpha0p5_unfiltered_offline_noent_std1p0_limits_xyz1_rot0p4_grip0p3_ports56500.yaml
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
  --config-name spatial_4_0514_runtime/spatial4_chunk_alpha0p1_unfiltered_offline_noent_std1p0_ports53100 \
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

### 准备 Long Task 3 动作限幅离线数据

这三组训练配置使用 `residual.action_limits=[1,1,1,0.4,0.4,0.4,0.3]`，不能复用旧的全 1.0 离线数据。先启动一个 OpenPI policy server：

```bash
source /vla/miniconda3/etc/profile.d/conda.sh
conda activate serl_torch
cd /vla/users/niejunnan/codebase/serl_torch

LOG_DIR=/vla/users/niejunnan/codebase/serl_torch/examples/libero/outputs/offline_prepare/policy
mkdir -p "${LOG_DIR}"

bash examples/libero/tools/serve_openpi_10000_policy.sh \
  --port 55101 \
  --gpu-id 6 \
  2>&1 | tee "${LOG_DIR}/policy_gpu6_port55101.log"
```

另一个终端依次生成 `alpha=0.1/0.2/0.5` 的 prepared replay：

```bash
source /vla/miniconda3/etc/profile.d/conda.sh
conda activate serl_torch
cd /vla/users/niejunnan/codebase/serl_torch

OFFLINE_ROOT=/vla/users/niejunnan/codebase/serl_torch/data/residual/offline_data_limits_xyz1_rot0p4_grip0p3
LOG_BASE=/vla/users/niejunnan/codebase/serl_torch/examples/libero/outputs/offline_prepare/long3_limits

for item in \
  "alpha0p1 long_3/long3_chunk_alpha0p1_unfiltered_offline_noent_std1p0_limits_xyz1_rot0p4_grip0p3_ports56100" \
  "alpha0p2 long_3/long3_chunk_alpha0p2_unfiltered_offline_noent_std1p0_limits_xyz1_rot0p4_grip0p3_ports56200" \
  "alpha0p5 long_3/long3_chunk_alpha0p5_unfiltered_offline_noent_std1p0_limits_xyz1_rot0p4_grip0p3_ports56500"; do
  set -- ${item}
  ALPHA_TAG="$1"
  CONFIG_NAME="$2"
  LOG_ROOT="${LOG_BASE}/${ALPHA_TAG}"
  mkdir -p "${LOG_ROOT}"

  python examples/libero/scripts/run_residual_offline_prepare.py \
    --config-name "${CONFIG_NAME}" \
    policy.host=127.0.0.1 \
    policy.port=55101 \
    offline.prepared_path=null \
    offline.prepare.output_root="${OFFLINE_ROOT}" \
    libero_root=/vla/users/niejunnan/codebase/serl_torch/third_party/LIBERO \
    libero_datasets_root=/vla/users/niejunnan/datasets \
    hydra.run.dir="${LOG_ROOT}" \
    2>&1 | tee "${LOG_ROOT}/prepare.log"
done
```

生成后应该出现：

```text
/vla/users/niejunnan/codebase/serl_torch/data/residual/offline_data_limits_xyz1_rot0p4_grip0p3/libero_10_task_3/openpi_chunk5_alpha0p1
/vla/users/niejunnan/codebase/serl_torch/data/residual/offline_data_limits_xyz1_rot0p4_grip0p3/libero_10_task_3/openpi_chunk5_alpha0p2
/vla/users/niejunnan/codebase/serl_torch/data/residual/offline_data_limits_xyz1_rot0p4_grip0p3/libero_10_task_3/openpi_chunk5_alpha0p5
```

快速检查 manifest：

```bash
python - <<'PY'
from pathlib import Path
import json

root = Path("/vla/users/niejunnan/codebase/serl_torch/data/residual/offline_data_limits_xyz1_rot0p4_grip0p3/libero_10_task_3")
expected_limits = [1.0, 1.0, 1.0, 0.4, 0.4, 0.4, 0.3]

def find_key(obj, key):
    if isinstance(obj, dict):
        if key in obj:
            return obj[key]
        for value in obj.values():
            found = find_key(value, key)
            if found is not None:
                return found
    if isinstance(obj, list):
        for value in obj:
            found = find_key(value, key)
            if found is not None:
                return found
    return None

for alpha in ("alpha0p1", "alpha0p2", "alpha0p5"):
    data_dir = root / f"openpi_chunk5_{alpha}"
    manifest = data_dir / "manifest.json"
    data = json.loads(manifest.read_text())
    limits = find_key(data, "action_limits")
    episodes = len(list(data_dir.glob("episode_*.pkl")))
    print(alpha, "episodes=", episodes, "action_limits=", limits)
    assert limits == expected_limits
    assert episodes > 0
PY
```

## 查看日志

```bash
RUN_ROOT=/vla/users/niejunnan/codebase/serl_torch/examples/libero/outputs/spatial_4_0514/spatial4_chunk_alpha0p1_unfiltered_offline_noent_std1p0

# key log files:
# ${RUN_ROOT}/services/train_env.log
# ${RUN_ROOT}/services/eval_env.log
# ${RUN_ROOT}/services/policy.log
# ${RUN_ROOT}/learner/launcher.log
# ${RUN_ROOT}/learner/train_residual_chunk.log
# ${RUN_ROOT}/learner/learner_timers.jsonl
# ${RUN_ROOT}/learner/async_eval_results.jsonl
# ${RUN_ROOT}/actor/launcher.log
# ${RUN_ROOT}/actor/train_residual_chunk.log
# ${RUN_ROOT}/actor/actor_timers.jsonl
# ${RUN_ROOT}/actor/episode_logs.jsonl
```

常用监控：

```bash
# learner
tail -f "${RUN_ROOT}/learner/train_residual_chunk.log"

# actor
tail -f "${RUN_ROOT}/actor/train_residual_chunk.log"

# 成功率
tail -f "${RUN_ROOT}/actor/episode_logs.jsonl"

# 异步评估
tail -f "${RUN_ROOT}/learner/async_eval_results.jsonl"
```

## 停止训练

```bash
cd /vla/users/niejunnan/codebase/serl_torch

bash examples/libero/tools/stop_launched_training.sh \
  --output-root /vla/users/niejunnan/codebase/serl_torch/examples/libero/outputs/spatial_4_0514/spatial4_chunk_alpha0p1_unfiltered_offline_noent_std1p0
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
