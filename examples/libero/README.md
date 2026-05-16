# LIBERO 使用说明

这份说明只覆盖当前常用路径：用 OpenPI 作为 base policy，在 LIBERO 上跑 residual RL。

默认路径如下。如果你的机器路径不同，只替换这些变量即可：

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

如果是新环境，先安装本仓库：

```bash
pip install -e ./serl_launcher
pip install -e .
```

## 训练入口

日常 residual 实验使用：

```text
examples/libero/scripts/run_residual_training_2_chunk_local.py
```

不要手动分别启动 env、policy、learner、actor，优先用统一启动脚本：

```text
examples/libero/tools/launch_residual_training.sh
```

这个脚本会拉起：

```text
train env
eval env
OpenPI policy server
learner
actor
```

并把命令、日志、pid 写到指定输出目录。

## 启动一组训练

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

这一步是在启动完整训练流程。

如果换机器、换数据盘、换 OpenPI 权重，只替换：

```text
--libero-root
--libero-datasets-root
--openpi-root
--policy-config
--policy-dir
```

如果换实验，只替换：

```text
--config-file
--output-root
--learner-gpu
--actor-gpu
--env-gpu
--eval-env-gpu
--policy-gpu
```

如果输出目录已经存在并且你要重跑，保留：

```text
--clean-output-dir
```

如果不想清空旧目录，删掉这个参数，并换一个新的 `--output-root`。

## 常用配置

配置文件放在：

```text
examples/libero/configs/
```

当前常用的是带端口的 runtime 配置：

```text
examples/libero/configs/spatial_4_0514_runtime/
examples/libero/configs/long_3/
```

### Spatial Task 4

任务是：

```text
libero_spatial task_id=4
pick up the black bowl in the top drawer of the wooden cabinet and place it on the plate
```

常用配置示例：

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

这组是在固定 `std_max=1.0`、`backup_entropy=false`、`unfiltered offline` 的情况下，对 residual 动作加每维限幅：

```text
xyz: 1.0
rotation: 0.4
gripper: 0.3
```

配置：

```text
examples/libero/configs/spatial_4_0514_runtime/spatial4_scripts_2_alpha0p1_unfiltered_offline_noent_std1p0_limits_xyz1_rot0p4_grip0p3_ports54100.yaml
examples/libero/configs/spatial_4_0514_runtime/spatial4_scripts_2_alpha0p2_unfiltered_offline_noent_std1p0_limits_xyz1_rot0p4_grip0p3_ports54200.yaml
examples/libero/configs/spatial_4_0514_runtime/spatial4_scripts_2_alpha0p3_unfiltered_offline_noent_std1p0_limits_xyz1_rot0p4_grip0p3_ports54300.yaml
```

### Long Task 3

配置：

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

这一步是把 LIBERO raw demo 转成 residual replay。脚本会读取专家动作，调用 OpenPI base policy，计算 residual action，然后写成训练可直接加载的 prepared replay。

先启动 OpenPI policy server：

```bash
cd /vla/users/niejunnan/codebase/serl_torch

LOG_DIR=/vla/users/niejunnan/codebase/serl_torch/examples/libero/outputs/offline_prepare/policy
mkdir -p "${LOG_DIR}"

bash examples/libero/tools/serve_openpi_10000_policy.sh \
  --port 55101 \
  --gpu-id 6 \
  2>&1 | tee "${LOG_DIR}/policy_gpu6_port55101.log"
```

再开另一个终端跑 prepare：

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

如果要生成别的 alpha，只换：

```text
--config-name
LOG_ROOT
policy.port
```

如果要换 prepared replay 输出位置，在命令里加：

```text
offline.prepare.output_root=/abs/path/to/offline_data_root
```

生成完成后，训练配置里的 `offline.prepared_path` 要指向具体数据目录，例如：

```text
/vla/users/niejunnan/codebase/serl_torch/data/residual/offline_data_limits_xyz1_rot0p4_grip0p3/libero_spatial_task_4/openpi_chunk5_alpha0p3
```

## 日志位置

假设输出目录是：

```text
RUN_ROOT=/vla/users/niejunnan/codebase/serl_torch/examples/libero/outputs/spatial_4_0514/spatial4_scripts_2_alpha0p1_unfiltered_offline_noent_std1p0
```

常看这些文件：

```text
${RUN_ROOT}/services/train_env.log
${RUN_ROOT}/services/eval_env.log
${RUN_ROOT}/services/policy.log
${RUN_ROOT}/learner/launcher.log
${RUN_ROOT}/learner/run_residual_training_2_chunk_local.log
${RUN_ROOT}/learner/learner_timers.jsonl
${RUN_ROOT}/learner/async_eval_results.jsonl
${RUN_ROOT}/actor/launcher.log
${RUN_ROOT}/actor/run_residual_training_2_chunk_local.log
${RUN_ROOT}/actor/actor_timers.jsonl
${RUN_ROOT}/actor/episode_logs.jsonl
```

看 learner：

```bash
tail -f "${RUN_ROOT}/learner/run_residual_training_2_chunk_local.log"
```

看 actor：

```bash
tail -f "${RUN_ROOT}/actor/run_residual_training_2_chunk_local.log"
```

看成功率：

```bash
tail -f "${RUN_ROOT}/actor/episode_logs.jsonl"
```

看异步评估：

```bash
tail -f "${RUN_ROOT}/learner/async_eval_results.jsonl"
```

## 停止训练

在启动训练的同一台机器上执行：

```bash
cd /vla/users/niejunnan/codebase/serl_torch

bash examples/libero/tools/stop_launched_training.sh \
  --output-root /vla/users/niejunnan/codebase/serl_torch/examples/libero/outputs/spatial_4_0514/spatial4_scripts_2_alpha0p1_unfiltered_offline_noent_std1p0
```

这一步会读取：

```text
${RUN_ROOT}/.launcher/pids/*.pid
```

然后停止 actor、learner、policy、env 等进程。

## 单独评估 checkpoint

这一步是加载一个 residual checkpoint，跑独立评估。

先启动 env server 和 OpenPI policy server，再运行：

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

如果只评估 base policy，不加载 residual checkpoint：

```text
eval.checkpoint_path=null
```

## 常见替换

换任务：

```text
task.suite_name=libero_spatial task.task_id=4
```

换 env 端口：

```text
env.remote.host=127.0.0.1 env.remote.port=30000
```

换 OpenPI 端口：

```text
policy.host=127.0.0.1 policy.port=30001
```

换输出目录：

```text
launch.output_root=/abs/path/to/output
hydra.run.dir=/abs/path/to/output/learner
```

换 ResNet 本地路径：

```text
encoder.resnet.model_name=/abs/path/to/microsoft--resnet-18
```

SwanLab 在线上传需要先登录：

```bash
conda activate serl_torch
swanlab login --relogin
```

训练指标由 SwanLab 上传；WandB 目录主要作为本地兼容日志保留。
