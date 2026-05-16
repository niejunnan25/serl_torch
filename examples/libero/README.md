## LIBERO — Residual RL with OpenPI Base Policy

在 [LIBERO](https://github.com/Lifelong-Robot-Learning/LIBERO) 仿真基准上，以 OpenPI (pi0) 作为 frozen base policy，用 DRQ 学习残差强化学习：`final_action = base_action + alpha × residual_action`。

### 前置条件

- NVIDIA GPU（至少 2 张，推荐 4 张）
- Python ≥ 3.10，CUDA ≥ 12.0
- LIBERO 数据集
- OpenPI 模型权重（pi0_libero 或 pi0_10000）
- 仓库已安装：`pip install -e ./serl_launcher && pip install -e .`

```bash
CODE_ROOT=/path/to/serl_torch
cd $CODE_ROOT
source /path/to/conda/etc/profile.d/conda.sh
conda activate serl_torch
```

运行前必须登录实验管理平台（二选一）：

```bash
swanlab login --relogin     # 默认，已全面转向 SwanLab
wandb login                 # 可选，仍支持
```

### 配置文件

配置文件位于 `examples/libero/configs/`，使用 Hydra YAML，文件名编码实验参数：

```
{task}_{scripts}_{alpha}_{data}_{entropy}_{std}_ports{prefix}.yaml
```

| 字段 | 含义 |
|------|------|
| `spatial4` / `long3` | 任务 |
| `scripts_2` / `scripts_5` | 训练脚本版本 |
| `alpha0p1` / `alpha0p2` / `alpha0p5` | 残差缩放系数 |
| `unfiltered_offline` | 离线数据过滤策略 |
| `noent` | backup_entropy=false |
| `std0p5` / `std1p0` / `std5p0` | SAC std_max |
| `ports53100` | 端口前缀（多实验共存） |

CLI 覆盖：

```bash
policy.type=joyra
policy.port=9001
task.suite_name=libero_10
residual.alpha=0.3
```

### 关键超参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `residual.alpha` | 0.1 | 残差缩放系数 |
| `residual.chunk_horizon` | 5 | 每次决策预测的动作块长度 |
| `sac.learning_rate` | 3e-4 | 学习率 |
| `sac.std_max` | 5.0 | 策略标准差上限，常用 1.0 或 0.5 |
| `sac.backup_entropy` | true | TD target 中是否包含熵项 |
| `sac.utd_ratio` | 1 | 每步更新次数，常用 4 |
| `training.training_starts` | 1000 | 收集多少步后开始训练 |
| `training.max_env_steps` | 300000 | 最大环境交互步数 |
| `offline.enabled` | false | 是否混入离线专家数据 |
| `offline.ratio` | 0.5 | 每个 batch 中离线数据占比 |
| `offline.prepared_path` | null | 预处理离线数据目录 |

### 准备离线数据

离线数据本质上是**残差 MDP 的 replay buffer**：对 LIBERO 专家演示的每一帧，用 base policy 推理出 `base_action_chunk`，将专家动作投影为残差动作 `residual = (expert - base) / alpha`，再构建残差 observation（robot state + 图像 + base_action_chunk + alpha），写入 `(obs, residual_action, next_obs, reward, mask)` 的 transition 序列。

**哪些参数变化后需要重新生成？** 以下任一参数变了，离线数据就不可复用（训练启动时 `manifest.json` 会校验，不匹配直接报错）：

| 类别 | 参数 | 为什么影响数据 |
|------|------|---------------|
| 任务 | `task.suite_name` + `task.task_id` | 不同的专家演示数据 |
| Base policy | `policy.type` + 具体 checkpoint | 不同 checkpoint 输出的 base_action_chunk 不同，残差 observation 里的 base 部分就不同 |
| 残差 | `residual.alpha` | 残差 observation 中包含 alpha |
| 残差 | `residual.chunk_horizon` | base_action_chunk 的长度 |
| 残差 | `residual.action_mask` | 哪些动作维度被残差控制 |
| 残差 | `residual.action_limits` | 投影时每维的幅度上限 |
| 残差 | `residual.clip_gripper` | 夹爪维度是否裁剪 |
| 观测 | `obs.image_keys` | 哪些摄像头图像被编码 |
| 观测 | `obs.vector_obs_keys` | 哪些本体感受状态被编码 |
| 离线准备 | `offline.prepare.expert_reference_scale` | 专家动作投影公式中的缩放系数 |
| 离线准备 | `offline.prepare.clip_residual_to_unit` | 投影后是否裁剪到 [-1, 1] |
| 离线准备 | `offline.prepare.filter_unrepresentable_steps` | 是否过滤掉 base policy 无法表示的步骤 |

目录名只编码了其中 5 个维度（task、backend、chunk、alpha），其余通过 `manifest.json` 的兼容性指纹校验。

**终端 1 — 启动 Policy Server：**

```bash
bash examples/libero/tools/serve_openpi_10000_policy.sh \
  --port 55101 \
  --gpu-id 6
```

**终端 2 — 运行 prepare：**

```bash
python examples/libero/scripts/run_residual_offline_prepare.py \
  --config-name spatial_4_0514_runtime/spatial4_scripts_2_alpha0p1_unfiltered_offline_noent_std1p0_ports53100 \
  policy.host=127.0.0.1 \
  policy.port=55101 \
  offline.prepare.output_root=data/residual/offline_data
```

以 spatial task 4、alpha=0.1、pi0_10000、chunk_horizon=5 为例，生成后目录树：

```
data/residual/offline_data/
└── libero_spatial_task_4/
    └── openpi_pi0_10000_chunk5_alpha0p1/
        ├── manifest.json           # 上述所有 12 项参数的兼容性指纹
        ├── episode_000000.pkl      # 每个 episode 的 transition 列表
        ├── episode_000001.pkl
        └── ...
```

每个 `.pkl` 是 `list[dict]`，每条 transition 包含 `observations`、`actions`（残差动作）、`next_observations`、`rewards`、`masks`、`dones`。

prepare 完成后，将生成的目录路径（`manifest.json` 所在的父目录）填入训练配置：

```yaml
offline:
  enabled: true
  ratio: 0.5
  prepared_path: data/residual/offline_data/libero_spatial_task_4/openpi_pi0_10000_chunk5_alpha0p1
```

### 启动训练

提供两种方式：一键脚本（推荐日常使用）和手动多终端（适合调试）。

#### 方式一：一键启动

```bash
bash examples/libero/tools/launch_residual_training.sh \
  --script-id 2 \
  --config-file examples/libero/configs/spatial_4_0514_runtime/spatial4_scripts_2_alpha0p1_unfiltered_offline_noent_std1p0_ports53100.yaml \
  --output-root examples/libero/outputs/my_first_experiment \
  --learner-gpu 1 \
  --actor-gpu 0 \
  --env-gpu 0 \
  --policy-gpu 0 \
  --libero-root third_party/LIBERO \
  --libero-datasets-root /path/to/libero/datasets \
  --openpi-root /path/to/openpi \
  --policy-dir /path/to/openpi/checkpoint \
  --with-eval-env \
  --clean-output-dir
```

| 参数 | 说明 |
|------|------|
| `--script-id` | 训练脚本版本（`2` = 最常用） |
| `--config-file` | Hydra YAML 配置文件路径 |
| `--output-root` | 输出目录（日志、checkpoint、结果） |
| `--learner-gpu` / `--actor-gpu` / `--env-gpu` / `--policy-gpu` | 各进程 GPU 分配 |
| `--libero-root` / `--libero-datasets-root` | LIBERO 安装和数据路径 |
| `--openpi-root` / `--policy-dir` | OpenPI 仓库和模型权重路径 |
| `--with-eval-env` | 启用异步评估 |
| `--clean-output-dir` | 重跑时清空旧输出目录 |

#### 方式二：手动多终端

适合调试、单步验证，每个进程在独立终端中运行。

默认 Hydra 输出路径由 `train_residual.yaml` 中的 `hydra.run.dir` 决定：

```
${launch.output_root}/${hydra:job.config_name}/${now:%Y-%m-%d_%H-%M-%S}
```

以下示例使用端口前缀 31000，手动覆盖 `hydra.run.dir` 使输出结构与一键启动一致。

**终端 1 — Train Env：**

```bash
python examples/libero/scripts/serve_env.py \
  task.suite_name=libero_spatial \
  task.task_id=4 \
  env.remote.port=31000
```

**终端 2 — Eval Env（异步评估，可选）：**

```bash
python examples/libero/scripts/serve_env.py \
  task.suite_name=libero_spatial \
  task.task_id=4 \
  env.remote.port=31003
```

**终端 3 — Policy Server：**

```bash
bash examples/libero/tools/serve_openpi_policy.sh \
  --port 31001 \
  --gpu-id 0 \
  --policy-dir /path/to/openpi/checkpoint
```

**终端 4 — Learner：**

```bash
python examples/libero/scripts/run_residual_training_2_chunk_local.py \
  --config-name spatial_4_0514_runtime/spatial4_scripts_2_alpha0p1_unfiltered_offline_noent_std1p0_ports53100 \
  runtime.role=learner \
  runtime.trainer_port=31004 \
  runtime.broadcast_port=31005 \
  runtime.data_port=31006 \
  task.suite_name=libero_spatial \
  task.task_id=4 \
  policy.port=31001 \
  env.backend=remote \
  env.remote.port=31000 \
  training.async_eval.enabled=true \
  training.async_eval.env.backend=remote \
  training.async_eval.env.remote.port=31003 \
  launch.output_root=examples/libero/outputs/spatial_4_0514/spatial4_scripts_2_alpha0p1_unfiltered_offline_noent_std1p0 \
  hydra.run.dir=examples/libero/outputs/spatial_4_0514/spatial4_scripts_2_alpha0p1_unfiltered_offline_noent_std1p0/learner
```

**终端 5 — Actor：**

```bash
python examples/libero/scripts/run_residual_training_2_chunk_local.py \
  --config-name spatial_4_0514_runtime/spatial4_scripts_2_alpha0p1_unfiltered_offline_noent_std1p0_ports53100 \
  runtime.role=actor \
  runtime.trainer_port=31004 \
  runtime.broadcast_port=31005 \
  runtime.data_port=31006 \
  task.suite_name=libero_spatial \
  task.task_id=4 \
  policy.port=31001 \
  env.backend=remote \
  env.remote.port=31000 \
  launch.output_root=examples/libero/outputs/spatial_4_0514/spatial4_scripts_2_alpha0p1_unfiltered_offline_noent_std1p0 \
  hydra.run.dir=examples/libero/outputs/spatial_4_0514/spatial4_scripts_2_alpha0p1_unfiltered_offline_noent_std1p0/actor
```

### 单独评估 Checkpoint

**终端 1 — Env Server：**

```bash
python examples/libero/scripts/serve_env.py \
  task.suite_name=libero_spatial \
  task.task_id=4 \
  env.remote.port=41000
```

**终端 2 — Policy Server：**

```bash
bash examples/libero/tools/serve_openpi_policy.sh \
  --port 41001 \
  --gpu-id 0 \
  --policy-dir /path/to/openpi/checkpoint
```

**终端 3 — 评估 residual checkpoint：**

```bash
python examples/libero/scripts/run_residual_eval.py \
  --config-name eval_residual \
  eval.checkpoint_path=/path/to/checkpoint \
  eval.episodes=50 \
  eval.deterministic=true \
  task.suite_name=libero_spatial \
  task.task_id=4 \
  policy.host=127.0.0.1 \
  policy.port=41001 \
  env.remote.host=127.0.0.1 \
  env.remote.port=41000 \
  hydra.run.dir=examples/libero/outputs/eval/spatial_4/checkpoint_$(basename /path/to/checkpoint)
```

**只评估 base policy（不加载 residual）：**

```bash
python examples/libero/scripts/run_residual_eval.py \
  --config-name eval_residual \
  eval.checkpoint_path=null \
  eval.episodes=50 \
  eval.deterministic=true \
  task.suite_name=libero_spatial \
  task.task_id=4 \
  policy.host=127.0.0.1 \
  policy.port=41001 \
  env.remote.host=127.0.0.1 \
  env.remote.port=41000 \
  hydra.run.dir=examples/libero/outputs/eval/spatial_4/base_policy_only
```

### 输出目录结构

```
${RUN_ROOT}/
├── services/
│   ├── train_env.log
│   ├── eval_env.log
│   └── policy.log
├── learner/
│   ├── run_residual_training_*.log
│   ├── learner_timers.jsonl
│   ├── async_eval_results.jsonl
│   └── checkpoints/
├── actor/
│   ├── run_residual_training_*.log
│   ├── actor_timers.jsonl
│   └── episode_logs.jsonl
└── .launcher/
    ├── commands/
    └── pids/
```

### 日志与监控

训练指标通过 SwanLab 上传（也兼容 WandB）。

| 日志文件 | 内容 |
|----------|------|
| `actor/episode_logs.jsonl` | 每 episode 成功率、步数、奖励 |
| `learner/async_eval_results.jsonl` | 异步评估确定性成功率 |
| `learner/learner_timers.jsonl` | Learner 各阶段耗时 |
| `actor/actor_timers.jsonl` | Actor 各阶段耗时 |

### 常见问题

**GPU 显存不足** — 降低 `replay.batch_size`（如 64），或减少 encoder bottleneck_dim（如 128）。

**端口冲突** — 每个实验使用唯一的端口前缀，如 `ports53100` 和 `ports53300` 可同时运行。

**训练不收敛** — 排查顺序：
1. 确认 base policy 单独评估成功率（`eval.checkpoint_path=null`）
2. 查看 `entropy_per_dim` 是否持续下降
3. 调高 `residual.alpha`（0.2 → 0.3 → 0.5）
4. 确保 `offline.enabled=true` 且 `offline.ratio=0.5`
5. 降低 `sac.std_max`（1.0 或 0.5）

**异步评估没有输出** — 检查 `training.async_eval.enabled=true`，确认 `services/eval_env.log` 正常。

### 相关文档

- [个人训练备忘](commands.md)
- [Chunk Residual MDP 讨论](docs/chunk_residual_mdp_discussion.md)
- [SERL Launcher 库文档](../../serl_launcher/README.md)
