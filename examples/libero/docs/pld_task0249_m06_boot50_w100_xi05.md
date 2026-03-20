# PLD Task 0/2/4/9 配置与日志说明（m06 模板）

## 1. 新增的 YAML 文件

- `/vla/users/niejunnan/codebase/serl_torch/examples/libero/conf/train_pld_task0_m06_boot50_w100_xi05.yaml`
- `/vla/users/niejunnan/codebase/serl_torch/examples/libero/conf/train_pld_task2_m06_boot50_w100_xi05.yaml`
- `/vla/users/niejunnan/codebase/serl_torch/examples/libero/conf/train_pld_task4_m06_boot50_w100_xi05.yaml`
- `/vla/users/niejunnan/codebase/serl_torch/examples/libero/conf/train_pld_task9_m06_boot50_w100_xi05.yaml`

这 4 个文件都参考：

- `/vla/users/niejunnan/codebase/serl_torch/examples/libero/conf/train_pld_task6_m06_boot50_w100_xi05.yaml`

保持一致的核心训练逻辑：

- `offline.bootstrap_base.enabled=true`
- `offline.bootstrap_base.success_episodes=50`
- `training.warmup_base_episodes=100`
- `residual.xi=0.5`
- `training.async_eval.enabled=true`

## 2. 相比模板改了哪些配置项

仅改了以下几类字段：

- `task.task_id`
- `env.remote.port`（训练环境端口）
- `openpi.port`（OpenPI 端口）
- `training.async_eval.env_port`（异步评估环境端口）
- `hydra.run.dir`（输出目录分到对应 task 子目录）

端口分配如下：

| config | task_id | train env port | openpi port | async eval env port |
|---|---:|---:|---:|---:|
| `train_pld_task0_m06_boot50_w100_xi05` | 0 | 31600 | 32600 | 31700 |
| `train_pld_task2_m06_boot50_w100_xi05` | 2 | 31602 | 32602 | 31702 |
| `train_pld_task4_m06_boot50_w100_xi05` | 4 | 31604 | 32604 | 31704 |
| `train_pld_task9_m06_boot50_w100_xi05` | 9 | 31609 | 32609 | 31709 |

## 3. 日志和 ckpt 记录位置

运行时会有两类输出目录。

### 3.1 run_train 启动器日志（外层）

由 `tools/run_train.sh` 产生，目录格式：

- `/vla/users/niejunnan/codebase/serl_torch/examples/libero/outputs/libero/run_train_support/<timestamp>_<config>_gpu<id>_<pid>/`

其中常见文件：

- `launcher.log`
- `env_server.log`
- `async_eval_env_server.log`
- `openpi_server.log`

示例（task0）：

- `/vla/users/niejunnan/codebase/serl_torch/examples/libero/outputs/libero/run_train_support/2026-03-19_20-00-00_train_pld_task0_m06_boot50_w100_xi05_gpu0_123456/launcher.log`

### 3.2 Hydra 训练输出目录（内层）

由各 YAML 的 `hydra.run.dir` 决定。以 task0 为例：

- `/vla/users/niejunnan/codebase/serl_torch/examples/libero/outputs/libero/pld_matrix/task0/m06_boot50_w100_xi05/<YYYY-MM-DD>/<HH-MM-SS>/`

该目录下常见内容：

- `logs/step_logs.jsonl`
- `logs/episode_logs.jsonl`
- `logs/summary.json`
- `artifacts/checkpoints/checkpoint_*.pt`
- `tb/`（tensorboard）
- `async_eval_watch.log`
- `async_eval_results.jsonl`
- `async_eval/step_XXXXXXX/`（每个 checkpoint 的异步评估子目录，包含 `eval_runner.log` 和评估日志/summary）

示例（task2）：

- `/vla/users/niejunnan/codebase/serl_torch/examples/libero/outputs/libero/pld_matrix/task2/m06_boot50_w100_xi05/2026-03-19/20-05-00/logs/episode_logs.jsonl`
- `/vla/users/niejunnan/codebase/serl_torch/examples/libero/outputs/libero/pld_matrix/task2/m06_boot50_w100_xi05/2026-03-19/20-05-00/artifacts/checkpoints/checkpoint_5000.pt`
- `/vla/users/niejunnan/codebase/serl_torch/examples/libero/outputs/libero/pld_matrix/task2/m06_boot50_w100_xi05/2026-03-19/20-05-00/async_eval/step_0005000/eval_runner.log`

## 4. 启动命令（绝对路径）

```bash
bash /vla/users/niejunnan/codebase/serl_torch/examples/libero/tools/run_train.sh /vla/users/niejunnan/codebase/serl_torch/examples/libero/conf/train_pld_task0_m06_boot50_w100_xi05.yaml --gpu_id 4
bash /vla/users/niejunnan/codebase/serl_torch/examples/libero/tools/run_train.sh /vla/users/niejunnan/codebase/serl_torch/examples/libero/conf/train_pld_task2_m06_boot50_w100_xi05.yaml --gpu_id 5
bash /vla/users/niejunnan/codebase/serl_torch/examples/libero/tools/run_train.sh /vla/users/niejunnan/codebase/serl_torch/examples/libero/conf/train_pld_task4_m06_boot50_w100_xi05.yaml --gpu_id 6
bash /vla/users/niejunnan/codebase/serl_torch/examples/libero/tools/run_train.sh /vla/users/niejunnan/codebase/serl_torch/examples/libero/conf/train_pld_task9_m06_boot50_w100_xi05.yaml --gpu_id 7
```
