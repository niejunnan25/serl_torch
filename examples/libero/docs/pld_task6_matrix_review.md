# PLD Task6 Matrix Review (2026-03-19)

## 1) YAML/训练逻辑检查结论

结论：当前 8 组矩阵配置逻辑正确，可用于 Stage-1（Residual RL Specialist）对比实验。

关键对应关系（代码 -> 配置）：

- 全局随机种子：`seed`  
  - 代码：`set_global_seeds(int(cfg.seed))`  
  - 文件：`scripts/train_residual_sac.py`
- 在线 episode seed 起点：`task.seed_base`  
  - 代码：`seed_cursor = int(cfg.task.seed_base)`，每个 episode 递增
- 离线 bootstrap seed 起点：`offline.bootstrap_base.seed_base`  
  - 代码：`_bootstrap_offline_with_base_success(...)`
- 离线预热（成功轨迹）：
  - 专家数据模式：`offline.dataset_paths` 非空，`bootstrap_base.enabled=false`
  - bootstrap 模式：`offline.dataset_paths=[]`，`bootstrap_base.enabled=true, success_episodes=50`
- 在线 warmup：
  - `training.warmup_base_episodes`
  - 代码里表现为前 N 个 episode residual action 强制为 0（走 base）
- probing（你现在不需要）：
  - `training.enable_base_probing=false`
  - 代码：`sample_probing_steps(...)` 返回 0，不会进入 probing rollout

说明：你如果直接 `python scripts/train_residual_sac.py`，当前 shell 可能报 `tensorflow` 缺失；用 `tools/run_train.sh` 会先激活 `conda serl_torch`，这是正确入口。

## 2) 参数意义（你现在最关心的）

- `offline.dataset_paths`：专家离线数据路径（用于离线 buffer）
- `offline.bootstrap_base.enabled/success_episodes`：是否在线收集 base 成功轨迹做离线预热
- `offline.symmetric_replay=true`：在线/离线 batch 固定 1:1 混合
- `training.warmup_base_episodes`：在线初期前 N 个 episode 禁用 residual（更稳）
- `training.max_online_env_steps`：在线训练步数预算
- `residual.xi`：残差动作幅度（探索强度）
- `training.enable_base_probing`：是否启用 probing 初始化（Stage-2更常用）
- `training.async_eval.enabled`：是否开启异步评估 watcher（矩阵里已关闭）

## 3) 绝对路径启动命令（1任务=1GPU）

工作目录建议先切到：

```bash
cd /vla/users/niejunnan/codebase/serl_torch/examples/libero
```

逐条启动（绝对路径）：

```bash
bash /vla/users/niejunnan/codebase/serl_torch/examples/libero/tools/run_train.sh /vla/users/niejunnan/codebase/serl_torch/examples/libero/conf/train_pld_task6_m01_expert_w0_xi05.yaml --gpu_id 0
bash /vla/users/niejunnan/codebase/serl_torch/examples/libero/tools/run_train.sh /vla/users/niejunnan/codebase/serl_torch/examples/libero/conf/train_pld_task6_m02_expert_w50_xi05.yaml --gpu_id 1
bash /vla/users/niejunnan/codebase/serl_torch/examples/libero/tools/run_train.sh /vla/users/niejunnan/codebase/serl_torch/examples/libero/conf/train_pld_task6_m03_expert_w100_xi05.yaml --gpu_id 2
bash /vla/users/niejunnan/codebase/serl_torch/examples/libero/tools/run_train.sh /vla/users/niejunnan/codebase/serl_torch/examples/libero/conf/train_pld_task6_m04_boot50_w0_xi05.yaml --gpu_id 3
bash /vla/users/niejunnan/codebase/serl_torch/examples/libero/tools/run_train.sh /vla/users/niejunnan/codebase/serl_torch/examples/libero/conf/train_pld_task6_m05_boot50_w50_xi05.yaml --gpu_id 4
bash /vla/users/niejunnan/codebase/serl_torch/examples/libero/tools/run_train.sh /vla/users/niejunnan/codebase/serl_torch/examples/libero/conf/train_pld_task6_m06_boot50_w100_xi05.yaml --gpu_id 5
bash /vla/users/niejunnan/codebase/serl_torch/examples/libero/tools/run_train.sh /vla/users/niejunnan/codebase/serl_torch/examples/libero/conf/train_pld_task6_m07_boot50_w100_xi03.yaml --gpu_id 6
bash /vla/users/niejunnan/codebase/serl_torch/examples/libero/tools/run_train.sh /vla/users/niejunnan/codebase/serl_torch/examples/libero/conf/train_pld_task6_m08_boot50_w100_xi02.yaml --gpu_id 7
```

一键并行脚本：

```bash
bash /vla/users/niejunnan/codebase/serl_torch/examples/libero/tools/run_pld_task6_min_matrix.sh 0,1,2,3,4,5,6,7
```

## 4) 训练/评估环境自动启动逻辑

`tools/run_train.sh` 会自动：

1. 启动训练环境 server（`env.remote.host:env.remote.port`）
2. 启动 OpenPI server（`openpi.host:openpi.port`）
3. 启动训练脚本

仅当 `training.async_eval.enabled=true` 时，才会尝试再起一个“评估专用 env server”。

你当前 8 组矩阵已设为 `async_eval.enabled=true`，会启动评估专用环境。

## 5) 评估环境逻辑 review 结果

- 逻辑本身是合理的：训练 env 与评估 env 分端口隔离时，会分别服务训练和 eval。
- 我额外修了一个真实风险点：  
  `run_train.sh` 之前读取端口时没有应用命令行 overrides，可能造成“服务启动端口”和“训练实际端口”不一致。  
  现在已修复为读取端口时会带上 `EXTRA_ARGS` overrides。

## 6) 端口冲突检查

矩阵端口规划：

- 训练 env：`31341-31348`
- OpenPI：`32341-32348`
- 异步评估 env：`31441-31448`
- 都是唯一且不重复

检查时刻（2026-03-19）上述端口均处于 `FREE` 状态。

补充：基础配置 `train_pld_task6.yaml` 已改到独立端口，避免与矩阵冲突：

- 训练 env：`31340`
- OpenPI：`32340`
- async eval env：`31440`
