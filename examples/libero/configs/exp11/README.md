# exp11 使用说明

这个目录是 `LIBERO libero_10 task_id=8` 的一组异步 residual SAC 实验配置。当前目录下最常用的文件有：

- `chunk/`: 8 个具体配置文件
- `COMMANDS.sh`: 按 run 名称封装好的启动脚本
- `FULL_COMMANDS.sh`: 依次启动全部配置的批量脚本

`exp11` 的每次异步训练都会涉及 5 个角色：

1. `env`
2. `async_eval_env`
3. `openpi`
4. `learner`
5. `actor`

## 支持的 run

- `null`
- `null_utdeff1p`
- `null_unfreeze`
- `null_utdeff1p_unfreeze`
- `null_pi05`
- `null_utdeff1p_pi05`
- `null_unfreeze_pi05`
- `null_utdeff1p_unfreeze_pi05`

其中前 4 个 run 走默认的 `pi0` checkpoint，后 4 个 `*_pi05` run 会额外指定：

```bash
POLICY_CONFIG=pi05_libero
POLICY_DIR=/vla/users/niejunnan/openpi-assets/checkpoints/pi05_libero
```

## 数据目录约定

`exp11` 默认使用统一训练格式 `libero_residual_training`，并且把 `base_chunks`
和 offline projected final actions 都写进 PKL 里了，所以不同 base policy 以及
不同 `alpha` 需要分开存。

当前这组配置约定：

- `pi0` run 的 offline 数据目录：
  `/vla/users/niejunnan/codebase/serl_torch/examples/libero/data/residual_training/offline_pi0_alpha01/libero_10_task_8`
- `pi0` run 的 online warmup 目录：
  `/vla/users/niejunnan/codebase/serl_torch/examples/libero/data/residual_training/online_pi0/libero_10_task_8/stepchunk/manifest.json`
- `pi05` run 的 offline 数据目录：
  `/vla/users/niejunnan/codebase/serl_torch/examples/libero/data/residual_training/offline_pi05_alpha01/libero_10_task_8`
- `pi05` run 的 online warmup 目录：
  `/vla/users/niejunnan/codebase/serl_torch/examples/libero/data/residual_training/online_pi05/libero_10_task_8/stepchunk/manifest.json`

如果你改了下面任意一项，就应该重新生成对应数据：

- `residual.alpha`
- OpenPI / base policy checkpoint
- residual 投影相关参数，例如 `action_mask`、`action_limits`、`expert_reference_scale`

离线数据生成示例：

```bash
cd /vla/users/niejunnan/codebase/serl_torch/examples/libero
bash tools/convert_offline.sh \
  --suite_name libero_10 \
  --task_id 8 \
  --chunk_horizon 5 \
  --residual_alpha 0.10 \
  --output_dir data/residual_training/offline_pi0_alpha01
```

如果当前 OpenPI 服务换成 `pi05_libero`，就把输出目录改成：

```bash
data/residual_training/offline_pi05_alpha01
```

在线 warmup / prefill 收集示例：

```bash
cd /vla/users/niejunnan/codebase/serl_torch/examples/libero
bash tools/collect_online_prefill.sh \
  /vla/users/niejunnan/codebase/serl_torch/examples/libero/configs/exp11/chunk/libero_10_task_8_chunk_async_null.yaml \
  --episodes 100 \
  --output_dir data/residual_training/online_pi0
```

如果是 `pi05` 版本，就把 `output_dir` 换成：

```bash
data/residual_training/online_pi05
```

## 推荐方式：一键启动

如果你只是想直接把 `exp11` 跑起来，最省事的方法是用封装脚本。

示例：

```bash
bash /vla/users/niejunnan/codebase/serl_torch/examples/libero/configs/exp11/COMMANDS.sh launch
```

这条命令现在等价于：

```bash
bash /vla/users/niejunnan/codebase/serl_torch/examples/libero/configs/exp11/COMMANDS.sh null launch
```

也就是默认启动 `null` 这组配置。脚本会自动拉起上面那 5 个角色，内部实际调用的是：

```bash
bash /vla/users/niejunnan/codebase/serl_torch/examples/libero/tools/launch_async_train.sh \
  /vla/users/niejunnan/codebase/serl_torch/examples/libero/configs/exp11/chunk/libero_10_task_8_chunk_async_null.yaml
```

常见变体：

```bash
bash /vla/users/niejunnan/codebase/serl_torch/examples/libero/configs/exp11/COMMANDS.sh null_utdeff1p launch
bash /vla/users/niejunnan/codebase/serl_torch/examples/libero/configs/exp11/COMMANDS.sh null_unfreeze launch
bash /vla/users/niejunnan/codebase/serl_torch/examples/libero/configs/exp11/COMMANDS.sh null_pi05 launch
```

说明：

- `launch` 模式会自动启动 `env / async_eval_env / openpi / learner / actor`
- 日志和输出会落到 Hydra run 目录下，默认在 `examples/libero/outputs/exp11/<config_name>/<timestamp>/`
- 前台会显示 `actor` 的日志；其余后台服务日志在同一个 run 目录的 `support/` 下面
- 停止时直接在当前终端 `Ctrl-C` 即可，launcher 会一并清理它拉起的后台进程

## 原始方式：手动开 5 个终端

如果你想完全手动控制每个角色，可以按下面的方法分别启动。这里仍然以 `null` 这个 run 为例。

建议统一先约定下面这些路径：

```bash
ROOT=/vla/users/niejunnan/codebase/serl_torch
CFG=$ROOT/examples/libero/configs/exp11/chunk/libero_10_task_8_chunk_async_null.yaml
RUN_ROOT=$ROOT/examples/libero/outputs/exp11/manual/null
BOOTSTRAP=$RUN_ROOT/agentlace_bootstrap.pkl
```

你可以先在任意一个终端执行一次：

```bash
mkdir -p "$RUN_ROOT" "$RUN_ROOT/learner" "$RUN_ROOT/actor"
```

### 终端 1：训练环境服务

```bash
cd /vla/users/niejunnan/codebase/serl_torch
bash examples/libero/tools/serve_env.sh \
  --host 127.0.0.1 \
  --port 36790
```

### 终端 2：异步评测环境服务

```bash
cd /vla/users/niejunnan/codebase/serl_torch
bash examples/libero/tools/serve_env.sh \
  --host 127.0.0.1 \
  --port 36792
```

### 终端 3：OpenPI 服务

```bash
cd /vla/users/niejunnan/codebase/serl_torch
bash examples/libero/tools/serve_openpi.sh \
  --port 36791 \
  --gpu-id 0
```

### 终端 4：Learner

```bash
cd /vla/users/niejunnan/codebase/serl_torch
bash examples/libero/tools/run_learner.sh \
  /vla/users/niejunnan/codebase/serl_torch/examples/libero/configs/exp11/chunk/libero_10_task_8_chunk_async_null.yaml \
  --bootstrap /vla/users/niejunnan/codebase/serl_torch/examples/libero/outputs/exp11/manual/null/agentlace_bootstrap.pkl \
  --gpu_id 1 \
  hydra.run.dir=/vla/users/niejunnan/codebase/serl_torch/examples/libero/outputs/exp11/manual/null/learner
```

`learner` 启动后如果看到它在等待 bootstrap 文件是正常的，因为这个文件会由 `actor` 初始化。

### 终端 5：Actor

```bash
cd /vla/users/niejunnan/codebase/serl_torch
bash examples/libero/tools/run_actor.sh \
  /vla/users/niejunnan/codebase/serl_torch/examples/libero/configs/exp11/chunk/libero_10_task_8_chunk_async_null.yaml \
  --bootstrap /vla/users/niejunnan/codebase/serl_torch/examples/libero/outputs/exp11/manual/null/agentlace_bootstrap.pkl \
  --gpu_id 0 \
  hydra.run.dir=/vla/users/niejunnan/codebase/serl_torch/examples/libero/outputs/exp11/manual/null/actor
```

推荐启动顺序就是上面的顺序：

1. `env`
2. `async_eval_env`
3. `openpi`
4. `learner`
5. `actor`

停止时需要分别在 5 个终端里 `Ctrl-C`。

## 如果你想切换到别的 run

最简单的方法就是替换 run 名称或者替换配置文件路径。

例如：

- 一键启动 `pi05` 版本：`bash examples/libero/configs/exp11/COMMANDS.sh null_pi05 launch`
- 手动方式则把 `CFG` 换成对应的 `*_pi05.yaml`
- `pi05` 手动启动时，`openpi` 终端需要额外带上：

```bash
POLICY_CONFIG=pi05_libero \
POLICY_DIR=/vla/users/niejunnan/openpi-assets/checkpoints/pi05_libero \
bash examples/libero/tools/serve_openpi.sh --port 40011 --gpu-id 0
```

## 批量运行

如果你就是想把这 8 组配置依次都跑一遍，可以直接看：

```bash
/vla/users/niejunnan/codebase/serl_torch/examples/libero/configs/exp11/FULL_COMMANDS.sh
```

它里面就是一组按顺序执行的 `launch_async_train.sh` 命令。
