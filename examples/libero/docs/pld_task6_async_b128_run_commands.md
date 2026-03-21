# PLD Task6 Async B128 运行命令（8 卡）

说明：
- 以下 8 条命令均为绝对路径。
- 不使用 `nohup`。
- 显式指定 `training.max_online_env_steps=300000`。
- 需要 8 个终端会话分别执行（或用 tmux/screen 开 8 个 pane）。

## GPU 0

```bash
bash /vla/users/niejunnan/codebase/serl_torch/examples/libero/tools/run_train.sh /vla/users/niejunnan/codebase/serl_torch/examples/libero/conf/train_pld_task6_async_b128_m01_expert_w0_xi05.yaml --gpu_id 0 training.max_online_env_steps=300000
```

## GPU 1

```bash
bash /vla/users/niejunnan/codebase/serl_torch/examples/libero/tools/run_train.sh /vla/users/niejunnan/codebase/serl_torch/examples/libero/conf/train_pld_task6_async_b128_m02_expert_w50_xi05.yaml --gpu_id 1 training.max_online_env_steps=300000
```

## GPU 2

```bash
bash /vla/users/niejunnan/codebase/serl_torch/examples/libero/tools/run_train.sh /vla/users/niejunnan/codebase/serl_torch/examples/libero/conf/train_pld_task6_async_b128_m03_expert_w100_xi05.yaml --gpu_id 2 training.max_online_env_steps=300000
```

## GPU 3

```bash
bash /vla/users/niejunnan/codebase/serl_torch/examples/libero/tools/run_train.sh /vla/users/niejunnan/codebase/serl_torch/examples/libero/conf/train_pld_task6_async_b128_m04_boot50_w0_xi05.yaml --gpu_id 3 training.max_online_env_steps=300000
```

## GPU 4

```bash
bash /vla/users/niejunnan/codebase/serl_torch/examples/libero/tools/run_train.sh /vla/users/niejunnan/codebase/serl_torch/examples/libero/conf/train_pld_task6_async_b128_m05_boot50_w50_xi05.yaml --gpu_id 4 training.max_online_env_steps=300000
```

## GPU 5

```bash
bash /vla/users/niejunnan/codebase/serl_torch/examples/libero/tools/run_train.sh /vla/users/niejunnan/codebase/serl_torch/examples/libero/conf/train_pld_task6_async_b128_m06_boot50_w100_xi05.yaml --gpu_id 5 training.max_online_env_steps=300000
```

## GPU 6

```bash
bash /vla/users/niejunnan/codebase/serl_torch/examples/libero/tools/run_train.sh /vla/users/niejunnan/codebase/serl_torch/examples/libero/conf/train_pld_task6_async_b128_m07_boot50_w100_xi03.yaml --gpu_id 6 training.max_online_env_steps=300000
```

## GPU 7

```bash
bash /vla/users/niejunnan/codebase/serl_torch/examples/libero/tools/run_train.sh /vla/users/niejunnan/codebase/serl_torch/examples/libero/conf/train_pld_task6_async_b128_m08_boot50_w100_xi02.yaml --gpu_id 7 training.max_online_env_steps=300000
```
