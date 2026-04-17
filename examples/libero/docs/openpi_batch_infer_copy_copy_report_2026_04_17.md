# OpenPI `infer_many` + LIBERO `copy_copy` 实施与实测记录

## 这次改了什么

这次改动只动了两个地方：

- `serl_torch`
- `/vla/users/niejunnan/codebase/openpi-modified`

没有改 `/vla/users/niejunnan/codebase/openpi`。

目标是把之前只在 JoyRA 路径可用的 chunk 级 batch backfill 能力，补到 OpenPI 上，让 `examples/libero/scripts/run_residual_training_copy_copy.py` 在 `policy.type=openpi` 时也能真正走：

- 主 actor：单样本 `infer`
- backfill：每个 chunk 一次 `infer_many`

## 代码改动

### `openpi-modified`

- `src/openpi/policies/policy.py`
  - 新增 `Policy.infer_many(observations)`。
  - 实现方式不是直接改现有 transform 语义，而是：
    1. 每个 raw example 先各自跑一遍 input transform。
    2. transform 后的 tensor tree 再按 batch 维堆叠。
    3. 只调用一次模型 `sample_actions(...)`。
    4. 再按样本拆开，分别跑 output transform。
  - 这样避免了 `TokenizePrompt` / `TokenizeFASTInputs` 这类 transform 被强行改成“吃 batched raw prompt”的复杂改造。
- `src/openpi/serving/websocket_policy_server.py`
  - 新增 batch wire protocol：
    - 请求：`{"examples": [single_request_dict, ...]}`
    - 响应：`{"actions": [B,H,D], "policy_timing": {...}, "server_timing": {...}, "batch_size": B}`
  - 单样本老协议保持不变。
  - 顺手修了一个潜在 bug：原文件里 `except websockets.ConnectionClosed` 但没有 `import websockets`。
- `scripts/serve_policy.py`
  - metadata 里补了 `batch_infer_supported` 和 `batch_request_key=examples`。
- `scripts/benchmark_batch_infer.py`
  - 新增 direct policy benchmark，用 fake raw LIBERO-style obs 比较 `30 serial` 和 `1 batch of 30`。

### `serl_torch`

- `serl_launcher/serl_launcher/policy/openpi/request_builder.py`
  - 新增 `build_openpi_batch_request(...)`。
- `serl_launcher/serl_launcher/policy/openpi/client.py`
  - 新增 `infer_many(...)`。
  - 保持单样本 `infer(...)` 行为不变。
  - 加了 `close()`。
  - batch path 直接复用现有 websocket client 的 `infer(send_data)`，只是发送 payload 改成 `{"examples": ...}`，不依赖额外 client 包升级。
- `examples/libero/scripts/run_residual_training_copy_copy.py`
  - 修了一个 review 时发现的真实 bug：
    - actor 之前会先调用 `_maybe_enable_torch_compile(...)`
    - 然后再打印“actor 忽略 compile”
    - 这两个语义是冲突的
  - 现在 actor 只打印 ignore，不再真的 compile。
- `serl_launcher/tests/policy/test_openpi_batch_client.py`
  - 新增 OpenPI batch client 单测。

## 当前数据流

### 主 actor 路径

`run_residual_training_copy_copy.py` 的 actor 仍然保持：

1. 当前 observation 走 OpenPI 单样本 `infer(...)`
2. 得到 base chunk
3. residual agent 出 residual chunk
4. 合成最终 action chunk
5. `env.step_chunk(...)`

这里没有改成 batch，主决策语义不变。

### backfill 路径

当 `backfill_policy.enabled=true` 且 backend client 支持 `infer_many(...)` 时：

1. chunk 执行完成后，收集该 chunk 的 `post_step_observations`
2. 组装成 `Sequence[PolicyInput]`
3. 通过 OpenPI client 发一次 batch websocket 请求：
   - `{"examples": [...]}`
4. `openpi-modified` server 调 `policy.infer_many(...)`
5. server 返回 `actions[B,H,D]`
6. LIBERO 侧按样本顺序回填 `next_residual_observations`

也就是说，现在 OpenPI 路径已经和 JoyRA 一样，具备了“每 chunk 一次 batch backfill RPC”的能力。

## Review 结论

### 已修的问题

- `copy_copy` actor 的 `torch_compile` 语义 bug 已修。
- OpenPI server batch path 已补齐。
- OpenPI client batch path 已补齐。
- `PolicyRecorder.infer_many(...)` 已补，单样本 `PolicyRecorder.infer(...)` 的保存方式也顺手统一成 object save，避免原来 `np.asarray(data)` 这种不稳定存法。

### 还存在的潜在问题

- 端到端主瓶颈现在不只在 policy infer。
  - 这次两条 1000-step 真实 run 里，actor 都出现了大量：
    - `Failed to send message to 127.0.0.1:<trainer_port>: Resource temporarily unavailable`
    - `Failed to get last update id`
  - 说明 actor/learner 的 trainer socket 或 request/backpressure 也是明显瓶颈。
- `copy_copy` learner 这次 summary 没有完全追上 actor 的 1000 env steps。
  - 因为我在 actor 跑完后手动中断了 learner，summary 停在：
    - `env_steps=520`
    - `replay_size=701`
  - 这不代表 actor 没跑到 1000，而是说明 async backfill + learner drain 在这次手动结束前还没完全追平。
- batch server 目前不是 dynamic batching。
  - 现在只是 caller 显式把一个 chunk 的 obs 打成一个 batch。
  - 没有做“等 16 个请求再合并”的 server-side waiting window。

## 测试与结果

### 1. 单测 / 静态检查

通过：

- `conda run -n serl_torch python -m py_compile ...`
- `conda run -n serl_torch python serl_launcher/tests/policy/test_openpi_batch_client.py`
- `conda run -n serl_torch python serl_launcher/tests/libero/test_transition_assembly.py`
- `conda run -n serl_torch python serl_launcher/tests/libero/test_config.py`
- `conda run -n serl_torch python test/test_libero_eval_config_compile.py`
- `conda activate openpi-modified && python -m py_compile ...`

### 2. fake raw obs: 30 串行 vs 30 打包 batch

结果文件：

- [openpi_batch_infer_fake_obs_b30.json](/home/hello/codebase/serl_torch/test/results/openpi_batch_infer_fake_obs_b30.json)

实测：

- `serial_median_s = 2.8405`
- `batch_median_s = 1.2160`
- `speedup_vs_serial = 2.336x`

这个测试跑的是 direct policy benchmark，使用的是 fake 的 LIBERO-style raw obs：

- `observation/image`
- `observation/wrist_image`
- `observation/state`
- `prompt`

不是 HTTP env，也不是完整 actor/learner 链路。

### 3. 1000-step 端到端：`copy_copy`

运行目录：

- [libero_copy_copy_1000_actor](/home/hello/codebase/serl_torch/test/results/libero_copy_copy_1000_actor)
- [libero_copy_copy_1000_learner](/home/hello/codebase/serl_torch/test/results/libero_copy_copy_1000_learner)

说明：

- `wandb.debug=true`
- 没有在线写 W&B
- 用的是 `openpi-modified` policy server，不是 `/vla/users/niejunnan/codebase/openpi`

关键数字：

- actor `env_steps=1000`
- actor 外部 wall time：`182.98s`
- 从 actor log 的 `Hydra run dir` 到 `episode=2 ... env_steps=1000`：
  - `172.331s`
  - `1000 / 172.331 = 5.803 step/s`

actor timer 最后一条平均值：

- `sample_actions ~= 0.1627s`
- `step_env ~= 0.1517s`
- `build_decision_obs ~= 0.000078s`
- `total ~= 0.5025s`

这个很关键，说明 `copy_copy` 的 post-hoc 组装开销已经被压得很低了。

### 4. 1000-step 端到端：原始 `run_residual_training.py`

运行目录：

- [libero_run_residual_1000_actor](/home/hello/codebase/serl_torch/test/results/libero_run_residual_1000_actor)
- [libero_run_residual_1000_learner](/home/hello/codebase/serl_torch/test/results/libero_run_residual_1000_learner)

关键数字：

- actor `env_steps=1000`
- actor 因 shutdown 阶段卡住，我手动中断了进程
- 所以这里使用 actor log 里真正到达 `env_steps=1000` 的时间：
  - `195.902s`
  - `1000 / 195.902 = 5.105 step/s`

actor timer 最后一条平均值：

- `sample_actions ~= 0.0165s`
- `step_env ~= 0.0345s`
- `build_decision_obs ~= 0.1077s`
- `total ~= 0.8509s`

这也很关键：原始 step-wise 训练线里，`build_decision_obs` 明显比 `copy_copy` 大很多。

## 速度结论

### 模型侧

OpenPI 真 batch 已经生效：

- `30 serial` -> `2.8405s`
- `1 batch of 30` -> `1.2160s`
- 约 `2.34x` 加速

### 端到端 actor

按真正跑到 `env_steps=1000` 的日志时间计算：

- `copy_copy`: `5.803 step/s`
- `run_residual_training.py`: `5.105 step/s`

提升约：

- `5.803 / 5.105 = 1.137x`
- 约 `13.7%`

### 为什么端到端没有吃满 2.34x

因为现在端到端不只卡 policy infer，还卡：

- trainer socket timeout / backpressure
- learner 更新与 actor 请求之间的同步开销
- remote env RPC
- episode boundary / stats 发送

这也是为什么：

- 模型侧 batch 提速很明显
- actor 侧总吞吐只提升了一截，但没有翻倍

## 我对当前实现的判断

### 已经实现了吗

已经实现了。

更准确地说：

- OpenPI server 现在支持 batch request
- OpenPI client 现在支持 `infer_many`
- `copy_copy` 现在在 OpenPI backend 下，确实能走 chunk 级 batched backfill

### 当前是否正确

我认为“功能上是正确的”，因为：

- 单测过了
- fake batch benchmark 证明模型侧真 batch 生效
- 真实 1000-step 也跑通了
- `copy_copy` 的 `build_decision_obs` 平均耗时已经接近被清空

### 当前最值得继续做的事

如果你接下来继续追端到端吞吐，我建议优先看这两个方向：

1. trainer / actor 通信瓶颈
   - 这次两条线都出现了大量 timeout warning
   - 这已经不是 OpenPI batch 自己能解决的问题了
2. async backfill 的 episode-end drain 语义
   - 这次 `copy_copy` actor 已到 1000 steps，但 learner summary 没完全追平
   - 如果要做更干净的 benchmark，最好让 learner 在 actor 完成后再自然 drain 一小段，而不是我这样手动打断

## 相关文件

- [serl_launcher/serl_launcher/policy/openpi/client.py](/home/hello/codebase/serl_torch/serl_launcher/serl_launcher/policy/openpi/client.py)
- [serl_launcher/serl_launcher/policy/openpi/request_builder.py](/home/hello/codebase/serl_torch/serl_launcher/serl_launcher/policy/openpi/request_builder.py)
- [serl_launcher/tests/policy/test_openpi_batch_client.py](/home/hello/codebase/serl_torch/serl_launcher/tests/policy/test_openpi_batch_client.py)
- [examples/libero/scripts/run_residual_training_copy_copy.py](/home/hello/codebase/serl_torch/examples/libero/scripts/run_residual_training_copy_copy.py)
- [openpi-modified/src/openpi/policies/policy.py](/vla/users/niejunnan/codebase/openpi-modified/src/openpi/policies/policy.py)
- [openpi-modified/src/openpi/serving/websocket_policy_server.py](/vla/users/niejunnan/codebase/openpi-modified/src/openpi/serving/websocket_policy_server.py)
- [openpi-modified/scripts/serve_policy.py](/vla/users/niejunnan/codebase/openpi-modified/scripts/serve_policy.py)
- [openpi-modified/scripts/benchmark_batch_infer.py](/vla/users/niejunnan/codebase/openpi-modified/scripts/benchmark_batch_infer.py)
