# 2026-04-13 Mainline Review Findings

## 文档信息

- 文档类型：主线代码 review 记录
- 记录时间：北京时间 2026-04-13 16:53:39（UTC+08:00）
- 记录分支：`refactor/agibot-real-libero-alignment`
- 记录提交：`3a7fa89`
- 复核范围：
  - `serl_launcher/`
  - `examples/libero/`
  - `examples/agibot_real/`
- 明确不包含：
  - `examples/RoboTwin/`
  - `reference/`
  - vendored 第三方仓库的深度审计

这份文档是基于当前主线代码的复核记录，不是一次正式的安全审计或真机端到端联调报告。这里记录的是“当前仍然能在代码里定位到、并且值得优先关注的问题”。

另外说明一下：上一次口头梳理里提到过一个和外部 `pickle.load` 相关的历史数据加载风险；但在本次对当前主线的复核里，那个旧路径已经不再出现在当前主线上，所以本文不再把它列为当前主线问题。

## 总结

当前主线里，我认为最值得优先处理的风险有 6 类：

1. 高风险：remote env RPC 直接使用 `pickle` 作为 HTTP 边界协议，存在反序列化执行和 DoS 风险
2. 高风险：checkpoint / classifier 加载仍然直接 `torch.load`，对不可信权重文件没有防护
3. 高风险：AgiBot 真机场景在传感器读取失败时会静默回退到旧观测
4. 中风险：AgiBot learner 的 `update_steps` 计数存在一拍偏移
5. 中风险：AgiBot controller 在终止信号和首个 executed transition 之间存在竞态
6. 中风险：LIBERO async eval 的 `eval_index` 没有持久化，重启后可能静默丢评测

如果只能先修一部分，我建议优先顺序是：

1. remote HTTP `pickle`
2. 不安全的 `torch.load`
3. 真机 stale observation
4. controller 终止竞态
5. learner `update_steps`
6. async eval `eval_index`

---

## 1. 高风险：remote env RPC 直接把 `pickle` 暴露在 HTTP 边界上

### 代码位置

- [`serl_launcher/serl_launcher/envs/remote_http.py`](../serl_launcher/serl_launcher/envs/remote_http.py)
- [`examples/libero/scripts/serve_env.py`](../examples/libero/scripts/serve_env.py)

### 现象

当前 remote env client 和 server 都在 HTTP 边界直接做 `pickle` 反序列化：

- client 收到响应后直接 `pickle.loads(resp_bytes)`
- server 收到请求体后直接 `pickle.loads(raw)`

服务端代码在真正校验 RPC method 之前，就已经执行了反序列化。

### 为什么这是问题

`pickle` 不是给“不可信输入”设计的协议。只要边界外部的数据源不完全可信，`pickle.loads(...)` 本身就可能触发任意对象构造、副作用执行，或者至少造成大内存 / 大对象分配压力。

这里的问题有三层：

1. 协议层不安全  
   HTTP body 直接是 pickle payload，本质上等于“谁能发请求，谁就能喂反序列化输入”。

2. 没有访问控制  
   当前实现没有 token、签名、mTLS、白名单来源校验。

3. 没有 body 大小上限  
   `Content-Length` 会被直接读取，代码里没有请求大小限制，DoS 面也在。

### 当前风险边界

这里要客观一点讲：当前 [`examples/libero/scripts/serve_env.py`](../examples/libero/scripts/serve_env.py) 的默认 `--host` 是 `127.0.0.1`，所以“默认配置下”风险面比绑定 `0.0.0.0` 小很多。

但这个边界仍然很脆弱。只要出现下面任意一种情况，风险就会重新暴露出来：

- 把 `--host` 改成 `0.0.0.0`
- 做端口映射 / SSH 转发 / 容器共享网络
- 同机存在其他不可信进程
- 后续有人把这套 remote env server 复用到更开放的内网环境里

### 可能后果

- 远程代码执行风险
- 非预期对象构造
- 单机内存打爆或请求阻塞
- env server 被恶意请求拖死，训练 actor 连带卡住

### 建议修复

推荐修复分三层：

1. 根修方案  
   不再把 `pickle` 放在网络边界上。改成：
   - JSON + numpy base64
   - `msgpack`
   - 明确 schema 的二进制协议

2. 过渡方案  
   如果短期内必须保留 pickle：
   - 明确限制只能绑定 `127.0.0.1`
   - 加 token / 签名校验
   - 对 `Content-Length` 增加硬上限
   - 在 README 和代码注释里标明“trusted local only”

3. 运行时隔离  
   - 只允许本机 loopback
   - 尽量放在单用途进程或容器里
   - 训练机和环境机分权

---

## 2. 高风险：checkpoint / classifier 加载仍然直接 `torch.load`

### 代码位置

- [`serl_launcher/serl_launcher/utils/checkpoint_utils.py`](../serl_launcher/serl_launcher/utils/checkpoint_utils.py)
- [`serl_launcher/serl_launcher/networks/reward_classifier.py`](../serl_launcher/serl_launcher/networks/reward_classifier.py)

### 现象

当前主线仍然有直接加载外部 `.pt` 文件的路径：

- `load_checkpoint_payload(...)` 里直接 `torch.load(checkpoint_path, map_location=...)`
- `load_classifier_func(...)` 里直接 `torch.load(ckpt_path, map_location="cpu")`

### 为什么这是问题

PyTorch 的默认 `torch.load` 底层同样基于 pickle 语义。也就是说，只要 checkpoint 文件来源不可信，它就不是“普通数据文件”，而是“可能在加载阶段执行任意构造逻辑的对象载体”。

在实验环境里，这个风险经常被低估，因为大家默认 checkpoint 来自自己训练。但真实协作里很常见下面几种情况：

- 权重来自共享盘
- 权重来自别人导出的实验目录
- 权重来自下载资产
- 权重来自历史缓存，来源已经说不清

### 可能后果

- 本地任意代码执行
- 加载时污染 Python 进程状态
- 训练 / 评估机被恶意 checkpoint 拿下

### 建议修复

1. 优先使用只含 tensor 的安全格式  
   例如 `safetensors`，或者只存 `state_dict`。

2. 如果继续使用 PyTorch checkpoint  
   评估是否可以迁到：
   - `torch.load(..., weights_only=True)`  
     前提是当前 PyTorch 版本和 payload 结构支持

3. 增加来源约束和 schema 校验  
   - 只接受特定目录
   - 只接受预期字段
   - 对 checkpoint 做 hash / manifest 校验

4. 文档上明确 trust boundary  
   当前这类接口不应该默认被理解为“可以加载任意外部文件”。

---

## 3. 高风险：AgiBot 真机在传感器失败时会静默复用旧观测

### 代码位置

- [`examples/agibot_real/env/task_env.py`](../examples/agibot_real/env/task_env.py)

### 现象

`_get_obs()` 当前逻辑是：

- 拉 head / left wrist / right wrist 图像
- 拉 joint state
- 只要其中任意一个返回 `None`
  - 如果有 `_last_obs`，就直接返回 `_last_obs`
  - 否则抛异常

### 为什么这是问题

对离线仿真来说，拿上一帧补一下也许只是“数据不新鲜”；但对真机 residual RL，这个语义危险得多，因为它会把“观测获取失败”伪装成“正常观测”。

这意味着策略会继续基于过期图像和过期状态做决策，而机器人在现实世界里已经继续运动了。

### 可能后果

- 动作基于旧画面继续执行
- replay buffer 记录下错误的 `(obs, action, next_obs)` 对应关系
- 真机出现视觉-动作错位
- 调试时很难第一时间定位，因为系统没有显式报“观测失败”

### 为什么我把它评成高风险

这不是单纯的数据质量问题，而是“系统在真机上悄悄进入降级模式，但上层训练和控制逻辑完全不知情”。对机器人系统来说，这类 silent fallback 往往比显式 fail-fast 更危险。

### 建议修复

更稳妥的实现一般是：

1. 先做有限次短重试  
   例如 2 到 3 次，间隔几十毫秒

2. 超过重试阈值后显式失败  
   例如：
   - 直接抛异常中断 episode
   - 或返回一个带错误标记的 terminal / truncated transition

3. 把观测失败写入日志和 episode info  
   让后续排查能明确看到是 sensor fault，而不是 reward / policy 问题

如果团队确实想保留“允许短暂回退上一帧”的降级模式，也应该：

- 显式加配置开关
- 在 info 里打标
- 限制连续回退次数

---

## 4. 中风险：AgiBot learner 的 `update_steps` 计数存在一拍偏移

### 代码位置

- [`examples/agibot_real/scripts/run_residual_training.py`](../examples/agibot_real/scripts/run_residual_training.py)

### 现象

当前 learner 主循环顺序是：

1. 训练一次
2. 用旧的 `update_steps` 做：
   - publish network
   - wandb log
   - checkpoint
3. 最后才 `update_steps += 1`

### 为什么这是问题

`update_steps` 语义上应该表示“已经完成的参数更新次数”。但当前实现实际更像“这次更新之前的计数值”。

直接后果是：

- 第一次更新后，`update_steps` 仍然是 `0`
- publish / checkpoint / logging 都会落后一拍
- 当 `steps_per_update == 1` 时，第一次更新后的网络不会马上广播出去

### 影响

这类问题通常不会把系统直接跑挂，但会让下面几件事变得很混乱：

- checkpoint 名字和真实参数状态不一致
- actor 可能晚一拍收到新参数
- 训练曲线和实际模型状态对不上

### 建议修复

最简单的修法是把顺序改成：

1. 完成一次 update
2. 立刻 `update_steps += 1`
3. 后续 publish / log / checkpoint 全部使用新的计数

如果需要区分“当前循环序号”和“已完成更新数”，那就拆成两个变量，不要混用一个 `update_steps`。

---

## 5. 中风险：AgiBot controller 在终止信号与首个 executed transition 之间有竞态

### 代码位置

- [`examples/agibot_real/env/task_env.py`](../examples/agibot_real/env/task_env.py)

### 现象

`_controller_execute_chunk_blocking(...)` 的行为大致是：

- 先 enqueue 一段 action chunk
- 轮询 controller transition
- 如果拿到 executed payload，就组装 transition
- 如果看到 terminal signal，就尝试结束

问题在于：如果 terminal signal 先到，但第一条 executed transition 还没被读到，那么：

- `pending` 仍然是 `None`
- 函数可能直接返回空列表
- 上层 `step()` / `step_chunk()` 会把它当成 internal error
  - `controller step produced no executed transition`
  - `controller step_chunk produced no executed transitions`

### 为什么这是问题

这类情况在真机上并不罕见，比如：

- 人工很快按了 success / fail
- reset 信号先被消费
- controller 正在切状态，但 transition 还没来得及出队

从业务语义看，这应该是“episode 终止得比较早”，不是“程序内部错误”。但当前实现会把它升级成 `RuntimeError`。

### 可能后果

- actor 进程被异常打断
- episode 没有被干净收尾
- 上层训练逻辑把人为成功 / 失败误判成系统故障

### 建议修复

比较合理的处理方式有两类：

1. 合成终止 transition  
   当 terminal meta 已经明确出现时，即使还没有 executed payload，也返回一个结构完整的终止结果。

2. 上层接受“零 executed step 的合法终止”  
   `step()` / `step_chunk()` 不要直接把空列表视为 bug，而是进一步检查 controller meta 是否已经终止。

这个问题本质上是“终止语义”和“执行确认语义”耦得太紧，应该拆开。

---

## 6. 中风险：LIBERO async eval 的 `eval_index` 没有持久化

### 代码位置

- [`examples/libero/runtime/async_eval_runtime.py`](../examples/libero/runtime/async_eval_runtime.py)
- [`examples/libero/scripts/run_residual_training_1_baseline.py`](../examples/libero/scripts/run_residual_training_1_baseline.py)
- [`examples/libero/runtime/async_eval_worker.py`](../examples/libero/runtime/async_eval_worker.py)

### 现象

当前 async eval runtime 启动时：

- `AsyncEvalRuntime.triggered_count` 从 `0` 开始
- learner 组装请求时直接用当前 `triggered_count` 作为 `eval_index`
- worker 侧把 `summary_jsonl` 中已经完成的 `eval_index` 当作去重键

这意味着：如果复用同一个 run dir 重启 learner，新的 async eval 请求会从 `0` 重新编号。

### 为什么这是问题

worker 的去重逻辑是“只看 `eval_index` 是否已经完成过”，不看：

- checkpoint 路径
- checkpoint step
- 请求生成时间

于是只要历史 `summary_jsonl` 里已经存在同编号记录，新的请求就会被默默当成“已处理过”。

### 可能后果

- 重启后某些 checkpoint 的 eval 根本没跑
- 训练 summary 看起来正常，但 async eval 实际缺失
- 后续分析成功率曲线时会以为是“没有触发”，而不是“触发了但被静默去重”

### 建议修复

至少要做下面一件：

1. 持久化单调递增计数器  
   启动时从历史结果里取 `max(eval_index) + 1`

2. 改用更稳定的主键  
   例如：
   - `checkpoint_step`
   - `checkpoint_path`
   - `(checkpoint_step, enqueue_timestamp)`
   - UUID

3. worker 去重不要只看 `eval_index`  
   当前这个键太容易碰撞。

---

## 一个具体例子：为什么 remote HTTP `pickle` 不是“小问题”

假设有这样一个很现实的使用场景：

1. 某台机器上运行了：
   - `python examples/libero/scripts/serve_env.py --host 0.0.0.0 --port 30000`
2. 训练 actor 通过 remote env 连接这个服务
3. 这台机器同时在实验室内网里，对同网段其他机器可见

这时只要有人能够向这个端口发 POST 请求，就可以向 server 提交一个“不是普通数据，而是带反序列化副作用的 pickle body”。

关键点在于：

- server 会先 `pickle.loads(raw)`
- method 名字是否合法，是在反序列化之后才检查的

也就是说，即使最终 method 不存在，风险也已经发生在更前面了。

这就是为什么这个问题不能简单理解成“RPC method 没鉴权”。更准确地说，它是“把不可信输入直接喂给了一个不安全反序列化器”。

这个例子里我没有给利用代码，因为重点不是如何利用，而是说明这个边界在设计上本身就不适合承载不可信网络输入。

---

## 再补一个更贴近真机的例子：stale observation 会怎么坑人

假设 AgiBot 正在执行抓取：

1. 策略基于当前头部相机画面，判断目标在右手前方
2. 机器人开始移动
3. 这一时刻右腕相机读取失败，`_get_obs()` 没有报错，而是直接返回上一帧 `_last_obs`
4. 上层策略和 replay 逻辑都以为这是一帧正常新观测
5. 系统继续输出下一步动作

结果是：

- 机器人真实状态已经变化了
- 但算法看到的还是旧图像和旧状态
- 这会把控制错误和训练数据污染同时引入系统

从排查体验上看，这类问题很烦，因为日志里不会明确写“这一帧观测失败了，我回退到了旧值”，最后大家很容易把问题误判成：

- residual policy 学坏了
- reward 有问题
- 相机标定不准
- OpenPI / JoyRA 推理不稳定

实际上根因只是“观测失败被 silent fallback 掩盖了”。

---

## 建议的后续动作

如果把这份文档转成工程动作，我建议按下面顺序推进：

1. 先给 `remote_http.py` 和 `torch.load` 路径补 trust-boundary 注释与 README 警告
2. 处理 AgiBot `_get_obs()` 的 stale fallback，改成有限重试 + 显式失败
3. 修 AgiBot learner `update_steps` 计数顺序
4. 修 controller 终止竞态，保证“早终止”不会被抛成内部错误
5. 给 async eval 增加持久化计数器或稳定请求 ID
6. 后续再考虑把 remote env RPC 从 pickle 迁走

## 结论

当前主线最大的问题不是“算法逻辑完全跑不通”，而是：

- 有几处 trust boundary 设计得还不够安全
- 有几处真机错误处理是 silent degradation，而不是 fail-fast
- 有几处训练元数据和异步流程的 bookkeeping 还不够稳

这些问题单独看都不一定会立刻炸，但叠在一起时，会明显增加：

- 安全面
- 真机调试成本
- 训练结果解释成本
- 故障排查时间

所以这份 review 的核心建议不是“大改架构”，而是优先把这些边界条件收紧，让系统在出问题时更早、更明确地暴露出来。
