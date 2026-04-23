# 2026-04-23 LIBERO Residual Training 1-5 脚本差异、演化顺序与推荐结论

## 文档信息

- 范围：
  `examples/libero/scripts/run_residual_training_1_baseline.py`
  `examples/libero/scripts/run_residual_training_2_chunk_local.py`
  `examples/libero/scripts/run_residual_training_3_split_proto.py`
  `examples/libero/scripts/run_residual_training_4_split_refined.py`
  `examples/libero/scripts/run_residual_training_5_split_pipeline.py`
- 同时参考的关键辅助模块：
  `examples/libero/config.py`
  `examples/libero/runtime/transition_assembly.py`
  `examples/libero/runtime/processor_protocol.py`
  `examples/libero/runtime/processor_pipeline.py`
  `examples/libero/runtime/processor_dispatch.py`
  `examples/libero/runtime/cadence.py`
  `serl_launcher/serl_launcher/common/trainer_session.py`
  `serl_launcher/serl_launcher/rollout/processor_runtime.py`
- 判断边界：
  这份文档的前半部分主要基于代码结构、数据流、协议边界和工程演进来判断；但在 2026-04-23 这次整理里，我又补跑了同配置下的 1500-step 训练，所以后半部分已经包含真实运行结果，而不只是工程推断。

## 0.1 2026-04-23 实跑结果

### 运行设置

- 统一任务：
  `libero_10 / task_id=8`
- 统一训练上限：
  `training.max_env_steps=1500`
  `training.max_update_steps=1500`
- 统一 policy backend：
  `openpi-modified`
- 统一显存设置：
  `XLA_PYTHON_CLIENT_MEM_FRACTION=0.3`
- 统一拓扑约束：
  每组 `actor + decision policy + backfill policy` 放在第一张卡；
  `learner` 放在第二张卡；
  `processor` 仅用于 `3/4/5`，放 CPU 路径
- 实际 GPU 分配：
  `5 -> (6,7)`
  `4 -> (4,5)`
  `3 -> (2,3)`
  `2 -> (0,1)`
  `1 -> (0,1)`，在 `2` 完整结束后复用同一对卡
- 运行产物目录：
  `outputs/libero_ablation_1500/parallel_reverse_2026-04-23/`

### 特殊修复

- `run_residual_training_3_split_proto.py`
  原始版本在冷启动背压下，`submit-chunk` 只允许固定 `5` 次重试，processor 队列短暂打满时 actor 会过早 abort。
- 本次补了一个很小的鲁棒性修复：
  把 `submit-chunk` 的重试预算改成和长请求一致的 `long_request_retry_limit`。
- 修复文件：
  `examples/libero/scripts/run_residual_training_3_split_proto.py`

### 吞吐量口径

- `1_baseline` 的 actor timer 是按 step 记时，所以 `actor env_steps/s = 1 / actor.timer.total`
- `2/3/4/5` 的 actor / processor timer 是按 chunk 记时，而 `chunk_horizon=5`
- 所以：
  `actor env_steps/s = 5 / actor.timer.total`
  `processor env_steps/s = 5 / processor.timer.total`
- learner 直接使用最后一条 `learner_timers.jsonl` 的 `updates_per_sec`

### 1500-step 实测结果

| 版本 | 运行目录 | actor env_steps/s | processor env_steps/s | learner updates/s | 结果 | 说明 |
| --- | --- | ---: | ---: | ---: | --- | --- |
| `1_baseline` | `1_baseline` | `2.12` | `-` | `1.093` | 跑通 | reference baseline，step-wise actor 明显最慢 |
| `2_chunk_local` | `2_chunk_local` | `15.43` | `-` | `1.116` | 跑通 | 当前最稳、也最省工程复杂度的优化线 |
| `3_split_proto` | `3_split_proto_fix1` | `12.73` | `13.55` | `1.075` | 跑通 | 需补重试预算修复；原型版背压处理明显偏脆弱 |
| `4_split_refined` | `4_split_refined` | `12.88` | `13.57` | `1.096` | 跑通 | split 线里最均衡、最“正常”的一版 |
| `5_split_pipeline` | `5_split_pipeline` | `16.46` | `16.31` | `1.090` | 跑通 | actor 原始推进最快；最终完成性要看 processor/learner drain，而不是 actor 早期 summary |

### 结果解读

1. 如果只看 actor 原始推进速度：
   `5 > 2 > 4 ≈ 3 >> 1`
2. 如果看 learner 稳态更新速度：
   `2` 略高，`4/5/1` 基本同一档，`3` 稍慢。
3. `5_split_pipeline` 的 actor timer 最漂亮，但它把 actor/processor 解耦得最彻底，所以不能只看 actor 自己的 summary；必须以 processor 最终 `committed_env_steps=1500` 和 learner `update_steps=1500` 为准。
4. `2_chunk_local` 仍然是我最推荐的“日常实验入口”：
   没有 split runtime 的复杂度，但 actor 吞吐和 learner 速度都已经很强。
5. `4_split_refined` 是 split 线里最值得继续维护的版本：
   它没有 `3` 那么原型化，也没有 `5` 那么容易把 raw actor throughput 和 end-to-end drain 解读混在一起。
6. `3_split_proto` 说明 split 方向本身没错，但原型实现对启动期背压的容忍度不够，工程上离“默认可用”还差一截。

## 0. 先给结论

如果只看一句话结论，可以直接记住下面这 6 点。

1. `1_baseline` 是语义最直接、最适合做对照和 debug 的 reference baseline。
2. `2_chunk_local` 是当前最适合日常跑实验的版本，也是我最推荐的版本。
3. `3_split_proto` 是 split 架构的第一代原型，方向对，但明显还在原型期。
4. `4_split_refined` 是把 `3` 的原型逻辑收敛成较统一工程结构的过渡版。
5. `5_split_pipeline` 是 split 线里最成熟的一版，职责边界最清楚，但它更像“架构演进线”，不是默认起跑线。
6. 这五个脚本的核心 RL 算法并没有根本变化，变化主要发生在 runtime、rollout dataflow、transition assembly、transport 和进程边界上。

换句话说：

- `1 -> 2` 的核心是性能优化，但尽量不改训练语义。
- `2 -> 3 -> 4 -> 5` 的核心是架构拆分和工程收敛。

## 1. 五个脚本的共同核心

先把“不变的部分”讲清楚，否则很容易误以为 `1/2/3/4/5` 是五套完全不同的算法。

这五个脚本在核心训练语义上大体共享下面这条主线：

1. base policy 输出一个 `chunk_horizon` 长度的 `base_action_chunk`
2. 当前观测被整理成 residual observation
3. residual agent 输出 residual action chunk
4. `ResidualActionSpec` 把 base action 和 residual action 组合成最终动作
5. replay 中最终存的仍然是按 step 展开的 transition
6. learner 仍然从 step replay 中按 chunk window 采样训练 batch

这里最重要的一点是：

- 这些版本都不是“直接按 chunk 存 replay 再按 chunk 训练”的另一套算法
- 它们仍然是“chunk 执行 + step transition 存储 + chunk-window 采样”

这也是为什么：

- `1/2/3/4/5` 的主要差异不是 SAC/DRQ 更新公式
- 而是 actor 何时推环境、何时构造 `next_residual_obs`、何时提交 replay、由谁来做 assembly、以及进程之间如何同步

如果训练结果出现差异，更可能来自下面这些因素：

- actor 热路径是否更短
- backfill 是否异步
- replay commit 是否有延迟
- stats/progress 是否和 replay commit 对齐
- split 架构是否引入了额外背压或更好的解耦

而不是因为这五版用了五种完全不同的 residual RL 算法。

## 2. 总览对比表

| 版本 | 角色拓扑 | env 执行单元 | transition assembly 在哪里做 | stats / progress 由谁发给 learner | 工程定位 | 我的判断 |
| --- | --- | --- | --- | --- | --- | --- |
| `1_baseline` | `actor + learner` | `step` | actor 主线程逐 step 内联完成 | actor 直接发 `send-stats` | reference baseline | 最适合做对照和 debug |
| `2_chunk_local` | `actor + learner` | `chunk` | actor 本地 post-hoc 组装，可选 async backfill | actor 直接发 `send-stats` | 当前最稳的优化线 | 最推荐的日常实验入口 |
| `3_split_proto` | `actor + processor + learner` | `chunk` | 独立 processor | actor 仍直接发 `send-stats` | 第一代 split 原型 | 方向正确，但原型味重 |
| `4_split_refined` | `actor + processor + learner` | `chunk` | 独立 processor | actor 直接发 `send-stats` 和 `actor-progress` | split 过渡整理版 | 比 `3` 明显更成熟 |
| `5_split_pipeline` | `actor + processor + learner` | `chunk` | 独立 processor pipeline | processor 在 flush 后转发 `send-stats` 和 `actor-progress` | 最新 split / pipeline 版 | split 线最成熟，但不是默认实验入口 |

## 3. 关键差异按维度梳理

### 3.1 角色拓扑：从两段式到三段式

#### `1_baseline`

- 只有 `actor` 和 `learner`
- actor 既负责环境推进，也负责 transition 构造与 replay 提交

这个结构的优点是简单直接，排障成本最低。

#### `2_chunk_local`

- 仍然只有 `actor` 和 `learner`
- 但 actor 内部已经出现了一个明确的“transition assembly 子阶段”

它还没有拆出第三个进程，但已经在 actor 内部把“控制推进”和“transition 后处理”分开了。

#### `3/4/5`

- 正式变成 `actor + processor + learner`
- actor 主要负责控制推进与 chunk rollout
- processor 主要负责把 chunk 恢复成 step-level residual transitions
- learner 继续做 replay 采样和训练

这条线的本质目标，不是再追求一点点单机优化，而是把 runtime 职责拆干净，让 actor 的控制路径更纯粹。

### 3.2 env 执行单元：`step` 到 `chunk`

#### `1_baseline`

`1_baseline` 的 actor 是最传统的做法：

- residual agent 先产出一个动作 chunk
- 然后 actor 仍然按 step 执行 `env.step(action)`
- 每一步之后立刻为下一步重新构造 decision obs

这意味着 actor 热路径里反复出现：

- `policy_client.infer(...)`
- `build_chunk_residual_obs(...)`
- `env.step(...)`

语义很清楚，但控制路径负担重。

#### `2/3/4/5`

这几版都切到了 `env.step_chunk(...)`：

- actor 一次拿到整个 chunk 的执行结果
- 再根据 chunk 的执行轨迹去恢复 step-level transition

这样做的本质收益是：

- actor 与环境交互的系统调用次数减少
- 下一步 residual observation 的构造可以从逐 step 热路径中搬出去
- 更适合把后处理做成 batch-aware 或异步

这里 `2` 是最关键的分水岭，因为它第一次把这件事做对了，同时又没有把整个系统拆得过重。

### 3.3 `next_residual_obs` 是怎么来的

这是五个脚本最关键的差别之一。

#### `1_baseline`：逐 step 现算

在 `1_baseline` 里，每执行一步：

1. 得到 `next_obs`
2. 立刻再跑一次 base policy
3. 构造 `next_base_actions`
4. 构造 `next_residual_obs`
5. 立刻写一条 step transition

优点：

- 语义最直观
- 没有额外中间层

缺点：

- actor 热路径很重
- `policy_client.infer(...)` 的次数最多

#### `2_chunk_local`：chunk 执行后统一回填

`2_chunk_local` 把 actor 数据流明确改成：

`step_chunk -> raw chunk -> post-hoc assembly -> step replay`

这版最重要的点不是“改成 chunk replay”，而是：

- env 的执行单位变成 chunk
- replay 的存储单位仍然是 step transition
- `next_residual_obs` 变成了 chunk 后处理的一部分

如果不开 async backfill：

- actor 在本地同步完成整段 assembly

如果开 async backfill：

- actor 主线程只负责推进环境和产生 raw chunk
- 后台线程负责整段 backfill 与 assembly
- 但 replay commit 仍保持顺序

这是一个很好的折中：

- 把最耗时的观测回填从 step 热路径里拿掉
- 又没有引入独立 processor 进程的额外运维复杂度

#### `3/4/5`：由 processor 重构 raw chunk 再 assembly

split 线里，actor 不再自己负责把 chunk 展成 step transition。

典型路径变成：

1. actor 执行 `env.step_chunk(...)`
2. actor 把 `chunk_result`、`episode_id`、`chunk_seq`、`task_prompt` 之类的上下文发给 processor
3. processor 用起始观测重建 decision obs
4. processor 做 post-step backfill
5. processor 组装出 step transition 并写入 replay

这里 `3/4/5` 的关键共同点是：

- actor 只要把 rollout chunk 交出去
- processor 再根据 chunk payload 恢复训练所需的 residual transition

这让 actor 真正接近“控制进程”，而不是又控环境又做数据预处理的混合体。

### 3.4 replay 写入与 commit 语义

#### `1_baseline`

- 直接用 `TrainerClient` / `TrainerServer`
- actor 写入 `QueuedDataStore`
- learner 作为 trainer server 注册 replay buffer
- 控制语义最简单

但这条线没有显式强化“actor 已结束并且所有在线数据都 committed 完毕后 learner 如何退出”的问题。

#### `2_chunk_local`

`2_chunk_local` 的一大改进是把 transport 也升级了：

- 默认使用 `train_residual_optimized.yaml`
- `runtime.trainer_transport.mode=async_commit`
- learner 有显式 `_should_stop_after_actor_done(...)`
- 停止条件会同时看：
  - 当前 `env_steps`
  - replay 已提交条数
  - transport 的 `accepted_update_id` / `committed_update_id`

也就是说，它不只是让 actor 更快，还补上了比较像样的 transport 完整性语义。

#### `3_split_proto`

`3` 把 processor 也纳入 commit 语义，但实现还比较原型化：

- processor 在脚本里自己维护 queue、condition、accepted/committed 状态
- episode end 和 shutdown flush 也写在脚本里

从功能上说，这版已经把“processor 负责 replay 写入”讲通了。
但从工程上说，协议和状态机还过于手工。

#### `4_split_refined`

`4` 的关键价值是把上面的手工逻辑沉到更稳定的公共抽象里：

- `ProcessorClient`
- `ProcessorServer`
- `TrainerClientSession`

这样 `actor`、`processor` 和 `learner` 不再各自重复写“失败重试 + status + flush”。

#### `5_split_pipeline`

`5` 不只是继续 split，而是把 split 的边界真正讲清楚：

- actor 把 chunk 和 episode end marker 发给 processor
- processor 只有在相关 chunk 真正处理完成之后，才 flush 对应 episode 的 stats/progress

这是 `5` 相对 `4` 的一个非常重要的提升：

- learner 看到的 `send-stats` / `actor-progress`
- 不再只是“actor 说我这个 episode 结束了”
- 而是“processor 确认这批 chunk 已经按顺序处理并刷进 replay 了”

这个一致性比 `4` 更强。

### 3.5 stats 与 progress 的归属是怎么变化的

这也是 `3 -> 4 -> 5` 里很值得单独看的一个点。

#### `1` 和 `2`

- actor 直接向 learner 发 `send-stats`

结构简单，但 stats 的含义就是“actor 这边 episode 结束了”。

#### `3`

虽然已经有 processor 了，但 actor 仍然自己向 learner 发 `send-stats`。

这说明 `3` 的 split 还只是把 replay assembly 拆出去，episode-level reporting 还没有一起收口。

#### `4`

actor 继续直接向 learner 发：

- `send-stats`
- `actor-progress`

相比 `3` 的进步在于：

- learner 不只知道 episode stats
- 还知道 actor 报告的 env progress 和 actor_done 语义

但 reporting 责任仍然在 actor。

#### `5`

`5` 则进一步把 episode-end reporting 也纳入 processor 的 flush 语义：

- actor 不再在 episode 结束后直接请求 learner
- actor 改成调用 `mark_episode_end(...)`
- processor 在确认对应 chunk 处理完成后，才把 `actor-progress` 和 `send-stats` 发给 learner

这是一条非常漂亮的边界：

- actor 只负责声明“这个 episode 到这里结束，并附上相关 marker”
- processor 负责决定“什么时候这件事在数据语义上算真正完成”

从一致性角度看，`5` 明显优于 `4`。

### 3.6 processor 通信实现的成熟度

#### `3_split_proto`

这版是最典型的 prototype 风格：

- script 内自己定义 `_resolve_processor_transport_cfg(...)`
- 自己拼 payload
- 自己写 `_ReqRepClient` / `_ReqRepServer` 相关流程
- 自己维护 queue、condition、accepting_submissions、stop_requested

代码当然能跑，但维护成本高，而且重复逻辑多。

#### `4_split_refined`

`4` 的一大进步，是开始承认“processor 是一个长期 runtime 概念，而不是这次试验脚本里的局部 hack”：

- `ProcessorTransportConfig` 进入 typed config
- `parse_train_cfg_allow_processor(...)` 和 `get_runtime_role(...)` 进入 `config.py`
- processor 的 control runtime 下沉到 `serl_launcher.rollout`

这一步的意义非常大，因为它把 split 架构从“脚本级试验”变成了“repo 里承认的一种 runtime 结构”。

#### `5_split_pipeline`

`5` 在 `4` 的基础上进一步做了两件事：

1. 把 processor 的 chunk 处理过程拆成显式 pipeline stage
2. 把 actor -> processor 的发送也抽成 `QueuedProcessorSubmitter`

这意味着：

- processor 处理链的阶段化更清晰
- actor 可以异步交付 chunk，不必在每次 processor control request 上同步阻塞

代价是：

- 系统更强大
- 但也更复杂，需要更清楚的监控和背压控制

### 3.7 config 与 runtime 抽象是怎么收口的

#### `1_baseline`

- `runtime.role` 只有 `actor | learner`
- 配置是当前 canonical baseline 配置 `train_residual.yaml`

#### `2_chunk_local`

- 仍然只有 `actor | learner`
- 但默认切到 `train_residual_optimized.yaml`
- `trainer_transport` 默认为 `async_commit`

这里已经能看出 repo 的态度：

- `1` 是参考线
- `2` 是优化实验线

#### `3_split_proto`

- processor 角色还不是 typed config 里的 first-class role
- 脚本里自己写 `_parse_train_cfg_allow_processor(...)`

这说明 `3` 虽然功能上已经 split，但配置系统还没真正把它接纳进去。

#### `4/5`

- `get_runtime_role(...)` 和 `parse_train_cfg_allow_processor(...)` 已进入 `examples/libero/config.py`
- `RuntimeConfig` 里已有 `processor_transport`

这说明 split 线的配置边界已经明显成熟。

不过也要注意一个现实：

- `RuntimeRole` 类型本身仍然是 `Literal["actor", "learner"]`
- `processor` 仍然是通过 allow-processor 兼容路径接入

也就是说，split 虽然已经相当成熟，但还没有完全收口到“typed config 的正式主干角色”。

### 3.8 backfill policy 的定位变化

#### `2`

`2` 的 backfill policy 是一个很实用的优化：

- 默认同步
- 可选开启 async backfill
- 还可以额外起 dedicated backfill policy server

这是“低侵入高收益”的典型做法。

#### `3` 和 `4`

processor 端可以：

- 如果 `backfill_policy.enabled=true`，使用 dedicated backfill backend
- 否则继续用主 policy backend

也就是说，这两版还保留了比较宽松的部署策略。

#### `5`

`5` 明确要求 processor 侧使用 dedicated backfill policy backend。

我理解这背后的意图是对的：

- 避免 processor 的 backfill 推理与 actor 的主 decision 推理抢同一个服务
- 让 split 架构的职责边界更清楚

但它也明显提高了部署门槛：

- 必须再起一个 backfill policy 服务
- 这个服务最好和主 decision 服务 checkpoint 对齐

所以这是一个典型的“架构更整洁，但日常试验成本更高”的 tradeoff。

### 3.9 learner 侧真正变化了什么

五个版本里，learner 的训练更新主干其实变化不大：

- 都是 residual DRQ-SAC
- 都是从 replay sample batch
- 都支持 offline replay 混合
- 都有 async eval、checkpoint、wandb logging

真正变化更明显的是 learner 的“外部世界感知”：

#### `1`

- 主要通过 `send-stats` 感知 env progress
- 缺少显式的“actor_done 且所有在线数据 committed 完成”退出语义

#### `2`

- 引入了 `_should_stop_after_actor_done(...)`
- learner 会结合 replay、transport accepted/committed 状态来判断是否应该正常收尾

#### `3`

- 延续了这套 stop 语义
- 但 actor 仍然直接发 stats，processor 只负责 replay data path

#### `4`

- learner 开始处理 `actor-progress`
- stop 条件从“env_steps 达到上限”升级成“actor 明确报告 done，并且对应在线数据已 committed”

#### `5`

- learner 仍消费 `actor-progress` 和 `send-stats`
- 但这两类消息现在由 processor 在正确 flush 时机转发

所以从 learner 一致性角度看：

`5 > 4 > 3 > 2 > 1`

但这只是“分布式运行时语义一致性”的排序，不是“默认日常实验入口”的排序。

## 4. 每个版本分别在解决什么问题

### 4.1 `1_baseline`：把语义讲清楚

`1_baseline` 的价值不是快，而是清楚。

它最适合回答：

- 当前 reference 训练语义到底是什么
- 一步 transition 是怎么形成的
- actor 和 learner 的基础接口是什么

我认可它的地方：

- 逻辑直接
- 调试友好
- 对照价值很高

我不满意的地方：

- actor 热路径太重
- step 内联构造 `next_residual_obs` 成本高
- transport 与 graceful shutdown 语义不够现代

结论：

- 这是最好的 baseline
- 不是最好的生产实验入口

### 4.2 `2_chunk_local`：把最值钱的优化落地

我认为 `2_chunk_local` 是五版里“投入产出比最高”的一版。

它真正做到的是：

1. 把环境执行从 step 切到 chunk
2. 把 transition 组装从逐 step 热路径移到 post-hoc assembly
3. 让 assembly 支持 batch-aware backfill
4. 保持 replay 语义基本不变
5. 接上更合理的 `async_commit` transport 和 stop 语义

它的优点几乎都很实在：

- actor 更轻
- learner 基本不需要为此重写
- 整体系统复杂度没有爆炸
- 如果要进一步优化，还可以开 async backfill 和 dedicated backfill server

它的局限也很清楚：

- transition assembly 仍在 actor 所在进程
- 还没有把控制推进和后处理彻底解耦成独立 runtime

但正因为它没有把系统拆得太碎，所以它最适合日常跑实验。

这是我最认可的一版。

### 4.3 `3_split_proto`：证明 split 是可行的

`3` 的核心贡献不是“已经做完”，而是“证明这条路走得通”。

它完成了几个很重要的验证：

- actor 可以只负责 rollout chunk 提交
- processor 可以独立重建 decision obs 和 step transition
- replay 提交与 actor 控制推进可以拆开
- split 架构在 LIBERO residual 训练线上是可实现的

但 `3` 的原型特征也很明显：

- 太多协议和状态机逻辑直接写在脚本里
- 配置系统还没有真正容纳 processor 角色
- 一些 transport/runtime 抽象仍是脚本私有

我的评价是：

- 我认可它作为原型
- 我不认可把它当长期版本停下来

### 4.4 `4_split_refined`：把原型整理成可维护结构

`4` 的价值是工程整理，而不是概念创新。

它做对了三件事：

1. 把 processor runtime 下沉到公共模块
2. 把 trainer client 的失败重试、flush、status 收口到 `TrainerClientSession`
3. 把 `actor-progress` 也纳入 learner 可消费的协议

我认可这版的地方：

- 明显比 `3` 更整洁
- 许多易错的重复逻辑被抽象掉了
- split 架构终于像一个可以维护的系统

我保留意见的地方：

- episode-end reporting 责任仍然在 actor
- actor 直接发 stats/progress，processor 只管 replay data path
- 边界比 `3` 清楚很多，但还没有到最顺的状态

所以 `4` 是一版很好的过渡整理版，但仍然不是我心里的最终形态。

### 4.5 `5_split_pipeline`：把 split 的边界真正讲顺

`5` 是 split 线里我最认可的一版。

它相对 `4` 最有价值的改进，不是“多加了一个模块”，而是把三件事串顺了：

1. actor 用 `QueuedProcessorSubmitter` 异步交付 chunk
2. processor 用显式 pipeline 处理 normalize -> reconstruct -> assemble
3. processor 在 flush episode marker 时再转发 `actor-progress` 和 `send-stats`

这样一来：

- actor 更像控制进程
- processor 更像 rollout-to-replay 转换进程
- learner 更像纯训练进程

我很认可这版的地方：

- 角色职责最清楚
- episode-end 语义最一致
- pipeline 形式更利于后续继续插 stage
- 共享抽象更干净，像是长期结构

我保留意见的地方：

- 它要求 dedicated backfill backend，部署复杂度提升
- actor 侧的 `QueuedProcessorSubmitter` 默认是无界队列，这意味着背压有可能从“同步阻塞”变成“内存积压”
- split 架构整体的运维复杂度仍高于 `2_chunk_local`

所以我的结论是：

- 如果你要继续做 split 架构演进，应该以 `5` 为起点
- 但如果你只是想稳定跑实验，我仍然不会优先让你从 `5` 开始

## 5. 演化顺序应该怎么理解

这五版最合理的阅读顺序就是名字里的数字顺序：

`1 -> 2 -> 3 -> 4 -> 5`

但这条顺序表达的是演化历史，不是推荐程度。

更准确地说：

### `1 -> 2`

这是“优化 actor 数据流”的阶段。

目标是：

- 让 actor 更快
- 把热路径清掉
- 尽量不重写 learner

这是整个演化链里最值钱的一步。

### `2 -> 3`

这是“尝试把 assembly 拆成独立 processor”的阶段。

目标是：

- 把控制推进和 replay assembly 分开
- 验证 split 架构可行

这是一次正确的原型试探。

### `3 -> 4`

这是“把原型收敛成可维护工程结构”的阶段。

目标是：

- 去掉脚本里重复的 transport/runtime 样板
- 把 processor 与 trainer session 变成可复用抽象

这是一次很健康的工程整理。

### `4 -> 5`

这是“把 split 架构边界讲清楚”的阶段。

目标是：

- 把 episode-end reporting 也纳入 processor flush 语义
- 引入 pipeline 和 queued submitter
- 让 actor/processor/learner 的边界更纯

这是一次更偏“长期架构设计”的推进。

## 6. 我是否认可这些修改

下面是我的直接判断。

### 6.1 我最认可的改动

#### 最认可：`1 -> 2`

这是五版里我最认可的一步。

原因：

- 收益大
- 改动目标清晰
- 没有过度改变训练语义
- 对日常实验最有帮助

如果只让我选一条修改线保留，我会优先保留 `2_chunk_local` 这条线。

#### 也很认可：`3 -> 4 -> 5` 的工程收敛思路

我也认可 split 线继续做下去，因为：

- 把 actor 从 transition assembly 中彻底解放出来，本身就是合理目标
- processor 作为长期 runtime 概念是成立的
- `5` 的 pipeline + marker flush 语义说明这条线不是乱拆，而是在逐步变清楚

### 6.2 我不太认可的地方

#### 不建议把 `3_split_proto` 当最终版本

原因很直接：

- 原型逻辑太多
- 脚本内协议太重
- 配置边界还没收口

它适合证明概念，不适合长期停留。

#### 不建议把 `5_split_pipeline` 当默认跑实验入口

这不是说 `5` 不好，而是说它的复杂度更高：

- 多一个 processor 进程
- 多一个 dedicated backfill backend
- actor 侧还有异步提交队列要观察

如果你只是要稳定产实验结果，这些复杂度通常得不偿失。

## 7. 我最推荐哪个版本

### 7.1 日常实验默认推荐：`2_chunk_local`

这是我的第一推荐。

适用场景：

- 你要稳定跑 LIBERO residual 训练
- 你要在不引入太多运维复杂度的前提下吃到 actor 优化收益
- 你想保留较强的 debug 能力

推荐理由：

- 训练语义变化小
- actor 热路径明显更干净
- 支持 async backfill 和 `async_commit`
- 比 split 线更容易启动、排障、复现实验

一句话概括：

它是“把最值钱的优化吃到手，但不过度架构化”的版本。

### 7.2 如果要做 baseline / debug：`1_baseline`

适用场景：

- 做对照实验
- 排查训练语义问题
- 向别人解释 reference 数据流

这版虽然慢一些，但语义最清楚。

### 7.3 如果要继续做 split 架构演进：`5_split_pipeline`

适用场景：

- 你明确要继续推进 actor / processor / learner 三段式架构
- 你关心 processor flush 与 episode reporting 的一致性
- 你准备继续在 runtime 层扩展更多处理 stage

这时不要从 `3` 或 `4` 开始，应该直接以 `5` 为基线继续演进。

这一判断也和当前 `examples/libero/README.md` 的入口定位一致：

- `1_baseline` 被保留为 reference / baseline
- `2_chunk_local` 被定位成当前更稳的优化实验线
- `5_split_pipeline` 被定位成最新 split / pipeline 演化版本，但不是默认起跑线

## 8. 如果现在要做收敛，我会怎么建议

如果后面准备继续整理这条线，我会建议按下面的思路收敛。

1. 对外默认入口继续保留：
   - baseline 对照：`1_baseline`
   - 日常优化实验：`2_chunk_local`
2. split 架构演进线只保留 `5_split_pipeline`
3. `3_split_proto` 和 `4_split_refined` 更适合作为历史演进参考，不建议再继续堆功能
4. 如果 `5` 要转正成默认线，至少还需要补两件事：
   - 把 `processor` 变成 typed config 里的正式 runtime role
   - 给 `QueuedProcessorSubmitter` 加明确的 bounded queue / 监控 / 背压策略

## 9. 最后的总结

这五个脚本不是五套互相割裂的训练算法，而是一条很清楚的 runtime/dataflow 演化链：

- `1` 负责把 reference 语义讲清楚
- `2` 负责把最重要的 actor 热路径优化落地
- `3` 负责证明 split 架构可行
- `4` 负责把原型整理成较可维护的结构
- `5` 负责把 split 架构的职责边界讲顺

如果只问“我最推荐哪个版本”，答案是：

- 默认日常实验：`run_residual_training_2_chunk_local.py`

如果只问“split 线里最成熟的是哪个版本”，答案是：

- `run_residual_training_5_split_pipeline.py`

如果只问“哪一个最适合做 reference baseline 和 debug”，答案是：

- `run_residual_training_1_baseline.py`

这也是我对这条演化线最核心的判断：

- 我认可这条演化路径
- 我最认可 `1 -> 2` 这一步
- 我认可 `3 -> 4 -> 5` 的长期架构方向
- 但我不会把“最新版本”自动等同于“默认最推荐版本”
