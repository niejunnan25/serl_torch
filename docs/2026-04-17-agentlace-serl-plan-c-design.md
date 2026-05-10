# 2026-04-17 Agentlace + SERL Plan C 设计与落地说明

## 0. 当前状态

这份文档最初写于 Plan C 还处于设计阶段时，后续实现已经在当前仓库内落地了一版 repo-local transport。

当前代码状态是：

- 训练数据/控制传输已经不再直接依赖 `agentlace.trainer.TrainerClient` / `TrainerServer`
- 当前实验训练线已经改为通过 [serl_launcher/serl_launcher/common/trainer_transport.py](/home/hello/codebase/serl_torch/serl_launcher/serl_launcher/common/trainer_transport.py) 创建 transport
- 外部 `agentlace` 现在只继续负责参数广播，不再负责 trainer data/control path
- LIBERO `optimized` 和 AgiBot `copy` 都已经接入 typed `runtime.trainer_transport`

因此，下面的很多章节应当理解为：

- 前半部分：为什么当时需要 Plan C
- 中间部分：协议与实现设计
- 后半部分：哪些设计已经实际落地，哪些仍然属于后续可继续演进的方向

## 1. 背景

当前 `serl_torch` 里的 copy 训练链路，已经通过 repo-local transport 封装了 trainer 通信；外部 `agentlace` 仅继续用于参数广播。

当前接入点主要在：

- LIBERO actor / learner:
  [examples/libero/scripts/run_residual_training_2_chunk_local.py](/home/hello/codebase/serl_torch/examples/libero/scripts/run_residual_training_2_chunk_local.py)
- AgiBot actor / learner:
  [examples/agibot_real/scripts/run_residual_training.py](/home/hello/codebase/serl_torch/examples/agibot_real/scripts/run_residual_training.py)
- transport 实现：
  [serl_launcher/serl_launcher/common/trainer_transport.py](/home/hello/codebase/serl_torch/serl_launcher/serl_launcher/common/trainer_transport.py)

当前 typed runtime config 里已经包含 `runtime.trainer_transport`：

- LIBERO `RuntimeConfig`: [examples/libero/config.py](/home/hello/codebase/serl_torch/examples/libero/config.py:30)
- AgiBot `RuntimeConfig`: [examples/agibot_real/config.py](/home/hello/codebase/serl_torch/examples/agibot_real/config.py:38)

现状问题是：

1. `client.update()`、`send-stats`、`get_last_update_id` 都挤在同一个 `req/rep` 控制通道里
2. learner 在 `req/rep` callback 内同步执行 `batch_insert`
3. replay insert 与 learner sample/train 争同一个 replay 锁
4. 大 payload 的 datastore update 会把小控制请求一起堵住

`agentlace` 当前 server 侧的关键路径是：

- `datastore` 在 callback 内直接 `batch_insert`: [trainer.py](/vla/miniconda3/envs/serl_torch/lib/python3.10/site-packages/agentlace/trainer.py:86)
- `get_last_update_id` 只有一个 last-update 语义: [trainer.py](/vla/miniconda3/envs/serl_torch/lib/python3.10/site-packages/agentlace/trainer.py:103)
- client `update()` 先查 last update id，再发 datastore: [trainer.py](/vla/miniconda3/envs/serl_torch/lib/python3.10/site-packages/agentlace/trainer.py:274)
- `request("send-stats", ...)` 仍然走同一个 req/rep socket: [trainer.py](/vla/miniconda3/envs/serl_torch/lib/python3.10/site-packages/agentlace/trainer.py:358)

这就是之前看到 socket timeout 和 `send-stats` 被连坐的根因。

## 2. Plan C 是什么

Plan C 不是单纯“换个 socket”，而是四件事一起做：

1. 控制面继续走 `req/rep`
2. 数据面改成独立 `push/pull`
3. learner 收到数据后先进入 bounded queue，再由 worker 线程真正写 replay
4. 协议上显式区分 `accepted` 与 `committed`

一句话概括：

> 让控制请求永远轻量，让大数据单独走，让 replay 提交异步化，并把“已收到”与“已提交”这两种状态从协议上分开。

目标结构如下：

```text
actor
  |-- control req/rep --> learner control port
  |                      get_accepted_update_id / get_committed_update_id
  |                      send-stats / checkpoint / health
  |
  |-- data push -------> learner data port
                         replay transition batch

learner
  |-- control thread
  |     只处理小请求，快速返回
  |
  |-- data receiver thread
  |     收到 data batch 后放入 bounded queue
  |
  |-- replay commit worker
        从 queue 取 batch，真正 batch_insert 到 replay
        写完以后更新 committed_update_id
```

## 3. 为什么 Plan C 比 Plan A / Plan B 更完整

### 3.1 Plan A

Plan A 是：

- control 走 `req/rep`
- data 走 `push/pull`
- data receiver 收到后立刻 insert replay

优点：

- 控制请求不再被 data 包直接堵住

问题：

- data receiver 线程自己仍会被 replay insert 卡住
- 没有显式 accepted / committed 双语义

### 3.2 Plan B

Plan B 是：

- 仍然只有一条 `req/rep`
- 只是把 `datastore` callback 改成 enqueue-and-ack

优点：

- 改动小
- `update()` 很快就能返回

问题：

- 控制和数据仍然共用同一个入口
- 大 payload 仍会和小控制请求共享同一通道

### 3.3 Plan C

Plan C 同时吸收了两者的好处：

- 控制和数据物理拆通道
- replay insert 异步化
- 队列 bounded，可形成真实背压
- 协议层区分 accepted / committed

因此 Plan C 是更完整、更适合长期在线 RL 的形态。

## 4. 协议语义：accepted 与 committed

Plan C 里必须显式区分两个 update id：

### 4.1 `accepted_update_id`

含义：

- learner 已经收到并接管了这批数据
- actor 之后不用重复发送这一批

它只代表“已被系统接收”，不代表“已经可被 replay sample 看到”。

### 4.2 `committed_update_id`

含义：

- worker 已经把这批数据真正写进 replay
- learner 现在理论上可以从 replay 采到它

### 4.3 为什么一定要两个值

如果只有一个值，就会出现语义混乱：

1. actor 发出 batch
2. learner 说“收到了”
3. 但 replay 还没 insert 完

这时如果 actor 误以为“收到 == 提交完成”，就会：

- 统计上认为 replay 已跟上
- 实际 learner 采样仍看不到最新数据

### 4.4 生产实现不建议复用 `get_last_update_id`

这次 benchmark 原型里，为了快速做实验，`get_last_update_id` 在不同 scenario 下会返回 accepted 或 committed 中的一种。

但真正落生产时，不建议延续这个做法。

更稳妥的协议应该是显式新增：

- `get_accepted_update_id`
- `get_committed_update_id`

而不是让 `get_last_update_id` 在不同模式下改变语义。

原因很简单：

- 老代码和日志容易误解
- 回归时不容易判断到底看的是哪一个状态
- 后续做监控和报警时会混乱

## 5. 推荐的生产改造范围

### 5.1 当前实现策略

当前实现已经采用了“仓库内 transport wrapper + 外部 agentlace 仅保留广播”的路线，而不是直接手改 site-packages。

也就是说：

- trainer control/data path 在本仓库内实现
- 参数广播继续复用外部 `agentlace`
- copy 训练脚本统一通过本地 factory 创建 actor / learner transport

这条路线已经足够支撑当前 rollout；是否后续再把协议层抽回可控的 `agentlace` fork，是单独的维护决策，不影响这版实现本身。

## 6. 具体改造清单

下面按“要改哪些类、哪些字段、actor/learner 各自怎么变”来列。

---

## 7. 第一层：agentlace 协议与传输层

### 7.1 `TrainerConfig` 要新增的字段

当前 `TrainerConfig` 只有：

- `port_number`
- `broadcast_port`
- `request_types`
- `rate_limit`
- `version`
- `experimental_pipeline_port`

建议新增：

```python
@dataclass
class TrainerConfig:
    port_number: int = 5555
    broadcast_port: int = 5556
    request_types: list[str] = field(default_factory=list)
    rate_limit: int | None = None
    version: str = "0.0.3"

    transport_mode: Literal["legacy_reqrep", "split_queue"] = "legacy_reqrep"
    data_port: int | None = None
    control_timeout_ms: int = 800
    data_queue_capacity: int = 8
    data_socket_hwm: int = 8
    commit_poll_ms: float = 5.0
```

字段含义：

- `transport_mode`
  当前是旧模式还是 Plan C 模式
- `data_port`
  新的数据面端口
- `control_timeout_ms`
  control req/rep 超时
- `data_queue_capacity`
  learner 侧接收队列上限
- `data_socket_hwm`
  ZMQ data socket 的 high-water mark
- `commit_poll_ms`
  actor 等待 committed 时的轮询周期

### 7.2 `TrainerServer` 要新增的状态

当前 server 里只有：

- `data_stores`
- `last_update_id_map`

Plan C 需要新增：

```python
self.accepted_update_id_map: dict[str, int]
self.committed_update_id_map: dict[str, int]
self.data_queue_map: dict[str, queue.Queue]
self.data_receiver_thread: threading.Thread | None
self.commit_worker_threads: dict[str, threading.Thread]
```

另外还要增加一个 data-side callback 路径，不再只靠 control callback 处理 datastore。

### 7.3 `TrainerServer` 要新增的 control 请求

当前有：

- `hash`
- `get_last_update_id`
- 自定义 request

建议新增：

- `get_accepted_update_id`
- `get_committed_update_id`
- `get_transport_status`

其中 `get_transport_status` 返回：

```python
{
  "accepted_update_id": ...,
  "committed_update_id": ...,
  "queue_depth": ...,
  "queue_capacity": ...,
}
```

这样后面调试 backlog 会轻松很多。

### 7.4 `TrainerClient` 要怎么改

当前 `TrainerClient.update()` 是：

1. `get_server_last_update_id`
2. `get_latest_data(from_id)`
3. `req_rep_client.send_msg(datastore)`

见 [trainer.py](/vla/miniconda3/envs/serl_torch/lib/python3.10/site-packages/agentlace/trainer.py:274)。

Plan C 下建议拆成：

```python
class TrainerClient:
    def get_server_accepted_update_id(self, name: str) -> int | None: ...
    def get_server_committed_update_id(self, name: str) -> int | None: ...
    def wait_until_committed(self, name: str, target_id: int, timeout_s: float | None) -> bool: ...
```

然后 `update()` 的语义改成：

1. 问 `get_server_accepted_update_id`
2. 从本地 datastore 取增量
3. 将数据通过 data socket 发到 learner
4. 立即返回成功 / 失败
5. 不同步等待 replay insert 完成

### 7.5 data message 建议字段

建议 data plane 里的消息显式带这些字段：

```python
{
  "type": "datastore",
  "store_name": "actor_env",
  "first_id": 1200,
  "last_id": 1229,
  "batch_count": 30,
  "payload_kind": "packed",
  "payload": ...
}
```

原因：

- learner 可以更清楚地做顺序和日志检查
- 将来如果要做去重、断点恢复、稀疏重发，更容易

### 7.6 queue + worker 的行为

Plan C 的关键不是“收到后立刻 insert”，而是：

1. data receiver 收到消息
2. 更新 `accepted_update_id`
3. 放入 bounded queue
4. worker 从 queue 取出 payload
5. 真正 `batch_insert`
6. 更新 `committed_update_id`

注意：

- queue 必须是 bounded
- queue 满时，data receiver 不能无限吞
- 最终要形成自然背压

### 7.7 replay insert 本身建议同步优化

Plan C 最好和“真正的 vectorized batch insert”一起做。

否则即便通道拆开了，worker 线程里仍然会花太多时间逐条 insert。

所以推荐：

1. 优先给 `ReplayBufferDataStore` / `MemoryEfficientStepWindowReplayBufferDataStore` 增加 batch-aware insert
2. 不要只依赖 `DataStoreBase.batch_insert()` 的逐条循环

---

## 8. 第二层：serl_torch typed config 怎么改

### 8.1 当前 config 状态

这一部分原本是在说明“为什么需要新增 transport typed config”。当前这件事已经完成。

现在 runtime 配置除了：

- `trainer_host`
- `trainer_port`
- `broadcast_port`
- `data_store_queue_size`

之外，还已经包含：

- `trainer_transport.mode`
- `trainer_transport.data_port`
- `trainer_transport.control_timeout_ms`
- `trainer_transport.data_queue_capacity`
- `trainer_transport.data_socket_hwm`
- `trainer_transport.commit_poll_ms`
- `trainer_transport.wait_committed_on_episode_end`
- `trainer_transport.wait_committed_on_shutdown`

### 8.2 `TrainerTransportConfig`

当前 LIBERO 和 AgiBot 两套 config 都已经增加：

```python
@dataclass(frozen=True, slots=True)
class TrainerTransportConfig:
    mode: Literal["legacy_reqrep", "split_queue"]
    data_port: int | None
    control_timeout_ms: int
    data_queue_capacity: int
    commit_poll_ms: float
    wait_committed_on_episode_end: bool
    wait_committed_on_shutdown: bool
```

当前 `RuntimeConfig` 也已经包含：

```python
@dataclass(frozen=True, slots=True)
class RuntimeConfig:
    role: RuntimeRole
    trainer_host: str
    trainer_port: int
    broadcast_port: int
    data_store_queue_size: int
    trainer_transport: TrainerTransportConfig
```

### 8.3 yaml 状态

当前 canonical yaml 和 copy 专用 yaml 都已经显式声明了 `runtime.trainer_transport`。

当前约定是：

- canonical `train_residual*.yaml` 默认 `mode: legacy_reqrep`
- `LIBERO optimized` 和当前 AgiBot mainline yaml 默认 `mode: async_commit`

对应文件分别是：

- LIBERO optimized:
  [examples/libero/configs/train_residual_optimized.yaml](/home/hello/codebase/serl_torch/examples/libero/configs/train_residual_optimized.yaml)
- AgiBot mainline:
  [examples/agibot_real/configs/train_residual.yaml](/home/hello/codebase/serl_torch/examples/agibot_real/configs/train_residual.yaml)

下面保留原始建议字段，作为设计记录。

例如：

```yaml
runtime:
  role: actor
  trainer_host: 127.0.0.1
  trainer_port: 5688
  broadcast_port: 5689
  data_store_queue_size: 2000
  trainer_transport:
    mode: split_queue
    data_port: 5690
    control_timeout_ms: 3000
    data_queue_capacity: 8
    commit_poll_ms: 5.0
    wait_committed_on_episode_end: false
    wait_committed_on_shutdown: true
```

对于 AgiBot 也类似：

```yaml
runtime:
  trainer_transport:
    mode: split_queue
    data_port: 5490
    control_timeout_ms: 3000
    data_queue_capacity: 8
    commit_poll_ms: 5.0
    wait_committed_on_episode_end: false
    wait_committed_on_shutdown: true
```

### 8.4 默认值建议

建议默认仍然保守：

- `mode: legacy_reqrep`
- `data_port: null`
- `wait_committed_on_episode_end: false`
- `wait_committed_on_shutdown: true`

这样不会默默改变现有训练语义。

---

## 9. 第三层：在 serl_launcher 里加本地 wrapper

### 9.1 为什么需要 wrapper

现在脚本里到处直接 `TrainerClient(...)` / `TrainerServer(...)`。

如果以后每个脚本都自己拼 transport 细节，会很乱。

建议新增一个本地模块，例如：

- `serl_launcher/common/trainer_transport.py`

提供统一工厂：

```python
def build_actor_trainer_client(cfg, data_store) -> ActorTrainerTransport: ...
def build_learner_trainer_server(cfg, replay_buffer, request_callback) -> LearnerTrainerTransport: ...
```

### 9.2 wrapper 暴露的最小接口

actor 侧 wrapper 建议暴露：

```python
class ActorTrainerTransport(Protocol):
    def update(self) -> bool: ...
    def request(self, type: str, payload: dict) -> dict | None: ...
    def recv_network_callback(self, callback) -> None: ...
    def wait_until_committed(self, timeout_s: float | None = None) -> bool: ...
    def stop(self) -> None: ...
```

learner 侧 wrapper 建议暴露：

```python
class LearnerTrainerTransport(Protocol):
    def register_data_store(self, name: str, data_store) -> None: ...
    def publish_network(self, payload: dict) -> None: ...
    def start(self, threaded: bool = True) -> None: ...
    def stop(self) -> None: ...
```

### 9.3 为什么要这样做

好处：

1. 训练脚本不需要知道 Plan C 的 socket / queue 细节
2. legacy 与 split_queue 可以通过 config 切换
3. rollout 风险更低

---

## 10. 第四层：actor 代码具体怎么改

### 10.1 受影响的 actor 脚本

第一批建议改这些：

- [examples/libero/scripts/run_residual_training_2_chunk_local.py](/home/hello/codebase/serl_torch/examples/libero/scripts/run_residual_training_2_chunk_local.py:472)
- [examples/agibot_real/scripts/run_residual_training.py](/home/hello/codebase/serl_torch/examples/agibot_real/scripts/run_residual_training.py:415)

第二批再考虑：

- `run_residual_training_1_baseline.py`
- 其他 eval / prepare 相关训练辅助脚本

### 10.2 actor 的 `update()` 语义怎么变

当前 actor 调用 `client.update()` 的位置很多，例如：

- LIBERO optimized: [examples/libero/scripts/run_residual_training_2_chunk_local.py](/home/hello/codebase/serl_torch/examples/libero/scripts/run_residual_training_2_chunk_local.py:535)
- AgiBot copy: [examples/agibot_real/scripts/run_residual_training.py](/home/hello/codebase/serl_torch/examples/agibot_real/scripts/run_residual_training.py:540)

Plan C 下，这些调用点原则上可以不改调用形状，但要改语义：

- 以前：`update()` 返回时，通常隐含“server 那边已经处理完 datastore callback”
- 以后：`update()` 返回时，只表示“数据已成功发往 learner，并已被 accepted”

也就是说，`update()` 从“同步提交”变成“快速投递”。

### 10.3 actor episode 边界要怎么处理

建议区分两个阶段：

#### 常规 episode 结束

默认不强制等 committed。

流程建议：

1. `client.update()`
2. `client.request("send-stats", episode_stats)`
3. 直接进入 reset 或下一轮 episode 准备

这样可以最大化重叠 reset / commit。

#### shutdown / summary / 最终 checkpoint

要显式等 committed。

流程建议：

1. `client.update()`
2. `client.wait_until_committed(timeout_s=...)`
3. 再写最终 summary / 退出

### 10.4 AgiBot copy 这条线的特别收益

对于 AgiBot：

- 当前 `run_residual_training.py` 本地还有 async backfill 与 `commit_replay` 的阶段
- 这些阶段之后再把 transition 从 actor 侧本地 `QueuedDataStore` 发到 learner

Plan C 对它的直接收益是：

1. `client.update()` 不再容易卡 actor 控制线程
2. episode 结束后更容易把 reset 时间和 learner replay commit 重叠
3. `send-stats` 不会再被 datastore 大包连坐

---

## 11. 第五层：learner 代码具体怎么改

### 11.1 learner 脚本改动原则

learner 训练主循环本身应尽量少改。

应该把改动集中在：

- trainer server 构建
- replay ingest
- observability

受影响的 learner 接入点：

- LIBERO learner: [examples/libero/scripts/run_residual_training_2_chunk_local.py](/home/hello/codebase/serl_torch/examples/libero/scripts/run_residual_training_2_chunk_local.py:910)
- AgiBot learner: [examples/agibot_real/scripts/run_residual_training.py](/home/hello/codebase/serl_torch/examples/agibot_real/scripts/run_residual_training.py:954)

### 11.2 learner 的 replay 可见性语义

learner 的训练逻辑仍然只依赖 replay 中真实已提交的数据。

所以：

- replay size
- sample iterator
- update loop

都只应该看 committed 后的数据。

accepted 只用于“actor 该不该重发数据”的通信语义，不参与训练采样语义。

### 11.3 learner heartbeat / logging 建议新增字段

建议在 learner heartbeat / logs 中额外记录：

- `accepted_update_id`
- `committed_update_id`
- `transport_backlog = accepted - committed`
- `data_queue_depth`

这样你以后看到“actor env_steps 已经 1000，learner replay_size 还没跟上”时，就能直接知道是 backlog 还是别的地方慢。

---

## 12. 第六层：replay insert 本身怎么改

Plan C 最好和 vectorized insert 一起做。

### 12.1 当前问题

目前 `DataStoreBase.batch_insert()` 默认是逐条循环。

对于图像 observation：

- Python 循环多
- 递归拷贝多
- 每条 transition 都拿锁

### 12.2 建议改法

给这些 datastore 增加真正的 batch-aware insert：

- `ReplayBufferDataStore`
- `MemoryEfficientReplayBufferDataStore`
- `StepWindowReplayBufferDataStore`
- `MemoryEfficientStepWindowReplayBufferDataStore`

实现方向：

1. 把 `list[transition]` 转成 packed batch dict
2. 一次拿锁
3. 一次切片写入 replay buffer
4. 一次更新 insert index / size / metadata

### 12.3 为什么 Plan C 仍然需要它

即便用了 queue worker：

- 如果 worker 自己仍然逐条 insert
- 那 committed 仍然会慢

Plan C 只是把慢操作从 control thread 挪开，不会凭空让 insert 变快。

---

## 13. 分阶段落地状态

### Phase 0：replay vectorized batch insert

目标：

- 不动协议
- 先降低真实 replay commit 成本

当前状态：

- 已完成
- 当前仓库里的 replay datastore 已经接入 batch-aware insert
- `MemoryEfficientReplayBufferDataStore` 仍然不是最彻底的整块切片写入，这部分仍可继续优化

### Phase 1：加 wrapper，但默认仍走 legacy

目标：

- 在 `serl_launcher` 里加统一 transport wrapper
- 不改现有训练脚本调用形状

当前状态：

- 已完成
- 当前实现位于 [serl_launcher/serl_launcher/common/trainer_transport.py](/home/hello/codebase/serl_torch/serl_launcher/serl_launcher/common/trainer_transport.py)

### Phase 2：repo-local split_queue transport

目标：

- control req/rep
- data push/pull
- bounded queue
- accepted / committed 双状态

当前状态：

- 已完成
- 当前实现没有去改外部 `agentlace`，而是在仓库内自管 trainer transport

### Phase 3：先在 LIBERO optimized 线上灰度

原因：

- 仿真验证成本低
- 端到端 smoke 容易跑

当前状态：

- 已完成代码接线
- benchmark / smoke 已覆盖
- 完整外部依赖链路的长跑 e2e 仍可继续补

### Phase 4：再接 AgiBot copy

原因：

- 真机 reset 与 replay commit 的重叠收益更明显
- 但真机 rollout 风险更高，应该放在仿真验证之后

当前状态：

- 已完成代码接线
- 仓库内静态验证、单测和 benchmark 已做
- 真机环境验收仍属于线下 manual checklist 范围

---

## 14. 需要补的测试

### 14.1 单元测试

1. `get_accepted_update_id` 与 `get_committed_update_id` 单调递增
2. data queue 满时不会静默丢包
3. worker commit 后 committed id 正确推进
4. actor `wait_until_committed()` 在目标 id 达成后返回成功

### 14.2 集成测试

1. actor 连续 update，learner replay 最终 size 正确
2. `send-stats` 在大数据包期间不被明显拖慢
3. shutdown 时 wait-for-committed 能正确 drain

### 14.3 benchmark 验收

建议继续复用：

- [test/benchmark_trainer_datastore_variants.py](/home/hello/codebase/serl_torch/test/benchmark_trainer_datastore_variants.py:1)

至少比较：

- legacy req/rep
- vectorized insert only
- Plan C split_queue

---

## 15. 这套改造实际落到代码上的最小变更列表

下面是最务实的一版“直接开改时的变更清单”。

### 15.1 trainer transport 层

当前实际改法不是直接修改外部 `agentlace`，而是：

- 新增 [serl_launcher/serl_launcher/common/trainer_transport.py](/home/hello/codebase/serl_torch/serl_launcher/serl_launcher/common/trainer_transport.py)
- 在这个文件里实现：
  - `LegacyReqRepTransport`
  - `SplitQueueTransport`
  - `accepted_update_id`
  - `committed_update_id`
  - transport status / wait-until-committed / lifecycle cleanup

### 15.2 serl_torch config 层

已改：

- `examples/libero/config.py`
- `examples/agibot_real/config.py`
- 对应 train yaml

### 15.3 serl_torch wrapper 层

已新增：

- `serl_launcher/common/trainer_transport.py`

### 15.4 脚本接入层

第一批，已接入：

- `examples/libero/scripts/run_residual_training_2_chunk_local.py`
- `examples/agibot_real/scripts/run_residual_training.py`

第二批，当前仍保留 legacy：

- `examples/libero/scripts/run_residual_training_1_baseline.py`
- `examples/agibot_real/scripts/run_residual_training.py`

---

## 16. 当前实现后的剩余重点

如果目标是：

- 尽快减少 timeout
- 保住 `send-stats` 的及时性
- 后面还能继续往更高吞吐走

当前主路径已经完成。后续如果继续推进，我推荐优先看这几件事：

1. 继续优化 `MemoryEfficientReplayBufferDataStore.batch_insert()`
2. 补完整的外部依赖 e2e 长跑验证
3. 视需要再把 canonical 非 copy 训练线也切到统一 transport

原因：

- 这条路径风险最可控
- 结构最干净
- 后续要做更强的 observability、backpressure、recoverability 也都更顺

## 17. 一句话结论

Plan C 不是“把 req/rep 改成 pipeline”这么简单。

它真正要做的是：

> 控制面和数据面拆通道，数据进入 bounded queue，worker 异步提交 replay，并在协议层明确区分 accepted 与 committed。

这才是把当前 `agentlace + serl_torch` 训练通信层从“能跑”提升到“适合长期在线 RL”的关键一步。
