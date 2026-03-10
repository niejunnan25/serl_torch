# SERL 从 JAX 迁移到 PyTorch：AI 执行指令

请按以下要求将项目 `/Users/niejunnan.25/Documents/codebase/serl` 从 **JAX/Flax** 栈迁移到 **PyTorch** 栈，保持算法逻辑、API 风格和目录结构一致，使原有示例与脚本在最小改动下可运行。

---

## 一、项目背景与范围

- **项目**：SERL（Sample-Efficient Robotic RL），用于机械臂操作的样本高效强化学习。
- **当前栈**：JAX、Flax、Optax、Distrax、Chex、Flax/Orbax checkpoints。
- **目标栈**：PyTorch、torch.nn、torch.optim（或可选 AdamW + 自定义 schedule）、PyTorch 分布（或 `torch.distributions`），PyTorch 原生 checkpoint。
- **迁移范围**：
  - **核心库**：`serl_launcher/` 下所有与 JAX/Flax 相关的代码（agents、networks、common、data、vision、utils、wrappers）。
  - **示例与入口**：`examples/` 下所有训练/评估脚本（如 `async_sac_state_sim`、各 `async_*_drq`、`bc_policy.py` 等）以及 `serl_launcher/utils/launcher.py` 中的 `make_*_agent`。
  - **不迁移**：`serl_robot_infra`、`franka_sim` 中与机器人/仿真环境接口相关的部分可保持 Python/NumPy；仅当其中直接调用 JAX（如 `jnp`/`jax`）时才改为 NumPy 或 PyTorch（例如推理时用 `torch` 张量并转为 numpy 与 env 交互）。

---

## 二、依赖与映射表

- **移除或替换**：`jax`、`jaxlib`、`flax`、`optax`、`distrax`、`chex`、`orbax-checkpoint`；若存在 `flax.training.checkpoints`，改为 PyTorch 保存/加载。
- **新增**：`torch`（>=2.0）、`numpy`；如需分布与 bijector，用 `torch.distributions`（及 `torch.distributions.transforms`）替代 distrax。
- **保留**：`gym`、`numpy`、`tqdm`、`wandb`、`absl-py`、`einops`、`imageio`、`moviepy`、`tensorflow`/`tf_keras`/`tensorflow_datasets` 等仅在不影响核心训练时保留；若某处仅用 TF 做数据或 I/O，可保留或后续改为 PyTorch DataLoader。
- **requirements.txt**：更新为以 `torch` 为核心的依赖列表，并注明 CUDA 版本建议（如 `torch` 与 `torchvision` 的安装方式）。

---

## 三、核心迁移规则（按模块）

### 3.1 张量与设备

- `jax.Array` / `jnp.ndarray` → `torch.Tensor`。
- `jnp.*` 逐函数替换：如 `jnp.concatenate`→`torch.cat`，`jnp.exp`→`torch.exp`，`jnp.clip`→`torch.clamp`，`jnp.sqrt`→`torch.sqrt`，`jnp.squeeze`→`squeeze`，`jnp.mean`→`torch.mean`，形状操作用 `reshape`/`view`/`permute` 等。
- 设备：用 `device = torch.device("cuda" if torch.cuda.is_available() else "cpu")`，模型与 batch 统一 `.to(device)`；数据加载时在 DataLoader 或迭代里将 batch 移到 device。
- 与 NumPy 互转：`.cpu().numpy()` 与 `torch.from_numpy(...).to(device)`，保持与 gym/env 接口一致（通常 env 用 numpy）。

### 3.2 模型定义（Flax → PyTorch）

- **Flax `nn.Module`**（`@nn.compact` / `setup`）→ `torch.nn.Module`，在 `__init__` 里定义子模块和参数，在 `forward(self, x, ...)` 里写前向。
- **参数**：Flax 的 `self.param("name", init_fn, shape)` → `nn.Parameter(...)` 或 `nn.Linear` 等；初始化在 `__init__` 或 `reset_parameters()` 中完成。
- **命名**：保持与现有逻辑对应的命名（如 `encoder`、`network`、`actor`、`critic`），便于对读 checkpoint 或下游脚本。
- **多模块共用一个 encoder**：Flax 的 `ModuleDict` 式共享 → 在 PyTorch 里同一个 `nn.Module` 实例被多个头（actor/critic）引用即可。
- **Ensemble**：Flax 的 `nn.vmap`/`ensemblize` → PyTorch 里用 `nn.ModuleList` 存多份 `Critic`，前向时对每个网络分别计算再 stack/mean。

### 3.3 优化器与学习率

- **Optax**：`optax.adam`、`optax.adamw`、`optax.chain`、`optax.clip_by_global_norm` → `torch.optim.Adam`/`AdamW` 与 `torch.nn.utils.clip_grad_norm_`。
- **Schedule**：`optax.warmup_cosine_decay_schedule`、`optax.linear_schedule`、`optax.join_schedules` → 用 `torch.optim.lr_scheduler` 或自定义 `scheduler.step()` 的 schedule；若当前是 per-step 的 schedule，在 trainer 里每 step 调用 `scheduler.step()` 并可选地 `optimizer.param_groups[0]["lr"] = scheduler.get_last_lr()[0]`。
- **common/optimizers.py**：重写为返回 PyTorch optimizer 和（可选）scheduler 的工厂函数，接口可与现有 `make_optimizer` 的语义对齐（warmup、cosine decay、weight decay、grad clip）。

### 3.4 训练状态与梯度

- **JaxRLTrainState**（`common/common.py`）：改为 PyTorch 风格的状态对象或简单 dataclass，包含：`model`（`nn.Module`）、`optimizer`、`scheduler`（可选）、`step`、`target_params`（或 target 网络）、`rng` 的替代（见下）。
- **apply_fn / params**：不再用 `apply_fn(params, x)`；改为 `model(x)`，梯度与更新用 `loss.backward()` + `optimizer.step()`；若需要“用另一套参数做前向”（如 target network），保留 `target_*` 的拷贝并在 update 时用 `tau` 软更新。
- **target 更新**：实现与现有逻辑一致的 polyak：`θ_target = τ * θ + (1-τ) * θ_target`，逐参数或使用 `copy.deepcopy`/`state_dict` 的软更新工具函数。
- **梯度**：`jax.grad(loss_fn)(params)` → `loss.backward()`；注意 `model.zero_grad()` 与 `optimizer.zero_grad()` 的调用时机。

### 3.5 随机数

- **JaxRNG / next_rng**（`utils/jax_utils.py`）：用 Python `random` 或 `torch.Generator` 替代；若需可复现，设置 `torch.manual_seed`/`np.random.seed`。在需要随机性的地方（如 dropout、采样动作）传入 generator 或使用全局/线程局部 RNG。
- **Dropout**：训练时 `model.train()`，推理时 `model.eval()`，与 Flax 的 `train=True/False` 对应。

### 3.6 分布与动作采样

- **distrax**：`distrax.MultivariateNormalDiag`、`distrax.Transformed`、`TanhMultivariateNormalDiag`、`distrax.Block`、`distrax.Tanh`、`distrax.Chain` 等 → `torch.distributions.MultivariateNormal`（或 `Normal` + `Independent`）、`torch.distributions.transforms.Transform`、`TanhTransform`、`ComposeTransform` 等，实现与现有策略相同的 squashing 与 log_prob 计算。
- **Policy 输出**：保持“返回一个分布对象，支持 `.sample()`、`.mode()`、`.log_prob(a)`”的接口，便于 SAC/DRQ 等里对熵、log_prob 的使用。
- **optax.sigmoid_binary_cross_entropy** 等 → `F.binary_cross_entropy` 或等价实现。

### 3.7 数据与 Replay Buffer

- **ReplayBuffer**（`data/replay_buffer.py`、`memory_efficient_replay_buffer.py`）：内部继续用 NumPy 存储；`get_iterator` 中不再使用 `jax.device_put`，改为在迭代时把 batch 转为 `torch.Tensor` 并 `.to(device)`；若存在 `batch_to_jax`，改为 `batch_to_torch(device)`。
- **Dataset/DataStore**：与 agentlace 等外部数据接口若原样保留，则仅保证喂给 PyTorch 模型的是 `torch.Tensor`；必要时在 dataloader 或 sampler 里做 to(device)。

### 3.8 视觉与编码器

- **vision/**：所有 Flax 编码器（如 ResNet、SmallEncoder、Mobilenet、Film 等）改为 `nn.Module`，卷积/归一化用 `nn.Conv2d`、`nn.LayerNorm` 或 `nn.BatchNorm2d` 等；保持输入/输出形状与原有逻辑一致，便于与 `EncodingWrapper`、actor-critic 对接。
- **data_augmentations**：`jax.lax.cond`、`jax.vmap` 等改为 PyTorch 分支与 `torch` 操作或简单循环；若对单张/批量图像做增强，用 `torchvision.transforms` 或手写 tensor 操作并注意 device。

### 3.9 公共类型与工具

- **common/typing.py**：`Params` 不再用 `FrozenDict`；可改为 `Dict[str, Any]` 或删除仅 JAX 用的类型；`Array` 改为 `Union[np.ndarray, torch.Tensor]`；`PRNGKey` 可改为 `Optional[torch.Generator]` 或删除。
- **common/encoding.py**：`EncodingWrapper`、`GCEncodingWrapper` 从 Flax 改为 PyTorch Module，内部 `jnp` 全改为 `torch`，`jax.lax.stop_gradient` → `x.detach()`。
- **common/common.py**：删除或重写 `shard_batch`（多 GPU 时可用 PyTorch DDP 或 `device_map`）；`ModuleDict` 用 `nn.ModuleDict`；`default_init` 用于 PyTorch 时在相应模块的 `reset_parameters` 或自定义 init 中实现。

### 3.10 Agent 类（SAC / DRQ / BC / VICE）

- **SACAgent / DrQAgent / BCAgent / VICEAgent**：从 Flax 的 `struct.PyTreeNode` + `JaxRLTrainState` 改为普通 Python 类，持有 PyTorch `model`、`optimizer`、`target_*` 等。
- **create / create_drq / create_vice**：构造函数里用 `model_def.init` 的等价：实例化 PyTorch 模型，用 dummy 输入做一次前向以确认形状，然后初始化 optimizer、target 拷贝等。
- **select_action**：输入为 numpy 或 tensor，输出为 numpy（与 env 兼容）；内部用 `model.eval()` 与 `torch.no_grad()`。
- **update**：从 `state.apply_loss_fns`/`jax.grad` 改为取 batch → 前向 → 算 loss → `backward` → `optimizer.step()`；需要多步 update（如 utd_ratio）时用循环；`@jax.jit` 可删除，PyTorch 默认 eager；若需编译可后续用 `torch.compile`。
- **update_target**：按现有周期或步数调用 target 的 polyak 更新。

### 3.11 Checkpoint

- **flax.training.checkpoints** / **orbax**：全部改为 `torch.save` / `torch.load` 或 `state_dict` + 额外保存 `step`、`optimizer.state_dict()` 等；恢复时 `model.load_state_dict(...)`，并恢复 step 与 optimizer 状态（若需要）。
- 保持与现有脚本兼容：例如 `checkpoint_path`、`eval_checkpoint_step`、`checkpoint_period` 等 flag 含义不变，仅内部实现改为 PyTorch 的 save/load。

### 3.12 其他工具

- **jax_utils.py**：`batch_to_jax` → `batch_to_torch(device)`；RNG 替代见 3.5。
- **train_utils.py**：若存在加载 ResNet 等预训练权重的逻辑，改为加载 PyTorch state_dict（格式需与当前实现一致或做 key 映射）。
- **launcher.py**：`make_*_agent` 返回的 agent 改为 PyTorch 版；`jax.random.PRNGKey(seed)` 改为 `torch.manual_seed(seed)` 等。

---

## 四、验收与兼容性

- **行为**：在相同 seed、相同数据下，迁移后的 SAC/DRQ/BC 与 JAX 版本在 1～2 个 epoch 内应得到相近的 loss 曲线（允许小幅数值差异）；若存在单元测试，需通过或更新为 PyTorch 版本。
- **API**：`examples/` 中通过 `make_*_agent` 创建 agent、调用 `agent.update(batch)`、`agent.select_action(obs)`、保存/加载 checkpoint 的代码，应尽量只改 import 和（如有）设备/路径配置，不改业务逻辑。
- **文档**：在 `README.md` 或 `docs/` 中增加“PyTorch 安装说明”（如 `pip install torch torchvision` 及 CUDA 版本），并注明已从 JAX 迁移为 PyTorch。

---

## 五、执行顺序建议

1. 更新 **requirements.txt** 和 README 中的安装说明。
2. 实现 **common**：typing、optimizer 工厂、训练状态/目标更新、encoding 的 PyTorch 版。
3. 实现 **networks**：MLP、Policy、Critic、ValueCritic、DistributionalCritic、Lagrange 等；分布与 Tanh 变换用 `torch.distributions`。
4. 实现 **vision**：编码器与 data_augmentations 的 PyTorch 版。
5. 实现 **data**：ReplayBuffer 的迭代器与 `batch_to_torch`。
6. 实现 **agents**：SAC → DRQ → BC → VICE；每个 agent 的 `create`/`update`/`select_action`/target 更新。
7. 实现 **utils**：jax_utils 替代、train_utils 中与权重加载相关的部分。
8. 更新 **launcher.py** 的 `make_*_agent`。
9. 逐个更新 **examples/** 下的脚本与 checkpoint 路径/保存加载逻辑。
10. 运行至少一个示例（如 `async_sac_state_sim` 或 `async_drq_sim`）做端到端验证，并更新文档。

---

请严格按照以上指令执行迁移，保证算法一致性与接口兼容性，并在关键处添加简短注释标出与 JAX 的对应关系，便于后续对照与调试。
