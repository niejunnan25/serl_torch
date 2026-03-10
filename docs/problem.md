# serl_torch 当前已知问题总结

本文档汇总 RoboTwin 残差 SAC（DrQ-SAC）训练流程中已发现的问题，按优先级排列，便于逐项修复与迭代。

---

## 一、算法正确性结论（无 bug 部分）

以下模块经逐行审查，**实现正确**，无需修改：

| 模块 | 结论 |
|------|------|
| **SAC 核心** (`sac.py`) | Critic TD target、Actor loss、Temperature 拉格朗日、Target 软更新、Tanh log_prob 修正、CriticEnsemble min、update_high_utd 流程均正确 |
| **DrQ** (`drq.py`) | 随机裁剪增强、obs/next_obs 同时增强，正确 |
| **ReplayBuffer** | 环形插入、均匀采样，正确 |
| **训练主循环** (`train_residual_sac.py`) | transition 构建、mask/done、Async 三锁、参数同步快照，正确 |
| **共享编码器梯度** | Actor 对图像 encoder 使用 stop_gradient，仅 Critic 更新图像编码器，正确 |

**结论：核心 SAC/DrQ 算法本身没有正确性 bug。** 训练不收敛/成功率下降主要来自下面列出的实现与配置问题。

---

## 二、已知问题列表（按优先级）

### P0（严重）—— 预训练 ResNet 权重未成功加载

**现象：** 使用 `encoder_type: resnet-pretrained` 时，视觉编码器实际是**随机初始化且冻结**的 ResNet-18，三路相机图像等价于随机特征，视觉信息几乎未被利用。

**原因简述：**

1. **架构名实不符**  
   `resnetv1_configs["resnetv1-10-frozen"]` 对应 `stage_sizes=(1,1,1,1)`，但 `ResNetEncoder` 中该分支实际构建的是 **torchvision ResNet-18**（`tv_models.resnet18(weights=None)`），并非“ResNet-10”，且 `weights=None` 为随机初始化。

2. **权重格式与 key 不匹配**  
   `load_resnet10_params` 从 `resnet10_params.pkl` 加载的是原始 SERL 的 **JAX/Flax** 格式参数，key 形如 `ResNetEncoder_0.Conv_0.kernel`；PyTorch ResNet-18 的 key 形如 `layer1.0.conv1.weight`。二者完全不兼容。

3. **静默失败**  
   `load_state_dict(..., strict=False)` 对不匹配的 key 直接忽略，不会报错，因此**没有任何参数被真正加载**。

4. **冻结的是随机权重**  
   `pre_trained_frozen=True` 将上述随机 backbone 冻结，仅后续的 SpatialLearnedEmbeddings + bottleneck 可训练，导致从随机特征中学习，效率极低。

**涉及文件：**

- `serl_launcher/serl_launcher/vision/resnet_v1.py`（ResNetEncoder 构建逻辑）
- `serl_launcher/serl_launcher/agents/continuous/drq.py`（resnet-pretrained 分支）
- `serl_launcher/serl_launcher/utils/train_utils.py`（`load_resnet10_params`）

**修复方向建议：**  
要么提供与当前 PyTorch ResNet 结构、命名一致的预训练权重并正确加载；要么改为使用 `torchvision` 自带的预训练权重（如 `weights=ResNet18_Weights.IMAGENET1K_V1`），并做好与下游 pooling/bottleneck 的衔接。

**✅ 已修复。** 具体修复内容如下：

1. **彻底重写 `ResNetEncoder`（`vision/resnet_v1.py`）**  
   废弃旧的 torchvision + JAX/Flax 权重加载方案，改为使用 **HuggingFace Transformers** 的 `ResNetModel`。通过 `ResNetModel.from_pretrained(model_name)` 加载预训练权重，架构与权重严格匹配，不再存在 key 不兼容问题。同时支持传入 HuggingFace model ID（如 `microsoft/resnet-18`）或本地已下载的模型目录绝对路径。

2. **消除全局变量，所有配置迁移至 YAML**  
   删除了旧的 `resnetv1_configs` 全局字典、`resolve_pretrained_path()`、`PreTrainedResNetEncoder` 等。`ResNetEncoder` 现在接受一个已构建好的 `backbone`（`nn.Module`）以及 `freeze_backbone`、`pooling_method`、`num_spatial_blocks`、`bottleneck_dim` 等参数。backbone 的创建由静态方法 `ResNetEncoder.create_backbone(model_name, pretrained, freeze)` 负责。所有配置由 YAML `sac.resnet` 块统一管理。

3. **多路相机共享 backbone，避免内存浪费**  
   `create_backbone()` 只调用一次，返回的 `ResNetModel` 实例被传给所有 image key 对应的 `ResNetEncoder`。3 路相机只有 **1 份** backbone 参数，每路各自拥有独立的 pooling + bottleneck 层。`vice.py` 中 `BinaryClassifier` 的 `backbone_encoder` 也复用同一份 backbone。

4. **支持灵活的 ResNet 变种选择**  
   可通过 YAML 中 `sac.resnet.model_name` 切换不同 ResNet 架构（如 `microsoft/resnet-18`、`microsoft/resnet-50` 等），无需修改代码。同时支持 `pretrained: true/false` 切换预训练 / 随机初始化。随机初始化时通过内置的 `VARIANT_CONFIGS` 字典离线创建网络结构，无需联网下载 config。

5. **冻结逻辑确保作用于真正的预训练权重**  
   `freeze_backbone: true` 仅在权重被正确加载之后执行 `requires_grad_(False)` + `eval()`，并在 `forward()` 中使用 `torch.no_grad()` + `.detach()` 确保梯度不回传。不再存在"冻结随机权重"的问题。

6. **`load_state_dict` 改为 `strict=True`**  
   `TorchRLTrainState` 的 `params` / `target_params` setter 以及 `reward_classifier.py` 的 checkpoint 加载中，`load_state_dict` 的 `strict` 参数均为 `True`，任何 key 不匹配都会立即报错，杜绝静默失败。

7. **提供离线权重下载脚本**  
   新增 `tools/download_resnet.py`，用于从 HuggingFace Hub 下载指定模型到 `pretrained_models/` 目录，供 Docker / 离线环境使用。

8. **全链路文件同步更新**  
   `drq.py`、`bc.py`、`vice.py`、`reward_classifier.py`、`launcher.py`、`config_utils.py`、`train_residual_sac.yaml`、`eval_residual_fast.yaml` 以及示例脚本中所有涉及旧 encoder 接口的调用均已同步适配新接口。

---

### P1（中等）—— xi_scheduler 退火在 warmup 期间被浪费

**现象：** 配置了 `xi_scheduler`（如从 `min_xi=0.1` 线性退火到 `base_xi=0.5`），但 warmup 期间残差动作被强制为 0，xi 不参与执行；等 warmup 结束，退火早已完成，残差一步到位到最大幅值，没有“逐步放大”的过程。

**典型配置：**

- `warmup_base_episodes: 100` → 约 2 万步内残差均为 0
- `xi_scheduler.anneal_steps: 2000`、`warmup_steps: 0` → 从 step 0 开始退火，2000 步即到 0.5

**结果：** warmup 结束后残差从 0 直接跳到满幅（xi=0.5），容易造成策略突变、不稳定。

**修复方向建议：**  
xi_scheduler 的退火起点应相对于“warmup 结束”计算，例如：  
`effective_step = max(0, global_policy_step - warmup_base_steps)`，用 `effective_step` 驱动 xi 退火；或单独配置 `xi_scheduler.warmup_steps` 与训练阶段的 warmup 对齐。

---

### P1（中等）—— warmup 零动作数据占满 replay

**现象：** 前约 100 个 episode（约 2 万条 transition）的 `action` 全为零向量写入 replay。这批数据在 buffer 中占比高，Critic 大量在“残差=0”的数据上训练。

**结果：** Critic 对**非零残差动作**的 Q 估计缺乏有效学习；warmup 结束后策略开始输出非零残差时，Q 值估计不准，影响策略更新质量。

**修复方向建议：**  
例如：warmup 数据不入 replay；或单独设一个小 buffer 存 warmup，正式训练只用 warmup 之后的 online/offline 混合；或显著缩短 warmup episode 数，并配合 xi_scheduler 在 warmup 之后再做退火。

---

### P2（轻微）—— proprio_proj 被双 Optimizer 同时更新

**现象：** `shared_encoder=True` 时，`EncodingWrapper` 中的 `proprio_proj` 被 Actor 和 Critic 共用。`stop_gradient=True` 只对**图像特征**做了 detach，**没有**对 `proprio_proj` 的输出做 detach，因此其参数同时出现在 Actor 与 Critic 的 optimizer 中。

**结果：** 两个 AdamW 各自维护该部分的动量/二阶估计，轮流更新同一组参数，可能带来额外噪声与不稳定。

**涉及文件：** `serl_launcher/serl_launcher/common/encoding.py`

**修复方向建议：**  
若希望仅由 Critic 更新共享表征，可对 `proprio_proj` 输出也做 detach（在 Actor 前向路径）；或明确约定仅由一方（如 Critic）更新 `proprio_proj`，另一方只读。

---

### P2（潜在 Bug）—— Critic/ValueCritic 的 init_final 在 forward 中重复初始化

**现象：** 在 `Critic._forward_single` / `ValueCritic.forward` 中，若 `self.init_final is not None`，**每次 forward 都会**对最后一层（Q-head / value head）执行 `nn.init.uniform_(...)`，导致该层权重被不断重置。

**结果：** 若将来某配置传入 `init_final`，会导致该层无法正常学习，属于严重逻辑错误。当前默认未使用 `init_final`，故未触发。

**涉及文件：** `serl_launcher/serl_launcher/networks/actor_critic_nets.py`

**修复方向建议：**  
若有“最后一层小范围初始化”的需求，应在 **`__init__` 或首次 forward 之前**执行一次初始化，而不是在每次 forward 中执行；或删除该分支，改用标准初始化。

---

### P3（设计/配置）—— backup_entropy: false

**现象：** 当前配置下 Critic 的 TD target 为  
`target_q = reward + discount * mask * min_next_q`，**不包含**熵项（即未减去 `alpha * log_pi(next_a)`）；而 Actor 目标为 `max (Q - alpha * log_pi)`。

**结果：** Critic 与 Actor 对“好动作”的度量不完全一致，属于设计/配置层面的不对齐。并非实现 bug，但可能影响稳定性和样本效率。

**修复方向建议：**  
若希望与标准 SAC 完全一致，可将 `sac.backup_entropy` 设为 `true`，使 Critic target 与 Actor 目标一致（含熵项）。

---

## 三、优先级与修复顺序建议

| 优先级 | 问题 | 建议处理顺序 |
|--------|------|--------------|
| **P0** | ResNet 预训练权重未加载 | 最先修复，否则视觉端几乎无效 |
| **P1** | xi_scheduler 退火与 warmup 错位 | 与 warmup 策略一起调整 |
| **P1** | warmup 零动作占满 replay | 同上，可一并设计 warmup/replay 策略 |
| **P2** | proprio_proj 双 optimizer | 在 P0/P1 之后优化稳定性时处理 |
| **P2** | init_final 在 forward 中初始化 | 代码清理时修复，避免后续误用 |
| **P3** | backup_entropy=false | 按需开启，与算法偏好一致即可 |

---

## 四、文档维护

- 本文档路径：`serl_torch/docs/problem.md`
- 每修复一项，建议在本文件中注明“已修复”及对应 commit/PR，或移至“已关闭问题”小节，以便后续只关注未解决问题。
