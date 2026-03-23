# 残差强化学习多环境接入：统一 I/O 合约落地指南（先不做代码整合）

本文给出一个“先统一输入输出协议，再考虑代码复用”的实施方案，目标是让你新增一个 residual RL example 时，主要只改环境 I/O 规则，而不是重写整套训练脚本。

适用范围：

- `examples/libero`
- `examples/RoboTwin`
- 未来新增环境（例如新的仿真任务或真机任务）

不在本文范围：

- 不做大规模目录重构
- 不做训练循环合并
- 不强行抽成单一 shared package

---

## 1. 目标与核心思路

你要的目标可以拆成一句话：

**训练主循环只依赖统一合约，不依赖具体环境字段。**

对齐 OpenPI 的干净设计，可以抽象成三层：

1. `repack`: 只做字段重排/命名对齐
2. `env_transform`: 环境语义变换（图像、状态、动作空间）
3. `model_transform`: 模型语义变换（tokenize/pad/normalize 等）

在你的 residual RL 场景里，同样可以采用这三层，只是 `model_transform` 更偏向 residual actor 的输入输出整理。

---

## 2. 第一步：定义统一 I/O 合约（文档 + 类型）

先把这 4 条链路定义成稳定接口，不要把它们散在脚本内部：

1. `EnvObs -> OpenPIInput`
2. `OpenPIActionChunk -> BaseActionWindow`
3. `(EnvObs, BaseAction) -> ResidualObs`
4. `(BaseAction, ResidualAction) -> FinalAction`

建议在文档层先固定以下类型（可后续转为 `TypedDict`/`Protocol`）：

```python
# 仅示意：先用于统一术语和字段，不要求立刻改代码
EnvObs = dict
OpenPIInput = dict
OpenPIActionChunk = np.ndarray          # [T, A_env]
BaseActionWindow = np.ndarray           # [H, A_env]
ResidualObs = dict[str, np.ndarray]     # {"state": [1, S], "image_*": ...}
ResidualAction = np.ndarray             # [A_res] 或 [H, A_res]
FinalAction = np.ndarray                # [A_env] 或 [H, A_env]
```

同时固定每个接口的输入/输出约束（重点是 shape、dtype、坐标系）：

- 图像通道顺序：`HWC` 或 `CHW`
- 图像 dtype：`uint8`（是否允许 `float32`）
- 动作维度：`A_env`、`A_res` 的定义和关系
- gripper 夹爪范围：`[-1,1]` 还是 `[0,1]`
- action chunk 规则：截断/末步填充策略

建议输出一个统一的“合约表”作为仓库内规范（新增环境前先填表）。

---

## 3. 第二步：每个 example 只维护一个 `env_io_spec`

每个环境定义一个 `env_io_spec`，它是“环境接入策略”的单点真相（Single Source of Truth）。

`env_io_spec` 最少应包含：

1. `env_name`
2. `action_dim_env`
3. `action_dim_residual`
4. `control_indices`
5. `image_keys`
6. `gripper_clip_rule`
7. `encode_obs_for_openpi(obs, prompt) -> OpenPIInput`
8. `select_base_window(chunk, horizon) -> BaseActionWindow`
9. `build_residual_obs(obs, base_action) -> ResidualObs`
10. `compose_final_action(base_action, residual_action) -> FinalAction`

你可以把它实现成 dataclass，也可以是模块函数集合。关键不是形式，而是做到：

- 主训练循环只调用 `spec.xxx(...)`
- 环境字段差异全部收口在 `spec`

### 3.1 对当前两个 example 的映射

`libero` 当前已有对应实现：

- OpenPI 编码：`examples/libero/policy/openpi_client.py`
- 观测构建：`examples/libero/policy/observation.py`
- 动作组合：`examples/libero/policy/action.py`

`RoboTwin` 也有对应实现：

- OpenPI 编码：`examples/RoboTwin/policy/openpi_client.py`
- 观测构建：`examples/RoboTwin/policy/observation.py`
- 动作组合：`examples/RoboTwin/policy/action.py`

下一步只需要把它们“命名为 spec 接口”，并让训练脚本通过 spec 调用，而不是直接 import 多个散函数。

---

## 4. 第三步：按 OpenPI 三段式改写你自己的 I/O 流

这里先做流程整理，不做代码合并。

### 4.1 输入正向流

训练/评估时统一按下面顺序处理：

1. `repack`: 从原始 `EnvObs` 提取规范字段
2. `env_transform`: 构造 OpenPI 输入或 residual actor 输入
3. `model_transform`: 归一化、pad、堆叠维度处理

可写成：

```text
raw_obs
  -> repack(raw_obs)
  -> openpi_input_transform(...)
  -> OpenPI infer
  -> base_window_transform(...)
  -> residual_obs_transform(...)
  -> residual actor infer
```

### 4.2 输出反向流

动作输出统一反向处理：

1. `model_out`: residual actor 输出动作
2. `env_out_transform`: 按 limits/indices/xi/scaling 映射到 env 动作空间
3. `final_action`: 应用 gripper clip/动作边界后执行到 env

可写成：

```text
residual_action
  -> model_out_transform(...)
  -> compose_final_action(...)
  -> final_action_for_env
```

这一步的收益是：新增环境时，你只改 transform 链条，不碰训练状态机和 replay/update 逻辑。

---

## 5. 第四步：把差异下沉到 Spec/配置，而不是写在训练脚本里

建议把以下差异项全部配置化，并挂到 `env_io_spec`：

1. `action_dim_env`
2. `action_dim_residual`
3. `control_indices`
4. `action_limits`（按维度的残差幅度上限）
5. `gripper_clip_min`、`gripper_clip_max`
6. `image_keys` 与图像来源路径
7. `image_layout` (`HWC`/`CHW`)
8. `image_resize_policy`（是否 `resize_with_pad`）
9. `state_builder` 规则（例如 `ee_pos||ee_ori||gripper`)
10. `openpi_input_schema`（`observation/*` 还是 `state+images`）

### 5.1 配置示意

```yaml
env_io:
  name: libero
  action_dim_env: 7
  action_dim_residual: 7
  control_indices: [0,1,2,3,4,5,6]
  gripper_clip:
    enabled: true
    min: -1.0
    max: 1.0
  images:
    keys: [image, wrist_image]
    layout: HWC
    resize_with_pad: [224, 224]
  openpi_schema: libero_observation_keys
```

这样做以后，训练脚本里就不应该再出现：

- 对具体 image key 的硬编码判断
- 对具体 gripper 维度位置的硬编码
- 对某个环境特定字段名的 if/else

---

## 6. 第五步：增加最小 contract test（新增环境先过门槛）

新增环境前必须通过一条最小链路测试：

`reset -> encode_openpi -> infer_chunk(mock) -> build_residual_obs -> compose_action`

### 6.1 最小测试建议

1. 构造一个 `fake env obs`（或真实 env reset 一帧）
2. 调用 `spec.encode_obs_for_openpi`，检查 key/shape/dtype
3. 用 mock OpenPI chunk（例如随机 `[T, A_env]`）走 `select_base_window`
4. 调用 `spec.build_residual_obs` 检查 residual actor 输入形状
5. 给一个 mock residual action，调用 `spec.compose_final_action`
6. 检查 final action 的维度与 clip 规则

### 6.2 通过标准（建议）

1. 无异常
2. 所有 shape 与 spec 一致
3. dtype 一致（避免隐式类型变化）
4. gripper clip 行为符合配置
5. chunk 截断/填充逻辑符合定义

只要这一组 contract tests 通过，新增 example 才允许进入训练阶段。

---

## 7. 推荐落地顺序（分 3 个小迭代）

### Iteration 1：先立规矩，不改大结构

1. 增加本文档定义的合约和字段表
2. 为 `libero`、`RoboTwin` 各写一份 `env_io_spec` 草案（可仍调用旧函数）
3. 增加最小 contract test（至少 1 个正例）

### Iteration 2：训练脚本改为“只调 spec”

1. 在 `train/eval` 脚本中，把 I/O 相关调用统一替换为 `spec.xxx`
2. 删除脚本中的环境字段硬编码
3. 保持 replay/update 逻辑不动

### Iteration 3：把离线链路也接到同一 spec

1. `convert_offline`、`offline_residual` 也改为调用 `spec`
2. 确保在线/离线使用同一组 state/image/action 变换
3. 新增离线 contract test（至少检查一条 episode）

---

## 8. 新增环境时的执行清单（Checklist）

1. 新建一个 `env_io_spec` 文件
2. 填写 `action_dim/image_keys/openpi_schema/gripper_clip` 配置
3. 实现 4 个核心函数：
   - `encode_obs_for_openpi`
   - `select_base_window`
   - `build_residual_obs`
   - `compose_final_action`
4. 通过最小 contract test
5. 运行一轮 smoke test（1 个 episode）
6. 再接入正式训练

如果这 6 步做完，你就可以实现“新增 example 只改输入输出，不重写一堆实现”。

---

## 9. 常见坑位（提前规避）

1. 同一环境在线与离线使用了不同 state 构造逻辑
2. 图像通道顺序在不同模块不一致（`HWC/CHW`）
3. gripper clip 区间没有配置化，脚本里硬编码
4. action_dim 与 control_indices 不一致
5. chunk padding 策略在线与评估不一致

建议把这 5 条做成 CI 检查项，至少在单元测试里覆盖。

---

## 10. 完成定义（Definition of Done）

当满足下面条件时，可认为这套改进达标：

1. 训练主循环不再直接依赖具体环境字段名
2. 每个环境新增只需新增/修改 `env_io_spec` + 配置
3. 在线、离线、评估三条链路共用同一套 I/O 规则
4. 新环境接入前有 contract test 门槛

这时你后续再做“代码复用/代码整合”，风险会显著更低，因为接口边界已经稳定。
