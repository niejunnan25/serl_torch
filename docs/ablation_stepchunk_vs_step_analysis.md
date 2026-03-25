# Ablation: stepchunk vs step (task6, xi=0.1, warmup) 训练分析报告（修订版）

> 分析日期: 2026-03-25  
> 实验目录: `examples/libero/outputs/libero/ablation_stepchunk_vs_step_task6_xi_warmup/`

---

## 一、四个实验当前态势（保留现象）

| 实验 | chunk_step | warmup | 当前状态 | 训练阶段最终成功率 | 近期成功率 |
|---|---|---|---|---|---|
| stepchunk_xi10_nowarmup | 开 | 0 ep | 已完成（300k env step） | **0.535** | 0.15 |
| stepchunk_xi10_warmup100ep | 开 | 100 ep | 已完成（300k env step） | **0.377** | 0.00 |
| step_xi10_nowarmup | 关 | 0 ep | 仍在跑（~98.8k env step） | 暂无最终值 | 0.00 |
| step_xi10_warmup100ep | 关 | 100 ep | 仍在跑（~111.4k env step） | 暂无最终值 | 0.00 |

结论（现象层）:
- 两个 `step` 组当前都处在长时间低成功率区间。
- `stepchunk_xi10_warmup100ep` 明显劣于 `stepchunk_xi10_nowarmup`。
- 但 `stepchunk_xi10_nowarmup` 是“后期退化但仍可用”，不是“完全崩坏”。

---

## 二、关键诊断（修正推导）

### 问题 1: 温度接近 0 的现象成立，但原文温度更新方向写反了

两个 stepchunk 完成实验的 `last_update_info` 都显示温度非常小:

- `stepchunk_xi10_nowarmup`: `temperature ≈ 2.13e-05`
- `stepchunk_xi10_warmup100ep`: `temperature ≈ 2.25e-05`

这说明熵正则权重几乎失效，actor 主要被 Q 项驱动。

但原文中“`entropy > target` 时温度增大”的推导不对。  
当前实现是:

- `temperature_loss = alpha * (entropy - target)`（`alpha >= 0`，softplus 参数化）
- 采用梯度下降优化

因此:
- 若 `entropy > target`，梯度为正，更新会让 `alpha` 变小
- 若 `entropy < target`，梯度为负，更新会让 `alpha` 变大

对应代码:
- `serl_launcher/serl_launcher/networks/lagrange.py`
- `serl_launcher/serl_launcher/agents/continuous/sac.py`

修正后解释:
- `temperature -> 0` 现象是事实；
- 但机制应描述为“温度动态与 actor-critic 联合更新耦合后，出现近零吸附/恢复困难”，而不是原文那条单向梯度叙述。

---

### 问题 2: step 组低迷，主要是“当前观测设计 + 任务形态”下难学，而不是可证明的理论不可行

原文“step 级别从根本上不可行”结论过强。更准确表述:

- 在当前实现中，step 模式只喂当前 `base_action`，不喂 `base_action_chunk`（chunk 模式会喂）；
- base policy 是 chunk 规划，step residual 的时间上下文更弱；
- task6 长时域 + 稀疏终奖，让 step residual 学习难度非常高；
- 结果表现为连续大量 520 步超时失败。

对应代码位置:
- `examples/libero/scripts/train_residual_sac.py`（step 分支构建 obs 与动作）
- `examples/libero/policy/observation.py`（`base_action_chunk` 融合逻辑）

---

### 问题 3: “Q 值偏小 => critic 退化”证据不足

原文把 `predicted_qs ~ 0.26~0.29` 直接解释为 critic 退化，这个证据不充分。  
在当前设置下（稀疏奖励、折扣、离线+在线混合），Q 的绝对量纲不能直接横向判断好坏。

更稳妥做法:
- 看同一实验内的 Q 趋势与成功率是否同向；
- 看 TD loss、Q gap、评估成功率一起变化；
- 不用“绝对 Q 数值”单独下诊断。

---

### 问题 4: warmup100ep 更差是“强相关现象”，但因果还需保守表达

现象成立:
- `stepchunk_xi10_warmup100ep` 最终显著低于 `stepchunk_xi10_nowarmup`。

合理假设（可讨论）:
- warmup + critic pretrain 让“低残差行为”成为早期吸引子；
- 后续在线偏离后，值函数泛化压力增大，性能下滑。

但这仍是“基于当前运行的解释”，不是已被对照实验严格证明的唯一因果。

---

### 问题 5: 离线 residual clipping 判断有道理

这部分原文基本正确，证据也够:

- `offline_stats.clipped_values = 13166`
- `denom = residual_limits * xi * expert_reference_scale`
- 当 `xi=0.1` 时，expert-base 差值较容易超出单位区间并被裁剪。

对应代码:
- `examples/libero/data/offline_residual.py`

这说明 `xi=0.1` 下离线监督信号存在明显截断，可能削弱“专家残差”信息质量。

---

## 三、实现层检查（保留并补充约束）

从代码看，没有发现“明显实现错误型 bug”，以下链路是自洽的:

1. chunk 动作合成与 step 动作合成逻辑一致（numpy/torch 对齐）
2. chunk replay 的窗口化、mask、discount 处理逻辑闭环
3. offline 数据进入 replay 的结构与训练读取结构匹配

补充:
- “无明显 bug”不等于“算法一定稳定”；
- 当前问题更像训练动力学与观测/任务匹配问题。

---

## 四、修订后的核心结论

1. 这组实验的主要问题不是单点代码 bug，而是训练稳定性与信息结构不匹配。
2. step 模式在当前实现下学习难度极高，短期内看不到恢复迹象。
3. stepchunk 模式可训练，但 warmup100ep 组合在当前配置下表现明显退化。
4. 温度近零是重要风险信号，但原文的温度梯度方向解释需要按代码修正。

---

## 五、建议（按优先级）

1. 先把 step 组视为“高风险/低收益”配置，避免继续大量算力投入。
2. 以 `stepchunk_xi10_nowarmup` 作为当前更稳的对照基线。
3. 增加训练过程观测:
   - 温度/熵/constraint gap 的时间曲线
   - `learner_update_steps / train_env_step` 曲线
   - async eval 成功率与训练内成功率并排
4. 对 `xi` 与离线 clipping 做小范围网格（如 0.1/0.2/0.3）而不是单点判断。
5. 若继续保留 warmup，建议单独做“是否 pretrain critic”的拆分对照，避免把 warmup 和 pretrain效应混在一起。

---

## 附录：本次修订修正了什么

- 保留了原文的大部分“现象观察”；
- 修正了温度更新方向的推导错误；
- 将“根本不可行/必然退化”改为“当前实现与证据下的高置信判断”；
- 把“Q 绝对值”从强诊断证据降级为辅助信号。
