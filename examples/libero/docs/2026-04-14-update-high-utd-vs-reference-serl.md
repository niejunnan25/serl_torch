# `update_high_utd` 逻辑梳理与 `reference/serl` 对照

生成时间（北京时间）: 2026-04-14 11:07:01 CST

## 结论

- 如果只看 `update_high_utd` 这个函数本身，当前仓库和 `reference/serl` 的核心思路是一样的。
- 真正容易混淆的地方不在函数本身，而在 learner 外层循环还额外做了多少次 `update_critics(...)`。
- 当前 LIBERO 默认配置下，当前仓库和 `reference/serl/examples/async_pcb_insert_drq` 这类写法，本质上是同一种配方。
- 如果对照的是 `reference/serl/examples/async_sac_state_sim` 这种“只调用 `update_high_utd(utd_ratio>1)`”的写法，总的 critic:actor 次数可以一样，但数据组织方式不完全一样。

## 当前仓库里 `update_high_utd` 本身在做什么

相关代码:

- `serl_launcher/serl_launcher/agents/continuous/drq.py:222`
- `serl_launcher/serl_launcher/agents/continuous/sac.py:1262`
- `serl_launcher/serl_launcher/agents/continuous/sac.py:362`

当前 Torch 版的流程可以拆成两层:

1. `DrQAgent.update_high_utd(...)`
   - 如果 batch 还是打包格式，先解包。
   - 对 `observations` 和 `next_observations` 做图像增强。
   - 然后调用 `SACAgent.update_high_utd(...)`。

2. `SACAgent.update_high_utd(...)`
   - 把 batch 转成 torch tensor。
   - 按 `utd_ratio` 沿 batch 维切成 `utd_ratio` 份。
   - 对每一份 minibatch 做一次 critic-only update。
   - 把这些 critic 的 info 做平均。
   - 再用完整原 batch 做一次 actor + temperature update。
   - 合并 info 返回。

因此，单看这个函数本身，它的语义是:

- critic 更新 `utd_ratio` 次
- actor 更新 `1` 次
- temperature 更新 `1` 次

## 当前 LIBERO learner 外层循环在做什么

相关代码:

- `examples/libero/scripts/run_residual_training.py:600`

当前 learner 并不是“每轮只调一次 `update_high_utd`”。它先做额外 critic update，再调一次 `update_high_utd`:

```python
for _ in range(max(0, critic_actor_ratio - 1)):
    agent, _critics_info = agent.update_critics(batch)

agent, update_info = agent.update_high_utd(
    batch,
    utd_ratio=cfg.sac.utd_ratio,
)
```

所以当前一轮 learner loop 的总更新次数其实是:

- critic: `(critic_actor_ratio - 1) + utd_ratio`
- actor: `1`
- temperature: `1`

这句话是理解区别的关键。

## 当前默认配置下的例子

相关配置:

- `examples/libero/configs/train_residual.yaml:83`
- `examples/libero/configs/train_residual.yaml:102`

当前默认值是:

- `sac.utd_ratio = 1`
- `training.critic_actor_ratio = 4`

那么一轮 learner loop 实际发生的是:

1. `update_critics(...)`
2. `update_critics(...)`
3. `update_critics(...)`
4. `update_high_utd(..., utd_ratio=1)`
   - critic 第 4 次
   - actor 第 1 次
   - temperature 第 1 次

最终总数:

- critic: `4`
- actor: `1`
- temperature: `1`

注意这里一个容易误解的点:

- 当前默认配置里，`update_high_utd` 本身其实并不“高”，因为 `utd_ratio=1`。
- 当前默认的 `4:1` 主要是由外层的 `critic_actor_ratio=4` 形成的。

## `reference/serl` 里的 `update_high_utd` 本身是什么逻辑

相关代码:

- `reference/serl/serl_launcher/serl_launcher/agents/continuous/sac.py:545`
- `reference/serl/serl_launcher/serl_launcher/agents/continuous/drq.py:256`

`reference/serl` 的 JAX 版 `SAC.update_high_utd(...)` 逻辑是:

1. 要求 batch size 能整除 `utd_ratio`
2. 把 batch reshape 成 `[utd_ratio, minibatch_size, ...]`
3. 用 `jax.lax.scan` 跑 `utd_ratio` 次 critic-only update
4. 对 critic infos 求平均
5. 再对原始 batch 做一次 actor + temperature update
6. 合并 info 返回

`reference/serl` 的 `DrQ.update_high_utd(...)` 也是:

- 先做图像增强
- 再调用 `SAC.update_high_utd(...)`

所以从函数定义层面说，当前仓库和 `reference/serl` 是同构的:

- critic `utd_ratio` 次
- actor `1` 次
- temperature `1` 次

## 为什么还会感觉“不一样”

原因在于:

- `update_high_utd` 本身是一层
- learner 外层怎么调用它，又是一层

`reference/serl/examples` 里并不是所有 example 都用同一种外层配方。

## 对照 1: `reference/serl/examples/async_pcb_insert_drq`

相关代码:

- `reference/serl/examples/async_pcb_insert_drq/async_drq_randomized.py:342`

它的 learner 外层写法是:

```python
for critic_step in range(critic_actor_ratio - 1):
    update_critics(...)

update_high_utd(batch, utd_ratio=1)
```

如果它的配置也是:

- `critic_actor_ratio = 4`
- `utd_ratio = 1`

那么这一轮总更新次数就是:

- critic: `4`
- actor: `1`
- temperature: `1`

这和当前仓库的默认 LIBERO learner 本质上一样。

## 对照 2: `reference/serl/examples/async_sac_state_sim`

相关代码:

- `reference/serl/examples/async_sac_state_sim/async_sac_state_sim.py:225`

它的 learner 外层写法是:

```python
update_high_utd(batch, utd_ratio=FLAGS.utd_ratio)
```

没有外层额外的 `update_critics(...)`。

如果它设:

- `utd_ratio = 4`

那么一轮总更新次数也是:

- critic: `4`
- actor: `1`
- temperature: `1`

所以从“总次数”看，它和上面的 `4:1` 也可以一样。

## 真正的区别: 数据是怎么组织的

即使总次数都一样，数据组织方式也可能不一样。

### 写法 A: 当前默认 / `async_pcb_insert_drq`

一轮可能是:

1. critic 用 batch `B1`
2. critic 用 batch `B2`
3. critic 用 batch `B3`
4. `update_high_utd(B4, utd_ratio=1)`
   - critic 用 `B4`
   - actor/temp 也用 `B4`

特点:

- 前 3 次 critic 是 3 个独立 sample
- 最后 1 次 critic 和 actor/temp 共享最后一个 batch

### 写法 B: 纯 `update_high_utd(B, utd_ratio=4)`

一轮可能是:

1. 先 sample 一个大 batch `B`
2. 把 `B` 切成 `B[1]`, `B[2]`, `B[3]`, `B[4]`
3. critic 用 `B[1]`
4. critic 用 `B[2]`
5. critic 用 `B[3]`
6. critic 用 `B[4]`
7. actor/temp 用完整 `B`

特点:

- 4 次 critic 和 1 次 actor/temp 围绕同一个大 batch 组织
- critic 的多次更新和 actor update 的数据“更成组”

## 一张总结表

| 情况 | 外层额外 `update_critics` | `update_high_utd` 内 critic 次数 | actor/temp 次数 | 总体结论 |
| --- | --- | --- | --- | --- |
| 当前 LIBERO 默认 | `critic_actor_ratio - 1 = 3` | `utd_ratio = 1` | `1` | 总共 critic `4` 次，actor/temp `1` 次 |
| `reference` `async_pcb_insert_drq` | `critic_actor_ratio - 1 = 3` | `utd_ratio = 1` | `1` | 与当前默认本质一致 |
| `reference` `async_sac_state_sim` 且 `utd_ratio=4` | `0` | `4` | `1` | 总次数可同为 `4:1`，但数据组织方式不同 |

## 最终结论

可以把这件事总结成两句话:

1. 当前仓库和 `reference/serl` 的 `update_high_utd` 函数本身，核心逻辑是一致的。
2. 真正的区别不一定在函数本身，而在 learner 外层是否还额外做了 `critic_actor_ratio - 1` 次 `update_critics(...)`，以及这些 critic 更新是不是围绕同一个大 batch 组织。

如果只看当前默认配置，那么:

- 当前仓库的 LIBERO learner
- `reference/serl/examples/async_pcb_insert_drq`

这两者在每轮总的 critic/actor/temp 次数上，本质是一致的。
