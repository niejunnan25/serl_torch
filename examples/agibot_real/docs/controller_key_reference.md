# AgiBot Controller 按键速查

这份文档只说明当前 `examples/agibot_real` 训练/评测主线里，terminal controller 支持哪些按键，以及它们的运行时语义。

## 生效前提

- `controller.enabled=true`
- `controller.interface=terminal`
- actor / eval 进程所在终端必须有 TTY

默认配置见：

- [../configs/train_residual_copy.yaml](../configs/train_residual_copy.yaml)
- [../configs/train_residual.yaml](../configs/train_residual.yaml)
- [../configs/eval_residual.yaml](../configs/eval_residual.yaml)

默认按键定义在：

- [../env/controller.py](../env/controller.py)
- [../config.py](../config.py)

## 默认按键

- `g`: ready / resume
- `p`: pause
- `r`: reset
- `s`: success
- `f`: fail
- `h`: help

## 每个按键的作用

### `g` ready / resume

- 在 `WAIT_READY` 状态下，按 `g` 会进入 `RUNNING`
- 在 `PAUSED` 状态下，按 `g` 会继续执行
- 常见用法是：`env.reset()` 之后，机器人回到初始位，等待你按 `g` 开始本局

### `p` pause

- 仅在 `RUNNING` 状态下生效
- 按下后 controller 进入 `PAUSED`
- 已经执行完的动作不会回滚，只是暂停继续发下一个动作

### `r` reset

- 当前 episode 立即结束
- 语义是人工中断，不算成功
- 训练/评测结果会写成：
  - `reward=0.0`
  - `done=false`
  - `truncated=true`
  - `info["human_reset"]=True`
  - `info["controller_terminal_signal"]="reset"`
- 外层流程随后会进入 reset

### `s` success

- 当前 episode 立即按“成功”结束
- 训练/评测结果会写成：
  - `reward=1.0`
  - `done=true`
  - `truncated=false`
  - `info["success"]=True`
  - `info["human_success"]=True`
  - `info["controller_terminal_signal"]="success"`

### `f` fail

- 当前 episode 立即按“失败”结束
- 训练/评测结果会写成：
  - `reward=0.0`
  - `done=true`
  - `truncated=false`
  - `info["success"]=False`
  - `info["human_fail"]=True`
  - `info["controller_terminal_signal"]="fail"`

### `h` help

- 只打印当前按键映射
- 不改变 controller 状态

## 状态流转

典型流程是：

1. `env.reset()`
2. reset hook 把机器人回到 task initial pose
3. controller 进入 `WAIT_READY`
4. 你按 `g`
5. controller 进入 `RUNNING`
6. 运行过程中可以按 `p` 暂停，再按 `g` 继续
7. 你按 `s` / `f` / `r` 结束当前局，或者达到 step limit 自动结束

## 自动结束语义

除了人工按键外，episode 也可能因为 step limit 结束。当前语义是：

- `reward=0.0`
- `done=false`
- `truncated=true`
- `info["time_limit_reached"]=True`
- `info["controller_terminal_signal"]="timeout"`

## 粘键保护

当前默认有一个很短的 terminal grace 窗口：

- `controller.terminal_grace_sec=0.15`

这表示在刚刚触发 `success` / `fail` / `reset` 之后，短时间内重复按这些终止键会被忽略，避免 episode 边界粘键。

## 自定义按键

如果你要改默认按键，可以直接改 yaml：

```yaml
controller:
  keys:
    ready: g
    pause: p
    reset: r
    success: s
    fail: f
    help: h
```

当前代码只支持这六类控制键，没有额外的“单步执行”“跳过当前 chunk”之类快捷键。
