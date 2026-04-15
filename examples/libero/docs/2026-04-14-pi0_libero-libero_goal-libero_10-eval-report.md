# pi0_libero LIBERO Eval Report

生成时间（北京时间）: 2026-04-14 02:28:12 CST

## 评测配置

- 策略名称: `pi0_libero`
- GPU: `2`
- Policy script: `/vla/users/niejunnan/codebase/serl_torch/examples/libero/tools/serve_openpi_policy.sh`
- Policy port: `42002`
- 结果目录: `/vla/users/niejunnan/codebase/serl_torch/outputs/openpi_libero_eval/2026-04-13_23-52-56_goal10_gpu1_gpu2/pi0_libero`
- 套件数: `2`
- 套件列表: `libero_goal, libero_10`

## 总览

- 总 episode 数: `1000`
- 总成功数: `876`
- 加权总成功率: `87.6%`
- Suite 平均成功率: `87.6%`

| Suite | Success Rate | Successes | Episodes |
| --- | --- | --- | --- |
| `libero_goal` | `92.6%` | `463` | `500` |
| `libero_10` | `82.6%` | `413` | `500` |

## libero_goal

- Suite 成功率: `92.6%` (`463/500`)

| Task | Success Rate | Successes | Episodes | Description |
| --- | --- | --- | --- | --- |
| `0` | `90.0%` | `45` | `50` | open the middle drawer of the cabinet |
| `1` | `98.0%` | `49` | `50` | put the bowl on the stove |
| `2` | `98.0%` | `49` | `50` | put the wine bottle on top of the cabinet |
| `3` | `86.0%` | `43` | `50` | open the top drawer and put the bowl inside |
| `4` | `100.0%` | `50` | `50` | put the bowl on top of the cabinet |
| `5` | `90.0%` | `45` | `50` | push the plate to the front of the stove |
| `6` | `92.0%` | `46` | `50` | put the cream cheese in the bowl |
| `7` | `100.0%` | `50` | `50` | turn on the stove |
| `8` | `94.0%` | `47` | `50` | put the bowl on the plate |
| `9` | `78.0%` | `39` | `50` | put the wine bottle on the rack |

## libero_10

- Suite 成功率: `82.6%` (`413/500`)

| Task | Success Rate | Successes | Episodes | Description |
| --- | --- | --- | --- | --- |
| `0` | `84.0%` | `42` | `50` | put both the alphabet soup and the tomato sauce in the basket |
| `1` | `98.0%` | `49` | `50` | put both the cream cheese box and the butter in the basket |
| `2` | `78.0%` | `39` | `50` | turn on the stove and put the moka pot on it |
| `3` | `96.0%` | `48` | `50` | put the black bowl in the bottom drawer of the cabinet and close it |
| `4` | `86.0%` | `43` | `50` | put the white mug on the left plate and put the yellow and white mug on the right plate |
| `5` | `94.0%` | `47` | `50` | pick up the book and place it in the back compartment of the caddy |
| `6` | `82.0%` | `41` | `50` | put the white mug on the plate and put the chocolate pudding to the right of the plate |
| `7` | `100.0%` | `50` | `50` | put both the alphabet soup and the cream cheese box in the basket |
| `8` | `26.0%` | `13` | `50` | put both moka pots on the stove |
| `9` | `82.0%` | `41` | `50` | put the yellow and white mug in the microwave and close it |

