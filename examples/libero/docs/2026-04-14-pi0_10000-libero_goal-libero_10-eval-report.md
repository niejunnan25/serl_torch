# pi0_10000 LIBERO Eval Report

生成时间（北京时间）: 2026-04-14 02:57:45 CST

## 评测配置

- 策略名称: `pi0_10000`
- GPU: `1`
- Policy script: `/home/hello/codebase/serl_torch/examples/libero/tools/serve_openpi_10000_policy.sh`
- Policy port: `42001`
- 结果目录: `/home/hello/codebase/serl_torch/outputs/openpi_libero_eval/2026-04-13_23-52-56_goal10_gpu1_gpu2/pi0_10000`
- 套件数: `2`
- 套件列表: `libero_goal, libero_10`

## 总览

- 总 episode 数: `1000`
- 总成功数: `608`
- 加权总成功率: `60.8%`
- Suite 平均成功率: `60.8%`

| Suite | Success Rate | Successes | Episodes |
| --- | --- | --- | --- |
| `libero_goal` | `67.4%` | `337` | `500` |
| `libero_10` | `54.2%` | `271` | `500` |

## libero_goal

- Suite 成功率: `67.4%` (`337/500`)

| Task | Success Rate | Successes | Episodes | Description |
| --- | --- | --- | --- | --- |
| `0` | `40.0%` | `20` | `50` | open the middle drawer of the cabinet |
| `1` | `100.0%` | `50` | `50` | put the bowl on the stove |
| `2` | `88.0%` | `44` | `50` | put the wine bottle on top of the cabinet |
| `3` | `12.0%` | `6` | `50` | open the top drawer and put the bowl inside |
| `4` | `88.0%` | `44` | `50` | put the bowl on top of the cabinet |
| `5` | `60.0%` | `30` | `50` | push the plate to the front of the stove |
| `6` | `72.0%` | `36` | `50` | put the cream cheese in the bowl |
| `7` | `94.0%` | `47` | `50` | turn on the stove |
| `8` | `90.0%` | `45` | `50` | put the bowl on the plate |
| `9` | `30.0%` | `15` | `50` | put the wine bottle on the rack |

## libero_10

- Suite 成功率: `54.2%` (`271/500`)

| Task | Success Rate | Successes | Episodes | Description |
| --- | --- | --- | --- | --- |
| `0` | `62.0%` | `31` | `50` | put both the alphabet soup and the tomato sauce in the basket |
| `1` | `96.0%` | `48` | `50` | put both the cream cheese box and the butter in the basket |
| `2` | `56.0%` | `28` | `50` | turn on the stove and put the moka pot on it |
| `3` | `74.0%` | `37` | `50` | put the black bowl in the bottom drawer of the cabinet and close it |
| `4` | `68.0%` | `34` | `50` | put the white mug on the left plate and put the yellow and white mug on the right plate |
| `5` | `34.0%` | `17` | `50` | pick up the book and place it in the back compartment of the caddy |
| `6` | `66.0%` | `33` | `50` | put the white mug on the plate and put the chocolate pudding to the right of the plate |
| `7` | `84.0%` | `42` | `50` | put both the alphabet soup and the cream cheese box in the basket |
| `8` | `0.0%` | `0` | `50` | put both moka pots on the stove |
| `9` | `2.0%` | `1` | `50` | put the yellow and white mug in the microwave and close it |

