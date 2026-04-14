# pi0_libero LIBERO Eval Report

生成时间（北京时间）: 2026-04-14 03:52:31 CST

## 评测配置

- 策略名称: `pi0_libero`
- GPU: `6`
- Policy script: `/vla/users/niejunnan/codebase/serl_torch/examples/libero/tools/serve_openpi_policy.sh`
- Policy port: `41002`
- 结果目录: `/vla/users/niejunnan/codebase/serl_torch/outputs/openpi_libero_eval/2026-04-13_22-46-33/pi0_libero`
- 套件数: `4`
- 套件列表: `libero_spatial, libero_object, libero_goal, libero_10`

## 总览

- 总 episode 数: `2000`
- 总成功数: `1841`
- 加权总成功率: `92.0%`
- Suite 平均成功率: `92.0%`

| Suite | Success Rate | Successes | Episodes |
| --- | --- | --- | --- |
| `libero_spatial` | `98.0%` | `490` | `500` |
| `libero_object` | `98.2%` | `491` | `500` |
| `libero_goal` | `92.2%` | `461` | `500` |
| `libero_10` | `79.8%` | `399` | `500` |

## libero_spatial

- Suite 成功率: `98.0%` (`490/500`)

| Task | Success Rate | Successes | Episodes | Description |
| --- | --- | --- | --- | --- |
| `0` | `98.0%` | `49` | `50` | pick up the black bowl between the plate and the ramekin and place it on the plate |
| `1` | `100.0%` | `50` | `50` | pick up the black bowl next to the ramekin and place it on the plate |
| `2` | `100.0%` | `50` | `50` | pick up the black bowl from table center and place it on the plate |
| `3` | `100.0%` | `50` | `50` | pick up the black bowl on the cookie box and place it on the plate |
| `4` | `94.0%` | `47` | `50` | pick up the black bowl in the top drawer of the wooden cabinet and place it on the plate |
| `5` | `96.0%` | `48` | `50` | pick up the black bowl on the ramekin and place it on the plate |
| `6` | `100.0%` | `50` | `50` | pick up the black bowl next to the cookie box and place it on the plate |
| `7` | `96.0%` | `48` | `50` | pick up the black bowl on the stove and place it on the plate |
| `8` | `98.0%` | `49` | `50` | pick up the black bowl next to the plate and place it on the plate |
| `9` | `98.0%` | `49` | `50` | pick up the black bowl on the wooden cabinet and place it on the plate |

## libero_object

- Suite 成功率: `98.2%` (`491/500`)

| Task | Success Rate | Successes | Episodes | Description |
| --- | --- | --- | --- | --- |
| `0` | `94.0%` | `47` | `50` | pick up the alphabet soup and place it in the basket |
| `1` | `100.0%` | `50` | `50` | pick up the cream cheese and place it in the basket |
| `2` | `100.0%` | `50` | `50` | pick up the salad dressing and place it in the basket |
| `3` | `100.0%` | `50` | `50` | pick up the bbq sauce and place it in the basket |
| `4` | `94.0%` | `47` | `50` | pick up the ketchup and place it in the basket |
| `5` | `98.0%` | `49` | `50` | pick up the tomato sauce and place it in the basket |
| `6` | `100.0%` | `50` | `50` | pick up the butter and place it in the basket |
| `7` | `100.0%` | `50` | `50` | pick up the milk and place it in the basket |
| `8` | `100.0%` | `50` | `50` | pick up the chocolate pudding and place it in the basket |
| `9` | `96.0%` | `48` | `50` | pick up the orange juice and place it in the basket |

## libero_goal

- Suite 成功率: `92.2%` (`461/500`)

| Task | Success Rate | Successes | Episodes | Description |
| --- | --- | --- | --- | --- |
| `0` | `96.0%` | `48` | `50` | open the middle drawer of the cabinet |
| `1` | `96.0%` | `48` | `50` | put the bowl on the stove |
| `2` | `98.0%` | `49` | `50` | put the wine bottle on top of the cabinet |
| `3` | `86.0%` | `43` | `50` | open the top drawer and put the bowl inside |
| `4` | `94.0%` | `47` | `50` | put the bowl on top of the cabinet |
| `5` | `90.0%` | `45` | `50` | push the plate to the front of the stove |
| `6` | `90.0%` | `45` | `50` | put the cream cheese in the bowl |
| `7` | `98.0%` | `49` | `50` | turn on the stove |
| `8` | `98.0%` | `49` | `50` | put the bowl on the plate |
| `9` | `76.0%` | `38` | `50` | put the wine bottle on the rack |

## libero_10

- Suite 成功率: `79.8%` (`399/500`)

| Task | Success Rate | Successes | Episodes | Description |
| --- | --- | --- | --- | --- |
| `0` | `84.0%` | `42` | `50` | put both the alphabet soup and the tomato sauce in the basket |
| `1` | `100.0%` | `50` | `50` | put both the cream cheese box and the butter in the basket |
| `2` | `78.0%` | `39` | `50` | turn on the stove and put the moka pot on it |
| `3` | `94.0%` | `47` | `50` | put the black bowl in the bottom drawer of the cabinet and close it |
| `4` | `80.0%` | `40` | `50` | put the white mug on the left plate and put the yellow and white mug on the right plate |
| `5` | `96.0%` | `48` | `50` | pick up the book and place it in the back compartment of the caddy |
| `6` | `82.0%` | `41` | `50` | put the white mug on the plate and put the chocolate pudding to the right of the plate |
| `7` | `96.0%` | `48` | `50` | put both the alphabet soup and the cream cheese box in the basket |
| `8` | `14.0%` | `7` | `50` | put both moka pots on the stove |
| `9` | `74.0%` | `37` | `50` | put the yellow and white mug in the microwave and close it |

