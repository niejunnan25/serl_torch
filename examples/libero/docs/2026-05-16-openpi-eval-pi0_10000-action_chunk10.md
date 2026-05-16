# pi0_10000 LIBERO Eval Report (action_chunk=10)

生成时间（北京时间）: 2026-05-16 08:03:32 CST

## 评测配置

- 策略名称: `pi0_10000`
- Action chunk: `10`
- 执行方式: `10推10`
- 服务器: `root@116.198.45.239`
- Policy GPU: `6`
- Env GPU: `7`
- Policy script: `/vla/users/niejunnan/codebase/serl_torch/examples/libero/tools/serve_openpi_10000_policy.sh`
- Policy port: `41061`
- Policy config: `pi0_libero_baseline_10_bs32_150000`
- Policy dir: `/vla/users/niejunnan/assets/openpi-assets/serl_torch_ckpt/pi0_10000`
- 结果目录: `/vla/users/niejunnan/codebase/serl_torch/examples/libero/outputs/openpi_eval_pi0_10000_10push10/all4_suites_gpu6_server_gpu7_env`
- 结果日志: `/vla/users/niejunnan/codebase/serl_torch/examples/libero/outputs/openpi_eval_pi0_10000_10push10/all4_suites_gpu6_server_gpu7_env/20260516_031004_spatial_object_goal_10_50_65.30.log`
- 套件数: `4`
- 套件列表: `libero_spatial, libero_object, libero_goal, libero_10`
- 每个 task episode 数: `50`

## 总览

- 总 episode 数: `2000`
- 总成功数: `1306`
- 加权总成功率: `65.30%`
- Suite 平均成功率: `65.30%`

| Suite | Success Rate | Successes | Episodes | Duration |
| --- | --- | --- | --- | --- |
| `libero_spatial` | `66.00%` | `330` | `500` | `58m 40s` |
| `libero_object` | `88.80%` | `444` | `500` | `49m 41s` |
| `libero_goal` | `61.80%` | `309` | `500` | `56m 51s` |
| `libero_10` | `44.60%` | `223` | `500` | `1h 57m 3s` |

## libero_spatial

- Suite 成功率: `66.00%` (`330/500`)

| Task | Success Rate | Successes | Episodes | Description |
| --- | --- | --- | --- | --- |
| `0` | `70.0%` | `35` | `50` | pick up the black bowl between the plate and the ramekin and place it on the plate |
| `1` | `62.0%` | `31` | `50` | pick up the black bowl next to the ramekin and place it on the plate |
| `2` | `90.0%` | `45` | `50` | pick up the black bowl from table center and place it on the plate |
| `3` | `72.0%` | `36` | `50` | pick up the black bowl on the cookie box and place it on the plate |
| `4` | `60.0%` | `30` | `50` | pick up the black bowl in the top drawer of the wooden cabinet and place it on the plate |
| `5` | `58.0%` | `29` | `50` | pick up the black bowl on the ramekin and place it on the plate |
| `6` | `68.0%` | `34` | `50` | pick up the black bowl next to the cookie box and place it on the plate |
| `7` | `54.0%` | `27` | `50` | pick up the black bowl on the stove and place it on the plate |
| `8` | `60.0%` | `30` | `50` | pick up the black bowl next to the plate and place it on the plate |
| `9` | `66.0%` | `33` | `50` | pick up the black bowl on the wooden cabinet and place it on the plate |

## libero_object

- Suite 成功率: `88.80%` (`444/500`)

| Task | Success Rate | Successes | Episodes | Description |
| --- | --- | --- | --- | --- |
| `0` | `82.0%` | `41` | `50` | pick up the alphabet soup and place it in the basket |
| `1` | `82.0%` | `41` | `50` | pick up the cream cheese and place it in the basket |
| `2` | `94.0%` | `47` | `50` | pick up the salad dressing and place it in the basket |
| `3` | `84.0%` | `42` | `50` | pick up the bbq sauce and place it in the basket |
| `4` | `96.0%` | `48` | `50` | pick up the ketchup and place it in the basket |
| `5` | `76.0%` | `38` | `50` | pick up the tomato sauce and place it in the basket |
| `6` | `92.0%` | `46` | `50` | pick up the butter and place it in the basket |
| `7` | `96.0%` | `48` | `50` | pick up the milk and place it in the basket |
| `8` | `98.0%` | `49` | `50` | pick up the chocolate pudding and place it in the basket |
| `9` | `88.0%` | `44` | `50` | pick up the orange juice and place it in the basket |

## libero_goal

- Suite 成功率: `61.80%` (`309/500`)

| Task | Success Rate | Successes | Episodes | Description |
| --- | --- | --- | --- | --- |
| `0` | `32.0%` | `16` | `50` | open the middle drawer of the cabinet |
| `1` | `90.0%` | `45` | `50` | put the bowl on the stove |
| `2` | `92.0%` | `46` | `50` | put the wine bottle on top of the cabinet |
| `3` | `16.0%` | `8` | `50` | open the top drawer and put the bowl inside |
| `4` | `90.0%` | `45` | `50` | put the bowl on top of the cabinet |
| `5` | `46.0%` | `23` | `50` | push the plate to the front of the stove |
| `6` | `66.0%` | `33` | `50` | put the cream cheese in the bowl |
| `7` | `92.0%` | `46` | `50` | turn on the stove |
| `8` | `78.0%` | `39` | `50` | put the bowl on the plate |
| `9` | `16.0%` | `8` | `50` | put the wine bottle on the rack |

## libero_10

- Suite 成功率: `44.60%` (`223/500`)

| Task | Success Rate | Successes | Episodes | Description |
| --- | --- | --- | --- | --- |
| `0` | `60.0%` | `30` | `50` | put both the alphabet soup and the tomato sauce in the basket |
| `1` | `92.0%` | `46` | `50` | put both the cream cheese box and the butter in the basket |
| `2` | `44.0%` | `22` | `50` | turn on the stove and put the moka pot on it |
| `3` | `48.0%` | `24` | `50` | put the black bowl in the bottom drawer of the cabinet and close it |
| `4` | `54.0%` | `27` | `50` | put the white mug on the left plate and put the yellow and white mug on the right plate |
| `5` | `14.0%` | `7` | `50` | pick up the book and place it in the back compartment of the caddy |
| `6` | `54.0%` | `27` | `50` | put the white mug on the plate and put the chocolate pudding to the right of the plate |
| `7` | `72.0%` | `36` | `50` | put both the alphabet soup and the cream cheese box in the basket |
| `8` | `0.0%` | `0` | `50` | put both moka pots on the stove |
| `9` | `8.0%` | `4` | `50` | put the yellow and white mug in the microwave and close it |
