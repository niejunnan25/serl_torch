# spatial_4 key_rl 消融实验

这些配置复用 `spatial_4_0514_runtime` 的基础训练设置，只修改 KeyRL gate、W&B 名称、服务端口和输出目录。

当前实验矩阵：

- `alpha`: `0.1`, `0.2`, `0.5`
- `sac.std_max`: `0.5`, `1.0`
- 窗口类型:
  - `single_stage_30_75`: 只在 `[30,75)` 启用 residual
  - `two_stage_30_75_110_160`: 在 `[30,75)` 和 `[110,160)` 启用 residual

总共 12 个配置：

1. `spatial4_keyrl_alpha0p1_std0p5_single_stage_30_75`
2. `spatial4_keyrl_alpha0p1_std0p5_two_stage_30_75_110_160`
3. `spatial4_keyrl_alpha0p1_std1p0_single_stage_30_75`
4. `spatial4_keyrl_alpha0p1_std1p0_two_stage_30_75_110_160`
5. `spatial4_keyrl_alpha0p2_std0p5_single_stage_30_75`
6. `spatial4_keyrl_alpha0p2_std0p5_two_stage_30_75_110_160`
7. `spatial4_keyrl_alpha0p2_std1p0_single_stage_30_75`
8. `spatial4_keyrl_alpha0p2_std1p0_two_stage_30_75_110_160`
9. `spatial4_keyrl_alpha0p5_std0p5_single_stage_30_75`
10. `spatial4_keyrl_alpha0p5_std0p5_two_stage_30_75_110_160`
11. `spatial4_keyrl_alpha0p5_std1p0_single_stage_30_75`
12. `spatial4_keyrl_alpha0p5_std1p0_two_stage_30_75_110_160`

已有 full prepared offline data 会被复用；KeyRL active-only 过滤发生在 load time。

完整实验动机、实验矩阵、已运行/待运行映射和分析计划见 `spatial4_keyrl_experiment_note.md`。
