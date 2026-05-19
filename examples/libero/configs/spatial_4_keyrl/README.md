# spatial_4 key_rl ablations

These configs reuse the `spatial_4_0514_runtime` settings and only change the
KeyRL gate, W&B names, and output roots.

The first-pass ablation is intentionally small:

1. `spatial4_keyrl_alpha0p2_single_stage_30_75`: single-stage baseline with residual only around the first interaction window.
2. `spatial4_keyrl_alpha0p2_two_stage_30_75_110_160`: primary two-stage hypothesis.
3. `spatial4_keyrl_alpha0p5_single_stage_30_75`: higher residual scale single-stage run.
4. `spatial4_keyrl_alpha0p5_two_stage_30_75_110_160`: higher residual scale two-stage run.

The recommended first run is the `spatial4_keyrl_alpha0p2_two_stage_30_75_110_160` config. Existing
full prepared offline data is reused; KeyRL active-only filtering happens at load time.
