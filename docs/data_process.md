[RoboTwin env.get_obs()]
        |
        v
[OpenPI infer_chunk(obs, prompt)]
  output: base_chunk (H x 14)
        |
        v
[for t in 0..H-1]  # 同一个 base_chunk 内循环
  |
  +--> [build_residual_step_obs(obs_t, base_action_t)]
  |      -> cam_high            (1,H,W,3)
  |      -> cam_left_wrist      (1,H,W,3)
  |      -> cam_right_wrist     (1,H,W,3)
  |      -> state               (1,28)    # 14 + 14 (joint + base_action_t)
  |
  +--> [agent.sample_actions(obs_input_t)]  # 每步推理一次残差
  |      -> residual_step_action (action_dim,)
  |
  +--> [compose_residual_action(base_chunk[t], residual_step_action)]
  |      -> final_action_t (14,)
  |
  +--> [env.step(final_action_t)]
  |      -> (obs_{t+1}, reward_t, done_t)
  |
  +--> [build next_obs_input_t]
  |      done=1: zero obs, mask=0
  |      done=0 & t<H-1: 使用当前 chunk 的下一步 base_action, mask=1
  |      done=0 & t=H-1: 预取下一次 OpenPI chunk 的第 0 步 base_action, mask=1
  |
  +--> [ReplayBuffer.insert(step transition)]
         observations=obs_input_t
         actions=residual_step_action (action_dim,)
         next_observations=next_obs_input_t
         rewards=reward_t
         masks=mask_t
         dones=done_t
        |
        v
[agent.update_high_utd(batch)]  # DrQ: random crop + SAC updates
