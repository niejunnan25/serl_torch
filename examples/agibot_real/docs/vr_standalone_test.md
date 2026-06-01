# VR Standalone Test

Use these scripts to verify the Quest VR controller and AgiBot EE teleop before
running residual RL.

Right hand:

```bash
cd /path/to/serl_torch
bash examples/agibot_real/tools/test_right_vr_base_pose_so3_grip_local_smooth.sh
```

Left hand:

```bash
cd /path/to/serl_torch
bash examples/agibot_real/tools/test_left_vr_base_pose_so3_grip_local_smooth.sh
```

The scripts use the copied smooth calibration at:

```text
examples/agibot_real/assets/vr_calibration/vr_robot_rotation_calibration_right_smooth.json
```

Default controls:

- Hold `RG` for right-hand control, or `LG` for left-hand control.
- The scripts execute robot commands and show live target/actual plots.
- Press `s` in the plot window to save the full plot animation.
- Use `Ctrl-C` to stop.
