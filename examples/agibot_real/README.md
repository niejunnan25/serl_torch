# AgiBot Real Residual RL

`examples/agibot_real` now has one canonical residual-RL training flow:

- config: [configs/train_residual.yaml](/vla/users/niejunnan/codebase/serl_torch/examples/agibot_real/configs/train_residual.yaml)
- entrypoint: [scripts/run_residual_training.py](/vla/users/niejunnan/codebase/serl_torch/examples/agibot_real/scripts/run_residual_training.py)

The training topology matches `examples/libero`:

- one typed config parser in [config.py](/vla/users/niejunnan/codebase/serl_torch/examples/agibot_real/config.py)
- one actor/learner script, selected by `runtime.role=actor|learner`
- observation and policy-input logic owned by this example:
  - [env/observation.py](/vla/users/niejunnan/codebase/serl_torch/examples/agibot_real/env/observation.py)
  - [env/policy_input.py](/vla/users/niejunnan/codebase/serl_torch/examples/agibot_real/env/policy_input.py)
  - [residual_observation.py](/vla/users/niejunnan/codebase/serl_torch/examples/agibot_real/residual_observation.py)

The old Agentlace split-training and eval stack has been removed. This README documents the only supported mainline for now. A new canonical eval entrypoint has not been added yet.

**Scope**

This example is for real-robot residual RL on AgiBot `camera_position` control.

- `env.backend` is local-only
- `env.action_dim` must be `14`
- base-policy images are `head + left wrist + right wrist`
- OpenPI uses 14D `state/pose`
- JoyRA can additionally consume 18D `pose + head + waist`

The residual learner trains on chunk-conditioned observations:

- `robot_proprio`
- `base_action`
- `base_action_chunk`
- `alpha`
- `image_rgb_0`
- `image_rgb_1`
- `image_rgb_2`

**Prereqs**

The current code assumes:

- `agentlace` is importable
- `serl_launcher` is importable
- the repo parent is importable as `serl_torch`
- the robot runtime environment has the AgiBot SDK dependencies available
- the robot-side Python environment is compatible with the vendored SDK wheels in `vendor/a2d_sdk/`

If you are not running from an editable install, export:

```bash
export PYTHONPATH=/vla/users/niejunnan/codebase:/vla/users/niejunnan/codebase/serl_torch/serl_launcher:$PYTHONPATH
```

**Canonical Config**

The canonical config is [configs/train_residual.yaml](/vla/users/niejunnan/codebase/serl_torch/examples/agibot_real/configs/train_residual.yaml).

Important defaults:

- `policy.type=openpi`
- `policy.host=127.0.0.1`
- `policy.port=30001`
- `residual.chunk_horizon=50`
- `controller.enabled=true`

To switch to JoyRA, override at launch:

```bash
policy.type=joyra policy.port=9001
```

**Startup Order**

1. Start the base-policy server.
2. Start AgiBot robot-service.
3. Start the learner.
4. Start the actor on the robot terminal.

**Base Policy Server**

OpenPI wrapper:

```bash
cd /vla/users/niejunnan/codebase/serl_torch/examples/agibot_real
OPENPI_ROOT=/path/to/openpi \
POLICY_DIR=/path/to/policy/checkpoint \
bash tools/serve_openpi.sh --port 30001
```

JoyRA wrapper:

```bash
cd /vla/users/niejunnan/codebase/serl_torch/examples/agibot_real
JOYRA_ROOT=/path/to/JoyRA \
JOYRA_CKPT_PATH=/path/to/checkpoints/steps_xxx.pt \
bash tools/serve_joyra.sh --port 9001
```

**Robot Service**

If the forwarder/runtime assets are not prepared yet:

```bash
cd /vla/users/niejunnan/codebase/serl_torch/examples/agibot_real
bash tools/prepare_robot_runtime.sh --from-dir /path/to/forwarder
```

Then start robot-service:

```bash
cd /vla/users/niejunnan/codebase/serl_torch/examples/agibot_real
source robot_service/env.sh
python scripts/services/start_robot_service.py -s -c robot_service/conf/copilot.pbtxt
```

**Learner**

```bash
cd /vla/users/niejunnan/codebase/serl_torch/examples/agibot_real
python scripts/run_residual_training.py runtime.role=learner
```

Equivalent wrapper:

```bash
bash tools/run_learner.sh
```

Common overrides:

```bash
python scripts/run_residual_training.py \
  runtime.role=learner \
  wandb.project=agibot_real \
  training.max_update_steps=300000 \
  training.checkpoint.dir=checkpoints
```

**Actor**

```bash
cd /vla/users/niejunnan/codebase/serl_torch/examples/agibot_real
python scripts/run_residual_training.py runtime.role=actor
```

Equivalent wrapper:

```bash
bash tools/run_actor.sh
```

Useful overrides:

```bash
python scripts/run_residual_training.py \
  runtime.role=actor \
  policy.type=openpi \
  policy.host=127.0.0.1 \
  policy.port=30001 \
  task.name=agibot_real_default \
  task.prompt='Pick up the object with the right hand and place it at the target location.'
```

If you want human gating before each episode, keep `controller.enabled=true`. The default terminal keys are:

- `g`: ready
- `p`: pause
- `r`: reset
- `s`: success
- `f`: fail
- `h`: help

If you want expert precheck before reset:

```bash
python scripts/run_residual_training.py runtime.role=actor training.expert_check=true
```

**Outputs**

Hydra writes runs under:

```text
outputs/agibot_real/train_residual/<timestamp>/
```

Typical contents:

- `summary.json`
- `checkpoints/`
- `wandb/`

**Implementation Notes**

The canonical training path does not use these older abstractions:

- `serl_launcher.policy.factory`
- `serl_launcher.agents.continuous.drq_config`
- `serl_launcher.residual.observation`
- `serl_launcher.residual.train.*`
- `serl_launcher.residual.runtime_agent`

The new entrypoint uses:

- `serl_launcher.agents.continuous.drq_typed_config`
- `serl_launcher.policy.typed_factory`
- `serl_launcher.residual.typed_action`
- `MemoryEfficientStepWindowReplayBufferDataStore`

This is the implementation shape that new AgiBot work should follow.

Default shell wrappers now follow the canonical mainline:

- `tools/run_actor.sh`
- `tools/run_learner.sh`

Canonical eval has not been ported yet. Until that lands, this example only documents the new training path.
