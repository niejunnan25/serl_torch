# AgiBot Main Training Startup

The copy training behavior is now folded into the main config:

- Config: [../configs/train_residual.yaml](../configs/train_residual.yaml)
- Actor/learner wrapper: [../tools/run_actor.sh](../tools/run_actor.sh) and [../tools/run_learner.sh](../tools/run_learner.sh)
- Script entry: [../scripts/run_residual_training.py](../scripts/run_residual_training.py)

Use Hydra overrides directly on the main config, for example:

```bash
bash examples/agibot_real/tools/run_learner.sh runtime.role=learner
bash examples/agibot_real/tools/run_actor.sh runtime.role=actor
```

Outputs are written under `output` inside `examples/agibot_real`.
