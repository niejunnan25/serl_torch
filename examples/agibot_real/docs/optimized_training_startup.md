# AgiBot Optimized Training Startup

This separate optimized yaml has been retired. The optimized/copy behavior now lives in the main training config:

- [../configs/train_residual.yaml](../configs/train_residual.yaml)

The default wrappers load that config:

```bash
bash examples/agibot_real/tools/run_learner.sh runtime.role=learner
bash examples/agibot_real/tools/run_actor.sh runtime.role=actor
```

Outputs are written under `output` inside `examples/agibot_real`.
