# AgiBot Chunk Boundary Startup

The separate chunk-boundary yaml has been retired. Use the main training config instead:

- [../configs/train_residual.yaml](../configs/train_residual.yaml)

The default wrappers load that config:

```bash
bash examples/agibot_real/tools/run_learner.sh runtime.role=learner
bash examples/agibot_real/tools/run_actor.sh runtime.role=actor
```

Outputs are written under `output` inside `examples/agibot_real`.
