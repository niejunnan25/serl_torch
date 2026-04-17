# LIBERO Thin Train Loop V3

## Modification Summary

- Added [serl_launcher/serl_launcher/residual/runtime/learner_service.py](/home/hello/codebase/serl_torch/serl_launcher/serl_launcher/residual/runtime/learner_service.py) to hold the standalone learner runtime logic.
- Slimmed [examples/libero/scripts/run_learner.py](/home/hello/codebase/serl_torch/examples/libero/scripts/run_learner.py) into a thin entrypoint that now only:
  - builds run context
  - sets seeds
  - passes LIBERO-specific `data_config` and image-key resolution into the learner service
- Kept the learner service generic over:
  - `data_config`
  - `resolve_cfg_image_keys`
  instead of hard-importing LIBERO-specific config inside `serl_launcher`.

## Review Notes

- `py_compile` passed for:
  - [examples/libero/scripts/run_learner.py](/home/hello/codebase/serl_torch/examples/libero/scripts/run_learner.py)
  - [serl_launcher/serl_launcher/residual/runtime/learner_service.py](/home/hello/codebase/serl_torch/serl_launcher/serl_launcher/residual/runtime/learner_service.py)
  - plus the already-split actor-side files
- The learner entry script is now 48 lines.
- `learner_service.py` no longer has Hydra or LIBERO-specific imports baked into the service layer itself.

## Commit Summary

- Theme: make the learner side match the actor side structurally.
- Result:
  - both `train_residual_sac.py` and `run_learner.py` are now thin entrypoints
  - the heavy runtime logic now lives under `serl_launcher/residual/runtime`
