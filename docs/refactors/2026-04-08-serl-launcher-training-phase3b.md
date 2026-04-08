# SERL Launcher Training Phase 3B

## Goal

Split the mixed loop and telemetry helpers into generic training helpers and
residual-train-specific helpers.

## New Modules

### Training

- `serl_launcher/serl_launcher/training/loop_utils.py`
  - generic env-step/update scheduling helpers
- `serl_launcher/serl_launcher/training/telemetry.py`
  - generic scalar logging helper

### Residual Train

- `serl_launcher/serl_launcher/residual/train/transitions.py`
  - residual online-transition insertion and chunk packing helpers
- `serl_launcher/serl_launcher/residual/train/telemetry.py`
  - residual step-window and update TensorBoard helpers

## Removed Modules

- `serl_launcher/serl_launcher/residual/runtime/train_loop_utils.py`
- `serl_launcher/serl_launcher/residual/runtime/tb_metrics.py`

## Import Updates

Updated:

- `agentlace_bridge.py` to use `training.loop_utils`
- `pretrain.py` to use `training.telemetry`
- actor loop / warmup / episode files to use
  `training.loop_utils`, `residual.train.transitions`, and
  `residual.train.telemetry`

## Review

I checked:

- `python -m py_compile` on the new loop/telemetry modules and updated call sites
- `conda run -n serl_torch python examples/libero/scripts/train_residual_sac.py --help`
- `conda run -n serl_torch python examples/libero/scripts/run_learner.py --help`
- `git diff --check`
- `rg` to confirm there are no remaining Python imports of
  `serl_launcher.residual.runtime.train_loop_utils` or
  `serl_launcher.residual.runtime.tb_metrics`

## Outcome

After this step, `serl_launcher.residual.runtime` is reduced to the async and
agentlace-facing mixed infrastructure files:

- `agentlace_bridge.py`
- `async_learning.py`

The scheduling and telemetry helpers now live on the intended sides of the
package boundary.
