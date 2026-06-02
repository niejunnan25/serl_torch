# RLT Migration Checklist

Date: 2026-06-02

Source reference: `/vla/users/yixin/5090`

Target branch: `codex/integrate-rlt-token`

## Migrated for LIBERO RLT

- RLT model core: `RLTokenEncoder`, `RLTokenDecoder`, `RLTActor`, and the
  chunk-level actor/critic agent wrapper.
- RLT observation schema: frozen `z_rl`, `proprio`, and `reference_action`
  fields, with actor/critic state using the RL token as in the source Stage 2
  path.
- VLA feature service: Pi0/OpenPI and frozen RLT encoder live behind a feature
  server; actor, learner, and eval consume compact features through the client.
- LIBERO actor/learner recipe: remote env, replay, checkpoint codec, Hydra
  config parsing, actor-finished signaling, and async eval integration.
- RLT Stage 2 loss shape: chunk-level critic target and actor objective
  `-Q + bc_reg_coeff * MSE(action, reference_action)`.
- Execution semantics: predicted chunk length and executed horizon are explicit
  config values, with TD bootstrap discount based on actual executed steps.

## Adapted to Current Infra

- Agent state is checkpoint-codec compatible with current `serl_torch`.
- Replay buffer accepts extra scalar fields for `discounts` and
  `executed_steps`.
- Async eval uses the shared queue/checkpoint/result helpers and a JSON config
  snapshot to avoid actor/learner Hydra metadata races.
- Launch uses tmux windows for env, VLA feature server, learner, actor, and
  optional eval env/VLA feature server.

## Deferred to Follow-Up

- `examples/agibot_real_rlt` and real-robot JoyRA/AgiBot handover or critical
  phase logic.
- Stage 1 RLT training beyond the reusable encoder/decoder/model definitions.
- Long-horizon convergence or paper-level reproduction experiments.
- Production watchdogs for actor crash detection; the current LIBERO path handles
  normal actor completion.

## Review Boundary

The first review should stage only LIBERO RLT and shared infrastructure needed by
that route. The untracked `examples/agibot_real_rlt` files should remain
unstaged until the real-robot path is normalized against the same interfaces.
