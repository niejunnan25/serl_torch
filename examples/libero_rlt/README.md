# LIBERO RLT Runbook

This example runs RL-Token on LIBERO with `serl_torch` actor/learner, replay,
checkpoint, and async-eval infrastructure. OpenPI is treated as an external VLA
provider: pass an `openpi_root` checkout and checkpoint paths, but do not vendor
or modify OpenPI inside this repo.

## Components

Stage 1 trains only `RLTokenEncoder + RLTokenDecoder` on frozen OpenPI VLA
embeddings. Stage 2 freezes the VLA and Stage 1 encoder, then trains the RLT
actor/critic over `z_rl + reference_action`.

Runtime services for Stage 2:

- LIBERO env server, started through `examples/libero/tools/serve_env.sh`.
- OpenPI VLA feature server, started by `examples/libero_rlt/scripts/serve_vla_features.py`.
- Learner process, running `examples/libero_rlt/scripts/run_rlt_training.py`.
- Actor process, running the same script with `runtime.role=actor`.
- Optional async eval env/VLA servers and eval worker.

The feature server websocket output remains:

- `z_rl`: frozen Stage 1 encoder output.
- `reference_action`: unnormalized OpenPI reference action chunk.
- `proprio`: raw proprio slice for logging/schema compatibility.

## Required Artifacts

- External OpenPI RLT checkout with a compatible embedding extractor (`model.extract_embeddings`), for example `/vla/users/niejunnan/codebase/openpi-rlt-github`.
- OpenPI policy checkpoint compatible with the chosen `vla.config_name`, e.g. `pi0_libero`.
- Stage 1 RLT encoder checkpoint for Stage 2. Existing checkpoints with `encoder_state_dict` remain supported.
- LIBERO runtime environment.

Recommended OpenPI fork:

```bash
git clone https://github.com/niejunnan25/openpi.git /vla/users/niejunnan/codebase/openpi-rlt-github
cd /vla/users/niejunnan/codebase/openpi-rlt-github
git checkout codex/rlt-extract-embeddings
```

This fork keeps OpenPI as the VLA provider and only adds the
`PI0Pytorch.extract_embeddings()` hook required by RLT.

## Token Length Compatibility

Stage 1 may train the RLT encoder on a truncated prefix of VLA tokens. The
default config uses `rlt.max_tokens: 512`. New Stage 1 checkpoints written by
`train_rlt_stage1.py` store this value in `config.rlt.max_tokens`; Stage 2 then
automatically applies the same truncation before calling the frozen encoder.
This keeps Stage 1 and online Stage 2 feature extraction aligned even when the
OpenPI feature extractor returns a longer token sequence.

Legacy checkpoints from the older yixin OpenPI path usually do not contain the
RLT config. Those checkpoints were trained with the Stage 1 prefix truncated to
512 tokens, so pass `--rlt-max-tokens 512` when launching Stage 2 with them.
Passing `--rlt-max-tokens 0` explicitly disables truncation.

## Stage 1 Training

Run Stage 1 with a Python environment that has the OpenPI dependencies installed. On the development cluster this is usually an OpenPI `.venv`; the `serl_torch` conda env may not contain packages such as `numpydantic`. The `vla.openpi_root` checkout must expose `model.extract_embeddings()`. For smoke tests without a cached LeRobot LIBERO dataset, set `vla.repo_id_override=fake`; real training should leave it unset and set `vla.lerobot_home` to the local LeRobot cache root. On the development cluster, `/vla/users/niejunnan/datasets/libero` is exposed as `physical-intelligence/libero` through `HF_LEROBOT_HOME=/vla/users/niejunnan/datasets`.

Example smoke command:

```bash
cd /vla/users/niejunnan/codebase/serl_torch-rlt-stage1-adapter

/vla/users/niejunnan/codebase/openpi-modified/.venv/bin/python3 \
  examples/libero_rlt/scripts/train_rlt_stage1.py \
  --config examples/libero_rlt/configs/stage1_rlt.yaml \
  vla.openpi_root=/vla/users/niejunnan/codebase/openpi-rlt-github \
  vla.config_name=pi0_libero \
  vla.checkpoint_path=/vla/users/yixin/base_model/openpi-assets/checkpoints/pi0_libero_pytorch \
  vla.repo_id_override=fake \
  vla.lerobot_home=/vla/users/niejunnan/datasets \
  training.output_dir=/tmp/rlt_stage1_smoke_codex \
  training.steps=2 \
  training.batch_size=1 \
  training.num_workers=0
```



Real-data Stage 1 smoke:

```bash
/vla/users/niejunnan/codebase/openpi-modified/.venv/bin/python3 \
  examples/libero_rlt/scripts/train_rlt_stage1.py \
  --config examples/libero_rlt/configs/stage1_rlt.yaml \
  vla.openpi_root=/vla/users/niejunnan/codebase/openpi-rlt-github \
  vla.lerobot_home=/vla/users/niejunnan/datasets \
  vla.repo_id_override=null \
  training.output_dir=/tmp/rlt_stage1_real_smoke \
  training.steps=20 \
  training.batch_size=1 \
  training.num_workers=0
```

A normal run removes the smoke overrides and sets `training.output_dir` to the experiment directory. The checkpoint format is:

- `encoder_state_dict`
- `decoder_state_dict`
- `optimizer_state_dict`
- `scheduler_state_dict`
- `config`, including `rlt.input_dim`

Stage 2 can load either `final_model.pt`, a numbered `checkpoint_*.pt`, or an encoder-only state dict if the architecture can be inferred.

## Stage 2 Launch

Start a smoke run without async eval:

```bash
cd /vla/users/niejunnan/codebase/serl_torch-rlt-stage1-adapter

bash examples/libero_rlt/tools/launch_rlt_training.sh \
  --session libero_rlt_smoke \
  --config-name smoke_rlt \
  --gpu 0 \
  --learner-gpu 1 \
  --openpi-root /vla/users/niejunnan/codebase/openpi-rlt-github \
  --vla-config pi0_libero \
  --vla-checkpoint /path/to/pi0_libero_checkpoint \
  --rlt-encoder-path /path/to/rlt_stage1_checkpoint.pt \
  -- \
  rlt.warmup_steps=0 \
  training.max_env_steps=1000 \
  training.max_update_steps=1000
```

Backward-compatible aliases are still accepted:

- `--pi0-config` == `--vla-config`
- `--pi0-path` == `--vla-checkpoint`

For an older yixin Stage 1 checkpoint that lacks `config.rlt.max_tokens`, add:

```bash
  --rlt-max-tokens 512
```

Start with async eval:

```bash
bash examples/libero_rlt/tools/launch_rlt_training.sh \
  --session libero_rlt_eval_smoke \
  --config-name smoke_rlt \
  --gpu 0 \
  --learner-gpu 1 \
  --with-eval \
  --eval-gpu 2 \
  --env-port 21000 \
  --vla-port 8777 \
  --eval-env-port 21001 \
  --eval-vla-port 8877 \
  --openpi-root /vla/users/niejunnan/codebase/openpi-rlt-github \
  --vla-config pi0_libero \
  --vla-checkpoint /path/to/pi0_libero_checkpoint \
  --rlt-encoder-path /path/to/rlt_stage1_checkpoint.pt \
  -- \
  rlt.warmup_steps=0 \
  training.max_env_steps=300 \
  training.max_update_steps=300 \
  training.async_eval.enabled=true \
  training.async_eval.every_episodes=1 \
  training.async_eval.episodes=1 \
  training.async_eval.max_env_steps_per_episode=20
```

## Key Config Semantics

- `rlt.chunk_size`: number of actions predicted by the RLT actor.
- `rlt.execute_horizon`: number of predicted actions executed before replanning.
- Replay transitions store `executed_steps` and `discounts = gamma ** executed_steps`; the learner uses the stored `discounts` for TD bootstrap.
- Terminal transitions skip next-state VLA inference because terminal TD targets do not bootstrap.
- The actor and learner share one run directory. The learner owns Hydra metadata; the actor is launched with `hydra.output_subdir=null`.
- Async eval writes queue, summary, worker log, and eval checkpoints inside the run directory. `training.async_eval.checkpoint.keep <= 0` keeps all eval checkpoints.

## Runtime Outputs

- `summary.json`: learner terminal summary with update/env/replay counts.
- `episode_logs.jsonl`: actor episode summaries.
- `actor_timers.jsonl` / `learner_timers.jsonl`: lightweight timing records.
- `checkpoints/`: learner checkpoints.
- `eval_summary.jsonl`: async eval results when enabled.

## Review Scope

This integration pass covers LIBERO RLT, the OpenPI backend adapter, and Stage 1 encoder/decoder training. Real-robot AgiBot/JoyRA code remains outside this review unit and should be staged in a separate follow-up.
