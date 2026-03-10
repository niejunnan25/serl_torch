#!/usr/bin/env bash
set -euo pipefail

# Paper-style Stage-1/2 repro runner:
# - 250k interaction steps
# - 3 seeds
# - mean + 95% CI aggregation

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

OUT_ROOT="${1:-outputs/paper_stage12_repro}"
SEEDS=(0 1 2)

for SEED in "${SEEDS[@]}"; do
  TRAIN_DIR="${OUT_ROOT}/train_seed${SEED}"
  EVAL_DIR="${OUT_ROOT}/eval_seed${SEED}"

  python scripts/train_residual_sac.py \
    seed="${SEED}" \
    task.seed_base="$((100000 + SEED * 10000))" \
    training.max_online_env_steps=250000 \
    hydra.run.dir="${TRAIN_DIR}"

  CKPT_PATH="$(ls -1 "${TRAIN_DIR}"/checkpoints/checkpoint_*.pt | sort -V | tail -n 1)"

  python scripts/eval_residual_fast.py \
    seed="${SEED}" \
    task.seed_base="$((200000 + SEED * 10000))" \
    eval.checkpoint_path="${CKPT_PATH}" \
    eval.episodes=50 \
    hydra.run.dir="${EVAL_DIR}"
done

python scripts/aggregate_eval_ci.py \
  --eval-dirs "${OUT_ROOT}"/eval_seed0 "${OUT_ROOT}"/eval_seed1 "${OUT_ROOT}"/eval_seed2 \
  --out "${OUT_ROOT}/eval_ci95.json"
