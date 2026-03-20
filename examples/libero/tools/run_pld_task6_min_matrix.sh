#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

CONFIGS=(
  train_pld_task6_m01_expert_w0_xi05
  train_pld_task6_m02_expert_w50_xi05
  train_pld_task6_m03_expert_w100_xi05
  train_pld_task6_m04_boot50_w0_xi05
  train_pld_task6_m05_boot50_w50_xi05
  train_pld_task6_m06_boot50_w100_xi05
  train_pld_task6_m07_boot50_w100_xi03
  train_pld_task6_m08_boot50_w100_xi02
)

if [[ "${1:-}" == "--dry-run" ]]; then
  DRY_RUN=1
  shift
else
  DRY_RUN=0
fi

if [[ $# -eq 0 ]]; then
  GPUS=(0 1 2 3 4 5 6 7)
else
  IFS=',' read -r -a GPUS <<<"$1"
fi

if [[ "${#GPUS[@]}" -ne 8 ]]; then
  echo "ERROR: need exactly 8 GPU ids, e.g. --dry-run 0,1,2,3,4,5,6,7"
  exit 1
fi

STAMP="$(date +%Y-%m-%d_%H-%M-%S)"
LAUNCH_DIR="$ROOT_DIR/outputs/libero/pld_matrix/launch_logs/${STAMP}"
mkdir -p "$LAUNCH_DIR"

echo "Launch logs: $LAUNCH_DIR"
for i in "${!CONFIGS[@]}"; do
  cfg="${CONFIGS[$i]}"
  gpu="${GPUS[$i]}"
  log_file="$LAUNCH_DIR/${cfg}.log"
  cmd=(bash tools/run_train.sh "${cfg}.yaml" --gpu_id "$gpu")
  echo "[$((i+1))/8] GPU=${gpu} CFG=${cfg}"
  echo "  CMD: ${cmd[*]}"
  echo "  LOG: $log_file"
  if [[ "$DRY_RUN" -eq 0 ]]; then
    nohup "${cmd[@]}" >"$log_file" 2>&1 &
    echo "  PID: $!"
  fi
done

if [[ "$DRY_RUN" -eq 1 ]]; then
  echo "Dry run only. No training started."
else
  echo "All 8 jobs launched."
fi
