#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
REPO_DIR="$(cd "$ROOT_DIR/.." && pwd)"
cd "$REPO_DIR"

SKIP_GPU_CHECK=0
TARGET_SUCCESSES=50
POLICY_SCRIPT="examples/libero/tools/serve_openpi_policy.sh"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --no-gpu-check) SKIP_GPU_CHECK=1; shift ;;
    --target-successes) TARGET_SUCCESSES="$2"; shift 2 ;;
    --policy-script) POLICY_SCRIPT="$2"; shift 2 ;;
    -h|--help)
      cat <<'EOF'
Usage:
  bash examples/libero_pld/tools/launch_libero90_pi0_libero_8tasks_300k.sh [--no-gpu-check]

Launches the 8 PLD Figure-5 LIBERO-90 tasks with official OpenPI pi0_libero.
Mapping: task01->GPU0, task02->GPU1, task17->GPU2, task21->GPU3,
         task33->GPU4, task34->GPU5, task46->GPU6, task47->GPU7.
Each task starts a tmux collect-then-train session.
EOF
      exit 0
      ;;
    *) echo "Unknown arg: $1" >&2; exit 2 ;;
  esac
done

labels=(01 02 17 21 33 34 46 47)
gpus=(0 1 2 3 4 5 6 7)

if [[ "$SKIP_GPU_CHECK" == "0" ]]; then
  echo "Checking target GPU memory usage..."
  for i in "${!gpus[@]}"; do
    gpu="${gpus[$i]}"
    used="$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "$gpu" | tr -d ' ')"
    if [[ "${used:-999999}" -gt 1024 ]]; then
      echo "ERROR: GPU${gpu} is not idle enough: memory.used=${used} MiB" >&2
      echo "Free GPU${gpu}, choose a different mapping manually, or rerun with --no-gpu-check if intentional." >&2
      exit 1
    fi
  done
fi

for i in "${!labels[@]}"; do
  label="${labels[$i]}"
  gpu="${gpus[$i]}"
  config="examples/libero_pld/configs/pld_libero90_task${label}_pi0_libero_300k.yaml"
  session="pld_libero90_task${label}_pi0_libero_300k_gpu${gpu}"
  echo "Launching task${label} on GPU${gpu}: ${session}"
  bash examples/libero_pld/tools/launch_pld_after_collect.sh     --config "$config"     --session "$session"     --gpu "$gpu"     --learner-gpu "$gpu"     --target-successes "$TARGET_SUCCESSES"     --policy-script "$POLICY_SCRIPT"
done

echo "All launch commands submitted. Use tmux ls and tmux attach -t <session> to inspect."
