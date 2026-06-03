#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
REPO_DIR="$(cd "$ROOT_DIR/.." && pwd)"

CONFIG="examples/libero_pld/configs/pld_libero_spatial_task4.yaml"
SESSION="pld_libero"
GPU_ID="0"
ENV_GPU=""
POLICY_GPU=""
LEARNER_GPU=""
ACTOR_GPU=""
START_ENV="1"
START_POLICY="1"
POLICY_SCRIPT="examples/libero/tools/serve_openpi_10000_policy.sh"
SERL_TORCH_ENV="${SERL_TORCH_ENV:-serl_torch}"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --config) CONFIG="$2"; shift 2 ;;
    --session) SESSION="$2"; shift 2 ;;
    --gpu) GPU_ID="$2"; shift 2 ;;
    --env-gpu) ENV_GPU="$2"; shift 2 ;;
    --policy-gpu) POLICY_GPU="$2"; shift 2 ;;
    --learner-gpu) LEARNER_GPU="$2"; shift 2 ;;
    --actor-gpu) ACTOR_GPU="$2"; shift 2 ;;
    --policy-script) POLICY_SCRIPT="$2"; shift 2 ;;
    --no-env) START_ENV="0"; shift ;;
    --no-policy) START_POLICY="0"; shift ;;
    *) echo "Unknown arg: $1" >&2; exit 2 ;;
  esac
done

ENV_GPU="${ENV_GPU:-$GPU_ID}"
POLICY_GPU="${POLICY_GPU:-$GPU_ID}"
LEARNER_GPU="${LEARNER_GPU:-$GPU_ID}"
ACTOR_GPU="${ACTOR_GPU:-$GPU_ID}"

cd "$REPO_DIR"

if [[ ! -f "$CONFIG" ]]; then
  echo "ERROR: config not found: $CONFIG" >&2
  exit 1
fi
CONFIG_ABS="$(cd "$(dirname "$CONFIG")" && pwd)/$(basename "$CONFIG")"

PYTHON_BIN="/vla/miniconda3/envs/${SERL_TORCH_ENV}/bin/python"
if [[ "$SERL_TORCH_ENV" == /* ]]; then
  PYTHON_BIN="${SERL_TORCH_ENV}/bin/python"
fi
if [[ ! -x "$PYTHON_BIN" ]]; then
  PYTHON_BIN="python"
fi

eval "$("$PYTHON_BIN" - "$CONFIG_ABS" <<'PY'
import shlex
import sys
from pathlib import Path

from omegaconf import OmegaConf

cfg = OmegaConf.load(sys.argv[1])
def get(path, default=None):
    cur = cfg
    for part in path.split("."):
        if not hasattr(cur, part):
            return default
        cur = getattr(cur, part)
    return cur

items = {
    "ENV_HOST": get("env.remote.host", "127.0.0.1"),
    "ENV_PORT": get("env.remote.port", 57100),
    "POLICY_PORT": get("policy.port", 57101),
    "TRAINER_PORT": get("runtime.trainer_port", 57104),
    "BROADCAST_PORT": get("runtime.broadcast_port", 57105),
    "DATA_PORT": get("runtime.trainer_transport.data_port", 57106),
    "POLICY_CONFIG": get("policy_config", "pi0_libero_baseline_10_bs32_150000"),
    "POLICY_DIR": get("policy_dir", "/vla/users/niejunnan/assets/openpi-assets/serl_torch_ckpt/pi0_10000"),
}

for key, value in items.items():
    print(f"{key}={shlex.quote(str(value))}")
PY
)"

CONFIG_NAME="$(basename "$CONFIG_ABS")"
CONFIG_NAME="${CONFIG_NAME%.*}"
CONFIG_DIR="$(dirname "$CONFIG_ABS")"

run_python() {
  local gpu="$1"
  shift
  printf 'source /vla/miniconda3/etc/profile.d/conda.sh; '
  printf 'conda activate %q; ' "$SERL_TORCH_ENV"
  printf 'cd %q; ' "$REPO_DIR"
  printf 'export CUDA_VISIBLE_DEVICES=%q; ' "$gpu"
  printf 'python examples/libero_pld/scripts/train_pld.py --config-path %q --config-name %q ' "$CONFIG_DIR" "$CONFIG_NAME"
  printf '%q ' "$@"
}

tmux has-session -t "$SESSION" 2>/dev/null && {
  echo "ERROR: tmux session already exists: $SESSION" >&2
  exit 1
}

tmux new-session -d -s "$SESSION" -n env

if [[ "$START_ENV" == "1" ]]; then
  tmux send-keys -t "$SESSION:0" "cd '$REPO_DIR'; bash examples/libero/tools/serve_env.sh --host '$ENV_HOST' --port '$ENV_PORT' --gpu-id '$ENV_GPU'" C-m
else
  tmux send-keys -t "$SESSION:0" "echo 'env server not started by launch_pld.sh'" C-m
fi

tmux new-window -t "$SESSION" -n policy
if [[ "$START_POLICY" == "1" ]]; then
  tmux send-keys -t "$SESSION:policy" "cd '$REPO_DIR'; POLICY_CONFIG='$POLICY_CONFIG' POLICY_DIR='$POLICY_DIR' bash '$POLICY_SCRIPT' --port '$POLICY_PORT' --gpu-id '$POLICY_GPU'" C-m
else
  tmux send-keys -t "$SESSION:policy" "echo 'policy server not started by launch_pld.sh'" C-m
fi

tmux new-window -t "$SESSION" -n learner
LEARNER_CMD="$(run_python "$LEARNER_GPU" runtime.role=learner runtime.trainer_port="$TRAINER_PORT" runtime.broadcast_port="$BROADCAST_PORT" runtime.trainer_transport.data_port="$DATA_PORT")"
tmux send-keys -t "$SESSION:learner" "$LEARNER_CMD" C-m

tmux new-window -t "$SESSION" -n actor
ACTOR_CMD="$(run_python "$ACTOR_GPU" runtime.role=actor runtime.trainer_host=127.0.0.1 runtime.trainer_port="$TRAINER_PORT" runtime.broadcast_port="$BROADCAST_PORT" runtime.trainer_transport.data_port="$DATA_PORT")"
tmux send-keys -t "$SESSION:actor" "$ACTOR_CMD" C-m

tmux select-window -t "$SESSION:learner"
echo "Started PLD tmux session: $SESSION"
echo "Attach with: tmux attach -t $SESSION"
