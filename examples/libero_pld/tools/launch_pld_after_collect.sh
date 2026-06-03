#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
REPO_DIR="$(cd "$ROOT_DIR/.." && pwd)"

CONFIG="examples/libero_pld/configs/pld_libero_spatial_task4.yaml"
SESSION="pld_spatial4"
GPU_ID="0"
ENV_GPU=""
POLICY_GPU=""
COLLECTOR_GPU=""
ACTOR_GPU=""
LEARNER_GPU="1"
START_ENV="1"
START_POLICY="1"
POLICY_SCRIPT="examples/libero/tools/serve_openpi_10000_policy.sh"
TARGET_SUCCESSES="50"
SERL_TORCH_ENV="${SERL_TORCH_ENV:-serl_torch}"

usage() {
  cat <<'EOF'
Usage:
  bash examples/libero_pld/tools/launch_pld_after_collect.sh [options]

Options:
  --config PATH              PLD config yaml.
  --session NAME             tmux session name.
  --gpu ID                   Default GPU for env/policy/collector/actor.
  --env-gpu ID               GPU for env server.
  --policy-gpu ID            GPU for OpenPI policy server.
  --collector-gpu ID         GPU for base-success collector.
  --actor-gpu ID             GPU for actor.
  --learner-gpu ID           GPU for learner.
  --target-successes N       Number of successful base-policy episodes to collect.
  --policy-script PATH       OpenPI policy server launcher.
  --no-env                   Do not start env server.
  --no-policy                Do not start policy server.
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --config) CONFIG="$2"; shift 2 ;;
    --session) SESSION="$2"; shift 2 ;;
    --gpu) GPU_ID="$2"; shift 2 ;;
    --env-gpu) ENV_GPU="$2"; shift 2 ;;
    --policy-gpu) POLICY_GPU="$2"; shift 2 ;;
    --collector-gpu) COLLECTOR_GPU="$2"; shift 2 ;;
    --actor-gpu) ACTOR_GPU="$2"; shift 2 ;;
    --learner-gpu) LEARNER_GPU="$2"; shift 2 ;;
    --target-successes) TARGET_SUCCESSES="$2"; shift 2 ;;
    --policy-script) POLICY_SCRIPT="$2"; shift 2 ;;
    --no-env) START_ENV="0"; shift ;;
    --no-policy) START_POLICY="0"; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown arg: $1" >&2; usage >&2; exit 2 ;;
  esac
done

ENV_GPU="${ENV_GPU:-$GPU_ID}"
POLICY_GPU="${POLICY_GPU:-$GPU_ID}"
COLLECTOR_GPU="${COLLECTOR_GPU:-$GPU_ID}"
ACTOR_GPU="${ACTOR_GPU:-$GPU_ID}"

cd "$REPO_DIR"

if [[ ! -f "$CONFIG" ]]; then
  echo "ERROR: config not found: $CONFIG" >&2
  exit 1
fi

if ! command -v tmux >/dev/null 2>&1; then
  echo "ERROR: tmux is required." >&2
  exit 1
fi

if tmux has-session -t "$SESSION" 2>/dev/null; then
  echo "ERROR: tmux session already exists: $SESSION" >&2
  echo "Attach with: tmux attach -t $SESSION" >&2
  exit 1
fi

CONFIG_ABS="$(cd "$(dirname "$CONFIG")" && pwd)/$(basename "$CONFIG")"
CONFIG_NAME="$(basename "$CONFIG_ABS")"
CONFIG_NAME="${CONFIG_NAME%.*}"
CONFIG_DIR="$(dirname "$CONFIG_ABS")"

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
    "POLICY_DIR": get("policy_dir", "/vla/users/niejunnan/assets/openpi-assets/serl_torch_ckpt/pi0_60000"),
    "OFFLINE_PATH": get("offline.prepared_path", ""),
}

for key, value in items.items():
    print(f"{key}={shlex.quote(str(value))}")
PY
)"

run_collect_cmd() {
  printf 'source /vla/miniconda3/etc/profile.d/conda.sh; '
  printf 'conda activate %q; ' "$SERL_TORCH_ENV"
  printf 'cd %q; ' "$REPO_DIR"
  printf 'export CUDA_VISIBLE_DEVICES=%q; ' "$COLLECTOR_GPU"
  printf 'python examples/libero_pld/scripts/collect_base_success_replay.py --config-path %q --config-name %q pld.base_success.target_successes=%q' \
    "$CONFIG_DIR" "$CONFIG_NAME" "$TARGET_SUCCESSES"
}

run_train_cmd() {
  local gpu="$1"
  shift
  printf 'source /vla/miniconda3/etc/profile.d/conda.sh; '
  printf 'conda activate %q; ' "$SERL_TORCH_ENV"
  printf 'cd %q; ' "$REPO_DIR"
  printf 'export CUDA_VISIBLE_DEVICES=%q; ' "$gpu"
  printf 'python examples/libero_pld/scripts/train_pld.py --config-path %q --config-name %q ' "$CONFIG_DIR" "$CONFIG_NAME"
  printf '%q ' "$@"
}

COLLECT_CMD="$(run_collect_cmd)"
LEARNER_CMD="$(run_train_cmd "$LEARNER_GPU" \
  runtime.role=learner \
  runtime.trainer_port="$TRAINER_PORT" \
  runtime.broadcast_port="$BROADCAST_PORT" \
  runtime.trainer_transport.data_port="$DATA_PORT")"
ACTOR_CMD="$(run_train_cmd "$ACTOR_GPU" \
  runtime.role=actor \
  runtime.trainer_host=127.0.0.1 \
  runtime.trainer_port="$TRAINER_PORT" \
  runtime.broadcast_port="$BROADCAST_PORT" \
  runtime.trainer_transport.data_port="$DATA_PORT")"

CONTROL_SCRIPT="/tmp/${SESSION}_collect_then_train.sh"
{
  echo '#!/usr/bin/env bash'
  echo 'set -euo pipefail'
  printf 'SESSION=%q\n' "$SESSION"
  printf 'COLLECT_CMD=%q\n' "$COLLECT_CMD"
  printf 'LEARNER_CMD=%q\n' "$LEARNER_CMD"
  printf 'ACTOR_CMD=%q\n' "$ACTOR_CMD"
  printf 'OFFLINE_PATH=%q\n' "$OFFLINE_PATH"
  cat <<'EOS'
echo "[controller] collecting base-success offline replay..."
echo "[controller] offline path: ${OFFLINE_PATH}"
eval "$COLLECT_CMD"
echo "[controller] collection finished successfully."
if [[ -n "$OFFLINE_PATH" && ! -f "$OFFLINE_PATH/manifest.json" ]]; then
  echo "[controller] ERROR: collection exited but manifest is missing: $OFFLINE_PATH/manifest.json" >&2
  exit 1
fi
echo "[controller] starting learner window..."
tmux send-keys -t "$SESSION:learner" "$LEARNER_CMD" C-m
sleep 5
echo "[controller] starting actor window..."
tmux send-keys -t "$SESSION:actor" "$ACTOR_CMD" C-m
echo "[controller] done. Attach with: tmux attach -t $SESSION"
EOS
} > "$CONTROL_SCRIPT"
chmod +x "$CONTROL_SCRIPT"

tmux new-session -d -s "$SESSION" -n env

if [[ "$START_ENV" == "1" ]]; then
  tmux send-keys -t "$SESSION:env" "cd '$REPO_DIR'; bash examples/libero/tools/serve_env.sh --host '$ENV_HOST' --port '$ENV_PORT' --gpu-id '$ENV_GPU'" C-m
else
  tmux send-keys -t "$SESSION:env" "echo 'env server not started by launch_pld_after_collect.sh'" C-m
fi

tmux new-window -t "$SESSION" -n policy
if [[ "$START_POLICY" == "1" ]]; then
  tmux send-keys -t "$SESSION:policy" "cd '$REPO_DIR'; POLICY_CONFIG='$POLICY_CONFIG' POLICY_DIR='$POLICY_DIR' bash '$POLICY_SCRIPT' --port '$POLICY_PORT' --gpu-id '$POLICY_GPU'" C-m
else
  tmux send-keys -t "$SESSION:policy" "echo 'policy server not started by launch_pld_after_collect.sh'" C-m
fi

tmux new-window -t "$SESSION" -n collect
tmux send-keys -t "$SESSION:collect" "bash '$CONTROL_SCRIPT'" C-m

tmux new-window -t "$SESSION" -n learner
tmux send-keys -t "$SESSION:learner" "echo 'waiting for collect window to finish, then learner starts on GPU $LEARNER_GPU'" C-m

tmux new-window -t "$SESSION" -n actor
tmux send-keys -t "$SESSION:actor" "echo 'waiting for collect window to finish, then actor starts on GPU $ACTOR_GPU'" C-m

tmux select-window -t "$SESSION:collect"

echo "Started PLD collect-then-train tmux session: $SESSION"
echo "  config      : $CONFIG_ABS"
echo "  env/policy  : GPU $ENV_GPU / GPU $POLICY_GPU"
echo "  collector   : GPU $COLLECTOR_GPU, target_successes=$TARGET_SUCCESSES"
echo "  learner     : GPU $LEARNER_GPU"
echo "  actor       : GPU $ACTOR_GPU"
echo "  offline path: $OFFLINE_PATH"
echo "Attach with: tmux attach -t $SESSION"
