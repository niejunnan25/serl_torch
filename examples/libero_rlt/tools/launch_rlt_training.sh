#!/usr/bin/env bash
set -euo pipefail

SESSION="libero_rlt"
CONFIG_NAME="smoke_rlt"
GPU="0"
LEARNER_GPU="1"
ENV_PORT="20000"
VLA_PORT="8775"
WITH_EVAL="0"
EVAL_GPU=""
EVAL_ENV_PORT="20001"
EVAL_VLA_PORT="8875"
RUN_DIR=""
PI0_PATH=""
RLT_ENCODER_PATH=""
PI0_CONFIG="pi0_libero"
CONDA_SH="/vla/miniconda3/etc/profile.d/conda.sh"
CONDA_ENV="serl_torch"
OPENPI_ROOT="/vla/users/niejunnan/codebase/openpi-modified"
VLA_PYTHON="/vla/users/niejunnan/codebase/openpi-modified/.venv/bin/python3"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
HYDRA_OVERRIDES=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --session) SESSION="$2"; shift 2 ;;
    --config-name) CONFIG_NAME="$2"; shift 2 ;;
    --gpu) GPU="$2"; shift 2 ;;
    --learner-gpu) LEARNER_GPU="$2"; shift 2 ;;
    --env-port) ENV_PORT="$2"; shift 2 ;;
    --vla-port) VLA_PORT="$2"; shift 2 ;;
    --with-eval) WITH_EVAL="1"; shift ;;
    --eval-gpu) EVAL_GPU="$2"; shift 2 ;;
    --eval-env-port) EVAL_ENV_PORT="$2"; shift 2 ;;
    --eval-vla-port) EVAL_VLA_PORT="$2"; shift 2 ;;
    --run-dir) RUN_DIR="$2"; shift 2 ;;
    --pi0-path) PI0_PATH="$2"; shift 2 ;;
    --rlt-encoder-path) RLT_ENCODER_PATH="$2"; shift 2 ;;
    --pi0-config) PI0_CONFIG="$2"; shift 2 ;;
    --conda-env) CONDA_ENV="$2"; shift 2 ;;
    --openpi-root) OPENPI_ROOT="$2"; shift 2 ;;
    --vla-python) VLA_PYTHON="$2"; shift 2 ;;
    --) shift; HYDRA_OVERRIDES=("$@"); break ;;
    *) echo "unknown argument: $1" >&2; exit 2 ;;
  esac
done

if [[ -z "$PI0_PATH" || -z "$RLT_ENCODER_PATH" ]]; then
  echo "--pi0-path and --rlt-encoder-path are required" >&2
  exit 2
fi

if [[ -z "$EVAL_GPU" ]]; then
  EVAL_GPU="$GPU"
fi
if [[ -z "$RUN_DIR" ]]; then
  RUN_DIR="${REPO_ROOT}/outputs/$(date +%Y-%m-%d)/$(date +%H-%M-%S)"
fi

PY_PREFIX="source ${CONDA_SH}; conda activate ${CONDA_ENV}; cd ${REPO_ROOT}; export PYTHONPATH=${REPO_ROOT}/serl_launcher:${REPO_ROOT}:${OPENPI_ROOT}/src:${OPENPI_ROOT}:\${PYTHONPATH:-}"
VLA_PREFIX="cd ${REPO_ROOT}; export PYTHONPATH=${REPO_ROOT}/serl_launcher:${REPO_ROOT}:${OPENPI_ROOT}/src:${OPENPI_ROOT}:\${PYTHONPATH:-}"

quote_args() {
  local quoted=""
  local arg
  local item
  for arg in "$@"; do
    printf -v item "%q" "$arg"
    quoted+=" ${item}"
  done
  printf "%s" "$quoted"
}

quote_cmd() {
  local quoted
  printf -v quoted "%q" "$1"
  printf "%s" "$quoted"
}

tmux_start_window() {
  local name="$1"
  local command="$2"
  tmux new-window -t "$SESSION" -n "$name" "bash -lc $(quote_cmd "$command")"
}

LEARNER_ARGS="$(quote_args \
  "${HYDRA_OVERRIDES[@]}" \
  "hydra.run.dir=${RUN_DIR}" \
  "env.remote.port=${ENV_PORT}" \
  "rlt.vla_server_port=${VLA_PORT}" \
  "training.async_eval.env.remote.port=${EVAL_ENV_PORT}" \
  "rlt.eval_vla_server_port=${EVAL_VLA_PORT}" \
  "runtime.role=learner" \
  "rlt.pi0_config_name=${PI0_CONFIG}" \
  "rlt.pi0_checkpoint_path=${PI0_PATH}" \
  "rlt.rlt_encoder_path=${RLT_ENCODER_PATH}")"
ACTOR_ARGS="$(quote_args \
  "${HYDRA_OVERRIDES[@]}" \
  "hydra.run.dir=${RUN_DIR}" \
  "hydra.output_subdir=null" \
  "env.remote.port=${ENV_PORT}" \
  "rlt.vla_server_port=${VLA_PORT}" \
  "training.async_eval.env.remote.port=${EVAL_ENV_PORT}" \
  "rlt.eval_vla_server_port=${EVAL_VLA_PORT}" \
  "runtime.role=actor" \
  "rlt.pi0_config_name=${PI0_CONFIG}" \
  "rlt.pi0_checkpoint_path=${PI0_PATH}" \
  "rlt.rlt_encoder_path=${RLT_ENCODER_PATH}")"

tmux has-session -t "$SESSION" 2>/dev/null && { echo "tmux session exists: $SESSION" >&2; exit 1; }
tmux new-session -d -s "$SESSION" -n env "bash -lc $(quote_cmd "cd ${REPO_ROOT}; LIBERO_CONDA_PREFIX=/vla/users/niejunnan/envs/libero bash examples/libero/tools/serve_env.sh --port ${ENV_PORT} --gpu-id ${GPU}")"
tmux_start_window vla "${VLA_PREFIX}; CUDA_VISIBLE_DEVICES=${GPU} ${VLA_PYTHON} examples/libero_rlt/scripts/serve_vla_features.py --pi0-config ${PI0_CONFIG} --pi0-path ${PI0_PATH} --rlt-encoder-path ${RLT_ENCODER_PATH} --port ${VLA_PORT}"
if [[ "$WITH_EVAL" == "1" ]]; then
  tmux_start_window eval-env "cd ${REPO_ROOT}; LIBERO_CONDA_PREFIX=/vla/users/niejunnan/envs/libero bash examples/libero/tools/serve_env.sh --port ${EVAL_ENV_PORT} --gpu-id ${EVAL_GPU}"
  tmux_start_window eval-vla "${VLA_PREFIX}; CUDA_VISIBLE_DEVICES=${EVAL_GPU} ${VLA_PYTHON} examples/libero_rlt/scripts/serve_vla_features.py --pi0-config ${PI0_CONFIG} --pi0-path ${PI0_PATH} --rlt-encoder-path ${RLT_ENCODER_PATH} --port ${EVAL_VLA_PORT}"
fi
tmux_start_window learner "${PY_PREFIX}; CUDA_VISIBLE_DEVICES=${LEARNER_GPU} python examples/libero_rlt/scripts/run_rlt_training.py --config-name ${CONFIG_NAME}${LEARNER_ARGS}"
tmux_start_window actor "${PY_PREFIX}; CUDA_VISIBLE_DEVICES=${GPU} python examples/libero_rlt/scripts/run_rlt_training.py --config-name ${CONFIG_NAME}${ACTOR_ARGS}"

echo "Started LIBERO RLT tmux session: ${SESSION}"
echo "run dir      : ${RUN_DIR}"
echo "train env/VLA: ${ENV_PORT} / ${VLA_PORT} on GPU ${GPU}"
if [[ "$WITH_EVAL" == "1" ]]; then
  echo "eval env/VLA : ${EVAL_ENV_PORT} / ${EVAL_VLA_PORT} on GPU ${EVAL_GPU}"
fi
echo "Attach with: tmux attach -t ${SESSION}"
if [[ ${#HYDRA_OVERRIDES[@]} -gt 0 ]]; then
  echo "Hydra overrides:${HYDRA_OVERRIDES[*]/#/ }"
fi
