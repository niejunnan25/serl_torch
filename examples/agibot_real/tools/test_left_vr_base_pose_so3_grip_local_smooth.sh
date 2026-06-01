#!/usr/bin/env bash
set -euo pipefail

AGIBOT_REAL_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SERL_ROOT="$(cd "${AGIBOT_REAL_ROOT}/../../.." && pwd)"

export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/matplotlib-${USER}}"
mkdir -p "${MPLCONFIGDIR}"

ROTATION_CALIBRATION="${ROTATION_CALIBRATION:-${AGIBOT_REAL_ROOT}/assets/vr_calibration/vr_robot_rotation_calibration_right_smooth.json}"

HZ="${HZ:-20}"
MAX_STEP="${MAX_STEP:-0.008}"
SMOOTHING="${SMOOTHING:-0.30}"
TRAJECTORY_TIME="${TRAJECTORY_TIME:-0.045}"
MAX_ROT_STEP_DEG="${MAX_ROT_STEP_DEG:-2.8}"
ROT_SMOOTHING="${ROT_SMOOTHING:-0.16}"
COMMAND_DEADBAND="${COMMAND_DEADBAND:-0.001}"
ROTATION_DEADBAND_DEG="${ROTATION_DEADBAND_DEG:-0.7}"
GRIP_LOCAL_ROT_MAP="${GRIP_LOCAL_ROT_MAP:--ry,-rx,-rz}"

echo "[agibot-vr-left] SO3 grip-local smooth standalone test"
echo "[agibot-vr-left] ROTATION_CALIBRATION=${ROTATION_CALIBRATION}"
echo "[agibot-vr-left] GRIP_LOCAL_ROT_MAP=${GRIP_LOCAL_ROT_MAP}"
echo "[agibot-vr-left] HZ=${HZ} TRAJECTORY_TIME=${TRAJECTORY_TIME}"
echo "[agibot-vr-left] MAX_STEP=${MAX_STEP} SMOOTHING=${SMOOTHING} COMMAND_DEADBAND=${COMMAND_DEADBAND}"
echo "[agibot-vr-left] MAX_ROT_STEP_DEG=${MAX_ROT_STEP_DEG} ROT_SMOOTHING=${ROT_SMOOTHING} ROTATION_DEADBAND_DEG=${ROTATION_DEADBAND_DEG}"

cd "${SERL_ROOT}"

python "${AGIBOT_REAL_ROOT}/scripts/run_vr_base_pose_so3_teleop_smooth.py" \
  --hand left \
  --hz "${HZ}" \
  --control-mode absolute \
  --so3-order base \
  --rotation-calibration "${ROTATION_CALIBRATION}" \
  --calibrated-rotation-mode grip-local \
  --max-delta 100.0 \
  --max-step "${MAX_STEP}" \
  --smoothing "${SMOOTHING}" \
  --command-deadband "${COMMAND_DEADBAND}" \
  --trajectory-time "${TRAJECTORY_TIME}" \
  --max-rot-delta-deg 360.0 \
  --max-rot-step-deg "${MAX_ROT_STEP_DEG}" \
  --rot-smoothing "${ROT_SMOOTHING}" \
  --rotation-deadband-deg "${ROTATION_DEADBAND_DEG}" \
  --grip-local-rot-map="${GRIP_LOCAL_ROT_MAP}" \
  --gripper-open 0 \
  --gripper-closed 120 \
  --print-delta \
  --plot \
  --execute
