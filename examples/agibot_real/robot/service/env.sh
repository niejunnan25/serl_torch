#!/usr/bin/env bash

# Repo-local AgiBot runtime environment bootstrap.
# Bundled AgiBot runtime env so actor/eval can start from this repo without
# sourcing files outside examples/agibot_real.

CURRENT_SCRIPT_DIR="$(dirname "$(realpath "${BASH_SOURCE[0]}")")"
LOG_DIR="$CURRENT_SCRIPT_DIR/log"
mkdir -p "$LOG_DIR"

local_ip="$(ip -o -4 addr list | grep '10.42.0.' | awk '{print $4}' | cut -d/ -f1)"
if [[ -z "$local_ip" ]]; then
    echo "no ip in 10.42.0.* found, can not communicate with robot"
else
    export LOCATOR_IP="$local_ip"
    export AORTA_DISCOVERY_URI="http://10.42.0.101:2379"
fi

export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python
export DYLOG_log_dir="$LOG_DIR"
export DYLOG_LOG_SIZE=20000
export DYLOG_DEFAULT_LEVEL=FATAL

if [[ -f "/opt/ros/humble/setup.bash" ]]; then
    # shellcheck source=/dev/null
    source /opt/ros/humble/setup.bash
    export ROS_VERSION=2
    export ROS_PYTHON_VERSION=3
    export ROS_DOMAIN_ID=0
    export ROS_LOCALHOST_ONLY=1
    export ROS_DISTRO=humble
    export RMW_IMPLEMENTATION=rmw_fastrtps_cpp
    ros2 daemon stop >/dev/null 2>&1 || true
    ros2 daemon start >/dev/null 2>&1 || true
fi
