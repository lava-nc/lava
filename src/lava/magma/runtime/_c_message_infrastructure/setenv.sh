#!/bin/bash
SCRIPTPATH=$(cd $(dirname -- "$BASH_SOURCE") && pwd)
export MSG_LOG_PATH="${SCRIPTPATH}/log"
if [ -d "$MSG_LOG_PATH" ]; then
  if [ -n $(find "$MSG_LOG_PATH" -maxdepth 1 -name 'lava_message_infrastructure_pid_*.log') ]; then
    rm "$MSG_LOG_PATH/lava_message_infrastructure_pid_*.log"
  fi
else
  mkdir -p "$MSG_LOG_PATH"
fi
