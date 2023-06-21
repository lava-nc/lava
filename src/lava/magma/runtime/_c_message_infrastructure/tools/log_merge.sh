#/bin/bash
LOG_FILE_FULL_NAME="lava_message_infrastructure.log"
LOG_FILE_SCATTERED_PATH=$1
LOG_LEVEL=$2
if [ "$LOG_LEVEL" = "" ];then
  LOG_LEVEL="ERRO|INFO|WARN|DBUG|DUMP"
fi
if [ "$LOG_FILE_SCATTERED_PATH" = "" ];then
  LOG_FILE_SCATTERED_PATH=$MSG_LOG_PATH
fi
grep -E "$LOG_LEVEL" "$LOG_FILE_SCATTERED_PATH"/lava_message_infrastructure_pid_*.log | sort -n > "$LOG_FILE_SCATTERED_PATH"/"$LOG_FILE_FULL_NAME"
