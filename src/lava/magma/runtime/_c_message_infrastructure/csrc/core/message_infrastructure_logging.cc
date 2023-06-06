// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: BSD-3-Clause
// See: https://spdx.org/licenses/

#include <core/message_infrastructure_logging.h>

namespace message_infrastructure {

namespace {

signed int GetPid() {
  return getpid();
}

std::string GetTime() {
  char buf[MAX_SIZE_LOG_TIME] = {};
  struct timespec ts;
  timespec_get(&ts, TIME_UTC);
  int end = strftime(buf, sizeof(buf), "%Y-%m-%d.%X", gmtime(&ts.tv_sec));
  snprintf(buf + end, MAX_SIZE_LOG_TIME-end, " %09ld", ts.tv_nsec);
  return std::string(buf);
}

}  // namespace

LogMsg::LogMsg(const std::string &msg_data,
               const char *log_file,
               const int &log_line,
               const char *log_level)
    : msg_data_(msg_data),
      msg_line_(log_line),
      msg_file_(log_file),
      msg_level_(log_level) {
  msg_time_ = GetTime();
}

std::string LogMsg::GetEntireLogMsg(const int &pid) {
  std::stringstream buf;
  buf << msg_time_    << " ";
  buf << msg_level_   << " ";
  buf << pid          << " ";
  buf << msg_file_    << ":";
  buf << msg_line_    << " ";
  buf << msg_data_;
  return buf.str();
}

MessageInfrastructureLog::MessageInfrastructureLog() {
  char *log_path = getenv("MSG_LOG_PATH");
  if (log_path == nullptr) {
    log_path = getcwd(nullptr, 0);
    log_path_ = log_path;
    free(log_path);
    return;
  }
  log_path_ = log_path;
}

// multithread safe
void MessageInfrastructureLog::LogWrite(const LogMsg& msg) {
  std::lock_guard<std::mutex> lg(log_lock_);
  log_queue_.push(msg);
  if (log_queue_.size() == MAX_SIZE_LOG) {
      WriteDown();
  }
}

// multithread unsafe
void MessageInfrastructureLog::Clear() {
  std::queue<LogMsg>().swap(log_queue_);
}

// multithread unsafe
void MessageInfrastructureLog::WriteDown() {
  if (log_queue_.empty()) return;
  signed int pid = GetPid();
  std::stringstream log_file_name;
  log_file_name << log_path_ << "/" << DEBUG_LOG_MODULE << "_pid_" << pid \
                << "." << DEBUG_LOG_FILE_SUFFIX;
  std::fstream log_file;
  log_file.open(log_file_name.str(), std::ios::app);
  while (!log_queue_.empty()) {
    std::string log_str = log_queue_.front().GetEntireLogMsg(pid);
    log_file << log_str;
    log_queue_.pop();
  }
  log_file.close();
}

MessageInfrastructureLog::~MessageInfrastructureLog() {
  WriteDown();
}

void LogClear() {
#if defined(MSG_LOG_FILE_ENABLE)
  GetLogInstance()->Clear();
#endif
}

MessageInfrastructureLogPtr log_instance;

MessageInfrastructureLogPtr GetLogInstance() {
  if (log_instance == nullptr) {
    log_instance = std::make_shared<MessageInfrastructureLog>();
  }
  return log_instance;
}

}  // namespace message_infrastructure
