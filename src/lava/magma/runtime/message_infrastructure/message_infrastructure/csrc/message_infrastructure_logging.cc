// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: BSD-3-Clause
// See: https://spdx.org/licenses/

#include <message_infrastructure_logging.h>

namespace message_infrastructure {

const char* LOG_MODULE_NAME[] = {
  "LOG_MULL_MODULE",
  "LOG_MPROC", // multiprocess
  "LOG_ACTOR",
  "LOG_LAYER",
  "LOG_SHMEM"
};

LogMsg::LogMsg(const std::string &msg_data, const char* log_file, int log_line, enum Log_Module log_module, const char* log_level){
    msg_data_ = msg_data;
    msg_time_ = getTime();
    msg_file_ = std::string(log_file);
    msg_level_ = std::string(log_level);
    msg_module_ = std::string(LOG_MODULE_NAME[log_module]);
    msg_pid_ = "0";
    msg_line_ = log_line;
}

std::string LogMsg::getEntireLogMsg(int pid){
    std::stringstream buf;
    buf << msg_time_    << " ";
    buf << msg_level_   << " ";
    buf << pid          << " ";
    buf << msg_module_  << " ";
    buf << msg_file_    << ":";
    buf << msg_line_    << " ";
    buf << msg_data_;
    return buf.str();
}

Log::Log(){
    current_length_ = 0;
    char * log_path = getenv("MSG_LOG_PATH");
    if (log_path == NULL){
        log_path = getcwd(NULL, 0);
        log_path_ = log_path;
        free(log_path);
        return;
    }
    log_path_ = log_path;
}

//multithread safe
void Log::log(LogMsg* msg){
    std::lock_guard<std::mutex> lg(log_lock_);
    log_buffer_[current_length_++] = msg;
        if(current_length_ == max_length_){
        write_down();
    }
}

//multithread unsafe
void Log::clear(){
    int i = 0;
    for(;i<current_length_;i++){
        delete log_buffer_[i];
    }
    current_length_ = 0;
}

//multithread unsafe
void Log::write_down(){
    int i = 0;
    signed int pid = getPid();
    std::stringstream log_file_name;
    log_file_name << log_path_ << "/" <<DEBUG_LOG_MODULE << "_pid_" << pid << "." << DEBUG_LOG_FILE_SUFFIX;
    std::fstream log_file;
    log_file.open(log_file_name.str(), std::ios::app);
    for(;i<current_length_;i++){
        std::string log_str = (*log_buffer_[i]).getEntireLogMsg(pid);
        log_file << log_str;
    }
    log_file.close();
    clear();
}

Log::~Log(){
    write_down();
    // clear();
}

signed int getPid(){
return getpid();
}

std::string getTime(){
  char buf[MAX_SIZE_LOG_TIME] = {};
  struct timespec ts;
  timespec_get(&ts, TIME_UTC);
  int end = strftime(buf, sizeof(buf), "%Y-%m-%d.%X", gmtime(&ts.tv_sec));
  snprintf(buf + end, MAX_SIZE_LOG_TIME-end," %09ld", ts.tv_nsec);
  std::string ret = std::string(buf);
  return ret;
}

void LogClear(){
#if DEBUG_MODE && DEBUG_LOG_TO_FILE
    getLogInstance()->clear();
#endif
}

LogPtr log_instance;

LogPtr getLogInstance(){
  if (log_instance == nullptr)
    log_instance = std::make_shared<Log>();
  return log_instance;
}

} // namespace message_infrastructure