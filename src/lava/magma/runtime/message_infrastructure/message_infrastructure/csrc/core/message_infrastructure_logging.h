// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: BSD-3-Clause
// See: https://spdx.org/licenses/

#ifndef MESSAGE_INFRASTRUCTURE_LOGGING_H_
#define MESSAGE_INFRASTRUCTURE_LOGGING_H_

#include <memory>
#include <mutex>
#include <iostream>
#include <sstream>
#include <fstream>
#include <string>
#include <cstdlib>
#include <ctime>
#include <queue>

#if _WIN32
#inlcude <process.h>
#include <direct.h>  // _getcwd()
#define getpid() _getpid()
#define getcwd() _getcwd()
#else  // __linux__ & __APPLE__
#include <unistd.h>
#include <sys/types.h>
#endif

#define MAX_SIZE_PER_LOG_MSG (1024)
#define MAX_SIZE_LOG (1)
#define MAX_SIZE_LOG_TIME (64)
#define LOG_MSG_SUBSTITUTION "!This message was displayed due to the failure of the malloc of this log message!\n"
#define LOG_GET_TIME_FAIL "Get log time failed."
#define DEBUG_LOG_MODULE "lava_message_infrastructure"
#define DEBUG_LOG_FILE_SUFFIX "log"
#define NULL_STRING ""

#define LOG_MASK_NULL (0)
#define LOG_MASK_INFO (1)
#define LOG_MASK_DUMP (1<<1)
#define LOG_MASK_DBUG (1<<2)
#define LOG_MASK_WARN (1<<3)
#define LOG_MASK_ERRO (1<<4)

#define LOG_PRINT_MASK_NULL (0)
#define LOG_PRINT_MASK_SHEL (1)
#define LOG_PRINT_MASK_FILE (2)

#if defined(MSG_LOG_LEVEL)

#elif defined(MSG_LOG_LEVEL_ALL)
  #define MSG_LOG_LEVEL (LOG_MASK_INFO | LOG_MASK_ERRO | LOG_MASK_DUMP | LOG_MASK_WARN | LOG_MASK_DBUG)
#elif defined(MSG_LOG_LEVEL_WARN_ERR)
  #define MSG_LOG_LEVEL (LOG_MASK_ERRO | LOG_MASK_WARN)
#else
  #define MSG_LOG_LEVEL (LOG_MASK_ERRO)  // default
#endif

#if defined(MSG_LOG_PRINT_MODE)

#elif defined(MSG_LOG_PRINT_MODE_ALL)
  #define MSG_LOG_PRINT_MODE (LOG_PRINT_MASK_SHEL | LOG_PRINT_MASK_FILE)
#elif defined(MSG_LOG_PRINT_MODE_FILE)
  #define MSG_LOG_PRINT_MODE (LOG_PRINT_MASK_FILE)
#elif defined(MSG_LOG_PRINT_MODE_NULL)
  #define MSG_LOG_PRINT_MODE (LOG_PRINT_MASK_NULL)
#else
  #define MSG_LOG_PRINT_MODE (LOG_PRINT_MASK_SHEL)  // default
#endif

#if (MSG_LOG_PRINT_MODE)
#define DEBUG_LOG_PRINT(_level, _module, _fmt, ...) do { \
  if (MSG_LOG_PRINT_MODE & LOG_PRINT_MASK_FILE) { \
    int length = 0; \
    char * log_data = (char*)malloc(sizeof(char)*MAX_SIZE_PER_LOG_MSG); \
    if (log_data != NULL) \
      length = std::snprintf(log_data, MAX_SIZE_PER_LOG_MSG, _fmt,  ## __VA_ARGS__); \
    if (log_data == NULL || length <0) { \
      getLogInstance()->LogWrite(LogMsg(std::string(LOG_MSG_SUBSTITUTION), __FILE__, __LINE__, _module, _level)); \
    } else { \
      getLogInstance()->LogWrite(LogMsg(std::string(log_data), __FILE__, __LINE__, _module, _level)); \
    } \
    free(log_data); \
  } \
  if (MSG_LOG_PRINT_MODE & LOG_PRINT_MASK_SHEL) { \
    std::printf(_fmt, ## __VA_ARGS__); \
  } \
}while(0)
#else
#define DEBUG_LOG_PRINT(_level, _module, ...)
#endif

#define LAVA_LOG(_module, _fmt, ...) do { \
  if ((_module) && (MSG_LOG_LEVEL & LOG_MASK_INFO)) { \
    DEBUG_LOG_PRINT("[CPP INFO]", _module, _fmt, ## __VA_ARGS__); \
  } \
}while(0)

#define LAVA_DUMP(_module, _fmt, ...) do { \
  if (_module && (MSG_LOG_LEVEL & LOG_MASK_DUMP)) { \
    DEBUG_LOG_PRINT("[CPP DUMP]", _module, _fmt, ## __VA_ARGS__); \
  } \
}while(0)

#define LAVA_DEBUG(_module, _fmt, ...) do { \
  if (_module && (MSG_LOG_LEVEL & LOG_MASK_DBUG)) { \
    DEBUG_LOG_PRINT("[CPP DBUG]", _module, _fmt, ## __VA_ARGS__); \
  } \
}while(0)

#define LAVA_LOG_WARN(_module, _fmt, ...) do { \
  if (_module && (MSG_LOG_LEVEL & LOG_MASK_WARN)) { \
    DEBUG_LOG_PRINT("[CPP WARN]", _module, _fmt, ## __VA_ARGS__); \
  } \
}while(0)

#define LAVA_LOG_ERR(_module, _fmt, ...) do { \
  if (_module && (MSG_LOG_LEVEL & LOG_MASK_ERRO)) { \
    DEBUG_LOG_PRINT("[CPP ERRO]", _module, _fmt, ## __VA_ARGS__); \
  } \
}while(0)

namespace message_infrastructure {

enum Log_Level{
  DISABLE_LOG,
  ERROR,
  WARN,
  LOG,
  DUMP,
  DEBUG
};

enum Log_Module{
  LOG_NULL,
  LOG_MP,  // log for multiprocessing
  LOG_ACTOR,
  LOG_LAYER,
  LOG_SMMP,  // log for shmemport
  LOG_SKP  // log for socketport
};

class LogMsg{
 public:
  LogMsg(const std::string&,
         const char* log_file,
         int log_line,
         enum Log_Module log_module,
         const char* log_level);
  std::string getEntireLogMsg(int pid);

 private:
  std::string msg_time_;
  std::string msg_data_;
  std::string msg_level_;
  std::string msg_module_;
  std::string msg_file_;
  int msg_line_ = 0;
};

class Log{
 public:
  Log();
  ~Log();
  void LogWrite(const LogMsg& msg);
  void Clear();
  void Write_down();

 private:
  std::string log_path_;
  std::mutex log_lock_;
  std::queue<LogMsg> log_queue_;
};

using LogPtr = std::shared_ptr<Log>;

LogPtr getLogInstance();

std::string getTime();

signed int getPid();

void LogClear();

}  // namespace message_infrastructure

#endif  // MESSAGE_INFRASTRUCTURE_LOGGING_H_
