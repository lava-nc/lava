// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: BSD-3-Clause
// See: https://spdx.org/licenses/

#ifndef CORE_MESSAGE_INFRASTRUCTURE_LOGGING_H_
#define CORE_MESSAGE_INFRASTRUCTURE_LOGGING_H_

#include <memory>
#include <mutex>  // NOLINT
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

#define MAX_SIZE_LOG (1)
#define MAX_SIZE_LOG_TIME (64)
#define MAX_SIZE_PER_LOG_MSG (1024)

#define NULL_STRING           ""
#define DEBUG_LOG_FILE_SUFFIX "log"
#define LOG_GET_TIME_FAIL     "Get log time failed."
#define DEBUG_LOG_MODULE      "lava_message_infrastructure"
#define LOG_MSG_SUBSTITUTION  "This message was displayed due to " \
                              "the failure of the malloc of this log message!"

// the following macros indicate if the specific log message need to be printed
// except the ERROR log message, ERROR log message will be printed whatever the
// macro value is
#define LOG_MP    (1)  // log for multiprocessing
#define LOG_ACTOR (1)
#define LOG_LAYER (1)
#define LOG_SMMP  (1)  // log for shmemport
#define LOG_SKP   (1)  // log for socketport
#define LOG_DDS   (1)  // lof for DDS Channel
#define LOG_UTTEST   (1)

#if defined(MSG_LOG_LEVEL)
#elif defined(MSG_LOG_LEVEL_ALL)
  #define MSG_LOG_LEVEL (LOG_MASK_DBUG)
#elif defined(MSG_LOG_LEVEL_WARN)
  #define MSG_LOG_LEVEL (LOG_MASK_WARN)
#elif defined(MSG_LOG_LEVEL_DUMP)
  #define MSG_LOG_LEVEL (LOG_MASK_DUMP)
#elif defined(MSG_LOG_LEVEL_INFO)
  #define MSG_LOG_LEVEL (LOG_MASK_INFO)
#else
  #define MSG_LOG_LEVEL (LOG_MASK_ERRO)  // default
#endif

#if defined(MSG_LOG_FILE_ENABLE)
#define DEBUG_LOG_PRINT(_level, _fmt, ...) do { \
    int length = 0; \
    char *log_data = reinterpret_cast<char *> \
                      (malloc(sizeof(char)*MAX_SIZE_PER_LOG_MSG)); \
    if (log_data != nullptr) { \
      length = std::snprintf(log_data, \
                             MAX_SIZE_PER_LOG_MSG, _fmt, ## __VA_ARGS__); \
    } \
    if (log_data == nullptr || length <0) { \
      GetLogInstance()->LogWrite(LogMsg(std::string(LOG_MSG_SUBSTITUTION), \
                                        __FILE__, \
                                        __LINE__, \
                                        _level)); \
    } else { \
      GetLogInstance()->LogWrite(LogMsg(std::string(log_data), \
                                        __FILE__, \
                                        __LINE__, \
                                        _level)); \
    } \
    free(log_data); \
    std::printf("%s[%d] ", _level, getpid()); \
    std::printf(_fmt, ## __VA_ARGS__); \
} while (0)
#else
#define DEBUG_LOG_PRINT(_level, _fmt, ...) do { \
  std::printf("%s", _level); \
  std::printf(_fmt, ## __VA_ARGS__); \
} while (0)
#endif

#define LAVA_LOG(_module, _fmt, ...) do { \
  if ((_module) && (MSG_LOG_LEVEL <= LOG_MASK_INFO)) { \
    DEBUG_LOG_PRINT("[CPP INFO]", _fmt, ## __VA_ARGS__); \
  } \
} while (0)

#define LAVA_DUMP(_module, _fmt, ...) do { \
  if (_module && (MSG_LOG_LEVEL <= LOG_MASK_DUMP)) { \
    DEBUG_LOG_PRINT("[CPP DUMP]", _fmt, ## __VA_ARGS__); \
  } \
} while (0)

#define LAVA_DEBUG(_module, _fmt, ...) do { \
  if (_module && (MSG_LOG_LEVEL <= LOG_MASK_DBUG)) { \
    DEBUG_LOG_PRINT("[CPP DBUG]", _fmt, ## __VA_ARGS__); \
  } \
} while (0)

#define LAVA_LOG_WARN(_module, _fmt, ...) do { \
  if (_module && (MSG_LOG_LEVEL <= LOG_MASK_WARN)) { \
    DEBUG_LOG_PRINT("[CPP WARN]", _fmt, ## __VA_ARGS__); \
  } \
} while (0)

#define LAVA_LOG_ERR(_fmt, ...) do { \
  DEBUG_LOG_PRINT("[CPP ERRO]", _fmt, ## __VA_ARGS__); \
} while (0)

#define LAVA_ASSERT_INT(result, expectation) do { \
  if (int r = (result) != expectation) { \
    LAVA_LOG_ERR("Assert failed, %d get, %d except. Errno: %d\n", \
                 r, 0, errno); \
    exit(-1); \
  } \
} while (0)

namespace message_infrastructure {

enum LogLevel {
  LOG_MASK_DBUG,
  LOG_MASK_INFO,
  LOG_MASK_DUMP,
  LOG_MASK_WARN,
  LOG_MASK_ERRO
};

class LogMsg{
 public:
  LogMsg(const std::string &msg_data,
         const char *log_file,
         const int &log_line,
         const char *log_level);
  std::string GetEntireLogMsg(const int &pid);

 private:
  std::string msg_time_;
  std::string msg_data_;
  std::string msg_level_;
  std::string msg_file_;
  int msg_line_ = 0;
};

class MessageInfrastructureLog {
 public:
  MessageInfrastructureLog();
  ~MessageInfrastructureLog();
  void LogWrite(const LogMsg &msg);
  void Clear();
  void WriteDown();

 private:
  std::string log_path_;
  std::mutex log_lock_;
  std::queue<LogMsg> log_queue_;
};

// MessageInfrastructureLog object should be handled by multiple actors.
// Use std::shared_ptr.
using MessageInfrastructureLogPtr = std::shared_ptr<MessageInfrastructureLog>;

MessageInfrastructureLogPtr GetLogInstance();

void LogClear();

}  // namespace message_infrastructure

#endif  // CORE_MESSAGE_INFRASTRUCTURE_LOGGING_H_
