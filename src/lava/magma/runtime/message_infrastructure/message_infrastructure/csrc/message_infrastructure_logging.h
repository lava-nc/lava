// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: BSD-3-Clause
// See: https://spdx.org/licenses/

#ifndef MESSAGE_INFRASTRUCTURE_LOGGING_H_
#define MESSAGE_INFRASTRUCTURE_LOGGING_H_

#include <memory>
#include <mutex>
#include <thread>
#include <iostream> 
#include <sstream> 
#include <fstream>
#include <string>

#include <stdio.h>
#include <memory.h>
#include <time.h>

#if _WIN32
#inlcude <process.h>
#define getpid() _getpid()

#else //__linux__ & __APPLE__
#include <unistd.h>
#include <sys/types.h>

#endif



// #define LOG_MP (0)  // log for multiprocessing
// #define LOG_ACTOR (0)
// #define LOG_LAYER (0)
#ifdef CMAKE_DEBUG
#define DEBUG_MODE (1)
#else
#define DEBUG_MODE (0)
#endif
// #define LOG_SMMP (0)  // log for shmemport

#define DEBUG_LOG_MODULE "lava_message_infrastructure"
#define DEBUG_LOG_FILE_SUFFIX "log"

//#if DEBUG_MODE
#define DEBUG_LOG_TO_FILE (1)

#define MAX_SIZE_PER_LOG_MSG (1024)
#define MAX_SIZE_LOG (1)
#define MAX_SIZE_LOG_TIME (64)

#define LOG_MSG_SUBSTITUTION "!This message was displayed due to the failure of the malloc of this log message!\n"
#define LOG_GET_TIME_FAIL "Get log time failed."

#if DEBUG_MODE && DEBUG_LOG_TO_FILE
// #define DEBUG_LOG_PRINT(...) do {\
//   int length = 0;\
//   char * log_data = (char*)malloc(sizeof(char)*MAX_SIZE_PER_LOG_MSG);\
//   struct logMsg * log_msg = (struct logMsg*)malloc(sizeof(struct logMsg));\
//   char * log_time = getTime();\
//   if (log_data != NULL)\
//     length = std::snprintf(log_data, MAX_SIZE_PER_LOG_MSG, ## __VA__ARGS__);\
//   if (log_data == NULL || log_msg == NULL || log_time == NULL || length < 0){\
//     if (log_time != NULL){\
//       std::printf(buf);\
//       std::printf(" ");\
//     } \
//     std::printf(## __VA_ARGS__);\
//     free(log_time);\
//     free(log_msg);\
//     free(log_data);\
//   }\
//   else{\
//     log_msg.msg_data = log_data;\
//     log_msg.msg_time = log_time;\
//     getLogInstance().log(log_msg);\
//   }\
// }while(0)
#define DEBUG_LOG_PRINT(_level,_module, _fmt, ...) do {\
  int length = 0;\
  char * log_data = (char*)malloc(sizeof(char)*MAX_SIZE_PER_LOG_MSG);\
  if (log_data != NULL)\
    length = std::snprintf(log_data, MAX_SIZE_PER_LOG_MSG, _fmt,  ## __VA_ARGS__);\
  if (log_data == NULL || length <0){\
    getLogInstance()->log(new LogMsg(std::string(LOG_MSG_SUBSTITUTION), __FILE__, __LINE__, _module, _level));\
  }else{\
    getLogInstance()->log(new LogMsg(std::string(log_data), __FILE__, __LINE__, _module, _level));\
  }\
  free(log_data);\
}while(0)

#else
#define DEBUG_LOG_PRINT(_level,_module, ...) //std::printf(__VA_ARGS__)
#endif




#define LAVA_LOG(_module, _fmt, ...) { \
  if ((_module)) { \
    DEBUG_LOG_PRINT("[CPP INFO]", _module, _fmt, ## __VA_ARGS__);\
  } \
}

#define LAVA_DUMP(_module, _fmt, ...) { \
  if ((_module && DEBUG_MODE)) { \
    DEBUG_LOG_PRINT("[CPP DUMP]", _module, _fmt, ## __VA_ARGS__);\
  } \
}

#define LAVA_DEBUG(_module, _fmt, ...) { \
  if ((_module && DEBUG_MODE)) { \
    DEBUG_LOG_PRINT("[CPP DEBUG]", _module, _fmt, ## __VA_ARGS__);\
  } \
}

#define LAVA_LOG_WARN(_module, _fmt, ...) { \
  if ((_module)) { \
    DEBUG_LOG_PRINT("[CPP WARN]", _module, _fmt, ## __VA_ARGS__);\
  } \
}

#define LAVA_LOG_ERR(_module, _fmt, ...) { \
  DEBUG_LOG_PRINT("[CPP ERROR]", _module, _fmt, ## __VA_ARGS__);\
}

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
  LOG_MP,       // log for multiprocessing
  LOG_ACTOR,
  LOG_LAYER,
  LOG_SMMP      // log for shmemport
};



// struct LogMsg{
//   char * msg_time;
//   char * msg_data;
//   char * msg_level;
//   char * msg_module;
// };

class LogMsg{
  public:
    LogMsg(const std::string&, const char* log_file, int log_line, enum Log_Module log_module, const char* log_level);
    // LogMsg(const char*);
    LogMsg();
    // ~LogMsg();
    std::string getEntireLogMsg(int pid);

  private:
  // std::string msg_time_ = nullptr; this way will cause a abort trouble...
    std::string msg_time_;
    std::string msg_data_;
    std::string msg_level_;
    std::string msg_module_;
    std::string msg_file_ ;
    std::string msg_pid_;
    int msg_line_ = 0;
    // std::string msg_tid = NULL;
};

class Log{
  public:
    Log();
    ~Log();
    void log(LogMsg*);
    void clear();
    void write_down();
    

  private:
    int current_length_ = 0;
    const int max_length_ = MAX_SIZE_LOG;
    std::mutex log_lock_;
    LogMsg * log_buffer_[MAX_SIZE_LOG];
    // static Log log_instance_;
};

using LogPtr = std::shared_ptr<Log>;



LogPtr getLogInstance();

std::string getTime();

signed int getPid();

void LogClear();

} // namespace message_infrastructure

#endif  // MESSAGE_INFRASTRUCTURE_LOGGING_H_


// { if ((LOG_LAYER)) {
//    do {
//      int length = 0;
//     char * log_data = (char*)malloc(sizeof(char)*(1024)); 
//     if (log_data != __null)
//       length = std::snprintf(log_data, (1024),"start MDataFromObject\n"); 
//       if (log_data == __null || length <0){
//        getLogInstance()->log(new LogMsg(std::string("!This message was displayed due to the failure of the malloc of this log message!\n"), "/home/wye/whye/hongda28/lava/src/lava/magma/runtime/message_infrastructure/message_infrastructure/csrc/port_proxy.cc", 94, LOG_LAYER, "[CPP INFO]"));
//       }
//       else{ 
//         getLogInstance()->log(new LogMsg(std::string(log_data), "/home/wye/whye/hongda28/lava/src/lava/magma/runtime/message_infrastructure/message_infrastructure/csrc/port_proxy.cc", 94, LOG_LAYER, "[CPP INFO]"));
//       } 
//       free(log_data);
//     }while(0); 
//     std::printf("[CPP INFO]" "start MDataFromObject\n"); 
//     } 
//   }