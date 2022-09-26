// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: BSD-3-Clause
// See: https://spdx.org/licenses/

#ifndef MESSAGE_INFRASTRUCTURE_LOGGING_H_
#define MESSAGE_INFRASTRUCTURE_LOGGING_H_

#include <stdio.h>
#define LOG_MP (0)  // log for multiprocessing
#define LOG_ACTOR (0)
#define LOG_LAYER (0)
#define DEBUG_MODE (0)
#define LOG_SMMP (0)  // log for shmemport

#define LAVA_LOG(_cond, _fmt, ...) { \
  if ((_cond)) { \
    printf("[CPP INFO]" _fmt, ## __VA_ARGS__); \
  } \
}

#define LAVA_DUMP(_cond, _fmt, ...) { \
  if ((_cond && DEBUG_MODE)) { \
    printf(_fmt, ## __VA_ARGS__); \
  } \
}

#define LAVA_DEBUG(_cond, _fmt, ...) { \
  if ((_cond && DEBUG_MODE)) { \
    printf("[CPP DEBUG]" _fmt, ## __VA_ARGS__); \
  } \
}

#define LAVA_LOG_WARN(_cond, _fmt, ...) { \
  if ((_cond)) { \
    printf("[CPP WARNING] %s " _fmt, __FUNCTION__, ## __VA_ARGS__); \
  } \
}

#define LAVA_LOG_ERR(_fmt, ...) { \
  printf("[CPP ERROR] %s " _fmt, __FUNCTION__, ## __VA_ARGS__); \
}

#endif  // MESSAGE_INFRASTRUCTURE_LOGGING_H_
