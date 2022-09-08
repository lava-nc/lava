// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: BSD-3-Clause
// See: https://spdx.org/licenses/

#ifndef MESSAGE_INFRASTRUCTURE_LOGGING_H_
#define MESSAGE_INFRASTRUCTURE_LOGGING_H_

#include <stdio.h>
#define LOG_MP (1)  // log for multiprocessing
#define LOG_LAYER (1)


#define LAVA_LOG(_cond, _fmt, ...) { \
  if ((_cond)) { \
    printf("[CPP INFO]" _fmt, ## __VA_ARGS__); \
  } \
}

#define LAVA_DUMP(_cond, _fmt, ...) { \
  if ((_cond)) { \
    printf(_fmt, ## __VA_ARGS__); \
  } \
}

#define LAVA_DEBUG(_cond, _fmt, ...) { \
  if ((_cond)) { \
    printf("[CPP DEBUG]" _fmt, ## __VA_ARGS__); \
  } \
}

#define LAVA_LOG_WARN(_cond, _fmt, ...) { \
  if ((_cond)) { \
    printf("[CPP WARNING]" _fmt, __FUNCTION__, ## __VA_ARGS__); \
  } \
}

#define LAVA_LOG_ERR(_fmt, ...) { \
  printf("[CPP ERROR]" _fmt, __FUNCTION__, ## __VA_ARGS__); \
}

#endif  // MESSAGE_INFRASTRUCTURE_LOGGING_H_
