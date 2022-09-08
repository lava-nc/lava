// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: BSD-3-Clause
// See: https://spdx.org/licenses/

#ifndef UTILS_H_
#define UTILS_H_

#include <sys/types.h>
#include <sys/shm.h>
#include <fcntl.h>
#include <semaphore.h>

#include <vector>

namespace message_infrastructure {

enum ProcessType {
  ErrorProcess = 0,
  ParentProcess = 1,
  ChildProcess = 2
};

enum ChannelType {
  SHMEMCHANNEL = 0,
  RPCCHANNEL = 1,
  DDSCHANNEL = 2
};

#define ACC_MODE (S_IRUSR | S_IWUSR | S_IXUSR | S_IRGRP | S_IWGRP | \
  S_IXGRP | S_IROTH | S_IWOTH | S_IXOTH)

#define CREAT_FLAG (O_CREAT | O_RDWR)

struct MetaData {
  int64_t nd;
  int64_t type;
  int64_t elsize;
  int64_t total_size;
  std::vector<int64_t> dims;
  std::vector<int64_t> strides;
  void* mdata;
};

}  // namespace message_infrastructure

#endif  // UTILS_H_
