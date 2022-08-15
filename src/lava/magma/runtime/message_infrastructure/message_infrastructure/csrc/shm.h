// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: BSD-3-Clause
// See: https://spdx.org/licenses/

#ifndef SHM_H_
#define SHM_H_

#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <sys/shm.h>
#include <fcntl.h>
#include <semaphore.h>
#include <unistd.h>

#include <vector>

namespace message_infrastructure {

class SharedMemory {
};

class SharedMemManager {
 public:
  int AllocSharedMemory(size_t mem_size) {
    int shmid = shmget(key_, mem_size, 0644|IPC_CREAT);
    if (shmid < 0)
      return -1;

    shms_.push_back(shmid);
    key_++;
    return shmid;
  }

  int DeleteSharedMemory(int key) {
    // Release specific shared memroy
    return 0;
  }
  
 private:
  key_t key_ = 0xdead;
  std::vector<int> shms_;
};

}  // namespace message_infrastructure

#endif  // SHM_H_
