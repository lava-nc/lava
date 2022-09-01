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

#include <memory>
#include <vector>

namespace message_infrastructure {

class SharedMemory {
 public:
  SharedMemory(size_t mem_size, int shmid) {
    data_ = reinterpret_cast<char*> (shmat(shmid, NULL, 0));
    shmid_ = shmid;
  }
  int GetDataElem(int offset) {
    return static_cast<int> (data_[offset]);
  }
  void* MemMap() {
    data_ = reinterpret_cast<char*> (shmat(shmid_, NULL, 0));
    return data_;
  }
 private:
  int shmid_;
  char* data_;
};

using SharedMemoryPtr = SharedMemory*;

class SharedMemManager {
 public:
  SharedMemManager() {
    this->key_ = 0x5555;
  }
  explicit SharedMemManager(int key) {
    this->key_ = key;
  }
  int AllocSharedMemory(size_t mem_size) {
    int shmid = shmget(key_, mem_size, 0644|IPC_CREAT);
    if (shmid < 0)
      return -1;

    shms_.push_back(shmid);
    key_++;
    return shmid;
  }

  int DeleteSharedMemory(int shmid) {
    // Release specific shared memory
    int del_cnt = 0;
    for (auto it = shms_.begin(); it != shms_.end(); it++) {
      if ((*it) == shmid) {
        shms_.erase(it);
        del_cnt++;
      }
    }
    return del_cnt;
  }

  int Stop() {
    int stop_cnt = shms_.size();
    shms_.clear();
    return stop_cnt;
  }

 private:
  key_t key_ = 0;
  std::vector<int> shms_;
};

}  // namespace message_infrastructure

#endif  // SHM_H_
