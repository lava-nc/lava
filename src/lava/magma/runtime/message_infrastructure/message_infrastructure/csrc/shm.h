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
#include <set>
#include <string>

namespace message_infrastructure {

class SharedMemory {
 public:
  SharedMemory();
  explicit SharedMemory(int shmid);
  int GetShmid();
  int SetShmid(int shmid);
  void* MemMap();
 private:
  int shmid_;
  size_t size_;
};

using SharedMemoryPtr = SharedMemory*;

class SharedMemManager {
 public:
  int AllocSharedMemory(const std::string &src_name, const size_t mem_size);
  int AllocSharedMemory(const size_t mem_size);

  int DeleteSharedMemory(const int shmid);
  int DeleteSharedMemory(const std::string &src_name);

  int Stop();

 private:
  std::set<std::string> shms_;
  std::set<int> shmids_;
  key_t key_ = 0xdead;
};

}  // namespace message_infrastructure

#endif  // SHM_H_
