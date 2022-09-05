// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: BSD-3-Clause
// See: https://spdx.org/licenses/

#ifndef SHM_H_
#define SHM_H_

#include <memory>
#include <set>
#include <string>
namespace message_infrastructure {

class SharedMemory {
 public:
  SharedMemory() {}
  SharedMemory(const int &shmid, const size_t &mem_size);
  int GetShmid();
  void* MemMap();
 private:
  int shmid_;
  size_t size_;
  sem_t req_;
  sem_t ack_;
};

class SharedMemManager {
 public:
  SharedMemManager() {}
  int Alloc(const size_t &mem_size);
  SharedMemory AllocSharedMemory(const size_t &mem_size);
  int DeleteSharedMemory(const int &shmid);
  int Stop();

 private:
  std::set<int> shmids_;
  key_t key_ = 0xdead;
};

}  // namespace message_infrastructure

#endif  // SHM_H_
