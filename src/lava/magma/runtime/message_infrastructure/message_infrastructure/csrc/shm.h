// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: BSD-3-Clause
// See: https://spdx.org/licenses/

#ifndef SHM_H_
#define SHM_H_

#include <semaphore.h>
#include <memory>
#include <set>
#include <string>
#include <atomic>
namespace message_infrastructure {

class SharedMemory {
 public:
  SharedMemory() {}
  SharedMemory(const size_t &mem_size, const int &shmid);
  int GetShmid();
  sem_t& GetReqSemaphore();
  sem_t& GetAckSemaphore();
  void* MemMap();
  void InitSemaphore();
  int GetDataElem(int offset);
 private:
  int shmid_;
  size_t size_;
  sem_t req_;
  sem_t ack_;
  void* data_;
};

using SharedMemoryPtr = std::shared_ptr<SharedMemory>;

class SharedMemManager {
 public:
  ~SharedMemManager();
  int AllocSharedMemory(const size_t &mem_size);
  SharedMemoryPtr AllocChannelSharedMemory(const size_t &mem_size);
  int DeleteSharedMemory(const int &shmid);
  friend SharedMemManager& GetSharedMemManager();

 private:
  SharedMemManager() {}
  std::set<int> shmids_;
  std::atomic<key_t> key_ {0x1111};
  static SharedMemManager smm_;
};

SharedMemManager& GetSharedMemManager();

using SharedMemManagerPtr = std::shared_ptr<SharedMemManager>;

}  // namespace message_infrastructure

#endif  // SHM_H_
