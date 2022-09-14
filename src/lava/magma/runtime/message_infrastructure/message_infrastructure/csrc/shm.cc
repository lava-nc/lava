// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: BSD-3-Clause
// See: https://spdx.org/licenses/

#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <sys/shm.h>
#include <fcntl.h>
#include <semaphore.h>
#include <unistd.h>

#include "shm.h"

namespace message_infrastructure {

SharedMemory::SharedMemory(const size_t &mem_size, const int &shmid) {
  shmid_ = shmid;
  size_ = mem_size;
  data_ = shmat(shmid, NULL, 0);
}
void SharedMemory::InitSemaphore() {
  sem_init(&req_, 1, 0);
  sem_init(&ack_, 1, 0);
}
int SharedMemory::GetShmid() {
  return shmid_;
}
void* SharedMemory::MemMap() {
  return (data_ = shmat(shmid_, NULL, 0));
}
sem_t& SharedMemory::GetReqSemaphore() {
  return req_;
}
sem_t& SharedMemory::GetAckSemaphore() {
  return ack_;
}
int SharedMemory::GetDataElem(int offset) {
  return static_cast<int> (*(((char*)data_) + offset));
}

int SharedMemManager::AllocSharedMemory(const size_t &mem_size) {
  int shmid = shmget(key_++, mem_size, 0644|IPC_CREAT);
  if (shmid < 0)
    return -1;

  shmids_.insert(shmid);
  return shmid;
}
SharedMemoryPtr SharedMemManager::AllocChannelSharedMemory(const size_t &mem_size) {
  int shmid = shmget(key_++, mem_size, 0644|IPC_CREAT);
  if (shmid < 0)
    exit(-1); // Log_Error

  SharedMemoryPtr shm = std::make_shared<SharedMemory>(mem_size, shmid);
  shm->InitSemaphore();

  shmids_.insert(shmid);
  return shm;
}

int SharedMemManager::DeleteSharedMemory(const int &shmid) {
  // Release specific shared memory
  int result = -1;
  if (shmids_.find(shmid) != shmids_.end()) {
    result = shmctl(shmid, IPC_RMID, NULL);
    shmids_.erase(shmid);
  } else {
    printf("There is no shmem whose id is %i.\n", shmid);
  }
  return result;
}

SharedMemManager::~SharedMemManager() {
  int result = 0;
  for (auto it = shmids_.begin(); it != shmids_.end(); it++) {
    result = shmctl(*it, IPC_RMID, NULL);
    if (result)
      exit(-1);
  }
  shmids_.clear();
}

SharedMemManager SharedMemManager::smm_;

SharedMemManager& GetSharedMemManager() {
  SharedMemManager &smm = SharedMemManager::smm_;
  return smm;
}
}  // namespace message_infrastructure
