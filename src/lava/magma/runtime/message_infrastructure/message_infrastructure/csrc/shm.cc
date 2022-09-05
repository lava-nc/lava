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

SharedMemory::SharedMemory(const int &shmid, const size_t &mem_size) {
  shmid_ = shmid;
  size_ = mem_size;
  sem_init(&req_, 1, 0);
  sem_init(&ack_, 1, 0);
}
int SharedMemory::GetShmid() {
  return shmid_;
}
void* SharedMemory::MemMap() {
  // Todo(hexu1) : return pointer of shmem in user space
}

SharedMemory SharedMemManager::AllocSharedMemory(const size_t &mem_size) {
  int shmid = shmget(key_, mem_size, 0644|IPC_CREAT);
  if (shmid < 0)
    exit(-1); // Log_Error

  SharedMemory shm(shmid, mem_size);

  shmids_.insert(shmid);
  key_++;
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

int SharedMemManager::Stop() {
  int result = 0;
  for (auto it = shmids_.begin(); it != shmids_.end(); it++) {
    result = shmctl(*it, IPC_RMID, NULL);
    if (result)
      exit(-1);
  }
  shmids_.clear();
  return result;
}
} // message_infrastructure
