// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: BSD-3-Clause
// See: https://spdx.org/licenses/

#include "shm.h"

namespace message_infrastructure {

SharedMemory::SharedMemory() {
  
}
SharedMemory::SharedMemory(int shmid) {
  shmid_ = shmid;
}
int SharedMemory::GetShmid() {
  return shmid_;
}
int SharedMemory::SetShmid(int shmid) {
  shmid_ = shmid;
}
void* SharedMemory::MemMap() {
  // Todo(hexu1) : return pointer of shmem in user space
}


int SharedMemManager::AllocSharedMemory(const std::string &src_name, const size_t mem_size) {
  std::string shm_name = src_name + "_shm";

  int shmid = shm_open(shm_name.c_str(),
    O_RDWR | O_CREAT, S_IRUSR | S_IWUSR | S_IRGRP | S_IWGRP);

  if (shmid == -1) {
    printf("Create shared memory object fail..\n");
    exit(-1);
  }

  int err_1 = ftruncate(shmid, mem_size);

  if (err_1 == -1) {
    printf("Resize shared memory segment fail..\n");
    exit(-1);
  }

  struct stat stat_shmem;

  err_1 = fstat(shmid, &stat_shmem);
  if (err_1 == -1) {
    printf("Detect shared memory segment size fail..\n");
    exit(-1);
  }

  shms_.insert(shm_name);
  return shmid;
}

int SharedMemManager::AllocSharedMemory(const size_t mem_size) {
  int shmid = shmget(key_, mem_size, 0644|IPC_CREAT);
  if (shmid < 0)
    return -1;

  shmids_.insert(shmid);
  key_++;
  return shmid;
}

int SharedMemManager::DeleteSharedMemory(const int shmid) {
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

int SharedMemManager::DeleteSharedMemory(const std::string &src_name) {
  // Release specific shared memory
  std::string shm_name = src_name + "_shm";
  int result = -1;
  if (shms_.find(shm_name) != shms_.end()) {
    result = shm_unlink(shm_name.c_str());
    shms_.erase(shm_name);
  } else {
    printf("There is no shmem named %s.\n", shm_name.c_str());
  }
  return result;
}

int SharedMemManager::Stop() {
  int result = 0;
  for (auto it = shms_.begin(); it != shms_.end(); it++) {
    result = shm_unlink(it->c_str());
    if (result)
      exit(-1);
  }
  shms_.clear();
  for (auto it = shmids_.begin(); it != shmids_.end(); it++) {
    result = shmctl(*it, IPC_RMID, NULL);
    if (result)
      exit(-1);
  }
  shmids_.clear();
  return result;
}
} // message_infrastructure
