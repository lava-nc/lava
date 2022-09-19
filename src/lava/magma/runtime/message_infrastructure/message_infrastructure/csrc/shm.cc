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
#include "message_infrastructure_logging.h"

namespace message_infrastructure {

#define SHM_FLAG O_RDWR | O_CREAT
#define SHM_MODE S_IRUSR | S_IWUSR | S_IRGRP | S_IWGRP | S_IROTH | S_IWOTH

SharedMemory::SharedMemory(const size_t &mem_size, const int &shmfd) {
  shmfd_ = shmfd;
  size_ = mem_size;
}
void SharedMemory::InitSemaphore() {
  sem_init(&req_, 1, 0);
  sem_init(&ack_, 1, 0);
}
int SharedMemory::GetShmfd() {
  return shmfd_;
}
void* SharedMemory::MemMap() {
  return (data_ = mmap(NULL, size_, PROT_READ | PROT_WRITE, MAP_SHARED, shmfd_, 0));
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
  std::string str = shm_str_ + std::to_string(key_++);
  int shmfd = shm_open(str.c_str(), SHM_FLAG, SHM_MODE);
  if (shmfd == -1) {
    LAVA_LOG_ERR("Create shared memory object failed.\n");
    exit(-1);
  }
  int err = ftruncate(shmfd, mem_size);
  if (err == -1) {
    LAVA_LOG_ERR("Resize shared memory segment failed.\n");
    exit(-1);
  }
  shm_strs_.insert(str);
  return shmfd;
}

SharedMemoryPtr SharedMemManager::AllocChannelSharedMemory(const size_t &mem_size) {
  std::string str = shm_str_ + std::to_string(key_++);
  int shmfd = shm_open(str.c_str(), SHM_FLAG, SHM_MODE);
  if (shmfd == -1) {
    LAVA_LOG_ERR("Create shared memory object failed.\n");
    exit(-1);
  }
  int err = ftruncate(shmfd, mem_size);
  if (err == -1) {
    LAVA_LOG_ERR("Resize shared memory segment failed.\n");
    exit(-1);
  }
  shm_strs_.insert(str);
  SharedMemoryPtr shm = std::make_shared<SharedMemory>(mem_size, shmfd);
  shm->InitSemaphore();

  return shm;
}

void SharedMemManager::DeleteSharedMemory(const std::string &shm_str) {
  // Release specific shared memory
  if (shm_strs_.find(shm_str) != shm_strs_.end()) {
    shm_unlink(shm_str.c_str());
    shm_strs_.erase(shm_str);
  } else {
    LAVA_LOG_WARN(LOG_SMMP,"There is no shmem whose name is %s.\n", shm_str.c_str());
  }
}

SharedMemManager::~SharedMemManager() {
  int result = 0;
  for (auto it = shm_strs_.begin(); it != shm_strs_.end(); it++) {
    shm_unlink(it->c_str());
  }
  shm_strs_.clear();
}

SharedMemManager SharedMemManager::smm_;

SharedMemManager& GetSharedMemManager() {
  SharedMemManager &smm = SharedMemManager::smm_;
  return smm;
}
}  // namespace message_infrastructure
