// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: BSD-3-Clause
// See: https://spdx.org/licenses/

#include <message_infrastructure/csrc/channel/shmem/shm.h>

namespace message_infrastructure {

SharedMemory::SharedMemory(const size_t &mem_size,
                           const int &shmfd,
                           const int &key) {
  shmfd_ = shmfd;
  size_ = mem_size;
  req_name_ += std::to_string(key);
  ack_name_ += std::to_string(key);
}

SharedMemory::SharedMemory(const size_t &mem_size, const int &shmfd) {
  shmfd_ = shmfd;
  size_ = mem_size;
}

void SharedMemory::InitSemaphore() {
  req_ = sem_open(req_name_.c_str(), O_CREAT, 0644, 0);
  ack_ = sem_open(ack_name_.c_str(), O_CREAT, 0644, 1);
}

void SharedMemory::Start() {
}

void SharedMemory::Store(HandleFn store_fn) {
  sem_wait(ack_);
  store_fn(MemMap());
  sem_post(req_);
}

bool SharedMemory::Load(HandleFn consume_fn) {
  bool ret = false;
  if (!sem_trywait(req_)) {
    consume_fn(MemMap());
    sem_post(ack_);
    ret = true;
  }
  return ret;
}

void SharedMemory::Close() {
  LAVA_ASSERT_INT(sem_close(req_), 0);
  LAVA_ASSERT_INT(sem_close(ack_), 0);
}

std::string SharedMemory::GetReq() {
  return req_name_;
}

std::string SharedMemory::GetAck() {
  return ack_name_;
}

void* SharedMemory::MemMap() {
  return (data_ = mmap(NULL, size_, PROT_READ | PROT_WRITE,
                       MAP_SHARED, shmfd_, 0));
}


int SharedMemory::GetDataElem(int offset) {
  return static_cast<int>(*(reinterpret_cast<char*>(data_) + offset));
}

RwSharedMemory::RwSharedMemory(const size_t &mem_size,
                               const int &shmfd,
                               const int &key)
  : size_(mem_size), shmfd_(shmfd) {
  sem_name_ += std::to_string(key);
}

void RwSharedMemory::InitSemaphore() {
  sem_ = sem_open(sem_name_.c_str(), O_CREAT, 0644, 0);
}

void RwSharedMemory::Start() {
  sem_post(sem_);
}

void RwSharedMemory::Handle(HandleFn handle_fn) {
  sem_wait(sem_);
  handle_fn(GetData());
  sem_post(sem_);
}

void RwSharedMemory::Close() {
  LAVA_ASSERT_INT(sem_close(sem_), 0);
}

void* RwSharedMemory::GetData() {
  return (data_ = mmap(NULL, size_, PROT_READ | PROT_WRITE,
                       MAP_SHARED, shmfd_, 0));
}

void SharedMemManager::DeleteAllSharedMemory() {
  int result = 0;
  LAVA_DEBUG(LOG_SMMP, "Delete: Number of shm to free: %zd.\n",
             shm_fd_strs_.size());
  LAVA_DEBUG(LOG_SMMP, "Delete: Number of sem to free: %zd.\n",
             sem_strs_.size());
  for (auto const& it : shm_fd_strs_) {
    LAVA_ASSERT_INT(shm_unlink(it.second.c_str()), 0);
    LAVA_DEBUG(LOG_SMMP, "Shm fd and name close: %s %d\n",
               it.second.c_str(), it.first);
    LAVA_ASSERT_INT(close(it.first), 0);
  }
  for (auto it = sem_strs_.begin(); it != sem_strs_.end(); it++) {
    LAVA_ASSERT_INT(sem_unlink(it->c_str()), 0);
  }
  sem_strs_.clear();
  shm_fd_strs_.clear();
}

SharedMemManager SharedMemManager::smm_;

SharedMemManager& GetSharedMemManager() {
  SharedMemManager &smm = SharedMemManager::smm_;
  return smm;
}
}  // namespace message_infrastructure
