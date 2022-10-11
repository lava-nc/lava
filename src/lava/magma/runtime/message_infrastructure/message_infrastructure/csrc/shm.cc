// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: BSD-3-Clause
// See: https://spdx.org/licenses/

#include "shm.h"

namespace message_infrastructure {

SharedMemory::SharedMemory(const size_t &mem_size, const int &shmfd, const int &key) {
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
  if (!sem_trywait(req_))
  {
      consume_fn(MemMap());
      sem_post(ack_);
      ret = true;
  }
  return ret;
}

void SharedMemory::Close() {
  sem_close(req_);
  sem_close(ack_);
}

std::string SharedMemory::GetReq() {
  return req_name_;
}

std::string SharedMemory::GetAck() {
  return ack_name_;
}

void* SharedMemory::MemMap() {
  return (data_ = mmap(NULL, size_, PROT_READ | PROT_WRITE, MAP_SHARED, shmfd_, 0));
}


int SharedMemory::GetDataElem(int offset) {
  return static_cast<int> (*(((char*)data_) + offset));
}

SharedMemory::~SharedMemory() {
  Close();
}

RwSharedMemory::RwSharedMemory(const size_t &mem_size, const int &shmfd, const int &key)
  : size_(mem_size), shmfd_(shmfd)
{
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
  sem_close(sem_);
}

void* RwSharedMemory::GetData() {
  return (data_ = mmap(NULL, size_, PROT_READ | PROT_WRITE, MAP_SHARED, shmfd_, 0));
}

RwSharedMemory::~RwSharedMemory() {
  Close();
  sem_unlink(sem_name_.c_str());
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
  for (auto it = sem_strs_.begin(); it != sem_strs_.end(); it++) {
    sem_unlink(it->c_str());
  }
  sem_strs_.clear();
  shm_strs_.clear();
}

SharedMemManager SharedMemManager::smm_;

SharedMemManager& GetSharedMemManager() {
  SharedMemManager &smm = SharedMemManager::smm_;
  return smm;
}
}  // namespace message_infrastructure
