// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: BSD-3-Clause
// See: https://spdx.org/licenses/

#include <channel/shmem/shm.h>

namespace message_infrastructure {

SharedMemory::SharedMemory(const size_t &mem_size,
                           void* mmap,
                           const int &key) {
  data_ = mmap;
  size_ = mem_size;
  req_name_ += std::to_string(key);
  ack_name_ += std::to_string(key);
}

SharedMemory::SharedMemory(const size_t &mem_size, void* mmap) {
  data_ = mmap;
  size_ = mem_size;
}

SharedMemory::~SharedMemory() {
}

void SharedMemory::InitSemaphore(sem_t *req, sem_t *ack) {
  req_ = req;
  ack_ = ack;
}

void SharedMemory::Start() {
}

void SharedMemory::Store(HandleFn store_fn) {
  sem_wait(ack_);
  store_fn(data_);
  sem_post(req_);
}

bool SharedMemory::Load(HandleFn consume_fn) {
  bool ret = false;
  if (!sem_trywait(req_)) {
    consume_fn(data_);
    sem_post(ack_);
    ret = true;
  }
  return ret;
}

void SharedMemory::BlockLoad(HandleFn consume_fn) {
  sem_wait(req_);
  consume_fn(data_);
  sem_post(ack_);
}

void SharedMemory::Read(HandleFn consume_fn) {
  sem_wait(req_);
  consume_fn(data_);
  sem_post(req_);
}

bool SharedMemory::TryProbe() {
  int val;
  sem_getvalue(req_, &val);
  return val > 0;
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

int SharedMemory::GetDataElem(int offset) {
  return static_cast<int>(*(reinterpret_cast<char*>(data_) + offset));
}

RwSharedMemory::RwSharedMemory(const size_t &mem_size,
                               void* mmap,
                               const int &key)
  : size_(mem_size), data_(mmap) {
  sem_name_ += std::to_string(key);
}

RwSharedMemory::~RwSharedMemory() {
  munmap(data_, size_);
}

void RwSharedMemory::InitSemaphore() {
  sem_ = sem_open(sem_name_.c_str(), O_CREAT, 0644, 0);
}

void RwSharedMemory::Start() {
  sem_post(sem_);
}

void RwSharedMemory::Handle(HandleFn handle_fn) {
  sem_wait(sem_);
  handle_fn(data_);
  sem_post(sem_);
}

void RwSharedMemory::Close() {
  LAVA_ASSERT_INT(sem_close(sem_), 0);
}

void SharedMemManager::DeleteAllSharedMemory() {
  if (alloc_pid_ != getpid())
    return;
  LAVA_DEBUG(LOG_SMMP, "Delete: Number of shm to free: %zd.\n",
             shm_fd_strs_.size());
  LAVA_DEBUG(LOG_SMMP, "Delete: Number of sem to free: %zd.\n",
             sem_p_strs_.size());
  for (auto const& it : shm_fd_strs_) {
    LAVA_ASSERT_INT(shm_unlink(it.second.c_str()), 0);
    LAVA_DEBUG(LOG_SMMP, "Shm fd and name close: %s %d\n",
               it.second.c_str(), it.first);
    LAVA_ASSERT_INT(close(it.first), 0);
  }
  for (auto const& it : shm_mmap_) {
    LAVA_ASSERT_INT(munmap(it.first, it.second), 0);
  }
  for (auto const& it : sem_p_strs_) {
    LAVA_ASSERT_INT(sem_close(it.first), 0);
    LAVA_ASSERT_INT(sem_unlink(it.second.c_str()), 0);
  }
  sem_p_strs_.clear();
  shm_fd_strs_.clear();
  shm_mmap_.clear();
}

SharedMemManager SharedMemManager::smm_;

SharedMemManager& GetSharedMemManagerSingleton() {
  SharedMemManager &smm = SharedMemManager::smm_;
  return smm;
}
}  // namespace message_infrastructure
