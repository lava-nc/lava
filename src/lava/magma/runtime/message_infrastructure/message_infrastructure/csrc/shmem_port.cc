// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: BSD-3-Clause
// See: https://spdx.org/licenses/

#include <pthread.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <sys/shm.h>
#include <memory.h>
#include <fcntl.h>
#include <semaphore.h>
#include <unistd.h>
#include <thread>
#include <mutex>
#include <memory>
#include <string>
#include <condition_variable>

#include "shmem_port.h"
#include "shm.h"
#include "utils.h"
#include "message_infrastructure_logging.h"

namespace message_infrastructure {
ShmemRecvQueue::ShmemRecvQueue(const std::string& name,
                          const size_t &size,
                          const size_t &nbytes) {
  size_ = size;
  name_ = name;
  nbytes_ = nbytes;
  array_.reserve(size_);
}

void ShmemRecvQueue::Push(void* src) {
  auto const curr_write_index = write_index_.load(std::memory_order_acquire);
  auto const curr_read_index = read_index_.load(std::memory_order_acquire);
  auto next_write_index = curr_write_index + 1;
  if (next_write_index == size_) {
      next_write_index = 0;
  }
  if (next_write_index == curr_read_index) {
    auto next_read_index = curr_read_index + 1;
    drop_array_.push_back(array_[curr_read_index]);
    if(next_read_index == size_) {
      next_read_index = 0;
    }
    read_index_.store(next_read_index, std::memory_order_release);
    LAVA_LOG_WARN(LOG_SMMP, "Drop data in ShmemChannel %s\n", name_.c_str());
  }
  void *ptr = malloc(nbytes_);
  memcpy(ptr, src, nbytes_);

  array_[curr_write_index] = ptr;
  write_index_.store(next_write_index, std::memory_order_release);
}

void ShmemRecvQueue::Pop() {
  if(Empty()) {
    LAVA_LOG_WARN(LOG_SMMP, "ShmemChannel %s is empty.\n", name_.c_str());
    return;
  }
  auto const curr_read_index = read_index_.load(std::memory_order_acquire);
  auto next_read_index = curr_read_index + 1;
  if(next_read_index == size_) {
    next_read_index = 0;
  }
  read_index_.store(next_read_index, std::memory_order_release);
  return;
}

void* ShmemRecvQueue::Front() {
  if(Empty()) {
    LAVA_LOG_WARN(LOG_SMMP, "ShmemChannel is empty.\n");
    return NULL;
  }
  auto const curr_read_index = read_index_.load(std::memory_order_acquire);
  void *ptr = array_[curr_read_index];
  return ptr;
}

void* ShmemRecvQueue::FrontPop() {
  if(Empty()) {
    LAVA_LOG_WARN(LOG_SMMP, "ShmemChannel is empty.\n");
    return NULL;
  }
  auto const curr_read_index = read_index_.load(std::memory_order_acquire);
  void *ptr = array_[curr_read_index];
  auto next_read_index = curr_read_index + 1;
  if(next_read_index == size_) {
    next_read_index = 0;
  }
  read_index_.store(next_read_index, std::memory_order_release);
  return ptr;
}

bool ShmemRecvQueue::Probe() {
  return ((write_index_.load(std::memory_order_acquire) + 1) % size_ ==  \
           read_index_.load(std::memory_order_acquire));
}

bool ShmemRecvQueue::Empty() {
  return (write_index_.load(std::memory_order_acquire) ==  \
                            read_index_.load(std::memory_order_acquire));
}

void ShmemRecvQueue::Free() {
  if(!Empty()) {
    auto const curr_read_index = read_index_.load(std::memory_order_acquire);
    auto const curr_write_index = write_index_.load(std::memory_order_acquire);
    int max, min;
    if(curr_read_index < curr_write_index) {
      max = curr_write_index;
      min = curr_read_index;
    } else {
      min = curr_write_index + 1;
      max = curr_read_index + 1;
    }
    for(int i = min; i < max; i++)
      if(array_[i]) free(array_[i]);
    read_index_.store(0, std::memory_order_release);
    write_index_.store(0, std::memory_order_release);
  }
  for(auto i = 0; i < drop_array_.size(); i++)
    free(drop_array_[i]);
  drop_array_.clear();
}

ShmemRecvQueue::~ShmemRecvQueue() {
  if(!Empty()) {
    auto const curr_read_index = read_index_.load(std::memory_order_acquire);
    auto const curr_write_index = write_index_.load(std::memory_order_acquire);
    int max, min;
    if(curr_read_index < curr_write_index) {
      max = curr_write_index;
      min = curr_read_index;
    } else {
      min = curr_write_index + 1;
      max = curr_read_index + 1;
    }
    for(int i = min; i < max; i++)
      if(array_[i]) free(array_[i]);
  }
  for(auto i = 0; i < drop_array_.size(); i++)
    free(drop_array_[i]);
}

ShmemSendPort::ShmemSendPort(const std::string &name,
                SharedMemoryPtr shm,
                const size_t &size,
                const size_t &nbytes) {
  name_ = name;
  shm_ = shm;
  nbytes_ = nbytes;
  size_ = size;

  done_ = false;

  array_ = shm_->MemMap();
}

void ShmemSendPort::Start() {
  sem_post(&shm_->GetAckSemaphore());
  // ack_callback_thread_ = std::make_shared<std::thread>(&message_infrastructure::ShmemSendPort::AckCallback, this);
}

int ShmemSendPort::Send(MetaDataPtr metadata) {
  char* cptr = (char*)array_;
  sem_wait(&shm_->GetAckSemaphore());
  memcpy(cptr, metadata.get(), offsetof(MetaData, mdata));
  cptr+=offsetof(MetaData, mdata);
  memcpy(cptr, metadata->mdata, nbytes_);
  sem_post(&shm_->GetReqSemaphore());
  return 0;
}

void ShmemSendPort::Join() {
  done_ = true;
  // ack_callback_thread_->join();
}

int ShmemSendPort::AckCallback() {
  while(!done_) {
  // Todo(hexu1) : CspSelector.Observer
  }
  return 0;
}

std::string ShmemSendPort::Name() {
  return name_;
}

size_t ShmemSendPort::Size() {
  return size_;
}

ShmemRecvPort::ShmemRecvPort(const std::string &name,
                SharedMemoryPtr shm,
                const size_t &size,
                const size_t &nbytes) {
  name_ = name;
  shm_ = shm;
  nbytes_ = nbytes;
  size_ = size;
  done_ = false;
  array_ = shm_->MemMap();
  queue_ = std::make_shared<ShmemRecvQueue>(name_, size_, nbytes);
}

void ShmemRecvPort::Start() {
  // req_callback_thread_ = std::make_shared<std::thread>(&message_infrastructure::ShmemRecvPort::ReqCallback, this);
  recv_queue_thread_ = std::make_shared<std::thread>(&message_infrastructure::ShmemRecvPort::QueueRecv, this);
}

void ShmemRecvPort::QueueRecv() {
  while(!done_) {
    sem_wait(&shm_->GetReqSemaphore());
    queue_->Push(array_);
    sem_post(&shm_->GetAckSemaphore());
  }
  queue_->Free();
}

bool ShmemRecvPort::Probe() {
  return queue_->Probe();
}

MetaDataPtr ShmemRecvPort::Recv() {
  char* cptr = (char*)queue_->FrontPop();
  MetaData *metadata_ptr = (MetaData*)cptr;
  cptr+=offsetof(MetaData, mdata);
  metadata_ptr->mdata = cptr;
  MetaDataPtr metadata(metadata_ptr);
  return metadata;
}

void ShmemRecvPort::Join() {
  done_ = true;
  // req_callback_thread_->join();
  recv_queue_thread_->join();
}

void* ShmemRecvPort::Peek() {
  return queue_->Front();
}

int ShmemRecvPort::ReqCallback() {
  while(!done_) {
  // Todo(hexu1) : CspSelector.Observer
  }
  return 0;
}

}  // namespace message_infrastructure
