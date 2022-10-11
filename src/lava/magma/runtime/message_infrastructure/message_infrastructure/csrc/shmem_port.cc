// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: BSD-3-Clause
// See: https://spdx.org/licenses/

#include <semaphore.h>
#include <unistd.h>
#include <thread>
#include <mutex>
#include <memory>
#include <string>
#include <condition_variable>
#include <cassert>
#include <cstring>

#include "shmem_port.h"
#include "utils.h"
#include "message_infrastructure_logging.h"

namespace message_infrastructure {
ShmemRecvQueue::ShmemRecvQueue(const std::string& name,
                          const size_t &size,
                          const size_t &nbytes)
  : name_(name), size_(size), nbytes_(nbytes), read_index_(0), write_index_(0), done_(false)
{
  array_.reserve(size_);
}

void ShmemRecvQueue::Push(void* src) {
  auto const curr_write_index = write_index_.load(std::memory_order_relaxed);
  auto next_write_index = curr_write_index + 1;
  if (next_write_index == size_) {
      next_write_index = 0;
  }

  if (next_write_index != read_index_.load(std::memory_order_acquire)) {
    void *ptr = malloc(nbytes_);
    std::memcpy(ptr, src, nbytes_);
    array_[curr_write_index] = ptr;
    write_index_.store(next_write_index, std::memory_order_release);
  }
}

void* ShmemRecvQueue::Pop(bool block) {
  while(block && Empty()) {
    helper::Sleep();
    if(done_)
      return NULL;
  }
  auto const curr_read_index = read_index_.load(std::memory_order_relaxed);
  assert(curr_read_index != write_index_.load(std::memory_order_acquire));
  void *ptr = array_[curr_read_index];
  auto next_read_index = curr_read_index + 1;
  if(next_read_index == size_) {
    next_read_index = 0;
  }
  read_index_.store(next_read_index, std::memory_order_release);
  return ptr;
}

void* ShmemRecvQueue::Front() {
  while(Empty()) {
    helper::Sleep();
    if(done_)
      return NULL;
  }
  auto curr_read_index = read_index_.load(std::memory_order_acquire);
  void *ptr = array_[curr_read_index];
  return ptr;
}

void ShmemRecvQueue::Stop() {
  done_ = true;
}

bool ShmemRecvQueue::Probe() {
  return !Empty();
}

int ShmemRecvQueue::AvailableCount() {
  auto const curr_read_index = read_index_.load(std::memory_order_acquire);
  auto const curr_write_index = write_index_.load(std::memory_order_acquire);
  if (curr_read_index == curr_write_index) {
    return size_;
  }
  if (curr_write_index > curr_read_index) {
    return size_ - curr_write_index + curr_read_index - 1;
  }
  return curr_read_index - curr_write_index - 1;
}

bool ShmemRecvQueue::Empty() {
  auto const curr_read_index = read_index_.load(std::memory_order_acquire);
  auto const curr_write_index = write_index_.load(std::memory_order_acquire);
  return curr_read_index == curr_write_index;
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
}

ShmemRecvQueue::~ShmemRecvQueue() {
  Free();
}

ShmemSendPort::ShmemSendPort(const std::string &name,
                SharedMemoryPtr shm,
                const size_t &size,
                const size_t &nbytes)
  : AbstractSendPort(name, size, nbytes), shm_(shm), done_(false)
{}

void ShmemSendPort::Start() {
  shm_->Start();
}

void ShmemSendPort::Send(MetaDataPtr metadata) {
  shm_->Store([this, &metadata](void* data){
    char* cptr = (char*)data;
    std::memcpy(cptr, metadata.get(), sizeof(MetaData));
    cptr += sizeof(MetaData);
    std::memcpy(cptr, metadata->mdata, this->nbytes_ - sizeof(MetaData));
  });
}

bool ShmemSendPort::Probe() {
  return false;
}

void ShmemSendPort::Join() {
  done_ = true;
}

ShmemRecvPort::ShmemRecvPort(const std::string &name,
                SharedMemoryPtr shm,
                const size_t &size,
                const size_t &nbytes)
  : AbstractRecvPort(name, size, nbytes), shm_(shm), done_(false)
{
  queue_ = std::make_shared<ShmemRecvQueue>(name_, size_, nbytes_);
}

void ShmemRecvPort::Start() {
  recv_queue_thread_ = std::make_shared<std::thread>(&message_infrastructure::ShmemRecvPort::QueueRecv, this);
}

void ShmemRecvPort::QueueRecv() {
  while(!done_.load()) {
    bool ret = false;
    if (this->queue_->AvailableCount() > 0) {
      ret = shm_->Load([this](void* data){
        this->queue_->Push(data);
      });
    }
    if (!ret) {
      // sleep
      helper::Sleep();
    }
  }
}

bool ShmemRecvPort::Probe() {
  return queue_->Probe();
}

MetaDataPtr ShmemRecvPort::Recv() {
  char *cptr = (char *)queue_->Pop(true);
  MetaDataPtr metadata_res = std::make_shared<MetaData>();
  std::memcpy(metadata_res.get(), cptr, sizeof(MetaData));
  metadata_res->mdata = (void*)(cptr + sizeof(MetaData));
  return metadata_res;
}

void ShmemRecvPort::Join() {
  if (!done_) {
    done_ = true;
    recv_queue_thread_->join();
    queue_->Stop();
  }
}

MetaDataPtr ShmemRecvPort::Peek() {
  char *cptr = (char *)queue_->Front();
  MetaDataPtr metadata_res = std::make_shared<MetaData>();
  std::memcpy(metadata_res.get(), cptr, sizeof(MetaData));
  metadata_res->mdata = (void*)(cptr + sizeof(MetaData));
  return metadata_res;
}

}  // namespace message_infrastructure
