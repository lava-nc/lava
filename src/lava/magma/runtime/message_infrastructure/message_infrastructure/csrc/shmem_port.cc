// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: BSD-3-Clause
// See: https://spdx.org/licenses/

#include <xmmintrin.h>
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
  done_ = false;
  overlap_ = false;
  read_index_.store(0, std::memory_order_release);
  write_index_.store(0, std::memory_order_release);
}

void ShmemRecvQueue::Push(void* src) {
  auto const curr_write_index = write_index_.load(std::memory_order_acquire);
  auto const curr_read_index = read_index_.load(std::memory_order_acquire);
  auto next_write_index = curr_write_index + 1;
  if (next_write_index == size_) {
      next_write_index = 0;
  }
  if(curr_write_index == curr_read_index && overlap_) {
    auto next_read_index = curr_read_index + 1;
    drop_array_.push_back(array_[curr_read_index]);
    next_read_index = next_write_index;
    read_index_.store(next_read_index, std::memory_order_release);
    LAVA_LOG_WARN(LOG_SMMP, "Drop data in ShmemChannel %s\n", name_.c_str());
  }
  if (next_write_index == curr_read_index) {
    overlap_ = true;
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
  overlap_ = false;
  return;
}

void* ShmemRecvQueue::Front() {
  if(Empty()) {
    LAVA_LOG_WARN(LOG_SMMP, "ShmemChannel is empty.\n");
    return NULL;
  }
  auto curr_read_index = read_index_.load(std::memory_order_acquire);
  void *ptr = array_[curr_read_index];
  return ptr;
}

void* ShmemRecvQueue::FrontPop() {
  while(Empty()) {
    _mm_pause();
    // LAVA_LOG_WARN(LOG_SMMP, "ShmemChannel is empty.\n");
    if(done_)
      return NULL;
  }
  auto curr_read_index = read_index_.load(std::memory_order_acquire);
  void *ptr = array_[curr_read_index];
  auto next_read_index = curr_read_index + 1;
  if(next_read_index == size_) {
    next_read_index = 0;
  }
  read_index_.store(next_read_index, std::memory_order_release);
  overlap_ = false;
  return ptr;
}

void ShmemRecvQueue::Stop() {
  done_ = true;
}

bool ShmemRecvQueue::Probe() {
  return !Empty();
}

bool ShmemRecvQueue::Empty() {
  auto const curr_read_index = read_index_.load(std::memory_order_acquire);
  auto const curr_write_index = write_index_.load(std::memory_order_acquire);
  return curr_read_index == curr_write_index && !overlap_;
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
  Free();
}

ShmemSendPort::ShmemSendPort(const std::string &name,
                SharedMemoryPtr shm,
                const size_t &size,
                const size_t &nbytes) : AbstractSendPort(name, size, nbytes), shm_(shm), done_(false)
{}

void ShmemSendPort::Start() {
  shm_->Start();
  // ack_callback_thread_ = std::make_shared<std::thread>(&message_infrastructure::ShmemSendPort::AckCallback, this);
}

void ShmemSendPort::Send(MetaDataPtr metadata) {
  shm_->Store([this, &metadata](void* data){
    char* cptr = (char*)data;
    memcpy(cptr, metadata.get(), sizeof(MetaData));
    cptr += sizeof(MetaData);
    memcpy(cptr, metadata->mdata, this->nbytes_);
  });
  printf("Send data finish.\n");
}

bool ShmemSendPort::Probe() {
  return false;
}

void ShmemSendPort::Join() {
  done_ = true;
}

int ShmemSendPort::AckCallback() {
  while(!done_) {
  // Todo(hexu1) : CspSelector.Observer
  }
  return 0;
}

ShmemRecvPort::ShmemRecvPort(const std::string &name,
                SharedMemoryPtr shm,
                const size_t &size,
                const size_t &nbytes): AbstractRecvPort(name, size, nbytes), shm_(shm), done_(false)
{
  queue_ = std::make_shared<ShmemRecvQueue>(name_, size_, nbytes_);
}

void ShmemRecvPort::Start() {
  // req_callback_thread_ = std::make_shared<std::thread>(&message_infrastructure::ShmemRecvPort::ReqCallback, this);
  recv_queue_thread_ = std::make_shared<std::thread>(&message_infrastructure::ShmemRecvPort::QueueRecv, this);
}

void ShmemRecvPort::QueueRecv() {
  while(!done_.load()) {
    auto ret = shm_->Load([this](void* data){
      this->queue_->Push(data);
    });
    if (!ret) {
      _mm_pause();
    }
  }
}

bool ShmemRecvPort::Probe() {
  return queue_->Probe();
}

MetaDataPtr ShmemRecvPort::Recv() {
  char *cptr = (char *)queue_->FrontPop();
  MetaDataPtr metadata_res = std::make_shared<MetaData>();
  memcpy(metadata_res.get(), cptr, sizeof(MetaData));
  metadata_res->mdata = (void*)(cptr + sizeof(MetaData));
  return metadata_res;
}

void ShmemRecvPort::Join() {
  done_ = true;
  recv_queue_thread_->join();
  queue_->Stop();
}

MetaDataPtr ShmemRecvPort::Peek() {
  // return queue_->Front();
  return NULL;
}

int ShmemRecvPort::ReqCallback() {
  while(!done_) {
  // Todo(hexu1) : CspSelector.Observer
  }
  return 0;
}

}  // namespace message_infrastructure
