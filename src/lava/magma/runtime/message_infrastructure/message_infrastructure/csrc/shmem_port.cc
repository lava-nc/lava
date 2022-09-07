// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: BSD-3-Clause
// See: https://spdx.org/licenses/

#include <pthread.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <sys/shm.h>
#include <fcntl.h>
#include <semaphore.h>
#include <memory.h>
#include <unistd.h>
#include <thread>
#include <mutex>
#include <string>
#include <condition_variable>

#include "shmem_port.h"
#include "shm.h"
#include "utils.h"

namespace message_infrastructure {
void ShmemRecvQueue::Init(const size_t &capacity, const size_t &nbytes) {
  capacity_ = capacity;
  nbytes_ = nbytes;
}

// void ShmemRecvQueue::Push(void* src) {
//   void *ptr = malloc(nbytes_);
//   memcpy(ptr, src, nbytes_);
//   lock_.lock();
//   if (queue_.empty()) {
//     cond_.notify_one();
//   }
//   while(queue_.size() >= capacity_) {
//     usleep(1);
//   }
//   queue_.push(ptr);
//   lock_.unlock();
// }

// void ShmemRecvQueue::Pop() {
//   std::unique_lock<std::mutex> l(lock_);
//   while (queue_.empty()) {
//     cond_.wait(l);
//   };
//   queue_.pop();
// }

// void* ShmemRecvQueue::Front() {
//   void *ptr = NULL;
//   std::unique_lock<std::mutex> l(lock_);
//   while (queue_.empty()) {
//     cond_.wait(l);
//   };
//   ptr = queue_.front();
//   return ptr;
// }

// void* ShmemRecvQueue::FrontPop() {
//   void *ptr = NULL;
//   std::unique_lock<std::mutex> l(lock_);
//   while (queue_.empty()) {
//     cond_.wait(l);
//   };
//   ptr = queue_.front();
//   queue_.pop();
//   return ptr;
// }

// bool ShmemRecvQueue::Probe() {
//   return (queue_.size() == capacity_);
// }


void ShmemRecvQueue::Push(void* src) {
  void *ptr = malloc(nbytes_);
  memcpy(ptr, src, nbytes_);
  std::unique_lock<std::mutex> l(lock_);
  cond_.wait(l, [this]() { return this->queue_.size() < this->capacity_; });
  queue_.push(ptr);
  cond_.notify_all();
}

void ShmemRecvQueue::Pop() {
  std::unique_lock<std::mutex> l(lock_);
  cond_.wait(l, [this]() { return !this->queue_.empty(); });
  queue_.pop();
  cond_.notify_all();
}

void* ShmemRecvQueue::Front() {
  void *ptr = NULL;
  std::unique_lock<std::mutex> l(lock_);
  cond_.wait(l, [this]() { return !this->queue_.empty(); });
  ptr = queue_.front();
  cond_.notify_all();
  return ptr;
}

void* ShmemRecvQueue::FrontPop() {
  void *ptr = NULL;
  std::unique_lock<std::mutex> l(lock_);
  cond_.wait(l, [this]() { return !this->queue_.empty(); });
  ptr = queue_.front();
  queue_.pop();
  cond_.notify_all();
  return ptr;
}

bool ShmemRecvQueue::Probe() {
  return (queue_.size() == capacity_);  // Will this be unsafe?
}

ShmemSendPort::ShmemSendPort(const std::string &name,
                SharedMemory shm,
                const size_t &size,
                const size_t &nbytes) {
  name_ = name;
  shm_ = shm;
  nbytes_ = nbytes;
  size_ = size;

  done_ = false;

  array_ = shmat(shm.GetShmid(), NULL, 0);
}

int ShmemSendPort::Start() {
  sem_post(&shm_.GetAckSemaphore());
  // std::thread ack_callback_thread(&message_infrastructure::ShmemSendPort::AckCallback, this);
  // ack_callback_thread.detach();
  return 0;
}

int ShmemSendPort::Send(void* data) {
  sem_wait(&shm_.GetAckSemaphore());
  memcpy(array_, data, nbytes_);
  sem_post(&shm_.GetReqSemaphore());
  return 0;
}

int ShmemSendPort::Join() {
  done_ = true;
  return 0;
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
                SharedMemory shm,
                const size_t &size,
                const size_t &nbytes) {
  name_ = name;
  shm_ = shm;
  nbytes_ = nbytes;
  size_ = size;
  done_ = false;
  array_ = shmat(shm.GetShmid(), NULL, 0);
  queue_.Init(size, nbytes);
}

int ShmemRecvPort::Start() {
  // std::thread req_callback_thread(&message_infrastructure::ShmemRecvPort::ReqCallback, this);
  // req_callback_thread.detach();
  std::thread recv_queue_thread(&message_infrastructure::ShmemRecvPort::QueueRecv, this);
  recv_queue_thread.detach();
  return 0;
}

void ShmemRecvPort::QueueRecv() {
  while(!done_) {
    sem_wait(&shm_.GetReqSemaphore());
    queue_.Push(array_);
    sem_post(&shm_.GetAckSemaphore());
  }
}

bool ShmemRecvPort::Probe() {
  return queue_.Probe();
}

void* ShmemRecvPort::Recv() {
  return queue_.FrontPop();
}

int ShmemRecvPort::Join() {
  done_ = true;
  return 0;
}

void* ShmemRecvPort::Peek() {
  return queue_.Front();
}

int ShmemRecvPort::ReqCallback() {
  while(!done_) {
  // Todo(hexu1) : CspSelector.Observer
  }
  return 0;
}

} // namespace message_infrastructure
