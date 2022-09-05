// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: BSD-3-Clause
// See: https://spdx.org/licenses/

#include <pthread.h>
// #include <pybind11/numpy.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <sys/shm.h>
#include <fcntl.h>
#include <semaphore.h>
#include <unistd.h>
#include <thread>
#include <mutex>
#include <string>

#include "shmem_port.h"
#include "shm.h"
#include "utils.h"

namespace message_infrastructure {

ShmemSendPort::ShmemSendPort(const std::string &name,
                const SharedMemory &shm,
                const size_t &size,
                const size_t &nbytes) {
  name_ = name;
  shm_ = shm;
  nbytes_ = nbytes;
  size_ = size;

  int shmid = shm_.GetShmid();
  
  struct stat stat_shmem;

  int err_1 = fstat(shmid, &stat_shmem);
  if (err_1 == -1) {
    printf("ShmemSendPort retrive shared memory segment info fail..\n");
    exit(-1);
  }
  
  void *ptr = mmap(NULL, nbytes_ * size, PROT_WRITE, MAP_SHARED, shmid, 0);

  for(int i = 0; i < size_; i++) {
    array_.push_back(ptr + nbytes_ * i);
  }
}

int ShmemSendPort::Start() {
  printf("SendPort start.\n");
  std::string semaphore_name = name_ + "_semaphore";
  semaphore_ = sem_open(semaphore_name.c_str(), CREAT_FLAG, ACC_MODE, 0);
  std::thread send_thread(&message_infrastructure::ShmemSendPort::AckCallback, this);
  send_thread.join();
  return 0;
}

int ShmemSendPort::Probe() {
  int result = sem_trywait(semaphore_);
  if(!result)
    sem_post(semaphore_);
  return result;
}

int ShmemSendPort::Send(void* data) {
  printf("SendPort send.\n");
  std::string req_name = name_ + "_req";
  req_ = sem_open(req_name.c_str(), 0);
  sem_wait(semaphore_);
  memcpy(array_[idx_], data, nbytes_);
  idx_ = (idx_ + 1) % size_;
  sem_post(req_);
  return 0;
}

int ShmemSendPort::Join() {
  done_ = true;
  return 0;
}

int ShmemSendPort::AckCallback() {
  std::string ack_name = name_ + "_ack";
  ack_ = sem_open(ack_name.c_str(), 0);

  while(!done_) {
    sem_wait(ack_);
    sem_post(semaphore_);
    // bool not_full = Probe();
    // if(>observer && !not_full)
    // Todo(hexu1) : CspSelector.Observer
    // printf("observer");
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
                const SharedMemory &shm,
                const size_t &size,
                const size_t &nbytes) {
  name_ = name;
  shm_ = shm;
  nbytes_ = nbytes;
  size_ = size;

  int shmid = shm_.GetShmid();

  struct stat stat_shmem;

  int err_1 = fstat(shmid, &stat_shmem);
  if (err_1 == -1) {
    printf("ShmemSendPort retrive shared memory segment info fail..\n");
    exit(-1);
  }
  
  void *ptr = mmap(NULL, nbytes_ * size, PROT_WRITE, MAP_SHARED, shmid, 0);

  for(int i = 0; i < size_; i++) {
    array_.push_back(ptr + nbytes_ * i);
  }
}

int ShmemRecvPort::Start() {
  printf("RecvPort start.\n");
  std::thread recv_thread(&message_infrastructure::ShmemRecvPort::ReqCallback, this);
  recv_thread.join();
  return 0;
}

bool ShmemRecvPort::Probe() {
  // return queue_.size() > 0;
  return true;
}

void* ShmemRecvPort::Recv() {
  // queue_.get();
  // todo
  printf("RecvPort recv.\n");
  int result[nbytes_ / sizeof(int)];
  memcpy(result, array_[idx_], nbytes_);
  idx_ = (idx_ + 1) % size_;
  sem_post(ack_);
  return result;
}

int ShmemRecvPort::Join() {
  done_ = true;
  return 0;
}

void* ShmemRecvPort::Peek() {
  // queue_.get(true);
  int result[nbytes_ / sizeof(int)];
  memcpy(result, array_[idx_], nbytes_);
  return result;
}

int ShmemRecvPort::ReqCallback() {
   while(!done_) {
    sem_wait(req_);
    // bool not_empty = Probe();
    // queue_.put_nowait(0);
    // if (observer && !not_empty)
      // Todo(hexu1) CspSelector.Observer
      // printf("observer");
  }
  return 0;
}

} // namespace message_infrastructure
