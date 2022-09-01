// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: BSD-3-Clause
// See: https://spdx.org/licenses/

#include <pthread.h>
#include <pybind11/numpy.h>
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

template<class T>
ShmemSendPort<T>::ShmemSendPort(const std::string &name,
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
  
  T *ptr = (T *)mmap(NULL, nbytes_ * size, PROT_WRITE, MAP_SHARED, shmid, 0);

  for(int i = 0; i < size_; i++) {
    array_.push_back(ptr + nbytes_ * i);
  }
}

template<class T>
int ShmemSendPort<T>::Start() {
  std::string semaphore_name = name_ + "_semaphore";
  semaphore_ = sem_open(semaphore_name.c_str(), CREAT_FLAG, ACC_MODE, 0);
  std::thread send_thread(this->AckCallback);
  send_thread.join();
  return 0;
}

template<class T>
int ShmemSendPort<T>::Probe() {
  int result = sem_trywait(semaphore_);
  if(!result)
    sem_post(semaphore_);
  return result;
}

template<class T>
int ShmemSendPort<T>::Send(T* data) {
  std::string req_name = name_ + "_req";
  req_ = sem_open(req_name.c_str(), 0);
  sem_wait(semaphore_);
  memcpy(array_[idx_], data, nbytes_);
  idx_ = (idx_ + 1) % size_;
  sem_post(req_);
  return 0;
}

template<class T>
int ShmemSendPort<T>::Join() {
  this->done_ = true;
  return 0;
}

template<class T>
int ShmemSendPort<T>::AckCallback() {
  std::string ack_name = name_ + "_ack";
  ack_ = sem_open(ack_name.c_str(), 0);

  while(!this->done_) {
    sem_wait(ack_);
    sem_post(semaphore_);
    // bool not_full = this->Probe();
    // if(this->observer && !not_full)
    // Todo(hexu1) : CspSelector.Observer
    // printf("this->observer");
  }
  return 0;
}

template<class T>
std::string ShmemSendPort<T>::Name() {
  return this->name_;
}

// template<class T>
// pybind11::dtype ShmemSendPort<T>::Dtype() {
//   return this->dtype_;
// }

// template<class T>
// ssize_t* ShmemSendPort<T>::Shape() {
//   return this->shape_;
// }

template<class T>
size_t ShmemSendPort<T>::Size() {
  return this->size_;
}

template<class T>
ShmemRecvPort<T>::ShmemRecvPort(const std::string &name,
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
  
  T *ptr = (T *)mmap(NULL, nbytes_ * size, PROT_WRITE, MAP_SHARED, shmid, 0);

  for(int i = 0; i < size_; i++) {
    array_.push_back(ptr + nbytes_ * i);
  }
}

template<class T>
int ShmemRecvPort<T>::Start() {
  std::thread recv_thread(this->ReqCallback);
  recv_thread.join();
  return 0;
}

template<class T>
bool ShmemRecvPort<T>::Probe() {
  // return queue_.size() > 0;
  return true;
}

template<class T>
T* ShmemRecvPort<T>::Recv() {
  // queue_.get();
  T result[nbytes_ / sizeof(T)];
  memcpy(result, array_[idx_], nbytes_);
  idx_ = (idx_ + 1) % size_;
  sem_post(ack_);
  return result;
}

template<class T>
int ShmemRecvPort<T>::Join() {
  this->done_ = true;
  return 0;
}

template<class T>
T* ShmemRecvPort<T>::Peek() {
  // queue_.get(true);
  T result[nbytes_ / sizeof(T)];
  memcpy(result, array_[idx_], nbytes_);
  return result;
}

template<class T>
int ShmemRecvPort<T>::ReqCallback() {
  while(!this->done_) {
    sem_wait(req_);
    // bool not_empty = this->Probe();
    // queue_.put_nowait(0);
    // if (this->observer && !not_empty)
      // Todo(hexu1) CspSelector.Observer
      // printf("this->observer");
  }
  return 0;
}

} // namespace message_infrastructure
