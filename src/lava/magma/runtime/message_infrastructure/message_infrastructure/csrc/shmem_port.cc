// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: BSD-3-Clause
// See: https://spdx.org/licenses/

#include "shmem_port.h"
#include "shm.h"
#include "utils.h"

namespace message_infrastructure {

template<class T>
ShmemSendPort<T>::ShmemSendPort(const std::string &name,
              SharedMemoryPtr shm,
              Proto *proto,
              const size_t &size,
              sem_t *req,
              sem_t *ack) {
                // Todo
              }
template<class T>
std::string ShmemSendPort<T>::Name() {
  return this->name_;
}
template<class T>
pybind11::dtype ShmemSendPort<T>::Dtype() {
  return this->dtype_;
}
template<class T>
ssize_t* ShmemSendPort<T>::Shape() {
  return this->shape_;
}
template<class T>
size_t ShmemSendPort<T>::Size() {
  return this->size_;
}
template<class T>
int ShmemSendPort<T>::Start() {
  // Todo
}
template<class T>
int ShmemSendPort<T>::Probe() {
  // Todo
}
template<class T>
int ShmemSendPort<T>::Send() {
  // Todo
}
template<class T>
int ShmemSendPort<T>::Join() {
  // Todo
}
template<class T>
int ShmemSendPort<T>::AckCallback() {
  // Todo
}

template<class T>
ShmemRecvPort<T>::ShmemRecvPort(const std::string &name,
              SharedMemoryPtr shm,
              Proto *proto,
              const size_t &size,
              sem_t *req,
              sem_t *ack) {
  // Todo
}
template<class T>
std::string ShmemRecvPort<T>::Name() {
  return this->name_;
}
template<class T>
pybind11::dtype ShmemRecvPort<T>::Dtype() {
  return this->dtype_;
}
template<class T>
ssize_t* ShmemRecvPort<T>::Shape() {
  return this->shape_;
}
template<class T>
size_t ShmemRecvPort<T>::Size() {
  return this->size_;
}
template<class T>
int ShmemRecvPort<T>::Start() {
  // Todo
}
template<class T>
int ShmemRecvPort<T>::Probe() {
  // Todo
}
template<class T>
int ShmemRecvPort<T>::Recv() {
  // Todo
}
template<class T>
int ShmemRecvPort<T>::Join() {
  // Todo
}
template<class T>
int ShmemRecvPort<T>::Peek() {
  // Todo
}
template<class T>
int ShmemRecvPort<T>::ReqCallback() {
  // Todo
}

} // namespace message_infrastructure
