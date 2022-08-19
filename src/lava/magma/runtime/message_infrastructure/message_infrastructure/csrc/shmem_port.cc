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
  name_ = name;
  shm_ = shm;
  dtype_ = proto->dtype_;
  shape_ = proto->shape_;
  size_ = size;
  req_ = req;
  ack_ = ack;
}
template<class T>
int ShmemSendPort<T>::Start() {
  // Todo
  return 0;
}
template<class T>
int ShmemSendPort<T>::Probe() {
  // Todo
  return 0;
}
template<class T>
int ShmemSendPort<T>::Send() {
  // Todo
  return 0;
}
template<class T>
int ShmemSendPort<T>::Join() {
  // Todo
  return 0;
}
template<class T>
int ShmemSendPort<T>::AckCallback() {
  // Todo
  return 0;
}

template<class T>
ShmemRecvPort<T>::ShmemRecvPort(const std::string &name,
              SharedMemoryPtr shm,
              Proto *proto,
              const size_t &size,
              sem_t *req,
              sem_t *ack) {
  name_ = name;
  shm_ = shm;
  dtype_ = proto->dtype_;
  shape_ = proto->shape_;
  size_ = size;
  req_ = req;
  ack_ = ack;
}
template<class T>
int ShmemRecvPort<T>::Start() {
  // Todo
  return 0;
}
template<class T>
int ShmemRecvPort<T>::Probe() {
  // Todo
  return 0;
}
template<class T>
int ShmemRecvPort<T>::Recv() {
  // Todo
  return 0;
}
template<class T>
int ShmemRecvPort<T>::Join() {
  // Todo
  return 0;
}
template<class T>
int ShmemRecvPort<T>::Peek() {
  // Todo
  return 0;
}
template<class T>
int ShmemRecvPort<T>::ReqCallback() {
  // Todo
  return 0;
}

} // namespace message_infrastructure
