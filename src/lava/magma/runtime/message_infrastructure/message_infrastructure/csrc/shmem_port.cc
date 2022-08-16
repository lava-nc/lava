// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: BSD-3-Clause
// See: https://spdx.org/licenses/

#include "shmem_port.h"

namespace message_infrastructure {

ShmemSendPort::ShmemSendPort(const std::string &name,
              SharedMemory *shm,
              Proto *proto,
              const size_t &size,
              sem_t *req,
              sem_t *ack);
int ShmemSendPort::Start();
int ShmemSendPort::Probe();
int ShmemSendPort::Send();
int ShmemSendPort::Join();
int ShmemSendPort::AckCallback();

ShmemRecvPort::ShmemRecvPort(const std::string &name,
              SharedMemory *shm,
              Proto *proto,
              const size_t &size,
              sem_t *req,
              sem_t *ack);
int ShmemRecvPort::Start();
int ShmemRecvPort::Probe();
int ShmemRecvPort::Send();
int ShmemRecvPort::Join();
int ShmemRecvPort::Peek();
int ShmemRecvPort::ReqCallback();

} // namespace message_infrastructure
