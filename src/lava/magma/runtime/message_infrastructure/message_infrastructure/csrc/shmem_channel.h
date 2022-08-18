// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: BSD-3-Clause
// See: https://spdx.org/licenses/

#ifndef SHMEM_CHANNEL_H_
#define SHMEM_CHANNEL_H_

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

#include <memory>
#include <string>

#include "abstract_channel.h"
#include "shm.h"
#include "port_proxy.h"

namespace message_infrastructure {

class ShmemChannel : public AbstractChannel {
 public:
  ShmemChannel(SharedMemoryPtr shm,
               const std::string &src_name,
               const std::string &dst_name,
               const ssize_t* shape,
               const pybind11::dtype &dtype,
               const size_t &size,
               const size_t &nbytes) {
    printf("Create ShmemChannel\n");
    Proto proto;
    proto.shape_ = shape;
    proto.dtype_ = dtype;
    proto.nbytes_ = nbytes;

    shm = NULL;

    AbstractSendPortPtr send_port;
    AbstractRecvPortPtr recv_port;
    send_port_proxy_ = std::make_shared<SendPortProxy>(
        ChannelType::SHMEMCHANNEL,
        send_port);
    recv_port_proxy_ = std::make_shared<RecvPortProxy>(
        ChannelType::SHMEMCHANNEL,
        recv_port);
  }
  SendPortProxyPtr GetSendPort() {
    printf("Get send_port.\n");
    return this->send_port_proxy_;
  }
  RecvPortProxyPtr GetRecvPort() {
    printf("Get recv_port.\n");
    return this->recv_port_proxy_;
  }

 private:
  SharedMemoryPtr shm_ = NULL;
  sem_t *req_ = NULL;
  sem_t *ack_ = NULL;
  SendPortProxyPtr send_port_proxy_;
  RecvPortProxyPtr recv_port_proxy_;
};

using ShmemChannelPtr = ShmemChannel *;

template <class T>
std::shared_ptr<ShmemChannel> GetShmemChannel(SharedMemoryPtr shm,
                              const pybind11::array_t<T> &data,
                              const size_t &size,
                              const size_t &nbytes,
                              const std::string &name = "test_channel") {
  return (std::make_shared<ShmemChannel>(shm,
                                         name,
                                         name,
                                         data.shape(),
                                         data.dtype(),
                                         size,
                                         nbytes));
}

}  // namespace message_infrastructure

#endif  // SHMEM_CHANNEL_H_

