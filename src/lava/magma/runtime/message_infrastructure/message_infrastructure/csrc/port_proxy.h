// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: BSD-3-Clause
// See: https://spdx.org/licenses/

#ifndef PORT_PROXY_H_
#define PORT_PROXY_H_

#include <memory>
#include <string>
#include "abstract_port.h"
#include "shmem_port.h"

namespace message_infrastructure {

class SendPortProxy {
 public:
  SendPortProxy(ChannelType channel_type, AbstractSendPortPtr send_port) :
                  channel_type_(channel_type),
                  send_port_(send_port) {
    printf("Create SendPortProxy\n");
  }
  ChannelType GetChannelType() {
    return channel_type_;
  }
  AbstractSendPortPtr GetSendPort() {
    return send_port_;
  }
  int Start() {
    return send_port_->Start();
  }
  int Probe() {
    return send_port_->Probe();
  }
  int Send() {
    return send_port_->Send();
  }
  int Join() {
    return send_port_->Join();
  }
  std::string Name() {
    return send_port_->Name();
  }
  pybind11::dtype Dtype() {
    return send_port_->Dtype();
  }
  ssize_t* Shape() {
    return send_port_->Shape();
  }
  size_t Size() {
    return send_port_->Size();
  }

 private:
  ChannelType channel_type_;
  AbstractSendPortPtr send_port_;
};

class RecvPortProxy {
 public:
  RecvPortProxy(ChannelType channel_type, AbstractRecvPortPtr recv_port) :
                  channel_type_(channel_type),
                  recv_port_(recv_port) {
    printf("Create RecvPortProxy\n");
  }

  ChannelType GetChannelType() {
    return channel_type_;
  }
  AbstractRecvPortPtr GetRecvPort() {
    return recv_port_;
  }
  int Start() {
    return recv_port_->Start();
  }
  int Probe() {
    return recv_port_->Probe();
  }
  int Recv() {
    return recv_port_->Recv();
  }
  int Join() {
    return recv_port_->Join();
  }
  int Peek() {
    return recv_port_->Peek();
  }
  std::string Name() {
    return recv_port_->Name();
  }
  pybind11::dtype Dtype() {
    return recv_port_->Dtype();
  }
  ssize_t* Shape() {
    return recv_port_->Shape();
  }
  size_t Size() {
    return recv_port_->Size();
  }

 private:
  ChannelType channel_type_;
  AbstractRecvPortPtr recv_port_;
};

using SendPortProxyPtr = std::shared_ptr<SendPortProxy>;
using RecvPortProxyPtr = std::shared_ptr<RecvPortProxy>;

}  // namespace message_infrastructure

#endif  // PORT_PROXY_H_
