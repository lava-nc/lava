// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: BSD-3-Clause
// See: https://spdx.org/licenses/

#ifndef PORT_PROXY_H_
#define PORT_PROXY_H_

#include <memory>
#include <string>
#include "abstract_port.h"
#include "abstract_channel.h"

namespace message_infrastructure {

class SendPortProxy {
 public:
  SendPortProxy(AbstractChannelPtr channel,
                ChannelType channel_type) :
    send_port_(channel->send_port_),
    channel_type_(channel_type) {}
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
  RecvPortProxy(AbstractChannelPtr channel,
                ChannelType channel_type) :
    recv_port_(channel->recv_port_),
    channel_type_(channel_type) {}
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

}  // namespace message_infrastructure

#endif  // PORT_PROXY_H_
