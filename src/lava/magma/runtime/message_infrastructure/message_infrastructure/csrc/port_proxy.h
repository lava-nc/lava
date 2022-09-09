// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: BSD-3-Clause
// See: https://spdx.org/licenses/

#ifndef PORT_PROXY_H_
#define PORT_PROXY_H_

#include <pybind11/pybind11.h>

#include <memory>
#include <string>
#include <vector>

#include "abstract_port.h"
#include "message_infrastructure_logging.h"
#include "shmem_port.h"
#include "utils.h"

namespace message_infrastructure {

namespace py = pybind11;

class PortProxy {
};

class SendPortProxy : public PortProxy {
 public:
  SendPortProxy() {}
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
  int Send(py::object* object) {
    MetaDataPtr metadata = MDataFromObject_(object);
    return send_port_->Send(metadata);
  }
  int Join() {
    return send_port_->Join();
  }
  std::string Name() {
    return send_port_->Name();
  }
  size_t Size() {
    return send_port_->Size();
  }

 private:
  MetaDataPtr MDataFromObject_(py::object*);
  ChannelType channel_type_;
  AbstractSendPortPtr send_port_;
};


class RecvPortProxy : public PortProxy {
 public:
  RecvPortProxy() {}
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
  py::object Recv() {
    MetaDataPtr metadata = recv_port_->Recv();
    return MDataToObject_(metadata);
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
  size_t Size() {
    return recv_port_->Size();
  }

 private:
  py::object MDataToObject_(MetaDataPtr);
  ChannelType channel_type_;
  AbstractRecvPortPtr recv_port_;
};

using SendPortProxyPtr = std::shared_ptr<SendPortProxy>;
using RecvPortProxyPtr = std::shared_ptr<RecvPortProxy>;
using SendPortProxyList = std::vector<SendPortProxyPtr>;
using RecvPortProxyList = std::vector<RecvPortProxyPtr>;
}  // namespace message_infrastructure

#endif  // PORT_PROXY_H_
