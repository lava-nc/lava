// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: BSD-3-Clause
// See: https://spdx.org/licenses/

#ifndef PORT_PROXY_H_
#define PORT_PROXY_H_

#include <pybind11/pybind11.h>
#include <memory>
#include <string>
#include <vector>

#include "abstract_port.h"
#include "utils.h"

namespace message_infrastructure {

namespace py = pybind11;

class PortProxy {};

class SendPortProxy : public PortProxy {
 public:
  SendPortProxy() {}
  SendPortProxy(ChannelType channel_type, AbstractSendPortPtr send_port) :
                                          channel_type_(channel_type),
                                          send_port_(send_port) {}
  ChannelType GetChannelType();
  void Start();
  bool Probe();
  void Send(py::object* object);
  void Join();
  std::string Name();
  size_t Size();

 private:
  MetaDataPtr MDataFromObject_(py::object* object);
  ChannelType channel_type_;
  AbstractSendPortPtr send_port_;
};


class RecvPortProxy : public PortProxy {
 public:
  RecvPortProxy() {}
  RecvPortProxy(ChannelType channel_type, AbstractRecvPortPtr recv_port) :
                                            channel_type_(channel_type),
                                            recv_port_(recv_port) {}

  ChannelType GetChannelType();
  void Start();
  bool Probe();
  py::object Recv();
  void Join();
  py::object Peek();
  std::string Name();
  size_t Size();

 private:
  py::object MDataToObject_(MetaDataPtr metadata);
  ChannelType channel_type_;
  AbstractRecvPortPtr recv_port_;
};

using SendPortProxyPtr = std::shared_ptr<SendPortProxy>;
using RecvPortProxyPtr = std::shared_ptr<RecvPortProxy>;
using SendPortProxyList = std::vector<SendPortProxyPtr>;
using RecvPortProxyList = std::vector<RecvPortProxyPtr>;

}  // namespace message_infrastructure

#endif  // PORT_PROXY_H_
