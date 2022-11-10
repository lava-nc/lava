// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: BSD-3-Clause
// See: https://spdx.org/licenses/

#ifndef PORT_PROXY_H_
#define PORT_PROXY_H_

#include <message_infrastructure/csrc/core/abstract_port.h>
#include <message_infrastructure/csrc/core/utils.h>
#include <pybind11/pybind11.h>

#include <memory>
#include <string>
#include <vector>

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

class GrpcSendPortProxy : public PortProxy {
 public:
  GrpcSendPortProxy() {}
  GrpcSendPortProxy(ChannelType channel_type,
                    GrpcAbstractSendPortPtr send_port) :
                    channel_type_(channel_type), send_port_(send_port) {}
  ChannelType GetChannelType();
  void Start();
  bool Probe();
  void Send(py::object* object);
  void Join();
  std::string Name();
  size_t Size();

 private:
  GrpcMetaDataPtr GrpcMDataFromObject_(py::object* object);
  ChannelType channel_type_;
  GrpcAbstractSendPortPtr send_port_;
};

using GrpcSendPortProxyPtr = std::shared_ptr<GrpcSendPortProxy>;
using GrpcSendPortProxyList = std::vector<GrpcSendPortProxyPtr>;

}  // namespace message_infrastructure

#endif  // PORT_PROXY_H_
