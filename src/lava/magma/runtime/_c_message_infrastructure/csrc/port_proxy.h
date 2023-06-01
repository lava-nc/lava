// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: BSD-3-Clause
// See: https://spdx.org/licenses/

#ifndef PORT_PROXY_H_
#define PORT_PROXY_H_

#include <core/abstract_port.h>
#include <core/utils.h>
#include <pybind11/pybind11.h>

#include <memory>
#include <string>
#include <vector>
#include <tuple>
#include <utility>
#include <mutex>  // NOLINT
#include <condition_variable>  // NOLINT

namespace message_infrastructure {

namespace py = pybind11;

class PortProxy {
 public:
  PortProxy() {}
  PortProxy(py::tuple shape, py::object d_type) :
            shape_(shape), d_type_(d_type) {}
  py::object DType();
  py::tuple Shape();
 private:
  py::object d_type_;
  py::tuple shape_;
};

class SendPortProxy : public PortProxy {
 public:
  SendPortProxy() {}
  SendPortProxy(ChannelType channel_type,
                AbstractSendPortPtr send_port,
                py::tuple shape = py::make_tuple(),
                py::object type = py::none()) :
                PortProxy(shape, type),
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
  DataPtr DataFromObject_(py::object* object);
  ChannelType channel_type_;
  AbstractSendPortPtr send_port_;
};


class RecvPortProxy : public PortProxy {
 public:
  RecvPortProxy() {}
  RecvPortProxy(ChannelType channel_type,
                AbstractRecvPortPtr recv_port,
                py::tuple shape = py::make_tuple(),
                py::object type = py::none()) :
                PortProxy(shape, type),
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
  void SetObserver(std::function<void()> obs);

 private:
  py::object MDataToObject_(MetaDataPtr metadata);
  ChannelType channel_type_;
  AbstractRecvPortPtr recv_port_;
};

// Users should be allowed to copy port objects.
// Use std::shared_ptr.
using SendPortProxyPtr = std::shared_ptr<SendPortProxy>;
using RecvPortProxyPtr = std::shared_ptr<RecvPortProxy>;
using SendPortProxyList = std::vector<SendPortProxyPtr>;
using RecvPortProxyList = std::vector<RecvPortProxyPtr>;


class Selector {
 public:
  pybind11::object Select(std::vector<std::tuple<RecvPortProxyPtr,
                          py::function>> *args) {
    while (true) {
      for (auto it = args->begin(); it != args->end(); ++it) {
        if (std::get<0>(*it)->Probe()) {
          return std::get<1>(*it)();
        }
      }
    }
  }
};

}  // namespace message_infrastructure

#endif  // PORT_PROXY_H_
