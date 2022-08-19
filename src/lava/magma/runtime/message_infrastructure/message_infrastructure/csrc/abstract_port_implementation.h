// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: BSD-3-Clause
// See: https://spdx.org/licenses/

#ifndef ABSTRACT_PORT_IMPLEMENTATION_H_
#define ABSTRACT_PORT_IMPLEMENTATION_H_

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

#include <vector>

#include "port_proxy.h"
#include "utils.h"

namespace message_infrastructure {

class AbstractPortImplementation {
 public:
  explicit AbstractPortImplementation(const SendPortProxyList &send_ports,
                                      const RecvPortProxyList &recv_ports);
  explicit AbstractPortImplementation(const RecvPortProxyList &recv_ports);
  explicit AbstractPortImplementation(const SendPortProxyList &send_ports);
  int Start();
  int Join();
  // std::vector<int> GetShape();

 protected:
  SendPortProxyList send_ports_;
  RecvPortProxyList recv_ports_;
  Proto proto_;
};

}  // namespace message_infrastructure

#endif  // ABSTRACT_PORT_IMPLEMENTATION_H_
