// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: BSD-3-Clause
// See: https://spdx.org/licenses/

#ifndef ABSTRACT_PORT_IMPLEMENTATION_H_
#define ABSTRACT_PORT_IMPLEMENTATION_H_

#include "port_proxy.h"

namespace message_infrastructure {

class AbstractPortImplementation {
 public:
  explicit AbstractPortImplementation(const SendPortProxyList &send_ports,
                                      const RecvPortProxyList &recv_ports);
  explicit AbstractPortImplementation(const RecvPortProxyList &recv_ports);
  explicit AbstractPortImplementation(const SendPortProxyList &send_ports);
  int Start();
  int Join();

 protected:
  SendPortProxyList send_ports_;
  RecvPortProxyList recv_ports_;
};

}  // namespace message_infrastructure

#endif  // ABSTRACT_PORT_IMPLEMENTATION_H_
