// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: BSD-3-Clause
// See: https://spdx.org/licenses/

#ifndef CORE_ABSTRACT_PORT_IMPLEMENTATION_H_
#define CORE_ABSTRACT_PORT_IMPLEMENTATION_H_

#include <message_infrastructure/csrc/core/abstract_port.h>

namespace message_infrastructure {

class AbstractPortImplementation {
 public:
  explicit AbstractPortImplementation(const SendPortList &send_ports,
                                      const RecvPortList &recv_ports);
  explicit AbstractPortImplementation(const RecvPortList &recv_ports);
  explicit AbstractPortImplementation(const SendPortList &send_ports);
  virtual ~AbstractPortImplementation() = default;
  int Start();
  int Join();

 protected:
  SendPortList send_ports_;
  RecvPortList recv_ports_;
};

}  // namespace message_infrastructure

#endif  // CORE_ABSTRACT_PORT_IMPLEMENTATION_H_
