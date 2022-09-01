// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: BSD-3-Clause
// See: https://spdx.org/licenses/

#include "abstract_port_implementation.h"

namespace message_infrastructure {

AbstractPortImplementation::AbstractPortImplementation(const SendPortProxyList &send_ports,
                                                       const RecvPortProxyList &recv_ports)
  : send_ports_(send_ports), recv_ports_(recv_ports)
  {}
AbstractPortImplementation::AbstractPortImplementation(const SendPortProxyList &send_ports)
  : send_ports_(send_ports)
  {}
AbstractPortImplementation::AbstractPortImplementation(const RecvPortProxyList &recv_ports)
  : recv_ports_(recv_ports)
  {}

int AbstractPortImplementation::Start() {
  for (auto port : this->send_ports_){
    port->Start();
  }
  for (auto port : this->recv_ports_){
    port->Start();
  }
}

int AbstractPortImplementation::Join() {
  for (auto port : this->send_ports_){
    port->Join();
  }
  for (auto port : this->recv_ports_){
    port->Join();
  }
}

/*
std::vector<int> AbstractPortImplementation::GetShape() {
  return this->shape_;
}
*/

}  // namespace message_infrastructure