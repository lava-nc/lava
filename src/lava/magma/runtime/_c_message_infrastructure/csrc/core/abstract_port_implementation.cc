// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: BSD-3-Clause
// See: https://spdx.org/licenses/

#include <core/abstract_port_implementation.h>

namespace message_infrastructure {

AbstractPortImplementation::AbstractPortImplementation(
  const SendPortList &send_ports,
  const RecvPortList &recv_ports)
  : send_ports_(send_ports), recv_ports_(recv_ports)
  {}
AbstractPortImplementation::AbstractPortImplementation(
  const SendPortList &send_ports)
  : send_ports_(send_ports)
  {}
AbstractPortImplementation::AbstractPortImplementation(
  const RecvPortList &recv_ports)
  : recv_ports_(recv_ports)
  {}

int AbstractPortImplementation::Start() {
  for (auto port : send_ports_) {
    port->Start();
  }
  for (auto port : recv_ports_) {
    port->Start();
  }
  return 0;
}

int AbstractPortImplementation::Join() {
  for (auto port : send_ports_) {
    port->Join();
  }
  for (auto port : recv_ports_) {
    port->Join();
  }
  return 0;
}
}  // namespace message_infrastructure
