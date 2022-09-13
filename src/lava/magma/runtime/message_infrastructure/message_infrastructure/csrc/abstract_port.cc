// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: BSD-3-Clause
// See: https://spdx.org/licenses/

#include <pybind11/pybind11.h>
#include <string>
#include "abstract_port.h"

namespace message_infrastructure {
namespace py = pybind11;

std::string AbstractSendPort::Name() {
  return this->name_;
}
size_t AbstractSendPort::Size() {
  return this->size_;
}
void AbstractSendPort::Start() {
  printf("AbstractSendPort Start.\n");
}
bool AbstractSendPort::Probe() {
  printf("AbstractSendPort Probe.\n");
  return true;
}
void AbstractSendPort::Join() {
  printf("AbstractPort Join.\n");
}
void AbstractSendPort::Send(MetaDataPtr data) {
  printf("AbstractPort Send.\n");
}
std::string AbstractRecvPort::Name() {
  return this->name_;
}
size_t AbstractRecvPort::Size() {
  return this->size_;
}
void AbstractRecvPort::Start() {
  printf("AbstractRecvPort Start.\n");
}
bool AbstractRecvPort::Probe() {
  printf("AbstractRecvPort Probe.\n");
  return true;
}
void AbstractRecvPort::Join() {
  printf("AbstractRecvPort Join.\n");
}
MetaDataPtr AbstractRecvPort::Recv() {
  printf("AbstractPort Recv.\n");
  return NULL;
}
MetaDataPtr AbstractRecvPort::Peek() {
  printf("AbstractPort Peek.\n");
  return NULL;
}
}  // namespace message_infrastructure
