// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: BSD-3-Clause
// See: https://spdx.org/licenses/

#include "abstract_port.h"
#include <string>

namespace message_infrastructure {
std::string AbstractPort::Name() {
  return this->name_;
}
size_t AbstractPort::Size() {
  return this->size_;
}
int AbstractPort::Start() {
  printf("AbstractPort Start.\n");
  return 0;
}
int AbstractPort::Probe() {
  printf("AbstractPort Probe.\n");
  return 0;
}
int AbstractPort::Join() {
  printf("AbstractPort Join.\n");
  return 0;
}
int AbstractSendPort::Send(void* data) {
  printf("AbstractPort Send.\n");
  return 0;
}
void* AbstractRecvPort::Recv() {
  printf("AbstractPort Recv.\n");
  return NULL;
}
int AbstractRecvPort::Peek() {
  printf("AbstractPort Peek.\n");
  return 0;
}
}  // namespace message_infrastructure
