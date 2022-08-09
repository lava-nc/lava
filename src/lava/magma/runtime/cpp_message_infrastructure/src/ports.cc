// Copyright (C) 2021-22 Intel Corporation
// SPDX-License-Identifier: BSD-3-Clause
// See: https://spdx.org/licenses/

#include <ports.h>

std::vector<int> AbstractCppPort::GetShape() {
  return this->shape_;
}

std::vector<int> AbstractCppIOPort::GetShape() {
  return this->shape_;
}

bool CppInPort::Probe() {
  return;
}

void CppInPort::Recv() {
}

void CppInPort::Peek() {
}


