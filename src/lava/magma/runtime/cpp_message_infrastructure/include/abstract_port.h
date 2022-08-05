// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: BSD-3-Clause
// See: https://spdx.org/licenses/

#ifndef ABSTRACT_PORT_H_
#define ABSTRACT_PORT_H_

#include <string>
#include <vector>

#include "shm.h"
#include "utils.h"

class AbstractPort {
 public:
  virtual int Start() = 0;
  virtual int Join() = 0;

  std::string name_;
  DataType dtype_;
  std::vector<int> shape_;
  size_t size_;
};

class AbstractSendPort : public AbstractPort {
 public:
  virtual int Send() = 0;
};

class AbstractRecvPort : public AbstractPort {
 public:
  virtual int Recv() = 0;
};

#endif