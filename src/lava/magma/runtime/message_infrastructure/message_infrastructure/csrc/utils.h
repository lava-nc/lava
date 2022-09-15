// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: BSD-3-Clause
// See: https://spdx.org/licenses/

#ifndef UTILS_H_
#define UTILS_H_

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

#include <vector>

namespace message_infrastructure {

enum ProcessType {
  ErrorProcess = 0,
  ParentProcess = 1,
  ChildProcess = 2
};

enum ChannelType {
  SHMEMCHANNEL = 0,
  RPCCHANNEL = 1,
  DDSCHANNEL = 2
};

struct Proto {
  const ssize_t *shape_;
  pybind11::dtype dtype_;
  size_t nbytes_;
};

struct MetaData {
  int64_t nd;
  int64_t type;
  int64_t elsize;
  int64_t total_size;
  std::vector<int64_t> dims;
  std::vector<int64_t> strides;
  void* mdata;
};

}  // namespace message_infrastructure

#endif  // UTILS_H_
