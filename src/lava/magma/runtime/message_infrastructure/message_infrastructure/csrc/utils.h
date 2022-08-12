// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: BSD-3-Clause
// See: https://spdx.org/licenses/

#ifndef UTILS_H_
#define UTILS_H_

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

namespace message_infrastructure {

enum ChannelType {
  SHMEMCHANNEL = 0,
  RPCCHANNEL = 1,
  DDSCHANNEL = 2
};

struct Proto {
  ssize_t *shape_ = NULL;
  pybind11::dtype dtype_;
  size_t nbytes_;
};

}  // namespace message_infrastructure

#endif  // UTILS_H_
