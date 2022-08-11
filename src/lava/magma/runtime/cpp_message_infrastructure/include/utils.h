// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: BSD-3-Clause
// See: https://spdx.org/licenses/

#ifndef UTILS_H_
#define UTILS_H_

namespace message_infrastructure {

enum ChannelType {
  ShmemChannel = 0,
  RpcChannel = 1,
  DdsChannel = 2
};

enum DataType {
  // dtype
};

} // namespace message_infrastructure

#endif