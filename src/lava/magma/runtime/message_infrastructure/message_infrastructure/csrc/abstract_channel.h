// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: BSD-3-Clause
// See: https://spdx.org/licenses/

#ifndef ABSTRACT_CHANNEL_H_
#define ABSTRACT_CHANNEL_H_

#include <memory>

#include "abstract_port.h"
#include "utils.h"

namespace message_infrastructure {

class AbstractChannel {
 public:
  std::shared_ptr<AbstractSendPort> src_port_;
  std::shared_ptr<AbstractRecvPort> dst_port_;
};

using AbstractChannelPtr = AbstractChannel *;

} // namespace message_infrastructure

#endif  // ABSTRACT_CHANNEL_H_
