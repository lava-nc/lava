// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: BSD-3-Clause
// See: https://spdx.org/licenses/

#ifndef ABSTACT_CHANNEL_H_
#define ABSTACT_CHANNEL_H_

#include "abstract_port.h"
#include "utils.h"
#include <memory>

namespace message_infrastructure {

class AbstractChannel {
 public:
  std::shared_ptr<AbstractSendPort> src_port_;
  std::shared_ptr<AbstractRecvPort> dst_port_;
};

} // namespace message_infrastructure

#endif