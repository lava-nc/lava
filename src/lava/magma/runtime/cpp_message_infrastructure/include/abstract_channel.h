// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: BSD-3-Clause
// See: https://spdx.org/licenses/

#ifndef ABSTACT_CHANNEL_H_
#define ABSTACT_CHANNEL_H_

#include "abstract_port.h"
#include "utils.h"

class AbstractChannel {
 public:
  AbstractSendPort *src_port_ = NULL;
  AbstractSendPort *dst_port_ = NULL;
};

#endif