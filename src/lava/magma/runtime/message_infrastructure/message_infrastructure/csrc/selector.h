// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: BSD-3-Clause
// See: https://spdx.org/licenses/

#ifndef SELECTOR_H_
#define SELECTOR_H_

#include "port_proxy.h"
#include <functional>

namespace message_infrastructure {

class AbstractSelector {
};

class ShmemSelector : public AbstractSelector {
 public:
  char* select(RecvPortProxyPtr recv_ptr, char* str) {
    return str;
  }
};

using ShmemSelectorPtr = ShmemSelector *;

}  // namespace message_infrastructure

#endif  // SELECTOR_H_
