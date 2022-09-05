// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: BSD-3-Clause
// See: https://spdx.org/licenses/

#ifndef TRANSFORMER_H_
#define TRANSFORMER_H_

#include <vector>

#include "abstract_port_implementation.h"

namespace message_infrastructure {

class AbstractTransformer {
 public:
    int Transform() {
        return 0;
    }
};

class IdentityTransformer: public AbstractTransformer {
 public:
    int Transform() {
         return 0;
    }
};

class VirtualPortTransformer: public AbstractTransformer {
 public:
    int Transform() {
        return 0;
    }
    int _Get_Transform() {
        return 0;
    }
};

}  // namespace message_infrastructure

#endif  // TRANSFORMER_H_
