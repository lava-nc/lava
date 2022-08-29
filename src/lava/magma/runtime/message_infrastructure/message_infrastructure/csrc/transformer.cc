// Copyright (C) 2021-22 Intel Corporation
// SPDX-License-Identifier: BSD-3-Clause
// See: https://spdx.org/licenses/

#include <transformer.h>

namespace message_infrastructure {

// IdentityTransformer
std::vector<int> IdentityTransformer::Transform(std::vector<int> data) {
    return data;
}

}  // namespace message_infrastructure