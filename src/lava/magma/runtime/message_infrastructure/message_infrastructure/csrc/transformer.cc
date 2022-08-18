// Copyright (C) 2021-22 Intel Corporation
// SPDX-License-Identifier: BSD-3-Clause
// See: https://spdx.org/licenses/

#include <transformer.h>

namespace message_infrastructure {

// IdentityTransformer
std::vector<pybind11::array_t<pybind11::dtype>> 
    IdentityTransformer::Transform(pybind11::array_t<pybind11::dtype> data) {
    return data;
}

}  // namespace message_infrastructure