// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: BSD-3-Clause
// See: https://spdx.org/licenses/

#include <pybind11/functional.h>
#include <pybind11/pybind11.h>

#include "multiprocessing.h"
//#include "abstract_port_implementation.h"

namespace message_infrastructure {

namespace py = pybind11;

PYBIND11_MODULE(MessageInfrastructurePywrapper, m) {
  py::class_<MultiProcessing> (m, "MultiProcessing")
    .def(py::init<>())
    .def("build_actor", &MultiProcessing::BuildActor)
    .def("check_actor", &MultiProcessing::CheckActor)
    .def("stop", &MultiProcessing::Stop);
}

} // namespace message_infrastructure
