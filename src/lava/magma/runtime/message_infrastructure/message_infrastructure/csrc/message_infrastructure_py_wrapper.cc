// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: BSD-3-Clause
// See: https://spdx.org/licenses/

#include <pybind11/functional.h>
#include <pybind11/pybind11.h>

#include "multiprocessing.h"
// #include "shm.h"
// #include "shmem_channel.h"
// #include "shmem_port.h"

namespace message_infrastructure {

namespace py = pybind11;

PYBIND11_MODULE(MessageInfrastructurePywrapper, m) {
  py::class_<MultiProcessing> (m, "MultiProcessing")
    .def(py::init<>())
    .def("build_actor", &MultiProcessing::BuildActor)
    .def("check_actor", &MultiProcessing::CheckActor)
    .def("stop", &MultiProcessing::Stop);
  /*
  py::class_<ShmemSendPort> (m, "ShmemSendPort")
    .def(py::init<std::string, SharedMemory*, Proto*, size_t, sem_t*, sem_t*>())
    .def("start", &ShmemSendPort::Start)
    .def("probe", &ShmemSendPort::Probe)
    .def("send", &ShmemSendPort::Send)
    .def("join", &ShmemSendPort::Join)
    .def("ack_callback", &ShmemSendPort::_ack_callback);
  py::class_<ShmemRecvPort> (m, "ShmemRecvPort")
    .def(py::init<std::string, SharedMemory*, Proto*, size_t, sem_t*, sem_t*>())
    .def("start", &ShmemRecvPort::Start)
    .def("probe", &ShmemRecvPort::Probe)
    .def("recv", &ShmemRecvPort::Recv)
    .def("join", &ShmemRecvPort::Join)
    .def("peek", &ShmemRecvPort::Peek)
    .def("req_callback", &ShmemRecvPort::_req_callback);
  py::class_<ShmemChannel> (m, "ShmemChannel")
    .def(py::init<SharedMemory*, std::string, std::string, ssize_t*, DataType, size_t>())
    .def("get_srcport", &ShmemChannel::GetSrcPort, return_value_policy::reference)
    .def("get_dstport", &ShmemChannel::GetDstPort, return_value_policy::reference);
  m.def("get_shmemchannel", &GetShmemChannel, return_value_policy::reference);
  */
}


} // namespace message_infrastructure
