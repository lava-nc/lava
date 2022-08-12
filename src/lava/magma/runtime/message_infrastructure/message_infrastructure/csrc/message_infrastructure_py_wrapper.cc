// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: BSD-3-Clause
// See: https://spdx.org/licenses/

#include <pybind11/functional.h>
#include <pybind11/pybind11.h>

#include "channel_factory.h"
#include "multiprocessing.h"
#include "shm.h"
#include "shmem_channel.h"
#include "shmem_port.h"

namespace message_infrastructure {

namespace py = pybind11;

PYBIND11_MODULE(MessageInfrastructurePywrapper, m) {
  py::class_<MultiProcessing> (m, "MultiProcessing")
    .def(py::init<>())
    .def("build_actor", &MultiProcessing::BuildActor)
    .def("check_actor", &MultiProcessing::CheckActor)
    .def("stop", &MultiProcessing::Stop);

  // using ShmemSendFloatPort = ShmemSendPort<float>;
  // using ShmemRecvFloatPort = ShmemRecvPort<float>;

  // py::class_<ShmemSendFloatPort> (m, "ShmemSendFloatPort")
  //   .def("start", &ShmemSendFloatPort::Start)
  //   .def("probe", &ShmemSendFloatPort::Probe)
  //   .def("send", &ShmemSendFloatPort::Send)
  //   .def("join", &ShmemSendFloatPort::Join);

  // py::class_<ShmemRecvFloatPort> (m, "ShmemRecvFloatPort")
  //   .def("start", &ShmemRecvFloatPort::Start)
  //   .def("probe", &ShmemRecvFloatPort::Probe)
  //   .def("recv", &ShmemRecvFloatPort::Recv)
  //   .def("join", &ShmemRecvFloatPort::Join)
  //   .def("peek", &ShmemRecvFloatPort::Peek);

  py::class_<ShmemChannel> (m, "ShmemChannel")
    .def("get_srcport", &ShmemChannel::GetSrcPort)
    .def("get_dstport", &ShmemChannel::GetDstPort);

  // py::class_<ChannelFactory> (m, "ChannelFactory")
  //   .def("get_channel_factory", &ChannelFactory::GetChannelFactory)
  //   .def("get_float_channel", &ChannelFactory::GetChannel<float>);
}

} // namespace message_infrastructure
