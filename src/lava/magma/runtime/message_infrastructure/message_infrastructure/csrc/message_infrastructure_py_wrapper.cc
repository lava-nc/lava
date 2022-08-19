// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: BSD-3-Clause
// See: https://spdx.org/licenses/

#include <memory>

#include <pybind11/functional.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "abstract_actor.h"
#include "channel_factory.h"
#include "multiprocessing.h"
#include "port_proxy.h"
#include "shm.h"
#include "shmem_channel.h"
#include "shmem_port.h"
#include "utils.h"

namespace message_infrastructure {

namespace py = pybind11;

PYBIND11_MODULE(MessageInfrastructurePywrapper, m) {
  py::class_<MultiProcessing> (m, "CppMultiProcessing")
    .def(py::init<>())
    .def("build_actor", &MultiProcessing::BuildActor)
    .def("check_actor", &MultiProcessing::CheckActor)
    .def("get_actors", &MultiProcessing::GetActors)
    .def("get_shmm", &MultiProcessing::GetSharedMemManager)
    .def("stop", &MultiProcessing::Stop);
  py::enum_<ProcessType> (m, "ProcessType")
    .value("ErrorProcess", ErrorProcess)
    .value("ChildProcess", ChildProcess)
    .value("ParentProcess", ParentProcess);
  py::class_<SharedMemManager> (m, "SharedMemManager")
    .def(py::init<>())
    .def("alloc_mem", &SharedMemManager::AllocSharedMemory)
    .def("stop", &SharedMemManager::Stop);
  py::class_<SharedMemory> (m, "SharedMemory")
    .def(py::init<>());
  py::class_<PosixActor> (m, "Actor")
    .def("wait", &PosixActor::Wait)
    .def("stop", &PosixActor::Stop)
    .def("get_status", &PosixActor::GetStatus);
    // .def("trace", &PosixActor::Trace);
  py::enum_<ChannelType> (m, "ChannelType")
    .value("SHMEMCHANNEL", SHMEMCHANNEL)
    .value("RPCCHANNEL", RPCCHANNEL)
    .value("DDSCHANNEL", DDSCHANNEL);
  py::class_<AbstractChannel> (m, "AbstractChannel")
    .def(py::init<>())
    .def("get_send_port", &AbstractChannel::GetSendPort)
    .def("get_recv_port", &AbstractChannel::GetRecvPort);
  py::class_<ShmemChannel> (m, "ShmemChannel")
    .def("get_send_port", &ShmemChannel::GetSendPort, py::return_value_policy::reference)
    .def("get_recv_port", &ShmemChannel::GetRecvPort, py::return_value_policy::reference);
  py::class_<SendPortProxy> (m, "SendPortProxy")
    .def(py::init<ChannelType, AbstractSendPortPtr>())
    .def("get_channel_type", &SendPortProxy::GetChannelType)
    .def("get_send_port", &SendPortProxy::GetSendPort, py::return_value_policy::reference)
    .def("start", &SendPortProxy::Start)
    .def("probe", &SendPortProxy::Probe)
    .def("send", &SendPortProxy::Send)
    .def("join", &SendPortProxy::Join)
    .def("name", &SendPortProxy::Name)
    .def("dtype", &SendPortProxy::Dtype)
    .def("shape", &SendPortProxy::Shape)
    .def("size", &SendPortProxy::Size);
  py::class_<RecvPortProxy> (m, "RecvPortProxy")
    .def(py::init<ChannelType, AbstractRecvPortPtr>())
    .def("get_channel_type", &RecvPortProxy::GetChannelType)
    .def("get_recv_port", &RecvPortProxy::GetRecvPort, py::return_value_policy::reference)
    .def("start", &RecvPortProxy::Start)
    .def("probe", &RecvPortProxy::Probe)
    .def("recv", &RecvPortProxy::Recv)
    .def("peek", &RecvPortProxy::Peek)
    .def("join", &RecvPortProxy::Join)
    .def("name", &RecvPortProxy::Name)
    .def("dtype", &RecvPortProxy::Dtype)
    .def("shape", &RecvPortProxy::Shape)
    .def("size", &RecvPortProxy::Size);
  py::class_<ChannelFactory> (m, "ChannelFactory")
    .def("get_channel", &ChannelFactory::GetChannel<double>, py::return_value_policy::reference)
    .def("get_channel", &ChannelFactory::GetChannel<std::int16_t>, py::return_value_policy::reference)
    .def("get_channel", &ChannelFactory::GetChannel<std::int32_t>, py::return_value_policy::reference)
    .def("get_channel", &ChannelFactory::GetChannel<std::int64_t>, py::return_value_policy::reference)
    .def("get_channel", &ChannelFactory::GetChannel<float>, py::return_value_policy::reference);
  m.def("get_channel_factory", GetChannelFactory, py::return_value_policy::reference);
  m.def("test_get_channel", TestGetChannel<double>, py::return_value_policy::reference);
  m.def("test_get_channel", TestGetChannel<std::int16_t>, py::return_value_policy::reference);
  m.def("test_get_channel", TestGetChannel<std::int32_t>, py::return_value_policy::reference);
  m.def("test_get_channel", TestGetChannel<std::int64_t>, py::return_value_policy::reference);
}

} // namespace message_infrastructure
