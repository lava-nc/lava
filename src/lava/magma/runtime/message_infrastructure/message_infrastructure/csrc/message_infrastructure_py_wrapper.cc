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
#include "ports.h"
#include "transformer.h"

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
  py::class_<AbstractChannel, std::shared_ptr<AbstractChannel>> (m, "AbstractChannel")
    .def(py::init<>())
    .def("get_send_port", &AbstractChannel::GetSendPort)
    .def("get_recv_port", &AbstractChannel::GetRecvPort);
  py::class_<ShmemChannel> (m, "ShmemChannel")
    .def("get_send_port", &ShmemChannel::GetSendPort, py::return_value_policy::reference)
    .def("get_recv_port", &ShmemChannel::GetRecvPort, py::return_value_policy::reference);
  py::class_<SendPortProxy, std::shared_ptr<SendPortProxy>> (m, "SendPortProxy")
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
  py::class_<RecvPortProxy, std::shared_ptr<RecvPortProxy>> (m, "RecvPortProxy")
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


  py::class_<CppInPortVectorDense, std::shared_ptr<CppInPortVectorDense>> (m, "PyInPortVectorDense")
    .def(py::init<RecvPortProxyList>())
    .def("recv", &CppInPortVectorDense::Recv)
    .def("peek", &CppInPortVectorDense::Peek);

  py::class_<CppInPortVectorSparse, std::shared_ptr<CppInPortVectorSparse>> (m, "PyInPortVectorSparse")
    .def(py::init<RecvPortProxyList>())
    .def("recv", &CppInPortVectorSparse::Recv)
    .def("peek", &CppInPortVectorSparse::Peek);

  py::class_<CppInPortScalarDense, std::shared_ptr<CppInPortScalarDense>> (m, "PyInPortScalarDense")
    .def(py::init<RecvPortProxyList>())
    .def("recv", &CppInPortScalarDense::Recv)
    .def("peek", &CppInPortScalarDense::Peek);

  py::class_<CppInPortScalarSparse, std::shared_ptr<CppInPortScalarSparse>> (m, "PyInPortScalarSparse")
    .def(py::init<RecvPortProxyList>())
    .def("recv", &CppInPortScalarSparse::Recv)
    .def("peek", &CppInPortScalarSparse::Peek);

  py::class_<CppOutPortVectorDense, std::shared_ptr<CppOutPortVectorDense>> (m, "PyOutPortVectorDense")
    .def(py::init<SendPortProxyList>())
    .def("send", &CppOutPortVectorDense::Send)
    .def("flush", &CppOutPortVectorDense::Flush);

  py::class_<CppOutPortVectorSparse, std::shared_ptr<CppOutPortVectorSparse>> (m, "PyOutPortVectorSparse")
    .def(py::init<SendPortProxyList>())
    .def("send", &CppOutPortVectorSparse::Send)
    .def("flush", &CppOutPortVectorSparse::Flush);

  py::class_<CppOutPortScalarDense, std::shared_ptr<CppOutPortScalarDense>> (m, "PyOutPortScalarDense")
    .def(py::init<SendPortProxyList>())
    .def("send", &CppOutPortScalarDense::Send)
    .def("flush", &CppOutPortScalarDense::Flush);

  py::class_<CppOutPortScalarSparse, std::shared_ptr<CppOutPortScalarSparse>> (m, "PyOutPortScalarSparse")
    .def(py::init<SendPortProxyList>())
    .def("send", &CppOutPortScalarSparse::Send)
    .def("flush", &CppOutPortScalarSparse::Flush);

  py::class_<CppRefPortVectorDense, std::shared_ptr<CppRefPortVectorDense>> (m, "PyRefPortVectorDense")
    .def(py::init<SendPortProxyList, RecvPortProxyList>())
    .def("read", &CppRefPortVectorDense::Read)
    .def("write", &CppRefPortVectorDense::Write);

  py::class_<CppRefPortVectorSparse, std::shared_ptr<CppRefPortVectorSparse>> (m, "PyRefPortVectorSparse")
    .def(py::init<SendPortProxyList, RecvPortProxyList>())
    .def("read", &CppRefPortVectorSparse::Read)
    .def("write", &CppRefPortVectorSparse::Write);

  py::class_<CppRefPortScalarDense, std::shared_ptr<CppRefPortScalarDense>> (m, "PyRefPortScalarDense")
    .def(py::init<SendPortProxyList, RecvPortProxyList>())
    .def("read", &CppRefPortScalarDense::Read)
    .def("write", &CppRefPortScalarDense::Write);

  py::class_<CppRefPortScalarSparse, std::shared_ptr<CppRefPortScalarSparse>> (m, "PyRefPortScalarSparse")
    .def(py::init<SendPortProxyList, RecvPortProxyList>())
    .def("read", &CppRefPortScalarSparse::Read)
    .def("write", &CppRefPortScalarSparse::Write);

  py::class_<CppVarPortVectorDense, std::shared_ptr<CppVarPortVectorDense>> (m, "PyVarPortVectorDense")
    .def(py::init<std::string, SendPortProxyList, RecvPortProxyList>())
    .def("service", &CppVarPortVectorDense::Service)
    .def("recv", &CppVarPortVectorDense::Recv)
    .def("peek", &CppVarPortVectorDense::Peek);

  py::class_<CppVarPortVectorSparse, std::shared_ptr<CppVarPortVectorSparse>> (m, "PyVarPortVectorSparse")
    .def(py::init<std::string, SendPortProxyList, RecvPortProxyList>())
    .def("service", &CppVarPortVectorSparse::Service)
    .def("recv", &CppVarPortVectorSparse::Recv)
    .def("peek", &CppVarPortVectorSparse::Peek);

  py::class_<CppVarPortScalarDense, std::shared_ptr<CppVarPortScalarDense>> (m, "PyVarPortScalarDense")
    .def(py::init<std::string, SendPortProxyList, RecvPortProxyList>())
    .def("service", &CppVarPortScalarDense::Service)
    .def("recv", &CppVarPortScalarDense::Recv)
    .def("peek", &CppVarPortScalarDense::Peek);

  py::class_<CppVarPortScalarSparse, std::shared_ptr<CppVarPortScalarSparse>> (m, "PyVarPortScalarSparse")
    .def(py::init<std::string, SendPortProxyList, RecvPortProxyList>())
    .def("service", &CppVarPortScalarSparse::Service)
    .def("recv", &CppVarPortScalarSparse::Recv)
    .def("peek", &CppVarPortScalarSparse::Peek);
}

}  // namespace message_infrastructure
