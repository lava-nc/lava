// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: BSD-3-Clause
// See: https://spdx.org/licenses/

#include <pybind11/functional.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "multiprocessing.h"
#include "abstract_actor.h"
#include "shm.h"
// #include "shmem_channel.h"
// #include "shmem_port.h"
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
  py::class_<PosixActor> (m, "Actor")
    .def("wait", &PosixActor::Wait)
    .def("stop", &PosixActor::Stop)
    .def("get_status", &PosixActor::GetStatus);
    // .def("trace", &PosixActor::Trace);
  
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

  py::class_<AbstractCppPort> (m, "AbstractPyPort")
    .def(py::init<>());

  py::class_<AbstractCppIOPort> (m, "AbstractPyIOPort")
    .def("ports", &AbstractCppIOPort::GetPorts);
  
  py::class_<CppInPort> (m, "PyInPort")
    .def("probe", &CppInPort::Probe);
  
  py::class_<CppInPortVectorDense> (m, "PyInPortVectorDense")
    .def("recv", &CppInPortVectorDense::Recv)
    .def("peek", &CppInPortVectorDense::Peek);
  
  py::class_<CppInPortVectorSparse> (m, "PyInPortVectorSparse")
    .def("recv", &CppInPortVectorSparse::Recv)
    .def("peek", &CppInPortVectorSparse::Peek);

  py::class_<CppInPortScalarDense> (m, "PyInPortScalarDense")
    .def("recv", &CppInPortScalarDense::Recv)
    .def("peek", &CppInPortScalarDense::Peek);
  
  py::class_<CppInPortScalarSparse> (m, "PyInPortScalarSparse")
    .def("recv", &CppInPortScalarSparse::Recv)
    .def("peek", &CppInPortScalarSparse::Peek);
  
  py::class_<CppOutPort> (m, "PyOutPort")
    .def(py::init<>());
  
  py::class_<CppOutPortVectorDense> (m, "PyOutPortVectorDense")
    .def("send", &CppOutPortVectorDense::Send);
  
  py::class_<CppOutPortVectorSparse> (m, "PyOutPortVectorSparse")
    .def("send", &CppOutPortVectorSparse::Send);
  
  py::class_<CppOutPortScalarDense> (m, "PyOutPortScalarDense")
    .def("send", &CppOutPortScalarDense::Send);

  py::class_<CppOutPortScalarSparse> (m, "PyOutPortScalarSparse")
    .def("send", &CppOutPortScalarSparse::Send);
  
  py::class_<CppRefPort> (m, "PyRefPort")
    .def(py::init<>())
    .def("ports", &CppRefPort::GetPorts);
  
  py::class_<CppRefPortVectorDense> (m, "PyRefPortVectorDense")
    .def("read", &CppRefPortVectorDense::Read)
    .def("write", &CppRefPortVectorDense::Write);
  
  py::class_<CppRefPortVectorSparse> (m, "PyRefPortVectorSparse")
    .def("read", &CppRefPortVectorSparse::Read)
    .def("write", &CppRefPortVectorSparse::Write);

  py::class_<CppRefPortScalarDense> (m, "PyRefPortScalarDense")
    .def("read", &CppRefPortScalarDense::Read)
    .def("write", &CppRefPortScalarDense::Write);
  
  py::class_<CppRefPortScalarSparse> (m, "PyRefPortScalarSparse")
    .def("read", &CppRefPortScalarSparse::Read)
    .def("write", &CppRefPortScalarSparse::Write);
  
  py::class_<CppVarPort> (m, "PyVarPort")
    .def(py::init<>());
    // .def("csp_ports", &CppVarPort::GetCspPorts);
  
  py::class_<CppVarPortVectorDense> (m, "PyVarPortVectorDense")
    .def("service", &CppVarPortVectorDense::Service);
  
  py::class_<CppVarPortVectorSparse> (m, "PyVarPortVectorSparse")
    .def("recv", &CppVarPortVectorSparse::Recv)
    .def("peek", &CppVarPortVectorSparse::Peek)
    .def("service", &CppVarPortVectorSparse::Service);
  
  py::class_<CppVarPortScalarDense> (m, "PyVarPortScalarDense")
    .def("recv", &CppVarPortScalarDense::Recv)
    .def("peek", &CppVarPortScalarDense::Peek)
    .def("service", &CppVarPortScalarDense::Service);

  py::class_<CppVarPortScalarSparse> (m, "PyVarPortScalarSparse")
    .def("recv", &CppVarPortScalarSparse::Recv)
    .def("peek", &CppVarPortScalarSparse::Peek)
    .def("service", &CppVarPortScalarSparse::Service);
  

  /*
  py::class_<AbstractTransformer> (m, "AbstractTransformer")
    .def("transform", &AbstractTransformer::Transform);

  py::class_<IdentityTransformer> (m, "IdentityTransformer")
    .def("transform", &IdentityTransformer::Transform);

  py::class_<VirtualPortTransformer> (m, "VirtualPortTransformer")
    .def("transform", &VirtualPortTransformer::Transform)
    .def("_get_transform", &VirtualPortTransformer::_Get_Transform);
  */
}

}  // namespace message_infrastructure
