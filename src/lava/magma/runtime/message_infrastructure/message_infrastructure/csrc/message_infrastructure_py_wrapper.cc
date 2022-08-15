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

  py::class_<AbstractPyPort> (m, "AbstractCppPort")
    .def(py::init<>());

  py::class_<AbstractPyIOPort> (m, "AbstractCppIOPort")
    .def("ports", &AbstractCppIOPort::GetPorts);
  
  py::class_<PyInPort> (m, "CppInPort")
    .def("probe", &CppInPort::Probe);
  
  py::class_<PyInPortVectorDense> (m, "CppInPortVectorDense")
    .def("recv", &CppInPortVectorDense::Recv)
    .def("peek", &CppInPortVectorDense::Peek);
  
  py::class_<PyInPortVectorSparse> (m, "CppInPortVectorSparse")
    .def("recv", &CppInPortVectorSparse::Recv)
    .def("peek", &CppInPortVectorSparse::Peek);

  py::class_<PyInPortScalarDense> (m, "CppInPortScalarDense")
    .def("recv", &CppInPortScalarDense::Recv)
    .def("peek", &CppInPortScalarDense::Peek);
  
  py::class_<PyInPortScalarSparse> (m, "CppInPortScalarSparse")
    .def("recv", &CppInPortScalarSparse::Recv)
    .def("peek", &CppInPortScalarSparse::Peek);
  
  py::class_<PyOutPort> (m, "CppOutPort")
    .def(py::init<>());
  
  py::class_<PyOutPortVectorDense> (m, "CppOutPortVectorDense")
    .def("send", &CppOutPortVectorDense::Send);
  
  py::class_<PyOutPortVectorSparse> (m, "CppOutPortVectorSparse")
    .def("send", &CppOutPortVectorSparse::Send);
  
  py::class_<PyOutPortScalarDense> (m, "CppOutPortScalarDense")
    .def("send", &CppOutPortScalarDense::Send);

  py::class_<PyOutPortScalarSparse> (m, "CppOutPortScalarSparse")
    .def("send", &CppOutPortScalarSparse::Send);
  
  py::class_<PyRefPort> (m, "CppRefPort")
    .def(py::init<>())
    .def("ports", &CppRefPort::GetPorts);
  
  py::class_<PyRefPortVectorDense> (m, "CppRefPortVectorDense")
    .def("read", &CppRefPortVectorDense::Read)
    .def("write", &CppRefPortVectorDense::Write);
  
  py::class_<PyRefPortVectorSparse> (m, "CppRefPortVectorSparse")
    .def("read", &CppRefPortVectorSparse::Read)
    .def("write", &CppRefPortVectorSparse::Write);

  py::class_<PyRefPortScalarDense> (m, "CppRefPortScalarDense")
    .def("read", &CppRefPortScalarDense::Read)
    .def("write", &CppRefPortScalarDense::Write);
  
  py::class_<PyRefPortScalarSparse> (m, "CppRefPortScalarSparse")
    .def("read", &CppRefPortScalarSparse::Read)
    .def("write", &CppRefPortScalarSparse::Write);
  
  py::class_<PyVarPort> (m, "CppVarPort")
    .def(py::init<>())
    .def("csp_ports", &CppVarPort::GetCspPorts);
  
  py::class_<PyVarPortVectorDense> (m, "CppVarPortVectorDense")
    .def("service", &CppVarPortVectorDense::Service);
  
  py::class_<PyVarPortVectorSparse> (m, "CppVarPortVectorSparse")
    .def("recv", &CppVarPortVectorSparse::Recv)
    .def("peek", &CppVarPortVectorSparse::Peek)
    .def("service", &CppVarPortVectorSparse::Service);
  
  py::class_<PyVarPortScalarDense> (m, "CppVarPortScalarDense")
    .def("recv", &CppVarPortScalarDense::Recv)
    .def("peek", &CppVarPortScalarDense::Peek)
    .def("service", &CppVarPortScalarDense::Service);

  py::class_<PyVarPortScalarSparse> (m, "CppVarPortScalarSparse")
    .def("recv", &CppVarPortScalarSparse::Recv)
    .def("peek", &CppVarPortScalarSparse::Peek)
    .def("service", &CppVarPortScalarSparse::Service);
}


}  // namespace message_infrastructure
