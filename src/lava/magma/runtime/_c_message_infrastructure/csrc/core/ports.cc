// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: BSD-3-Clause
// See: https://spdx.org/licenses/

#include <message_infrastructure/csrc/core/ports.h>

namespace message_infrastructure {

CppInPort::CppInPort(const RecvPortList &recv_ports)
  : AbstractPortImplementation(recv_ports)
{}

bool CppInPort::Probe() {
  return true;
}

int CppInPortVectorDense::Recv() {
  // Todo
  return 0;
}

int CppInPortVectorDense::Peek() {
  // Todo
  return 0;
}

int CppInPortVectorSparse::Recv() {
  // Todo
  return 0;
}

int CppInPortVectorSparse::Peek() {
  // Todo
  return 0;
}

int CppInPortScalarDense::Recv() {
  // Todo
  return 0;
}

int CppInPortScalarDense::Peek() {
  // Todo
  return 0;
}

int CppInPortScalarSparse::Recv() {
  // Todo
  return 0;
}

int CppInPortScalarSparse::Peek() {
  // Todo
  return 0;
}

CppOutPort::CppOutPort(const SendPortList &send_ports)
  : AbstractPortImplementation(send_ports)
{}

int CppOutPortVectorDense::Send() {
  // Todo
  return 0;
}

int CppOutPortVectorSparse::Send() {
  // Todo
  return 0;
}

int CppOutPortScalarDense::Send() {
  // Todo
  return 0;
}

int CppOutPortScalarSparse::Send() {
  // Todo
  return 0;
}

CppRefPort::CppRefPort(const SendPortList &send_ports,
                        const RecvPortList &recv_ports)
  : AbstractPortImplementation(send_ports, recv_ports)
{}

int CppRefPort::Wait() {
  // Todo
  return 0;
}

int CppRefPortVectorDense::Read() {
  // Todo
  return 0;
}

int CppRefPortVectorDense::Write() {
  // Todo
  return 0;
}

int CppRefPortVectorSparse::Read() {
  // Todo
  return 0;
}

int CppRefPortVectorSparse::Write() {
  // Todo
  return 0;
}

int CppRefPortScalarDense::Read() {
  // Todo
  return 0;
}

int CppRefPortScalarDense::Write() {
  // Todo
  return 0;
}

int CppRefPortScalarSparse::Read() {
  // Todo
  return 0;
}

int CppRefPortScalarSparse::Write() {
  // Todo
  return 0;
}

CppVarPort::CppVarPort(const std::string &name,
                       const SendPortList &send_ports,
                       const RecvPortList &recv_ports)
  : name_(name), AbstractPortImplementation(send_ports, recv_ports)
{}

int CppVarPortVectorDense::Service() {
  // Todo
  return 0;
}

int CppVarPortVectorDense::Recv() {
  // Todo
  return 0;
}

int CppVarPortVectorDense::Peek() {
  // Todo
  return 0;
}

int CppVarPortVectorSparse::Service() {
  // Todo
  return 0;
}

int CppVarPortVectorSparse::Recv() {
  // Todo
  return 0;
}

int CppVarPortVectorSparse::Peek() {
  // Todo
  return 0;
}

int CppVarPortScalarDense::Service() {
  // Todo
  return 0;
}

int CppVarPortScalarDense::Recv() {
  // Todo
  return 0;
}

int CppVarPortScalarDense::Peek() {
  // Todo
  return 0;
}

int CppVarPortScalarSparse::Service() {
  // Todo
  return 0;
}

int CppVarPortScalarSparse::Recv() {
  // Todo
  return 0;
}

int CppVarPortScalarSparse::Peek() {
  // Todo
  return 0;
}

}  // namespace message_infrastructure
