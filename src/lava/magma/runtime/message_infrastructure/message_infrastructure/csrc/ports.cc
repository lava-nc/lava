// Copyright (C) 2021-22 Intel Corporation
// SPDX-License-Identifier: BSD-3-Clause
// See: https://spdx.org/licenses/

#include <ports.h>
#include <vector>

namespace message_infrastructure {

std::vector<PortPtr> AbstractCppIOPort::GetPorts() {
  return this->ports_;
}

bool CppInPort::Probe() {
  auto lambda = [&](int acc, int port){return acc && port->Probe();};
  return std::accumulate(ports_.begin(), ports_.end(), true, lambda);
}
 //TODO: Implement transform and change ndarray type 
ndarray CppInPortVectorDense::Recv(){
  auto lambda = [&](int acc, int port){return acc + port->Transform();}
}

int CppInPortScalarDense::Recv() {
}

int CppInPortScalarDense::Peek() {
}

std::vector<int> CppInPortScalarSparse::Recv(){
}

std::vector<int> CppInPortScalarSparse::Peek(){
}

std::vector <ndarray> CppOutPortVectorDense::Send(ndarray data){
  for (auto port : this->ports_){
    port->Send(data);
  }
}

}
