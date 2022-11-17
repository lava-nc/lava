// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: BSD-3-Clause
// See: https://spdx.org/licenses/

#ifndef CORE_PORTS_H_
#define CORE_PORTS_H_

#include <message_infrastructure/csrc/core/abstract_port_implementation.h>

#include <vector>
#include <variant>
#include <memory>
#include <string>

namespace message_infrastructure {

class CppInPort : public AbstractPortImplementation {
 public:
  explicit CppInPort(const RecvPortList &recv_ports);
  virtual ~CppInPort() = default;
  bool Probe();
  virtual int Recv() = 0;
  virtual int Peek() = 0;
};


class CppInPortVectorDense final : public CppInPort {
 public:
  using CppInPort::CppInPort;
  ~CppInPortVectorDense() override {}
  int Recv() override;
  int Peek() override;
};


class CppInPortVectorSparse final : public CppInPort {
 public:
  using CppInPort::CppInPort;
  ~CppInPortVectorSparse() override {}
  int Recv() override;
  int Peek() override;
};


class CppInPortScalarDense final : public CppInPort {
 public:
  using CppInPort::CppInPort;
  ~CppInPortScalarDense() override {}
  int Recv() override;
  int Peek() override;
};


class CppInPortScalarSparse final : public CppInPort {
 public:
  using CppInPort::CppInPort;
  ~CppInPortScalarSparse() override {}
  int Recv() override;
  int Peek() override;
};


class CppOutPort : public AbstractPortImplementation {
 public:
  explicit CppOutPort(const SendPortList &send_ports);
  virtual ~CppOutPort() = default;
  virtual int Send() = 0;
  void Flush() {}
};


class CppOutPortVectorDense final : public CppOutPort {
 public:
  using CppOutPort::CppOutPort;
  ~CppOutPortVectorDense() override {}
  int Send() override;
};


class CppOutPortVectorSparse final : public CppOutPort {
 public:
  using CppOutPort::CppOutPort;
  ~CppOutPortVectorSparse() override {}
  int Send() override;
};


class CppOutPortScalarDense final : public CppOutPort {
 public:
  using CppOutPort::CppOutPort;
  ~CppOutPortScalarDense() override {}
  int Send() override;
};


class CppOutPortScalarSparse final : public CppOutPort {
 public:
  using CppOutPort::CppOutPort;
  ~CppOutPortScalarSparse() override {}
  int Send() override;
};


class CppRefPort : public AbstractPortImplementation {
 public:
  explicit CppRefPort(const SendPortList &send_ports,
                      const RecvPortList &recv_ports);
  virtual ~CppRefPort() = default;
  virtual int Read() = 0;
  virtual int Write() = 0;
  int Wait();
};


class CppRefPortVectorDense final : public CppRefPort {
 public:
  using CppRefPort::CppRefPort;
  ~CppRefPortVectorDense() override {}
  int Read() override;
  int Write() override;
};


class CppRefPortVectorSparse final : public CppRefPort {
 public:
  using CppRefPort::CppRefPort;
  ~CppRefPortVectorSparse() override {}
  int Read() override;
  int Write() override;
};


class CppRefPortScalarDense final : public CppRefPort {
 public:
  using CppRefPort::CppRefPort;
  ~CppRefPortScalarDense() override {}
  int Read() override;
  int Write() override;
};


class CppRefPortScalarSparse final : public CppRefPort {
 public:
  using CppRefPort::CppRefPort;
  ~CppRefPortScalarSparse() override {}
  int Read() override;
  int Write() override;
};


class CppVarPort : public AbstractPortImplementation {
 public:
  explicit CppVarPort(const std::string &name,
                      const SendPortList &send_ports,
                      const RecvPortList &recv_ports);
  virtual ~CppVarPort() = default;
  virtual int Service() = 0;
  virtual int Recv()  = 0;
  virtual int Peek() = 0;

 private:
  std::string name_;
};


class CppVarPortVectorDense final : public CppVarPort {
 public:
  using CppVarPort::CppVarPort;
  ~CppVarPortVectorDense() override {}
  int Service() override;
  int Recv() override;
  int Peek() override;
};


class CppVarPortVectorSparse final : public CppVarPort {
 public:
  using CppVarPort::CppVarPort;
  ~CppVarPortVectorSparse() override {}
  int Service() override;
  int Recv() override;
  int Peek() override;
};


class CppVarPortScalarDense final : public CppVarPort {
 public:
  using CppVarPort::CppVarPort;
  ~CppVarPortScalarDense() override {}
  int Service() override;
  int Recv() override;
  int Peek() override;
};


class CppVarPortScalarSparse final : public CppVarPort {
 public:
  using CppVarPort::CppVarPort;
  ~CppVarPortScalarSparse() override {}
  int Service() override;
  int Recv() override;
  int Peek() override;
};


}  // namespace message_infrastructure

#endif  // CORE_PORTS_H_
