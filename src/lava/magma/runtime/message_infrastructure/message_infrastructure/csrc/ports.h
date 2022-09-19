// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: BSD-3-Clause
// See: https://spdx.org/licenses/

#ifndef PORTS_H_
#define PORTS_H_

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

#include <vector>
#include <variant>
#include <memory>
#include <string>

#include "abstract_port_implementation.h"

namespace message_infrastructure {

class CppInPort : public AbstractPortImplementation {
 public:
    explicit CppInPort(const RecvPortProxyList &recv_ports);
    bool Probe();
    virtual int Recv() = 0;
    virtual int Peek() = 0;
};


class CppInPortVectorDense final : public CppInPort {
 public:
    using CppInPort::CppInPort;
    int Recv() override;
    int Peek() override;
};


class CppInPortVectorSparse final : public CppInPort {
 public:
    using CppInPort::CppInPort;
    int Recv() override;
    int Peek() override;
};


class CppInPortScalarDense final : public CppInPort {
 public:
    using CppInPort::CppInPort;
    int Recv() override;
    int Peek() override;
};


class CppInPortScalarSparse final : public CppInPort {
 public:
    using CppInPort::CppInPort;
    int Recv() override;
    int Peek() override;
};


class CppOutPort : public AbstractPortImplementation {
 public:
    explicit CppOutPort(const SendPortProxyList &send_ports);
    virtual int Send() = 0;
    void Flush() {}
};


class CppOutPortVectorDense final : public CppOutPort {
 public:
    using CppOutPort::CppOutPort;
    int Send() override;
};


class CppOutPortVectorSparse final : public CppOutPort {
 public:
    using CppOutPort::CppOutPort;
    int Send() override;
};


class CppOutPortScalarDense final : public CppOutPort {
 public:
    using CppOutPort::CppOutPort;
    int Send() override;
};


class CppOutPortScalarSparse final : public CppOutPort {
 public:
    using CppOutPort::CppOutPort;
    int Send() override;
};


class CppRefPort : public AbstractPortImplementation {
 public:
    explicit CppRefPort(const SendPortProxyList &send_ports,
                        const RecvPortProxyList &recv_ports);
    virtual int Read() = 0;
    virtual int Write() = 0;
    int Wait() { return 0; }
};


class CppRefPortVectorDense final : public CppRefPort {
 public:
    using CppRefPort::CppRefPort;
    int Read() override;
    int Write() override;
};


class CppRefPortVectorSparse final : public CppRefPort {
 public:
    using CppRefPort::CppRefPort;
    int Read() override;
    int Write() override;
};


class CppRefPortScalarDense final : public CppRefPort {
 public:
    using CppRefPort::CppRefPort;
    int Read() override;
    int Write() override;
};


class CppRefPortScalarSparse final : public CppRefPort {
 public:
    using CppRefPort::CppRefPort;
    int Read() override;
    int Write() override;
};


class CppVarPort : public AbstractPortImplementation {
 public:
    explicit CppVarPort(const std::string &name,
                        const SendPortProxyList &send_ports,
                        const RecvPortProxyList &recv_ports);

    virtual int Service() = 0;
    virtual int Recv()  = 0;
    virtual int Peek() = 0;

 private:
  std::string name_;
};


class CppVarPortVectorDense final : public CppVarPort {
 public:
    using CppVarPort::CppVarPort;
    int Service() override;
    int Recv() override;
    int Peek() override;
};


class CppVarPortVectorSparse final : public CppVarPort {
 public:
    using CppVarPort::CppVarPort;
    int Service() override;
    int Recv() override;
    int Peek() override;
};


class CppVarPortScalarDense final : public CppVarPort {
 public:
    using CppVarPort::CppVarPort;
    int Service() override;
    int Recv() override;
    int Peek() override;
};


class CppVarPortScalarSparse final : public CppVarPort {
 public:
    using CppVarPort::CppVarPort;
    int Service() override;
    int Recv() override;
    int Peek() override;
};


}  // namespace message_infrastructure

#endif  // PORTS_H_
