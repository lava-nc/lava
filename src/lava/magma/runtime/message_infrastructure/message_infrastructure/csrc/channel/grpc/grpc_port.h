// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: BSD-3-Clause
// See: https://spdx.org/licenses/

#ifndef CHANNEL_GRPC_GRPC_PORT_H_
#define CHANNEL_GRPC_GRPC_PORT_H_

#include <grpcpp/grpcpp.h>
#include <grpcpp/ext/proto_server_reflection_plugin.h>
#include <grpcpp/health_check_service_interface.h>

#include <message_infrastructure/csrc/core/utils.h>
#include <message_infrastructure/csrc/core/message_infrastructure_logging.h>
#include <message_infrastructure/csrc/core/abstract_port.h>
#include <message_infrastructure/csrc/channel/grpc/build/grpcchannel.grpc.pb.h>

#include <atomic>
#include <thread> //NOLINT
#include <iostream>
#include <memory>
#include <string>
#include <vector>

namespace message_infrastructure {

using grpc::Server;
using grpc::ServerBuilder;
using grpc::ServerContext;
using grpc::Channel;
using grpc::ClientContext;
using grpc::Status;
using grpcchannel::DataReply;
using grpcchannel::GrpcChannelServer;
using grpcchannel::GrpcMetaData;

using ThreadPtr = std::shared_ptr<std::thread>;
using GrpcMetaDataPtr = std::shared_ptr<GrpcMetaData>;

class GrpcChannelServerImpl final: public GrpcChannelServer::Service{
 public:
  GrpcChannelServerImpl(const std::string& name,
                        const size_t &size,
                        const size_t &nbytes);
  Status RecvArrayData(ServerContext* context,
                       const GrpcMetaData* request,
                       DataReply* reply) override;
  void Push(const GrpcMetaData *src);
  int AvailableCount();
  bool Empty();
  GrpcMetaDataPtr Pop(bool block);
  GrpcMetaDataPtr Front();
  bool Probe();
  void Stop();
  std::vector<GrpcMetaDataPtr> array_;
  std::atomic<uint32_t> read_index_;
  std::atomic<uint32_t> write_index_;

 private:
  std::string name_;
  size_t size_;
  size_t nbytes_;
  std::atomic_bool done_;
};

using ServerImplPtr = std::shared_ptr<GrpcChannelServerImpl>;

class GrpcRecvPort final : public AbstractRecvPort{
 public:
  GrpcRecvPort(const std::string& name,
               const size_t &size,
               const size_t &nbytes,
               const std::string& url);
  ~GrpcRecvPort();
    void Start();
    MetaDataPtr Recv();
    MetaDataPtr Peek();
    void Join();
    bool Probe();
    void GrpcMetaData2MetaData(MetaDataPtr metadata, GrpcMetaDataPtr grpcdata);
 private:
  ServerBuilder builder;
  std::atomic_bool done_;
  std::unique_ptr<Server> server;
  ServerImplPtr serviceptr;
  std::string url_;
};

using GrpcRecvPortPtr = std::shared_ptr<GrpcRecvPort>;

class GrpcSendPort final : public AbstractSendPort{
 public:
  GrpcSendPort(const std::string &name,
               const size_t &size,
               const size_t &nbytes, const std::string& url);

  void Start();
  void Send(MetaDataPtr metadata);
  void Join();
  bool Probe();
  GrpcMetaData MetaData2GrpcMetaData(MetaDataPtr metadata);

 private:
  std::shared_ptr<Channel> channel;
  std::atomic_bool done_;
  std::unique_ptr<GrpcChannelServer::Stub> stub_;
  ThreadPtr ack_callback_thread_ = nullptr;
  std::string url_;
};

using GrpcSendPortPtr = std::shared_ptr<GrpcSendPort>;

}  // namespace message_infrastructure

#endif  // CHANNEL_GRPC_GRPC_PORT_H_

