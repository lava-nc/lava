
// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: BSD-3-Clause
// See: https://spdx.org/licenses/
#include <grpcpp/ext/proto_server_reflection_plugin.h>
#include <grpcpp/grpcpp.h>
#include <grpcpp/health_check_service_interface.h>
#include <message_infrastructure/csrc/core/utils.h>
#include <message_infrastructure/csrc/core/message_infrastructure_logging.h>
#include <message_infrastructure/csrc/core/abstract_port.h>
#include <atomic>
#include <thread> //NOLINT
#include<iostream>
#include <memory>
#include <string>

#include <vector>
#include "message_infrastructure/csrc/channel/grpc_channel/build/grpcchannel.grpc.pb.h"

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
                        const size_t &nbytes)\
                        :name_(name), size_(size), nbytes_(nbytes) {}
  Status RecvArrayData(ServerContext* context, const GrpcMetaData* request,
                      DataReply* reply) override;
  bool Writable();
  GrpcMetaDataPtr getdata();

 private:
  GrpcMetaDataPtr data_;
  std::string name_;
  size_t size_;
  size_t nbytes_;
};
using ServerImplPtr = std::shared_ptr<GrpcChannelServerImpl>;

class GrpcRecvPort final : public AbstractRecvPort{
 public:
  ~GrpcRecvPort();
  GrpcRecvPort(const std::string& name,
                 const size_t &size,
                 const size_t &nbytes,
                 const std::string &url);
  void StartSever();
  void Start();
  MetaDataPtr Recv();
  void Join();
  bool Probe();
  MetaDataPtr Peek();
  void GrpcMetaData2MetaData(MetaDataPtr metadata, GrpcMetaDataPtr grpcdata);
 private:
  ServerBuilder builder;
  std::atomic_bool done_;
  std::unique_ptr<Server> server;
  ServerImplPtr serviceptr;
  ThreadPtr grpcthreadptr = nullptr;
  std::string url_;
};
using GrpcRecvPortPtr = std::shared_ptr<GrpcRecvPort>;

class GrpcSendPort final : public AbstractSendPort{
 public:
  GrpcSendPort(const std::string &name,
            const size_t &size,
            const size_t &nbytes, const std::string& url)
            :AbstractSendPort(name, size, nbytes), done_(false), url_(url) {}

  void Start();
  void MetaData2GrpcMetaData(MetaDataPtr metadata, GrpcMetaDataPtr grpcdata);
  void Send(MetaDataPtr metadata);
  bool Probe();
  void Join();

 private:
  int idx_ = 0;
  std::shared_ptr<Channel> channel;
  std::atomic_bool done_;
  std::unique_ptr<GrpcChannelServer::Stub> stub_;
  ThreadPtr ack_callback_thread_ = nullptr;
  std::string url_;
};
using GrpcSendPortPtr = std::shared_ptr<GrpcSendPort>;
}   // namespace message_infrastructure
