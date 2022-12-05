// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: BSD-3-Clause
// See: https://spdx.org/licenses/

#ifndef CHANNEL_GRPC_GRPC_PORT_H_
#define CHANNEL_GRPC_GRPC_PORT_H_

#include <grpcpp/grpcpp.h>
#include <grpcpp/ext/proto_server_reflection_plugin.h>
#include <grpcpp/health_check_service_interface.h>

#include <core/utils.h>
#include <core/common.h>
#include <core/abstract_port.h>
#include <channel/grpc/grpcchannel.grpc.pb.h>

#include <atomic>
#include <thread> //NOLINT
#include <memory>
#include <string>

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

using GrpcMetaDataPtr = std::shared_ptr<GrpcMetaData>;

template class RecvQueue<GrpcMetaDataPtr>;

inline GrpcMetaDataPtr MetaData2GrpcMetaData(MetaDataPtr metadata) {
  GrpcMetaDataPtr grpcdata = std::make_shared<GrpcMetaData>();
  grpcdata->set_nd(metadata->nd);
  grpcdata->set_type(metadata->type);
  grpcdata->set_elsize(metadata->elsize);
  grpcdata->set_total_size(metadata->total_size);
  // char* data = reinterpret_cast<char*>(metadata->mdata);
  for (int i = 0; i < metadata->nd; i++) {
    grpcdata->add_dims(metadata->dims[i]);
    grpcdata->add_strides(metadata->strides[i]);
  }
  grpcdata->set_value(metadata->mdata, metadata->elsize*metadata->total_size);
  return grpcdata;
}

class GrpcChannelServerImpl final : public GrpcChannelServer::Service {
 public:
  GrpcChannelServerImpl(const std::string& name,
                        const size_t &size);
  ~GrpcChannelServerImpl() override {}
  Status RecvArrayData(ServerContext* context,
                       const GrpcMetaData* request,
                       DataReply* reply) override;
  GrpcMetaDataPtr Pop(bool block);
  GrpcMetaDataPtr Front();
  bool Probe();
  void Stop();

 private:
  std::shared_ptr<RecvQueue<GrpcMetaDataPtr>> recv_queue_;
  std::string name_;
  size_t size_;
  std::atomic_bool done_;
};

using ServerImplPtr = std::shared_ptr<GrpcChannelServerImpl>;

class GrpcRecvPort final : public AbstractRecvPort {
 public:
  GrpcRecvPort() = delete;
  GrpcRecvPort(const std::string& name,
               const size_t &size,
               const std::string& url);
  ~GrpcRecvPort() override {}
  void Start();
  MetaDataPtr Recv();
  MetaDataPtr Peek();
  void Join();
  bool Probe();

 private:
  ServerBuilder builder_;
  std::atomic_bool done_;
  std::unique_ptr<Server> server_;
  ServerImplPtr service_ptr_;
  std::string url_;
  std::string name_;
  size_t size_;
};

// Users should be allowed to copy port objects.
// Use std::shared_ptr.
using GrpcRecvPortPtr = std::shared_ptr<GrpcRecvPort>;

class GrpcSendPort final : public AbstractSendPort {
 public:
  GrpcSendPort() = delete;
  GrpcSendPort(const std::string &name,
               const size_t &size,
               const std::string& url)
  :name_(name), size_(size), done_(false), url_(url) {}
  ~GrpcSendPort() override {}

  void Start();
  void Send(DataPtr grpcdata);
  void Join();
  bool Probe();

 private:
  std::shared_ptr<Channel> channel_;
  std::atomic_bool done_;
  std::unique_ptr<GrpcChannelServer::Stub> stub_;
  std::string url_;
  std::string name_;
  size_t size_;
};

// Users should be allowed to copy port objects.
// Use std::shared_ptr.
using GrpcSendPortPtr = std::shared_ptr<GrpcSendPort>;

}  // namespace message_infrastructure

#endif  // CHANNEL_GRPC_GRPC_PORT_H_
