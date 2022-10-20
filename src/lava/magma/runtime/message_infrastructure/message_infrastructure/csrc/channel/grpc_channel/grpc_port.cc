// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: BSD-3-Clause
// See: https://spdx.org/licenses/
#include <grpcpp/ext/proto_server_reflection_plugin.h>
#include <grpcpp/grpcpp.h>
#include <grpcpp/health_check_service_interface.h>
#include <message_infrastructure/csrc/channel/grpc_channel/grpc_port.h>
#include <message_infrastructure/csrc/core/utils.h>
#include <message_infrastructure/csrc/core/message_infrastructure_logging.h>
#include <iostream>
#include <memory>
#include <string>
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
using GrpcMetaDataPtr = std::shared_ptr<GrpcMetaData>;

Status GrpcChannelServerImpl::RecvArrayData(ServerContext* context, \
                      const GrpcMetaData *request, DataReply* reply) {
  bool ret = false;
  if (Writable()) {
    data_ = std::make_shared<GrpcMetaData>(*request);
    ret = true;
  }
  reply->set_ack(ret);
  return Status::OK;
}
bool GrpcChannelServerImpl::Writable() {
  return data_== nullptr;
}
GrpcMetaDataPtr GrpcChannelServerImpl::getdata() {
  return data_;
}

GrpcRecvPort::~GrpcRecvPort() {
  free(data_->mdata);
  server->Shutdown();
}
GrpcRecvPort::GrpcRecvPort(const std::string& name,
                 const size_t &size,
                 const size_t &nbytes, const std::string &url)
  :AbstractRecvPort(name, size, nbytes), url_(url) {
  serviceptr = std::make_shared<GrpcChannelServerImpl>(name_, size_, nbytes_);
  data_ = std::make_shared<MetaData>();
  data_->mdata = malloc(nbytes_);
}
void GrpcRecvPort::StartSever() {
  grpc::EnableDefaultHealthCheckService(true);
  grpc::reflection::InitProtoReflectionServerBuilderPlugin();
  builder.AddListeningPort(url_, grpc::InsecureServerCredentials());
  builder.RegisterService(serviceptr.get());
  server = builder.BuildAndStart();
  server->Wait();
}
void GrpcRecvPort::Start() {
  grpcthreadptr = std::make_shared<std::thread>(\
      &message_infrastructure::GrpcRecvPort::StartSever, this);
}
MetaDataPtr GrpcRecvPort::Recv() {
  while (serviceptr->Writable()) {
    helper::Sleep();
    if (done_)
      return NULL;
  }
  memset(data_->mdata, 0, nbytes_);
  GrpcMetaData2MetaData(data_, serviceptr->getdata());
  serviceptr->getdata() = nullptr;
  return data_;
}
void GrpcRecvPort::Join() {
  if (!done_) {
    done_ = true;
    server->Shutdown();
    grpcthreadptr->join();
  }
}
bool GrpcRecvPort::Probe() {
  return false;
}
MetaDataPtr GrpcRecvPort::Peek() {
  return Recv();
}
void GrpcRecvPort::GrpcMetaData2MetaData(\
    MetaDataPtr metadata, GrpcMetaDataPtr grpcdata) {
  metadata->nd = grpcdata->nd();
  metadata->type = grpcdata->type();
  metadata->elsize = grpcdata->elsize();
  metadata->total_size = grpcdata->total_size();
  for (int i = 0; i < MAX_ARRAY_DIMS; i++) {
    metadata->dims[i] = grpcdata->dims(i);
    metadata->strides[i] = grpcdata->strides(i);
  }
  std::memcpy(metadata->mdata, grpcdata->value().c_str(), nbytes_);
}

void GrpcSendPort::Start() {
  channel = grpc::CreateChannel(url_, grpc::InsecureChannelCredentials());
  stub_ = GrpcChannelServer::NewStub(channel);
}

void GrpcSendPort::MetaData2GrpcMetaData(\
                        MetaDataPtr metadata, \
                        GrpcMetaDataPtr grpcdata) {
  grpcdata->set_nd(metadata->nd);
  grpcdata->set_type(metadata->type);
  grpcdata->set_elsize(metadata->elsize);
  grpcdata->set_total_size(metadata->total_size);
  char* data = (char*)metadata->mdata;
  for (int i = 0; i < MAX_ARRAY_DIMS; i++) {
    grpcdata->add_dims(metadata->dims[i]);
    grpcdata->add_strides(metadata->strides[i]);
  }
  grpcdata->set_value(data, nbytes_);
}
void GrpcSendPort::Send(MetaDataPtr metadata) {
  GrpcMetaDataPtr request = std::make_shared<GrpcMetaData>();
  MetaData2GrpcMetaData(metadata, request);
  DataReply reply;
  ClientContext context;
  Status status = stub_->RecvArrayData(&context, *request, &reply);

  // if (status.ok()&&reply.ack()) {
  //  std::cout<<"Successed,Recv ack is "<<reply.ack()<<std::endl;
  //} else {
  //  std::cout<<"Recv ack is "<<reply.ack()<<"ERROR! Send fail!"<<std::endl;
  //}
}
bool GrpcSendPort::Probe() {
  return false;
}
void GrpcSendPort::Join() {
  done_ = true;
}
}  // namespace message_infrastructure
