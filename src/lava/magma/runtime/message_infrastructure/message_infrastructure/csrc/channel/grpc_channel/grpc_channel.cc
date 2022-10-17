#include <iostream>
#include <memory>
#include <string>
#include <message_infrastructure/csrc/core/utils.h>
#include <message_infrastructure/csrc/core/message_infrastructure_logging.h>

#include <grpcpp/ext/proto_server_reflection_plugin.h>
#include <grpcpp/grpcpp.h>
#include <grpcpp/health_check_service_interface.h>
#include "grpcchannel.grpc.pb.h"
#include <message_infrastructure/csrc/channel/grpc_channel/grpc_channel.h>

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
std::string target_str = "localhost:50051";


GrpcChannelServerImpl::GrpcChannelServerImpl(const std::string& name,
												const size_t &size,
												const size_t &nbytes)
:name_(name), size_(size), nbytes_(nbytes), read_index_(0), write_index_(0), done_(false)
{
array_.reserve(size_);
}

//Server function tmplement
Status GrpcChannelServerImpl::RecvArrayData(ServerContext* context, const GrpcMetaData* request,
						DataReply* reply) override {
	bool ret = false;
	if (AvailableCount() > 0) {
		GrpcChannelServerImpl:::Push(request);
		ret=true;
	}
	reply->set_ack(ret);
	return Status::OK;
}

void GrpcChannelServerImpl::Push(GrpcMetaData* src) {
  auto const curr_write_index = write_index_.load(std::memory_order_relaxed);
  auto next_write_index = curr_write_index + 1;
  if (next_write_index == size_) {
      next_write_index = 0;
  }
  if (next_write_index != read_index_.load(std::memory_order_acquire)) {
    array_[curr_write_index] = *src;
    write_index_.store(next_write_index, std::memory_order_release);
  }
}

int GrpcChannelServerImpl::AvailableCount() {
	auto const curr_read_index = read_index_.load(std::memory_order_acquire);
	auto const curr_write_index = write_index_.load(std::memory_order_acquire);
	if (curr_read_index == curr_write_index) {
	return size_;
	}
	if (curr_write_index > curr_read_index) {
			return size_ - curr_write_index + curr_read_index - 1;
	}
	return curr_read_index - curr_write_index - 1;
}

bool GrpcChannelServerImpl::Empty() {
	auto const curr_read_index = read_index_.load(std::memory_order_acquire);
	auto const curr_write_index = write_index_.load(std::memory_order_acquire);
	return curr_read_index == curr_write_index;
}

GrpcMetaData GrpcChannelServerImpl::Pop(bool block) {
	while(block && Empty()) {
	helper::Sleep();
	if(done_)
	return NULL;
	}
	auto const curr_read_index = read_index_.load(std::memory_order_relaxed);
	assert(curr_read_index != write_index_.load(std::memory_order_acquire));
	GrpcMetaData data_ = array_[curr_read_index];
	auto next_read_index = curr_read_index + 1;
	if(next_read_index == size_) {
			next_read_index = 0;
	}
	read_index_.store(next_read_index, std::memory_order_release);
	return data_;
}

GrpcMetaData GrpcChannelServerImpl::Front() {
  while(Empty()) {
    helper::Sleep();
    if(done_)
      return NULL;
  }
  auto curr_read_index = read_index_.load(std::memory_order_acquire);
  GrpcMetaData data_ = array_[curr_read_index];
  return data_;
}

void GrpcChannelServerImpl::Stop() {
  done_ = true;
}

bool GrpcChannelServerImpl::Probe() {
  return !Empty();
}


GrpcRecvPort::GrpcRecvPort(const std::string& name,
                 const size_t &size,
                 const size_t &nbytes):AbstractRecvPort(name, size, nbytes),done_(false){
                    serviceptr = std::make_shared<GrpcChannelServerImpl>(name_, size_, nbytes_);
                 }
void GrpcRecvPort::Start(){
    grpc::EnableDefaultHealthCheckService(true);
    grpc::reflection::InitProtoReflectionServerBuilderPlugin();
    ServerBuilder builder;
    builder.AddListeningPort(server_address, grpc::InsecureServerCredentials());
    builder.RegisterService(serviceptr.get());
    std::unique_ptr<Server> server(builder.BuildAndStart());
    server->wait()
}

void GrpcRecvPort::Startthread(){
    grpcthreadptr = std::make_shared<std::thread>(&GrpcRecvPort::Start(), this);
}

MetaDataPtr GrpcRecvPort::Recv(){
    GrpcMetaDataPtr recvdata = std::make_shared<GrpcMetaData>(serviceptr->Pop());
    MetaDataPtr data_ = std::make_share<MetaData>();
    GrpcMetaData2MetaData(data_,recvdata);
    return data_;
}

MetaDataPtr GrpcRecvPort::Peek(){
    GrpcMetaDataPtr peekdata = std::make_shared<GrpcMetaData>(serviceptr->Front());
    MetaDataPtr data_ = std::make_share<MetaData>();
    GrpcMetaData2MetaData(data_,peekdata);
    return data_;
}
void GrpcRecvPort::Stop(){
    serviceptr->Stop();
}
void GrpcRecvPort::Join(){
    if (!done_) {
        done_ = true;
        grpcthreadptr->join();
        grpcthreadptr->Stop();
    }
}
void GrpcRecvPort::Prob(){
    serviceptr->Probe();
}
void GrpcRecvPort::GrpcMetaData2MetaData(MetaDataPtr metadata, GrpcMetaDataPtr grpcdata){
    metadata->nd = grpcdata->nd;
    metadata->type = grpcdata->type;
    metadata->elsize = grpcdata->elsize;
    metadata->total_size = grpcdata->total_szie;
    int* data = (int*)metadata->mdata;
    for(int i=0; i<MAX_ARRAY_DIMS;i++){
            metadata->dims[i] = grpcdata->dims(i);
            metadata->strides[i] = grpcdata->strides(i);
    }
    for(int i =0; i< grpcdata->total_size; i++){
            *data++=grpcdata->value(i);
    }
}



GrpcSendPort::GrpcSendPort(const std::string &name,
                SharedMemoryPtr shm,
                const size_t &size,
                const size_t &nbytes)
  : AbstractSendPort(name, size, nbytes),done_(false)
{}

void GrpcSendPort::Start(){
        channel = grpc::CreateChannel(target_str, grpc::InsecureChannelCredentials());
        stub_ = GrpcChannelServer::NewStub(channel);
}

void GrpcSendPort::MetaData2GrpcMetaData(MetaDataPtr metadata, GrpcMetaDataPtr grpcdata){
    grpcdata->nd = metadata->nd;
    grpcdata->type = metadata->type;
    grpcdata->elsize = metadata->elsize;
    grpcdata->total_szie = metadata->total_size;
    int* data = (int*)metadata->mdata;
    for(int i=0; i<MAX_ARRAY_DIMS;i++){
            grpcdata->add_dims(metadata->dims[i]);
            grpcdata->add_strides(metadata->strides[i]);
    }
    for(int i =0; i< grpcdata->total_size; i++){
            grpcdata->add_value(*data++);
    }
}

void GrpcSendPort::send(MetaDataPtr metadata){
    GrpcMetaDataPtr request = std::make_shared<GrpcMetaData>();
    MetaData2GrpcMeta(metadata, request);
    DataReply reply;
    ClientContext context;
    Status status = stub_->RecvArrayData(&context, *request, &reply);
    if (status.ok()&&reply.ack()) {
        std::cout<<"Recv ack is "<<reply.ack()<<std::endl;
    } else {
        std::cout<<"ERROR! Send fail!"<<std::endl;
    }
}
bool GrpcSendPort::Probe() {
  return false;
}
void GrpcSendPort::Join() {
  done_ = true;
}



int main(){

}