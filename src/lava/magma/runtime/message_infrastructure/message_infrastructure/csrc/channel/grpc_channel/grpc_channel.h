#include <numpy/arrayobject.h>
#include <Python.h>
#include "port_proxy.h"
#include "message_infrastructure_logging.h"
#include "abstract_port.h"
#include <atomic>
#include "utils.h"
#include <thread>
#include<iostream>
#include <memory>
#include <string>
#include <grpcpp/ext/proto_server_reflection_plugin.h>
#include <grpcpp/grpcpp.h>
#include <grpcpp/health_check_service_interface.h>
#include "grpcchannel.grpc.pb.h"
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
std::string target_str = "localhost:50051";

class GrpcChannelServerImpl final: public GrpcChannelServer::Service{
public:
  GrpcChannelServerImpl(const std::string& name,
                          const size_t &size,
                          const size_t &nbytes);
  Status RecvArrayData(ServerContext* context, const GrpcMetaData* request,
							DataReply* reply) override;
  void Push(GrpcMetaData* src)
  int AvailableCount();
  bool Empty();
  GrpcMetaData Pop(bool block);
  GrpcMetaData Front();
  void Stop();
  bool Probe();

private:
  std::string name_;
  size_t size_;
  size_t nbytes_;
	std::vector<GrpcMetaData> array_;
	std::atomic<uint32_t> read_index_;
	std::atomic<uint32_t> write_index_;
  std::atomic_bool done_;
}
using ServerImplPtr = std::shared_ptr<GrpcChannelServerImpl>;


class GrpcRecvPort{
  public:
  GrpcRecvPort(const std::string& name,
                 const size_t &size,
                 const size_t &nbytes);
    void Start();
    void Startthread();
    MetaDataPtr Recv();//GrpcRecvPort will use this func to call the service's member function pop().
    MetaDataPtr Peek();
    void Stop();
    void Join();
    bool Probe();
    void GrpcMetaData2GrpcMeta(MetaDataPtr metadata, GrpcMetaDataPtr grpcdata);
  private:
    std::atomic_bool done_;
    std::unique_ptr<Server> server;
    ServerImplPtr serviceptr;
    ThreadPtr grpcthreadptr = nullptr;
}



class GrpcSendPort final : public AbstractSendPort{
  public:
    GrpcSendPort(const std::string &name,
              const size_t &size,
              const size_t &nbytes);

    void Start();
    void Send(MetaDataPtr metadata);
    void Join();
    bool Probe();
    void MetaData2GrpcMetaData(MetaDataPtr metadata, GrpcMetaDataPtr grpcdata);

  private:
    int idx_ = 0;
    std::shared_ptr<Channel> channel;
    std::atomic_bool done_;
    std::unique_ptr<GrpcChannelServer::Stub> stub_;
    ThreadPtr ack_callback_thread_ = nullptr;

}

