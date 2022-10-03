// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: BSD-3-Clause
// See: https://spdx.org/licenses/

#include <iostream>

#include <gtest/gtest.h>
#include <multiprocessing.h>
#include <abstract_actor.h>
#include <shmem_channel.h>
#include <channel_proxy.h>

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
namespace py = pybind11;

using namespace message_infrastructure;

class Builder {
  public:
    void Build() {};
};

void TargetFunction(Builder builder, AbstractActor* actor_ptr) {
  std::cout << "Target Function running..." << std::endl;
  actor_ptr->SetStatus(ActorStatus::StatusStopped);
  builder.Build();
}

void ActorStop(std::string actor_name) {
  std::cout << actor_name << " STOP" << std::endl;
}

int* DataInteger() {
  int data = 42;
  int *data_ptr;
  data_ptr = &data;
  return data_ptr;
}

void SendProc(ShmemSendPort send_port, MetaDataPtr data) {
  // Sends data
  send_port.Start();
  send_port.Send(data);
  send_port.Join();
}

auto RecvProc(ShmemRecvPort recv_port) {
  // Returns received data
  recv_port.Start();
  auto recv_data = recv_port.Recv();
  recv_port.Join();

  return recv_data;
}

TEST(TestSharedMemory, SharedMemSendReceive) {
  // Creates a pair of send and receive ports
  // TODO: Define success criteria
  MultiProcessing mp;
  Builder *builder = new Builder();

  AbstractActor::TargetFn target_fn;

  auto bound_fn = std::bind(&TargetFunction, (*builder),  std::placeholders::_1);
  target_fn = bound_fn;

  AbstractActor::StopFn stop_fn;
  AbstractActor::StopFn stop_fn2;

  auto actor_stop_fn = std::bind(&ActorStop, std::placeholders::_1);
  stop_fn = actor_stop_fn("Send");
  
  AbstractActor* send_actor_ptr;
  send_actor_ptr->SetStopFn(stop_fn);
  send_actor_ptr->SetStatus(ActorStatus::StatusStopped);

  AbstractActor* recv_actor_ptr;
  recv_actor_ptr->SetStopFn(stop_fn2);
  recv_actor_ptr->SetStatus(ActorStatus::StatusStopped);

  int size = 1;
  int nbytes = sizeof(int);
  std::string name = "test_shmem_channel";
  std::string src_name = "Source1";
  std::string dst_name = "Dest1";
  auto shmem_channel = ShmemChannel(
    src_name,
    dst_name,
    size,
    nbytes);

  // AbstractSendPortPtr send_port = *shmem_channel.GetSendPort();
  // AbstractRecvPortPtr recv_port = *shmem_channel->GetRecvPort();

  // auto send_port_fn = std::bind(&SendProc, send_port);
  // auto send_port_fn = std::bind(&SendProc, (*builder_send), SendPort);
  // auto recv_port_fn = std::bind(&RecvProc, (*builder_recv), RecvPort);
  // mp.BuildActor(send_port_fn);
  // mp.BuildActor(recv_port_fn);
  
  // Stop any currently running actors
  // mp.Stop(true);
}

TEST(TestSharedMemory, SharedMemSingleProcess){
  ChannelProxy ShmemChannel;

  SendPortProxyPtr SendPort = ShmemChannel.GetSendPort();
  RecvPortProxyPtr RecvPort = ShmemChannel.GetRecvPort();

  SendPort.Start();
  RecvPort.Start();
  auto data = DataInteger();
  SendPort.Send(data);
  auto received_data = RecvPort.Recv();
  EXPECT_EQ(data, received_data)
}
