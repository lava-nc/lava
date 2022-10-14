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

py::array_t<int32_t> Data() {
  py::array_t<int32_t> data = py::array_t<int32_t>({1, 2, 3, 4});
  return data;
}

void SendProc(AbstractSendPortPtr send_port, MetaDataPtr data, AbstractActor* actor_ptr) {
  AbstractActor::StopFn stop_fn;
  actor_ptr->SetStopFn(stop_fn);
  std::cout << "Here I am" << std::endl;

  // Sends data
  send_port->Start();
  send_port->Send(data);
  send_port->Join();

  actor_ptr->SetStatus(ActorStatus::StatusStopped);
  std::cout << "Status STOPPED" << std::endl;
}

void RecvProc(AbstractRecvPortPtr recv_port, AbstractActor* actor_ptr) {
  AbstractActor::StopFn stop_fn;
  actor_ptr->SetStopFn(stop_fn);
  std::cout << "Here I am (RECV)" << std::endl;

  // Returns received data
  recv_port->Start();
  auto recv_data = recv_port->Recv();
  recv_port->Join();

  actor_ptr->SetStatus(ActorStatus::StatusStopped);

  // TODO: Check data value
  recv_data->mdata;
  std::cout << "Mdata" << std::endl;
  // if (recv_data != Data()) {
  //   std::cout << "Received Data is incorrect" << std::endl;
  // }
}

TEST(TestSharedMemory, SharedMemSendReceive) {
  // Creates a pair of send and receive ports
  // TODO: Define success criteria

  // Create Shared Memory Channel
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

  AbstractSendPortPtr send_port = shmem_channel.GetSendPort();
  AbstractRecvPortPtr recv_port = shmem_channel.GetRecvPort();

  MultiProcessing mp;
  Builder *builder = new Builder();

  AbstractActor::TargetFn send_target_fn;
  AbstractActor::TargetFn recv_target_fn;

  // TODO: convert data into python to pass into function
  MetaDataPtr data;
  auto send_bound_fn = std::bind(&SendProc, send_port, data, std::placeholders::_1);
  send_target_fn = send_bound_fn;

  auto recv_bound_fn = std::bind(&RecvProc, recv_port, std::placeholders::_1);
  recv_target_fn = recv_bound_fn;

  mp.BuildActor(send_target_fn);
  mp.BuildActor(recv_target_fn);

  sleep(2);

  // Stop any currently running actors
  mp.Stop(true);
}

TEST(TestSharedMemory, SharedMemSingleProcess){
  // ChannelProxy ShmemChannel;

  // SendPortProxyPtr SendPort = ShmemChannel.GetSendPort();
  // RecvPortProxyPtr RecvPort = ShmemChannel.GetRecvPort();

  // SendPort.Start();
  // RecvPort.Start();
  // auto data = DataInteger();
  // SendPort.Send(data);
  // auto received_data = RecvPort.Recv();
  // EXPECT_EQ(data, received_data)
}
