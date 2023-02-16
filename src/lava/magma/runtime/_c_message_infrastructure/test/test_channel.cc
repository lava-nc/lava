// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: BSD-3-Clause
// See: https://spdx.org/licenses/

#include <core/multiprocessing.h>
#include <core/abstract_actor.h>
#include <channel/shmem/shmem_channel.h>
#include <gtest/gtest.h>
#include <iostream>

namespace message_infrastructure {

class Builder {
 public:
    void Build() {}
};

MetaDataPtr ExpectData() {
  auto metadata = std::make_shared<MetaData>();
  int32_t data[5] = {1, 3, 5, 7, 9};
  int32_t *data_ptr = data;
  metadata->mdata = reinterpret_cast<void*>(data);
  // metadata->mdata = (void*)data;
  return metadata;
}

void SendProc(AbstractSendPortPtr send_port,
              MetaDataPtr data) {
  AbstractActor::StopFn stop_fn;
  std::cout << "Here I am" << std::endl;

  // Sends data
  send_port->Start();
  send_port->Send(data);
  send_port->Join();
  std::cout << "Status STOPPED" << std::endl;
}

void RecvProc(AbstractRecvPortPtr recv_port) {
  std::cout << "Here I am (RECV)" << std::endl;

  // Returns received data
  recv_port->Start();
  auto recv_data = recv_port->Recv();
  recv_port->Join();

  // if (recv_data != Data()) {
  //   std::cout << "Received Data is incorrect" << std::endl;
  // }
}

TEST(TestSharedMemory, SharedMemSendReceive) {
  // Creates a pair of send and receive ports
  // Expects that data sent is the same as data received
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

  auto data = ExpectData();
  auto send_bound_fn = std::bind(&SendProc,
                                 send_port,
                                 data);
  send_target_fn = send_bound_fn;

  auto recv_bound_fn = std::bind(&RecvProc,
                                 recv_port);
  recv_target_fn = recv_bound_fn;

  mp.BuildActor(send_target_fn);
  mp.BuildActor(recv_target_fn);

  sleep(2);

  // Stop any currently running actors
  mp.Stop();
  mp.Cleanup(true);
}

TEST(TestSharedMemory, SharedMemSingleProcess) {
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

}  // namespace message_infrastructure
