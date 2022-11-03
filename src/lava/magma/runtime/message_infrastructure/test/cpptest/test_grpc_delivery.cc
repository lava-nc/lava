// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: BSD-3-Clause
// See: https://spdx.org/licenses/

#include <message_infrastructure/csrc/core/channel_factory.h>
#include <message_infrastructure/csrc/core/multiprocessing.h>
#include <message_infrastructure/csrc/channel/grpc/grpc_channel.h>
#include <message_infrastructure/csrc/core/message_infrastructure_logging.h>
#include <message_infrastructure/csrc/core/utils.h>
#include <gtest/gtest.h>
#include <iostream>

namespace message_infrastructure {

static void stop_fn() {
  // exit(0);
}

void target_fn1(
  int loop,
  AbstractChannelPtr mp_to_a1,
  AbstractChannelPtr a1_to_mp,
  AbstractChannelPtr a1_to_a2,
  AbstractChannelPtr a2_to_a1,
  AbstractActor* actor_ptr) {
    actor_ptr->SetStopFn(stop_fn);
    auto from_mp = mp_to_a1->GetRecvPort();
    auto to_mp = a1_to_mp->GetSendPort();
    auto to_a2   = a1_to_a2->GetSendPort();
    auto from_a2 = a2_to_a1->GetRecvPort();
    from_mp->Start();
    to_mp->Start();
    to_a2->Start();
    from_a2->Start();
    LAVA_DUMP(1, "grpc actor1, loop: %d\n", loop);
    while ((loop--)&&!actor_ptr->GetStatus()) {
      LAVA_DUMP(1, "grpc actor1 waitting\n");
      MetaDataPtr data = from_mp->Recv();
      LAVA_DUMP(1, "grpc actor1 recviced\n");
      (*reinterpret_cast<int64_t*>(data->mdata))++;
      to_a2->Send(data);
      free(reinterpret_cast<char*>(data->mdata));
      data = from_a2->Recv();
      (*reinterpret_cast<int64_t*>(data->mdata))++;
      to_mp->Send(data);
      free(reinterpret_cast<char*>(data->mdata));
    }
    from_mp->Join();
    to_mp->Join();
    to_a2->Join();
    from_a2->Join();
    // actor_ptr->SetStatus(ActorStatus::StatusStopped);
    while (!actor_ptr->GetStatus()) {
      helper::Sleep();
    }
  }

void target_fn2(
  int loop,
  AbstractChannelPtr a1_to_a2,
  AbstractChannelPtr a2_to_a1,
  AbstractActor* actor_ptr) {
    actor_ptr->SetStopFn(stop_fn);
    auto to_a1 = a2_to_a1->GetSendPort();
    auto from_a1 = a1_to_a2->GetRecvPort();
    from_a1->Start();
    to_a1->Start();
    LAVA_DUMP(1, "grpc actor2, loop: %d\n", loop);
    while ((loop--)&&!actor_ptr->GetStatus()) {
      LAVA_DUMP(1, "grpc actor2 waitting\n");
      MetaDataPtr data = from_a1->Recv();
      LAVA_DUMP(1, "grpc actor2 recviced\n");
      (*reinterpret_cast<int64_t*>(data->mdata))++;
      to_a1->Send(data);
      free(reinterpret_cast<char*>(data->mdata));
    }
    from_a1->Join();
    to_a1->Join();
    actor_ptr->SetStatus(ActorStatus::StatusStopped);
    while (!actor_ptr->GetStatus()) {
      helper::Sleep();
    }
  }

TEST(TestGRPCChannel, GRPCLoop) {
  MetaDataPtr metadata = std::make_shared<MetaData>();
  metadata->nd = 1;
  metadata->type = 7;
  metadata->elsize = 8;
  metadata->total_size = 1;
  metadata->dims[0] = 1;
  metadata->strides[0] = 1;
  metadata->mdata =
    reinterpret_cast<char*>
    (malloc(sizeof(int64_t)));
  *reinterpret_cast<int64_t*>(metadata->mdata) = 1;
  MultiProcessing mp;
  int loop = 10000;
  AbstractChannelPtr mp_to_a1 = GetChannelFactory().GetRPCChannel(
  "127.13.2.11", 8001, "mp_to_a1", "mp_to_a1", 6);
  AbstractChannelPtr a1_to_mp = GetChannelFactory().GetRPCChannel(
  "127.13.2.12", 8005, "a1_to_mp", "a1_to_mp", 6);
  AbstractChannelPtr a1_to_a2 = GetChannelFactory().GetRPCChannel(
  "127.13.2.13", 8003, "a1_to_a2", "a1_to_a2", 6);
  AbstractChannelPtr a2_to_a1 = GetChannelFactory().GetRPCChannel(
  "127.13.2.14", 8005, "a2_to_a1", "a2_to_a1", 6);
  auto target_fn_a1 = std::bind(&target_fn1,
                                loop,
                                mp_to_a1,
                                a1_to_mp,
                                a1_to_a2,
                                a2_to_a1,
                                std::placeholders::_1);
  auto target_fn_a2 = std::bind(&target_fn2,
                                loop,
                                a1_to_a2,
                                a2_to_a1,
                                std::placeholders::_1);
  int actor1 = mp.BuildActor(target_fn_a1);
  int actor2 = mp.BuildActor(target_fn_a2);
  auto to_a1 = mp_to_a1->GetSendPort();
  to_a1->Start();
  auto from_a1 = a1_to_mp->GetRecvPort();
  from_a1->Start();
  MetaDataPtr mptr;
  int expect_result = 1+loop*3;
  const clock_t start_time = std::clock();
  while (loop--) {
    to_a1->Send(metadata);
    LAVA_DUMP(1, "wait for response, remain loop: %d\n", loop);
    mptr = from_a1->Recv();
    LAVA_DUMP(1, "metadata:\n");
    LAVA_DUMP(1, "nd: %ld\n", mptr->nd);
    LAVA_DUMP(1, "type: %ld\n", mptr->type);
    LAVA_DUMP(1, "elsize: %ld\n", mptr->elsize);
    LAVA_DUMP(1, "total_size: %ld\n", mptr->total_size);
    LAVA_DUMP(1, "dims: {%ld, %ld, %ld, %ld, %ld}\n",
    mptr->dims[0], mptr->dims[1], mptr->dims[2], mptr->dims[3], mptr->dims[4]);
    LAVA_DUMP(1, "strides: {%ld, %ld, %ld, %ld, %ld}\n",
    mptr->strides[0], mptr->strides[1], mptr->strides[2], mptr->strides[3],
    mptr->strides[4]);
    LAVA_DUMP(1, "mdata: %p, *mdata: %ld\n", mptr->mdata,
              *reinterpret_cast<int64_t*>(mptr->mdata));
    free(reinterpret_cast<char*>(metadata->mdata));
    metadata = mptr;
  }

  const clock_t end_time = std::clock();
  to_a1->Join();
  from_a1->Join();
  int64_t result = *reinterpret_cast<int64_t*>(metadata->mdata);
  free(reinterpret_cast<char*>(metadata->mdata));
  mp.Stop(true);
  if (result != expect_result) {
    LAVA_DUMP(1, "expect_result: %d\n", expect_result);
    LAVA_DUMP(1, "result: %ld\n", result);
    LAVA_LOG_ERR("result != expect_result");
    throw;
  }
  LAVA_DUMP(1, "cpp loop timedelta: %ld\n", (end_time - start_time));
  LAVA_DUMP(1, "exit\n");
}

}  // namespace message_infrastructure
