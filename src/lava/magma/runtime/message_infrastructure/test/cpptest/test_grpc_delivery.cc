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

GrpcMetaDataPtr MetaData2GrpcMetaData(MetaDataPtr metadata) {
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

void grpc_target_fn1(
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
      LAVA_DUMP(LOG_UTTEST, "grpc actor1 waitting\n");
      MetaDataPtr data = from_mp->Recv();
      LAVA_DUMP(LOG_UTTEST, "grpc actor1 recviced\n");
      (*reinterpret_cast<int64_t*>(data->mdata))++;
      to_a2->Send(MetaData2GrpcMetaData(data));
      free(reinterpret_cast<char*>(data->mdata));
      data = from_a2->Recv();
      (*reinterpret_cast<int64_t*>(data->mdata))++;
      to_mp->Send(MetaData2GrpcMetaData(data));
      free(reinterpret_cast<char*>(data->mdata));
    }
    from_mp->Join();
    to_mp->Join();
    to_a2->Join();
    from_a2->Join();
    while (!actor_ptr->GetStatus()) {
      helper::Sleep();
    }
  }

void grpc_target_fn2(
  int loop,
  AbstractChannelPtr a1_to_a2,
  AbstractChannelPtr a2_to_a1,
  AbstractActor* actor_ptr) {
    actor_ptr->SetStopFn(stop_fn);
    auto to_a1 = a2_to_a1->GetSendPort();
    auto from_a1 = a1_to_a2->GetRecvPort();
    from_a1->Start();
    to_a1->Start();
    LAVA_DUMP(LOG_UTTEST, "grpc actor2, loop: %d\n", loop);
    while ((loop--)&&!actor_ptr->GetStatus()) {
      LAVA_DUMP(LOG_UTTEST, "grpc actor2 waitting\n");
      MetaDataPtr data = from_a1->Recv();
      LAVA_DUMP(LOG_UTTEST, "grpc actor2 recviced\n");
      (*reinterpret_cast<int64_t*>(data->mdata))++;
      to_a1->Send(MetaData2GrpcMetaData(data));
      free(reinterpret_cast<char*>(data->mdata));
    }
    from_a1->Join();
    to_a1->Join();
    while (!actor_ptr->GetStatus()) {
      helper::Sleep();
    }
  }

TEST(TestGRPCChannel, GRPCLoop) {
  MultiProcessing mp;
  int loop = 1000;
  AbstractChannelPtr mp_to_a1 = GetChannelFactory().GetDefRPCChannel(
    "mp_to_a1", "mp_to_a1", 6);
  AbstractChannelPtr a1_to_mp = GetChannelFactory().GetDefRPCChannel(
    "a1_to_mp", "a1_to_mp", 6);
  AbstractChannelPtr a1_to_a2 = GetChannelFactory().GetDefRPCChannel(
    "a1_to_a2", "a1_to_a2", 6);
  AbstractChannelPtr a2_to_a1 = GetChannelFactory().GetDefRPCChannel(
    "a2_to_a1", "a2_to_a1", 6);
  auto target_fn_a1 = std::bind(&grpc_target_fn1,
                                loop,
                                mp_to_a1,
                                a1_to_mp,
                                a1_to_a2,
                                a2_to_a1,
                                std::placeholders::_1);
  auto target_fn_a2 = std::bind(&grpc_target_fn2,
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
  int64_t array_[10000] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 0};
  std::fill(array_ + 10, array_ + 10000, 1);

  MetaDataPtr metadata = std::make_shared<MetaData>();
  int64_t* array = reinterpret_cast<int64_t*>(array_);
  int64_t dims[] = {10000, 0, 0, 0, 0};
  int64_t nd = 1;

  GetMetadata(metadata, array, nd, METADATA_TYPES::LONG, dims);
  int expect_result = 1 + loop * 3;
  const clock_t start_time = std::clock();
  while (loop--) {
    LAVA_DUMP(LOG_UTTEST, "wait for response, remain loop: %d\n", loop);
    to_a1->Send(MetaData2GrpcMetaData(metadata));
    metadata = from_a1->Recv();
    LAVA_DUMP(LOG_UTTEST, "metadata:\n");
    LAVA_DUMP(LOG_UTTEST, "nd: %ld\n", metadata->nd);
    LAVA_DUMP(LOG_UTTEST, "type: %ld\n", metadata->type);
    LAVA_DUMP(LOG_UTTEST, "elsize: %ld\n", metadata->elsize);
    LAVA_DUMP(LOG_UTTEST, "total_size: %ld\n", metadata->total_size);
    LAVA_DUMP(LOG_UTTEST, "dims: {%ld, %ld, %ld, %ld, %ld}\n",
              metadata->dims[0], metadata->dims[1], metadata->dims[2],
              metadata->dims[3], metadata->dims[4]);
    LAVA_DUMP(LOG_UTTEST, "strides: {%ld, %ld, %ld, %ld, %ld}\n",
              metadata->strides[0], metadata->strides[1], metadata->strides[2],
              metadata->strides[3], metadata->strides[4]);
    LAVA_DUMP(LOG_UTTEST, "grpc mdata: %p, grpc *mdata: %ld\n", metadata->mdata,
              *reinterpret_cast<int64_t*>(metadata->mdata));
    free(reinterpret_cast<char*>(metadata->mdata));
  }
  const clock_t end_time = std::clock();
  to_a1->Join();
  from_a1->Join();
  int64_t result = *reinterpret_cast<int64_t*>(metadata->mdata);
  LAVA_DUMP(LOG_UTTEST, "grpc result =%ld\n", result);
  mp.Stop(true);
  if (result != expect_result) {
    LAVA_DUMP(LOG_UTTEST, "expect_result: %d\n", expect_result);
    printf("result: %ld\n", result);
    LAVA_DUMP(LOG_UTTEST, "result: %ld\n", result);
    LAVA_LOG_ERR("result != expect_result\n");
    throw;
  }
  std::printf("grpc cpp loop timedelta: %f\n",
           ((end_time - start_time)/static_cast<double>(CLOCKS_PER_SEC)));
  LAVA_DUMP(LOG_UTTEST, "exit\n");
}

}  // namespace message_infrastructure
