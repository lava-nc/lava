// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: BSD-3-Clause
// See: https://spdx.org/licenses/

#include <message_infrastructure/csrc/core/channel_factory.h>
#include <message_infrastructure/csrc/core/multiprocessing.h>
#include <message_infrastructure/csrc/channel/dds/dds_channel.h>
#include <message_infrastructure/csrc/core/message_infrastructure_logging.h>
#include <gtest/gtest.h>

namespace message_infrastructure {

void stop_fn() {
  // exit(0);
}

void target_fn_a1_bound(
  int loop,
  AbstractChannelPtr mp_to_a1,
  AbstractChannelPtr a1_to_mp,
  AbstractChannelPtr a1_to_a2,
  AbstractChannelPtr a2_to_a1,
  AbstractActor* actor_ptr) {
    actor_ptr->SetStopFn(stop_fn);
    auto from_mp = mp_to_a1->GetRecvPort();
    from_mp->Start();
    auto to_mp   = a1_to_mp->GetSendPort();
    to_mp->Start();
    auto to_a2   = a1_to_a2->GetSendPort();
    to_a2->Start();
    auto from_a2 = a2_to_a1->GetRecvPort();
    from_a2->Start();
    LAVA_DUMP(1, "actor1, loop: %d\n", loop);
    while ((loop--)&&!actor_ptr->GetStatus()) {
      LAVA_DUMP(1, "actor1 waitting\n");
      MetaDataPtr data = from_mp->Recv();
      LAVA_DUMP(1, "actor1 recviced\n");
      (*reinterpret_cast<int64_t*>(data->mdata))++;
      to_a2->Send(data);
      // free(reinterpret_cast<char*>(data->mdata)-sizeof(MetaData));
      data = from_a2->Recv();
      (*reinterpret_cast<int64_t*>(data->mdata))++;
      to_mp->Send(data);
      // free(reinterpret_cast<char*>(data->mdata)-sizeof(MetaData));
    }
    from_mp->Join();
    from_a2->Join();
    while (!actor_ptr->GetStatus()) {
      helper::Sleep();
    }
  }

void target_fn_a2_bound(
  int loop,
  AbstractChannelPtr a1_to_a2,
  AbstractChannelPtr a2_to_a1,
  AbstractActor* actor_ptr) {
    actor_ptr->SetStopFn(stop_fn);
    auto from_a1 = a1_to_a2->GetRecvPort();
    from_a1->Start();
    auto to_a1   = a2_to_a1->GetSendPort();
    to_a1->Start();
    LAVA_DUMP(1, "actor2, loop: %d\n", loop);
    while ((loop--)&&!actor_ptr->GetStatus()) {
      LAVA_DUMP(1, "actor2 waitting\n");
      MetaDataPtr data = from_a1->Recv();
      LAVA_DUMP(1, "actor2 recviced\n");
      (*reinterpret_cast<int64_t*>(data->mdata))++;
      to_a1->Send(data);
      // free(reinterpret_cast<char*>(data->mdata)-sizeof(MetaData));
    }
    from_a1->Join();
    while (!actor_ptr->GetStatus()) {
      helper::Sleep();
    }
  }

TEST(TestDDSDelivery, DDSLoop) {
  GTEST_SKIP();
  MultiProcessing mp;
  int loop = 1;
  AbstractChannelPtr mp_to_a1 = GetChannelFactory()
																.GetDefDDSChannel(5,
                                       						8,
																									"mp_to_a1",
																									DDSSHM,
																									FASTDDSBackend);
  AbstractChannelPtr a1_to_mp = GetChannelFactory()
																.GetDefDDSChannel(5,
                                       						8,
																									"a1_to_mp",
																									DDSSHM,
																									FASTDDSBackend);
  AbstractChannelPtr a1_to_a2 = GetChannelFactory()
																.GetDefDDSChannel(5,
                                       						8,
																									"a1_to_a2",
																									DDSSHM,
																									FASTDDSBackend);
  AbstractChannelPtr a2_to_a1 = GetChannelFactory()
																.GetDefDDSChannel(5,
                                       						8,
																									"a2_to_a1",
																									DDSSHM,
																									FASTDDSBackend);

  auto target_fn_a1 = std::bind(&target_fn_a1_bound, loop,
                                mp_to_a1, a1_to_mp, a1_to_a2,
                                a2_to_a1, std::placeholders::_1);
  auto target_fn_a2 = std::bind(&target_fn_a2_bound, loop, a1_to_a2,
                                a2_to_a1, std::placeholders::_1);

  int actor1 = mp.BuildActor(target_fn_a1);
  int actor2 = mp.BuildActor(target_fn_a2);

  auto to_a1   = mp_to_a1->GetSendPort();
  to_a1->Start();
  auto from_a1 = a1_to_mp->GetRecvPort();
  from_a1->Start();

  MetaDataPtr metadata = std::make_shared<MetaData>();
  metadata->nd = 1;
  metadata->type = 7;
  metadata->elsize = 8;
  metadata->total_size = 1;
  metadata->dims[0] = 1;
  metadata->strides[0] = 1;
  metadata->mdata =
    (reinterpret_cast<char*>
    (malloc(sizeof(int64_t)+sizeof(MetaData)))+sizeof(MetaData));
  *reinterpret_cast<int64_t*>(metadata->mdata) = 1;

  MetaDataPtr mptr;
  LAVA_DUMP(1, "main process loop: %d\n", loop);
  const clock_t start_time = std::clock();
  while (loop--) {
    to_a1->Send(metadata);
    LAVA_DUMP(1, "wait for response, remain loop: %d\n", loop);
    mptr = from_a1->Recv();

    // to_a1->Join();
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
    // free(reinterpret_cast<char*>(metadata->mdata)-sizeof(MetaData));
    metadata = mptr;
  }
  const clock_t end_time = std::clock();
  // free(reinterpret_cast<char*>(mptr->mdata)-sizeof(MetaData));
  from_a1->Join();
  mp.Stop(true);
  LAVA_DUMP(1, "cpp loop timedelta: %ld", (end_time - start_time));
  LAVA_DUMP(1, "exit\n");
}

TEST(TestDDSSingleProcess, DDS1Process) {
	LAVA_DUMP(1, "TestDDSSingleProcess starts.\n");
  AbstractChannelPtr dds_channel = GetChannelFactory()
																	.GetDefDDSChannel(5,
																										8,
																										"test_DDSChannel",
																										DDSSHM,
																										FASTDDSBackend);

  auto send_port = dds_channel->GetSendPort();
  send_port->Start();
  auto recv_port = dds_channel->GetRecvPort();
  recv_port->Start();

  MetaDataPtr metadata = std::make_shared<MetaData>();
  metadata->nd = 1;
  metadata->type = 7;
  metadata->elsize = 8;
  metadata->total_size = 1;
  metadata->dims[0] = 1;
  metadata->strides[0] = 1;
  metadata->mdata =
    (reinterpret_cast<char*>
    (malloc(sizeof(int64_t)+sizeof(MetaData)))+sizeof(MetaData));
  *reinterpret_cast<int64_t*>(metadata->mdata) = 1;

  MetaDataPtr mptr;
  const clock_t start_time = std::clock();
	int loop = 10;
	int i = 0;
  while (loop--) {
		if(!(loop % 1000))
			printf("At iteration : %d * 1000\n", i++);
    send_port->Send(metadata);
    LAVA_DUMP(1, "wait for response, remain loop: %d\n", loop);
    mptr = recv_port->Recv();

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
		EXPECT_EQ(*reinterpret_cast<int64_t*>(mptr->mdata), *reinterpret_cast<int64_t*>(metadata->mdata));
	  (*reinterpret_cast<int64_t*>(metadata->mdata))++;
    free(reinterpret_cast<char*>(mptr->mdata));
  }
  const clock_t end_time = std::clock();
  recv_port->Join();
  send_port->Join();
  LAVA_DUMP(1, "cpp loop timedelta: %ld", (end_time - start_time));
  LAVA_DUMP(1, "exit\n");
}
}  // namespace message_infrastructure