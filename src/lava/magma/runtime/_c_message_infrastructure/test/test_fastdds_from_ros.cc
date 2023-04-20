// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: BSD-3-Clause
// See: https://spdx.org/licenses/

#include <channel/dds/fast_dds.h>
#include <core/channel_factory.h>
#include <channel/dds/dds_channel.h>

using namespace message_infrastructure;  // NOLINT

#define LOOP_NUM 100

int main() {
  auto dds_channel = GetChannelFactory()
    .GetDDSChannel("test_channel_src",
                   "test_channel_dst",
                   "rt/dds_topic",
                   10,
                   DEFAULT_NBYTES,
                   DDSTransportType::DDSUDPv4,
                   DDSBackendType::FASTDDSBackend);
  auto dds_recv = dds_channel->GetRecvPort();
  int loop = LOOP_NUM;

  dds_recv->Start();
  while (loop--) {
    MetaDataPtr res = dds_recv->Recv();
    printf("DDS recv : '%d'\n", *reinterpret_cast<char*>(res->mdata));
  }
  dds_recv->Join();
  return 0;
}
