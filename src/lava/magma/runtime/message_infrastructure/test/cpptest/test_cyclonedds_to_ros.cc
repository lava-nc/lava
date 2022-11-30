// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: BSD-3-Clause
// See: https://spdx.org/licenses/

#include <message_infrastructure/csrc/channel/dds/cyclone_dds.h>
#include <message_infrastructure/csrc/core/channel_factory.h>
#include <message_infrastructure/csrc/channel/dds/dds_channel.h>

using namespace message_infrastructure;  // NOLINT

#define LOOP_NUM 100

int main() {
  auto dds_channel = GetChannelFactory()
    .GetDDSChannel("rt/dds_topic",
                   DDSTransportType::DDSUDPv4,
                   DDSBackendType::CycloneDDSBackend,
                   10);
  auto dds_send = dds_channel->GetSendPort();
  int loop = LOOP_NUM;

  dds_send->Start();
  MetaDataPtr metadata = std::make_shared<MetaData>();
  metadata->nd = 1;
  metadata->type = 7;
  metadata->elsize = 1;
  metadata->total_size = 1;
  metadata->dims[0] = 1;
  metadata->strides[0] = 1;
  metadata->mdata = reinterpret_cast<char*>(malloc(sizeof(char)));
  while (loop--) {
    *reinterpret_cast<char*>(metadata->mdata) = loop % 255;
    dds_send->Send(metadata);
    printf("DDS send : '%d'\n", loop);
    sleep(1);
  }
  dds_send->Join();
  return 0;
}
