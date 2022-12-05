# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: BSD-3-Clause
# See: https://spdx.org/licenses/

from ctypes import CDLL, RTLD_GLOBAL
from os import path

lib_name = 'libmessage_infrastructure.so'
here = path.abspath(__file__)
lib_path = path.join(path.dirname(here), lib_name)
if (path.exists(lib_path)):
    CDLL(lib_path)
else:
    print("Warn: No library file")

from message_infrastructure.MessageInfrastructurePywrapper import (
    CppMultiProcessing,
    ProcessType,
    Actor,
    ActorStatus,
    ActorCmd,
    RecvPort,
    AbstractTransferPort,
    support_grpc_channel,
    support_fastdds_channel,
    support_cyclonedds_channel)
from message_infrastructure.MessageInfrastructurePywrapper \
    import ChannelType as ChannelBackend

from .ports import SendPort, Channel, getTempSendPort, getTempRecvPort

ChannelQueueSize = 1
SyncChannelBytes = 128
SupportGRPCChannel = support_grpc_channel()
SupportFastDDSChannel = support_fastdds_channel()
SupportCycloneDDSChannel = support_cyclonedds_channel()

if SupportGRPCChannel:
    from .ports import GetRPCChannel
if SupportFastDDSChannel or SupportCycloneDDSChannel:
    from .ports import GetDDSChannel
    from message_infrastructure.MessageInfrastructurePywrapper import (
        DDSTransportType,
        DDSBackendType)
