# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: BSD-3-Clause
# See: https://spdx.org/licenses/

from ctypes import CDLL, RTLD_GLOBAL
from os import path

lib_name = 'libmessage_infrastructure.so'
here = path.abspath(__file__)
lib_path = path.join(path.dirname(here), lib_name)
print(lib_path)
if (path.exists(lib_path)):
    CDLL(lib_path)
else:
    print("Warn: No library file")

from .MessageInfrastructurePywrapper import CppMultiProcessing
from .MessageInfrastructurePywrapper import ProcessType
from .MessageInfrastructurePywrapper import Actor
from .MessageInfrastructurePywrapper import ActorStatus
from .MessageInfrastructurePywrapper import ActorCmd
from .MessageInfrastructurePywrapper import ChannelType as ChannelBackend
from .MessageInfrastructurePywrapper import RecvPort
from .MessageInfrastructurePywrapper import AbstractTransferPort
from .MessageInfrastructurePywrapper import support_grpc_channel
from .MessageInfrastructurePywrapper import support_fastdds_channel
from .MessageInfrastructurePywrapper import support_cyclonedds_channel
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
    from .MessageInfrastructurePywrapper import DDSTransportType
    from .MessageInfrastructurePywrapper import DDSBackendType
