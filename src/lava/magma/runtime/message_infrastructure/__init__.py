# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: BSD-3-Clause
# See: https://spdx.org/licenses/

import os
import platform
from glob import glob
import warnings


def _get_pure_py() -> bool:
    pure_py_env = os.getenv("LAVA_PURE_PYTHON", 0)
    system_name = platform.system().lower()
    if system_name != "linux":
        return True
    return int(pure_py_env) > 0


PURE_PYTHON_VERSION = _get_pure_py()

if PURE_PYTHON_VERSION:
    from abc import ABC, abstractmethod
    import multiprocessing as mp

    if platform.system() != 'Windows':
        mp.set_start_method('fork')

    class Channel(ABC):
        @property
        @abstractmethod
        def src_port(self):
            pass

        @property
        @abstractmethod
        def dst_port(self):
            pass

    from .py_ports import AbstractTransferPort
    from .pypychannel import (
        SendPort,
        RecvPort,
        create_channel)
    from .pypychannel import CspSelector as Selector
    SupportGRPCChannel = False
    SupportFastDDSChannel = False
    SupportCycloneDDSChannel = False
    SupportTempChannel = False

    def getTempSendPort(addr_path: str):
        return None

    def getTempRecvPort():
        return None, None

else:
    from ctypes import CDLL, RTLD_GLOBAL

    def load_library():
        lib_name = 'libmessage_infrastructure.so'
        here = os.path.abspath(__file__)
        lib_path = os.path.join(os.path.dirname(here), lib_name)

        if not os.path.exists(lib_path):
            warnings.warn("No library file")
            return

        extra_lib_folder = os.path.join(os.path.dirname(here), "install", "lib")
        dds_libs = ["libfastcdr.so.*",
                    "libfastrtps.so.*",
                    "libddsc.so.*",
                    "libddscxx.so.*"]
        if os.path.exists(extra_lib_folder):
            for lib in dds_libs:
                files = glob(os.path.join(extra_lib_folder, lib))
                for file in files:
                    CDLL(file, mode=RTLD_GLOBAL)

        CDLL(lib_path, mode=RTLD_GLOBAL)

    load_library()

    from lava.magma.runtime.message_infrastructure. \
        MessageInfrastructurePywrapper import (  # noqa  # nosec
            RecvPort,  # noqa  # nosec
            AbstractTransferPort,  # noqa  # nosec
            support_grpc_channel,
            support_fastdds_channel,
            support_cyclonedds_channel)

    ChannelQueueSize = 128
    SyncChannelBytes = 128

    from .ports import (  # noqa  # nosec
        SendPort,  # noqa  # nosec
        Channel,  # noqa  # nosec
        Selector,  # noqa  # nosec
        getTempSendPort,  # noqa  # nosec
        getTempRecvPort,  # noqa  # nosec
        create_channel)  # noqa  # nosec
    SupportGRPCChannel = support_grpc_channel()
    SupportFastDDSChannel = support_fastdds_channel()
    SupportCycloneDDSChannel = support_cyclonedds_channel()
    SupportTempChannel = True

    if SupportGRPCChannel:
        from .ports import GetRPCChannel # noqa # nosec
    if SupportFastDDSChannel or SupportCycloneDDSChannel:
        from .ports import GetDDSChannel # noqa # nosec
        from lava.magma.runtime.message_infrastructure. \
            MessageInfrastructurePywrapper import (
                DDSTransportType,  # noqa  # nosec
                DDSBackendType)  # noqa  # nosec
