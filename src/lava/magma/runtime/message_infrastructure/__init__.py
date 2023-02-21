# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: BSD-3-Clause
# See: https://spdx.org/licenses/

import os
import platform


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
        create_channel,
        CspSelector,
        PyPyChannel)
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
        if os.path.exists(lib_path):
            CDLL(lib_path, mode=RTLD_GLOBAL)
        else:
            print("Warn: No library file")
        extra_lib_folder = os.path.join(os.path.dirname(here), "install", "lib")
        if os.path.exists(extra_lib_folder):
            extra_libs = os.listdir(extra_lib_folder)
            for lib in extra_libs:
                if '.so' in lib and ('idl' not in lib):
                    lib_file = os.path.join(extra_lib_folder, lib)
                    CDLL(lib_file, mode=RTLD_GLOBAL)

    load_library()

    from lava.magma.runtime.message_infrastructure. \
        MessageInfrastructurePywrapper import (  # noqa
            RecvPort,  # noqa
            AbstractTransferPort,  # noqa
            support_grpc_channel,
            support_fastdds_channel,
            support_cyclonedds_channel)

    ChannelQueueSize = 1
    SyncChannelBytes = 128

    from .ports import (  # noqa
        SendPort,  # noqa
        Channel,  # noqa
        Selector,  # noqa
        getTempSendPort,  # noqa
        getTempRecvPort,  # noqa
        create_channel)  # noqa
    from .pypychannel import (  # noqa
        CspSelector,  # noqa
        PyPyChannel)  # noqa
    SupportGRPCChannel = support_grpc_channel()
    SupportFastDDSChannel = support_fastdds_channel()
    SupportCycloneDDSChannel = support_cyclonedds_channel()
    SupportTempChannel = True

    if SupportGRPCChannel:
        from .ports import GetRPCChannel  # noqa
    if SupportFastDDSChannel or SupportCycloneDDSChannel:
        from .ports import GetDDSChannel  # noqa
        from lava.magma.runtime.message_infrastructure. \
            MessageInfrastructurePywrapper import (
                DDSTransportType,  # noqa
                DDSBackendType)  # noqa
