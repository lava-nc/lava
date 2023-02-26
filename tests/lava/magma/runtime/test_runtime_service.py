# Copyright (C) 2021-23 Intel Corporation
# SPDX-License-Identifier: BSD-3-Clause
# See: https://spdx.org/licenses/

import unittest

import numpy as np
from lava.magma.compiler.channels.pypychannel import PyPyChannel
from lava.magma.core.decorator import implements
from lava.magma.core.model.py.model import AbstractPyProcessModel
from lava.magma.core.process.process import AbstractProcess
from lava.magma.core.sync.protocol import AbstractSyncProtocol
from lava.magma.runtime.runtime_services.runtime_service import PyRuntimeService
from lava.magma.runtime.message_infrastructure.shared_memory_manager import (
    SharedMemoryManager,
)


class MockInterface:
    def __init__(self, smm):
        self.smm = smm


def create_channel(smm: SharedMemoryManager, name: str):
    mock = MockInterface(smm=smm)
    return PyPyChannel(
        mock,
        name,
        name,
        (1,),
        np.int32,
        8,
    )


class SimpleSyncProtocol(AbstractSyncProtocol):
    pass


class SimpleProcess(AbstractProcess):
    pass


@implements(proc=SimpleProcess, protocol=SimpleSyncProtocol)
class SimpleProcessModel(AbstractPyProcessModel):
    def run(self):
        pass

    def add_ports_for_polling(self):
        pass


class SimplePyRuntimeService(PyRuntimeService):
    def run(self):
        pass


class TestRuntimeService(unittest.TestCase):
    def test_runtime_service_construction(self):
        sp = SimpleSyncProtocol()
        rs = SimplePyRuntimeService(protocol=sp)
        self.assertEqual(rs.protocol, sp)
        self.assertEqual(rs.service_to_runtime, None)
        self.assertEqual(rs.service_to_process, [])
        self.assertEqual(rs.runtime_to_service, None)
        self.assertEqual(rs.process_to_service, [])

    def test_runtime_service_start_run(self):
        pm = SimpleProcessModel(proc_params={})
        sp = SimpleSyncProtocol()
        rs = SimplePyRuntimeService(protocol=sp)
        smm = SharedMemoryManager()
        smm.start()
        runtime_to_service = create_channel(smm, name="runtime_to_service")
        service_to_runtime = create_channel(smm, name="service_to_runtime")
        service_to_process = [create_channel(smm, name="service_to_process")]
        process_to_service = [create_channel(smm, name="process_to_service")]
        runtime_to_service.dst_port.start()
        service_to_runtime.src_port.start()

        pm.service_to_process = service_to_process[0].dst_port
        pm.process_to_service = process_to_service[0].src_port
        pm.py_ports = []
        pm.start()
        rs.runtime_to_service = runtime_to_service.src_port
        rs.service_to_runtime = service_to_runtime.dst_port
        rs.service_to_process = [service_to_process[0].src_port]
        rs.process_to_service = [process_to_service[0].dst_port]
        rs.join()
        pm.join()
        smm.shutdown()


if __name__ == '__main__':
    unittest.main()
