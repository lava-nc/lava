import unittest
from multiprocessing.managers import SharedMemoryManager

import numpy as np

from lava.magma.compiler.channels.pypychannel import PyPyChannel
from lava.magma.core.decorator import implements
from lava.magma.core.model.py.model import AbstractPyProcessModel
from lava.magma.core.process.process import AbstractProcess
from lava.magma.core.sync.protocol import AbstractSyncProtocol
from lava.magma.runtime.runtime_service import PyRuntimeService


def create_channel(messaging_infrastructure: SharedMemoryManager, name: str):
    return PyPyChannel(
        messaging_infrastructure,
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


class SimplePyRuntimeService(PyRuntimeService):
    def run(self):
        pass


class TestRuntimeService(unittest.TestCase):
    def test_runtime_service_construction(self):
        sp = SimpleSyncProtocol()
        rs = SimplePyRuntimeService(protocol=sp)
        self.assertEqual(rs.protocol, sp)
        self.assertEqual(rs.service_to_runtime_ack, None)
        self.assertEqual(rs.service_to_runtime_data, None)
        self.assertEqual(rs.service_to_process_cmd, [])
        self.assertEqual(rs.service_to_process_req, [])
        self.assertEqual(rs.service_to_process_data, [])
        self.assertEqual(rs.runtime_to_service_cmd, None)
        self.assertEqual(rs.runtime_to_service_req, None)
        self.assertEqual(rs.runtime_to_service_data, None)
        self.assertEqual(rs.process_to_service_ack, [])
        self.assertEqual(rs.process_to_service_data, [])

    def test_runtime_service_start_run(self):
        pm = SimpleProcessModel()
        sp = SimpleSyncProtocol()
        rs = SimplePyRuntimeService(protocol=sp)
        smm = SharedMemoryManager()
        smm.start()
        runtime_to_service_cmd = create_channel(smm,
                                                name="runtime_to_service_cmd")
        service_to_runtime_ack = create_channel(smm,
                                                name="service_to_runtime_ack")
        runtime_to_service_req = create_channel(smm,
                                                name="runtime_to_service_req")
        service_to_runtime_data = create_channel(smm,
                                                 name="service_to_runtime_data")
        runtime_to_service_data = create_channel(smm,
                                                 name="runtime_to_service_data")
        service_to_process_cmd = [
            create_channel(smm, name="service_to_process_cmd")]
        process_to_service_ack = [
            create_channel(smm, name="process_to_service_ack")]
        service_to_process_req = [
            create_channel(smm, name="service_to_process_req")]
        process_to_service_data = [
            create_channel(smm, name="process_to_service_data")]
        service_to_process_data = [
            create_channel(smm, name="service_to_process_data")]
        runtime_to_service_cmd.dst_port.start()
        service_to_runtime_ack.src_port.start()
        runtime_to_service_req.src_port.start()
        service_to_runtime_data.dst_port.start()
        runtime_to_service_data.src_port.start()

        pm.service_to_process_cmd = service_to_process_cmd[0].dst_port
        pm.service_to_process_req = service_to_process_req[0].dst_port
        pm.service_to_process_data = service_to_process_data[0].dst_port
        pm.process_to_service_ack = process_to_service_ack[0].src_port
        pm.process_to_service_data = process_to_service_data[0].src_port
        pm.py_ports = []
        pm.start()
        rs.runtime_to_service_cmd = runtime_to_service_cmd.src_port
        rs.service_to_runtime_ack = service_to_runtime_ack.dst_port
        rs.service_to_runtime_data = service_to_runtime_data.dst_port
        rs.runtime_to_service_req = runtime_to_service_req.src_port
        rs.runtime_to_service_data = runtime_to_service_data.dst_port
        rs.service_to_process_cmd = [service_to_process_cmd[0].src_port]
        rs.process_to_service_ack = [process_to_service_ack[0].dst_port]
        rs.service_to_process_req = [service_to_process_req[0].src_port]
        rs.process_to_service_data = [process_to_service_data[0].dst_port]
        rs.service_to_process_data = [service_to_process_data[0].src_port]
        rs.join()
        pm.join()
        smm.shutdown()


if __name__ == '__main__':
    unittest.main()
