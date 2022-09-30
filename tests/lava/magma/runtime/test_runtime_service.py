import unittest

from message_infrastructure import (
    ChannelBackend,
    Channel,
    ActorStatus
)
from lava.magma.core.decorator import implements
from lava.magma.core.model.py.model import AbstractPyProcessModel
from lava.magma.core.process.process import AbstractProcess
from lava.magma.core.sync.protocol import AbstractSyncProtocol
from lava.magma.runtime.runtime_services.runtime_service import \
    PyRuntimeService


def create_channel(name: str):
    return Channel(ChannelBackend.SHMEMCHANNEL,
                   8,
                   4,
                   name + "src",
                   name + "dst")


class MockActorInterface:

    def set_stop_fn(self, fn):
        pass

    def get_status(self):
        return ActorStatus.StatusRunning

    def status_stopped(self):
        pass

    def status_terminated(self):
        pass


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
        runtime_to_service = create_channel(name="runtime_to_service")
        service_to_runtime = create_channel(name="service_to_runtime")
        service_to_process = [create_channel(name="service_to_process")]
        process_to_service = [create_channel(name="process_to_service")]

        pm.service_to_process = service_to_process[0].dst_port
        pm.process_to_service = process_to_service[0].src_port
        pm.py_ports = []
        pm.start(MockActorInterface())
        rs.runtime_to_service = runtime_to_service.src_port
        rs.service_to_runtime = service_to_runtime.dst_port
        rs.service_to_process = [service_to_process[0].src_port]
        rs.process_to_service = [process_to_service[0].dst_port]
        rs.start(MockActorInterface())
        rs.join()
        pm.join()


if __name__ == '__main__':
    unittest.main()
