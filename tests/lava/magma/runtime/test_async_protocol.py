import unittest

from lava.magma.core.decorator import implements, requires
from lava.magma.core.model.py.model import PyAsyncProcessModel
from lava.magma.core.model.py.type import LavaPyType
from lava.magma.core.process.process import AbstractProcess
from lava.magma.core.process.variable import Var
from lava.magma.core.resources import CPU
from lava.magma.core.run_conditions import RunContinuous
from lava.magma.core.run_configs import RunConfig
from lava.magma.core.sync.domain import SyncDomain
from lava.magma.core.sync.protocols.async_protocol import AsyncProtocol


class SimpleProcess(AbstractProcess):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        shape = kwargs["shape"]
        self.u = Var(shape=shape, init=0)
        self.v = Var(shape=shape, init=0)


class SimpleRunConfig(RunConfig):
    def __init__(self, **kwargs):
        sync_domains = kwargs.pop("sync_domains")
        super().__init__(custom_sync_domains=sync_domains)
        self.model = None
        if "model" in kwargs:
            self.model = kwargs.pop("model")

    def select(self, process, proc_models):
        if self.model is not None:
            if self.model == "sub" and isinstance(process, SimpleProcess):
                return proc_models[1]

        return proc_models[0]


@implements(proc=SimpleProcess, protocol=AsyncProtocol)
@requires(CPU)
class SimpleProcessModel(PyAsyncProcessModel):
    u = LavaPyType(int, int)
    v = LavaPyType(int, int)

    def run_async(self):
        while True:
            self.u = self.u + 10
            self.v = self.v + 1000
            if self.check_for_stop_cmd():
                return


class TestProcess(unittest.TestCase):
    def test_async_process_model(self):
        """
        Verifies the working of Asynchronous Process
        """
        process = SimpleProcess(shape=(2, 2))
        simple_sync_domain = SyncDomain("simple", AsyncProtocol(), [process])
        run_config = SimpleRunConfig(sync_domains=[simple_sync_domain])
        process.run(condition=RunContinuous(), run_cfg=run_config)
        process.stop()

    def test_async_process_model_pause(self):
        """
        Verifies the working of Asynchronous Process, pause should have no
        effect
        """
        process = SimpleProcess(shape=(2, 2))
        simple_sync_domain = SyncDomain("simple", AsyncProtocol(), [process])
        run_config = SimpleRunConfig(sync_domains=[simple_sync_domain])
        process.run(condition=RunContinuous(), run_cfg=run_config)
        process.pause()
        process.stop()


if __name__ == "__main__":
    unittest.main()
