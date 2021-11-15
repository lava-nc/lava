import numpy as np
import unittest

from lava.magma.core.model.py.model import PyLoihiProcessModel
from lava.magma.core.model.py.ports import PyRefPort, PyVarPort
from lava.magma.core.model.py.type import LavaPyType
from lava.magma.core.process.ports.ports import RefPort, VarPort
from lava.magma.core.process.process import AbstractProcess
from lava.magma.core.process.variable import Var
from lava.magma.core.resources import CPU
from lava.magma.core.sync.domain import SyncDomain
from lava.magma.core.sync.protocols.loihi_protocol import LoihiProtocol
from lava.magma.core.run_configs import RunConfig
from lava.magma.core.run_conditions import RunSteps
from lava.magma.core.decorator import implements, requires


class P1(AbstractProcess):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.ref = RefPort(shape=(1,))
        self.v = Var(shape=(1,), init=17)
        self.var_port = VarPort(self.v)


# A minimal PyProcModel implementing P1
@implements(proc=P1, protocol=LoihiProtocol)
@requires(CPU)
class PyProcModel1(PyLoihiProcessModel):
    v: np.ndarray = LavaPyType(np.ndarray, np.int32)

    def pre_guard(self):
        return True

    def run_pre_mgmt(self):
        if self.current_ts > 1:
             self.v = self.current_ts


class Monitors(unittest.TestCase):
    def test_setup_monitor(self):
        some_proc = P1()
        monitor = Monitor.probe(var=P1.v)

        simple_sync_domain = SyncDomain("simple", LoihiProtocol(),
                                        [some_proc, monitor])
        # Dynamically create refPorts for each var to be monitored rather
        # than hard-coding the refports


if __name__ == '__main__':
    unittest.main()
