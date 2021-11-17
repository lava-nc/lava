import numpy as np
from lava.magma.core.model.py.ports import PyRefPort, PyVarPort
from lava.proc.monitor.process import Monitor
from lava.magma.core.sync.protocols.loihi_protocol import LoihiProtocol
from lava.magma.core.model.py.model import PyLoihiProcessModel
from lava.magma.core.model.py.type import LavaPyType
from lava.magma.core.decorator import implements, requires
from lava.magma.core.resources import CPU

# A minimal PyProcModel implementing P1
@implements(proc=Monitor, protocol=LoihiProtocol)
@requires(CPU)
class PyMonitorModel(PyLoihiProcessModel):
    var_read: np.ndarray = LavaPyType(np.ndarray, np.int32, precision=24)
    ref: PyRefPort = LavaPyType(PyRefPort.VEC_DENSE, int)

    def pre_guard(self):
        return True

    def run_pre_mgmt(self):
        if self.current_ts > 0:
            self.var_read[ self.current_ts-1] = self.ref.read()
            # self.proc_params