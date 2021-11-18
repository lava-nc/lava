import numpy as np
from lava.magma.core.model.py.ports import PyRefPort, PyVarPort, PyInPort
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
    # var_read_Process_1_v: np.ndarray = LavaPyType(np.ndarray,
    #                                               np.int32,
    #                                               precision=24)
    out_read_Process_1_s_out: np.ndarray = LavaPyType(np.ndarray,
                                                  np.int32,
                                                  precision=24)
    # ref_port_Process_1_v: PyRefPort = LavaPyType(PyRefPort.VEC_DENSE, int)
    in_port_Process_1_s_out: PyInPort = LavaPyType(PyInPort.VEC_DENSE, int)


    def pre_guard(self):
        return True

    def run_pre_mgmt(self):
        if self.current_ts > 0:
            num_ref_port_probes = len(self.proc_params["RefPorts"])
            for i in range(num_ref_port_probes):
                ref_port_name = self.proc_params["RefPorts"][i]
                var_read_name = self.proc_params["VarsRead"][i]
                getattr(self, var_read_name)[self.current_ts - 1] = \
                    getattr(self, ref_port_name).read()

    def run_spk(self):
        if self.current_ts > 0:
            num_in_port_probes = len(self.proc_params["InPorts"])
            for i in range(num_in_port_probes):
                in_port_name = self.proc_params["InPorts"][i]
                out_read_name = self.proc_params["OutsRead"][i]
                if self.in_port_Process_1_s_out.csp_ports[0].probe():
                    # a = getattr(self, in_port_name).recv()
                    getattr(self, out_read_name)[:,self.current_ts - 1] = \
                        np.squeeze(np.array(getattr(self, in_port_name).recv()))
                else:
                    getattr(self, out_read_name)[self.current_ts - 1] = 0