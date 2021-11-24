# Copyright (C) 2021 Intel Corporation
# SPDX-License-Identifier: BSD-3-Clause
# See: https://spdx.org/licenses/

import numpy as np
from lava.magma.core.model.py.ports import PyRefPort, PyInPort
from lava.proc.monitor.process import Monitor
from lava.magma.core.sync.protocols.loihi_protocol import LoihiProtocol
from lava.magma.core.model.py.model import PyLoihiProcessModel
from lava.magma.core.model.py.type import LavaPyType
from lava.magma.core.decorator import implements, requires, tag
from lava.magma.core.resources import CPU


@implements(proc=Monitor, protocol=LoihiProtocol)
@requires(CPU)
@tag('floating_pt')
class PyMonitorModel(PyLoihiProcessModel):
    """
    This process model contains prototypical Ports and Vars to have
    one-to-one correspondes with Monitor process.
    """
    var_read_0: np.ndarray = LavaPyType(np.ndarray,
                                        np.float,
                                        precision=24)
    out_read_0: np.ndarray = LavaPyType(np.ndarray,
                                        np.float,
                                        precision=24)
    ref_port_0: PyRefPort = LavaPyType(PyRefPort.VEC_DENSE, int)
    in_port_0: PyInPort = LavaPyType(PyInPort.VEC_DENSE, int)

    def pre_guard(self):
        return True

    def run_pre_mgmt(self):
        """
        During this phase, RefPorts of Monitor process collects data from
        monitored Vars
        """
        # Check if this Monitor Process instance has been assigned to monitor
        # any Var by checking proc_params["n_ref_ports"], if so loop over
        # those RefPorts; readout their values to correspoinding data-stroing
        # Var
        for i in range(self.proc_params["n_ref_ports"]):
            ref_port_name = self.proc_params["RefPorts"][i]
            var_read_name = self.proc_params["VarsData1"][i]
            getattr(self, var_read_name)[:, self.current_ts - 1] = \
                np.squeeze(np.array(getattr(self, ref_port_name).read()))

    def run_spk(self):
        """
        During this phase, InPorts of Monitor process collects data from
        monitored OutPorts
        """

        # Check if this Monitor Process instance has been assigned to monitor
        # any OutPort by checking proc_params["n_in_ports"], if so loop over
        # those InPorts; readout their values to correspoinding data-stroing
        # Var
        for i in range(self.proc_params["n_in_ports"]):
            in_port_name = self.proc_params["InPorts"][i]
            out_read_name = self.proc_params["VarsData2"][i]
            # Check if buffer is non-empty, otherwise add zero to the list
            if getattr(self, in_port_name).csp_ports[0].probe():
                getattr(self, out_read_name)[:, self.current_ts - 1] = \
                    np.squeeze(np.array(getattr(self, in_port_name).recv()))
            else:
                getattr(self, out_read_name)[:, self.current_ts - 1] = 0
