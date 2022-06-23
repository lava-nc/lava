# Copyright (C) 2021 Intel Corporation
# SPDX-License-Identifier: BSD-3-Clause
# See: https://spdx.org/licenses/

import numpy as np
from lava.magma.core.model.py.ports import PyRefPort, PyInPort
from lava.proc.monitor.process import Monitor
from lava.magma.core.sync.protocols.loihi_protocol import LoihiProtocol
from lava.magma.core.model.py.model import PyLoihiProcessModel
from lava.magma.core.model.py.type import LavaPyType
from lava.magma.core.decorator import implements, requires
from lava.magma.core.resources import CPU


@implements(proc=Monitor, protocol=LoihiProtocol)
@requires(CPU)
class PyMonitorModel(PyLoihiProcessModel):
    """
    This process model contains one LavaPyType for each of dynamically
    extended container of RefPorts, InPorts and their corresponding memory Vars.
    """

    ref_ports_list: list = LavaPyType(PyRefPort.VEC_DENSE, int, is_list=True)
    mem_vars_rp_list: list = LavaPyType(np.ndarray, float, precision=24,
                                        is_list=True)
    in_ports_list: list = LavaPyType(PyInPort.VEC_DENSE, int, is_list=True)
    mem_vars_ip_list: list = LavaPyType(np.ndarray, float, precision=24,
                                        is_list=True)

    def post_guard(self):
        return True

    def run_post_mgmt(self):
        """
        During this phase, RefPorts of Monitor process collects data from
        monitored Vars
        """
        # In each time step loop over those RefPorts of the Monitor Process,
        # readout their values to corresponding data-storing memory Var
        for ind in range(self.proc_params["n_ref_ports"]):
            ref_port_name = self.proc_params["ref_ports"][ind]
            memory_var_name = self.proc_params["mem_vars_rp"][ind]
            getattr(self, memory_var_name)[self.time_step - 1, ...] = \
                np.squeeze(np.array(getattr(self, ref_port_name).read()))

    def run_spk(self):
        """
        During this phase, InPorts of Monitor process collects data from
        monitored OutPorts
        """
        # At each time step loop over the InPorts of the Monitor process and
        # store their values in their corresponding memory Var
        for i in range(self.proc_params["n_in_ports"]):
            in_port_name = self.proc_params["in_ports"][i]
            out_read_name = self.proc_params["mem_vars_ip"][i]
            getattr(self, out_read_name)[self.time_step - 1, ...] = \
                np.squeeze(np.array(getattr(self, in_port_name).recv()))
