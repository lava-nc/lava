# Copyright (C) 2021 Intel Corporation
# SPDX-License-Identifier: BSD-3-Clause
# See: https://spdx.org/licenses/

from lava.magma.core.process.process import AbstractProcess
from lava.magma.core.process.variable import Var
from lava.magma.core.process.ports.ports import InPort, OutPort, RefPort

class Monitor(AbstractProcess):
    """Monitor process to probe/monitor a given variable of a process
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # self.ref = []
        self.data = {}

        self.proc_params["RefPorts"] = []
        self.proc_params["VarsRead"] = []
        self.proc_params["InPorts"] = []
        self.proc_params["OutsRead"] = []


    def probe(self, var, num_steps):
        new_ref_port_name = "ref_port_" + str(var.process.name) \
                            + "_" + str(var.name)
        new_var_read_name = "var_read_" + str(var.process.name) \
                            + "_" + str(var.name)

        # create and set new Refport and Var_read
        setattr(self, new_ref_port_name, RefPort(shape=var.shape))
        setattr(self, new_var_read_name, Var(shape=(num_steps,), init=0))

        # Add the names of new RefPort and Var_read name to proc_params dict
        self.proc_params["RefPorts"].append(new_ref_port_name)
        self.proc_params["VarsRead"].append(new_var_read_name)

        # Register them
        attrs = self._find_attr_by_type(RefPort)
        self._init_proc_member_obj(attrs)
        self.ref_ports.add_members(attrs)

        attrs = self._find_attr_by_type(Var)
        self._init_proc_member_obj(attrs)
        self.vars.add_members(attrs)

        # connect newly created Refport to the var to be monitored
        getattr(self, new_ref_port_name).connect_var(var)

        # self.data[str(var.process.name)] = {}
        # self.data[str(var.process.name)][str(var.name)] = []

    def probe_port(self, port, num_steps):
        new_in_port_name = "in_port_" + str(port.process.name) \
                            + "_" + str(port.name)
        new_out_read_name = "out_read_" + str(port.process.name) \
                            + "_" + str(port.name)

        # create and set new Refport and Var_read
        setattr(self, new_in_port_name, InPort(shape=port.shape))
        setattr(self, new_out_read_name, Var(shape=(port.shape[0],num_steps),
                                             init=0))

        # Add the names of new RefPort and Var_read name to proc_params dict
        self.proc_params["InPorts"].append(new_in_port_name)
        self.proc_params["OutsRead"].append(new_out_read_name)

        # Register them
        attrs = self._find_attr_by_type(InPort)
        self._init_proc_member_obj(attrs)
        self.in_ports.add_members(attrs)

        attrs = self._find_attr_by_type(Var)
        self._init_proc_member_obj(attrs)
        self.vars.add_members(attrs)

        # connect newly created InPort from the OutPort to be monitored
        getattr(self, new_in_port_name).connect_from(port)
        return