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


    def probe(self, var, num_steps):

        self.ref = RefPort(shape=var.shape)
        attrs = self._find_attr_by_type(RefPort)
        self._init_proc_member_obj(attrs)
        self.ref_ports.add_members(attrs)

        self.ref.connect_var(var)

        self.var_read = Var(shape=(num_steps,), init=0)
        attrs = self._find_attr_by_type(Var)
        self._init_proc_member_obj(attrs)
        self.vars.add_members(attrs)

        self.data[str(var.process.name)]={}
        self.data[str(var.process.name)][str(var.name)]=[]


        return