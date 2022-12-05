# Copyright (C) 2021-22 Intel Corporation
# SPDX-License-Identifier: BSD-3-Clause
# See: https://spdx.org/licenses/

from lava.magma.core.sync.protocols.loihi_protocol import LoihiProtocol
from lava.magma.core.resources import CPU
import numpy as np
from lava.magma.core.decorator import implements, requires, tag
from lava.proc.rf_iz.process import RF_IZ
from lava.proc.rf.models import AbstractPyRFModelFloat, AbstractPyRFModelFixed


@implements(proc=RF_IZ, protocol=LoihiProtocol)
@requires(CPU)
@tag('floating_pt')
class PyRF_IZModelFloat(AbstractPyRFModelFloat):
    """Float point implementation of Resonate and Fire Izhikevich Neuron"""
    def run_spk(self):
        a_real_in_data = self.a_real_in.recv()
        a_imag_in_data = self.a_imag_in.recv()

        new_real, new_imag = self.resonator_dynamics(a_real_in_data,
                                                     a_imag_in_data,
                                                     self.real,
                                                     self.imag)
        s_out = new_imag >= self.vth
        self.real[:] = new_real * (1 - s_out)  # reset dynamics

        # the 1e-5 insures we don't spike again
        self.imag[:] = s_out * (self.vth - 1e-5) + (1 - s_out) * new_imag
        self.s_out.send(s_out)


@implements(proc=RF_IZ, protocol=LoihiProtocol)
@requires(CPU)
@tag('fixed_pt')
class PyRF_IZModelFixed(AbstractPyRFModelFixed):
    """Fixed point implementation of Resonate and Fire Izhikevich Neuron"""
    def run_spk(self):
        a_real_in_data = np.left_shift(self.a_real_in.recv(),
                                       self.state_exp)
        a_imag_in_data = np.left_shift(self.a_imag_in.recv(),
                                       self.state_exp)

        new_real, new_imag = self.resonator_dynamics(a_real_in_data,
                                                     a_imag_in_data,
                                                     np.int64(self.real),
                                                     np.int64(self.imag))

        new_real = np.clip(new_real,
                           self.neg_voltage_limit, self.pos_voltage_limit)
        new_imag = np.clip(new_imag,
                           self.neg_voltage_limit, self.pos_voltage_limit)

        s_out = new_imag >= self.vth
        self.real[:] = new_real * (1 - s_out)  # reset dynamics
        self.imag[:] = s_out * (self.vth - 1) + (1 - s_out) * new_imag
        self.s_out.send(s_out)
