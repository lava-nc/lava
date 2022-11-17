# Copyright (C) 2021-22 Intel Corporation
# SPDX-License-Identifier: BSD-3-Clause
# See: https://spdx.org/licenses/

import numpy as np
from lava.magma.core.sync.protocols.loihi_protocol import LoihiProtocol
from lava.magma.core.model.py.ports import PyInPort, PyOutPort
from lava.magma.core.model.py.type import LavaPyType
from lava.magma.core.resources import CPU
from lava.magma.core.decorator import implements, requires, tag
from lava.magma.core.model.py.model import PyLoihiProcessModel
from lava.proc.rf.process import RF


class AbstractPyRFModelFloat(PyLoihiProcessModel):
    a_real_in: PyInPort = LavaPyType(PyInPort.VEC_DENSE, float)
    a_imag_in: PyInPort = LavaPyType(PyInPort.VEC_DENSE, float)
    s_out: PyOutPort = LavaPyType(PyOutPort.VEC_DENSE, bool, precision=1)
    real: np.ndarray = LavaPyType(np.ndarray, float)
    imag: np.ndarray = LavaPyType(np.ndarray, float)
    sin_decay: float = LavaPyType(float, float)
    cos_decay: float = LavaPyType(float, float)
    state_exp: np.ndarray = LavaPyType(np.ndarray, np.int32, precision=3)
    decay_bits: np.ndarray = LavaPyType(np.ndarray, np.int32, precision=3)
    vth: float = LavaPyType(float, float)

    def scale_volt(self, voltage):
        """No downscale of voltage needed for floating point implementation"""
        return voltage

    def resonator_dynamics(self, a_real_in_data, a_imag_in_data, real, imag):
        """Resonate and Fire real and imaginary voltage dynamics

        Parameters
        ----------
        a_real_in_data : np.ndarray
            Real component input current
        a_imag_in_data : np.ndarray
            Imaginary component input current
        real : np.ndarray
            Real component voltage to be updated
        imag : np.ndarray
            Imag component voltage to be updated

        Returns
        -------
        np.ndarray, np.ndarray
            updated real and imaginary components

        """

        decayed_real = self.scale_volt(self.cos_decay * real) \
            - self.scale_volt(self.sin_decay * imag) \
            + a_real_in_data
        decayed_imag = self.scale_volt(self.sin_decay * real) \
            + self.scale_volt(self.cos_decay * imag) \
            + a_imag_in_data

        return decayed_real, decayed_imag

    def run_spk(self):
        raise NotImplementedError("spiking activation() cannot be called from "
                                  "an abstract ProcessModel")


@implements(proc=RF, protocol=LoihiProtocol)
@requires(CPU)
@tag('floating_pt')
class PyRFModelFloat(AbstractPyRFModelFloat):
    """Implementation of Resonate-and-Fire neural process in floating
    point precision. This short and simple ProcessModel can be used for quick
    algorithmic prototyping, without engaging with the nuances of a fixed
    point implementation.
    """

    def run_spk(self):
        a_real_in_data = self.a_real_in.recv()
        a_imag_in_data = self.a_imag_in.recv()

        new_real, new_imag = self.resonator_dynamics(a_real_in_data,
                                                     a_imag_in_data,
                                                     self.real,
                                                     self.imag)

        s_out = (new_real >= self.vth) * (new_imag >= 0) * (self.imag < 0)
        self.real[:], self.imag[:] = new_real, new_imag
        self.s_out.send(s_out)


class AbstractPyRFModelFixed(AbstractPyRFModelFloat):
    a_real_in: PyInPort = LavaPyType(PyInPort.VEC_DENSE,
                                     np.int16, precision=24)
    a_imag_in: PyInPort = LavaPyType(PyInPort.VEC_DENSE,
                                     np.int16, precision=24)
    s_out = LavaPyType(PyOutPort.VEC_DENSE, np.int32, precision=24)
    real: np.ndarray = LavaPyType(np.ndarray, np.int32, precision=24)
    imag: np.ndarray = LavaPyType(np.ndarray, np.int32, precision=24)
    sin_decay: int = LavaPyType(int, np.uint16, precision=12)
    cos_decay: int = LavaPyType(int, np.uint16, precision=12)
    state_exp: np.ndarray = LavaPyType(np.ndarray, np.int32, precision=3)
    decay_bits: np.ndarray = LavaPyType(np.ndarray, np.int32, precision=3)
    vth: np.ndarray = LavaPyType(np.ndarray, np.int32, precision=24)

    def __init__(self, proc_params):
        super(AbstractPyRFModelFixed, self).__init__(proc_params)

        # real, imaginary bitwidth
        self.ri_bitwidth = self.sin_decay.precision * 2
        max_ri_val = 2 ** (self.ri_bitwidth - 1)
        self.neg_voltage_limit = -np.int32(max_ri_val) + 1
        self.pos_voltage_limit = np.int32(max_ri_val) - 1

    def scale_volt(self, voltage):
        return np.sign(voltage) * np.right_shift(np.abs(
            voltage), self.decay_bits)

    def run_spk(self):
        raise NotImplementedError("spiking activation() cannot be called from "
                                  "an abstract ProcessModel")


@implements(proc=RF, protocol=LoihiProtocol)
@requires(CPU)
@tag('fixed_pt')
class PyRFModelFixed(AbstractPyRFModelFixed):
    """Fixed point implementation of Resonate and Fire neuron."""
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
        s_out = (new_real >= self.vth) * (new_imag >= 0) * (self.imag < 0)
        self.real[:], self.imag[:] = new_real, new_imag

        self.s_out.send(s_out)
