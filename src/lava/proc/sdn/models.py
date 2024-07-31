# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: BSD-3-Clause
# See: https://spdx.org/licenses/

import numpy as np
from typing import Any, Dict

from lava.magma.core.sync.protocols.loihi_protocol import LoihiProtocol
from lava.magma.core.model.py.ports import PyInPort, PyOutPort
from lava.magma.core.model.py.type import LavaPyType
from lava.magma.core.resources import CPU
from lava.magma.core.decorator import implements, requires, tag
from lava.magma.core.model.py.model import PyLoihiProcessModel
from lava.proc.sdn.process import Sigma, Delta, SigmaDelta, ActivationMode


def ReLU(x: np.ndarray) -> np.ndarray:
    """ReLU activation implementation

    Parameters
    ----------
    x : np.ndarray
        input array

    Returns
    -------
    np.ndarray
        output array
    """
    return np.maximum(x, 0)


class AbstractSigmaModel(PyLoihiProcessModel):
    a_in = None
    a_out = None

    sigma = None

    def sigma_dynamics(self, a_in_data: np.ndarray) -> np.ndarray:
        """Sigma decoding dynamics method

        Parameters
        ----------
        a_in_data : np.ndarray
            Input data

        Returns
        -------
        np.ndarray
            decoded data
        """
        return a_in_data + self.sigma

    def run_spk(self) -> None:
        a_in_data = self.a_in.recv()
        self.sigma = self.sigma_dynamics(a_in_data)
        self.s_out.send(self.sigma)


class AbstractDeltaModel(PyLoihiProcessModel):
    a_in = None
    s_out = None

    vth = None
    act = None
    residue = None
    error = None
    spike_exp = None
    state_exp = None
    cum_error = None

    def delta_dynamics(self, act_data: np.ndarray) -> np.ndarray:
        """Delta encodind dynamics method

        Parameters
        ----------
        act_data : np.ndarray
            data to be encoded

        Returns
        -------
        np.ndarray
            delta encoded data
        """
        delta = act_data - self.act + self.residue

        if self.cum_error:
            self.error += delta
            s_out = np.where(
                np.abs(self.error) >= self.vth,
                delta, 0
            )
            self.error *= 1 - (np.abs(s_out) > 0)
        else:
            s_out = np.where(
                np.abs(delta) >= self.vth,
                delta, 0
            )
        self.residue = delta - s_out
        return s_out


class AbstractSigmaDeltaModel(AbstractSigmaModel, AbstractDeltaModel):
    a_in = None
    s_out = None

    vth = None
    sigma = None
    act = None
    residue = None
    error = None
    bias = None
    spike_exp = None
    state_exp = None
    cum_error = None

    def __init__(self, proc_params: Dict[str, Any]) -> None:
        super().__init__(proc_params)
        self.act_mode = self.proc_params['act_mode']

    def activation_dynamics(self, sigma_data: np.ndarray) -> np.ndarray:
        """Sigma Delta activation dynamics. UNIT and RELU activations are
        supported.

        Parameters
        ----------
        sigma_data : np.ndarray
            sigma decoded data

        Returns
        -------
        np.ndarray
            activation output

        Raises
        ------
        NotImplementedError
            if activation mode other than UNIT or RELU is encountered.
        """
        if self.act_mode == ActivationMode.UNIT:
            act = sigma_data + self.bias
        elif self.act_mode == ActivationMode.RELU:
            act = ReLU(sigma_data + self.bias)
        else:
            raise NotImplementedError(
                f'Activation mode {self.act_mode} is not implemented.'
            )
        return act

    def dynamics(self, a_in_data: np.ndarray) -> np.ndarray:
        self.sigma = self.sigma_dynamics(a_in_data)
        act = self.activation_dynamics(self.sigma)
        s_out = self.delta_dynamics(act)
        self.act = act

        return s_out


@implements(proc=Sigma, protocol=LoihiProtocol)
@requires(CPU)
@tag('floating_pt')
class PySigmaModelFloat(AbstractSigmaModel):
    """ Floating point implementation of Sigma decoding"""
    a_in = LavaPyType(PyInPort.VEC_DENSE, float)
    s_out = LavaPyType(PyOutPort.VEC_DENSE, float)
    sigma: np.ndarray = LavaPyType(np.ndarray, float)


@implements(proc=Sigma, protocol=LoihiProtocol)
@requires(CPU)
@tag('fixed_pt')
class PySigmaModelFixed(AbstractSigmaModel):
    """ Fixed point implementation of Sigma decoding"""
    a_in = LavaPyType(PyInPort.VEC_DENSE, np.int32, precision=24)
    s_out = LavaPyType(PyOutPort.VEC_DENSE, np.int32, precision=24)
    sigma: np.ndarray = LavaPyType(np.ndarray, np.int32, precision=24)


@implements(proc=Delta, protocol=LoihiProtocol)
@requires(CPU)
@tag('floating_pt')
class PyDeltaModelFloat(AbstractDeltaModel):
    """Floating point implementation of Delta encoding."""
    a_in = LavaPyType(PyInPort.VEC_DENSE, float)
    s_out = LavaPyType(PyOutPort.VEC_DENSE, float)

    vth: np.ndarray = LavaPyType(np.ndarray, float)
    sigma: np.ndarray = LavaPyType(np.ndarray, float)
    act: np.ndarray = LavaPyType(np.ndarray, float)
    residue: np.ndarray = LavaPyType(np.ndarray, float)
    error: np.ndarray = LavaPyType(np.ndarray, float)

    spike_exp: np.ndarray = LavaPyType(np.ndarray, np.int32, precision=3)
    state_exp: np.ndarray = LavaPyType(np.ndarray, np.int32, precision=3)
    cum_error: np.ndarray = LavaPyType(np.ndarray, bool, precision=1)

    def run_spk(self) -> None:
        # Receive synaptic input
        a_in_data = self.a_in.recv()
        s_out = self.delta_dynamics(a_in_data)
        self.act = a_in_data
        self.s_out.send(s_out)


@implements(proc=Delta, protocol=LoihiProtocol)
@requires(CPU)
@tag('fixed_pt')
class PyDeltaModelFixed(AbstractDeltaModel):
    """Fixed point implementation of Delta encoding."""
    a_in = LavaPyType(PyInPort.VEC_DENSE, np.int32, precision=24)
    s_out = LavaPyType(PyOutPort.VEC_DENSE, np.int32, precision=24)

    vth: np.ndarray = LavaPyType(np.ndarray, np.int32, precision=24)
    sigma: np.ndarray = LavaPyType(np.ndarray, np.int32, precision=24)
    act: np.ndarray = LavaPyType(np.ndarray, np.int32, precision=24)
    residue: np.ndarray = LavaPyType(np.ndarray, np.int32, precision=24)
    error: np.ndarray = LavaPyType(np.ndarray, np.int32, precision=24)

    spike_exp: np.ndarray = LavaPyType(np.ndarray, np.int32, precision=3)
    state_exp: np.ndarray = LavaPyType(np.ndarray, np.int32, precision=3)
    cum_error: np.ndarray = LavaPyType(np.ndarray, bool, precision=1)

    def run_spk(self) -> None:
        # Receive synaptic input
        a_in_data = np.left_shift(
            self.a_in.recv(), self.spike_exp + self.state_exp
        )
        s_out_scaled = self.delta_dynamics(a_in_data)
        s_out = np.right_shift(s_out_scaled, self.state_exp)
        self.act = a_in_data
        self.s_out.send(s_out)


@implements(proc=SigmaDelta, protocol=LoihiProtocol)
@requires(CPU)
@tag('floating_pt')
class PySigmaDeltaModelFloat(AbstractSigmaDeltaModel):
    """Floating point implementation of Sigma Delta neuron."""
    a_in = LavaPyType(PyInPort.VEC_DENSE, float)
    s_out = LavaPyType(PyOutPort.VEC_DENSE, float)

    vth: np.ndarray = LavaPyType(np.ndarray, float)
    sigma: np.ndarray = LavaPyType(np.ndarray, float)
    act: np.ndarray = LavaPyType(np.ndarray, float)
    residue: np.ndarray = LavaPyType(np.ndarray, float)
    error: np.ndarray = LavaPyType(np.ndarray, float)
    bias: np.ndarray = LavaPyType(np.ndarray, float)

    spike_exp: np.ndarray = LavaPyType(np.ndarray, np.int32, precision=3)
    state_exp: np.ndarray = LavaPyType(np.ndarray, np.int32, precision=3)
    cum_error: np.ndarray = LavaPyType(np.ndarray, bool, precision=1)

    def run_spk(self) -> None:
        # Receive synaptic input
        a_in_data = self.a_in.recv()
        s_out = self.dynamics(a_in_data)
        self.s_out.send(s_out)


@implements(proc=SigmaDelta, protocol=LoihiProtocol)
@requires(CPU)
@tag('fixed_pt')
class PySigmaDeltaModelFixed(AbstractSigmaDeltaModel):
    """Fixed point implementation of Sigma Delta neuron."""
    a_in = LavaPyType(PyInPort.VEC_DENSE, np.int32, precision=24)
    s_out = LavaPyType(PyOutPort.VEC_DENSE, np.int32, precision=24)

    vth: np.ndarray = LavaPyType(np.ndarray, np.int32, precision=24)
    sigma: np.ndarray = LavaPyType(np.ndarray, np.int32, precision=24)
    act: np.ndarray = LavaPyType(np.ndarray, np.int32, precision=24)
    residue: np.ndarray = LavaPyType(np.ndarray, np.int32, precision=24)
    error: np.ndarray = LavaPyType(np.ndarray, np.int32, precision=24)
    bias: np.ndarray = LavaPyType(np.ndarray, np.int32, precision=16)

    spike_exp: np.ndarray = LavaPyType(np.ndarray, np.int32, precision=3)
    state_exp: np.ndarray = LavaPyType(np.ndarray, np.int32, precision=3)
    cum_error: np.ndarray = LavaPyType(np.ndarray, bool, precision=1)

    def run_spk(self) -> None:
        # Receive synaptic input
        a_in_data = self.a_in.recv()
        s_out_scaled = self.dynamics(a_in_data)
        s_out = np.right_shift(s_out_scaled, self.state_exp)
        self.s_out.send(s_out)


@implements(proc=SigmaDelta, protocol=LoihiProtocol)
@requires(CPU)
@tag('fixed_pt')
class PySigmaDeltaModelFixedCorrected(AbstractSigmaDeltaModel):
    """Fixed point implementation of Sigma Delta neuron."""
    a_in = LavaPyType(PyInPort.VEC_DENSE, np.int32, precision=24)
    s_out = LavaPyType(PyOutPort.VEC_DENSE, np.int32, precision=24)

    vth: np.ndarray = LavaPyType(np.ndarray, np.int32, precision=24)
    sigma: np.ndarray = LavaPyType(np.ndarray, np.int32, precision=24)
    act: np.ndarray = LavaPyType(np.ndarray, np.int32, precision=24)
    residue: np.ndarray = LavaPyType(np.ndarray, np.int32, precision=24)
    error: np.ndarray = LavaPyType(np.ndarray, np.int32, precision=24)
    bias: np.ndarray = LavaPyType(np.ndarray, np.int32, precision=16)

    spike_exp: np.ndarray = LavaPyType(np.ndarray, np.int32, precision=3)
    state_exp: np.ndarray = LavaPyType(np.ndarray, np.int32, precision=3)
    cum_error: np.ndarray = LavaPyType(np.ndarray, bool, precision=1)

    def run_spk(self) -> None:
        # Receive synaptic input
        a_in_data = self.a_in.recv()
        self.sigma = self.sigma_dynamics(a_in_data)
        act = self.activation_dynamics(self.sigma)
        delta = act - self.act
        s_out_scaled = np.where(np.abs(delta) >= self.vth, delta, 0)
        s_out = np.right_shift(s_out_scaled, self.state_exp)
        delta = np.left_shift(s_out, self.state_exp)
        self.act += delta
        self.s_out.send(s_out)
