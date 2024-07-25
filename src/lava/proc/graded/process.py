# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: BSD-3-Clause
# See: https://spdx.org/licenses/

import numpy as np
import typing as ty

from lava.magma.core.process.process import AbstractProcess
from lava.magma.core.process.variable import Var
from lava.magma.core.process.ports.ports import InPort, OutPort


def loihi2round(vv: np.ndarray) -> np.ndarray:
    """
    Round values in numpy array the way loihi 2
    performs rounding/truncation.

    Parameters
    ----------
    vv : np.ndarray
        Input values to be rounded consistent with loihi2 rouding.

    Returns
    -------
    vv_r : np.ndarray
        Output values rounded consistent with loihi2 rouding.
    """
    vv_r = np.fix(vv + (vv > 0) - 0.5).astype('int')
    return vv_r


class GradedVec(AbstractProcess):
    """GradedVec
    Graded spike vector layer. Transmits accumulated input as
    graded spike with no dynamics.

    v[t] = a_in
    s_out = v[t] * (|v[t]| > vth)

    Parameters
    ----------
    shape: tuple(int)
        number and topology of neurons
    vth: int
        threshold for spiking
    exp: int
        fixed point base
    """

    def __init__(
            self,
            shape: ty.Tuple[int, ...],
            vth: ty.Optional[int] = 1,
            exp: ty.Optional[int] = 0) -> None:

        super().__init__(shape=shape)

        self.a_in = InPort(shape=shape)
        self.s_out = OutPort(shape=shape)

        self.v = Var(shape=shape, init=0)
        self.vth = Var(shape=(1,), init=vth)
        self.exp = Var(shape=(1,), init=exp)

    @property
    def shape(self) -> ty.Tuple[int, ...]:
        """Return shape of the Process."""
        return self.proc_params['shape']


class GradedReluVec(AbstractProcess):
    """GradedReluVec
    Graded spike vector layer. Transmits accumulated input as
    graded spike with no dynamics.

    v[t] = a_in
    s_out = v[t] * (v[t] > vth)

    Parameters
    ----------
    shape : tuple(int)
        Number and topology of neurons.
    vth : int
        Threshold for spiking.
    exp : int
        Fixed point base.
    """

    def __init__(
            self,
            shape: ty.Tuple[int, ...],
            vth: ty.Optional[int] = 1,
            exp: ty.Optional[int] = 0) -> None:

        super().__init__(shape=shape)

        self.a_in = InPort(shape=shape)
        self.s_out = OutPort(shape=shape)

        self.v = Var(shape=shape, init=0)
        self.vth = Var(shape=(1,), init=vth)
        self.exp = Var(shape=(1,), init=exp)

    @property
    def shape(self) -> ty.Tuple[int, ...]:
        """Return shape of the Process."""
        return self.proc_params['shape']


class NormVecDelay(AbstractProcess):
    """NormVec
    Normalizable graded spike vector. Used in conjunction with
    InvSqrt process to create normalization layer.

    When configured with InvSqrt, the process will output a normalized
    vector with graded spike values. The output is delayed by 2 timesteps.

    NormVecDelay has two input and two output channels. The process
    outputs the square input values on the second channel to the InvSqrt
    neuron. The process waits two timesteps to receive the inverse
    square root value returned by the InvSqrt process. The value received
    on the second input channel is multiplied by the primary input
    value, and the result is output on the primary output channel.

    v[t] = a_in1
    s2_out = v[t] ** 2
    s_out = v[t-2] * a_in2

    Parameters
    ----------
    shape : tuple(int)
        Number and topology of neurons.
    vth : int
        Threshold for spiking.
    exp : int
        Fixed point base.
    """

    def __init__(
            self,
            shape: ty.Tuple[int, ...],
            vth: ty.Optional[int] = 1,
            exp: ty.Optional[int] = 0) -> None:

        super().__init__(shape=shape)

        self.a_in1 = InPort(shape=shape)
        self.a_in2 = InPort(shape=shape)

        self.s_out = OutPort(shape=shape)
        self.s2_out = OutPort(shape=shape)

        self.vth = Var(shape=(1,), init=vth)
        self.exp = Var(shape=(1,), init=exp)

        self.v = Var(shape=shape, init=np.zeros(shape, 'int32'))
        self.v2 = Var(shape=shape, init=np.zeros(shape, 'int32'))

    @property
    def shape(self) -> ty.Tuple[int, ...]:
        """Return shape of the Process."""
        return self.proc_params['shape']


class InvSqrt(AbstractProcess):
    """InvSqrt
    Neuron model for computing inverse square root with 24-bit
    fixed point values. Designed to be used in conjunction with
    NormVecDelay.

    v[t] = a_in
    s_out = 1 / sqrt(v[t])

    Parameters
    ----------
    shape : tuple(int)
        Number and topology of neurons.

    fp_base : int
        Base of the fixed-point representation.
    """

    def __init__(
            self,
            shape: ty.Tuple[int, ...],
            fp_base: ty.Optional[int] = 12) -> None:
        super().__init__(shape=shape)

        # Base of the decimal point
        self.fp_base = Var(shape=(1,), init=fp_base)
        self.a_in = InPort(shape=shape)
        self.s_out = OutPort(shape=shape)

    @property
    def shape(self) -> ty.Tuple[int, ...]:
        """Return shape of the Process."""
        return self.proc_params['shape']
