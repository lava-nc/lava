# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: BSD-3-Clause
# See: https://spdx.org/licenses/

import numpy as np

from lava.magma.core.sync.protocols.loihi_protocol import LoihiProtocol
from lava.magma.core.model.py.ports import PyInPort, PyOutPort
from lava.magma.core.model.py.type import LavaPyType
from lava.magma.core.resources import CPU
from lava.magma.core.decorator import implements, requires, tag
from lava.magma.core.model.py.model import PyLoihiProcessModel

from lava.proc.graded.process import (GradedVec, GradedReluVec,
                                      NormVecDelay, InvSqrt)


class AbstractGradedVecModel(PyLoihiProcessModel):
    """Implementation of GradedVec"""

    a_in = None
    s_out = None
    v = None
    vth = None
    exp = None

    def run_spk(self) -> None:
        """The run function that performs the actual computation during
        execution orchestrated by a PyLoihiProcessModel using the
        LoihiProtocol.
        """
        a_in_data = self.a_in.recv()
        self.v += a_in_data

        is_spike = np.abs(self.v) > self.vth
        sp_out = self.v * is_spike

        self.v[:] = 0

        self.s_out.send(sp_out)


@implements(proc=GradedVec, protocol=LoihiProtocol)
@requires(CPU)
@tag('fixed_pt')
class PyGradedVecModelFixed(AbstractGradedVecModel):
    """Fixed point implementation of GradedVec"""
    a_in = LavaPyType(PyInPort.VEC_DENSE, np.int32, precision=24)
    s_out = LavaPyType(PyOutPort.VEC_DENSE, np.int32, precision=24)
    vth: np.ndarray = LavaPyType(np.ndarray, np.int32, precision=24)
    v: np.ndarray = LavaPyType(np.ndarray, np.int32, precision=24)
    exp: np.ndarray = LavaPyType(np.ndarray, np.int32, precision=24)


class AbstractGradedReluVecModel(PyLoihiProcessModel):
    """Implementation of GradedReluVec"""

    a_in = None
    s_out = None
    v = None
    vth = None
    exp = None

    def run_spk(self) -> None:
        """The run function that performs the actual computation during
        execution orchestrated by a PyLoihiProcessModel using the
        LoihiProtocol.
        """
        a_in_data = self.a_in.recv()
        self.v += a_in_data

        is_spike = self.v > self.vth
        sp_out = self.v * is_spike

        self.v[:] = 0

        self.s_out.send(sp_out)


@implements(proc=GradedReluVec, protocol=LoihiProtocol)
@requires(CPU)
@tag('fixed_pt')
class PyGradedReluVecModelFixed(AbstractGradedReluVecModel):
    """Fixed point implementation of GradedVec"""
    a_in = LavaPyType(PyInPort.VEC_DENSE, np.int32, precision=24)
    s_out = LavaPyType(PyOutPort.VEC_DENSE, np.int32, precision=24)
    vth: np.ndarray = LavaPyType(np.ndarray, np.int32, precision=24)
    v: np.ndarray = LavaPyType(np.ndarray, np.int32, precision=24)
    exp: np.ndarray = LavaPyType(np.ndarray, np.int32, precision=24)


@implements(proc=NormVecDelay, protocol=LoihiProtocol)
@requires(CPU)
@tag('fixed_pt')
class NormVecDelayModel(PyLoihiProcessModel):
    """Implementation of NormVecDelay. This process is typically part of
    a network for normalization.
    """
    a_in1 = LavaPyType(PyInPort.VEC_DENSE, np.int32, precision=24)
    a_in2 = LavaPyType(PyInPort.VEC_DENSE, np.int32, precision=24)
    s_out = LavaPyType(PyOutPort.VEC_DENSE, np.int32, precision=24)
    s2_out = LavaPyType(PyOutPort.VEC_DENSE, np.int32, precision=24)

    vth: np.ndarray = LavaPyType(np.ndarray, np.int32, precision=24)
    exp: np.ndarray = LavaPyType(np.ndarray, np.int32, precision=24)
    v: np.ndarray = LavaPyType(np.ndarray, np.int32, precision=24)
    v2: np.ndarray = LavaPyType(np.ndarray, np.int32, precision=24)

    def run_spk(self) -> None:
        """The run function that performs the actual computation during
        execution orchestrated by a PyLoihiProcessModel using the
        LoihiProtocol.
        """
        a_in_data1 = self.a_in1.recv()
        a_in_data2 = self.a_in2.recv()

        vsq = a_in_data1 ** 2
        self.s2_out.send(vsq)

        self.v2 = self.v
        self.v = a_in_data1

        output = self.v2 * a_in_data2

        is_spike = np.abs(output) > self.vth
        sp_out = output * is_spike

        self.s_out.send(sp_out)


@implements(proc=InvSqrt, protocol=LoihiProtocol)
@requires(CPU)
@tag('floating_pt')
class InvSqrtModelFloat(PyLoihiProcessModel):
    """Implementation of InvSqrt in floating point"""
    a_in = LavaPyType(PyInPort.VEC_DENSE, float)
    s_out = LavaPyType(PyOutPort.VEC_DENSE, float)

    fp_base: np.ndarray = LavaPyType(np.ndarray, np.int32, precision=24)

    def run_spk(self) -> None:
        """The run function that performs the actual computation during
        execution orchestrated by a PyLoihiProcessModel using the
        LoihiProtocol.
        """
        a_in_data = self.a_in.recv()
        sp_out = 1 / (a_in_data ** 0.5)

        self.s_out.send(sp_out)


def make_fpinv_table(fp_base: int) -> np.ndarray:
    """
    Creates the table for fp inverse square root algorithm.

    Parameters
    ----------
    fp_base : int
        Base of the fixed point.

    Returns
    -------
    Y_est : np.ndarray
        Initialization look-up table for fp inverse square root.
    """
    n_bits = 24
    B = 2**fp_base

    Y_est = np.zeros((n_bits), dtype='int')
    n_adj = 1.238982962

    for m in range(n_bits):  # Span the 24 bits, negate the decimal base
        Y_est[n_bits - m - 1] = 2 * int(B / (2**((m - fp_base) / 2) * n_adj))

    return Y_est


def clz(val: int) -> int:
    """
    Count lead zeros.

    Parameters
    ----------
    val : int
        Integer value for counting lead zeros.

    Returns
    -------
    out_val : int
        Number of leading zeros.
    """
    out_val = (24 - (int(np.log2(val)) + 1))
    return out_val


def inv_sqrt(s_fp: int,
             n_iters: int = 5,
             b_fraction: int = 12) -> int:
    """
    Runs the fixed point inverse square root algorithm.

    Parameters
    ----------
    s_fp : int
        Fixed point value to calulate inverse square root.
    n_iters : int, optional
        Number of iterations for fixed point inverse square root algorithm.
    b_fraction : int, optional
        Fixed point base.

    Returns
    -------
    y_i : int
        Approximate inverse square root in fixed point.
    """
    Y_est = make_fpinv_table(b_fraction)
    m = clz(s_fp)
    b_i = int(s_fp)
    Y_i = Y_est[m]
    y_i = Y_i // 2

    for _ in range(n_iters):
        b_i = np.right_shift(np.right_shift(b_i * Y_i,
                                            b_fraction + 1) * Y_i,
                             b_fraction + 1)
        Y_i = np.left_shift(3, b_fraction) - b_i
        y_i = np.right_shift(y_i * Y_i, b_fraction + 1)

    return y_i


@implements(proc=InvSqrt, protocol=LoihiProtocol)
@requires(CPU)
@tag('fixed_pt')
class InvSqrtModelFP(PyLoihiProcessModel):
    """Implementation of InvSqrt in fixed point"""

    a_in = LavaPyType(PyInPort.VEC_DENSE, np.int32, precision=24)
    s_out = LavaPyType(PyOutPort.VEC_DENSE, np.int32, precision=24)
    fp_base: np.ndarray = LavaPyType(np.ndarray, np.int32, precision=24)

    def run_spk(self) -> None:
        """The run function that performs the actual computation during
        execution orchestrated by a PyLoihiProcessModel using the
        LoihiProtocol.
        """
        a_in_data = self.a_in.recv()

        if np.any(a_in_data) == 0:
            sp_out = 0 * a_in_data
        else:
            sp_out = np.array([inv_sqrt(a_in_data, 5)])

        self.s_out.send(sp_out)
