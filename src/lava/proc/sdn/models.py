# Copyright (C) 2021 Intel Corporation
# SPDX-License-Identifier: BSD-3-Clause
# See: https://spdx.org/licenses/

import numpy as np
from typing import Any, Dict
from lava.magma.core.model.model import AbstractProcessModel

from lava.magma.core.sync.protocols.loihi_protocol import LoihiProtocol
from lava.magma.core.model.py.ports import PyInPort, PyOutPort
from lava.magma.core.model.py.type import LavaPyType
from lava.magma.core.resources import CPU
from lava.magma.core.decorator import implements, requires, tag
from lava.magma.core.model.py.model import PyLoihiProcessModel
from lava.proc.sdn.process import SigmaDelta, ACTIVATION_MODE


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


@implements(proc=SigmaDelta, protocol=LoihiProtocol)
class AbstractSigmaDeltaModel(PyLoihiProcessModel):
    a_in = None
    s_out = None

    vth = None
    sigma = None
    act = None
    residue = None
    error = None
    cum_error = None
    bias = None
    wgt_exp: np.ndarray = LavaPyType(np.ndarray, np.int32, precision=3)
    state_exp: np.ndarray = LavaPyType(np.ndarray, np.int32, precision=3)
    cum_error: np.ndarray = LavaPyType(np.ndarray, bool, precision=1)

    def __init__(self, proc_params: Dict[str, Any]) -> None:
        super().__init__(proc_params)
        self.act_mode = self.proc_params['act_fn']
        self.isbiasscaled = False
        self.isthrscaled = False
        self.effective_bias = None

    def dynamics(self, a_in_data: np.ndarray) -> np.ndarray:
        self.sigma = a_in_data + self.sigma

        if self.act_mode == ACTIVATION_MODE.Unit:
            act = self.sigma + self.effective_bias
        elif self.act_mode == ACTIVATION_MODE.ReLU:
            act = ReLU(self.sigma + self.effective_bias)
        else:
            raise NotImplementedError(
                f'Activation mode {self.act_mode} is not implemented.'
            )

        delta = act - self.act + self.residue

        if self.cum_error:
            self.error += delta
            s_out = np.where(
                np.abs(self.error) >= self.effective_vth,
                delta, 0
            )
            self.error *= 1 - (np.abs(s_out) > 0)
        else:
            s_out = np.where(
                np.abs(delta) >= self.effective_vth,
                delta, 0
            )
        self.residue = delta - s_out
        self.act = act

        return s_out


@requires(CPU)
@tag('fixed_pt')
class PySigmaDeltaModel(AbstractSigmaDeltaModel):
    """Fixed point implementation of Sigma Delta neuron."""
    a_in = LavaPyType(PyInPort.VEC_DENSE, np.int32, precision=24)
    s_out = LavaPyType(PyOutPort.VEC_DENSE, np.int32, precision=24)

    vth: np.ndarray = LavaPyType(np.ndarray, np.int32, precision=24)
    sigma: np.ndarray = LavaPyType(np.ndarray, np.int32, precision=24)
    act: np.ndarray = LavaPyType(np.ndarray, np.int32, precision=24)
    residue: np.ndarray = LavaPyType(np.ndarray, np.int32, precision=24)
    error: np.ndarray = LavaPyType(np.ndarray, np.int32, precision=24)
    bias: np.ndarray = LavaPyType(np.ndarray, np.int32, precision=16)

    def scale_bias(self):
        """Scale bias with bias exponent by taking into account sign of the
        exponent.
        """
        self.effective_bias = np.left_shift(
            self.bias,
            self.wgt_exp + self.state_exp
        )
        self.isbiasscaled = True

    def scale_threshold(self):
        """Scale threshold according to the way Loihi hardware scales it. In
        Loihi hardware, threshold is left-shifted by 6-bits to MSB-align it
        with other state variables of higher precision.
        """
        self.effective_vth = np.left_shift(
            self.vth,
            self.wgt_exp + self.state_exp
        )
        self.isthrscaled = True

    def run_spk(self) -> None:
        # Receive synaptic input
        a_in_data = self.a_in.recv()

        self.scale_bias()
        # # Compute effective bias and threshold only once, not every time-step
        # if not self.isbiasscaled:
        #     self.scale_bias()

        if not self.isthrscaled:
            self.scale_threshold()

        s_out_scaled = self.dynamics(a_in_data)

        s_out = np.right_shift(s_out_scaled, self.wgt_exp)
        # if not np.array_equal(s_out_scaled, np.left_shift(s_out, self.wgt_exp)):
        #     print(f'Possible quantization loss.')
        #     print(
        #         f's_out = {s_out_scaled.flatten()[:50] / (1 << self.wgt_exp)}'
        #     )
        #     print(f'{np.argwhere(np.left_shift(s_out, self.wgt_exp) != s_out_scaled)=}')

        self.s_out.send(s_out)
