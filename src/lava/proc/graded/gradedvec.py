# INTEL CORPORATION CONFIDENTIAL AND PROPRIETARY
#
# Copyright © 2021-2023 Intel Corporation.
#
# This software and the related documents are Intel copyrighted
# materials, and your use of them is governed by the express
# license under which they were provided to you (License). Unless
# the License provides otherwise, you may not use, modify, copy,
# publish, distribute, disclose or transmit  this software or the
# related documents without Intel's prior written permission.
#
# This software and the related documents are provided as is, with
# no express or implied warranties, other than those that are
# expressly stated in the License.
# See: https://spdx.org/licenses/

import os
import numpy as np
import typing as ty
from typing import Any, Dict
from enum import IntEnum, unique

from lava.magma.core.process.process import AbstractProcess
from lava.magma.core.process.variable import Var
from lava.magma.core.process.ports.ports import InPort, OutPort

from lava.magma.core.sync.protocols.loihi_protocol import LoihiProtocol
from lava.magma.core.model.py.ports import PyInPort, PyOutPort
from lava.magma.core.model.py.type import LavaPyType
from lava.magma.core.resources import CPU
from lava.magma.core.decorator import implements, requires, tag
from lava.magma.core.model.py.model import PyLoihiProcessModel

def loihi2round(vv):
    """round values in numpy array the way loihi 2 performs rounding/truncation."""
    return np.fix(vv + (vv > 0) - 0.5).astype('int')


class GradedVec(AbstractProcess):
    def __init__(
            self,
            *,
            shape: ty.Tuple[int, ...],
            vth=1,
            exp=0) -> None:
        """GradedVec
        Thresholded graded spike vector
        
        Parameters
        ----------
        shape: tuple(int)
            number and topology of neurons
        vth: int
            threshold for spiking
        exp: int
            fixed point base
        """
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


class AbstractGradedVecModel(PyLoihiProcessModel):
    a_in = None
    s_out = None

    v = None
    vth = None
    exp = None
    
    def run_spk(self) -> None:
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
    """ Fixed point implementation of GradedVec"""
    a_in = LavaPyType(PyInPort.VEC_DENSE, np.int32, precision=24)
    s_out = LavaPyType(PyOutPort.VEC_DENSE, np.int32, precision=24)
    vth: np.ndarray = LavaPyType(np.ndarray, np.int32, precision=24)
    v: np.ndarray = LavaPyType(np.ndarray, np.int32, precision=24)
    exp: np.ndarray = LavaPyType(np.ndarray, np.int32, precision=24)
