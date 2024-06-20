# Copyright (C) 2022-23 Intel Corporation
# SPDX-License-Identifier: BSD-3-Clause
# See: https://spdx.org/licenses/

import numpy as np
import typing as ty

from lava.proc.graded.process import InvSqrt
from lava.proc.graded.process import NormVecDelay
from lava.proc.sparse.process import Sparse
from lava.proc.dense.process import Dense
from lava.proc.prodneuron.process import ProdNeuron
from lava.proc.graded.process import GradedVec as GradedVecProc
from lava.proc.lif.process import LIF
from lava.proc.io import sink, source

from .network import Network, AlgebraicVector, AlgebraicMatrix


class InputVec(AlgebraicVector):
    """InputVec
    Simple input vector. Adds algebraic syntax to RingBuffer

    Parameters
    ----------
    vec : np.ndarray
        NxM array of input values. Input will repeat every M steps.
    exp : int, optional
        Set the fixed point base value
    loihi2 : bool, optional
        Flag to create the adapters for loihi 2.
    """

    def __init__(self,
                 vec: np.ndarray,
                 loihi2: ty.Optional[bool] = False,
                 exp: ty.Optional[int] = 0,
                 **kwargs) -> None:

        self.loihi2 = loihi2
        self.shape = np.atleast_2d(vec).shape
        self.exp = exp

        # Convert it to fixed point base
        vec *= 2**self.exp

        self.inport_plug = source.RingBuffer(data=np.atleast_2d(vec))

        if self.loihi2:
            from lava.proc import embedded_io as eio
            self.inport_adapter = eio.spike.PyToNxAdapter(
                shape=(self.shape[0],),
                num_message_bits=24)
            self.inport_plug.s_out.connect(self.inport_adapter.inp)
            self.out_port = self.inport_adapter.out

        else:
            self.out_port = self.inport_plug.s_out

    def __lshift__(self, other):
        # Maybe this could be done with a numpy array and call set_data?
        return NotImplemented


class OutputVec(Network):
    """OutputVec
    Records spike output. Adds algebraic syntax to RingBuffer

    Parameters
    ----------
    shape : tuple(int)
        shape of the output to record
    buffer : int, optional
        length of the recording.
        (buffer is overwritten if shorter than sim time).
    loihi2 : bool, optional
        Flag to create the adapters for loihi 2.
    num_message_bits : int
        size of output message. ("0" is for unary spike event).
    """

    def __init__(self,
                 shape: ty.Tuple[int, ...],
                 buffer: int = 1,
                 loihi2: ty.Optional[bool] = False,
                 num_message_bits: ty.Optional[int] = 24,
                 **kwargs) -> None:

        self.shape = shape
        self.buffer = buffer
        self.loihi2 = loihi2
        self.num_message_bits = num_message_bits

        self.outport_plug = sink.RingBuffer(
            shape=self.shape, buffer=self.buffer, **kwargs)

        if self.loihi2:
            from lava.proc import embedded_io as eio
            self.outport_adapter = eio.spike.NxToPyAdapter(
                shape=self.shape, num_message_bits=self.num_message_bits)
            self.outport_adapter.out.connect(self.outport_plug.a_in)
            self.in_port = self.outport_adapter.inp
        else:
            self.in_port = self.outport_plug.a_in

    def get_data(self):
        return (self.outport_plug.data.get().astype(np.int32) << 8) >> 8


class LIFVec(AlgebraicVector):
    """LIFVec
    Network wrapper to LIF neuron.

    Parameters
    ----------
    See lava.proc.lif.process.LIF
    """

    def __init__(self, **kwargs):
        self.main = LIF(**kwargs)

        self.in_port = self.main.a_in
        self.out_port = self.main.s_out


class GradedVec(AlgebraicVector):
    """GradedVec
    Simple graded threshold vector with no dynamics.

    Parameters
    ----------
    shape : tuple(int)
        Number and topology of neurons.
    vth : int, optional
        Threshold for spiking.
    exp : int, optional
        Fixed point base of the vector.
    """

    def __init__(self,
                 shape: ty.Tuple[int, ...],
                 vth: int = 10,
                 exp: int = 0,
                 **kwargs):

        self.shape = shape
        self.vth = vth
        self.exp = exp

        self.main = GradedVecProc(shape=self.shape, vth=self.vth, exp=self.exp)
        self.in_port = self.main.a_in
        self.out_port = self.main.s_out

        super().__init__()

    def __mul__(self, other):
        if isinstance(other, GradedVec):
            # Create the product network
            prod_layer = ProductVec(shape=self.shape, vth=1, exp=self.exp)

            weightsI = np.eye(self.shape[0])

            weights_A = GradedSparse(weights=weightsI)
            weights_B = GradedSparse(weights=weightsI)
            weights_out = GradedSparse(weights=weightsI)

            prod_layer << (weights_A @ self, weights_B @ other)
            weights_out @ prod_layer
            return weights_out
        else:
            return NotImplemented


class ProductVec(AlgebraicVector):
    """ProductVec

    Neuron that will multiply values on two input channels.

    Parameters
    ----------
    shape : tuple(int)
        Number and topology of neurons.
    vth : int
        Threshold for spiking.
    exp : int
        Fixed point base of the vector.
    """

    def __init__(self,
                 shape: ty.Tuple[int, ...],
                 vth: ty.Optional[int] = 10,
                 exp: ty.Optional[int] = 0,
                 **kwargs):
        self.shape = shape
        self.vth = vth
        self.exp = exp

        self.main = ProdNeuron(shape=self.shape, vth=self.vth, exp=self.exp)

        self.in_port = self.main.a_in1
        self.in_port2 = self.main.a_in2

        self.out_port = self.main.s_out

    def __lshift__(self, other):
        # We're going to override the behavior here,
        # since there are two ports the API idea is:
        # prod_layer << (conn1, conn2)
        if isinstance(other, (list, tuple)):
            # It should be only length 2, and a Network object,
            # TODO: add checks
            other[0].out_port.connect(self.in_port)
            other[1].out_port.connect(self.in_port2)
        else:
            return NotImplemented


class GradedDense(AlgebraicMatrix):
    """GradedDense
    Network wrapper for Dense. Adds algebraic syntax to Dense.

    Parameters
    ----------
    See lava.proc.dense.process.Dense

    weights : numpy.ndarray
        Weight matrix expressed as floating point. Weights will be automatically
        reconfigured to fixed point (may lead to changes due to rounding).
    exp : int, optional
        Fixed point base of the weight (reconfigures weights/weight_exp).
    """

    def __init__(self,
                 weights: np.ndarray,
                 exp: int = 7,
                 **kwargs):
        self.exp = exp

        # Adjust the weights to the fixed point
        w = weights * 2 ** self.exp

        self.main = Dense(weights=w,
                          num_message_bits=24,
                          num_weight_bits=8,
                          weight_exp=-self.exp)

        self.in_port = self.main.s_in
        self.out_port = self.main.a_out


class GradedSparse(AlgebraicMatrix):
    """GradedSparse
    Network wrapper for Sparse. Adds algebraic syntax to Sparse.

    Parameters
    ----------
    See lava.proc.sparse.process.Sparse

    weights : numpy.ndarray
        Weight matrix expressed as floating point. Weights will be automatically
        reconfigured to fixed point (may lead to changes due to rounding).
    exp : int, optional
        Fixed point base of the weight (reconfigures weights/weight_exp).
    """

    def __init__(self,
                 weights: np.ndarray,
                 exp: int = 7,
                 **kwargs):

        self.exp = exp

        # Adjust the weights to the fixed point
        w = weights * 2 ** self.exp
        self.main = Sparse(weights=w,
                           num_message_bits=24,
                           num_weight_bits=8,
                           weight_exp=-self.exp)

        self.in_port = self.main.s_in
        self.out_port = self.main.a_out


class NormalizeNet(AlgebraicVector):
    """NormalizeNet
    Creates a layer for normalizing vector inputs

    Parameters
    ----------
    shape : tuple(int)
        Number and topology of neurons.
    exp : int
        Fixed point base of the vector.
    """

    def __init__(self,
                 shape: ty.Tuple[int, ...],
                 exp: ty.Optional[int] = 12,
                 **kwargs):
        self.shape = shape
        self.fpb = exp

        vec_to_fpinv_w = np.ones((1, self.shape[0]))
        fpinv_to_vec_w = np.ones((self.shape[0], 1))
        weight_exp = 0

        self.vfp_dense = Dense(weights=vec_to_fpinv_w,
                               num_message_bits=24,
                               weight_exp=-weight_exp)
        self.fpv_dense = Dense(weights=fpinv_to_vec_w,
                               num_message_bits=24,
                               weight_exp=-weight_exp)

        self.main = NormVecDelay(shape=self.shape, vth=1,
                                 exp=self.fpb)
        self.fp_inv_neuron = InvSqrt(shape=(1,), fp_base=self.fpb)

        self.main.s2_out.connect(self.vfp_dense.s_in)
        self.vfp_dense.a_out.connect(self.fp_inv_neuron.a_in)
        self.fp_inv_neuron.s_out.connect(self.fpv_dense.s_in)
        self.fpv_dense.a_out.connect(self.main.a_in2)

        self.in_port = self.main.a_in1
        self.out_port = self.main.s_out
